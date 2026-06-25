"""Combined delay calculation using JAX JIT compilation.

This module contains the performance-critical JAX-compiled delay calculation
that combines DM, solar wind, FD, and binary delays into a single kernel.
This is the key optimization that makes JUG 100x faster than PINT.
"""

# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax

import jax.numpy as jnp
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, T_SUN_SEC, PC_TO_LIGHT_SEC, AU_KM, AU_PC
from jug.delays.binary_bt import bt_binary_delay_from_tt0
from jug.delays.binary_dd import (
    dd_binary_delay,
    ddk_binary_delay,
    dd_binary_delay_from_tt0,
)
# Note: Kopeikin corrections (K96 proper motion, annual orbital parallax) are
# implemented inline in branch_ddk() below, not as separate importable functions.
from jug.delays.binary_t2 import t2_binary_delay

# Highest ELL1H H3/H4 Shapiro harmonic JUG evaluates when a par sets NHARMS.
# Tempo2's calcDH sums harmonics 3..NHARMS; realistic NHARMS is <=7, so 12 is a
# safe ceiling. Harmonics k>nharm are masked (base->0) so the partial sum is exact
# for the requested nharm and never diverges (the H3/H4 series grows as (H4/H3)^k).
_ELL1H_MAX_NHARM = 12


@jax.jit
def combined_delays(
    tdbld, freq_bary, obs_sun_pos, L_hat,
    dm_coeffs, dm_factorials, dm_epoch,
    ne_sw, fd_coeffs, has_fd,
    roemer_shapiro, has_binary, binary_model_id,
    pb, a1, tasc, eps1, eps2, eps1dot, eps2dot, pbdot, xdot, gamma, r_shap, s_shap,
    ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
    fb_coeffs, fb_factorials, fb_epoch, use_fb,
    # DDK Kopeikin parameters (optional, for model_id 5)
    obs_pos_ls=None, px=0.0, sin_ra=0.0, cos_ra=1.0, sin_dec=0.0, cos_dec=1.0,
    # K96 proper motion parameters (Kopeikin 1996)
    k96=True, pmra_rad_per_sec=0.0, pmdec_rad_per_sec=0.0,
    # Tropospheric delay (for PINT-compatible pre-binary time)
    tropo_sec=None,
    # DMX delay (for PINT-compatible pre-binary time)
    dmx_sec=None,
    # Precomputed (tdb - binary_epoch)*86400 in longdouble, then cast to float64.
    # Avoids float64 cancellation when computing t - T0 inside JAX at MJD ~58000.
    tt_binary_sec=None,
    # tt_binary_sec reduced by a whole number of orbital periods in longdouble
    # (jug.utils.orbit_reduction.reduce_binary_time_sec). Used for the LINEAR
    # orbital phase term only — integer orbits drop out of all trig — removing
    # the ~ps float64 phase-quantization floor at ~1e4 orbits. Falls back to
    # tt_binary_sec when not provided.
    tt_binary_red_sec=None,
    # DD relativistic-deformation parameters (DDGR-derived; standard DD = 0).
    # er = ecc*(1+dr), eTheta = ecc*(1+dth) in the DD Roemer.
    dr=0.0, dth=0.0,
    # ELL1H H3/H4 Shapiro harmonic count (par NHARMS, Tempo2 default 4). Only the
    # H3+H4 branch uses it; H3+STIGMA (exact) and H3-only are unaffected.
    nharm=4.0
):
    """Combined delay calculation - single JAX kernel for maximum performance.

    Now updated to support Universal Binary Kernel (ELL1, DD, DDK, T2, BT) via jax.lax.switch.

    Binary Model IDs:
    0: None
    1: ELL1 / ELL1H
    2: DD / DDGR / DDH (standard DD without Kopeikin)
    3: T2
    4: BT / BTX
    5: DDK (DD with Kopeikin annual orbital parallax + K96 proper motion)

    K96 Corrections (Kopeikin 1996):
    When k96=True and proper motion is provided, applies secular corrections
    to KIN, a1, and omega due to the pulsar's proper motion.
    """
    # === DM Delay ===
    dt_years = (tdbld - dm_epoch) / 365.25
    powers = jnp.arange(len(dm_coeffs))
    dt_powers = dt_years[:, jnp.newaxis] ** powers[jnp.newaxis, :]
    dm_eff = jnp.sum(dm_coeffs * dt_powers / dm_factorials, axis=1)
    dm_sec = K_DM_SEC * dm_eff / (freq_bary ** 2)

    # === Solar Wind Delay ===
    r_km = jnp.sqrt(jnp.sum(obs_sun_pos**2, axis=1))
    r_au = r_km / AU_KM
    sun_dir = obs_sun_pos / r_km[:, jnp.newaxis]
    cos_elong = jnp.sum(sun_dir * L_hat, axis=1)
    elong = jnp.arccos(jnp.clip(cos_elong, -1.0, 1.0))
    rho = jnp.pi - elong
    sin_rho = jnp.maximum(jnp.sin(rho), 1e-10)
    geometry_pc = AU_PC * rho / (r_au * sin_rho)
    dm_sw = ne_sw * geometry_pc
    sw_sec = jnp.where(ne_sw != 0, K_DM_SEC * dm_sw / (freq_bary ** 2), 0.0)

    # === FD Delay ===
    log_freq = jnp.log(freq_bary / 1000.0)
    fd_sec = jnp.where(
        has_fd,
        jnp.polyval(jnp.concatenate([fd_coeffs[::-1], jnp.array([0.0])]), log_freq),
        0.0
    )

    # Handle troposphere array - use zeros if not provided
    tropo_arr = jnp.where(
        tropo_sec is None,
        jnp.zeros_like(tdbld),
        tropo_sec
    ) if tropo_sec is not None else jnp.zeros_like(tdbld)

    # Handle DMX array - use zeros if not provided
    dmx_arr = jnp.where(
        dmx_sec is None,
        jnp.zeros_like(tdbld),
        dmx_sec
    ) if dmx_sec is not None else jnp.zeros_like(tdbld)

    # High-precision binary epoch offset (precomputed in longdouble outside JAX)
    tt_binary_arr = jnp.where(
        tt_binary_sec is None,
        jnp.zeros_like(tdbld),
        tt_binary_sec
    ) if tt_binary_sec is not None else jnp.zeros_like(tdbld)

    # Orbit-count-reduced binary time; falls back to the full time (the
    # reduced linear-phase formulas are then algebraically identical to the
    # original ones, differing only at the float64 rounding level).
    tt_binary_red_arr = (tt_binary_red_sec
                         if tt_binary_red_sec is not None
                         else tt_binary_arr)

    # === Universal Binary Delay Dispatch ===
    def compute_binary_universal(args):
        (tdbld_val, roemer_shapiro_val, obs_pos_ls_val, dm_val, sw_val, tropo_val, dmx_val, tt_binary_val, tt_binary_red_val) = args

        # Pre-binary delay sum: sum of all delays before BinaryDD in PINT's order.
        # roemer_shapiro_val includes: Roemer + SS Shapiro (Sun + planets)
        # We add: DM, DMX, Solar Wind, Troposphere
        prebinary_sum = roemer_shapiro_val + dm_val + dmx_val + sw_val + tropo_val

        # High-precision time for binary model:
        #   tt_binary_val = (tdb - binary_epoch) * SECS_PER_DAY, precomputed in longdouble.
        # Subtract the pre-binary delays to get the time at which the binary model is evaluated.
        # This avoids float64 cancellation when computing (tdb_mjd - T0) inside JAX at MJD ~58000.
        tt_binary_prebinary = tt_binary_val - prebinary_sum

        # Reduced counterpart: same prebinary shift, integer orbits already
        # subtracted in longdouble outside JIT. Only valid in the LINEAR
        # phase term of each model (secular terms must keep the full time).
        tt_binary_red_prebinary = tt_binary_red_val - prebinary_sum

        # MJD-based prebinary time (still needed for DDK observer position geometry)
        t_prebinary = tdbld_val - prebinary_sum / SECS_PER_DAY

        # Branch 0: None
        def branch_none(tt_pair): return 0.0

        # Branch 1: ELL1 / ELL1H (Inline Optimized)
        # tt_binary_val was computed as (tdb - TASC)*86400, so tt_binary_prebinary
        # is already (t_prebinary - TASC) in seconds.
        def branch_ell1(tt_pair):
            dt_sec_bin, dt_red_bin = tt_pair
            dt_days = dt_sec_bin / SECS_PER_DAY

            # Phase calculation (FB or PB). The LINEAR term uses the
            # orbit-count-reduced time (integer orbits drop out of all trig);
            # PBDOT / higher-order FB terms keep the full time.
            def compute_phase_fb():
                n_coeffs = len(fb_coeffs)
                indices = jnp.arange(n_coeffs)
                powers_plus1 = indices + 1
                dt_powers_plus1 = dt_sec_bin ** powers_plus1
                factorials_plus1 = fb_factorials * (indices + 1)
                terms = jnp.where(indices > 0,
                                  fb_coeffs * dt_powers_plus1 / factorials_plus1,
                                  0.0)
                # Shape check is static; both lax.switch branches are traced
                # even for non-FB pulsars, where fb_coeffs can be empty.
                fb0 = fb_coeffs[0] if fb_coeffs.shape[0] > 0 else 0.0
                phase_integral = jnp.sum(terms) + fb0 * dt_red_bin
                return 2.0 * jnp.pi * phase_integral

            def compute_phase_pb():
                n0_local = 2.0 * jnp.pi / (pb * SECS_PER_DAY)
                return n0_local * dt_red_bin - n0_local * dt_sec_bin * (pbdot / 2.0 / pb * dt_days)

            Phi = jnp.where(use_fb, compute_phase_fb(), compute_phase_pb())

            # n0 calculation
            def compute_n0_fb():
                n_coeffs = len(fb_coeffs)
                indices = jnp.arange(n_coeffs)
                dt_powers = dt_sec_bin ** indices
                F_orb = jnp.sum(fb_coeffs * dt_powers / fb_factorials)
                return 2.0 * jnp.pi * F_orb

            def compute_n0_pb():
                return 2.0 * jnp.pi / (pb * SECS_PER_DAY)
            
            n0 = jnp.where(use_fb, compute_n0_fb(), compute_n0_pb())

            sin_Phi, cos_Phi = jnp.sin(Phi), jnp.cos(Phi)
            sin_2Phi, cos_2Phi = jnp.sin(2*Phi), jnp.cos(2*Phi)
            sin_3Phi, cos_3Phi = jnp.sin(3*Phi), jnp.cos(3*Phi)
            sin_4Phi, cos_4Phi = jnp.sin(4*Phi), jnp.cos(4*Phi)

            a1_eff = jnp.where(xdot != 0.0, a1 + xdot * dt_sec_bin, a1)
            # Apply EPS1DOT/EPS2DOT time evolution
            eps1_eff = eps1 + eps1dot * dt_sec_bin
            eps2_eff = eps2 + eps2dot * dt_sec_bin
            eps1_sq, eps2_sq = eps1_eff**2, eps2_eff**2
            eps1_cu, eps2_cu = eps1_eff**3, eps2_eff**3

            Dre_a1 = (
                sin_Phi + 0.5 * (eps2_eff * sin_2Phi - eps1_eff * cos_2Phi)
                - (1.0/8.0) * (5*eps2_sq*sin_Phi - 3*eps2_sq*sin_3Phi - 2*eps2_eff*eps1_eff*cos_Phi
                              + 6*eps2_eff*eps1_eff*cos_3Phi + 3*eps1_sq*sin_Phi + 3*eps1_sq*sin_3Phi)
                - (1.0/12.0) * (5*eps2_cu*sin_2Phi + 3*eps1_sq*eps2_eff*sin_2Phi
                               - 6*eps1_eff*eps2_sq*cos_2Phi - 4*eps1_cu*cos_2Phi
                               - 4*eps2_cu*sin_4Phi + 12*eps1_sq*eps2_eff*sin_4Phi
                               + 12*eps1_eff*eps2_sq*cos_4Phi - 4*eps1_cu*cos_4Phi)
            )
            Drep_a1 = (
                cos_Phi + eps1_eff * sin_2Phi + eps2_eff * cos_2Phi
                - (1.0/8.0) * (5*eps2_sq*cos_Phi - 9*eps2_sq*cos_3Phi + 2*eps1_eff*eps2_eff*sin_Phi
                              - 18*eps1_eff*eps2_eff*sin_3Phi + 3*eps1_sq*cos_Phi + 9*eps1_sq*cos_3Phi)
                - (1.0/12.0) * (10*eps2_cu*cos_2Phi + 6*eps1_sq*eps2_eff*cos_2Phi
                               + 12*eps1_eff*eps2_sq*sin_2Phi + 8*eps1_cu*sin_2Phi
                               - 16*eps2_cu*cos_4Phi + 48*eps1_sq*eps2_eff*cos_4Phi
                               - 48*eps1_eff*eps2_sq*sin_4Phi + 16*eps1_cu*sin_4Phi)
            )
            Drepp_a1 = (
                -sin_Phi + 2*eps1_eff*cos_2Phi - 2*eps2_eff*sin_2Phi
                - (1.0/8.0) * (-5*eps2_sq*sin_Phi + 27*eps2_sq*sin_3Phi + 2*eps1_eff*eps2_eff*cos_Phi
                              - 54*eps1_eff*eps2_eff*cos_3Phi - 3*eps1_sq*sin_Phi - 27*eps1_sq*sin_3Phi)
                - (1.0/12.0) * (-20*eps2_cu*sin_2Phi - 12*eps1_sq*eps2_eff*sin_2Phi
                               + 24*eps1_eff*eps2_sq*cos_2Phi + 16*eps1_cu*cos_2Phi
                               + 64*eps2_cu*sin_4Phi - 192*eps1_sq*eps2_eff*sin_4Phi
                               - 192*eps1_eff*eps2_sq*cos_4Phi + 64*eps1_cu*cos_4Phi)
            )

            Dre = a1_eff * Dre_a1
            Drep = a1_eff * Drep_a1
            Drepp = a1_eff * Drepp_a1
            binary_roemer = Dre * (1.0 - n0*Drep + (n0*Drep)**2 + 0.5*n0**2*Dre*Drepp)

            einstein_binary = jnp.where(gamma != 0.0, gamma * sin_Phi, 0.0)
            
            # ELL1H Shapiro delay, H3 + STIGMA -> "3rd-harmonic-and-up" EXACT form
            # (Freire & Wex 2010 Eq. 28), matching Tempo2 ELL1Hmodel.C mode 1:
            #   lsc = log(1+stig^2-2*stig*sin(Phi)) + 2*stig*sin(Phi)
            #                                       - stig^2*cos(2*Phi)
            #   ds  = -2*(H3/stig^3) * lsc
            # The k=1,2 harmonics (+2*stig*sin(Phi) - stig^2*cos(2*Phi)) are
            # EXACTLY degenerate with the ELL1 Roemer a1/EPS1/EPS2 terms, so the
            # par's Keplerian values (fit by Tempo2 with this Eq.28 form) already
            # absorb them. Using the FULL log (Eq.29, as PINT's default
            # delayS_H3_STIGMA_exact does) at these fixed params DOUBLE-COUNTS k=1,2
            # -- verified on J1902-5105 (STIG=1.154): Eq.29 -> own WRMS 2.356 us
            # with sin(Phi)/cos(2Phi) structure (corr -0.52/+0.28); Eq.28 -> 1.694 us
            # flat == par TRES 1.637. (An earlier change to Eq.29 chased PINT parity
            # but was only ever validated on stig->0 pulsars where Eq.28 == Eq.29.)
            # PINT can match via its delayS3p_H3_STIGMA_exact (Eq.28); the batch
            # harness selects it. stig->0 (H3-only) path below is unaffected.
            fs = 1.0 + stig**2 - 2.0 * stig * sin_Phi
            lsc = jnp.log(fs) + 2.0 * stig * sin_Phi - stig**2 * cos_2Phi
            r_ell1h = h3 / jnp.maximum(stig**3, 1e-30)
            shapiro_ell1h = -2.0 * r_ell1h * lsc
            
            # Standard log formula for M2/SINI (non-ELL1H)
            shapiro_standard = jnp.where(
                (r_shap > 0.0) & (s_shap > 0.0),
                -2.0 * r_shap * jnp.log(1.0 - s_shap * sin_Phi),
                0.0
            )
            # Orthometric H3-only Shapiro delay (Freire & Wex 2010 Eq. 19)
            shapiro_h3only = jnp.where(
                (r_shap == 0.0) & (s_shap == 0.0) & (h3 > 0.0) & (stig == 0.0) & (h4 == 0.0),
                -(4.0 / 3.0) * h3 * sin_3Phi,
                0.0
            )
            # ELL1H mode 2/3: H3/H4 harmonic Shapiro (Tempo2 ELL1Hmodel.C calcDH):
            #   ds = sd3 + sd4 + sd5
            #   sd3 = -4/3*H3*sin(3Phi),  sd4 = H4*cos(4Phi),  sd5 = 4*H4*fs
            #   fs  = sum_{k=5}^{nharm} c_k * s^(k-4) * trig_k(k*Phi),  s = H4/H3
            #     k odd : c_k = (-1)^((k-1)/2)/k, trig = sin
            #     k even: c_k = (-1)^(k/2)/k,     trig = cos
            # sd5 vanishes for nharm<=4 (Tempo2 default), recovering the classic
            # -4/3*H3*sin(3Phi) + H4*cos(4Phi). nharm is a runtime scalar (par
            # NHARMS, default 4); harmonics k>nharm are masked by zeroing the base
            # so s^(k-4)=0 even when s>1 and the series diverges (e.g. J0613-0200
            # s=H4/H3=1.11) -- no inf*0=nan.
            cos_4Phi = jnp.cos(4.0 * Phi)
            s_h4 = h4 / jnp.where(h3 != 0.0, h3, 1.0)
            fs_h3h4 = 0.0
            for _k in range(5, _ELL1H_MAX_NHARM + 1):
                base = jnp.where(_k <= nharm, s_h4, 0.0)
                term = base ** (_k - 4)
                if _k % 2 == 1:
                    coeff = ((-1.0) ** ((_k - 1) // 2)) / _k
                    fs_h3h4 = fs_h3h4 + coeff * term * jnp.sin(_k * Phi)
                else:
                    coeff = ((-1.0) ** (_k // 2)) / _k
                    fs_h3h4 = fs_h3h4 + coeff * term * jnp.cos(_k * Phi)
            sd5_h3h4 = 4.0 * h4 * fs_h3h4
            shapiro_h3h4 = jnp.where(
                (h4 != 0.0) & (stig == 0.0),
                -(4.0 / 3.0) * h3 * sin_3Phi + h4 * cos_4Phi + sd5_h3h4,
                0.0
            )
            # Select: ELL1H lsc if stig > 0 (H3+STIGMA), else h3h4/h3only/standard.
            # No stig<=1 upper bound: PINT's delayS_H3_STIGMA_exact applies the
            # exact log for ANY STIGMA (ELL1H does not validate STIGMA<=1), and the
            # log argument (1+stig^2-2*stig*sin(Phi)) stays positive for the
            # mildly-superunity STIGMA that fits sometimes produce (e.g. J1902-5105
            # STIG=1.154). Guarding stig<=1 silently dropped the entire ELL1H
            # Shapiro for those pulsars (~1.2 us at orbital frequency vs PINT).
            shapiro_binary = jnp.where(
                stig > 0.0,
                shapiro_ell1h,
                shapiro_standard + shapiro_h3only + shapiro_h3h4
            )
            return binary_roemer + einstein_binary + shapiro_binary

        # Branch 2: DD / DDK
        # tt0 = (t_prebinary - T0) * SECS_PER_DAY (precomputed in longdouble)
        def branch_dd(tt_pair):
            tt0, tt0_red = tt_pair
            return dd_binary_delay_from_tt0(
                tt0, pb, a1, ecc, om, gamma, pbdot, omdot, xdot, edot,
                sini, m2, h3, h4, stig, tt0_red_sec=tt0_red, dr=dr, dth=dth
            )

        # Branch 3: T2. Evaluate in a relative-day coordinate so both the
        # PB/T0 and FB/TASC parameterizations use the longdouble-derived
        # binary epoch offset supplied by the caller.
        # NOTE: T2 does not yet take the orbit-count-reduced time, so it keeps
        # the ~ps float64 phase floor (t2_binary_delay handles its own phase).
        def branch_t2(tt_pair):
            tt0, _ = tt_pair
            t = tt0 / SECS_PER_DAY
            return t2_binary_delay(
                t, pb, a1, ecc, om, 0.0, gamma, pbdot, xdot, edot, omdot,
                m2, sini, kin, kom,
                fb_coeffs, fb_factorials, 0.0, use_fb
            )

        # Branch 4: BT
        def branch_bt(tt_pair):
            tt0, tt0_red = tt_pair
            return bt_binary_delay_from_tt0(
                tt0, pb, a1, ecc, om, gamma, pbdot, omdot, xdot, edot,
                tt0_red_sec=tt0_red
            )

        # Branch 5: DDK (DD with Kopeikin annual orbital parallax + K96 proper motion)
        def branch_ddk(tt_pair):
            tt0, tt0_red = tt_pair
            # Apply Kopeikin corrections if we have the required parameters
            # obs_pos_ls_val is the per-TOA observer position in light-seconds

            # Time since T0 in seconds (precomputed in longdouble for precision)
            tt0_sec = tt0

            # Base values
            kin_rad = jnp.deg2rad(kin)
            kom_rad = jnp.deg2rad(kom)

            # =====================================================================
            # K96 Proper Motion Corrections (Kopeikin 1996)
            # These are secular corrections that accumulate over time
            # =====================================================================

            # delta_kin from proper motion (Eq 10)
            # delta_KIN = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * (t - T0)
            sin_kom = jnp.sin(kom_rad)
            cos_kom = jnp.cos(kom_rad)

            delta_kin_pm = jnp.where(
                k96,
                (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
                0.0
            )

            # Effective inclination including K96 correction
            kin_eff_rad = kin_rad + delta_kin_pm

            # delta_a1 from proper motion (Eq 8)
            # delta_a1 = a1 * delta_KIN / tan(KIN)
            tan_kin_eff = jnp.tan(kin_eff_rad)
            tan_kin_eff_safe = jnp.where(jnp.abs(tan_kin_eff) < 1e-10, 1e-10, tan_kin_eff)

            delta_a1_pm = jnp.where(
                k96,
                a1 * delta_kin_pm / tan_kin_eff_safe,
                0.0
            )

            # delta_omega from proper motion (Eq 9)
            # delta_omega = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * (t - T0)
            sin_kin_eff = jnp.sin(kin_eff_rad)
            sin_kin_eff_safe = jnp.where(jnp.abs(sin_kin_eff) < 1e-10, 1e-10, sin_kin_eff)

            delta_omega_pm_rad = jnp.where(
                k96,
                (1.0 / sin_kin_eff_safe) * (pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom) * tt0_sec,
                0.0
            )

            # =====================================================================
            # Kopeikin 1995 Annual Orbital Parallax Corrections
            # These are periodic corrections based on Earth's position
            # =====================================================================

            # Kopeikin projection terms
            x = obs_pos_ls_val[0]
            y = obs_pos_ls_val[1]
            z = obs_pos_ls_val[2]

            delta_I0 = -x * sin_ra + y * cos_ra
            delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec

            # Distance in light-seconds from parallax
            px_safe = jnp.maximum(jnp.abs(px), 1e-10)
            d_ls = 1000.0 * PC_TO_LIGHT_SEC / px_safe

            # Use effective KIN (with K96 correction) for parallax corrections
            tan_kin_for_px = jnp.tan(kin_eff_rad)
            tan_kin_for_px_safe = jnp.where(jnp.abs(tan_kin_for_px) < 1e-10, 1e-10, tan_kin_for_px)

            # delta_a1 from parallax (Eq 17)
            delta_a1_px = jnp.where(
                (px > 0.0) & (jnp.abs(kin) > 0.0),
                (a1 / tan_kin_for_px_safe / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom),
                0.0
            )

            # delta_omega from parallax (Eq 19)
            sin_kin_for_px = jnp.sin(kin_eff_rad)
            sin_kin_for_px_safe = jnp.where(jnp.abs(sin_kin_for_px) < 1e-10, 1e-10, sin_kin_for_px)

            delta_omega_px_rad = jnp.where(
                (px > 0.0) & (jnp.abs(kin) > 0.0),
                -(1.0 / sin_kin_for_px_safe / d_ls) * (delta_I0 * cos_kom + delta_J0 * sin_kom),
                0.0
            )

            # =====================================================================
            # Apply all corrections
            # =====================================================================

            # Total corrections: K96 proper motion + Kopeikin 1995 parallax
            a1_eff = a1 + delta_a1_pm + delta_a1_px
            om_eff = om + jnp.rad2deg(delta_omega_pm_rad) + jnp.rad2deg(delta_omega_px_rad)

            # For DDK, SINI = sin(KIN_eff) if SINI is not provided
            sini_eff = jnp.where(
                (sini == 0.0) & (jnp.abs(kin) > 0.0),
                jnp.sin(kin_eff_rad),
                sini
            )

            return dd_binary_delay_from_tt0(
                tt0, pb, a1_eff, ecc, om_eff, gamma, pbdot, omdot, xdot, edot,
                sini_eff, m2, h3, h4, stig, tt0_red_sec=tt0_red
            )

        # Switch logic (6 branches: 0=None, 1=ELL1, 2=DD, 3=T2, 4=BT, 5=DDK)
        # All branches receive (tt_binary_prebinary, tt_binary_red_prebinary):
        # the full and orbit-count-reduced (t_prebinary - binary_epoch) * 86400,
        # both precomputed in longdouble, avoiding float64 cancellation at
        # MJD ~58000 and the ~ps float64 phase floor respectively.
        return jax.lax.switch(
            binary_model_id,
            [branch_none, branch_ell1, branch_dd, branch_t2, branch_bt, branch_ddk],
            (tt_binary_prebinary, tt_binary_red_prebinary)
        )

    # Prepare observer position - use zeros if not provided (for non-DDK models)
    obs_pos_ls_arr = jnp.where(
        obs_pos_ls is None,
        jnp.zeros((len(tdbld), 3)),
        obs_pos_ls
    ) if obs_pos_ls is not None else jnp.zeros((len(tdbld), 3))

    binary_sec = jnp.where(
        has_binary,
        jax.vmap(compute_binary_universal)((tdbld, roemer_shapiro, obs_pos_ls_arr, dm_sec, sw_sec, tropo_arr, dmx_arr, tt_binary_arr, tt_binary_red_arr)),
        0.0
    )

    total_delay = dm_sec + sw_sec + fd_sec + binary_sec
    return total_delay



@jax.jit
def compute_total_delay_jax(
    tdbld, freq_bary, obs_sun, L_hat,
    dm_coeffs, dm_factorials, dm_epoch,
    ne_sw, fd_coeffs, has_fd,
    roemer_shapiro, has_binary, binary_model_id,
    pb, a1, tasc, eps1, eps2, eps1dot, eps2dot, pbdot, xdot, gamma, r_shap, s_shap,
    ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
    fb_coeffs, fb_factorials, fb_epoch, use_fb,
    # DDK Kopeikin parameters (optional)
    obs_pos_ls=None, px=0.0, sin_ra=0.0, cos_ra=1.0, sin_dec=0.0, cos_dec=1.0,
    # K96 proper motion parameters (Kopeikin 1996)
    k96=True, pmra_rad_per_sec=0.0, pmdec_rad_per_sec=0.0,
    # Tropospheric delay (for PINT-compatible pre-binary time)
    tropo_sec=None,
    # DMX delay (for PINT-compatible pre-binary time)
    dmx_sec=None,
    # Precomputed (tdb - binary_epoch)*86400 in longdouble then cast to float64.
    tt_binary_sec=None,
    # Orbit-count-reduced tt_binary_sec (see combined_delays / orbit_reduction).
    tt_binary_red_sec=None,
    # DD relativistic-deformation parameters (DDGR-derived; standard DD = 0).
    dr=0.0, dth=0.0,
    # ELL1H H3/H4 Shapiro harmonic count (par NHARMS, Tempo2 default 4).
    nharm=4.0
):
    """Compute total delay in a single JAX kernel.

    This wrapper adds the Roemer+Shapiro delay to the combined delays
    from DM, solar wind, FD, and binary.

    For DDK model (binary_model_id=5), additional parameters are needed:
    - obs_pos_ls: Observer position relative to SSB in light-seconds, shape (N, 3)
    - px: Parallax in milliarcseconds
    - sin_ra, cos_ra: Sine/cosine of pulsar RA
    - sin_dec, cos_dec: Sine/cosine of pulsar DEC

    K96 proper motion corrections (Kopeikin 1996):
    - k96: Boolean flag to enable proper motion corrections (default True)
    - pmra_rad_per_sec: Proper motion in RA (radians/second), PMRA/cos(DEC)
    - pmdec_rad_per_sec: Proper motion in DEC (radians/second)
    
    Tropospheric delay:
    - tropo_sec: Tropospheric delay in seconds (for PINT-compatible pre-binary time)
                 If None, zeros are used internally.
    
    DMX delay:
    - dmx_sec: DMX delay in seconds (for PINT-compatible pre-binary time)
               If None, zeros are used internally.
    """
    combined_sec = combined_delays(
        tdbld, freq_bary, obs_sun, L_hat,
        dm_coeffs, dm_factorials, dm_epoch,
        ne_sw, fd_coeffs, has_fd,
        roemer_shapiro, has_binary, binary_model_id,
        pb, a1, tasc, eps1, eps2, eps1dot, eps2dot, pbdot, xdot, gamma, r_shap, s_shap,
        ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
        fb_coeffs, fb_factorials, fb_epoch, use_fb,
        obs_pos_ls, px, sin_ra, cos_ra, sin_dec, cos_dec,
        k96, pmra_rad_per_sec, pmdec_rad_per_sec,
        tropo_sec,
        dmx_sec,
        tt_binary_sec,
        tt_binary_red_sec,
        dr,
        dth,
        nharm
    )

    return roemer_shapiro + combined_sec
