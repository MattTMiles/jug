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
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, T_SUN_SEC
from jug.delays.binary_bt import bt_binary_delay
from jug.delays.binary_dd import (
    dd_binary_delay,
    ddk_binary_delay,
)
# Note: Kopeikin corrections (K96 proper motion, annual orbital parallax) are
# implemented inline in branch_ddk() below, not as separate importable functions.
from jug.delays.binary_t2 import t2_binary_delay


@jax.jit
def combined_delays(
    tdbld, freq_bary, obs_sun_pos, L_hat,
    dm_coeffs, dm_factorials, dm_epoch,
    ne_sw, fd_coeffs, has_fd,
    roemer_shapiro, has_binary, binary_model_id,
    pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap,
    ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
    fb_coeffs, fb_factorials, fb_epoch, use_fb,
    # DDK Kopeikin parameters (optional, for model_id 5)
    obs_pos_ls=None, px=0.0, sin_ra=0.0, cos_ra=1.0, sin_dec=0.0, cos_dec=1.0,
    # K96 proper motion parameters (Kopeikin 1996)
    k96=True, pmra_rad_per_sec=0.0, pmdec_rad_per_sec=0.0,
    # Tropospheric delay (for PINT-compatible pre-binary time)
    tropo_sec=None
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
    AU_KM_local = 1.495978707e8
    AU_PC = 4.84813681e-6
    r_km = jnp.sqrt(jnp.sum(obs_sun_pos**2, axis=1))
    r_au = r_km / AU_KM_local
    sun_dir = obs_sun_pos / r_km[:, jnp.newaxis]
    cos_elong = jnp.sum(sun_dir * L_hat, axis=1)
    elong = jnp.arccos(jnp.clip(cos_elong, -1.0, 1.0))
    rho = jnp.pi - elong
    sin_rho = jnp.maximum(jnp.sin(rho), 1e-10)
    geometry_pc = AU_PC * rho / (r_au * sin_rho)
    dm_sw = ne_sw * geometry_pc
    sw_sec = jnp.where(ne_sw > 0, K_DM_SEC * dm_sw / (freq_bary ** 2), 0.0)

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

    # === Universal Binary Delay Dispatch ===
    def compute_binary_universal(args):
        (tdbld_val, roemer_shapiro_val, obs_pos_ls_val, dm_val, sw_val, tropo_val) = args

        # Binary evaluation time: PINT-compatible "pre-binary" time
        #
        # PINT's delay component order is:
        #   AstrometryEquatorial -> TroposphereDelay -> SolarSystemShapiro ->
        #   SolarWindDispersion -> DispersionDM -> BinaryDD -> FD
        #
        # So PINT evaluates BinaryDD at:
        #   t_prebinary = tdbld - (all delays before BinaryDD) / 86400
        #
        # This includes: Roemer (astrometry), Troposphere, SS Shapiro, Solar Wind, DM
        # but NOT FD (which comes after BinaryDD).
        #
        # roemer_shapiro_val includes: Roemer + SS Shapiro (Sun + planets)
        # We add: DM, Solar Wind, Troposphere
        t_prebinary = tdbld_val - (roemer_shapiro_val + dm_val + sw_val + tropo_val) / SECS_PER_DAY

        # Branch 0: None
        def branch_none(t): return 0.0

        # Branch 1: ELL1 / ELL1H (Inline Optimized)
        def branch_ell1(t):
            dt_days = t - tasc
            dt_sec_bin = dt_days * SECS_PER_DAY

            # Phase calculation (FB or PB)
            def compute_phase_fb():
                dt_fb = dt_sec_bin
                n_coeffs = len(fb_coeffs)
                indices = jnp.arange(n_coeffs)
                powers_plus1 = indices + 1
                dt_powers_plus1 = dt_fb ** powers_plus1
                factorials_plus1 = fb_factorials * (indices + 1)
                phase_integral = jnp.sum(fb_coeffs * dt_powers_plus1 / factorials_plus1)
                return 2.0 * jnp.pi * phase_integral

            def compute_phase_pb():
                n0_local = 2.0 * jnp.pi / (pb * SECS_PER_DAY)
                return n0_local * dt_sec_bin * (1.0 - pbdot / 2.0 / pb * dt_days)

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
            eps1_sq, eps2_sq = eps1**2, eps2**2
            eps1_cu, eps2_cu = eps1**3, eps2**3

            Dre_a1 = (
                sin_Phi + 0.5 * (eps2 * sin_2Phi - eps1 * cos_2Phi)
                - (1.0/8.0) * (5*eps2_sq*sin_Phi - 3*eps2_sq*sin_3Phi - 2*eps2*eps1*cos_Phi
                              + 6*eps2*eps1*cos_3Phi + 3*eps1_sq*sin_Phi + 3*eps1_sq*sin_3Phi)
                - (1.0/12.0) * (5*eps2_cu*sin_2Phi + 3*eps1_sq*eps2*sin_2Phi
                               - 6*eps1*eps2_sq*cos_2Phi - 4*eps1_cu*cos_2Phi
                               - 4*eps2_cu*sin_4Phi + 12*eps1_sq*eps2*sin_4Phi
                               + 12*eps1*eps2_sq*cos_4Phi - 4*eps1_cu*cos_4Phi)
            )
            Drep_a1 = (
                cos_Phi + eps1 * sin_2Phi + eps2 * cos_2Phi
                - (1.0/8.0) * (5*eps2_sq*cos_Phi - 9*eps2_sq*cos_3Phi + 2*eps1*eps2*sin_Phi
                              - 18*eps1*eps2*sin_3Phi + 3*eps1_sq*cos_Phi + 9*eps1_sq*cos_3Phi)
                - (1.0/12.0) * (10*eps2_cu*cos_2Phi + 6*eps1_sq*eps2*cos_2Phi
                               + 12*eps1*eps2_sq*sin_2Phi + 8*eps1_cu*sin_2Phi
                               - 16*eps2_cu*cos_4Phi + 48*eps1_sq*eps2*cos_4Phi
                               - 48*eps1*eps2_sq*sin_4Phi + 16*eps1_cu*sin_4Phi)
            )
            Drepp_a1 = (
                -sin_Phi + 2*eps1*cos_2Phi - 2*eps2*sin_2Phi
                - (1.0/8.0) * (-5*eps2_sq*sin_Phi + 27*eps2_sq*sin_3Phi + 2*eps1*eps2*cos_Phi
                              - 54*eps1*eps2*cos_3Phi - 3*eps1_sq*sin_Phi - 27*eps1_sq*sin_3Phi)
                - (1.0/12.0) * (-20*eps2_cu*sin_2Phi - 12*eps1_sq*eps2*sin_2Phi
                               + 24*eps1*eps2_sq*cos_2Phi + 16*eps1_cu*cos_2Phi
                               + 64*eps2_cu*sin_4Phi - 192*eps1_sq*eps2*sin_4Phi
                               - 192*eps1*eps2_sq*cos_4Phi + 64*eps1_cu*cos_4Phi)
            )

            Dre = a1_eff * Dre_a1
            Drep = a1_eff * Drep_a1
            Drepp = a1_eff * Drepp_a1
            binary_roemer = Dre * (1.0 - n0*Drep + (n0*Drep)**2 + 0.5*n0**2*Dre*Drepp)

            einstein_binary = jnp.where(gamma != 0.0, gamma * sin_Phi, 0.0)
            shapiro_binary = jnp.where(
                (r_shap > 0.0) & (s_shap > 0.0),
                -2.0 * r_shap * jnp.log(1.0 - s_shap * sin_Phi),
                0.0
            )
            # Orthometric H3-only Shapiro delay (Freire & Wex 2010 Eq. 19)
            # When r_shap=0 and s_shap=0 but h3>0, use harmonic expansion.
            # Leading non-absorbed harmonic is the 3rd:
            #   Δ_S = -(4/3)*H3*sin(3Φ)
            # Coefficient 4/3 = 2 * (2/3) from Fourier basis factor 2/k for k=3.
            # Matches Tempo2 ELL1Hmodel.C mode 0 and PINT ELL1H_model.py.
            shapiro_h3only = jnp.where(
                (r_shap == 0.0) & (s_shap == 0.0) & (h3 > 0.0),
                -(4.0 / 3.0) * h3 * sin_3Phi,
                0.0
            )
            shapiro_binary = shapiro_binary + shapiro_h3only
            return binary_roemer + einstein_binary + shapiro_binary

        # Branch 2: DD / DDK
        def branch_dd(t):
            return dd_binary_delay(
                t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
                sini, m2, h3, h4, stig
            )

        # Branch 3: T2
        def branch_t2(t):
            return t2_binary_delay(
                t, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
                m2, sini, kin, kom,
                fb_coeffs, fb_factorials, fb_epoch, use_fb
            )

        # Branch 4: BT
        def branch_bt(t):
            return bt_binary_delay(
                t, pb, a1, ecc, om, t0, gamma, pbdot, m2, sini, omdot, xdot
            )

        # Branch 5: DDK (DD with Kopeikin annual orbital parallax + K96 proper motion)
        def branch_ddk(t):
            # Apply Kopeikin corrections if we have the required parameters
            # obs_pos_ls_val is the per-TOA observer position in light-seconds

            # Time since T0 in seconds (for K96 proper motion corrections)
            dt_days = t - t0
            tt0_sec = dt_days * SECS_PER_DAY

            # Base values
            kin_rad = jnp.deg2rad(kin)
            kom_rad = jnp.deg2rad(kom)

            # =====================================================================
            # K96 Proper Motion Corrections (Kopeikin 1996)
            # These are secular corrections that accumulate over time
            # =====================================================================

            # delta_kin from proper motion (Eq 10)
            # δ_KIN = (-μ_RA * sin(KOM) + μ_DEC * cos(KOM)) * (t - T0)
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
            # δ_a1 = a1 * δ_KIN / tan(KIN)
            tan_kin_eff = jnp.tan(kin_eff_rad)
            tan_kin_eff_safe = jnp.where(jnp.abs(tan_kin_eff) < 1e-10, 1e-10, tan_kin_eff)

            delta_a1_pm = jnp.where(
                k96,
                a1 * delta_kin_pm / tan_kin_eff_safe,
                0.0
            )

            # delta_omega from proper motion (Eq 9)
            # δ_ω = (1/sin(KIN)) * (μ_RA * cos(KOM) + μ_DEC * sin(KOM)) * (t - T0)
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
            PC_TO_LIGHT_SEC = 3.0857e16 / 2.99792458e8
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

            return dd_binary_delay(
                t, pb, a1_eff, ecc, om_eff, t0, gamma, pbdot, omdot, xdot, edot,
                sini_eff, m2, h3, h4, stig
            )

        # Switch logic (6 branches: 0=None, 1=ELL1, 2=DD, 3=T2, 4=BT, 5=DDK)
        return jax.lax.switch(
            binary_model_id,
            [branch_none, branch_ell1, branch_dd, branch_t2, branch_bt, branch_ddk],
            t_prebinary
        )

    # Prepare observer position - use zeros if not provided (for non-DDK models)
    obs_pos_ls_arr = jnp.where(
        obs_pos_ls is None,
        jnp.zeros((len(tdbld), 3)),
        obs_pos_ls
    ) if obs_pos_ls is not None else jnp.zeros((len(tdbld), 3))

    binary_sec = jnp.where(
        has_binary,
        jax.vmap(compute_binary_universal)((tdbld, roemer_shapiro, obs_pos_ls_arr, dm_sec, sw_sec, tropo_arr)),
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
    pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap,
    ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
    fb_coeffs, fb_factorials, fb_epoch, use_fb,
    # DDK Kopeikin parameters (optional)
    obs_pos_ls=None, px=0.0, sin_ra=0.0, cos_ra=1.0, sin_dec=0.0, cos_dec=1.0,
    # K96 proper motion parameters (Kopeikin 1996)
    k96=True, pmra_rad_per_sec=0.0, pmdec_rad_per_sec=0.0,
    # Tropospheric delay (for PINT-compatible pre-binary time)
    tropo_sec=None
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
    """
    combined_sec = combined_delays(
        tdbld, freq_bary, obs_sun, L_hat,
        dm_coeffs, dm_factorials, dm_epoch,
        ne_sw, fd_coeffs, has_fd,
        roemer_shapiro, has_binary, binary_model_id,
        pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap,
        ecc, om, t0, omdot, edot, m2, sini, kin, kom, h3, h4, stig,
        fb_coeffs, fb_factorials, fb_epoch, use_fb,
        obs_pos_ls, px, sin_ra, cos_ra, sin_dec, cos_dec,
        k96, pmra_rad_per_sec, pmdec_rad_per_sec,
        tropo_sec
    )

    return roemer_shapiro + combined_sec

