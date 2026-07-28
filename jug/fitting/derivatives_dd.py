"""Analytical derivatives for DD binary orbital parameters (JAX implementation).

The DD (Damour-Deruelle) binary model uses traditional Keplerian orbital elements:
- T0: Time of periastron passage (MJD)
- PB: Orbital period (days)
- A1: Projected semi-major axis (light-seconds)
- ECC: Orbital eccentricity
- OM: Longitude of periastron (degrees)
- OMDOT: Rate of periastron advance (deg/yr, optional)
- GAMMA: Time dilation + gravitational redshift (s, optional)

This contrasts with ELL1 which uses Laplace-Lagrange parameters (TASC, EPS1, EPS2).

The DD delay consists of:
1. Roemer delay: Light travel time across orbit
2. Einstein delay: Time dilation + gravitational redshift (GAMMA term)
3. Shapiro delay: Signal delay from companion's gravitational field

Reference: Damour & Deruelle (1986), PINT src/pint/models/binary_dd.py
"""

import warnings
from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from jug.io.par_reader import get_longdouble
from jug.utils.constants import SECS_PER_DAY, SECS_PER_YEAR, T_SUN, DEG_TO_RAD, PC_TO_LIGHT_SEC
from jug.utils.orbit_reduction import reduce_binary_time_sec

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)


def _as_f64(x):
    """Cast a concrete or traced scalar to float64 for JAX kernels."""
    if isinstance(x, (np.longdouble, np.float128)):
        return jnp.float64(float(x))
    return jnp.asarray(x, dtype=jnp.float64)


def _orthometric_values_active(params) -> bool:
    """True iff any orthometric parameter carries a nonzero value."""
    return any(
        float(params.get(key, 0.0) or 0.0) != 0.0
        for key in ("H3", "H4", "STIG", "STIGMA")
    )


_DDK_ORTHOMETRIC_REJECTION = (
    "Orthometric Shapiro parameters (H3/H4/STIG) are not supported for "
    "DDK/Kopeikin binaries: DDK derives the inclination from KIN. "
    "Fit KIN and M2 instead."
)


def _compute_tt0_sec(toas_bary_mjd: np.ndarray, t0: float) -> np.ndarray:
    """Return (toas_bary_mjd - t0) in seconds using longdouble to avoid float64 cancellation.

    Direct float64 subtraction at MJD ~58000 loses ~4 decimal digits (~600 ns in tt0,
    ~67 ps Roemer error).  Computing in longdouble before returning float64 eliminates
    this; if the caller provides longdouble prebin_mjd the result is sub-ns accurate.
    """
    t0_ld = np.longdouble(t0)
    return np.asarray(
        (np.asarray(toas_bary_mjd, dtype=np.longdouble) - t0_ld) * np.longdouble(SECS_PER_DAY),
        dtype=np.float64,
    )


def _resolve_pb_days(params: Dict) -> float:
    """Orbital period in days, FB-aware (PB = 1/FB0 when PB is absent).

    Prevents a bare params.get('PB', 1.0) from silently returning 1 day for
    FB-parameterized binaries, which would corrupt pb-based formulas in the DD
    derivative routines. See also _extract_dd_params (which additionally derives
    PBDOT from FB1 for the forward model).
    """
    if 'PB' in params:
        return float(params['PB'])
    fb0 = params.get('FB0')
    if fb0 is not None and float(fb0) != 0.0:
        return 1.0 / (float(fb0) * SECS_PER_DAY)
    return 1.0


# =============================================================================
# Eccentric Anomaly Solver (Kepler's Equation)
# =============================================================================

@jax.jit
def solve_kepler(mean_anomaly: jnp.ndarray, ecc: float, tol: float = 1e-12) -> jnp.ndarray:
    """Solve Kepler's equation: E - ecc*sin(E) = M
    
    Uses Newton-Raphson iteration.
    
    Parameters
    ----------
    mean_anomaly : jnp.ndarray
        Mean anomaly M in radians
    ecc : float
        Orbital eccentricity (0 <= ecc < 1)
    tol : float
        Convergence tolerance
        
    Returns
    -------
    E : jnp.ndarray
        Eccentric anomaly in radians
    """
    # Initial guess
    E = mean_anomaly + ecc * jnp.sin(mean_anomaly)
    
    # Newton-Raphson iterations (fixed count for JIT compatibility)
    for _ in range(10):
        f = E - ecc * jnp.sin(E) - mean_anomaly
        fp = 1.0 - ecc * jnp.cos(E)
        E = E - f / fp
    
    return E


@jax.jit
def compute_true_anomaly(E: jnp.ndarray, ecc: float) -> jnp.ndarray:
    """Compute true anomaly from eccentric anomaly.
    
    tan(theta/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    ecc : float
        Orbital eccentricity
        
    Returns
    -------
    theta : jnp.ndarray
        True anomaly in radians
    """
    half_E = E / 2.0
    theta = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc) * jnp.sin(half_E),
        jnp.sqrt(1.0 - ecc) * jnp.cos(half_E),
    )
    return theta


# =============================================================================
# DD Model Orbital Phase and Delay
# =============================================================================

@jax.jit
def compute_mean_anomaly_dd(
    toas_bary_mjd: jnp.ndarray,
    pb: float,
    t0: float,
    pbdot: float = 0.0
) -> jnp.ndarray:
    """Compute mean anomaly for DD model.
    
    M = 2pi * (t - T0) / PB * (1 - 0.5 * PBDOT * (t - T0) / PB)
    
    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    pb : float
        Orbital period in days
    t0 : float
        Time of periastron passage in MJD
    pbdot : float
        Period derivative (dimensionless)
        
    Returns
    -------
    M : jnp.ndarray
        Mean anomaly in radians
    """
    dt = toas_bary_mjd - t0  # days
    orbits = dt / pb * (1.0 - 0.5 * pbdot * dt / pb)
    return 2 * jnp.pi * orbits


@jax.jit
def compute_dd_roemer_delay(
    E: jnp.ndarray,
    theta: jnp.ndarray,
    a1: float,
    ecc: float,
    om_rad: float
) -> jnp.ndarray:
    """Compute Roemer delay for DD model.
    
    Roemer delay = a1 * (sin(omega) * (cos(E) - ecc) + 
                         cos(omega) * sqrt(1-ecc^2) * sin(E))
    
    Or equivalently using true anomaly:
    Roemer delay = a1 * sin(omega + theta) * (1 - ecc^2) / (1 + ecc*cos(theta))
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    theta : jnp.ndarray
        True anomaly in radians
    a1 : float
        Projected semi-major axis in light-seconds
    ecc : float
        Orbital eccentricity
    om_rad : float
        Longitude of periastron in radians
        
    Returns
    -------
    roemer : jnp.ndarray
        Roemer delay in seconds
    """
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # Using eccentric anomaly formulation (more stable)
    roemer = a1 * (sin_omega * (jnp.cos(E) - ecc) + 
                   cos_omega * sqrt_1_e2 * jnp.sin(E))
    return roemer


@jax.jit
def compute_dd_einstein_delay(
    E: jnp.ndarray,
    gamma: float,
    ecc: float
) -> jnp.ndarray:
    """Compute Einstein delay (time dilation + gravitational redshift).
    
    Einstein delay = GAMMA * sin(E)
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    gamma : float
        Einstein delay amplitude in seconds
    ecc : float
        Orbital eccentricity (not used directly, but kept for API)
        
    Returns
    -------
    einstein : jnp.ndarray
        Einstein delay in seconds
    """
    return gamma * jnp.sin(E)


@jax.jit
def compute_dd_shapiro_delay(
    E: jnp.ndarray,
    theta: jnp.ndarray,
    om_rad: float,
    ecc: float,
    sini: float,
    m2: float
) -> jnp.ndarray:
    """Compute Shapiro delay for DD model.

    Implements Damour & Deruelle (1986) equation [26]:
        Shapiro = -2 * r * log(1 - e*cos(E) - s*(sin(omega)*(cos(E)-e) + sqrt(1-e^2)*cos(omega)*sin(E)))

    where r = T_SUN * M2 and s = SINI.

    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    theta : jnp.ndarray
        True anomaly in radians (unused, kept for clarity)
    om_rad : float
        Longitude of periastron in radians
    ecc : float
        Orbital eccentricity
    sini : float
        Sine of orbital inclination
    m2 : float
        Companion mass in solar masses

    Returns
    -------
    shapiro : jnp.ndarray
        Shapiro delay in seconds
    """
    r = T_SUN * m2  # Range parameter

    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)

    # D&D 1986 eq. [26]: argument of log
    arg = 1 - ecc * cos_E - sini * (sin_omega * (cos_E - ecc) + sqrt_1_e2 * cos_omega * sin_E)
    arg = jnp.maximum(arg, 1e-10)  # Avoid log(0) for edge-on orbits

    return -2 * r * jnp.log(arg)


def _extract_dd_params(params: Dict):
    """Extract common DD binary parameters from params dict.

    Returns a dict of floats: a1, pb, t0, ecc, om_deg, pbdot, gamma, m2,
    sini, omdot, xdot, edot.
    """
    a1 = float(params.get('A1', 0.0))
    pb = _resolve_pb_days(params)
    t0 = get_longdouble(params, 'T0', default=0.0)
    ecc = float(params.get('ECC', params.get('E', 0.0)))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    m2 = float(params.get('M2', 0.0))

    # Handle SINI - can be numeric or 'KIN' (DDK convention: SINI = sin(KIN))
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        kin_deg = float(params.get('KIN', 0.0))
        sini = float(jnp.sin(jnp.deg2rad(kin_deg)))
    else:
        sini = float(sini_raw)

    # DDS model: Shapiro inclination parameterized by SHAPMAX = -log(1 - sin i)
    # (Kramer et al. 2006; PINT DDS_model.SINI). Better-conditioned than SINI
    # near edge-on. Without this, a DDS par (which has no SINI) would read
    # sini=0 and silently drop the Shapiro delay.
    if sini == 0.0 and 'SHAPMAX' in params:
        sini = float(1.0 - np.exp(-float(params['SHAPMAX'])))

    # Check for orthometric parameters if SINI/M2 not set
    if sini == 0.0 or m2 == 0.0:
        h3 = float(params.get('H3', 0.0))
        stig = float(params.get('STIG', params.get('STIGMA', 0.0)))
        h4 = float(params.get('H4', 0.0))

        if h3 != 0.0 and stig > 0.0:
            if h4 != 0.0:
                warnings.warn(
                    "Both STIG and H4 are nonzero; using H3/STIG parameterization (H4 ignored)",
                    UserWarning, stacklevel=2
                )
            sini = 2 * stig / (1 + stig**2)
            m2 = h3 / (stig**3 * T_SUN)
        elif h3 != 0.0 and h4 != 0.0 and (h4 / h3) > 0.0:
            stig_derived = h4 / h3
            sini = 2.0 * stig_derived / (1.0 + stig_derived**2)
            m2 = h3 / (stig_derived**3 * T_SUN)
        elif h3 != 0.0 and h4 == 0.0 and stig == 0.0:
            warnings.warn(
                "H3/H4 parameterization with H4=0: M2 is ill-conditioned; derivative will be zero",
                UserWarning, stacklevel=2
            )

    omdot = float(params.get('OMDOT', 0.0))
    xdot = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    edot = float(params.get('EDOT', 0.0))

    # FB-parameterization guard: this DD core takes PB directly, so if a DD/BT
    # pulsar uses FB instead of PB, PB would default to 1.0 day and silently
    # corrupt the orbital phase, periastron accumulation, and the eq[52]
    # inverse-correction nhat (this is the FB bug that hit ELL1; see
    # derivatives_binary._compute_ell1_binary_delay_jit). Derive PB from FB0 and
    # PBDOT from FB1 (PB = 1/FB0; PBDOT = dPB/dt = -FB1/FB0^2) so the DD core gets
    # the right period. FB2+ are not representable by the DD PB/PBDOT pair.
    if 'PB' not in params and 'FB0' in params:
        fb0 = float(params['FB0'])
        if fb0 != 0.0:
            pb = 1.0 / (fb0 * SECS_PER_DAY)  # days
            if 'PBDOT' not in params and 'FB1' in params:
                fb1 = float(params['FB1'])
                pbdot = -fb1 / fb0 ** 2  # dimensionless dPB/dt
            if any(
                key.startswith('FB') and key[2:].isdigit() and int(key[2:]) >= 2
                for key in params
            ):
                warnings.warn(
                    "DD binary with FB2+ terms: the DD core only supports PB/PBDOT, "
                    "so FB2+ orbital-frequency evolution is ignored. Use ELL1/T2 for "
                    "full FB support.", UserWarning, stacklevel=2)

    return dict(a1=a1, pb=pb, t0=t0, ecc=ecc, om_deg=om_deg, pbdot=pbdot,
                gamma=gamma, m2=m2, sini=sini, omdot=omdot, xdot=xdot, edot=edot)


@dataclass(frozen=True)
class KopeikinStructure:
    kin_deg_ref: float
    kom_deg_ref: float
    px_mas_ref: float
    pmra_ref: float
    pmdec_ref: float
    pm_factor: float
    pmra_keys: tuple[str, ...]
    pmdec_keys: tuple[str, ...]
    use_k96: bool
    has_parallax: bool
    is_ecliptic: bool
    sin_ra: float
    cos_ra: float
    sin_dec: float
    cos_dec: float
    obl_rad: float
    sini_explicit: float


def resolve_kopeikin_flags(params: Dict) -> KopeikinStructure:
    """Resolve concrete DDK structure once from reference params."""
    import math
    from jug.io.par_reader import OBLIQUITY_ARCSEC, parse_ra, parse_dec

    kin_deg_ref = float(params.get("KIN", 0.0))
    kom_deg_ref = float(params.get("KOM", 0.0))
    px_mas_ref = float(params.get("PX", 0.0))
    pm_factor = float((math.pi / 180.0 / 3600.0 / 1000.0) / SECS_PER_YEAR)

    is_ecliptic = bool(params.get("_ecliptic_coords", False))
    if is_ecliptic:
        pmra_keys = ("_ecliptic_pm_lon", "PMLAMBDA", "PMELONG")
        pmdec_keys = ("_ecliptic_pm_lat", "PMBETA", "PMELAT")
        pmra_ref = float(
            params.get("_ecliptic_pm_lon", params.get("PMLAMBDA", params.get("PMELONG", 0.0)))
        )
        pmdec_ref = float(
            params.get("_ecliptic_pm_lat", params.get("PMBETA", params.get("PMELAT", 0.0)))
        )
    else:
        pmra_keys = ("PMRA",)
        pmdec_keys = ("PMDEC",)
        pmra_ref = float(params.get("PMRA", 0.0))
        pmdec_ref = float(params.get("PMDEC", 0.0))

    k96_flag = True
    if "K96" in params and params["K96"] is not None:
        k96_param = params["K96"]
        if isinstance(k96_param, bool):
            k96_flag = k96_param
        elif isinstance(k96_param, str):
            k96_flag = k96_param.upper() not in ("N", "NO", "FALSE", "0", "F")
        else:
            k96_flag = bool(k96_param)
    use_k96 = k96_flag and (pmra_ref != 0.0 or pmdec_ref != 0.0)
    # Structural gate: Kopeikin parallax terms exist whenever KIN is defined.
    # The correction is linear in PX, so a zero or negative reference must not
    # disable the sector (same class of bug as the old PX>0 delay branch).
    has_parallax = abs(kin_deg_ref) > 0.0

    obl_rad = 0.0
    if is_ecliptic:
        ecl_frame = str(params.get("_ecliptic_frame", "IERS2010")).upper()
        obl_rad = (
            OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC["IERS2010"])
            * math.pi
            / (180.0 * 3600.0)
        )
        lon_rad = math.pi / 180.0 * float(params.get("_ecliptic_lon_deg", 0.0))
        lat_rad = math.pi / 180.0 * float(params.get("_ecliptic_lat_deg", 0.0))
        sin_ra, cos_ra = math.sin(lon_rad), math.cos(lon_rad)
        sin_dec, cos_dec = math.sin(lat_rad), math.cos(lat_rad)
    else:
        raj_val = params.get("RAJ", 0.0)
        decj_val = params.get("DECJ", 0.0)
        ra_rad = (
            parse_ra(raj_val)
            if isinstance(raj_val, str) and ":" in raj_val
            else float(raj_val)
        )
        dec_rad = (
            parse_dec(decj_val)
            if isinstance(decj_val, str) and ":" in decj_val
            else float(decj_val)
        )
        sin_ra, cos_ra = math.sin(ra_rad), math.cos(ra_rad)
        sin_dec, cos_dec = math.sin(dec_rad), math.cos(dec_rad)

    sini_raw = params.get("SINI", 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == "KIN":
        sini_explicit = 0.0
    else:
        sini_explicit = float(sini_raw)

    return KopeikinStructure(
        kin_deg_ref=kin_deg_ref,
        kom_deg_ref=kom_deg_ref,
        px_mas_ref=px_mas_ref,
        pmra_ref=pmra_ref,
        pmdec_ref=pmdec_ref,
        pm_factor=pm_factor,
        pmra_keys=pmra_keys,
        pmdec_keys=pmdec_keys,
        use_k96=use_k96,
        has_parallax=has_parallax,
        is_ecliptic=is_ecliptic,
        sin_ra=sin_ra,
        cos_ra=cos_ra,
        sin_dec=sin_dec,
        cos_dec=cos_dec,
        obl_rad=obl_rad,
        sini_explicit=sini_explicit,
    )


def _compute_kopeikin_corrections_traceable(
    toas_bary_mjd,
    a1,
    t0,
    kin_deg,
    kom_deg,
    px_mas,
    pmra_rad_per_sec,
    pmdec_rad_per_sec,
    obs_pos_ls,
    struct: KopeikinStructure,
):
    """Compute DDK Kopeikin corrections with traceable numeric inputs."""
    toas_bary_mjd = jnp.asarray(toas_bary_mjd, dtype=jnp.float64)
    a1 = _as_f64(a1)
    t0 = _as_f64(t0)
    kin_deg = _as_f64(kin_deg)
    kom_deg = _as_f64(kom_deg)
    px_mas = _as_f64(px_mas)
    pmra_rad_per_sec = _as_f64(pmra_rad_per_sec)
    pmdec_rad_per_sec = _as_f64(pmdec_rad_per_sec)
    kin_rad = jnp.deg2rad(kin_deg)
    kom_rad = jnp.deg2rad(kom_deg)
    tt0_sec = (toas_bary_mjd - t0) * SECS_PER_DAY
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    delta_kin_pm = jnp.where(
        struct.use_k96,
        (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
        0.0,
    )
    kin_eff_rad = kin_rad + delta_kin_pm
    tan_safe = jnp.where(
        jnp.abs(jnp.tan(kin_eff_rad)) < 1e-10, 1e-10, jnp.tan(kin_eff_rad)
    )
    sin_safe = jnp.where(
        jnp.abs(jnp.sin(kin_eff_rad)) < 1e-10, 1e-10, jnp.sin(kin_eff_rad)
    )
    delta_a1_pm = jnp.where(struct.use_k96, a1 * delta_kin_pm / tan_safe, 0.0)
    delta_om_pm = jnp.where(
        struct.use_k96,
        (1.0 / sin_safe)
        * (pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom)
        * tt0_sec,
        0.0,
    )

    if obs_pos_ls is None:
        obs = jnp.zeros((toas_bary_mjd.shape[0], 3))
    else:
        obs = jnp.asarray(obs_pos_ls)
    if struct.is_ecliptic:
        c, s = jnp.cos(struct.obl_rad), jnp.sin(struct.obl_rad)
        obs = jnp.column_stack(
            [obs[:, 0], obs[:, 1] * c + obs[:, 2] * s, -obs[:, 1] * s + obs[:, 2] * c]
        )

    x, y, z = obs[:, 0], obs[:, 1], obs[:, 2]
    dI0 = -x * struct.sin_ra + y * struct.cos_ra
    dJ0 = (
        -x * struct.sin_dec * struct.cos_ra
        - y * struct.sin_dec * struct.sin_ra
        + z * struct.cos_dec
    )
    inv_d_ls = px_mas / (1000.0 * PC_TO_LIGHT_SEC)
    delta_a1_px = jnp.where(
        struct.has_parallax,
        (a1 / tan_safe) * inv_d_ls * (dI0 * sin_kom - dJ0 * cos_kom),
        0.0,
    )
    delta_om_px = jnp.where(
        struct.has_parallax,
        -(1.0 / sin_safe) * inv_d_ls * (dI0 * cos_kom + dJ0 * sin_kom),
        0.0,
    )

    delta_a1 = delta_a1_pm + delta_a1_px
    delta_om_deg = jnp.rad2deg(delta_om_pm) + jnp.rad2deg(delta_om_px)
    sini_eff = jnp.where(
        (struct.sini_explicit == 0.0) & (jnp.abs(kin_deg) > 0.0),
        jnp.sin(kin_eff_rad),
        struct.sini_explicit,
    )
    return delta_a1, delta_om_deg, sini_eff


def _compute_kopeikin_corrections(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    a1: float,
    t0: float,
    obs_pos_ls: jnp.ndarray = None,
):
    """Back-compatible wrapper around traceable Kopeikin kernel."""
    struct = resolve_kopeikin_flags(params)
    return _compute_kopeikin_corrections_traceable(
        jnp.asarray(toas_bary_mjd),
        _as_f64(a1),
        _as_f64(t0),
        struct.kin_deg_ref,
        struct.kom_deg_ref,
        struct.px_mas_ref,
        struct.pmra_ref * struct.pm_factor,
        struct.pmdec_ref * struct.pm_factor,
        obs_pos_ls,
        struct,
    )


def compute_dd_binary_delay(
    toas_bary_mjd: jnp.ndarray,
    params: Dict,
    **kwargs,
) -> jnp.ndarray:
    """Compute total DD binary delay.
    
    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    params : Dict
        DD model parameters
        
    Returns
    -------
    delay : jnp.ndarray
        Total binary delay in seconds
    """
    p = _extract_dd_params(params)
    tt0_ld = ((np.asarray(toas_bary_mjd, dtype=np.longdouble) - np.longdouble(p['t0']))
              * np.longdouble(SECS_PER_DAY))
    tt0_sec = np.asarray(tt0_ld, dtype=np.float64)
    has_shapiro = (p['sini'] > 0.0 and p['m2'] != 0.0)

    return _compute_dd_binary_delay_jit(
        jnp.asarray(tt0_sec),
        p['a1'], p['pb'], p['ecc'], p['om_deg'], p['omdot'],
        p['pbdot'], p['gamma'], p['sini'], p['m2'], p['xdot'], p['edot'],
        tt0_red_sec=jnp.asarray(reduce_binary_time_sec(tt0_ld, pb_days=p['pb'])),
        has_shapiro=has_shapiro,
    )


def compute_ddk_binary_delay(
    toas_bary_mjd: jnp.ndarray,
    params: Dict,
    obs_pos_ls: jnp.ndarray = None,
    **kwargs,
) -> jnp.ndarray:
    """Compute DDK binary delay (DD + Kopeikin corrections to A1/OM).

    Applies K96 proper motion and K95 annual orbital parallax corrections
    to the projected semi-major axis and longitude of periastron before
    computing the standard DD delay.

    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    params : Dict
        DDK model parameters (must include KIN, KOM)
    obs_pos_ls : jnp.ndarray, optional
        Observer position in light-seconds relative to SSB, shape (N, 3).
        Required for Kopeikin 1995 annual parallax corrections.

    Returns
    -------
    delay : jnp.ndarray
        Total binary delay in seconds
    """
    if _orthometric_values_active(params):
        raise NotImplementedError(
            "Orthometric Shapiro parameters (H3/H4/STIG) are not supported for "
            "DDK/Kopeikin binaries: DDK derives the inclination from KIN. "
            "Fit KIN and M2 instead."
        )
    p = _extract_dd_params(params)
    toas_bary_mjd_np = np.asarray(toas_bary_mjd)
    tt0_ld = ((np.asarray(toas_bary_mjd, dtype=np.longdouble) - np.longdouble(p['t0']))
              * np.longdouble(SECS_PER_DAY))
    tt0_sec = np.asarray(tt0_ld, dtype=np.float64)

    delta_a1, delta_om_deg, sini_eff = _compute_kopeikin_corrections(
        params, toas_bary_mjd, p['a1'], p['t0'], obs_pos_ls
    )

    # Per-TOA effective A1 and OM
    a1_eff = p['a1'] + delta_a1
    om_eff_deg = p['om_deg'] + delta_om_deg
    has_shapiro = (p['sini'] > 0.0 and p['m2'] != 0.0)

    return _compute_dd_binary_delay_jit(
        jnp.asarray(tt0_sec),
        a1_eff, p['pb'], p['ecc'], om_eff_deg, p['omdot'],
        p['pbdot'], p['gamma'], sini_eff, p['m2'], p['xdot'], p['edot'],
        tt0_red_sec=jnp.asarray(reduce_binary_time_sec(tt0_ld, pb_days=p['pb'])),
        has_shapiro=has_shapiro,
    )


@partial(jax.jit, static_argnames=("has_shapiro",))
def _compute_dd_binary_delay_jit(
    tt0_sec: jnp.ndarray,
    a1: float, pb: float, ecc: float, om_deg: float, omdot_deg_yr: float,
    pbdot: float, gamma: float, sini: float, m2: float,
    xdot: float, edot: float,
    tt0_red_sec: jnp.ndarray = None,
    has_shapiro: bool = True,
) -> jnp.ndarray:
    """JIT-compiled DD binary delay computation.

    tt0_sec is (toas_bary_mjd - T0) in seconds, precomputed via _compute_tt0_sec
    to avoid float64 catastrophic cancellation at MJD ~58000.
    tt0_red_sec (optional) is the longdouble orbit-count-reduced tt0
    (jug.utils.orbit_reduction.reduce_binary_time_sec with the same float64
    pb); when given, the fractional orbit comes from it, removing the ~ps
    float64 phase floor. norbits (secular OMDOT/Ae term only) keeps full tt0.
    """
    pb_sec = pb * SECS_PER_DAY

    # Apply secular changes to a1 and eccentricity
    a1_current = a1 + xdot * tt0_sec
    ecc_current = ecc + edot * tt0_sec

    # Mean anomaly: divide by pb_sec (single division, matching binary_dd.py kernel)
    # then reduce to [0, 2π) so ULP(M) ~ ULP(frac_orbits*2π) ≪ ULP(orbits*2π).
    orbits = tt0_sec / pb_sec - 0.5 * pbdot * (tt0_sec / pb_sec) ** 2
    norbits = jnp.floor(orbits)
    if tt0_red_sec is None:
        frac_orbits = orbits - norbits
    else:
        # Linear term from the reduced time; PBDOT quadratic from full tt0.
        # Differs from orbits by an integer (drops out of M); re-wrap to
        # [0, 1) to keep the nu/Ae branch structure identical.
        orbit_shift = jnp.rint((tt0_sec - tt0_red_sec) / pb_sec)
        orbits_hp = (orbit_shift + tt0_red_sec / pb_sec
                     - 0.5 * pbdot * (tt0_sec / pb_sec) ** 2)
        norbits = jnp.floor(orbits_hp)
        frac_orbits = orbits_hp - norbits
    M = 2.0 * jnp.pi * frac_orbits

    # Solve Kepler's equation for eccentric anomaly
    E = solve_kepler(M, ecc_current)

    # True anomaly
    theta = compute_true_anomaly(E, ecc_current)

    # Periastron advance: D&D 1986 eq [25]: omega = omega_0 + k*Ae
    # k = OMDOT / n (dimensionless); Ae = accumulated true anomaly
    Ae = 2.0 * jnp.pi * norbits + theta  # accumulated true anomaly
    k_omdot = omdot_deg_yr * pb / (360.0 * 365.25)
    om_rad = jnp.deg2rad(om_deg) + k_omdot * Ae

    # Roemer delay
    roemer = compute_dd_roemer_delay(E, theta, a1_current, ecc_current, om_rad)

    # Einstein delay
    einstein = compute_dd_einstein_delay(E, gamma, ecc_current)

    # Damour & Deruelle (1986) eq [52] proper-time -> coordinate-time inverse
    # correction. The kernel (jug/delays/binary_dd.py:dd_binary_delay) and PINT
    # both apply this; omitting it here previously left this standalone helper
    # ~200 us out of sync with the kernel for wide binaries (e.g. B1953+29).
    # Dre = Roemer + Einstein; alpha/beta are the D&D [46],[47] terms.
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    alpha = a1_current * jnp.sin(om_rad)
    beta = a1_current * jnp.sqrt(1.0 - ecc_current ** 2) * jnp.cos(om_rad)
    Dre = roemer + einstein
    Drep = -alpha * sinE + (beta + gamma) * cosE
    Drepp = -alpha * cosE - (beta + gamma) * sinE
    pb_prime_sec = pb_sec + pbdot * tt0_sec
    nhat = (2.0 * jnp.pi / pb_prime_sec) / (1.0 - ecc_current * cosE)
    correction_factor = (
        1.0
        - nhat * Drep
        + (nhat * Drep) ** 2
        + 0.5 * nhat ** 2 * Dre * Drepp
        - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat ** 2 * Dre * Drep
    )
    delay_inverse = Dre * correction_factor

    # Shapiro delay — structural presence is a static plan/wrapper flag.
    if has_shapiro:
        shapiro = compute_dd_shapiro_delay(E, theta, om_rad, ecc_current, sini, m2)
    else:
        shapiro = 0.0

    return delay_inverse + shapiro


# =============================================================================
# DD Model Derivatives
# =============================================================================

def compute_binary_derivatives_dd(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str]
) -> Dict[str, jnp.ndarray]:
    """Compute DD binary parameter derivatives.
    
    Uses hand-coded analytical derivatives, JIT-compiled with JAX.
    
    Parameters
    ----------
    params : Dict
        DD model parameters
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    fit_params : List[str]
        Parameters to compute derivatives for
        
    Returns
    -------
    derivatives : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to derivative arrays
    """
    toas_bary_mjd_np = np.asarray(toas_bary_mjd)
    
    # Extract base parameters
    a1 = float(params.get('A1', 0.0))
    pb = _resolve_pb_days(params)
    t0_ld = get_longdouble(
        params, 'T0', default=float(np.mean(toas_bary_mjd_np, dtype=np.float64))
    )
    # All DD derivative formulas depend on time only through (TOA - T0).
    # Shift to a relative-day coordinate before entering JAX: this preserves
    # the longdouble subtraction without exposing float128 scalars to JAX.
    toas_bary_mjd = jnp.asarray(
        _compute_tt0_sec(toas_bary_mjd_np, t0_ld) / SECS_PER_DAY
    )
    t0 = 0.0
    ecc = float(params.get('ECC', params.get('E', 0.0)))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))
    
    # Handle SINI - can be numeric or 'KIN' (DDK convention: SINI = sin(KIN))
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        kin_deg = float(params.get('KIN', 0.0))
        sini = float(jnp.sin(jnp.deg2rad(kin_deg)))
    else:
        sini = float(sini_raw)

    # DDS: SHAPMAX = -log(1 - sin i) (see _extract_dd_params).
    if sini == 0.0 and 'SHAPMAX' in params:
        sini = float(1.0 - np.exp(-float(params['SHAPMAX'])))

    # Apply XDOT/EDOT secular evolution to get effective per-TOA a1 and ecc
    xdot = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    edot = float(params.get('EDOT', 0.0))
    dt_sec = _compute_tt0_sec(np.asarray(toas_bary_mjd), t0)
    a1_eff = a1 + xdot * dt_sec
    ecc_eff = ecc + edot * dt_sec

    # Apply periastron advance for omega
    dt_yr = dt_sec / SECS_PER_YEAR
    om_rad = (om_deg + omdot * dt_yr) * DEG_TO_RAD
    
    derivatives = {}
    
    for param in fit_params:
        param_upper = param.upper()
        
        if param_upper == 'A1':
            # d(delay)/d(A1) - simple scaling
            deriv = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'PB':
            deriv = _d_delay_d_PB(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv  # Already in s/day units
            
        elif param_upper == 'T0':
            deriv = _d_delay_d_T0(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv  # Already in s/day units
            
        elif param_upper == 'ECC':
            deriv = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, pbdot, gamma, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'OM':
            deriv = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv * DEG_TO_RAD  # Convert to per-degree units
            
        elif param_upper == 'PBDOT':
            deriv = _d_delay_d_PBDOT(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'GAMMA':
            deriv = _d_delay_d_GAMMA(toas_bary_mjd, pb, t0, ecc_eff, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'SINI':
            deriv = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'M2':
            deriv = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, sini)
            derivatives[param] = deriv

        elif param_upper == 'SHAPMAX':
            # DDS: d(delay)/d(SHAPMAX) = d(delay)/d(SINI) * d(SINI)/d(SHAPMAX),
            # SINI = 1 - exp(-SHAPMAX) -> d(SINI)/d(SHAPMAX) = exp(-SHAPMAX)
            # = 1 - SINI. (PINT DDS_model.d_SINI_d_SHAPMAX = exp(-SHAPMAX).)
            d_sini = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, sini, m2)
            derivatives[param] = d_sini * (1.0 - sini)

        elif param_upper == 'H3':
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            h4_val = float(params.get('H4', 0.0))
            if stig_val > 0.0:
                if h4_val != 0.0:
                    warnings.warn(
                        "Both STIG and H4 are nonzero; using H3/STIG parameterization (H4 ignored)",
                        UserWarning, stacklevel=2
                    )
                # DDH model: H3/STIG parameterization — valid STIG
                deriv = _d_delay_d_H3(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, stig_val)
            elif h3_val != 0.0 and h4_val != 0.0 and (h4_val / h3_val) > 0.0:
                # H3/H4 parameterization — same-sign pair
                deriv = _d_delay_d_H3_h3h4(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, h3_val, h4_val)
            else:
                # Invalid STIG — Shapiro delay is unconstrained, return zero derivative
                if h3_val != 0.0:
                    warnings.warn(
                        "H3/H4 parameterization with H4=0: M2 is ill-conditioned; derivative will be zero",
                        UserWarning, stacklevel=2
                    )
                deriv = jnp.zeros_like(toas_bary_mjd)
            derivatives[param] = deriv

        elif param_upper in ('STIG', 'STIGMA'):
            # DDH model: H3/STIG parameterization
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            if stig_val > 0.0 and h3_val != 0.0:
                deriv = _d_delay_d_STIG(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, h3_val, stig_val)
            else:
                deriv = jnp.zeros_like(toas_bary_mjd)
            derivatives[param] = deriv

        elif param_upper == 'OMDOT':
            deriv = _d_delay_d_OMDOT(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_deg, omdot, pbdot, sini, m2)
            # OMDOT is in deg/yr, convert appropriately
            derivatives[param] = deriv
            
        elif param_upper == 'XDOT' or param_upper == 'A1DOT':
            # d(delay)/d(XDOT) = d(delay)/d(A1) * dt_sec (chain rule through a1_eff)
            d_a1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot)
            derivatives[param] = d_a1 * dt_sec

        elif param_upper == 'EDOT':
            # d(delay)/d(EDOT) = d(delay)/d(ECC) * dt_sec (chain rule through ecc_eff)
            d_ecc = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad, pbdot, gamma, sini, m2)
            derivatives[param] = d_ecc * dt_sec

        elif param_upper == 'H4':
            # Orthometric Shapiro parameter H4 (DD/DDH model, H3/H4 parameterization)
            h3 = float(params.get('H3', 0.0))
            h4 = float(params.get('H4', 0.0))
            if h3 != 0.0 and h4 != 0.0 and (h4 / h3) > 0.0:
                deriv = _d_delay_d_H4(toas_bary_mjd, pb, t0, ecc_eff, om_rad, pbdot, h3, h4)
            else:
                # Invalid STIG — return zero derivative
                deriv = jnp.zeros_like(toas_bary_mjd)
            derivatives[param] = deriv

    return derivatives


# =============================================================================
# Individual Derivative Functions (analytical, JIT-compiled)
# =============================================================================

@jax.jit
def _d_delay_d_A1(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray, pbdot: float
) -> jnp.ndarray:
    """d(Roemer delay)/d(A1) = Roemer_delay / A1"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # d(Roemer)/d(A1) = (Roemer/A1) since Roemer ~ A1
    return sin_omega * (jnp.cos(E) - ecc) + cos_omega * sqrt_1_e2 * jnp.sin(E)


@jax.jit
def _d_delay_d_PB(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(PB) via chain rule through mean anomaly."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(PB) = -2pi * dt / PB^2 * (1 - PBDOT * dt / PB)
    dM_dPB = -2 * jnp.pi * dt / pb**2 * (1 - pbdot * dt / pb)
    
    # Need d(delay)/d(M) = d(delay)/d(E) * d(E)/d(M)
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    # d(E)/d(M) = 1 / (1 - ecc*cos(E))
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    # d(Roemer)/d(E)
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    # d(Einstein)/d(E) = GAMMA * cos(E) -- but GAMMA doesn't depend on PB in first order
    
    # Shapiro derivative through theta
    # d(theta)/d(E) = sqrt(1-e^2) / (1 - e*cos(E))
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    
    # d(Shapiro)/d(theta) for the (1 - s sin(omega+theta)) factor of the DD
    # Shapiro log-arg, where r = T_Sun M2, s = sin(i):
    # d/dtheta[-2r ln(1 - s sin(omega+theta))] = 2r s cos(omega+theta) / [1 - s sin(omega+theta)]
    #
    # NOTE: PINT's DD model includes this cos(omega+theta) factor — its
    # d_delayS_d_par chains through omega/E and DD_model.dsDelay_domega equals
    # 2*TM2*SINI*cos(omega+theta)/(1-s*sin(omega+theta)). The cos-factor omission
    # is specific to PINT's *ELL1* base model (ELL1_model.d_delayS_d_Phi drops
    # cos(Phi)); PINT's ELL1H model includes it. JUG includes it everywhere.
    #
    # Wolfram Alpha: d/dx [-2*a*ln(1 - b*sin(x))]  ->  2*a*b*cos(x)/(1-b*sin(x))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = 1 - sini * sin_omega_theta
    denom = jnp.maximum(denom, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom

    # Full DD Shapiro log-arg factors as (1 - e*cosE)(1 - s*sin(omega+theta)); the
    # (1 - e*cosE) factor's E-dependence (-2r * e*sinE/(1 - e*cosE)) was previously
    # dropped. Include it so d(Shapiro)/dE is the complete D&D 1986 eq.[26] derivative
    # (matches dlogArg_dE in _d_delay_d_ECC and PINT DD_model.dsDelay_dE).
    dShapiro_dE = (dShapiro_dtheta * dtheta_dE
                   - 2 * r * ecc * jnp.sin(E) / (1 - ecc * jnp.cos(E)))

    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dPB


@jax.jit
def _d_delay_d_T0(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(T0) via chain rule."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(T0) = -2pi/PB * (1 - PBDOT * dt / PB)
    dM_dT0 = -2 * jnp.pi / pb * (1 - pbdot * dt / pb)
    
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    # Shapiro
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom
    # Include the (1 - e*cosE) factor's E-dependence (full D&D 1986 eq.[26]
    # Shapiro log-arg); previously dropped. See _d_delay_d_PB for the derivation.
    dShapiro_dE = (dShapiro_dtheta * dtheta_dE
                   - 2 * r * ecc * jnp.sin(E) / (1 - ecc * jnp.cos(E)))

    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dT0


@jax.jit 
def _d_delay_d_ECC(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, gamma: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(ECC) - includes Roemer, Einstein, and Shapiro terms."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # Chain rule: d(E)/d(ecc) at fixed M
    # From E - ecc*sin(E) = M: d(E)/de = sin(E) / (1 - ecc*cos(E))
    oneMecccosE = 1 - ecc * cosE
    dE_de = sinE / oneMecccosE
    
    # --- Roemer term ---
    # d(Roemer)/d(ecc) has two parts:
    # 1. Direct: d/de[sin(om)*(cos(E)-e) + cos(om)*sqrt(1-e^2)*sin(E)]
    # 2. Chain rule through E: d(Roemer)/d(E) * d(E)/d(ecc)
    dRoemer_de_direct = a1 * (
        -sin_omega  # from d(-ecc)/de
        - cos_omega * ecc / sqrt_1_e2 * sinE  # from d(sqrt(1-e^2))/de
    )
    dRoemer_dE = a1 * (-sin_omega * sinE + cos_omega * sqrt_1_e2 * cosE)
    dRoemer_de = dRoemer_de_direct + dRoemer_dE * dE_de
    
    # --- Einstein term ---
    # d(GAMMA*sin(E))/d(ecc) = GAMMA * cos(E) * d(E)/d(ecc)
    dEinstein_de = gamma * cosE * dE_de
    
    # --- Shapiro term ---
    # Shapiro delay: -2*r*ln(1 - e*cosE - s*(sin(om)*(cosE-e) + sqrt(1-e^2)*cos(om)*sinE))
    # where r = T_SUN * M2, s = SINI
    # d(Shapiro)/d(ecc) = d(Shapiro)/d(ecc)|_E + d(Shapiro)/d(E) * dE/de
    r = T_SUN * m2
    logArg = 1 - ecc * cosE - sini * (sin_omega * (cosE - ecc) + sqrt_1_e2 * cos_omega * sinE)
    logArg = jnp.maximum(logArg, 1e-10)
    
    # Direct partial (holding E constant):
    # d(logArg)/de = -cosE - sini*(-sin(om) - e*cos(om)*sinE/sqrt(1-e^2))
    #              = -cosE - sini*(-sin(om) + e/(sqrt(1-e^2))*(-cos(om)*sinE))
    dlogArg_de = -cosE - sini * (-sin_omega - ecc * cos_omega * sinE / sqrt_1_e2)
    dShapiro_de_direct = -2 * r * dlogArg_de / logArg
    
    # Chain rule through E:
    # d(logArg)/dE = e*sinE - sini*(sqrt(1-e^2)*cosE*cos(om) - sinE*sin(om))
    dlogArg_dE = ecc * sinE - sini * (sqrt_1_e2 * cosE * cos_omega - sinE * sin_omega)
    dShapiro_dE = -2 * r * dlogArg_dE / logArg
    
    dShapiro_de = dShapiro_de_direct + dShapiro_dE * dE_de
    
    return dRoemer_de + dEinstein_de + dShapiro_de


@jax.jit
def _d_delay_d_OM(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(OM) - affects Roemer and Shapiro."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # d(Roemer)/d(omega) = a1 * (cos(om)*(cos(E)-e) - sin(om)*sqrt(1-e^2)*sin(E))
    dRoemer_dom = a1 * (cos_omega * (jnp.cos(E) - ecc) - sin_omega * sqrt_1_e2 * jnp.sin(E))
    
    # d(Shapiro)/d(omega) = 2*r*sini*cos(om+theta) / (1 - sini*sin(om+theta))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dom = 2 * r * sini * cos_omega_theta / denom
    
    return dRoemer_dom + dShapiro_dom


@jax.jit
def _d_delay_d_PBDOT(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(PBDOT) via chain rule."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(PBDOT) = -pi * dt^2 / PB^2
    dM_dPBDOT = -jnp.pi * dt**2 / pb**2
    
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, 0.0)  # Use PBDOT=0 for base
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom
    # Include the (1 - e*cosE) factor's E-dependence (full D&D 1986 eq.[26]
    # Shapiro log-arg); previously dropped. See _d_delay_d_PB for the derivation.
    dShapiro_dE = (dShapiro_dtheta * dtheta_dE
                   - 2 * r * ecc * jnp.sin(E) / (1 - ecc * jnp.cos(E)))

    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dPBDOT


@jax.jit
def _d_delay_d_GAMMA(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, pbdot: float
) -> jnp.ndarray:
    """d(delay)/d(GAMMA) = sin(E)"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    return jnp.sin(E)


@jax.jit
def _d_delay_d_SINI(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(Shapiro)/d(SINI)"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    
    return 2 * r * sin_omega_theta / denom


@jax.jit
def _d_delay_d_M2(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float
) -> jnp.ndarray:
    """d(Shapiro)/d(M2) = Shapiro / M2 = -2 T_SUN log(arg).

    Must use the FULL Damour & Deruelle (1986) eq.[26] argument that the
    forward model (compute_dd_shapiro_delay) uses, not the simplified ELL1
    form 1 - s*sin(omega+theta). The DD argument factors as
        1 - e cosE - s[sinw(cosE-e) + sqrt(1-e^2)cosw sinE]
            = (1 - e cosE) * (1 - s sin(omega+theta)),
    so the simplified form drops the -2 T_SUN log(1 - e cosE) term. That
    omission made the M2 design column ~3% off per-TOA, which the
    eccentric DD block amplified into a sign-oscillating ~0.73 linear
    convergence (M2 carried the whole residual step: 27 iters on J1946+3417
    even when started AT the solution). The (1-e cosE) factor cancels in
    d/dSINI, so only M2 was affected."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)

    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    sqrt_1_e2 = jnp.sqrt(1 - ecc ** 2)
    arg = jnp.maximum(
        1 - ecc * cos_E
        - sini * (jnp.sin(om_rad) * (cos_E - ecc)
                  + sqrt_1_e2 * jnp.cos(om_rad) * sin_E),
        1e-10)

    return -2 * T_SUN * jnp.log(arg)


@jax.jit
def _d_delay_d_OMDOT(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_deg: float, omdot: float,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(OMDOT) = d(delay)/d(om_rad) * d(om_rad)/d(OMDOT).

    The forward model advances periastron as omega = omega_0 + k*Ae with
    k = OMDOT * pb / (360 * 365.25) and Ae the accumulated TRUE anomaly
    (D&D 1986 eq.[25]; see _compute_dd_binary_delay_jit), so

        d(om_rad)/d(OMDOT) = (pb / (360 * 365.25)) * Ae.

    The previous implementation used d(om_rad)/d(OMDOT) = dt_yr * DEG_TO_RAD,
    a linear-in-time approximation that is exact only for circular orbits.
    For eccentric orbits it disagrees with the forward model (and PINT) by the
    per-orbit true-anomaly oscillation (~0.16% on J1903+0327, ecc~0.44), which
    the near-degenerate Keplerian block amplifies into ~10-16% uncertainty
    errors on A1/T0/OM/ECC. Computing Ae here keeps the OMDOT column consistent
    with the delay it differentiates."""
    # Accumulated true anomaly Ae = 2*pi*norbits + theta, mirroring the forward
    # kernel. toas_bary_mjd is (t - T0) in days here (t0 == 0 in caller coords).
    dt = toas_bary_mjd - t0  # days
    orbits = dt / pb * (1.0 - 0.5 * pbdot * dt / pb)
    norbits = jnp.floor(orbits)
    M = 2.0 * jnp.pi * (orbits - norbits)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    Ae = 2.0 * jnp.pi * norbits + theta

    # Operating-point omega consistent with the forward model's Ae evolution.
    k_omdot = omdot * pb / (360.0 * 365.25)
    om_rad = om_deg * DEG_TO_RAD + k_omdot * Ae

    d_om_rad = _d_delay_d_OM(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, sini, m2)

    # OMDOT in deg/yr: d(om_rad)/d(OMDOT) = (pb / (360 * 365.25)) * Ae.
    return d_om_rad * (pb / (360.0 * 365.25)) * Ae


# =============================================================================
# DDK (Kopeikin 1995/1996) Partial Derivatives for KIN and KOM
# =============================================================================
# The DDK model applies corrections to A1 and OM based on:
#   1. K96 proper motion corrections (Kopeikin 1996)
#   2. Annual orbital parallax corrections (Kopeikin 1995)
#
# The total derivatives use the chain rule:
#   d(delay)/d(KIN) = d(delay)/d(A1_eff) * d(A1_eff)/d(KIN) 
#                   + d(delay)/d(OM_eff) * d(OM_eff)/d(KIN)
#                   + d(delay)/d(SINI_eff) * d(SINI_eff)/d(KIN)
#
# where A1_eff = A1 + delta_A1_pm + delta_A1_px
#       OM_eff = OM + delta_OM_pm + delta_OM_px  (in degrees)
#       SINI_eff = sin(KIN_eff) for DDK when SINI not explicitly set
#
# References:
#   - Kopeikin 1995: Annual orbital parallax
#   - Kopeikin 1996: Proper motion (K96) corrections
#   - PINT src/pint/models/binary_ddk.py for implementation details


@jax.jit
def _compute_ddk_correction_derivatives_KIN(
    tt0_sec: jnp.ndarray,
    a1: float,
    kin_rad: float,
    kom_rad: float,
    pmra_rad_per_sec: float,
    pmdec_rad_per_sec: float,
    delta_I0: jnp.ndarray,
    delta_J0: jnp.ndarray,
    d_ls: float,
    use_k96: bool,
    has_parallax: bool
) -> tuple:
    """
    Compute d(delta_A1)/d(KIN) and d(delta_OM)/d(KIN) for DDK corrections.
    
    The K96 proper motion corrections (Kopeikin 1996) are:
        delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * (t - T0)
        delta_A1_pm = A1 * delta_KIN_pm / tan(KIN_eff)
        delta_OM_pm = (1/sin(KIN_eff)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * (t - T0)
    
    The Kopeikin 1995 parallax corrections are:
        delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
        delta_OM_px = -(1 / sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    
    For the chain rule, we need:
        d(delta_A1_pm)/d(KIN) - involves d/d(KIN)[delta_KIN_pm / tan(KIN_eff)]
        d(delta_OM_pm)/d(KIN) - involves d/d(KIN)[1/sin(KIN_eff)]
        d(delta_A1_px)/d(KIN) - involves d/d(KIN)[1/tan(KIN)]
        d(delta_OM_px)/d(KIN) - involves d/d(KIN)[1/sin(KIN)]
    
    Returns
    -------
    d_A1_eff_d_KIN : array
        d(A1_eff)/d(KIN) in light-seconds per radian
    d_OM_eff_d_KIN : array
        d(OM_eff)/d(KIN) in radians per radian (dimensionless)
    d_SINI_eff_d_KIN : array
        d(SINI_eff)/d(KIN) in 1/radian
    """
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    sin_kin = jnp.sin(kin_rad)
    cos_kin = jnp.cos(kin_rad)
    sin2_kin = sin_kin ** 2
    
    # K96 proper motion corrections
    # delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0_sec
    pm_term = -pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom
    delta_kin_pm = jnp.where(use_k96, pm_term * tt0_sec, 0.0)
    
    # d(delta_KIN_pm)/d(KIN) = 0 (no explicit KIN dependence in the definition)
    
    # delta_A1_pm = A1 * delta_KIN_pm / tan(KIN_eff)
    # where KIN_eff = KIN + delta_KIN_pm
    # Approximate: d(KIN_eff)/d(KIN) ~= 1 (delta_KIN_pm doesn't depend on KIN)
    # d(A1 * delta_KIN / tan(KIN))/d(KIN) = A1 * delta_KIN * d(1/tan(KIN))/d(KIN)
    #                                      = A1 * delta_KIN * (-1/sin^2(KIN))
    d_A1_pm_d_KIN = jnp.where(
        use_k96,
        -a1 * delta_kin_pm / sin2_kin,
        0.0
    )
    
    # delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0_sec
    # d/d(KIN)[1/sin(KIN)] = -cos(KIN)/sin^2(KIN)
    pm_omega_term = pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom
    d_OM_pm_d_KIN = jnp.where(
        use_k96,
        -cos_kin / sin2_kin * pm_omega_term * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995 parallax corrections
    # delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
    # d/d(KIN)[A1 / tan(KIN) / d] = A1 / d * (-1/sin^2(KIN))
    parallax_a1_term = delta_I0 * sin_kom - delta_J0 * cos_kom
    d_A1_px_d_KIN = jnp.where(
        has_parallax,
        -a1 / d_ls / sin2_kin * parallax_a1_term,
        0.0
    )
    
    # delta_OM_px = -(1/sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    # d/d(KIN)[1/sin(KIN)] = -cos(KIN)/sin^2(KIN)
    parallax_om_term = delta_I0 * cos_kom + delta_J0 * sin_kom
    d_OM_px_d_KIN = jnp.where(
        has_parallax,
        cos_kin / sin2_kin / d_ls * parallax_om_term,  # Note: negative from chain rule cancels original negative
        0.0
    )
    
    # Total derivatives
    d_A1_eff_d_KIN = d_A1_pm_d_KIN + d_A1_px_d_KIN
    d_OM_eff_d_KIN = d_OM_pm_d_KIN + d_OM_px_d_KIN  # in radians/radian
    
    # SINI_eff = sin(KIN_eff) where KIN_eff ~= KIN for small corrections
    # d(sin(KIN))/d(KIN) = cos(KIN)
    d_SINI_eff_d_KIN = cos_kin
    
    return d_A1_eff_d_KIN, d_OM_eff_d_KIN, d_SINI_eff_d_KIN


@jax.jit
def _compute_ddk_correction_derivatives_KOM(
    tt0_sec: jnp.ndarray,
    a1: float,
    kin_rad: float,
    kom_rad: float,
    pmra_rad_per_sec: float,
    pmdec_rad_per_sec: float,
    delta_I0: jnp.ndarray,
    delta_J0: jnp.ndarray,
    d_ls: float,
    use_k96: bool,
    has_parallax: bool
) -> tuple:
    """
    Compute d(delta_A1)/d(KOM) and d(delta_OM)/d(KOM) for DDK corrections.
    
    K96 proper motion:
        delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0
        delta_A1_pm = A1 * delta_KIN_pm / tan(KIN)
        delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0
    
    Kopeikin 1995 parallax:
        delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
        delta_OM_px = -(1/sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    
    Returns
    -------
    d_A1_eff_d_KOM : array
        d(A1_eff)/d(KOM) in light-seconds per radian
    d_OM_eff_d_KOM : array
        d(OM_eff)/d(KOM) in radians per radian (dimensionless)
    """
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    sin_kin = jnp.sin(kin_rad)
    tan_kin = jnp.tan(kin_rad)
    
    # Safe denominators
    sin_kin_safe = jnp.where(jnp.abs(sin_kin) < 1e-10, 1e-10, sin_kin)
    tan_kin_safe = jnp.where(jnp.abs(tan_kin) < 1e-10, 1e-10, tan_kin)
    
    # K96 proper motion: delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0
    # d(delta_KIN_pm)/d(KOM) = (-mu_RA * cos(KOM) - mu_DEC * sin(KOM)) * tt0
    d_delta_kin_pm_d_KOM = jnp.where(
        use_k96,
        (-pmra_rad_per_sec * cos_kom - pmdec_rad_per_sec * sin_kom) * tt0_sec,
        0.0
    )
    
    # delta_A1_pm = A1 * delta_KIN_pm / tan(KIN)
    # d(delta_A1_pm)/d(KOM) = A1 / tan(KIN) * d(delta_KIN_pm)/d(KOM)
    d_A1_pm_d_KOM = jnp.where(
        use_k96,
        a1 / tan_kin_safe * d_delta_kin_pm_d_KOM,
        0.0
    )
    
    # delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0
    # d/d(KOM)[mu_RA * cos(KOM) + mu_DEC * sin(KOM)] = -mu_RA * sin(KOM) + mu_DEC * cos(KOM)
    d_OM_pm_d_KOM = jnp.where(
        use_k96,
        (1.0 / sin_kin_safe) * (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995: delta_A1_px = (A1/tan(KIN)/d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
    # d/d(KOM)[delta_I0 * sin(KOM) - delta_J0 * cos(KOM)] = delta_I0 * cos(KOM) + delta_J0 * sin(KOM)
    d_A1_px_d_KOM = jnp.where(
        has_parallax,
        a1 / tan_kin_safe / d_ls * (delta_I0 * cos_kom + delta_J0 * sin_kom),
        0.0
    )
    
    # delta_OM_px = -(1/sin(KIN)/d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    # d/d(KOM)[delta_I0 * cos(KOM) + delta_J0 * sin(KOM)] = -delta_I0 * sin(KOM) + delta_J0 * cos(KOM)
    d_OM_px_d_KOM = jnp.where(
        has_parallax,
        -(1.0 / sin_kin_safe / d_ls) * (-delta_I0 * sin_kom + delta_J0 * cos_kom),
        0.0
    )
    
    # Total derivatives
    d_A1_eff_d_KOM = d_A1_pm_d_KOM + d_A1_px_d_KOM
    d_OM_eff_d_KOM = d_OM_pm_d_KOM + d_OM_px_d_KOM  # in radians/radian
    
    # SINI_eff = sin(KIN_eff) where KIN_eff = KIN + delta_KIN_pm(KOM)
    # d(SINI_eff)/d(KOM) = cos(KIN_eff) * d(delta_KIN_pm)/d(KOM)
    cos_kin = jnp.cos(kin_rad)
    d_SINI_eff_d_KOM = jnp.where(use_k96, cos_kin * d_delta_kin_pm_d_KOM, 0.0)
    
    return d_A1_eff_d_KOM, d_OM_eff_d_KOM, d_SINI_eff_d_KOM


def compute_binary_derivatives_ddk(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str],
    obs_pos_ls: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """
    Compute DDK binary parameter derivatives including KIN and KOM.
    
    DDK extends DD with Kopeikin corrections. For standard DD parameters,
    we use the DD derivatives evaluated at the effective A1/OM values.
    For KIN and KOM, we use the chain rule through the Kopeikin corrections.
    
    Parameters
    ----------
    params : Dict
        DDK model parameters (must include KIN, KOM, and optionally PX, PMRA, PMDEC)
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    fit_params : List[str]
        Parameters to compute derivatives for
    obs_pos_ls : jnp.ndarray, optional
        Observer position in light-seconds relative to SSB, shape (N, 3).
        Required for Kopeikin 1995 parallax corrections.
        
    Returns
    -------
    derivatives : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to derivative arrays
    """
    fit_params_upper = [p.upper() for p in fit_params]
    if _orthometric_values_active(params) or any(
        name in fit_params_upper for name in ("H3", "H4", "STIG", "STIGMA")
    ):
        raise NotImplementedError(
            "Orthometric Shapiro parameters (H3/H4/STIG) are not supported for "
            "DDK/Kopeikin binaries: DDK derives the inclination from KIN. "
            "Fit KIN and M2 instead."
        )
    toas_bary_mjd_np = np.asarray(toas_bary_mjd)
    n_toas = len(toas_bary_mjd_np)
    
    # Extract base DD parameters
    a1 = float(params.get('A1', 0.0))
    pb = _resolve_pb_days(params)
    t0_ld = get_longdouble(
        params, 'T0', default=float(np.mean(toas_bary_mjd_np, dtype=np.float64))
    )
    toas_bary_mjd = jnp.asarray(
        _compute_tt0_sec(toas_bary_mjd_np, t0_ld) / SECS_PER_DAY
    )
    t0 = 0.0
    ecc = float(params.get('ECC', params.get('E', 0.0)))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))
    
    # Apply EDOT secular evolution to get effective per-TOA ecc
    edot = float(params.get('EDOT', 0.0))
    dt_sec_from_t0 = _compute_tt0_sec(np.asarray(toas_bary_mjd), t0)
    ecc_eff = ecc + edot * dt_sec_from_t0
    
    # DDK-specific parameters
    kin_deg = float(params.get('KIN', 0.0))
    
    kom_deg = float(params.get('KOM', 0.0))
    kin_rad = jnp.deg2rad(kin_deg)
    kom_rad = jnp.deg2rad(kom_deg)
    
    px_mas = float(params.get('PX', 0.0))
    
    # Proper motion (for K96).
    # For ecliptic pulsars, use PMELONG/PMELAT (stored as _ecliptic_pm_lon/lat)
    # so that the K96 formula uses the same coordinate frame as KOM.
    MAS_PER_YR_TO_RAD_PER_SEC = (jnp.pi / 180.0 / 3600.0 / 1000.0) / SECS_PER_YEAR
    _is_ecliptic = bool(params.get('_ecliptic_coords', False))
    if _is_ecliptic:
        pmra_mas_yr = float(params.get('_ecliptic_pm_lon', 0.0))   # PMELONG
        pmdec_mas_yr = float(params.get('_ecliptic_pm_lat', 0.0))  # PMELAT
    else:
        pmra_mas_yr = float(params.get('PMRA', 0.0))
        pmdec_mas_yr = float(params.get('PMDEC', 0.0))
    pmra_rad_per_sec = pmra_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    pmdec_rad_per_sec = pmdec_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    
    # K96 flag
    k96_flag = True
    if 'K96' in params:
        k96_param = params['K96']
        if isinstance(k96_param, bool):
            k96_flag = k96_param
        elif isinstance(k96_param, str):
            k96_flag = k96_param.upper() not in ('N', 'NO', 'FALSE', '0', 'F')
        else:
            k96_flag = bool(k96_param)
    use_k96 = k96_flag and (pmra_mas_yr != 0 or pmdec_mas_yr != 0)

    # Structural gate (match resolve_kopeikin_flags): KIN defines the sector;
    # PX value only scales the linear correction.
    has_parallax = abs(kin_deg) > 0.0

    # Linear signed factor (same form as the autodiff Kopeikin path).
    inv_d_ls = px_mas / (1000.0 * PC_TO_LIGHT_SEC)
    d_ls = 1.0 / inv_d_ls if inv_d_ls != 0.0 else float("inf")
    
    # Time since T0
    tt0_sec = _compute_tt0_sec(np.asarray(toas_bary_mjd), t0)
    
    # Observer position for Kopeikin projections.
    # For ecliptic pulsars, rotate ICRS obs_pos to ecliptic frame.
    if obs_pos_ls is None:
        obs_pos_ls = jnp.zeros((n_toas, 3))
    obs_pos_ls = jnp.asarray(obs_pos_ls)
    if _is_ecliptic:
        from jug.io.par_reader import OBLIQUITY_ARCSEC
        _ecl_frame = str(params.get('_ecliptic_frame', 'IERS2010')).upper()
        _obl_rad = OBLIQUITY_ARCSEC.get(_ecl_frame, OBLIQUITY_ARCSEC['IERS2010']) * float(jnp.pi) / (180.0 * 3600.0)
        _cos_obl = jnp.cos(_obl_rad)
        _sin_obl = jnp.sin(_obl_rad)
        _x = obs_pos_ls[:, 0]
        _y = obs_pos_ls[:, 1] * _cos_obl + obs_pos_ls[:, 2] * _sin_obl
        _z = -obs_pos_ls[:, 1] * _sin_obl + obs_pos_ls[:, 2] * _cos_obl
        obs_pos_ls = jnp.column_stack([_x, _y, _z])

    # Get pulsar position for K95 projections (delta_I0, delta_J0).
    # For ecliptic pulsars, use ecliptic lon/lat instead of RA/DEC so that
    # the projections are consistent with the ecliptic KOM frame.
    if _is_ecliptic:
        _ecl_lon_rad = float(jnp.pi) / 180.0 * float(params.get('_ecliptic_lon_deg', 0.0))
        _ecl_lat_rad = float(jnp.pi) / 180.0 * float(params.get('_ecliptic_lat_deg', 0.0))
        sin_ra = jnp.sin(_ecl_lon_rad)
        cos_ra = jnp.cos(_ecl_lon_rad)
        sin_dec = jnp.sin(_ecl_lat_rad)
        cos_dec = jnp.cos(_ecl_lat_rad)
    else:
        # Handle both radians (float) and sexagesimal strings
        from jug.io.par_reader import parse_ra, parse_dec
        raj_val = params.get('RAJ', 0.0)
        decj_val = params.get('DECJ', 0.0)

        if isinstance(raj_val, str) and ':' in raj_val:
            ra_rad = parse_ra(raj_val)
        else:
            ra_rad = float(raj_val)

        if isinstance(decj_val, str) and ':' in decj_val:
            dec_rad = parse_dec(decj_val)
        else:
            dec_rad = float(decj_val)

        sin_ra = jnp.sin(ra_rad)
        cos_ra = jnp.cos(ra_rad)
        sin_dec = jnp.sin(dec_rad)
        cos_dec = jnp.cos(dec_rad)
    
    # Kopeikin projection terms (per-TOA)
    x = obs_pos_ls[:, 0]
    y = obs_pos_ls[:, 1]
    z = obs_pos_ls[:, 2]
    delta_I0 = -x * sin_ra + y * cos_ra
    delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
    
    # Compute effective parameters (matching combined.py branch_ddk)
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    
    # K96 corrections
    delta_kin_pm = jnp.where(
        use_k96,
        (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
        0.0
    )
    kin_eff_rad = kin_rad + delta_kin_pm
    
    tan_kin_eff = jnp.tan(kin_eff_rad)
    tan_kin_eff_safe = jnp.where(jnp.abs(tan_kin_eff) < 1e-10, 1e-10, tan_kin_eff)
    sin_kin_eff = jnp.sin(kin_eff_rad)
    sin_kin_eff_safe = jnp.where(jnp.abs(sin_kin_eff) < 1e-10, 1e-10, sin_kin_eff)
    
    delta_a1_pm = jnp.where(use_k96, a1 * delta_kin_pm / tan_kin_eff_safe, 0.0)
    delta_omega_pm_rad = jnp.where(
        use_k96,
        (1.0 / sin_kin_eff_safe) * (pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom) * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995 parallax corrections
    delta_a1_px = jnp.where(
        has_parallax,
        (a1 / tan_kin_eff_safe / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom),
        0.0
    )
    delta_omega_px_rad = jnp.where(
        has_parallax,
        -(1.0 / sin_kin_eff_safe / d_ls) * (delta_I0 * cos_kom + delta_J0 * sin_kom),
        0.0
    )
    
    # Effective parameters (Kopeikin + XDOT)
    xdot = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    a1_eff = a1 + delta_a1_pm + delta_a1_px + xdot * dt_sec_from_t0
    om_eff_deg = om_deg + jnp.rad2deg(delta_omega_pm_rad) + jnp.rad2deg(delta_omega_px_rad)
    
    # For SINI: use sin(KIN_eff) if SINI not explicitly set or if SINI='KIN'
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        # DDK convention: SINI derived from KIN
        sini_explicit = 0.0  # Treat as not explicitly set
    else:
        sini_explicit = float(sini_raw)
    sini_eff = jnp.where(
        (sini_explicit == 0.0) & (jnp.abs(kin_deg) > 0.0),
        jnp.sin(kin_eff_rad),
        sini_explicit
    )
    
    # Get the base DD derivatives evaluated at effective parameters
    # We need to handle the time-varying omega
    dt_yr = (toas_bary_mjd - t0) / 365.25
    om_rad_eff = (om_eff_deg + omdot * dt_yr) * DEG_TO_RAD
    
    derivatives = {}
    
    # Check which parameters need KIN/KOM-specific handling
    needs_kin = 'KIN' in fit_params_upper
    needs_kom = 'KOM' in fit_params_upper
    
    # First, handle standard DD parameters using effective values
    dd_params = [p for p in fit_params if p.upper() not in ('KIN', 'KOM')]
    
    for param in dd_params:
        param_upper = param.upper()
        
        if param_upper == 'A1':
            # Get base derivative for A1
            deriv = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot)
            
            # Adjustment factor for effective A1 dependence on A1
            d_A1_eff_d_A1 = 1.0
            if use_k96:
                d_A1_eff_d_A1 = d_A1_eff_d_A1 + delta_kin_pm / tan_kin_eff_safe
            if has_parallax:
                d_A1_eff_d_A1 = d_A1_eff_d_A1 + (1.0 / tan_kin_eff_safe / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom)
            
            derivatives[param] = deriv * d_A1_eff_d_A1
            
        elif param_upper == 'PB':
            deriv = _d_delay_d_PB(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'T0':
            deriv = _d_delay_d_T0(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'ECC':
            deriv = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, gamma, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'OM':
            # d(delay)/d(OM) - OM_eff = OM + corrections, so d(OM_eff)/d(OM) = 1
            deriv = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv * DEG_TO_RAD
            
        elif param_upper == 'PBDOT':
            deriv = _d_delay_d_PBDOT(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'GAMMA':
            deriv = _d_delay_d_GAMMA(toas_bary_mjd, pb, t0, ecc_eff, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'SINI':
            deriv = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'M2':
            deriv = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff)
            derivatives[param] = deriv
            
        elif param_upper == 'OMDOT':
            deriv = _d_delay_d_OMDOT(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, float(jnp.mean(om_eff_deg)), omdot, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'XDOT' or param_upper == 'A1DOT':
            d_a1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot)
            derivatives[param] = d_a1 * dt_sec_from_t0

        elif param_upper == 'EDOT':
            d_ecc = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, gamma, sini_eff, m2)
            derivatives[param] = d_ecc * dt_sec_from_t0

    # Now handle KIN and KOM using chain rule
    if needs_kin:
        # d(delay)/d(KIN) = d(delay)/d(A1_eff) * d(A1_eff)/d(KIN)
        #                 + d(delay)/d(OM_eff) * d(OM_eff)/d(KIN)
        #                 + d(delay)/d(SINI_eff) * d(SINI_eff)/d(KIN)
        
        # Compute correction derivatives
        d_A1_eff_d_KIN, d_OM_eff_d_KIN_rad, d_SINI_eff_d_KIN = _compute_ddk_correction_derivatives_KIN(
            tt0_sec, a1, float(kin_rad), float(kom_rad),
            pmra_rad_per_sec, pmdec_rad_per_sec,
            delta_I0, delta_J0, d_ls,
            use_k96, has_parallax
        )
        
        # Get base derivatives
        d_delay_d_A1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot)
        d_delay_d_OM = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
        d_delay_d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
        
        # Chain rule (note: d_OM_eff_d_KIN_rad is in radians/radian, d_delay_d_OM is in sec/radian)
        d_delay_d_KIN_rad = (
            d_delay_d_A1 * d_A1_eff_d_KIN +
            d_delay_d_OM * d_OM_eff_d_KIN_rad +
            d_delay_d_SINI * d_SINI_eff_d_KIN
        )
        
        # Convert from per-radian to per-degree (KIN is in degrees)
        derivatives['KIN'] = d_delay_d_KIN_rad * DEG_TO_RAD
    
    if needs_kom:
        # d(delay)/d(KOM) = d(delay)/d(A1_eff) * d(A1_eff)/d(KOM)
        #                 + d(delay)/d(OM_eff) * d(OM_eff)/d(KOM)
        
        d_A1_eff_d_KOM, d_OM_eff_d_KOM_rad, _ = _compute_ddk_correction_derivatives_KOM(
            tt0_sec, a1, float(kin_rad), float(kom_rad),
            pmra_rad_per_sec, pmdec_rad_per_sec,
            delta_I0, delta_J0, d_ls,
            use_k96, has_parallax
        )
        
        d_delay_d_A1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc_eff, om_rad_eff, pbdot)
        d_delay_d_OM = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc_eff, om_rad_eff, pbdot, sini_eff, m2)
        
        d_delay_d_KOM_rad = (
            d_delay_d_A1 * d_A1_eff_d_KOM +
            d_delay_d_OM * d_OM_eff_d_KOM_rad
        )
        
        # Convert from per-radian to per-degree
        derivatives['KOM'] = d_delay_d_KOM_rad * DEG_TO_RAD
    
    return derivatives


# =============================================================================
# H3/STIG Orthometric Shapiro Delay Derivatives
# =============================================================================
# DDH uses orthometric parameterization instead of SINI/M2:
#   SINI = 2 * STIG / (1 + STIG^2)
#   M2 = H3 / (STIG^3 * T_SUN)
#
# We use chain rule:
#   d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
#   d(delay)/d(STIG) = d(delay)/d(M2) * d(M2)/d(STIG) + d(delay)/d(SINI) * d(SINI)/d(STIG)


@jax.jit
def _d_delay_d_H3(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, stig: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H3) for DDH orthometric parameterization.
    
    From M2 = H3 / (STIG^3 * T_SUN):
        d(M2)/d(H3) = 1 / (STIG^3 * T_SUN)
    
    So:
        d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
                       = d_delay_d_M2 / (STIG^3 * T_SUN)
    """
    # Compute SINI and M2 from H3/STIG for the Shapiro delay calculation
    sini = 2 * stig / (1 + stig**2)
    
    # d(delay)/d(M2) = -2 * T_SUN * log(1 - sini * sin(omega + theta))
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    
    # d(M2)/d(H3) = 1 / (STIG^3 * T_SUN)
    dM2_dH3 = 1.0 / (stig**3 * T_SUN)
    
    return d_M2 * dM2_dH3


@jax.jit
def _d_delay_d_STIG(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, stig: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(STIG) for DDH orthometric parameterization.
    
    From:
        SINI = 2 * STIG / (1 + STIG^2)
        M2 = H3 / (STIG^3 * T_SUN)
    
    Derivatives:
        d(SINI)/d(STIG) = 2 * (1 - STIG^2) / (1 + STIG^2)^2
        d(M2)/d(STIG) = -3 * H3 / (STIG^4 * T_SUN) = -3 * M2 / STIG
    
    Chain rule:
        d(delay)/d(STIG) = d(delay)/d(M2) * d(M2)/d(STIG) + d(delay)/d(SINI) * d(SINI)/d(STIG)
    """
    # Compute derived quantities
    stig2 = stig**2
    sini = 2 * stig / (1 + stig2)
    m2 = h3 / (stig**3 * T_SUN)
    
    # Get individual derivatives
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)
    
    # Compute Jacobian terms
    dM2_dSTIG = -3 * m2 / stig  # = -3 * H3 / (STIG^4 * T_SUN)
    dSINI_dSTIG = 2 * (1 - stig2) / (1 + stig2)**2
    
    return d_M2 * dM2_dSTIG + d_SINI * dSINI_dSTIG


@jax.jit
def _d_delay_d_H3_h3h4(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, h4: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H3) for H3/H4 orthometric parameterization.

    Freire & Wex (2010), PINT/Tempo2 convention:
        SINI = 2*H3*H4 / (H3^2 + H4^2)
        M2   = H3^4 / (H4^3 * T_SUN)

    Derivatives:
        d(SINI)/d(H3) = 2*H4*(H4^2 - H3^2) / (H3^2 + H4^2)^2
        d(M2)/d(H3)   = 4*H3^3 / (H4^3 * T_SUN) = 4*M2/H3

    Chain rule:
        d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
                        + d(delay)/d(SINI) * d(SINI)/d(H3)
    """
    h4_safe = jnp.maximum(jnp.abs(h4), 1e-30)
    h3h4_denom = jnp.maximum(h3**2 + h4**2, 1e-60)
    sini = jnp.clip(2.0 * h3 * h4 / h3h4_denom, 0.0, 1.0)
    m2 = h3**4 / (h4_safe**3 * T_SUN)

    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)

    dM2_dH3 = 4.0 * h3**3 / (h4_safe**3 * T_SUN)
    dSINI_dH3 = 2.0 * h4 * (h4**2 - h3**2) / h3h4_denom**2

    return d_M2 * dM2_dH3 + d_SINI * dSINI_dH3


@jax.jit
def _d_delay_d_H4(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, h4: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H4) for H3/H4 orthometric parameterization.

    Freire & Wex (2010), PINT/Tempo2 convention:
        SINI = 2*H3*H4 / (H3^2 + H4^2)
        M2   = H3^4 / (H4^3 * T_SUN)

    Derivatives:
        d(SINI)/d(H4) = 2*H3*(H3^2 - H4^2) / (H3^2 + H4^2)^2
        d(M2)/d(H4)   = -3*M2/H4

    Chain rule:
        d(delay)/d(H4) = d(delay)/d(M2) * d(M2)/d(H4)
                        + d(delay)/d(SINI) * d(SINI)/d(H4)
    """
    # Compute SINI and M2 from H3/H4 (PINT convention)
    h4_safe = jnp.maximum(jnp.abs(h4), 1e-30)
    h3h4_denom = jnp.maximum(h3**2 + h4**2, 1e-60)
    sini = jnp.clip(2.0 * h3 * h4 / h3h4_denom, 0.0, 1.0)
    m2 = h3**4 / (h4_safe**3 * T_SUN)

    # Get individual derivatives of delay w.r.t. SINI and M2
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)

    # Jacobian terms
    dM2_dH4 = -3.0 * m2 / h4_safe
    dSINI_dH4 = 2.0 * h3 * (h3**2 - h4**2) / h3h4_denom**2

    return d_M2 * dM2_dH4 + d_SINI * dSINI_dH4


def compute_binary_derivatives_ddgr(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str]
) -> Dict[str, jnp.ndarray]:
    """DDGR binary derivatives: chain the GR mass fit onto the DD per-PK columns.

    In DDGR the post-Keplerian parameters SINI, GAMMA, PBDOT, OMDOT are DERIVED
    from MTOT, M2 (+ Keplerian PB, A1, ECC), so a fit for any of those must
    include the chain terms

        d(delay)/d(p) = [direct DD d(delay)/d(p)]  +  sum_PK d(delay)/d(PK) * d(PK)/d(p)

    The per-PK delay columns d(delay)/d(SINI|GAMMA|PBDOT|OMDOT|M2|PB|A1|ECC|...)
    come from the standard DD derivatives (evaluated at the GR-derived PK
    values); the scalar d(PK)/d(p) come from jug.delays.ddgr. MTOT/XOMDOT/XPBDOT
    are pure chain terms. (DR/DTH chain omitted -- negligible, see
    compute_ddgr_pk_derivatives.)
    """
    from jug.delays.ddgr import compute_ddgr_pk_params, compute_ddgr_pk_derivatives

    mtot = float(params.get('MTOT', 0.0))
    m2 = float(params.get('M2', 0.0))
    pb = _resolve_pb_days(params)
    a1 = float(params.get('A1', 0.0))
    ecc = float(params.get('ECC', params.get('E', 0.0)))

    # Not a well-posed DDGR system -> fall back to plain DD derivatives.
    if not (mtot > 0.0 and m2 > 0.0 and pb > 0.0 and a1 > 0.0):
        return compute_binary_derivatives_dd(params, toas_bary_mjd, fit_params)

    pk = compute_ddgr_pk_params(
        mtot, m2, pb, a1, ecc,
        xomdot_deg_yr=float(params.get('XOMDOT', 0.0)),
        xpbdot=float(params.get('XPBDOT', 0.0)))
    dpk = compute_ddgr_pk_derivatives(mtot, m2, pb, a1, ecc)

    # Per-PK + direct DD delay columns, evaluated at the GR-derived PK values.
    dd_params = dict(params)
    dd_params['SINI'] = pk['sini']
    dd_params['GAMMA'] = pk['gamma_sec']
    dd_params['PBDOT'] = pk['pbdot']
    dd_params['OMDOT'] = pk['omdot_deg_yr']
    need = ['SINI', 'GAMMA', 'PBDOT', 'OMDOT', 'M2', 'PB', 'A1', 'ECC', 'T0', 'OM']
    cols = compute_binary_derivatives_dd(dd_params, toas_bary_mjd, need)

    def C(name):
        return cols.get(name, 0.0)

    out = {}
    for p in fit_params:
        pu = p.upper()
        if pu == 'MTOT':
            out[p] = (C('SINI') * dpk['sini_mtot'] + C('GAMMA') * dpk['gamma_mtot']
                      + C('PBDOT') * dpk['pbdot_mtot'] + C('OMDOT') * dpk['omdot_mtot'])
        elif pu == 'M2':
            out[p] = (C('M2') + C('SINI') * dpk['sini_m2'] + C('GAMMA') * dpk['gamma_m2']
                      + C('PBDOT') * dpk['pbdot_m2'] + C('OMDOT') * dpk['omdot_m2'])
        elif pu == 'PB':
            out[p] = (C('PB') + C('SINI') * dpk['sini_pb'] + C('GAMMA') * dpk['gamma_pb']
                      + C('PBDOT') * dpk['pbdot_pb'] + C('OMDOT') * dpk['omdot_pb'])
        elif pu == 'A1':
            out[p] = C('A1') + C('SINI') * dpk['sini_a1']
        elif pu == 'ECC':
            out[p] = (C('ECC') + C('GAMMA') * dpk['gamma_ecc']
                      + C('PBDOT') * dpk['pbdot_ecc'] + C('OMDOT') * dpk['omdot_ecc'])
        elif pu in ('T0', 'OM'):
            out[p] = C(pu)
        elif pu == 'XOMDOT':
            out[p] = C('OMDOT')        # omdot_total = omdot_GR + XOMDOT
        elif pu == 'XPBDOT':
            out[p] = C('PBDOT')        # pbdot_total = pbdot_GR + XPBDOT
        else:
            # Non-DDGR-coupled param (e.g. astrometry handled elsewhere): use DD.
            _d = compute_binary_derivatives_dd(dd_params, toas_bary_mjd, [p])
            out[p] = _d.get(p, 0.0)
    return out


def compute_ddgr_binary_delay(
    toas_bary_mjd: jnp.ndarray,
    params: Dict,
    **kwargs,
) -> jnp.ndarray:
    """DDGR binary delay for the fitter's forward path.

    Derives the GR post-Keplerian parameters (SINI/GAMMA/PBDOT/OMDOT) from
    MTOT/M2 + Keplerian and feeds them to the standard DD delay. Without this,
    the fitter's binary forward (compute_dd_binary_delay) would read the absent
    SINI/GAMMA from the par as 0 and silently drop Shapiro+Einstein, optimizing
    a wrong model. (DR/DTH ~0.65 ns are carried by the residual-calculator
    forward path, not needed for fit convergence.)
    """
    binary_model = str(params.get('BINARY', '')).upper()
    mtot = float(params.get('MTOT', 0.0))
    m2 = float(params.get('M2', 0.0))
    pb = _resolve_pb_days(params)
    a1 = float(params.get('A1', 0.0))
    if binary_model == 'DDGR' and mtot > 0.0 and m2 > 0.0 and pb > 0.0 and a1 > 0.0:
        from jug.delays.ddgr import compute_ddgr_pk_params
        _pk = compute_ddgr_pk_params(
            mtot, m2, pb, a1, float(params.get('ECC', params.get('E', 0.0))),
            xomdot_deg_yr=float(params.get('XOMDOT', 0.0)),
            xpbdot=float(params.get('XPBDOT', 0.0)))
        params = dict(params)
        params['SINI'] = _pk['sini']
        params['GAMMA'] = _pk['gamma_sec']
        params['PBDOT'] = _pk['pbdot']
        params['OMDOT'] = _pk['omdot_deg_yr']
    return compute_dd_binary_delay(toas_bary_mjd, params, **kwargs)
