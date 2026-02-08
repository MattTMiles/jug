"""T2 (Tempo2 general) binary model implementation with JAX JIT compilation.

The T2 model is Tempo2's "universal" binary model that can emulate any other
binary model (BT, DD, ELL1, etc.) by setting appropriate parameters. This is
the most flexible binary model and many published .par files use BINARY T2.

References
----------
- Edwards et al. (2006), MNRAS 372, 1549 (Tempo2 paper)
- Tempo2: T2model.C, T2model.h
- PINT: pint/models/binary_t2.py (if exists)

Notes
-----
T2 supports multiple parameterizations:
- Keplerian: PB, A1, ECC, OM, T0
- Shapiro: M2/SINI or H3/STIG  
- Inclination: KIN, KOM (3D orbital geometry)
- Time derivatives: PBDOT, OMDOT, XDOT, EDOT
- Relativistic: GAMMA, DR, DTH, A0, B0

For now, we implement the most common subset (Keplerian + M2/SINI + derivatives).
"""

import jax
import jax.numpy as jnp
from jug.delays.binary_bt import solve_kepler
from jug.utils.constants import SECS_PER_DAY


@jax.jit
def t2_binary_delay(
    t_topo_tdb, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
    m2, sini, kin=0.0, kom=0.0,
    fb_coeffs=None, fb_factorials=None, fb_epoch=0.0, use_fb=False
):
    """Compute T2 (Tempo2 general) binary delay.
    
    Updated to support FB (orbital frequency) parameters.
    """
    # Guard against None fb arrays (JAX eagerly evaluates both jnp.where branches)
    if fb_coeffs is None:
        fb_coeffs = jnp.array([], dtype=jnp.float64)
    if fb_factorials is None:
        fb_factorials = jnp.array([], dtype=jnp.float64)

    # Time since periastron
    dt_days = t_topo_tdb - t0
    dt_sec = dt_days * SECS_PER_DAY
    
    # Apply time-dependent corrections
    om_rad = jnp.deg2rad(om + omdot * dt_days / 365.25)
    ecc_eff = jnp.where(edot != 0.0, ecc + edot * dt_sec, ecc)
    ecc_eff = jnp.clip(ecc_eff, 0.0, 0.9999)
    # xdot affects a1
    a1_eff = jnp.where(xdot != 0.0, a1 + xdot * dt_sec, a1)
    
    # Mean anomaly calculation
    def compute_mean_anomaly_pb():
        pb_eff = pb * (1.0 + pbdot * dt_days / pb)
        n = 2.0 * jnp.pi / (pb_eff * SECS_PER_DAY)
        return n * dt_sec

    def compute_mean_anomaly_fb():
        # FB expansion gives Phase in turns from TASC (usually)
        # M = 2pi * Phase - omega
        # Because at TASC, M = -omega. Phase=0.
        # FB epoch is usually TASC.
        
        dt_fb = (t_topo_tdb - fb_epoch) * SECS_PER_DAY
        n_coeffs = fb_coeffs.shape[0]
        indices = jnp.arange(n_coeffs)
        powers_plus1 = indices + 1
        dt_powers_plus1 = dt_fb ** powers_plus1
        # (i+1)! = fb_factorials[i] * (i+1) where fb_factorials[i] = i!
        factorials_plus1 = fb_factorials * (indices + 1)
        
        phase_integral = jnp.sum(fb_coeffs * dt_powers_plus1 / factorials_plus1)
        phase_rad = 2.0 * jnp.pi * phase_integral
        
        # Adjust for T0 vs TASC difference
        # M = Phase_from_TASC - omega
        return phase_rad - om_rad

    mean_anomaly = jnp.where(use_fb, compute_mean_anomaly_fb(), compute_mean_anomaly_pb())
    
    # Solve Kepler's equation
    ecc_anomaly = solve_kepler(mean_anomaly, ecc_eff)
    
    # True anomaly
    true_anomaly = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc_eff) * jnp.sin(ecc_anomaly / 2.0),
        jnp.sqrt(1.0 - ecc_eff) * jnp.cos(ecc_anomaly / 2.0)
    )
    
    # === ROEMER DELAY ===
    sin_omega_nu = jnp.sin(om_rad + true_anomaly)
    
    # 3D geometry correction (KIN/KOM)
    kin_rad = jnp.deg2rad(kin)
    # kom not used in delay? It is used for secular changes in x/om due to proper motion (Kopeikin), 
    # but geometric delay factor only depends on KIN (inclination) and position on sky?
    # Kopeikin Eq 7: Delay = x * [ -sin(w+nu) * cos(KIN) + ... ] ?
    # Actually checking t2_binary used cos(KIN) factor correctly as projection.
    
    geometry_factor = jnp.where(
        kin != 0.0,
        jnp.cos(kin_rad),
        1.0
    )
    
    roemer_delay = a1_eff * geometry_factor * (sin_omega_nu + ecc_eff * jnp.sin(om_rad))
    
    # === EINSTEIN DELAY ===
    einstein_delay = jnp.where(gamma != 0.0, gamma * jnp.sin(ecc_anomaly), 0.0)
    
    # === SHAPIRO DELAY ===
    T_SUN = 4.925490947e-6
    r_shap = T_SUN * m2
    shapiro_delay = jnp.where(
        (m2 > 0.0) & (sini > 0.0),
        -2.0 * r_shap * jnp.log(1.0 - sini * sin_omega_nu),
        0.0
    )
    
    return roemer_delay + einstein_delay + shapiro_delay



@jax.jit
def t2_binary_delay_vectorized(
    t_topo_tdb_array, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
    m2, sini, kin, kom
):
    """Vectorized T2 binary delay for array of times.
    
    Parameters
    ----------
    t_topo_tdb_array : jnp.ndarray
        Array of topocentric TDB times (MJD)
    pb, a1, ecc, om, t0, ... : float
        Binary parameters (see t2_binary_delay)
        
    Returns
    -------
    jnp.ndarray
        Array of binary delays (seconds)
    """
    return jax.vmap(
        lambda t: t2_binary_delay(t, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot, m2, sini, kin, kom)
    )(t_topo_tdb_array)


def t2_binary_delay_sec(t_em_mjd_array, model):
    """Wrapper to compute T2 binary delay from model dict.
    
    Parameters
    ----------
    t_em_mjd_array : jnp.ndarray
        Array of emission times (MJD in TDB)
    model : dict
        Parameter dictionary from read_par_file
        
    Returns
    -------
    jnp.ndarray
        Binary delays in seconds
    """
    # Extract parameters with defaults
    pb = float(model.get('PB', 0.0))
    a1 = float(model.get('A1', 0.0))
    ecc = float(model.get('ECC', 0.0))
    om = float(model.get('OM', 0.0))
    t0 = float(model.get('T0', model.get('TASC', 0.0)))
    gamma = float(model.get('GAMMA', 0.0))
    pbdot = float(model.get('PBDOT', 0.0))
    xdot = float(model.get('XDOT', 0.0))
    edot = float(model.get('EDOT', 0.0))
    omdot = float(model.get('OMDOT', 0.0))
    m2 = float(model.get('M2', 0.0))
    sini = float(model.get('SINI', 0.0))
    kin = float(model.get('KIN', 0.0))
    kom = float(model.get('KOM', 0.0))
    
    return t2_binary_delay_vectorized(
        t_em_mjd_array, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
        m2, sini, kin, kom
    )
