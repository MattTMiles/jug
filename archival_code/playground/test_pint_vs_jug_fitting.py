"""
Test if PINT and JUG converge to same parameters when fitting the same residuals.

Strategy:
1. Compute JUG baseline residuals using longdouble precision
2. Replace PINT's residuals with JUG's residuals (by adjusting TOA times)
3. Fit with both PINT and JUG
4. Compare fitted parameters
"""

import numpy as np
import pint.models
import pint.toa
import pint.residuals
from pint.fitter import WLSFitter
import astropy.units as u
from astropy.time import TimeDelta
import jax
import jax.numpy as jnp

# Enable float64 in JAX
jax.config.update('jax_enable_x64', True)

# Import JUG modules
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds  
from jug.io.clock import clock_correction_seconds
from jug.delays.barycentric import barycentric_correction
from jug.delays.combined import ell1_delay_third_order

print("=" * 70)
print("TESTING: Do PINT and JUG converge to same fitted parameters?")
print("=" * 70)
print()

# Load data
parfile = 'data/J1909-3744.par'
timfile = 'data/J1909-3744.tim'

# ============================================================================
# 1. Compute JUG baseline residuals with longdouble precision
# ============================================================================
print("1. Computing JUG baseline residuals (longdouble precision)...")

params = parse_par_file(parfile)
toas_list = parse_tim_file_mjds(timfile)

# Convert to dict format
toas_table = []
for toa in toas_list:
    mjd_value = toa.mjd_int + toa.mjd_frac
    toas_table.append({
        'mjd': mjd_value,
        'freq': toa.freq_mhz,
        'error': toa.error_us,
        'obs': toa.observatory
    })

# Extract data
mjd_utc = np.array([t['mjd'] for t in toas_table], dtype=np.longdouble)
freq_mhz = np.array([t['freq'] for t in toas_table], dtype=np.longdouble)
errors_us = np.array([t['error'] for t in toas_table], dtype=np.longdouble)
obs_codes = [t['obs'] for t in toas_table]

n_toas = len(mjd_utc)
print(f"   Loaded {n_toas} TOAs")

# Clock corrections
mjd_tt = np.zeros(n_toas, dtype=np.longdouble)
for i in range(n_toas):
    corr_sec = clock_correction_seconds(mjd_utc[i], obs_codes[i])
    mjd_tt[i] = mjd_utc[i] + corr_sec / 86400.0

# Barycentric correction
ssb_delays_sec = np.zeros(n_toas, dtype=np.longdouble)
for i in range(n_toas):
    ssb_delay = barycentric_correction(
        mjd_tt[i],
        obs_codes[i],
        params['RAJ'],
        params['DECJ'],
        params.get('PMRA', 0.0),
        params.get('PMDEC', 0.0),
        params.get('PX', 0.0)
    )
    ssb_delays_sec[i] = ssb_delay

mjd_tdb = mjd_tt + ssb_delays_sec / 86400.0

# Binary delay
if 'BINARY' in params and params['BINARY'] in ['ELL1', 'ELL1H']:
    binary_delays_sec = np.zeros(n_toas, dtype=np.longdouble)
    for i in range(n_toas):
        binary_delays_sec[i] = ell1_delay_third_order(
            mjd_tdb[i],
            params['PB'],
            params['A1'],
            params['TASC'],
            params['EPS1'],
            params['EPS2'],
            params.get('PBDOT', 0.0),
            params.get('XDOT', 0.0),
            params.get('EPS1DOT', 0.0),
            params.get('EPS2DOT', 0.0),
            params.get('M2', 0.0),
            params.get('SINI', 0.0)
        )
    mjd_emission = mjd_tdb - binary_delays_sec / 86400.0
else:
    mjd_emission = mjd_tdb.copy()

# DM delay (dispersion measure)
K_DM_SEC = 4.148808e3  # MHz^2 pc^-1 cm^3 s
dm_delays_sec = (K_DM_SEC * params.get('DM', 0.0) / freq_mhz**2).astype(np.longdouble)
mjd_infinite_freq = mjd_emission - dm_delays_sec / 86400.0

# Spin phase (use JAX for this part)
F0 = params['F0']
F1 = params.get('F1', 0.0)
F2 = params.get('F2', 0.0)
PEPOCH = params['PEPOCH']

dt_days = mjd_infinite_freq - PEPOCH
dt_sec = dt_days * 86400.0

# Use JAX for phase computation
dt_sec_jax = jnp.array(dt_sec, dtype=jnp.float64)

@jax.jit
def compute_phase_jax(dt_sec, f0, f1, f2):
    """Compute spin phase with JAX."""
    phase = f0 * dt_sec + 0.5 * f1 * dt_sec**2 + (1.0/6.0) * f2 * dt_sec**3
    return phase

phase_cycles = np.array(compute_phase_jax(dt_sec_jax, F0, F1, F2))

# Phase offset (TZR)
if 'TZRMJD' in params:
    tzr_mjd = params['TZRMJD']
    tzr_phase = np.array(compute_phase_jax(
        jnp.array((tzr_mjd - PEPOCH) * 86400.0, dtype=jnp.float64),
        F0, F1, F2
    ))
    phase_offset = np.round(tzr_phase)
else:
    phase_offset = 0.0

phase_residual_cycles = phase_cycles - phase_offset

# Convert to time residuals
residuals_jug_sec = phase_residual_cycles / F0

print(f"   JUG baseline RMS: {np.std(residuals_jug_sec)*1e6:.3f} μs")
print()

# ============================================================================
# 2. Load PINT and replace its residuals with JUG's
# ============================================================================
print("2. Loading PINT and replacing its residuals with JUG's...")

model_pint = pint.models.get_model(parfile)
toas_pint = pint.toa.get_TOAs(timfile, model=model_pint)

# Compute original PINT residuals
res_pint_original = pint.residuals.Residuals(toas_pint, model_pint)
residuals_pint_original = res_pint_original.time_resids.to_value('s')

print(f"   PINT original RMS: {np.std(residuals_pint_original)*1e6:.3f} μs")
print(f"   JUG-PINT difference: {np.std(residuals_jug_sec - residuals_pint_original)*1e6:.3f} μs")

# Adjust TOAs so PINT computes JUG residuals
delta_residuals = residuals_jug_sec - residuals_pint_original
toas_pint.adjust_TOAs(TimeDelta(delta_residuals * u.s))

# Verify
res_check = pint.residuals.Residuals(toas_pint, model_pint)
residuals_check = res_check.time_resids.to_value('s')
print(f"   After adjustment, PINT RMS: {np.std(residuals_check)*1e6:.3f} μs")
print(f"   Match to JUG: {np.std(residuals_check - residuals_jug_sec)*1e6:.6f} μs")
print()

# ============================================================================
# 3. Fit with PINT
# ============================================================================
print("3. Fitting with PINT...")

fitter_pint = WLSFitter(toas_pint, model_pint)
fitter_pint.fit_toas()

pint_f0 = model_pint.F0.value
pint_f1 = model_pint.F1.value
pint_chi2 = fitter_pint.resids.chi2

print(f"   PINT F0: {pint_f0:.15f} Hz")
print(f"   PINT F1: {pint_f1:.15e} Hz/s")
print(f"   PINT chi2: {pint_chi2:,.1f}")
print()

# ============================================================================
# 4. Fit with JUG
# ============================================================================
print("4. Fitting with JUG...")

from jug.fitting.gauss_newton_jax import fit_gauss_newton_jax

# Prepare data for JUG fitter
mjd_inf_jax = jnp.array(mjd_infinite_freq, dtype=jnp.float64)
freq_jax = jnp.array(freq_mhz, dtype=jnp.float64)
errors_sec_jax = jnp.array(errors_us * 1e-6, dtype=jnp.float64)

# Initial parameters (from .par file)
initial_params = jnp.array([F0, F1], dtype=jnp.float64)
param_names = ['F0', 'F1']

# Fit
result = fit_gauss_newton_jax(
    mjd_inf_jax,
    freq_jax,
    errors_sec_jax,
    initial_params,
    param_names,
    params,
    max_iterations=10,
    tolerance=1e-12
)

jug_f0 = float(result['fitted_params'][0])
jug_f1 = float(result['fitted_params'][1])
jug_chi2 = float(result['chi2'])

print(f"   JUG F0: {jug_f0:.15f} Hz")
print(f"   JUG F1: {jug_f1:.15e} Hz/s")
print(f"   JUG chi2: {jug_chi2:,.1f}")
print()

# ============================================================================
# 5. Compare results
# ============================================================================
print("=" * 70)
print("COMPARISON: PINT vs JUG fitting the same residuals")
print("=" * 70)
print()
print(f"F0:")
print(f"  PINT:      {pint_f0:.15f} Hz")
print(f"  JUG:       {jug_f0:.15f} Hz")
print(f"  Difference: {jug_f0 - pint_f0:.3e} Hz")
print()
print(f"F1:")
print(f"  PINT:      {pint_f1:.15e} Hz/s")
print(f"  JUG:       {jug_f1:.15e} Hz/s")
print(f"  Difference: {jug_f1 - pint_f1:.3e} Hz/s")
print()
print(f"Chi2:")
print(f"  PINT:      {pint_chi2:,.1f}")
print(f"  JUG:       {jug_chi2:,.1f}")
print(f"  Difference: {jug_chi2 - pint_chi2:,.1f}")
print()

# Interpret
tol_f0 = 1e-12  # 1 pHz
tol_f1 = 1e-20  # 1e-20 Hz/s

if abs(jug_f0 - pint_f0) < tol_f0 and abs(jug_f1 - pint_f1) < tol_f1:
    print("✅ SUCCESS: JUG and PINT converge to same fitted parameters!")
    print("   The fitters are equivalent within numerical precision.")
else:
    print("⚠️  DIFFERENCE: JUG and PINT find different fitted parameters")
    print(f"   ΔF0 = {abs(jug_f0 - pint_f0):.3e} Hz (tolerance: {tol_f0:.3e})")
    print(f"   ΔF1 = {abs(jug_f1 - pint_f1):.3e} Hz/s (tolerance: {tol_f1:.3e})")
    print()
    print("   This may indicate:")
    print("   - Different fitting algorithms")
    print("   - Different stopping criteria")
    print("   - Numerical precision differences")
