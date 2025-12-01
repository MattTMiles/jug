"""Test Levenberg-Marquardt fitter against PINT."""

# CRITICAL: Enable JAX float64 BEFORE imports
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
import pint.residuals

# Load data
print("Loading data...")
par_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par')
tim_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim')

model_pint = pm.get_model(str(par_file))
toas_pint = pt.get_TOAs(str(tim_file), planets=True, ephem='de421')

# Get PINT pre-fit residuals
print("\nComputing PINT pre-fit residuals...")
res_pint_prefit = pint.residuals.Residuals(toas_pint, model_pint)
pint_prefit_res_us = res_pint_prefit.time_resids.to_value('us')
print(f"PINT pre-fit RMS: {np.std(pint_prefit_res_us):.3f} μs")

# Fit with PINT
print("\nFitting with PINT...")
fitter = pf.WLSFitter(toas_pint, model_pint)
fitter.fit_toas(maxiter=10)

# Get PINT post-fit values
pint_f0 = fitter.model.F0.value
pint_f1 = fitter.model.F1.value
pint_dm = fitter.model.DM.value
pint_ra = fitter.model.RAJ.value
pint_dec = fitter.model.DECJ.value

print("\nPINT fitted values:")
print(f"F0:  {pint_f0:.15f} Hz")
print(f"F1:  {pint_f1:.15e} Hz/s")
print(f"DM:  {pint_dm:.15f} pc/cm^3")
print(f"RAJ: {pint_ra:.15f} rad")
print(f"DECJ: {pint_dec:.15f} rad")

# Get PINT post-fit residuals
res_pint_postfit = fitter.resids
pint_postfit_res_us = res_pint_postfit.time_resids.to_value('us')
print(f"\nPINT post-fit RMS: {np.std(pint_postfit_res_us):.3f} μs")

# Setup JUG
print("\n" + "="*60)
print("Fitting with JUG Levenberg-Marquardt...")
print("="*60)

# Get baseline JUG residuals
result_baseline = compute_residuals_simple(
    str(par_file), str(tim_file),
    clock_dir="data/clock",
    observatory="meerkat"
)

toas_mjd = result_baseline['t_utc_mjd']
freqs_mhz = result_baseline['freqs_mhz']
errors_us = result_baseline['errors_us']

print(f"\nLoaded {len(toas_mjd)} TOAs")
print(f"JUG baseline RMS: {result_baseline['rms_us']:.3f} μs")

# Prepare fixed data for JAX
fixed_data = prepare_fixed_data(
    str(par_file), str(tim_file),
    clock_dir="data/clock",
    observatory="meerkat"
)

# Get initial parameter values
initial_f0 = fixed_data['initial_params']['F0']
initial_f1 = fixed_data['initial_params']['F1']
initial_dm = fixed_data['initial_params']['DM']
initial_raj = fixed_data['initial_params']['RAJ']
initial_decj = fixed_data['initial_params']['DECJ']

print("\nInitial parameters:")
print(f"F0:   {initial_f0:.15e}")
print(f"F1:   {initial_f1:.15e}")
print(f"DM:   {initial_dm:.15e}")
print(f"RAJ:  {initial_raj:.15e}")
print(f"DECJ: {initial_decj:.15e}")

# Create residual function for fitter
def residual_func(toas, freqs, params_dict):
    """Compute residuals for given parameters."""
    # Use JAX residual function with updated parameters
    params_array = jnp.array([
        params_dict['F0'],
        params_dict['F1'],
        params_dict['DM'],
        params_dict['RAJ'],
        params_dict['DECJ']
    ], dtype=jnp.float64)
    
    return compute_residuals_jax(
        toas, freqs, params_array, 
        fixed_data['t_tdb_mjd_ref'],
        fixed_data
    )

# Initial parameters
initial_params = {
    'F0': initial_f0,
    'F1': initial_f1,
    'DM': initial_dm,
    'RAJ': initial_raj,
    'DECJ': initial_decj,
}

# Fit with Levenberg-Marquardt
fitted_params = fit_levenberg_marquardt(
    residual_func,
    initial_params,
    toas_mjd,
    freqs_mhz,
    errors_us,
    max_iterations=20,
    tolerance=1e-9,
    initial_damping=1e-3,
    verbose=True,
)

print("\n" + "="*60)
print("COMPARISON: PINT vs JUG L-M")
print("="*60)

print("\nFitted values:")
print(f"{'Parameter':<10} {'PINT':>20} {'JUG L-M':>20} {'Difference':>15} {'Sigma':>8}")
print("-" * 80)

params_compare = [
    ('F0', pint_f0, fitted_params['F0'], fitter.model.F0.uncertainty.value),
    ('F1', pint_f1, fitted_params['F1'], fitter.model.F1.uncertainty.value),
    ('DM', pint_dm, fitted_params['DM'], fitter.model.DM.uncertainty.value),
    ('RAJ', pint_ra, fitted_params['RAJ'], fitter.model.RAJ.uncertainty.value),
    ('DECJ', pint_dec, fitted_params['DECJ'], fitter.model.DECJ.uncertainty.value),
]

for name, pint_val, jug_val, unc in params_compare:
    diff = jug_val - pint_val
    rel_diff_pct = 100 * abs(diff / pint_val) if pint_val != 0 else 0
    sigma = abs(diff / unc) if unc > 0 else 0
    print(f"{name:<10} {pint_val:>20.12e} {jug_val:>20.12e} {diff:>15.6e} {sigma:>8.2f}σ")

print("\nRelative differences:")
for name, pint_val, jug_val, unc in params_compare:
    rel_diff_pct = 100 * abs((jug_val - pint_val) / pint_val) if pint_val != 0 else 0
    print(f"{name}: {rel_diff_pct:.6f}%")
