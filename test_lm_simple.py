"""Simpler L-M test - just check if fitter works."""

# CRITICAL: Enable JAX float64 BEFORE imports
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

print("Preparing data...")
par_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par')
tim_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim')

# Get baseline
result = compute_residuals_simple(str(par_file), str(tim_file))
print(f"\nBaseline JUG RMS: {result['rms_us']:.3f} μs")

# Prepare fixed data
fixed_data = prepare_fixed_data(str(par_file), str(tim_file))

print("\nAvailable keys in fixed_data:")
for key in sorted(fixed_data.keys()):
    val = fixed_data[key]
    if isinstance(val, (np.ndarray, jnp.ndarray)):
        print(f"  {key}: array shape={val.shape}, dtype={val.dtype}")
    elif isinstance(val, dict):
        print(f"  {key}: dict with keys {list(val.keys())[:5]}...")
    else:
        print(f"  {key}: {type(val).__name__}")

print("\nInitial parameters:")
for key, val in fixed_data['initial_params'].items():
    print(f"  {key}: {val}")

# Create residual function
def residual_func(toas, freqs, params_dict):
    """Compute residuals for given parameters."""
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

# Test residual function with initial params
print("\nTesting residual function...")
toas_test = fixed_data['tdb_mjd'][:10]
freqs_test = fixed_data['freqs_mhz'][:10]
params_test = fixed_data['initial_params']

res_test = residual_func(toas_test, freqs_test, params_test)
print(f"  Computed {len(res_test)} test residuals")
print(f"  Range: [{float(res_test.min()):.6e}, {float(res_test.max()):.6e}] sec")

