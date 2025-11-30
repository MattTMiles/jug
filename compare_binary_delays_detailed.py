"""Compare JUG and PINT binary delays in detail."""

import numpy as np
import jax
import jax.numpy as jnp

# Configure JAX
jax.config.update('jax_enable_x64', True)

# Import JUG DD model
from jug.delays.binary_dd import dd_binary_delay_vectorized

# Import PINT
from pint.models import get_model
from pint.toa import get_TOAs

# Test files
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("DETAILED BINARY DELAY COMPARISON: JUG vs PINT")
print("="*80)

# Load PINT model and TOAs
print("\nLoading PINT model and TOAs...")
model = get_model(PAR_FILE)
toas = get_TOAs(TIM_FILE, planets=True)

# Get barycentric TOAs from PINT (convert to float MJD values)
# Use tdbld column which has high-precision TDB as long double
tdb_mjd_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Get binary parameters from PINT model
binary_comp = model.get_components_by_category()['pulsar_system'][0]
print(f"\nBinary model: {binary_comp.binary_model_name}")

# Extract parameters
params = {}
for param_name in ['PB', 'A1', 'ECC', 'OM', 'T0', 'GAMMA', 'PBDOT', 'OMDOT', 'M2', 'SINI']:
    if hasattr(binary_comp, param_name):
        param_obj = getattr(binary_comp, param_name)
        if param_obj.value is not None:
            params[param_name] = float(param_obj.value)
        else:
            params[param_name] = 0.0
    else:
        params[param_name] = 0.0

print(f"\nBinary parameters:")
for k, v in params.items():
    print(f"  {k:10s} = {v}")

# Compute PINT binary delays
print(f"\nComputing PINT binary delays...")
pint_binary_delays = binary_comp.binarymodel_delay(toas, None)
pint_binary_delays_sec = pint_binary_delays.to_value('s')
print(f"  Range: [{np.min(pint_binary_delays_sec):.6f}, {np.max(pint_binary_delays_sec):.6f}] s")
print(f"  Mean: {np.mean(pint_binary_delays_sec):.6f} s")
print(f"  Std: {np.std(pint_binary_delays_sec):.6f} s")

# Compute JUG binary delays
print(f"\nComputing JUG binary delays...")
jug_binary_delays_sec = np.array(dd_binary_delay_vectorized(
    jnp.array(tdb_mjd_pint),
    pb_days=params['PB'],
    a1_lt_sec=params['A1'],
    ecc=params['ECC'],
    omega_deg=params['OM'],
    t0_mjd=params['T0'],
    gamma_sec=params['GAMMA'],
    pbdot=params['PBDOT'],
    omdot_deg_yr=params['OMDOT'],
    xdot=0.0,  # Not in par file
    edot=0.0,  # Not in par file
    sini=params['SINI'],
    m2_msun=params['M2'],
))
print(f"  Range: [{np.min(jug_binary_delays_sec):.6f}, {np.max(jug_binary_delays_sec):.6f}] s")
print(f"  Mean: {np.mean(jug_binary_delays_sec):.6f} s")
print(f"  Std: {np.std(jug_binary_delays_sec):.6f} s")

# Compare
print(f"\n" + "="*80)
print("BINARY DELAY COMPARISON")
print("="*80)

diff_sec = jug_binary_delays_sec - pint_binary_delays_sec
diff_us = diff_sec * 1e6

print(f"\nJUG - PINT difference:")
print(f"  Mean: {np.mean(diff_us):.6f} μs = {np.mean(diff_us)*1000:.3f} ns")
print(f"  RMS: {np.std(diff_us):.6f} μs = {np.std(diff_us)*1000:.3f} ns")
print(f"  Min: {np.min(diff_us):.6f} μs")
print(f"  Max: {np.max(diff_us):.6f} μs")
print(f"  Median: {np.median(diff_us):.6f} μs")

# Check for single TOA
print(f"\n" + "="*80)
print("SINGLE TOA ANALYSIS (first TOA)")
print("="*80)

idx = 0
print(f"\nTOA #{idx}:")
print(f"  TDB: {tdb_mjd_pint[idx]:.10f} MJD")
print(f"  PINT binary delay: {pint_binary_delays_sec[idx]:.12f} s")
print(f"  JUG binary delay: {jug_binary_delays_sec[idx]:.12f} s")
print(f"  Difference: {diff_us[idx]:.6f} μs = {diff_us[idx]*1000:.3f} ns")

# Check for correlation with orbital phase
print(f"\n" + "="*80)
print("ORBITAL PHASE CORRELATION")
print("="*80)

# Compute orbital phase
dt_days = tdb_mjd_pint - params['T0']
orbital_phase = (dt_days / params['PB']) % 1.0

corr_phase = np.corrcoef(orbital_phase, diff_us)[0, 1]
print(f"Correlation with orbital phase: {corr_phase:.6f}")

# Check percentiles
print(f"\nBinary delay difference percentiles (μs):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(diff_us, p)
    print(f"  {p:2d}%: {val:10.6f} μs = {val*1000:10.3f} ns")

# Save results
np.savez('binary_delay_comparison.npz',
         jug_delays=jug_binary_delays_sec,
         pint_delays=pint_binary_delays_sec,
         diff_us=diff_us,
         tdb_mjd=tdb_mjd_pint,
         orbital_phase=orbital_phase)
print(f"\nSaved to: binary_delay_comparison.npz")

print("\n" + "="*80)
