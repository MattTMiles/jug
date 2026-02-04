#!/usr/bin/env python
"""
Further investigation: The DM time offset doesn't explain the 1.1 μs variance.
Let's check other possible causes:
1. DD algorithm difference (delayInverse vs simple Roemer)
2. Different Kepler solver precision
3. Different time-dependent parameter evolution
4. Something else entirely
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
import astropy.units as u

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("Loading PINT model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"Number of TOAs: {len(toas)}")

# Get PINT and JUG residuals
pint_resids = Residuals(toas, model)
pint_resid_us = pint_resids.time_resids.to('us').value

import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

from jug.residuals.simple_calculator import compute_residuals_simple

jug_result = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)
jug_resid_us = jug_result['residuals_us']

diff_us = jug_resid_us - pint_resid_us
diff_centered = diff_us - np.mean(diff_us)

print(f"\nResidual difference (centered) std: {np.std(diff_centered):.3f} μs")

# Get binary delays directly from both
print("\n" + "="*80)
print("Comparing Binary Delays Directly")
print("="*80)

# PINT binary delay
binary_comp = model.components['BinaryDD']
pint_binary_delay = binary_comp.binarymodel_delay(toas, None).to('s').value
print(f"PINT binary delay: mean={np.mean(pint_binary_delay):.6f} s, std={np.std(pint_binary_delay):.6f} s")

# Get JUG's binary delay - need to trace through the computation
# Check what's in the result
print(f"\nJUG result keys: {list(jug_result.keys())}")

# Get the barycentric time JUG uses for binary
tdb_mjd = jug_result['tdb_mjd']
roemer_shapiro_sec = jug_result.get('roemer_shapiro_sec', None)

if roemer_shapiro_sec is not None:
    t_binary_jug = tdb_mjd - roemer_shapiro_sec / 86400.0
else:
    print("WARNING: No roemer_shapiro_sec in JUG result!")
    t_binary_jug = tdb_mjd

# Get PINT's barycentric time
t_binary_pint = np.array([b.value for b in model.get_barycentric_toas(toas)])

print(f"\nBinary evaluation time comparison:")
print(f"  JUG t_binary[0]:  {t_binary_jug[0]:.15f} MJD")
print(f"  PINT t_binary[0]: {t_binary_pint[0]:.15f} MJD")
print(f"  Difference: {(t_binary_jug[0] - t_binary_pint[0]) * 86400 * 1000:.6f} ms")

# Now compute JUG binary delay at PINT's time to isolate algorithm difference
print("\n" + "="*80)
print("Computing JUG DD at PINT's barycentric time")
print("="*80)

from jug.delays.binary_dd import dd_binary_delay

# Extract binary parameters
pb = float(binary_comp.PB.value)
a1 = float(binary_comp.A1.value)
ecc = float(binary_comp.ECC.value)
om = float(binary_comp.OM.value)
t0 = float(binary_comp.T0.value)
gamma = float(binary_comp.GAMMA.value) if binary_comp.GAMMA.value is not None else 0.0
pbdot = float(binary_comp.PBDOT.value) if binary_comp.PBDOT.value is not None else 0.0
omdot = float(binary_comp.OMDOT.value) if binary_comp.OMDOT.value is not None else 0.0
xdot = float(binary_comp.A1DOT.value) if hasattr(binary_comp, 'A1DOT') and binary_comp.A1DOT.value is not None else 0.0
edot = float(binary_comp.EDOT.value) if hasattr(binary_comp, 'EDOT') and binary_comp.EDOT.value is not None else 0.0
sini = float(binary_comp.SINI.value) if binary_comp.SINI.value is not None else 0.0
m2 = float(binary_comp.M2.value) if binary_comp.M2.value is not None else 0.0

print(f"\nBinary parameters:")
print(f"  PB = {pb} d")
print(f"  A1 = {a1} ls")
print(f"  ECC = {ecc}")
print(f"  OM = {om} deg")
print(f"  T0 = {t0} MJD")
print(f"  GAMMA = {gamma} s")
print(f"  SINI = {sini}")
print(f"  M2 = {m2} Msun")

# Compute JUG binary delay at PINT's barycentric time
# Need to ensure float64, not float128
t_binary_pint_f64 = np.array(t_binary_pint, dtype=np.float64)

jug_binary_at_pint_time = []
for t in t_binary_pint_f64:
    delay = float(dd_binary_delay(
        float(t), pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
    ))
    jug_binary_at_pint_time.append(delay)
jug_binary_at_pint_time = np.array(jug_binary_at_pint_time)

print(f"\nJUG binary delay at PINT's time:")
print(f"  Mean: {np.mean(jug_binary_at_pint_time):.6f} s")
print(f"  Std:  {np.std(jug_binary_at_pint_time):.6f} s")

# Compare
binary_diff = (jug_binary_at_pint_time - pint_binary_delay) * 1e6  # μs
print(f"\nBinary delay difference (JUG - PINT) at same time:")
print(f"  Mean: {np.mean(binary_diff):.3f} μs")
print(f"  Std:  {np.std(binary_diff):.3f} μs")
print(f"  Range: [{np.min(binary_diff):.3f}, {np.max(binary_diff):.3f}] μs")

# This binary delay difference should directly contribute to residual difference
# Check correlation
binary_diff_centered = binary_diff - np.mean(binary_diff)
x = binary_diff_centered
y = diff_centered
r = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
print(f"\nCorrelation between binary delay diff and residual diff: r = {r:.6f}")

# If binary delay diff explains the residual diff, subtracting should help
unexplained = diff_centered - binary_diff_centered
print(f"\nResidual after removing binary delay difference:")
print(f"  Std: {np.std(unexplained):.3f} μs")

# Now let's look at orbital phase dependence of the binary delay difference
print("\n" + "="*80)
print("Orbital Phase Analysis of Binary Delay Difference")
print("="*80)

binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance
orbital_phase = (bo.orbits() % 1).value

# Bin by phase
n_bins = 20
phase_bins = np.linspace(0, 1, n_bins + 1)

print(f"\n{'Phase':<10} {'BinaryDiff(μs)':<15} {'ResidDiff(μs)':<15} {'N':<5}")
print("-" * 50)

for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 5:
        phase_center = (phase_bins[i] + phase_bins[i+1]) / 2
        bin_binary_diff = np.mean(binary_diff_centered[mask])
        bin_resid_diff = np.mean(diff_centered[mask])
        print(f"{phase_center:.2f}      {bin_binary_diff:+8.3f}        {bin_resid_diff:+8.3f}        {np.sum(mask)}")

# Check if the issue is in how JUG computes other delays (not binary)
print("\n" + "="*80)
print("Checking Other Delay Components")
print("="*80)

# Get PINT's total delay
pint_total_delay = model.delay(toas).to('s').value

# Get JUG's total delay
jug_total_delay = jug_result.get('total_delay_sec', None)

if jug_total_delay is not None:
    delay_diff = (jug_total_delay - pint_total_delay) * 1e6
    print(f"\nTotal delay difference (JUG - PINT):")
    print(f"  Mean: {np.mean(delay_diff):.3f} μs")
    print(f"  Std:  {np.std(delay_diff):.3f} μs")
    
    delay_diff_centered = delay_diff - np.mean(delay_diff)
    r = np.sum(delay_diff_centered * diff_centered) / np.sqrt(np.sum(delay_diff_centered**2) * np.sum(diff_centered**2))
    print(f"  Correlation with residual diff: r = {r:.6f}")
else:
    print("  total_delay_sec not in JUG result")

# Summary
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)
print(f"""
Binary delay difference (JUG - PINT at same time):
  - Mean: {np.mean(binary_diff):.3f} μs
  - Std:  {np.std(binary_diff):.3f} μs
  
Residual difference (centered):
  - Std:  {np.std(diff_centered):.3f} μs
  
Correlation: r = {r:.3f}

If r ≈ 1: The DD algorithm difference is the main cause.
If r ≈ 0: The issue is elsewhere (non-binary delays, phase computation, etc.)
""")
