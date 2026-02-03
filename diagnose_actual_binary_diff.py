#!/usr/bin/env python
"""
Compare the ACTUAL binary delays used by JUG vs PINT.
Not "JUG DD at PINT's time" but "JUG DD at JUG's time vs PINT DD at PINT's time"
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

import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.delays.binary_dd import dd_binary_delay

jug_result = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)

# Get PINT's binary delay (at PINT's barycentric time)
binary_comp = model.components['BinaryDD']
pint_binary_delay_s = binary_comp.binarymodel_delay(toas, None).to('s').value

# Get JUG's barycentric time for binary
tdb_mjd = jug_result['tdb_mjd']
roemer_shapiro_sec = jug_result['roemer_shapiro_sec']
t_binary_jug = tdb_mjd - roemer_shapiro_sec / 86400.0

# Get PINT's barycentric time for binary
t_binary_pint = np.array([b.value for b in model.get_barycentric_toas(toas)])

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

# Compute JUG binary delay at JUG's barycentric time (what JUG actually uses)
jug_binary_at_jug_time = []
for t in t_binary_jug:
    delay = float(dd_binary_delay(
        float(t), pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
    ))
    jug_binary_at_jug_time.append(delay)
jug_binary_at_jug_time = np.array(jug_binary_at_jug_time)

# ACTUAL binary delay difference (JUG at JUG's time - PINT at PINT's time)
actual_binary_diff = (jug_binary_at_jug_time - pint_binary_delay_s) * 1e6  # μs

print("\n" + "="*80)
print("ACTUAL Binary Delay Difference (JUG@JUG_time - PINT@PINT_time)")
print("="*80)
print(f"  Mean: {np.mean(actual_binary_diff):.3f} μs")
print(f"  Std:  {np.std(actual_binary_diff):.3f} μs")
print(f"  Range: [{np.min(actual_binary_diff):.3f}, {np.max(actual_binary_diff):.3f}] μs")

# Get residual difference
pint_resids = Residuals(toas, model)
pint_resid_us = pint_resids.time_resids.to('us').value
jug_resid_us = jug_result['residuals_us']
diff_us = jug_resid_us - pint_resid_us
diff_centered = diff_us - np.mean(diff_us)

print(f"\nResidual difference (centered):")
print(f"  Std: {np.std(diff_centered):.3f} μs")

# Check correlation
actual_binary_diff_centered = actual_binary_diff - np.mean(actual_binary_diff)
r = np.corrcoef(actual_binary_diff_centered, diff_centered)[0, 1]
print(f"\nCorrelation between ACTUAL binary delay diff and residual diff: r = {r:.6f}")

# If binary delay diff explains residual diff, they should be negatively correlated
# (larger binary delay -> smaller residual)
# And the magnitudes should match

# Check if subtracting binary diff (with sign flip) explains residual diff
unexplained = diff_centered + actual_binary_diff_centered  # + because of sign relationship
print(f"\nResidual after accounting for binary delay difference:")
print(f"  Std: {np.std(unexplained):.3f} μs")
print(f"  Variance explained: {100 * (1 - np.var(unexplained)/np.var(diff_centered)):.1f}%")

# Orbital phase analysis
print("\n" + "="*80)
print("Orbital Phase Analysis")
print("="*80)

binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance
orbital_phase = (bo.orbits() % 1).value

n_bins = 12
phase_bins = np.linspace(0, 1, n_bins + 1)

print(f"\n{'Phase':<8} {'BinaryDiff':<12} {'ResidDiff':<12} {'Sum':<12} {'N':<5}")
print("-" * 55)

for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 5:
        phase_center = (phase_bins[i] + phase_bins[i+1]) / 2
        bin_binary_diff = np.mean(actual_binary_diff_centered[mask])
        bin_resid_diff = np.mean(diff_centered[mask])
        bin_sum = bin_binary_diff + bin_resid_diff  # Should be ~0 if binary explains all
        print(f"{phase_center:.2f}     {bin_binary_diff:+8.3f}    {bin_resid_diff:+8.3f}    {bin_sum:+8.3f}    {np.sum(mask)}")

# Time offset analysis
print("\n" + "="*80)
print("Time Offset Analysis")
print("="*80)

time_diff_ms = (t_binary_jug - t_binary_pint) * 86400 * 1000  # ms
print(f"Binary evaluation time difference (JUG - PINT):")
print(f"  Mean: {np.mean(time_diff_ms):.3f} ms")
print(f"  Std:  {np.std(time_diff_ms):.3f} ms")
print(f"  Range: [{np.min(time_diff_ms):.3f}, {np.max(time_diff_ms):.3f}] ms")

# Numerical derivative of binary delay
dt_day = 1e-8  # 0.864 ms
d_binary_dt = []
for i, t in enumerate(t_binary_pint):
    delay_plus = float(dd_binary_delay(
        float(t + dt_day), pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
    ))
    delay_minus = float(dd_binary_delay(
        float(t - dt_day), pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
    ))
    deriv = (delay_plus - delay_minus) / (2 * dt_day * 86400)  # d(delay)/dt (dimensionless)
    d_binary_dt.append(deriv)
d_binary_dt = np.array(d_binary_dt)

# Expected binary delay error from time offset
time_diff_sec = time_diff_ms / 1000
expected_binary_error_us = d_binary_dt * time_diff_sec * 1e6

print(f"\nExpected binary delay error from time offset:")
print(f"  Mean: {np.mean(expected_binary_error_us):.3f} μs")
print(f"  Std:  {np.std(expected_binary_error_us):.3f} μs")

print(f"\nActual binary delay difference:")
print(f"  Mean: {np.mean(actual_binary_diff):.3f} μs")
print(f"  Std:  {np.std(actual_binary_diff):.3f} μs")

# Correlation
r_expected = np.corrcoef(expected_binary_error_us - np.mean(expected_binary_error_us), 
                         actual_binary_diff_centered)[0, 1]
print(f"\nCorrelation between expected and actual binary diff: r = {r_expected:.6f}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"""
If |r| ≈ 1 between actual binary diff and residual diff, and binary diff std ≈ residual diff std,
then the binary delay evaluated at different times IS the cause of the 1.1 μs error.

Actual binary delay diff std: {np.std(actual_binary_diff):.3f} μs
Residual diff std: {np.std(diff_centered):.3f} μs
Correlation: r = {r:.3f}
Variance explained: {100 * (1 - np.var(unexplained)/np.var(diff_centered)):.1f}%
""")
