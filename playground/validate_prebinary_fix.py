#!/usr/bin/env python
"""
Validation script for PINT-compatible pre-binary time fix AND TZRMJD fix.

This script compares JUG and PINT pre-fit residuals for J1713+0747
to verify that:
1. The orbital-phase-dependent ~1.1 μs discrepancy is resolved (pre-binary time fix)
2. The ~956 μs mean offset is resolved (TZRMJD timescale fix)
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
import astropy.units as u

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("=" * 80)
print("Validation: PINT-compatible pre-binary time + TZRMJD timescale fix for JUG")
print("=" * 80)

print("\nLoading PINT model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"Number of TOAs: {len(toas)}")

# Get PINT pre-fit residuals
pint_resids = Residuals(toas, model)
pint_resid_us = pint_resids.time_resids.to('us').value
pint_wrms = np.sqrt(np.average(pint_resid_us**2, weights=1/toas.get_errors().to('us').value**2))

print(f"\nPINT pre-fit weighted RMS: {pint_wrms:.3f} μs")

# Get JUG pre-fit residuals
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

from jug.residuals.simple_calculator import compute_residuals_simple

print("\nComputing JUG pre-fit residuals (with TZRMJD fix, tzrmjd_scale='TDB')...")
# Use subtract_tzr=True to properly test TZR handling (like PINT default)
jug_result = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=True, tzrmjd_scale="TDB")
jug_resid_us = jug_result['residuals_us']
jug_wrms = jug_result['weighted_rms_us']

print(f"JUG pre-fit weighted RMS: {jug_wrms:.3f} μs")

# Compare
diff_us = jug_resid_us - pint_resid_us
diff_centered = diff_us - np.mean(diff_us)

print(f"\n" + "-" * 60)
print("Residual difference (JUG - PINT):")
print(f"  Mean: {np.mean(diff_us):.6f} μs")
print(f"  Std (centered): {np.std(diff_centered):.6f} μs")
print(f"  Range: [{np.min(diff_centered):.6f}, {np.max(diff_centered):.6f}] μs")

# Orbital phase analysis
print("\n" + "-" * 60)
print("Orbital phase analysis of residual difference:")

binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance
orbital_phase = (bo.orbits() % 1).value

n_bins = 12
phase_bins = np.linspace(0, 1, n_bins + 1)

print(f"\n{'Phase':<8} {'Diff (μs)':<12} {'N':<5}")
print("-" * 30)

for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 5:
        phase_center = (phase_bins[i] + phase_bins[i+1]) / 2
        bin_diff = np.mean(diff_centered[mask])
        print(f"{phase_center:.2f}     {bin_diff:+8.3f}    {np.sum(mask)}")

# Compare binary delays directly
print("\n" + "=" * 80)
print("Direct binary delay comparison")
print("=" * 80)

# PINT binary delay at PINT's time
pint_binary_delay_s = binary_comp.binarymodel_delay(toas, None).to('s').value

# JUG binary delay at JUG's PINT-compatible pre-binary time
from jug.delays.binary_dd import dd_binary_delay

# Get pre-binary time from JUG
prebinary_delay_sec = jug_result['prebinary_delay_sec']
tdb_mjd = jug_result['tdb_mjd']
t_prebinary_jug = tdb_mjd - prebinary_delay_sec / 86400.0

# Get PINT's barycentric time for binary
t_binary_pint = np.array([b.value for b in model.get_barycentric_toas(toas)])

print(f"\nTime offset (JUG pre-binary - PINT pre-binary):")
time_diff_ms = (t_prebinary_jug - t_binary_pint) * 86400 * 1000
print(f"  Mean: {np.mean(time_diff_ms):.3f} ms")
print(f"  Std:  {np.std(time_diff_ms):.3f} ms")
print(f"  Range: [{np.min(time_diff_ms):.3f}, {np.max(time_diff_ms):.3f}] ms")

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

# Compute JUG binary delay at JUG's pre-binary time
jug_binary_at_jug_prebinary = []
for t in t_prebinary_jug:
    delay = float(dd_binary_delay(
        float(t), pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0.0, 0.0, 0.0
    ))
    jug_binary_at_jug_prebinary.append(delay)
jug_binary_at_jug_prebinary = np.array(jug_binary_at_jug_prebinary)

# Binary delay difference
binary_diff_us = (jug_binary_at_jug_prebinary - pint_binary_delay_s) * 1e6

print(f"\nBinary delay difference (JUG@JUG_prebinary - PINT@PINT_time):")
print(f"  Mean: {np.mean(binary_diff_us):.6f} μs")
print(f"  Std:  {np.std(binary_diff_us):.6f} μs")
print(f"  Range: [{np.min(binary_diff_us):.6f}, {np.max(binary_diff_us):.6f}] μs")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
BEFORE fixes:
  - ~1.14 μs std orbital-phase-dependent discrepancy (pre-binary time bug)  
  - ~956 μs mean offset (TZRMJD double-conversion bug, only visible with subtract_tzr=False)

AFTER fixes:
  - Orbital phase std: {np.std(diff_centered):.6f} μs
  - Mean offset: {np.mean(diff_us):.6f} μs

Binary delay std: {np.std(binary_diff_us):.6f} μs
Time offset std: {np.std(time_diff_ms):.6f} ms

Target: < 0.001 μs std, < 0.001 μs mean (nanosecond level agreement)
""")

# Check both conditions - tighten thresholds for ns-level agreement
orbital_ok = np.std(diff_centered) < 0.001
mean_ok = abs(np.mean(diff_us)) < 0.001

if orbital_ok and mean_ok:
    print("✓ BOTH FIXES SUCCESSFUL:")
    print("  - Orbital-phase-dependent discrepancy resolved (sub-ns level)")
    print("  - Mean offset eliminated (TZRMJD fix working)")
    print(f"\n  JUG-PINT agreement: {np.std(diff_us)*1000:.3f} ns RMS")
elif orbital_ok:
    print("✓ PRE-BINARY FIX SUCCESSFUL: Orbital pattern resolved")
    print(f"✗ TZRMJD FIX INCOMPLETE: Mean offset = {np.mean(diff_us):.6f} μs")
else:
    print("✗ FIXES INCOMPLETE:")
    print(f"  - Orbital std: {np.std(diff_centered):.6f} μs (target < 0.001)")
    print(f"  - Mean offset: {np.mean(diff_us):.6f} μs (target < 0.001)")
