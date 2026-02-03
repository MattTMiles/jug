#!/usr/bin/env python
"""
Diagnose the ~1 μs pre-fit residual difference between JUG and PINT/Tempo2.

Key finding from previous run:
- JUG pre-fit weighted RMS: 956.6 μs (huge mean offset!)
- PINT pre-fit weighted RMS: 0.167 μs

After removing mean offset:
- Residual difference std: 1.14 μs  <-- This is the problem!

The mean offset (~956 μs) is likely a TZR phase issue.
The varying part (~1.1 μs std) is the orbital-phase-dependent error.

Binary evaluation time difference (JUG - PINT): ~44 ms (the DM delay!)
This is what's causing the orbital-phase-dependent error.
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
import astropy.units as u

# Dataset paths
par_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim"

print("="*80)
print("Loading PINT model and TOAs...")
print("="*80)

model = get_model(par_file)
toas = get_TOAs(tim_file, ephem="de440", planets=True)

print(f"Number of TOAs: {len(toas)}")

# Compute PINT pre-fit residuals
print("\n" + "="*80)
print("PINT Pre-fit Residuals")
print("="*80)

pint_resids = Residuals(toas, model)
pint_resid_us = pint_resids.time_resids.to('us').value

errors_us = toas.get_errors().to('us').value
weights = 1.0 / errors_us**2
pint_wrms = np.sqrt(np.sum(weights * pint_resid_us**2) / np.sum(weights))

print(f"PINT pre-fit weighted RMS: {pint_wrms:.6f} μs")
print(f"PINT pre-fit mean: {np.mean(pint_resid_us):.6f} μs")

# Get JUG pre-fit residuals
print("\n" + "="*80)
print("JUG Pre-fit Residuals")
print("="*80)

import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

from jug.residuals.simple_calculator import compute_residuals_simple

jug_result = compute_residuals_simple(
    par_file, tim_file,
    verbose=False, 
    subtract_tzr=False,
)

jug_resid_us = jug_result['residuals_us']
jug_errors_us = jug_result['errors_us']
jug_weights = 1.0 / jug_errors_us**2
jug_wrms = np.sqrt(np.sum(jug_weights * jug_resid_us**2) / np.sum(jug_weights))

print(f"JUG pre-fit weighted RMS: {jug_wrms:.6f} μs")
print(f"JUG pre-fit mean: {np.mean(jug_resid_us):.6f} μs")

# Compare
diff_us = jug_resid_us - pint_resid_us
mean_offset = np.mean(diff_us)
diff_centered = diff_us - mean_offset

print("\n" + "="*80)
print("Residual Difference Analysis")
print("="*80)
print(f"Mean offset (JUG - PINT): {mean_offset:.3f} μs")
print(f"After removing mean:")
print(f"  Std:  {np.std(diff_centered):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(diff_centered**2)):.6f} μs")

# The ~1.1 μs varying part - analyze its structure
print("\n" + "="*80)
print("Analyzing the ~1.1 μs Varying Component")
print("="*80)

# Get orbital phase
binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance
orbital_phase = (bo.orbits() % 1).value

# Get frequency
freq_mhz = jug_result['freq_bary_mhz']
tdb_mjd = jug_result['tdb_mjd']

# Bin by orbital phase
n_bins = 20
phase_bins = np.linspace(0, 1, n_bins + 1)
bin_means = []
bin_stds = []
bin_centers = []

for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 5:
        bin_means.append(np.mean(diff_centered[mask]))
        bin_stds.append(np.std(diff_centered[mask]))
        bin_centers.append((phase_bins[i] + phase_bins[i+1]) / 2)

bin_means = np.array(bin_means)
bin_stds = np.array(bin_stds)
bin_centers = np.array(bin_centers)

print(f"\nResidual difference vs orbital phase (binned):")
print(f"  Phase range: [{orbital_phase.min():.3f}, {orbital_phase.max():.3f}]")
print(f"  Bin mean range: [{bin_means.min():.3f}, {bin_means.max():.3f}] μs")
print(f"  Peak-to-peak: {bin_means.max() - bin_means.min():.3f} μs")

# Check frequency dependence
print("\n" + "="*80)
print("Frequency Dependence")
print("="*80)

# Split by frequency
freq_median = np.median(freq_mhz)
low_freq_mask = freq_mhz < freq_median
high_freq_mask = freq_mhz >= freq_median

print(f"Median frequency: {freq_median:.1f} MHz")
print(f"Low freq ({freq_mhz[low_freq_mask].mean():.1f} MHz) diff std: {np.std(diff_centered[low_freq_mask]):.3f} μs")
print(f"High freq ({freq_mhz[high_freq_mask].mean():.1f} MHz) diff std: {np.std(diff_centered[high_freq_mask]):.3f} μs")

# The key insight: DM delay is frequency-dependent
# If JUG isn't subtracting DM before computing binary delay, different 
# frequencies will have different time offsets in the binary model

print("\n" + "="*80)
print("ROOT CAUSE ANALYSIS: DM delay in binary time")
print("="*80)

# Get DM delay from PINT
dm_delay_sec = model.constant_dispersion_delay(toas, None).to('s').value
dm_delay_ms = dm_delay_sec * 1000

print(f"\nDM delay statistics:")
print(f"  Mean: {np.mean(dm_delay_ms):.3f} ms")
print(f"  Std:  {np.std(dm_delay_ms):.3f} ms")
print(f"  Range: [{np.min(dm_delay_ms):.3f}, {np.max(dm_delay_ms):.3f}] ms")

# Compute how this time offset affects binary delay at each orbital phase
# Binary delay derivative: d(delay)/d(time) ≈ A1 * n * sin(omega + nu)
# where n = 2π/PB

PB_sec = float(binary_comp.PB.value) * 86400
A1 = float(binary_comp.A1.value)  # light-seconds
n = 2 * np.pi / PB_sec

# nu = true anomaly
nu = bo.nu().value  # radians
omega = bo.omega().value  # radians

# Derivative of Roemer delay w.r.t. time
# dR/dt = -A1 * sin(omega + nu) * (d nu/dt)
# d nu/dt = n * (1 + e*cos(nu))^2 / (1-e^2)^1.5
ecc = float(binary_comp.ECC.value)
dnu_dt = n * (1 + ecc * np.cos(nu))**2 / (1 - ecc**2)**1.5

# Roemer delay derivative
dR_dt = -A1 * np.sin(omega + nu) * dnu_dt

print(f"\nBinary delay derivative (d(Roemer)/dt):")
print(f"  Mean: {np.mean(dR_dt):.9f}")
print(f"  Std:  {np.std(dR_dt):.9f}")
print(f"  Range: [{np.min(dR_dt):.9f}, {np.max(dR_dt):.9f}]")

# Expected residual error from DM time offset
# If JUG uses t_binary = tdbld - (roemer + shapiro)
# but PINT uses t_binary = tdbld - (roemer + shapiro + dm + sw + tropo)
# Then JUG's binary time is LATER by dm_delay

# This means JUG computes binary delay at wrong orbital phase
# Error = dR/dt * dm_delay
expected_error_us = dR_dt * dm_delay_sec * 1e6

print(f"\nExpected residual error from DM time offset:")
print(f"  Mean: {np.mean(expected_error_us):.3f} μs")
print(f"  Std:  {np.std(expected_error_us):.3f} μs")
print(f"  Range: [{np.min(expected_error_us):.3f}, {np.max(expected_error_us):.3f}] μs")

print(f"\nActual residual difference (centered):")
print(f"  Mean: {np.mean(diff_centered):.3f} μs")
print(f"  Std:  {np.std(diff_centered):.3f} μs")
print(f"  Range: [{np.min(diff_centered):.3f}, {np.max(diff_centered):.3f}] μs")

# Correlation between expected and actual
try:
    # Manual Pearson correlation to avoid scipy issues
    x = expected_error_us - np.mean(expected_error_us)
    y = diff_centered - np.mean(diff_centered)
    r = np.sum(x * y) / np.sqrt(np.sum(x**2) * np.sum(y**2))
    print(f"\nCorrelation between expected and actual: r = {r:.6f}")
except Exception as e:
    print(f"\nCorrelation error: {e}")

# Residual after accounting for expected error
unexplained = diff_centered - expected_error_us
print(f"\nUnexplained residual (after DM correction):")
print(f"  Std: {np.std(unexplained):.3f} μs")
print(f"  RMS: {np.sqrt(np.mean(unexplained**2)):.3f} μs")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
The ~1.1 μs pre-fit difference has two components:

1. CONSTANT OFFSET: {mean_offset:.1f} μs
   - Likely TZR phase reference issue
   - Not orbital-phase dependent

2. VARYING COMPONENT: {np.std(diff_centered):.3f} μs std
   - Caused by DM delay NOT being subtracted from binary evaluation time
   - JUG uses: t_binary = tdbld - (roemer + shapiro)
   - PINT uses: t_binary = tdbld - (roemer + shapiro + DM + SW + tropo)
   - The missing DM delay (~{np.mean(dm_delay_ms):.1f} ms, varies with frequency)
     causes different frequencies to evaluate binary delay at slightly
     different orbital phases.

FIX: Subtract DM delay (and SW, tropo) from binary evaluation time in JUG.

Expected improvement: {np.std(diff_centered):.3f} μs → {np.std(unexplained):.3f} μs
""")

# Additional check: what fraction of the 1.1 μs is explained by DM offset?
explained_variance = 1 - np.var(unexplained) / np.var(diff_centered)
print(f"Fraction of variance explained by DM time offset: {explained_variance*100:.1f}%")
