#!/usr/bin/env python
"""
Diagnose the ~1 μs pre-fit residual difference between JUG and PINT/Tempo2.

Tempo2 pre-fit RMS: 0.167 μs
JUG pre-fit RMS: 1.158 μs

This script will trace through the delay calculations to find the source.
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

# Compute PINT pre-fit residuals (no fitting, just evaluate model)
print("\n" + "="*80)
print("PINT Pre-fit Residuals")
print("="*80)

pint_resids = Residuals(toas, model)
pint_resid_us = pint_resids.time_resids.to('us').value

# Weighted RMS
errors_us = toas.get_errors().to('us').value
weights = 1.0 / errors_us**2
pint_wrms = np.sqrt(np.sum(weights * pint_resid_us**2) / np.sum(weights))

print(f"PINT pre-fit weighted RMS: {pint_wrms:.6f} μs")
print(f"PINT pre-fit unweighted RMS: {np.std(pint_resid_us):.6f} μs")
print(f"PINT pre-fit mean: {np.mean(pint_resid_us):.6f} μs")

# Now compute JUG pre-fit residuals
print("\n" + "="*80)
print("JUG Pre-fit Residuals")
print("="*80)

import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

from jug.residuals.simple_calculator import compute_residuals_simple

# Compute JUG residuals (no fitting)
jug_result = compute_residuals_simple(
    par_file, tim_file,
    verbose=False, 
    subtract_tzr=False,  # Don't subtract mean - want raw pre-fit
)

jug_resid_us = jug_result['residuals_us']
jug_errors_us = jug_result['errors_us']
jug_weights = 1.0 / jug_errors_us**2
jug_wrms = np.sqrt(np.sum(jug_weights * jug_resid_us**2) / np.sum(jug_weights))

print(f"JUG pre-fit weighted RMS: {jug_wrms:.6f} μs")
print(f"JUG pre-fit unweighted RMS: {np.std(jug_resid_us):.6f} μs")
print(f"JUG pre-fit mean: {np.mean(jug_resid_us):.6f} μs")

# Compare residuals directly
print("\n" + "="*80)
print("Direct Comparison (JUG - PINT)")
print("="*80)

# Ensure same ordering (should be same if loaded same .tim file)
diff_us = jug_resid_us - pint_resid_us

print(f"Residual difference (JUG - PINT):")
print(f"  Mean: {np.mean(diff_us):.6f} μs")
print(f"  Std:  {np.std(diff_us):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(diff_us**2)):.6f} μs")
print(f"  Min:  {np.min(diff_us):.6f} μs")
print(f"  Max:  {np.max(diff_us):.6f} μs")

# Check if mean offset explains most of the difference
diff_centered = diff_us - np.mean(diff_us)
print(f"\nAfter removing mean offset:")
print(f"  Std:  {np.std(diff_centered):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(diff_centered**2)):.6f} μs")

# Check correlation with various quantities
print("\n" + "="*80)
print("Correlation Analysis")
print("="*80)

# Get quantities to check correlation
tdb_mjd = jug_result['tdb_mjd']
freq_mhz = jug_result['freq_bary_mhz']

# Orbital phase
binary_comp = model.components['BinaryDD']
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance
orbital_phase = (bo.orbits() % 1).value

from scipy.stats import pearsonr, spearmanr

correlations = {
    'MJD': tdb_mjd,
    'Frequency (MHz)': freq_mhz,
    'Orbital Phase': orbital_phase,
    'PINT residual': pint_resid_us,
}

print(f"\n{'Quantity':<20} {'Pearson r':<12} {'p-value':<12}")
print("-"*50)
for name, values in correlations.items():
    try:
        r, p = pearsonr(values, diff_us)
        print(f"{name:<20} {r:+.6f}     {p:.2e}")
    except Exception as e:
        print(f"{name:<20} ERROR: {e}")

# Check if difference correlates with binary delay itself
print("\n" + "="*80)
print("Binary Delay Comparison")
print("="*80)

# Get PINT binary delay
pint_binary_delay = binary_comp.binarymodel_delay(toas, None).to('s').value

# Get JUG binary delay (need to trace through the computation)
# First, let's see what time JUG is using for binary
roemer_shapiro_sec = jug_result.get('roemer_shapiro_sec', None)
if roemer_shapiro_sec is not None:
    t_binary_jug = tdb_mjd - roemer_shapiro_sec / 86400.0
    print(f"JUG binary evaluation time: tdbld - roemer_shapiro")
else:
    print("WARNING: roemer_shapiro_sec not in JUG result")
    t_binary_jug = tdb_mjd

# Get PINT's barycentric time for binary
t_binary_pint = np.array([b.value for b in model.get_barycentric_toas(toas)])

time_diff_ms = (t_binary_jug - t_binary_pint) * 86400 * 1000
print(f"\nBinary evaluation time difference (JUG - PINT):")
print(f"  Mean: {np.mean(time_diff_ms):.6f} ms")
print(f"  Std:  {np.std(time_diff_ms):.6f} ms")
print(f"  Range: [{np.min(time_diff_ms):.6f}, {np.max(time_diff_ms):.6f}] ms")

# What delays is PINT subtracting that JUG isn't?
print("\n" + "="*80)
print("Delay Differences (what PINT subtracts that JUG doesn't)")
print("="*80)

# PINT subtracts these before binary:
# ss_geo, tropo, ss_shapiro, sw, dm
ss_geo = model.solar_system_geometric_delay(toas, None).to('s').value
ss_shapiro = model.solar_system_shapiro_delay(toas, None).to('s').value
tropo = model.troposphere_delay(toas).to('s').value
sw = model.solar_wind_delay(toas, None).to('s').value
dm = model.constant_dispersion_delay(toas, None).to('s').value

# JUG subtracts: roemer + shapiro (which should match ss_geo + ss_shapiro)
# So the MISSING delays are: tropo + sw + dm

missing_delay_sec = tropo + sw + dm
missing_delay_ms = missing_delay_sec * 1000

print(f"\nMissing delays (tropo + sw + dm):")
print(f"  Troposphere: mean={np.mean(tropo)*1e9:.3f} ns")
print(f"  Solar wind:  mean={np.mean(sw)*1e6:.3f} μs, std={np.std(sw)*1e6:.3f} μs")
print(f"  DM:          mean={np.mean(dm)*1000:.3f} ms, std={np.std(dm)*1000:.3f} ms")
print(f"\n  Total missing: mean={np.mean(missing_delay_ms):.6f} ms, std={np.std(missing_delay_ms):.6f} ms")

# Check if the time difference matches missing delays
computed_time_diff_ms = (ss_geo + ss_shapiro - roemer_shapiro_sec) * 1000 if roemer_shapiro_sec is not None else None
if computed_time_diff_ms is not None:
    print(f"\nVerification: (PINT roemer+shapiro) - (JUG roemer_shapiro):")
    print(f"  Mean: {np.mean(computed_time_diff_ms):.9f} ms")
    print(f"  Std:  {np.std(computed_time_diff_ms):.9f} ms")

# The time difference should be approximately the missing delays
print(f"\nExpected time difference (missing delays): {np.mean(missing_delay_ms):.6f} ms")
print(f"Actual time difference (JUG - PINT):       {np.mean(time_diff_ms):.6f} ms")

# Now compute how this time difference affects binary delay
print("\n" + "="*80)
print("Impact of Time Difference on Binary Delay")
print("="*80)

# d(binary_delay)/dt at each TOA
# Approximate by: (binary_delay(t + dt) - binary_delay(t)) / dt
# But easier: use the fact that d(delay)/dt ≈ nhat * Drep

nhat = bo.nhat().to('1/s').value
Drep = bo.Drep()
# Drep has weird units, let's just compute empirically

# Binary delay sensitivity: how much does binary delay change per second of time offset?
# For small dt: d(binary_delay) ≈ (d binary_delay / d orbital_phase) * (d orbital_phase / dt)
# d(phase)/dt = 1/PB, so d(delay)/dt = (d delay / d phase) / PB

PB_sec = float(binary_comp.PB.value) * 86400

# Get the derivative of binary delay numerically
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

# Compute derivative numerically for a few TOAs
dt_test = 1e-6  # 1 microsecond
print(f"\nNumerical derivative d(binary_delay)/dt at selected TOAs:")
for i in [0, 100, 500, 1000]:
    t = t_binary_pint[i]
    delay1 = float(dd_binary_delay(t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0, 0, 0))
    delay2 = float(dd_binary_delay(t + dt_test/86400, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0, 0, 0))
    deriv = (delay2 - delay1) / dt_test  # d(delay_sec) / d(time_sec)
    
    # Expected residual error from time offset
    time_offset_sec = time_diff_ms[i] / 1000
    expected_delay_error_us = deriv * time_offset_sec * 1e6
    
    print(f"  TOA {i}: d(delay)/dt = {deriv:.9f}, time_offset = {time_offset_sec*1000:.3f} ms")
    print(f"          => expected delay error = {expected_delay_error_us:.3f} μs")
    print(f"          Actual residual diff = {diff_us[i]:.3f} μs")

# Compute expected error for all TOAs
print("\n" + "="*80)
print("Full Prediction of Residual Difference")
print("="*80)

# Compute derivative for all TOAs
derivatives = []
for i, t in enumerate(t_binary_pint):
    delay1 = float(dd_binary_delay(t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0, 0, 0))
    delay2 = float(dd_binary_delay(t + dt_test/86400, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, 0, 0, 0))
    derivatives.append((delay2 - delay1) / dt_test)
derivatives = np.array(derivatives)

# Expected residual error
time_offset_sec = time_diff_ms / 1000
expected_error_us = derivatives * time_offset_sec * 1e6

print(f"Expected residual error from time difference:")
print(f"  Mean: {np.mean(expected_error_us):.6f} μs")
print(f"  Std:  {np.std(expected_error_us):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(expected_error_us**2)):.6f} μs")

print(f"\nActual residual difference (JUG - PINT):")
print(f"  Mean: {np.mean(diff_us):.6f} μs")
print(f"  Std:  {np.std(diff_us):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(diff_us**2)):.6f} μs")

# Check if expected error explains the difference
unexplained = diff_us - expected_error_us
print(f"\nUnexplained residual (actual - expected):")
print(f"  Mean: {np.mean(unexplained):.6f} μs")
print(f"  Std:  {np.std(unexplained):.6f} μs")
print(f"  RMS:  {np.sqrt(np.mean(unexplained**2)):.6f} μs")

# Correlation between expected and actual
r, p = pearsonr(expected_error_us, diff_us)
print(f"\nCorrelation between expected and actual: r={r:.6f}, p={p:.2e}")
