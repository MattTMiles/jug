#!/usr/bin/env python
"""Direct comparison of JUG and PINT DD binary delays."""

import numpy as np

# PINT imports
import pint
from pint.models import get_model
from pint import toa as pint_toa
import pint.residuals

# JUG imports
from jug.delays.binary_dd import dd_binary_delay
from jug.utils.constants import SECS_PER_DAY

par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.tim"

print("=" * 70)
print("Loading PINT model and TOAs...")
print("=" * 70)

model = get_model(par_file)
toas = pint_toa.get_TOAs(tim_file, model=model)

# Get binary parameters from PINT model
params = {
    'PB': model.PB.value,  # days
    'T0': model.T0.value,  # MJD (with units)
    'A1': model.A1.value,  # light-seconds
    'OM': model.OM.value,  # degrees
    'ECC': model.ECC.value,
    'SINI': model.SINI.value,
    'M2': model.M2.value,  # solar masses
    'OMDOT': model.OMDOT.value if hasattr(model, 'OMDOT') and model.OMDOT.value is not None else 0.0,
    'PBDOT': model.PBDOT.value if hasattr(model, 'PBDOT') and model.PBDOT.value is not None else 0.0,
    'XDOT': model.XDOT.value if hasattr(model, 'XDOT') and model.XDOT.value is not None else 0.0,
    'GAMMA': model.GAMMA.value if hasattr(model, 'GAMMA') and model.GAMMA.value is not None else 0.0,
    'EDOT': model.EDOT.value if hasattr(model, 'EDOT') and model.EDOT.value is not None else 0.0,
}

# Convert to float values
for k, v in params.items():
    try:
        params[k] = float(v)
    except TypeError:
        # Already a float or has no units
        pass

print(f"\nBinary parameters from PINT:")
for k, v in params.items():
    print(f"  {k}: {v}")

# Get TDB times (as MJD floats)
tdb_mjd = toas.get_mjds()
tdb_float = np.array([t.value for t in tdb_mjd])  # Convert from Quantity to float

print(f"\nLoaded {len(tdb_float)} TOAs")
print(f"  First TDB: {tdb_float[0]:.10f}")
print(f"  Last TDB: {tdb_float[-1]:.10f}")

# Get PINT binary delays
print("\n" + "=" * 70)
print("Computing PINT binary delays...")
print("=" * 70)

# PINT computes binary delay via model.binarymodel_delay()
pint_binary_delay = model.binarymodel_delay(toas, None)
pint_binary_us = np.array([d.to('us').value for d in pint_binary_delay])

print(f"PINT binary delays:")
print(f"  Mean: {np.mean(pint_binary_us):.3f} μs")
print(f"  Std: {np.std(pint_binary_us):.3f} μs")
print(f"  Min: {np.min(pint_binary_us):.3f} μs")
print(f"  Max: {np.max(pint_binary_us):.3f} μs")

# Compute JUG binary delays at the same TDB times
print("\n" + "=" * 70)
print("Computing JUG binary delays at TDB times...")
print("=" * 70)

jug_binary_us = []
for t in tdb_float:
    delay = dd_binary_delay(
        t_bary_mjd=t,
        pb_days=params['PB'],
        a1_lt_sec=params['A1'],
        ecc=params['ECC'],
        omega_deg=params['OM'],
        t0_mjd=params['T0'],
        gamma_sec=params['GAMMA'],
        pbdot=params['PBDOT'],
        omdot_deg_yr=params['OMDOT'],
        xdot=params['XDOT'],
        edot=params['EDOT'],
        sini=params['SINI'],
        m2_msun=params['M2'],
    )
    jug_binary_us.append(float(delay) * 1e6)

jug_binary_us = np.array(jug_binary_us)

print(f"JUG binary delays:")
print(f"  Mean: {np.mean(jug_binary_us):.3f} μs")
print(f"  Std: {np.std(jug_binary_us):.3f} μs")
print(f"  Min: {np.min(jug_binary_us):.3f} μs")
print(f"  Max: {np.max(jug_binary_us):.3f} μs")

# Compare
print("\n" + "=" * 70)
print("Comparing JUG vs PINT binary delays")
print("=" * 70)

diff = jug_binary_us - pint_binary_us

print(f"\nDifferences (JUG - PINT):")
print(f"  Mean: {np.mean(diff):.6f} μs")
print(f"  Std: {np.std(diff):.6f} μs")
print(f"  Min: {np.min(diff):.6f} μs")
print(f"  Max: {np.max(diff):.6f} μs")

# Sample comparison at different orbital phases
print("\n" + "-" * 70)
print("Sample binary delay comparison:")
print("-" * 70)
print(f"{'TOA':>6} {'MJD':>14} {'PINT (μs)':>16} {'JUG (μs)':>16} {'Diff (μs)':>14}")

# Select a range of TOAs spanning different orbital phases
indices = np.linspace(0, len(tdb_float)-1, 20, dtype=int)
for i in indices:
    print(f"{i:>6} {tdb_float[i]:>14.6f} {pint_binary_us[i]:>16.3f} {jug_binary_us[i]:>16.3f} {diff[i]:>14.6f}")

# Check if difference correlates with orbital phase
print("\n" + "=" * 70)
print("Checking if difference correlates with orbital phase")
print("=" * 70)

pb = params['PB']
t0 = params['T0']

# Compute orbital phase
orbital_phase = ((tdb_float - t0) / pb) % 1.0

print(f"\nOrbital phase range: {orbital_phase.min():.4f} to {orbital_phase.max():.4f}")

# Bin by orbital phase and compute mean difference in each bin
n_bins = 10
phase_bins = np.linspace(0, 1, n_bins + 1)
print(f"\nMean difference by orbital phase bin:")
print(f"{'Phase bin':>15} {'Mean diff (μs)':>15} {'Std diff (μs)':>15} {'N':>6}")
for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 0:
        mean_diff = np.mean(diff[mask])
        std_diff = np.std(diff[mask])
        print(f"{phase_bins[i]:.2f}-{phase_bins[i+1]:.2f}:".rjust(15) + f"{mean_diff:>15.6f} {std_diff:>15.6f} {np.sum(mask):>6}")

# Let's also look at the Roemer delay specifically
print("\n" + "=" * 70)
print("Investigating Roemer delay component specifically")
print("=" * 70)

# Compute just the Roemer delay (without Shapiro or Einstein)
def compute_roemer_only(t_mjd, pb, a1, ecc, om, t0, omdot, pbdot, xdot, edot):
    """Compute just the Roemer delay for debugging."""
    dt_days = t_mjd - t0
    dt_sec = dt_days * SECS_PER_DAY

    # Periastron advance
    dt_years = dt_days / 365.25
    omega_rad = np.deg2rad(om + omdot * dt_years)

    # Mean anomaly
    pb_sec = pb * SECS_PER_DAY
    tt0 = dt_sec
    orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
    frac_orbits = orbits - np.floor(orbits)
    mean_anomaly = frac_orbits * 2.0 * np.pi

    # Solve Kepler
    E = mean_anomaly
    for _ in range(30):
        E = mean_anomaly + ecc * np.sin(E)

    # Apply secular changes
    a1_current = a1 + xdot * dt_sec
    ecc_current = ecc + edot * dt_sec

    # Roemer delay
    sinE = np.sin(E)
    cosE = np.cos(E)
    sinOm = np.sin(omega_rad)
    cosOm = np.cos(omega_rad)

    alpha = a1_current * sinOm
    beta = a1_current * np.sqrt(1.0 - ecc_current**2) * cosOm

    roemer = alpha * (cosE - ecc_current) + beta * sinE
    return roemer

# Test at first TOA
t_test = tdb_float[0]
roemer_jug = compute_roemer_only(t_test, params['PB'], params['A1'], params['ECC'],
                                 params['OM'], params['T0'], params['OMDOT'],
                                 params['PBDOT'], params['XDOT'], params['EDOT'])

print(f"\nAt first TOA (MJD {t_test:.6f}):")
print(f"  JUG Roemer delay: {roemer_jug * 1e6:.6f} μs")

# Now compare with PINT internal values
# PINT stores the orbit parameters
binary_instance = model.binary_instance
if binary_instance is not None:
    print(f"\nPINT binary instance: {type(binary_instance)}")

    # PINT computes delays differently - let's look at what it calculates
    # The binary model has methods like delayR(), delayS() etc.

# Check if there's a systematic offset vs orbital phase
from scipy.stats import pearsonr
corr, p_value = pearsonr(orbital_phase, diff)
print(f"\nCorrelation between orbital phase and difference:")
print(f"  Pearson r: {corr:.4f}")
print(f"  p-value: {p_value:.2e}")

# Also check derivative (might correlate with orbital velocity)
phase_deriv = np.gradient(orbital_phase)
corr2, p2 = pearsonr(np.abs(phase_deriv), np.abs(diff))
print(f"\nCorrelation between |phase derivative| and |difference|:")
print(f"  Pearson r: {corr2:.4f}")
print(f"  p-value: {p2:.2e}")

# Look at the first few TOAs in detail
print("\n" + "=" * 70)
print("Detailed comparison for first TOA")
print("=" * 70)

t = tdb_float[0]
print(f"TDB time: {t:.10f} MJD")
print(f"T0: {params['T0']:.10f} MJD")
print(f"PB: {params['PB']:.15f} days")

dt_days = t - params['T0']
print(f"\ndt_days = t - T0 = {dt_days:.10f} days")
print(f"Number of orbits = {dt_days / params['PB']:.10f}")

# Check what PINT calculates for M (mean anomaly)
# The orbital phase should match
print(f"Fractional orbit (JUG): {(dt_days / params['PB']) % 1:.10f}")

# If PINT has a different way of computing orbital phase, that could be the issue!
