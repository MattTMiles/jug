"""Test that the binary delay patch is working correctly."""

import numpy as np
from pathlib import Path
from pint.models import get_model
from pint.toa import get_TOAs

from jug.residuals.simple_calculator import compute_residuals_simple

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

print("="*80)
print("TESTING BINARY DELAY PATCH")
print("="*80)

# ===== PINT RESIDUALS =====
print("\n[PINT] Computing residuals...")
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

# Compute PINT residuals
from pint.residuals import Residuals
residuals_pint = Residuals(toas, model)
resid_us_pint = residuals_pint.time_resids.to_value('us')

print(f"  PINT residuals: RMS={np.std(resid_us_pint):.3f} μs, mean={np.mean(resid_us_pint):.3f} μs")
print(f"  Range: [{np.min(resid_us_pint):.3f}, {np.max(resid_us_pint):.3f}] μs")

# ===== JUG RESIDUALS =====
print("\n[JUG] Computing residuals...")
result = compute_residuals_simple(par_file, tim_file, verbose=False)
resid_us_jug = result['residuals_us']

print(f"  JUG residuals: RMS={np.std(resid_us_jug):.3f} μs, mean={np.mean(resid_us_jug):.3f} μs")
print(f"  Range: [{np.min(resid_us_jug):.3f}, {np.max(resid_us_jug):.3f}] μs")

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Remove means for comparison
resid_pint_centered = resid_us_pint - np.mean(resid_us_pint)
resid_jug_centered = resid_us_jug - np.mean(resid_us_jug)

diff = resid_jug_centered - resid_pint_centered
print(f"\nResidual difference (JUG - PINT):")
print(f"  Mean: {np.mean(diff):.3f} μs")
print(f"  RMS:  {np.std(diff):.3f} μs")
print(f"  Range: [{np.min(diff):.3f}, {np.max(diff):.3f}] μs")

# Check for orbital-period correlation
from jug.io.par_reader import parse_par_file
params = parse_par_file(par_file)
pb_days = float(params.get('PB', 67.825))

# Get TDB times for phase calculation
tdb_mjd = result['tdb_mjd']
t0 = float(params.get('T0', 0.0))
if t0 == 0.0:
    t0 = float(params.get('TASC', 0.0))

# Compute orbital phase
orbital_phase = ((tdb_mjd - t0) / pb_days) % 1.0

# Sort by orbital phase and look for patterns
sorted_idx = np.argsort(orbital_phase)
phase_sorted = orbital_phase[sorted_idx]
diff_sorted = diff[sorted_idx]

# Bin by orbital phase
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
phase_bin_idx = np.digitize(phase_sorted, bins) - 1
bin_means = []
bin_phases = []
for i in range(n_bins):
    mask = phase_bin_idx == i
    if np.sum(mask) > 0:
        bin_means.append(np.mean(diff_sorted[mask]))
        bin_phases.append((bins[i] + bins[i+1]) / 2)

print(f"\nOrbital-phase-binned difference:")
print(f"  Phase range  |  Mean diff (μs)")
print(f"  -------------|----------------")
for p, m in zip(bin_phases, bin_means):
    bar = "*" * int(abs(m) / 0.5) if abs(m) > 0.1 else ""
    print(f"  {p:.2f}          |  {m:+.3f}  {bar}")

print("\n" + "="*80)
print("RESULT")
print("="*80)
if np.std(diff) < 1.0:
    print(f"✓ JUG and PINT residuals agree to within {np.std(diff):.3f} μs RMS")
else:
    print(f"⚠️  Significant disagreement: {np.std(diff):.1f} μs RMS")
    print(f"   Max difference: {np.max(np.abs(diff)):.1f} μs")
