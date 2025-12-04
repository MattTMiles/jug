#!/usr/bin/env python3
"""
Compare TDB calculation between JUG and PINT.
This is the first place where delays are applied.
"""
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# PINT
from pint.toa import get_TOAs

# JUG TDB calculation
from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
from jug.io.clock import parse_clock_file
from jug.utils.constants import OBSERVATORIES
from astropy.coordinates import EarthLocation
from astropy import units as u

# File paths
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"
clock_dir = "data/clock"

print("=" * 80)
print("TDB CALCULATION COMPARISON: JUG vs PINT")
print("=" * 80)

# ============================================================================
# JUG TDB calculation
# ============================================================================
print("\n1. Computing JUG TDB (standalone)...")

# Load clock files
mk_clock = parse_clock_file(f"{clock_dir}/mk2utc.clk")
gps_clock = parse_clock_file(f"{clock_dir}/gps2utc.clk")
bipm_clock = parse_clock_file(f"{clock_dir}/tai2tt_bipm2024.clk")

# Parse TOAs
toas = parse_tim_file_mjds(tim_file)
mjd_ints = [toa.mjd_int for toa in toas]
mjd_fracs = [toa.mjd_frac for toa in toas]

# Observatory location
obs_itrf_km = OBSERVATORIES.get('meerkat')
location = EarthLocation.from_geocentric(
    obs_itrf_km[0] * u.km,
    obs_itrf_km[1] * u.km,
    obs_itrf_km[2] * u.km
)

# Compute TDB
jug_tdb_mjd = compute_tdb_standalone_vectorized(
    mjd_ints, mjd_fracs,
    mk_clock, gps_clock, bipm_clock,
    location
)

print(f"   Computed TDB for {len(jug_tdb_mjd)} TOAs")
print(f"   TDB range: {jug_tdb_mjd.min():.6f} to {jug_tdb_mjd.max():.6f}")

# ============================================================================
# PINT TDB calculation
# ============================================================================
print("\n2. Computing PINT TDB (BIPM2024)...")

pint_toas = get_TOAs(tim_file, ephem='DE440', bipm_version='BIPM2024', planets=True)
pint_tdb_mjd = pint_toas.get_mjds(high_precision=False).value

print(f"   Computed TDB for {len(pint_tdb_mjd)} TOAs")
print(f"   TDB range: {pint_tdb_mjd.min():.6f} to {pint_tdb_mjd.max():.6f}")

# ============================================================================
# Compare TDB values
# ============================================================================
print("\n3. TDB difference analysis...")

# Convert to seconds
tdb_diff_sec = np.array((jug_tdb_mjd - pint_tdb_mjd) * 86400, dtype=np.float64)
tdb_diff_ms = tdb_diff_sec * 1000
tdb_diff_ns = tdb_diff_sec * 1e9
mjd_utc = np.array([toa.mjd_int + toa.mjd_frac for toa in toas], dtype=np.float64)

print(f"\nJUG - PINT TDB:")
print(f"  Mean: {np.mean(tdb_diff_ns):.3f} ns")
print(f"  RMS:  {np.std(tdb_diff_ns):.3f} ns")
print(f"  Max:  {np.max(np.abs(tdb_diff_ns)):.3f} ns")

# Check for time-dependent trend

# Full dataset
slope_full, _, r2_full, _, _ = stats.linregress(mjd_utc, tdb_diff_ns)
print(f"\nFull dataset TDB trend (MJD {mjd_utc.min():.1f} - {mjd_utc.max():.1f}):")
print(f"  Slope: {slope_full:.6f} ns/day = {slope_full * 365.25:.3f} ns/yr")
print(f"  R²: {r2_full:.6f}")

# Late data
late_mask = mjd_utc > 60700
if np.sum(late_mask) > 10:
    late_mjd = mjd_utc[late_mask]
    late_diff = tdb_diff_ns[late_mask]
    slope_late, _, r2_late, _, _ = stats.linregress(late_mjd, late_diff)
    print(f"\nLate data TDB trend (MJD > 60700, N={np.sum(late_mask)}):")
    print(f"  Slope: {slope_late:.6f} ns/day = {slope_late * 365.25:.3f} ns/yr")
    print(f"  R²: {r2_late:.6f}")

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Full TDB difference
ax = axes[0]
ax.plot(mjd_utc, tdb_diff_ns, 'o', markersize=2, alpha=0.6)
fit_line = slope_full * mjd_utc + (np.mean(tdb_diff_ns) - slope_full * np.mean(mjd_utc))
ax.plot(mjd_utc, fit_line, 'r-', linewidth=2, alpha=0.8,
        label=f'Trend: {slope_full:.6f} ns/day ({slope_full*365.25:.3f} ns/yr)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(60700, color='red', linestyle=':', alpha=0.5, label='MJD 60700')
ax.set_xlabel('MJD (UTC)')
ax.set_ylabel('JUG - PINT TDB (ns)')
ax.set_title(f'TDB Calculation Difference (R²={r2_full:.4f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Late data
ax = axes[1]
if np.sum(late_mask) > 0:
    ax.plot(late_mjd, late_diff, 'o', markersize=3, alpha=0.6)
    fit_late = slope_late * late_mjd + (np.mean(late_diff) - slope_late * np.mean(late_mjd))
    ax.plot(late_mjd, fit_late, 'r-', linewidth=2, alpha=0.8,
            label=f'Trend: {slope_late:.6f} ns/day ({slope_late*365.25:.3f} ns/yr)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('MJD (UTC)')
    ax.set_ylabel('JUG - PINT TDB (ns)')
    ax.set_title(f'Late Data (MJD > 60700, R²={r2_late:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tdb_calculation_comparison.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 80)
print("Plot saved: tdb_calculation_comparison.png")
print("=" * 80)

# ============================================================================
# Conclusion
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if abs(slope_full * 365.25) > 1.0:
    print(f"\n⚠️  TDB CALCULATION HAS A TREND: {slope_full * 365.25:.3f} ns/yr")
    print("    This is likely where the residual trend originates.")
    print("    Possible causes:")
    print("      - Clock file interpolation differences")
    print("      - TT -> TDB transformation differences")
    print("      - Observatory position calculation differences")
else:
    print(f"\n✓ TDB calculation is stable: {slope_full * 365.25:.3f} ns/yr")
    print("  The residual trend must come from delays AFTER TDB calculation.")

print("=" * 80)
