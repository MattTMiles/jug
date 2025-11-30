#!/usr/bin/env python3
"""
Analyze the time-dependent trend in JUG vs PINT residuals.
Focus on late-time data (MJD > 60500) to isolate the trend.
"""
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# PINT imports
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple

# File paths
par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("=" * 80)
print("RESIDUAL TREND ANALYSIS: JUG vs PINT")
print("=" * 80)

# Load JUG residuals
print("\n1. Computing JUG residuals...")
jug_result = compute_residuals_simple(par_file, tim_file)
jug_res_us = jug_result['residuals_us']
mjd_tdb = jug_result['tdb_mjd']
errors_us = jug_result['errors_us']
print(f"   JUG RMS: {jug_result['rms_us']:.3f} μs")

# Load PINT residuals (BIPM2024)
print("\n2. Computing PINT residuals (BIPM2024)...")
pint_toas = get_TOAs(tim_file, ephem='DE440', bipm_version='BIPM2024', planets=True)
pint_model = get_model(par_file)
pint_resids = Residuals(pint_toas, pint_model)
pint_res_us = pint_resids.time_resids.to_value('us')
print(f"   PINT RMS: {np.std(pint_res_us):.3f} μs")

# Compute difference
diff_us = np.array(jug_res_us - pint_res_us, dtype=np.float64)
diff_ns = diff_us * 1000
mjd_tdb = np.array(mjd_tdb, dtype=np.float64)

print("\n3. Overall difference statistics:")
print(f"   Mean: {np.mean(diff_ns):.3f} ns")
print(f"   Std:  {np.std(diff_ns):.3f} ns")
print(f"   Max:  {np.max(diff_ns):.3f} ns")
print(f"   Min:  {np.min(diff_ns):.3f} ns")

# Analyze full dataset trend
slope_full, intercept_full, r_full, p_full, stderr_full = stats.linregress(mjd_tdb, diff_ns)
print(f"\n4. Full dataset trend (MJD {mjd_tdb.min():.1f} - {mjd_tdb.max():.1f}):")
print(f"   Slope: {slope_full:.6f} ns/day = {slope_full * 365.25:.3f} ns/yr")
print(f"   R²: {r_full**2:.6f}")
print(f"   Total drift: {slope_full * (mjd_tdb.max() - mjd_tdb.min()):.2f} ns")

# Analyze early data (MJD < 60500)
early_mask = mjd_tdb < 60500
if np.sum(early_mask) > 10:
    early_mjd = mjd_tdb[early_mask]
    early_diff = diff_ns[early_mask]
    slope_early, intercept_early, r_early, p_early, stderr_early = stats.linregress(early_mjd, early_diff)
    print(f"\n5. Early data trend (MJD < 60500, N={np.sum(early_mask)}):")
    print(f"   Slope: {slope_early:.6f} ns/day = {slope_early * 365.25:.3f} ns/yr")
    print(f"   R²: {r_early**2:.6f}")
    print(f"   Mean: {np.mean(early_diff):.3f} ns, Std: {np.std(early_diff):.3f} ns")

# Analyze late data (MJD > 60500)
late_mask = mjd_tdb > 60500
if np.sum(late_mask) > 10:
    late_mjd = mjd_tdb[late_mask]
    late_diff = diff_ns[late_mask]
    slope_late, intercept_late, r_late, p_late, stderr_late = stats.linregress(late_mjd, late_diff)
    print(f"\n6. Late data trend (MJD > 60500, N={np.sum(late_mask)}):")
    print(f"   Slope: {slope_late:.6f} ns/day = {slope_late * 365.25:.3f} ns/yr")
    print(f"   R²: {r_late**2:.6f}")
    print(f"   Mean: {np.mean(late_diff):.3f} ns, Std: {np.std(late_diff):.3f} ns")

# Analyze very late data (MJD > 60700)
very_late_mask = mjd_tdb > 60700
if np.sum(very_late_mask) > 10:
    very_late_mjd = mjd_tdb[very_late_mask]
    very_late_diff = diff_ns[very_late_mask]
    slope_very_late, intercept_very_late, r_very_late, p_very_late, stderr_very_late = stats.linregress(very_late_mjd, very_late_diff)
    print(f"\n7. Very late data trend (MJD > 60700, N={np.sum(very_late_mask)}):")
    print(f"   Slope: {slope_very_late:.6f} ns/day = {slope_very_late * 365.25:.3f} ns/yr")
    print(f"   R²: {r_very_late**2:.6f}")
    print(f"   Mean: {np.mean(very_late_diff):.3f} ns, Std: {np.std(very_late_diff):.3f} ns")

# Create diagnostic plots
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Plot 1: Full difference vs time
ax = axes[0]
ax.plot(mjd_tdb, diff_ns, 'o', markersize=3, alpha=0.6, label='JUG - PINT')
fit_line = slope_full * mjd_tdb + intercept_full
ax.plot(mjd_tdb, fit_line, 'r-', linewidth=2, alpha=0.8,
        label=f'Trend: {slope_full:.6f} ns/day ({slope_full*365.25:.3f} ns/yr)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(60500, color='orange', linestyle=':', alpha=0.5, label='MJD 60500')
ax.axvline(60700, color='red', linestyle=':', alpha=0.5, label='MJD 60700')
ax.set_xlabel('MJD (TDB)')
ax.set_ylabel('JUG - PINT (ns)')
ax.set_title(f'Full Dataset: Residual Difference vs Time (R²={r_full**2:.4f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Late data only (MJD > 60500)
ax = axes[1]
if np.sum(late_mask) > 0:
    ax.plot(late_mjd, late_diff, 'o', markersize=3, alpha=0.6, label='JUG - PINT')
    fit_line_late = slope_late * late_mjd + intercept_late
    ax.plot(late_mjd, fit_line_late, 'r-', linewidth=2, alpha=0.8,
            label=f'Trend: {slope_late:.6f} ns/day ({slope_late*365.25:.3f} ns/yr)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(60700, color='red', linestyle=':', alpha=0.5, label='MJD 60700')
    ax.set_xlabel('MJD (TDB)')
    ax.set_ylabel('JUG - PINT (ns)')
    ax.set_title(f'Late Data (MJD > 60500): Trend (R²={r_late**2:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Plot 3: Distribution histogram
ax = axes[2]
ax.hist(diff_ns, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(diff_ns), color='red', linestyle='--', linewidth=2,
          label=f'Mean: {np.mean(diff_ns):.3f} ns')
ax.axvline(np.median(diff_ns), color='orange', linestyle='--', linewidth=2,
          label=f'Median: {np.median(diff_ns):.3f} ns')
ax.set_xlabel('JUG - PINT (ns)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution (Std: {np.std(diff_ns):.3f} ns)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('residual_trend_analysis.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 80)
print("Plot saved: residual_trend_analysis.png")
print("=" * 80)

# Critical question: Is this acceptable?
print("\n" + "=" * 80)
print("ASSESSMENT")
print("=" * 80)
if abs(slope_full * 365.25) < 1.0:
    print(f"✓ Trend is SMALL: {slope_full*365.25:.3f} ns/yr")
    print("  This is ~1000x smaller than the previously reported 6 ns/yr issue.")
    print("  Over 10 years: ~{:.1f} ns cumulative error".format(slope_full * 365.25 * 10))
    print("  Over 25 years: ~{:.1f} ns cumulative error".format(slope_full * 365.25 * 25))
    if abs(slope_full * 365.25 * 25) < 10:
        print("\n  ✓✓ THIS IS ACCEPTABLE for production timing!")
    else:
        print("\n  ⚠ May be problematic for very long baselines (25+ years)")
else:
    print(f"✗ Trend is SIGNIFICANT: {slope_full*365.25:.3f} ns/yr")
    print("  This needs investigation.")
    print(f"  Over 10 years: ~{slope_full * 365.25 * 10:.1f} ns cumulative error")
    print(f"  Over 25 years: ~{slope_full * 365.25 * 25:.1f} ns cumulative error")

print("=" * 80)
