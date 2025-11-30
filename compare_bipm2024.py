#!/usr/bin/env python3
"""Compare JUG vs PINT using BIPM2024 clock files"""

import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import pint.toa as toa
import pint.models as models
import pint.residuals
import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple

# File paths
par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("=" * 60)
print("Comparing JUG vs PINT with BIPM2024")
print("=" * 60)

# Compute JUG residuals
print("\n1. Computing JUG residuals...")
jug_result = compute_residuals_simple(par_file, tim_file)
jug_res_us = jug_result['residuals_us']
mjds = jug_result['tdb_mjd']  # Use TDB MJDs
errors_us = jug_result['errors_us']
print(f"   JUG RMS: {np.std(jug_res_us):.3f} μs")

# Compute PINT residuals with BIPM2024
print("\n2. Computing PINT residuals (BIPM2024)...")
t_bipm = toa.get_TOAs(tim_file, ephem="DE440", bipm_version="BIPM2024", planets=True)
m = models.get_model(par_file)
pint_res_bipm = pint.residuals.Residuals(t_bipm, m)
pint_res_bipm_us = pint_res_bipm.time_resids.to_value('us')
print(f"   PINT (BIPM2024) RMS: {np.std(pint_res_bipm_us):.3f} μs")

# Compute PINT residuals with BIPM2019 (default)
print("\n3. Computing PINT residuals (BIPM2019 default)...")
t_default = toa.get_TOAs(tim_file, ephem="DE440", planets=True)
pint_res_default = pint.residuals.Residuals(t_default, m)
pint_res_default_us = pint_res_default.time_resids.to_value('us')
print(f"   PINT (BIPM2019) RMS: {np.std(pint_res_default_us):.3f} μs")

# Compute differences
diff_bipm = jug_res_us - pint_res_bipm_us
diff_default = jug_res_us - pint_res_default_us

print("\n" + "=" * 60)
print("DIFFERENCE STATISTICS")
print("=" * 60)

print("\nJUG - PINT (BIPM2024):")
print(f"  Mean: {np.mean(diff_bipm):.6f} μs")
print(f"  Std:  {np.std(diff_bipm):.6f} μs")
print(f"  Min:  {np.min(diff_bipm):.6f} μs")
print(f"  Max:  {np.max(diff_bipm):.6f} μs")

print("\nJUG - PINT (BIPM2019 default):")
print(f"  Mean: {np.mean(diff_default):.6f} μs")
print(f"  Std:  {np.std(diff_default):.6f} μs")
print(f"  Min:  {np.min(diff_default):.6f} μs")
print(f"  Max:  {np.max(diff_default):.6f} μs")

# Check late data trend
late_mask = mjds > 60500
if np.any(late_mask):
    late_diff_bipm = diff_bipm[late_mask]
    late_diff_default = diff_default[late_mask]
    late_mjd = mjds[late_mask]
    
    print(f"\nLate data (MJD > 60500, N={np.sum(late_mask)}):")
    print(f"  JUG - PINT (BIPM2024):")
    print(f"    Mean: {np.mean(late_diff_bipm):.6f} μs")
    print(f"    Std:  {np.std(late_diff_bipm):.6f} μs")
    print(f"    Trend: {(late_diff_bipm[-1] - late_diff_bipm[0]) / (late_mjd[-1] - late_mjd[0]) * 1000:.3f} ns/day")
    
    print(f"  JUG - PINT (BIPM2019):")
    print(f"    Mean: {np.mean(late_diff_default):.6f} μs")
    print(f"    Std:  {np.std(late_diff_default):.6f} μs")
    print(f"    Trend: {(late_diff_default[-1] - late_diff_default[0]) / (late_mjd[-1] - late_mjd[0]) * 1000:.3f} ns/day")

# Create comparison plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Residuals
ax = axes[0]
ax.errorbar(mjds, jug_res_us, yerr=errors_us, fmt='o', alpha=0.5, 
            label=f'JUG (RMS={np.std(jug_res_us):.3f} μs)', markersize=2)
ax.errorbar(mjds, pint_res_bipm_us, yerr=errors_us, fmt='x', alpha=0.5,
            label=f'PINT BIPM2024 (RMS={np.std(pint_res_bipm_us):.3f} μs)', markersize=2)
ax.errorbar(mjds, pint_res_default_us, yerr=errors_us, fmt='+', alpha=0.5,
            label=f'PINT BIPM2019 (RMS={np.std(pint_res_default_us):.3f} μs)', markersize=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(60500, color='red', linestyle=':', alpha=0.3, label='MJD 60500')
ax.set_xlabel('MJD')
ax.set_ylabel('Residual (μs)')
ax.set_title('J1909-3744: JUG vs PINT Residuals (Different BIPM Versions)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: JUG - PINT (BIPM2024)
ax = axes[1]
ax.plot(mjds, diff_bipm, 'o', markersize=2, alpha=0.7, label='JUG - PINT (BIPM2024)')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.axhline(np.mean(diff_bipm), color='orange', linestyle='--', alpha=0.5,
           label=f'Mean={np.mean(diff_bipm):.4f} μs')
ax.fill_between(mjds, np.mean(diff_bipm) - np.std(diff_bipm), 
                np.mean(diff_bipm) + np.std(diff_bipm),
                alpha=0.2, label=f'±1σ={np.std(diff_bipm):.4f} μs')
ax.axvline(60500, color='red', linestyle=':', alpha=0.3)
ax.set_xlabel('MJD')
ax.set_ylabel('Difference (μs)')
ax.set_title('Residual Difference: JUG - PINT (BIPM2024)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: JUG - PINT (BIPM2019 default)
ax = axes[2]
ax.plot(mjds, diff_default, 'o', markersize=2, alpha=0.7, label='JUG - PINT (BIPM2019)')
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.axhline(np.mean(diff_default), color='orange', linestyle='--', alpha=0.5,
           label=f'Mean={np.mean(diff_default):.4f} μs')
ax.fill_between(mjds, np.mean(diff_default) - np.std(diff_default),
                np.mean(diff_default) + np.std(diff_default),
                alpha=0.2, label=f'±1σ={np.std(diff_default):.4f} μs')
ax.axvline(60500, color='red', linestyle=':', alpha=0.3)
ax.set_xlabel('MJD')
ax.set_ylabel('Difference (μs)')
ax.set_title('Residual Difference: JUG - PINT (BIPM2019 default)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('J1909-3744_bipm_comparison.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 60)
print("Plot saved: J1909-3744_bipm_comparison.png")
print("=" * 60)
