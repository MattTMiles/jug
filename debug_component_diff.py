#!/usr/bin/env python3
"""Debug the 15 ns time-correlated difference between JUG and PINT."""

import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds

# PINT imports
import pint.models
import pint.toa
import pint.residuals
import astropy.units as u

# File paths
PAR_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
TIM_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("="*60)
print("Component-Level Comparison: JUG vs PINT")
print("="*60)

# Compute JUG residuals
print("\nComputing JUG residuals...")
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']

# Compute PINT residuals
print("\nComputing PINT residuals...")
pint_model = pint.models.get_model(PAR_FILE)
pint_toas = pint.toa.get_TOAs(TIM_FILE, model=pint_model)
pint_res = pint.residuals.Residuals(pint_toas, pint_model, use_weighted_mean=False)
pint_res_us = pint_res.time_resids.to(u.us).value

# Compare
diff_us = jug_res_us - pint_res_us
diff_ns = diff_us * 1000

print(f"\n{'='*60}")
print("RESIDUAL COMPARISON")
print(f"{'='*60}")
print(f"Difference (JUG - PINT):")
print(f"  Mean:  {np.mean(diff_ns):8.3f} ns")
print(f"  Std:   {np.std(diff_ns):8.3f} ns")
print(f"  Min:   {np.min(diff_ns):8.3f} ns")
print(f"  Max:   {np.max(diff_ns):8.3f} ns")
print(f"  Range: {np.max(diff_ns) - np.min(diff_ns):8.3f} ns")

# Parse TOAs to get MJDs
toas = parse_tim_file_mjds(TIM_FILE)
mjds = np.array([t.mjd_int + t.mjd_frac for t in toas])

# Check time correlation
correlation = np.corrcoef(mjds, diff_ns)[0,1]
print(f"\nTime correlation: {correlation:.6f}")

if abs(correlation) > 0.5:
    print("⚠️  SIGNIFICANT time correlation detected!")
    print("   This suggests a systematic error in one of the delay components.")
else:
    print("✓  No significant time correlation.")

# Plot the difference
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top: Difference vs time
ax1.plot(mjds, diff_ns, 'o', ms=3, alpha=0.6, color='red')
ax1.axhline(0, color='k', ls='--', alpha=0.3)
ax1.set_xlabel('MJD', fontsize=12)
ax1.set_ylabel('Residual Difference (JUG - PINT) [ns]', fontsize=12)
ax1.set_title(f'J1909-3744: Residual Difference (corr={correlation:.4f})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Bottom: Both residuals overlaid
ax2.plot(mjds, jug_res_us, 'o', ms=2, alpha=0.5, label='JUG', color='blue')
ax2.plot(mjds, pint_res_us, 'x', ms=2, alpha=0.5, label='PINT', color='orange')
ax2.axhline(0, color='k', ls='--', alpha=0.3)
ax2.set_xlabel('MJD', fontsize=12)
ax2.set_ylabel('Residual [μs]', fontsize=12)
ax2.set_title('Residuals Comparison', fontsize=12)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jug_pint_component_debug.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: jug_pint_component_debug.png")

print(f"\n{'='*60}")
print("Analysis complete.")
print(f"{'='*60}")
