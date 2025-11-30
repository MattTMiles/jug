"""Test J1909-3744 (ELL1 binary) to verify it matches PINT < 50 ns."""

import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# J1909-3744 files (ELL1 binary)
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim"

print("="*80)
print("J1909-3744 (ELL1 BINARY) VALIDATION TEST")
print("="*80)

# Compute JUG residuals
print("\nComputing JUG residuals...")
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']
print(f"  JUG: {len(jug_res_us)} TOAs")
print(f"  JUG RMS: {np.std(jug_res_us):.6f} μs")

# Compute PINT residuals (use BIPM2024 to match JUG)
print("\nComputing PINT residuals...")
model_pint = get_model(PAR_FILE)
toas_pint = get_TOAs(TIM_FILE, planets=True, ephem="DE440", include_bipm=True, bipm_version="BIPM2024")
res_pint = Residuals(toas_pint, model_pint)
pint_res_us = res_pint.time_resids.to_value('us')
print(f"  PINT: {len(pint_res_us)} TOAs")
print(f"  PINT RMS: {np.std(pint_res_us):.6f} μs")

# Compute difference
diff_us = jug_res_us - pint_res_us
diff_ns = diff_us * 1000.0

print("\n" + "="*80)
print("COMPARISON (JUG - PINT)")
print("="*80)
print(f"\nDifference statistics:")
print(f"  Mean: {np.mean(diff_us):.6f} μs = {np.mean(diff_ns):.3f} ns")
print(f"  RMS:  {np.std(diff_us):.6f} μs = {np.std(diff_ns):.3f} ns")
print(f"  Max:  {np.max(np.abs(diff_us)):.6f} μs = {np.max(np.abs(diff_ns)):.3f} ns")

# Check success
print("\n" + "="*80)
if np.std(diff_ns) < 50:
    print(f"✅ SUCCESS! RMS difference ({np.std(diff_ns):.1f} ns) < 50 ns target")
    print("   ELL1 binary model is working correctly!")
else:
    print(f"❌ FAILED! RMS difference ({np.std(diff_ns):.1f} ns) > 50 ns target")
    print(f"   Factor over target: {np.std(diff_ns) / 50:.1f}×")

# Create comparison plots
print("\nCreating comparison plots...")
mjds = toas_pint.get_mjds().value
dt_years = (mjds - mjds[0]) / 365.25

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: JUG and PINT residuals
ax = axes[0]
ax.plot(dt_years, jug_res_us, 'b.', markersize=2, alpha=0.5, label='JUG')
ax.plot(dt_years, pint_res_us, 'r.', markersize=2, alpha=0.5, label='PINT')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_ylabel('Residuals (μs)')
ax.set_title(f'J1909-3744 (ELL1): JUG vs PINT Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Difference (μs)
ax = axes[1]
ax.plot(dt_years, diff_us, 'g.', markersize=2, alpha=0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_ylabel('Difference (μs)')
ax.set_title(f'Residual Difference (JUG - PINT): RMS = {np.std(diff_us):.6f} μs')
ax.grid(True, alpha=0.3)

# Plot 3: Difference (ns)
ax = axes[2]
ax.plot(dt_years, diff_ns, 'g.', markersize=2, alpha=0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.axhline(50, color='r', linestyle='--', linewidth=1, alpha=0.5, label='±50 ns target')
ax.axhline(-50, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Difference (ns)')
ax.set_title(f'Residual Difference (JUG - PINT): RMS = {np.std(diff_ns):.3f} ns')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Histogram of differences (ns)
ax = axes[3]
ax.hist(diff_ns, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.axvline(50, color='r', linestyle='--', linewidth=1, alpha=0.5, label='±50 ns')
ax.axvline(-50, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Difference (ns)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution: Mean = {np.mean(diff_ns):.3f} ns, RMS = {np.std(diff_ns):.3f} ns')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('j1909_ell1_comparison.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to: j1909_ell1_comparison.png")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nJ1909-3744 (ELL1 binary, {len(jug_res_us)} TOAs):")
print(f"  JUG RMS:  {np.std(jug_res_us):.6f} μs")
print(f"  PINT RMS: {np.std(pint_res_us):.6f} μs")
print(f"  Difference RMS: {np.std(diff_ns):.3f} ns")
if np.std(diff_ns) < 50:
    print(f"\n✅ ELL1 model validated - working correctly!")
    print(f"   This confirms the 1.4 μs issue with J1012-4235 is DD-specific.")
else:
    print(f"\n⚠️  ELL1 model also has issues - need to investigate further")
