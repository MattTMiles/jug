"""Test DD binary model implementation against PINT.

This script tests JUG's DD/DDH/DDGR/DDK binary model support using J0437-4715,
which is one of the best-timed millisecond pulsars and uses the DDK model.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Test paths
PAR_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715_tdb.par"
TIM_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715_tdb.tim"

print("=" * 70)
print("Testing DD Binary Model Implementation")
print("=" * 70)
print(f"\nPulsar: J0437-4715 (DDK model)")
print(f"Par file: {Path(PAR_FILE).name}")
print(f"Tim file: {Path(TIM_FILE).name}")

# Check if files exist
if not Path(PAR_FILE).exists():
    print(f"\n❌ ERROR: Par file not found: {PAR_FILE}")
    exit(1)
if not Path(TIM_FILE).exists():
    print(f"\n❌ ERROR: Tim file not found: {TIM_FILE}")
    exit(1)

# Import JUG
try:
    from jug.residuals.simple_calculator import compute_residuals_simple
    print("\n✓ JUG imported successfully")
except ImportError as e:
    print(f"\n❌ ERROR importing JUG: {e}")
    exit(1)

# Import PINT
try:
    import pint.models
    import pint.toa
    print("✓ PINT imported successfully")
except ImportError as e:
    print(f"\n❌ ERROR importing PINT: {e}")
    exit(1)

# Compute JUG residuals
print("\n" + "=" * 70)
print("Computing JUG Residuals")
print("=" * 70)
try:
    jug_result = compute_residuals_simple(
        PAR_FILE, TIM_FILE,
        clock_dir="/home/mattm/soft/JUG/data/clock",
        observatory="meerkat"
    )
    jug_res_us = jug_result['residuals_us']
    jug_rms_us = jug_result['wrms_us']
    jug_tdb_mjd = jug_result['tdb_mjd']
    jug_errors_us = jug_result.get('errors_us', np.ones(len(jug_res_us)))
    print(f"\n✓ JUG: RMS = {jug_rms_us:.3f} μs (weighted)")
    print(f"  N_TOAs = {jug_result['n_toas']}")
except Exception as e:
    print(f"\n❌ ERROR computing JUG residuals: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compute PINT residuals
print("\n" + "=" * 70)
print("Computing PINT Residuals")
print("=" * 70)
try:
    model = pint.models.get_model(PAR_FILE)
    toas = pint.toa.get_TOAs(TIM_FILE)
    
    # Compute residuals
    pint_res = pint.residuals.Residuals(toas, model)
    pint_res_us = pint_res.time_resids.to('us').value
    pint_errors_us = toas.get_errors().to('us').value
    
    # Weighted RMS
    weights = 1.0 / (pint_errors_us ** 2)
    pint_wrms_us = np.sqrt(np.average(pint_res_us**2, weights=weights))
    
    print(f"\n✓ PINT: RMS = {pint_wrms_us:.3f} μs (weighted)")
    print(f"  N_TOAs = {len(pint_res_us)}")
except Exception as e:
    print(f"\n❌ ERROR computing PINT residuals: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Compare residuals
print("\n" + "=" * 70)
print("Comparison: JUG vs PINT")
print("=" * 70)

diff_us = jug_res_us - pint_res_us
rms_diff = np.sqrt(np.mean(diff_us**2))
max_diff = np.max(np.abs(diff_us))

print(f"\nRMS difference: {rms_diff:.6f} μs")
print(f"Max difference: {max_diff:.6f} μs")
print(f"Mean difference: {np.mean(diff_us):.6f} μs")
print(f"Std difference: {np.std(diff_us):.6f} μs")

# Success criteria
if rms_diff < 1.0:  # Less than 1 microsecond RMS difference
    print(f"\n✅ SUCCESS: DD binary model matches PINT to {rms_diff:.3f} μs")
    status = "✅ PASS"
else:
    print(f"\n⚠️  WARNING: DD model has {rms_diff:.3f} μs RMS difference")
    status = "⚠️ REVIEW"

# Create comparison plot
print("\n" + "=" * 70)
print("Creating Comparison Plot")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: JUG residuals
axes[0].errorbar(jug_tdb_mjd, jug_res_us, yerr=jug_errors_us, 
                 fmt='o', markersize=3, alpha=0.6, capsize=2)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('JUG Residuals (μs)')
axes[0].set_title(f'J0437-4715 DDK Binary Model Test\nJUG: WRMS = {jug_rms_us:.3f} μs')
axes[0].grid(True, alpha=0.3)

# Plot 2: PINT residuals  
axes[1].errorbar(jug_tdb_mjd, pint_res_us, yerr=pint_errors_us,
                 fmt='o', markersize=3, alpha=0.6, capsize=2, color='orange')
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('PINT Residuals (μs)')
axes[1].set_title(f'PINT: WRMS = {pint_wrms_us:.3f} μs')
axes[1].grid(True, alpha=0.3)

# Plot 3: Difference
axes[2].errorbar(jug_tdb_mjd, diff_us, yerr=np.sqrt(jug_errors_us**2 + pint_errors_us**2),
                 fmt='o', markersize=3, alpha=0.6, capsize=2, color='red')
axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[2].set_xlabel('TDB (MJD)')
axes[2].set_ylabel('JUG - PINT (μs)')
axes[2].set_title(f'Difference: RMS = {rms_diff:.6f} μs, Max = {max_diff:.6f} μs [{status}]')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = "J0437-4715_DD_test.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: {plot_file}")
plt.close()

print("\n" + "=" * 70)
print("Test Complete!")
print("=" * 70)
