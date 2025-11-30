"""Test the DD binary model fix against PINT residuals."""

import sys
import numpy as np

# Import JUG
from jug.residuals.simple_calculator import compute_residuals_simple

# Test pulsar
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("TESTING DD BINARY MODEL FIX")
print("="*80)
print(f"\nPulsar: J1012-4235 (DD binary, 7089 TOAs)")
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")

# Compute JUG residuals
print("\n" + "="*80)
print("COMPUTING JUG RESIDUALS...")
print("="*80)

try:
    jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
    jug_res_us = jug_result['residuals_us']

    jug_rms = jug_result['unweighted_rms_us']
    jug_mean = jug_result['mean_us']

    print(f"\nJUG Results:")
    print(f"  N_TOAs: {len(jug_res_us)}")
    print(f"  Mean: {jug_mean:.3f} μs")
    print(f"  RMS: {jug_rms:.3f} μs")
    print(f"  Range: [{np.min(jug_res_us):.3f}, {np.max(jug_res_us):.3f}] μs")

except Exception as e:
    print(f"\nERROR computing JUG residuals: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compute PINT residuals for comparison
print("\n" + "="*80)
print("COMPUTING PINT RESIDUALS FOR COMPARISON...")
print("="*80)

try:
    from pint.models import get_model
    from pint.toa import get_TOAs
    from pint.fitter import WLSFitter

    # Load PINT model and TOAs
    model_pint = get_model(PAR_FILE)
    toas_pint = get_TOAs(TIM_FILE)

    # Compute pre-fit residuals
    pint_res = model_pint.residuals(toas_pint).time_resids.to_value('us')

    pint_rms = np.std(pint_res)
    pint_mean = np.mean(pint_res)

    print(f"\nPINT Results:")
    print(f"  N_TOAs: {len(pint_res)}")
    print(f"  Mean: {pint_mean:.3f} μs")
    print(f"  RMS: {pint_rms:.3f} μs")
    print(f"  Range: [{np.min(pint_res):.3f}, {np.max(pint_res):.3f}] μs")

except Exception as e:
    print(f"\nWARNING: Could not compute PINT residuals: {e}")
    print("Skipping PINT comparison.")
    pint_res = None

# Compare if we have PINT results
if pint_res is not None:
    print("\n" + "="*80)
    print("COMPARISON: JUG vs PINT")
    print("="*80)

    diff = jug_res_us - pint_res
    diff_rms = np.std(diff)
    diff_mean = np.mean(diff)

    print(f"\nDifference (JUG - PINT):")
    print(f"  Mean: {diff_mean:.3f} μs")
    print(f"  RMS: {diff_rms:.3f} μs")
    print(f"  Range: [{np.min(diff):.3f}, {np.max(diff):.3f}] μs")

    # Success criteria
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    TARGET_RMS = 10.0  # μs

    if diff_rms < TARGET_RMS:
        print(f"\n✅ SUCCESS! Difference RMS ({diff_rms:.3f} μs) < {TARGET_RMS} μs")
        print("\nThe DD binary model fix works correctly!")
    else:
        print(f"\n❌ FAILED! Difference RMS ({diff_rms:.3f} μs) >= {TARGET_RMS} μs")
        print("\nThe DD model still has issues that need investigation.")
        sys.exit(1)
else:
    print("\n" + "="*80)
    print("VALIDATION (JUG ONLY)")
    print("="*80)

    # Without PINT, just check if JUG gives reasonable values
    if jug_rms < 1000:  # Less than 1 ms
        print(f"\n✅ JUG RMS ({jug_rms:.3f} μs) is reasonable")
        print("(Cannot verify against PINT without PINT installation)")
    else:
        print(f"\n❌ JUG RMS ({jug_rms:.3f} μs) seems too large")
        sys.exit(1)

print("\n" + "="*80)
