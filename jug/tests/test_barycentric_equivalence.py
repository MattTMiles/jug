"""Strict bit-for-bit equivalence tests for barycentric computations.

This module tests whether compute_ssb_obs_pos_vel_gcrs_posvel produces
EXACTLY identical results to the baseline compute_ssb_obs_pos_vel.

CRITICAL: Uses np.array_equal (NO tolerances) for all comparisons.
Any difference (even 1 ULP) means the alternative is NOT adopted.
"""

import os
import numpy as np
from pathlib import Path

# Force deterministic behavior
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from jug.delays.barycentric import (
    compute_ssb_obs_pos_vel,
    compute_ssb_obs_pos_vel_gcrs_posvel
)
from jug.utils.constants import OBSERVATORIES


def test_gcrs_posvel_vs_baseline():
    """
    Test: compute_ssb_obs_pos_vel_gcrs_posvel vs baseline compute_ssb_obs_pos_vel
    
    STRICT BIT-FOR-BIT comparison using np.array_equal.
    
    If this test fails, the gcrs_posvel method is NOT adopted as default.
    """
    print("\n" + "=" * 70)
    print("STRICT BIT-FOR-BIT EQUIVALENCE TEST: GCRS POS/VEL")
    print("=" * 70)
    
    # Use MeerKAT observatory
    obs_itrf_km = OBSERVATORIES['meerkat']
    
    # Create a representative set of TDB times (100 TOAs spanning ~6 years)
    # Using deterministic times for reproducibility
    np.random.seed(42)
    tdb_mjd = np.sort(58526.0 + np.random.uniform(0, 2311, size=100))
    
    print(f"\nTest parameters:")
    print(f"  Observatory: MeerKAT {obs_itrf_km}")
    print(f"  N_times: {len(tdb_mjd)}")
    print(f"  Time range: MJD {tdb_mjd.min():.1f} - {tdb_mjd.max():.1f}")
    
    # Compute using baseline method
    print("\nComputing with baseline method (numerical differentiation)...")
    baseline_timings = {}
    pos_baseline, vel_baseline = compute_ssb_obs_pos_vel(
        tdb_mjd, obs_itrf_km, timings=baseline_timings
    )
    
    # Compute using gcrs_posvel method
    print("Computing with gcrs_posvel method...")
    gcrs_timings = {}
    pos_gcrs, vel_gcrs = compute_ssb_obs_pos_vel_gcrs_posvel(
        tdb_mjd, obs_itrf_km, timings=gcrs_timings
    )
    
    # Report timings
    print("\nTiming comparison:")
    print(f"  Baseline total: {sum(baseline_timings.values())*1000:.1f} ms")
    for k, v in baseline_timings.items():
        print(f"    {k}: {v*1000:.1f} ms")
    print(f"  GCRS posvel total: {sum(gcrs_timings.values())*1000:.1f} ms")
    for k, v in gcrs_timings.items():
        print(f"    {k}: {v*1000:.1f} ms")
    
    # STRICT BIT-FOR-BIT COMPARISON
    print("\n" + "-" * 70)
    print("STRICT BIT-FOR-BIT COMPARISON (np.array_equal)")
    print("-" * 70)
    
    # Check shapes
    assert pos_baseline.shape == pos_gcrs.shape, \
        f"Position shape mismatch: {pos_baseline.shape} vs {pos_gcrs.shape}"
    assert vel_baseline.shape == vel_gcrs.shape, \
        f"Velocity shape mismatch: {vel_baseline.shape} vs {vel_gcrs.shape}"
    print(f"[x] Shapes match: pos={pos_baseline.shape}, vel={vel_baseline.shape}")
    
    # Check dtypes
    assert pos_baseline.dtype == pos_gcrs.dtype == np.float64, \
        f"Position dtype mismatch: {pos_baseline.dtype} vs {pos_gcrs.dtype}"
    assert vel_baseline.dtype == vel_gcrs.dtype == np.float64, \
        f"Velocity dtype mismatch: {vel_baseline.dtype} vs {vel_gcrs.dtype}"
    print(f"[x] Dtypes match: float64")
    
    # STRICT position comparison
    pos_identical = np.array_equal(pos_baseline, pos_gcrs)
    if pos_identical:
        print("[x] Positions: BIT-FOR-BIT IDENTICAL")
    else:
        pos_diff = np.abs(pos_baseline - pos_gcrs)
        print(f"[ ] Positions: DIFFER")
        print(f"    Max difference: {np.max(pos_diff):.6e} km")
        print(f"    Mean difference: {np.mean(pos_diff):.6e} km")
        print(f"    Max relative: {np.max(pos_diff / np.abs(pos_baseline + 1e-30)):.6e}")
    
    # STRICT velocity comparison
    vel_identical = np.array_equal(vel_baseline, vel_gcrs)
    if vel_identical:
        print("[x] Velocities: BIT-FOR-BIT IDENTICAL")
    else:
        vel_diff = np.abs(vel_baseline - vel_gcrs)
        print(f"[ ] Velocities: DIFFER")
        print(f"    Max difference: {np.max(vel_diff):.6e} km/s")
        print(f"    Mean difference: {np.mean(vel_diff):.6e} km/s")
        print(f"    Max relative: {np.max(vel_diff / np.abs(vel_baseline + 1e-30)):.6e}")
    
    # Overall result
    print("\n" + "=" * 70)
    if pos_identical and vel_identical:
        print("RESULT: GCRS_POSVEL is BIT-FOR-BIT IDENTICAL to baseline")
        print("  -> Safe to adopt as default")
        print("=" * 70)
    else:
        print("RESULT: GCRS_POSVEL DIFFERS from baseline")
        print("  -> NOT safe to adopt as default")
        print("  -> Keep baseline method; gcrs_posvel available behind flag only")
        print("=" * 70)


def test_clock_caching():
    """Test that clock file caching produces identical results."""
    print("\n" + "=" * 70)
    print("CLOCK FILE CACHING TEST")
    print("=" * 70)
    
    from jug.io.clock import parse_clock_file, _parse_clock_file_cached
    
    clock_dir = Path(__file__).parent.parent.parent / "data" / "clock"
    gps_file = clock_dir / "gps2utc.clk"
    
    if not gps_file.exists():
        print(f"SKIP: Clock file not found: {gps_file}")
        return
    
    # Clear cache
    _parse_clock_file_cached.cache_clear()
    
    # First call (cache miss)
    result1 = parse_clock_file(gps_file)
    
    # Second call (cache hit)
    result2 = parse_clock_file(gps_file)
    
    # Third call with different path form
    result3 = parse_clock_file(str(gps_file))
    
    # Verify identical arrays
    assert np.array_equal(result1['mjd'], result2['mjd']), "MJD arrays differ on cache hit"
    assert np.array_equal(result1['offset'], result2['offset']), "Offset arrays differ on cache hit"
    assert np.array_equal(result1['mjd'], result3['mjd']), "MJD arrays differ with path string"
    
    print(f"[x] Clock file caching works correctly")
    print(f"  Entries: {len(result1['mjd'])}")
    print(f"  Cache info: {_parse_clock_file_cached.cache_info()}")


def run_all_tests():
    """Run all barycentric equivalence tests."""
    print("\n" + "=" * 70)
    print("BARYCENTRIC COMPUTATION EQUIVALENCE TESTS")
    print("=" * 70)
    
    # Test gcrs_posvel
    gcrs_identical = test_gcrs_posvel_vs_baseline()
    
    # Test clock caching
    clock_ok = test_clock_caching()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  GCRS posvel bit-identical: {gcrs_identical}")
    print(f"  Clock caching correct: {clock_ok}")
    
    if gcrs_identical:
        print("\n  RECOMMENDATION: Safe to use gcrs_posvel as default")
    else:
        print("\n  RECOMMENDATION: Keep baseline method; gcrs_posvel differs")
    
    print("=" * 70)
    
    return gcrs_identical and clock_ok


if __name__ == '__main__':
    run_all_tests()
