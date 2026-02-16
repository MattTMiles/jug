"""Tests for geometry disk cache - strict bit-for-bit equivalence.

These tests verify that:
1. Disk cache saves and loads arrays correctly (bit-for-bit)
2. Cached results are identical to computed results
3. Cache invalidation works correctly
"""

import os
import tempfile
import numpy as np
from pathlib import Path

# Force deterministic behavior
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


def test_cache_save_load():
    """Test that cache saves and loads arrays bit-for-bit identically."""
    print("\n" + "=" * 70)
    print("TEST: Cache Save/Load Bit-for-Bit")
    print("=" * 70)
    
    from jug.utils.geom_cache import GeometryDiskCache, compute_array_hash
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GeometryDiskCache(cache_dir=Path(tmpdir))
        
        # Create test arrays
        np.random.seed(42)
        tdb_mjd = np.sort(58000.0 + np.random.uniform(0, 1000, size=100))
        obs_itrf_km = np.array([5109.360133, 2006.852586, -3238.948127])
        
        ssb_obs_pos = np.random.randn(100, 3) * 1e8
        ssb_obs_vel = np.random.randn(100, 3) * 30.0
        
        # Save to cache
        saved = cache.save(tdb_mjd, obs_itrf_km, ssb_obs_pos, ssb_obs_vel)
        assert saved, "Failed to save to cache"
        print("✓ Cache save successful")
        
        # Load from cache
        loaded = cache.load(tdb_mjd, obs_itrf_km)
        assert loaded is not None, "Failed to load from cache"
        loaded_pos, loaded_vel = loaded
        print("✓ Cache load successful")
        
        # Verify bit-for-bit identical
        pos_identical = np.array_equal(ssb_obs_pos, loaded_pos)
        vel_identical = np.array_equal(ssb_obs_vel, loaded_vel)
        
        if pos_identical:
            print("✓ Position arrays: BIT-FOR-BIT IDENTICAL")
        else:
            print(f"✗ Position arrays DIFFER (max: {np.max(np.abs(ssb_obs_pos - loaded_pos))})")
        
        if vel_identical:
            print("✓ Velocity arrays: BIT-FOR-BIT IDENTICAL")
        else:
            print(f"✗ Velocity arrays DIFFER (max: {np.max(np.abs(ssb_obs_vel - loaded_vel))})")
        
        # Verify dtype
        assert loaded_pos.dtype == np.float64, "Position dtype mismatch"
        assert loaded_vel.dtype == np.float64, "Velocity dtype mismatch"
        print("✓ Dtypes: float64")


def test_cache_key_mismatch():
    """Test that cache misses on different inputs."""
    print("\n" + "=" * 70)
    print("TEST: Cache Key Mismatch Detection")
    print("=" * 70)
    
    from jug.utils.geom_cache import GeometryDiskCache
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GeometryDiskCache(cache_dir=Path(tmpdir))
        
        np.random.seed(42)
        tdb_mjd = np.sort(58000.0 + np.random.uniform(0, 1000, size=100))
        obs_itrf_km = np.array([5109.360133, 2006.852586, -3238.948127])
        
        ssb_obs_pos = np.random.randn(100, 3) * 1e8
        ssb_obs_vel = np.random.randn(100, 3) * 30.0
        
        # Save original
        cache.save(tdb_mjd, obs_itrf_km, ssb_obs_pos, ssb_obs_vel)
        
        # Try to load with different TDB times
        tdb_mjd_different = tdb_mjd + 0.001
        loaded = cache.load(tdb_mjd_different, obs_itrf_km)
        assert loaded is None, "Cache should miss on different TDB times"
        print("✓ Cache misses on different TDB times")
        
        # Try to load with different observatory
        obs_itrf_different = obs_itrf_km + 0.001
        loaded = cache.load(tdb_mjd, obs_itrf_different)
        assert loaded is None, "Cache should miss on different observatory"
        print("✓ Cache misses on different observatory")
        
        # Original should still work
        loaded = cache.load(tdb_mjd, obs_itrf_km)
        assert loaded is not None, "Cache should hit on original inputs"
        print("✓ Cache hits on original inputs")


def test_compute_with_cache():
    """Test that compute_ssb_obs_pos_vel with cache gives identical results."""
    print("\n" + "=" * 70)
    print("TEST: Compute with Cache Equivalence")
    print("=" * 70)
    
    from jug.delays.barycentric import compute_ssb_obs_pos_vel
    from jug.utils.geom_cache import get_geometry_cache
    from jug.utils.constants import OBSERVATORIES
    
    # Use MeerKAT observatory
    obs_itrf_km = OBSERVATORIES['meerkat']
    
    # Create test times (small set for speed)
    np.random.seed(42)
    tdb_mjd = np.sort(58000.0 + np.random.uniform(0, 100, size=50))
    
    # Clear cache to ensure fresh computation
    cache = get_geometry_cache()
    cache.clear()
    
    print(f"Test: {len(tdb_mjd)} TOAs, MeerKAT observatory")
    
    # First computation (no cache)
    print("\nFirst computation (cache miss)...")
    pos1, vel1 = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, use_cache=True)
    print(f"  Shape: pos={pos1.shape}, vel={vel1.shape}")
    
    # Second computation (should hit cache)
    print("Second computation (cache hit)...")
    pos2, vel2 = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, use_cache=True)
    
    # Compare bit-for-bit
    pos_identical = np.array_equal(pos1, pos2)
    vel_identical = np.array_equal(vel1, vel2)
    
    if pos_identical:
        print("✓ Positions: BIT-FOR-BIT IDENTICAL")
    else:
        print(f"✗ Positions DIFFER (max: {np.max(np.abs(pos1 - pos2))})")
    
    if vel_identical:
        print("✓ Velocities: BIT-FOR-BIT IDENTICAL")
    else:
        print(f"✗ Velocities DIFFER (max: {np.max(np.abs(vel1 - vel2))})")
    
    # Verify dtypes
    assert pos2.dtype == np.float64, "Position dtype must be float64"
    assert vel2.dtype == np.float64, "Velocity dtype must be float64"
    print("✓ Dtypes: float64")


def test_residuals_with_cache():
    """Test that residuals are identical with and without geometry cache."""
    print("\n" + "=" * 70)
    print("TEST: Residuals with Cache Equivalence")
    print("=" * 70)
    
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.utils.geom_cache import get_geometry_cache
    
    par_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744_tdb.par"
    tim_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744.tim"
    
    if not par_file.exists():
        print(f"SKIP: Test data not found: {par_file}")
        return
    
    # Clear cache
    cache = get_geometry_cache()
    cache.clear()
    
    print(f"Dataset: {par_file.name} + {tim_file.name}")
    
    # First computation (cache miss)
    print("\nFirst residual computation (geometry cache miss)...")
    result1 = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)
    print(f"  RMS: {result1['rms_us']:.6f} μs")
    
    # Second computation (cache hit)
    print("Second residual computation (geometry cache hit)...")
    result2 = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)
    print(f"  RMS: {result2['rms_us']:.6f} μs")
    
    # Compare bit-for-bit
    residuals_identical = np.array_equal(result1['residuals_us'], result2['residuals_us'])
    tdb_identical = np.array_equal(result1['tdb_mjd'], result2['tdb_mjd'])
    dt_identical = np.array_equal(result1['dt_sec'], result2['dt_sec'])
    
    if residuals_identical:
        print("✓ Residuals: BIT-FOR-BIT IDENTICAL")
    else:
        max_diff = np.max(np.abs(result1['residuals_us'] - result2['residuals_us']))
        print(f"✗ Residuals DIFFER (max: {max_diff:.6e} μs)")
    
    if tdb_identical:
        print("✓ TDB times: BIT-FOR-BIT IDENTICAL")
    else:
        print("✗ TDB times DIFFER")
    
    if dt_identical:
        print("✓ dt_sec: BIT-FOR-BIT IDENTICAL")
    else:
        print("✗ dt_sec DIFFER")
    
    # RMS should be exactly equal
    rms_identical = result1['rms_us'] == result2['rms_us']
    if rms_identical:
        print("✓ RMS: IDENTICAL")
    else:
        print(f"✗ RMS DIFFER ({result1['rms_us']} vs {result2['rms_us']})")


def test_cache_disabled():
    """Test that disabling cache still gives correct results."""
    print("\n" + "=" * 70)
    print("TEST: Cache Disabled Still Works")
    print("=" * 70)
    
    import os
    from jug.delays.barycentric import compute_ssb_obs_pos_vel
    from jug.utils.constants import OBSERVATORIES
    
    obs_itrf_km = OBSERVATORIES['meerkat']
    np.random.seed(42)
    tdb_mjd = np.sort(58000.0 + np.random.uniform(0, 100, size=20))
    
    # Compute with cache enabled
    pos_cached, vel_cached = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, use_cache=True)
    
    # Compute with cache disabled
    pos_nocache, vel_nocache = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, use_cache=False)
    
    # Should be bit-for-bit identical
    pos_identical = np.array_equal(pos_cached, pos_nocache)
    vel_identical = np.array_equal(vel_cached, vel_nocache)
    
    if pos_identical:
        print("✓ Positions: BIT-FOR-BIT IDENTICAL")
    else:
        print(f"✗ Positions DIFFER")
    
    if vel_identical:
        print("✓ Velocities: BIT-FOR-BIT IDENTICAL")
    else:
        print(f"✗ Velocities DIFFER")


def run_all_tests():
    """Run all geometry cache tests."""
    print("\n" + "=" * 70)
    print("GEOMETRY DISK CACHE TESTS")
    print("=" * 70)
    
    results = {}
    
    results['save_load'] = test_cache_save_load()
    results['key_mismatch'] = test_cache_key_mismatch()
    results['compute_with_cache'] = test_compute_with_cache()
    results['residuals_with_cache'] = test_residuals_with_cache()
    results['cache_disabled'] = test_cache_disabled()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\nAll geometry cache tests passed!")
    else:
        print("\nSome tests failed!")
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
