"""
Test that TimingSession caching correctly handles subtract_tzr mode separation.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.engine.session import TimingSession


def test_subtract_tzr_cache_separation():
    """
    Test that subtract_tzr=True and subtract_tzr=False results are cached separately.
    
    This is critical for correctness - we don't want to accidentally return
    subtract_tzr=True results when fitting asks for subtract_tzr=False.
    """
    par_file = Path("data/pulsars/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    if not par_file.exists() or not tim_file.exists():
        print(f"SKIP: Test data not found")
        return
    
    print("Testing subtract_tzr cache separation...")
    
    # Create session
    session = TimingSession(par_file, tim_file, verbose=False)
    
    # Compute with subtract_tzr=True
    print("\n1. Computing residuals with subtract_tzr=True...")
    result_tzr_true = session.compute_residuals(subtract_tzr=True)
    residuals_tzr_true = result_tzr_true['residuals_us'].copy()
    rms_tzr_true = result_tzr_true['rms_us']
    print(f"   RMS (subtract_tzr=True): {rms_tzr_true:.6f} μs")
    
    # Compute with subtract_tzr=False
    print("\n2. Computing residuals with subtract_tzr=False...")
    result_tzr_false = session.compute_residuals(subtract_tzr=False)
    residuals_tzr_false = result_tzr_false['residuals_us'].copy()
    rms_tzr_false = result_tzr_false['rms_us']
    print(f"   RMS (subtract_tzr=False): {rms_tzr_false:.6f} μs")
    
    # Verify they differ (they should!)
    print("\n3. Verifying results differ...")
    if np.array_equal(residuals_tzr_true, residuals_tzr_false):
        raise AssertionError("Results are identical - cache is not separating subtract_tzr modes!")
    
    max_diff = np.max(np.abs(residuals_tzr_true - residuals_tzr_false))
    print(f"   ✓ Results differ (max diff: {max_diff:.6f} μs)")
    
    # Now request again with subtract_tzr=True (should use cache)
    print("\n4. Re-requesting subtract_tzr=True (should use cache)...")
    result_tzr_true_again = session.compute_residuals(subtract_tzr=True)
    residuals_tzr_true_again = result_tzr_true_again['residuals_us']
    
    if not np.array_equal(residuals_tzr_true, residuals_tzr_true_again):
        raise AssertionError("Cached result differs from original - cache broken!")
    print(f"   ✓ Cache returned correct result (subtract_tzr=True)")
    
    # Request again with subtract_tzr=False (should use different cache entry)
    print("\n5. Re-requesting subtract_tzr=False (should use cache)...")
    result_tzr_false_again = session.compute_residuals(subtract_tzr=False)
    residuals_tzr_false_again = result_tzr_false_again['residuals_us']
    
    if not np.array_equal(residuals_tzr_false, residuals_tzr_false_again):
        raise AssertionError("Cached result differs from original - cache broken!")
    print(f"   ✓ Cache returned correct result (subtract_tzr=False)")
    
    # Final check: ensure they still differ
    if np.array_equal(residuals_tzr_true_again, residuals_tzr_false_again):
        raise AssertionError("Cache contamination detected!")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED: Cache correctly separates subtract_tzr modes!")
    print("="*80)


if __name__ == '__main__':
    test_subtract_tzr_cache_separation()
