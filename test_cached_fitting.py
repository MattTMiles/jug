#!/usr/bin/env python3
"""
Test cached residual calculator performance
============================================

Compare performance of cached vs uncached fitting.
"""

import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from jug.fitting.cached_residuals import OptimizedFitter


def test_cached_fitting():
    """Test fitting with cached residual calculator."""
    print("="*80)
    print("TESTING CACHED RESIDUAL CALCULATOR")
    print("="*80)
    
    par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    print(f"\nInput files:")
    print(f"  Par: {par_file}")
    print(f"  Tim: {tim_file}")
    
    # Create fitter
    print(f"\nCreating optimized fitter...")
    fitter = OptimizedFitter(par_file, tim_file)
    
    # Fit F0 and F1
    print(f"\nFitting F0 + F1 with cached residuals...")
    start_time = time.time()
    
    results = fitter.fit(['F0', 'F1'], max_iter=25)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    print(f"\nConvergence:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Time per iteration: {total_time/results['iterations']:.3f} seconds")
    
    print(f"\nFinal parameters:")
    for param, value in results['final_params'].items():
        if param == 'F0':
            print(f"  {param} = {value:.20f} Hz")
        else:
            print(f"  {param} = {value:.20e} Hz/s")
    
    print(f"\nFinal RMS: {results['final_rms']:.6f} μs")
    
    print(f"\nUncertainties:")
    for i, param in enumerate(['F0', 'F1']):
        unc = np.sqrt(results['covariance'][i, i])
        if param == 'F0':
            print(f"  {param}: {unc:.3e} Hz")
        else:
            print(f"  {param}: {unc:.3e} Hz/s")
    
    # Compare to expected
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nExpected (uncached): ~21 seconds")
    print(f"Actual (cached):     {total_time:.3f} seconds")
    
    if total_time < 21:
        speedup = 21.0 / total_time
        print(f"\n✅ SPEEDUP: {speedup:.2f}x faster!")
    else:
        print(f"\n⚠️  No speedup yet (still implementing optimizations)")
    
    return results


if __name__ == '__main__':
    try:
        results = test_cached_fitting()
        print(f"\n{'='*80}")
        print("TEST COMPLETE")
        print(f"{'='*80}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
