#!/usr/bin/env python3
"""
Profile JUG residual computation to find bottlenecks.
"""

import time
import cProfile
import pstats
from pathlib import Path
from jug.residuals.simple_calculator import compute_residuals_simple

def profile_residuals():
    """Profile the residual computation."""
    par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    print("Profiling JUG residual computation...")
    print("="*80)
    
    # Time with manual breakdown
    import time
    
    # Total time
    t_total = time.time()
    result = compute_residuals_simple(par_file, tim_file)
    elapsed_total = time.time() - t_total
    
    print(f"\nTotal time: {elapsed_total:.3f}s")
    print(f"RMS: {result['rms_us']:.6f} μs")
    
    # Now profile with cProfile
    print("\n" + "="*80)
    print("DETAILED PROFILING")
    print("="*80)
    
    profiler = cProfile.Profile()
    profiler.enable()
    result = compute_residuals_simple(par_file, tim_file)
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\nTop 30 functions by cumulative time:")
    print("-"*80)
    stats.print_stats(30)
    
    print("\n" + "="*80)
    print("Top 20 functions by internal time:")
    print("-"*80)
    stats.sort_stats('time')
    stats.print_stats(20)

if __name__ == '__main__':
    profile_residuals()
