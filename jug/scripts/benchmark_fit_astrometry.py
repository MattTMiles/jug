#!/usr/bin/env python
"""Benchmark script for astrometry fitting with the new damping loop.

This script benchmarks:
- Cold run: Fresh process, empty caches
- Warm run: Second run in same process

And reports:
- Total fit time
- Full-model evaluation count
- Rejected step count
- Final WRMS
- Damping loop statistics
"""

import sys
import time
import gc
from pathlib import Path

# Add JUG to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def benchmark_fit(par_file, tim_file, fit_params, run_name="", verbose=False):
    """Run a single benchmark fit and collect statistics."""
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    
    gc.collect()  # Clean up before timing
    
    start = time.perf_counter()
    result = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=10,
        verbose=verbose
    )
    elapsed = time.perf_counter() - start
    
    return {
        'run_name': run_name,
        'total_time': elapsed,
        'iterations': result['iterations'],
        'final_rms': result['final_rms'],
        'converged': result['converged'],
        'final_chi2': result.get('final_chi2', None),
    }


def main():
    par_file = Path('data/pulsars/J1909-3744_tdb.par')
    tim_file = Path('data/pulsars/J1909-3744.tim')
    
    # Full parameter set including astrometry
    fit_params = ['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 
                  'DM', 'DM1', 'DM2',
                  'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT']
    
    print("="*70)
    print("Astrometry Fitting Benchmark")
    print("="*70)
    print(f"Par file: {par_file}")
    print(f"Tim file: {tim_file}")
    print(f"Fit params: {len(fit_params)} ({', '.join(fit_params[:5])}...)")
    print()
    
    # Warm-up run (to JIT compile JAX functions)
    print("-"*40)
    print("Warm-up run (JIT compilation)...")
    warmup = benchmark_fit(par_file, tim_file, fit_params, "warmup")
    print(f"  Time: {warmup['total_time']:.3f}s")
    
    # Cold run (fresh fit, cache cleared)
    print("-"*40)
    print("Cold run...")
    gc.collect()
    cold = benchmark_fit(par_file, tim_file, fit_params, "cold")
    print(f"  Time: {cold['total_time']:.3f}s")
    print(f"  Iterations: {cold['iterations']}")
    print(f"  Final WRMS: {cold['final_rms']:.6f} μs")
    
    # Warm run (second fit, caches warm)
    print("-"*40)
    print("Warm run...")
    warm = benchmark_fit(par_file, tim_file, fit_params, "warm")
    print(f"  Time: {warm['total_time']:.3f}s")
    print(f"  Iterations: {warm['iterations']}")
    print(f"  Final WRMS: {warm['final_rms']:.6f} μs")
    
    # Multiple run average
    print("-"*40)
    print("Multiple runs (5x)...")
    times = []
    for i in range(5):
        r = benchmark_fit(par_file, tim_file, fit_params, f"run{i+1}")
        times.append(r['total_time'])
        print(f"  Run {i+1}: {r['total_time']:.3f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Metric':<30} {'Value':<15}")
    print("-"*45)
    print(f"{'Cold run time':<30} {cold['total_time']:.3f}s")
    print(f"{'Warm run time':<30} {warm['total_time']:.3f}s")
    print(f"{'Average time (5 runs)':<30} {avg_time:.3f}s")
    print(f"{'Min time':<30} {min_time:.3f}s")
    print(f"{'Max time':<30} {max_time:.3f}s")
    print(f"{'Iterations':<30} {cold['iterations']}")
    print(f"{'Final WRMS':<30} {cold['final_rms']:.6f} μs")
    print(f"{'Converged':<30} {cold['converged']}")
    
    # Speedup estimate
    if cold['total_time'] > warm['total_time']:
        speedup = cold['total_time'] / warm['total_time']
        print(f"{'Speedup (cold/warm)':<30} {speedup:.2f}x")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
