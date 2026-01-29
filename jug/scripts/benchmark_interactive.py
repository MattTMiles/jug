#!/usr/bin/env python3
"""
Interactive Workflow Benchmark
==============================

Measures performance for the typical GUI/interactive workflow:
1. Load data (cold start)
2. Compute residuals
3. Fit parameters
4. Re-fit with different parameters (should be fast!)

This simulates what happens in the GUI when a user loads data and
performs multiple fits.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


def run_interactive_benchmark(
    par_file: Path,
    tim_file: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run interactive workflow benchmark."""
    
    results = {}
    
    if verbose:
        print("=" * 70)
        print("INTERACTIVE WORKFLOW BENCHMARK")
        print("=" * 70)
        print(f"Par: {par_file}")
        print(f"Tim: {tim_file}")
    
    # =========================================================================
    # PHASE 1: Cold Start
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("PHASE 1: Cold Start (first time loading data)")
        print("-" * 70)
    
    # Import timing (cold)
    t0 = time.perf_counter()
    from jug.engine.session import TimingSession
    import_time = time.perf_counter() - t0
    results['import_time_ms'] = import_time * 1000
    if verbose:
        print(f"  Import TimingSession: {import_time*1000:.1f} ms")
    
    # Create session
    t0 = time.perf_counter()
    session = TimingSession(par_file, tim_file, verbose=False)
    session_time = time.perf_counter() - t0
    results['session_creation_ms'] = session_time * 1000
    if verbose:
        print(f"  Create session: {session_time*1000:.1f} ms")
    
    # First residual computation (cold - includes all JIT)
    t0 = time.perf_counter()
    result = session.compute_residuals(subtract_tzr=True)
    first_residuals_time = time.perf_counter() - t0
    results['first_residuals_ms'] = first_residuals_time * 1000
    if verbose:
        print(f"  First residuals: {first_residuals_time*1000:.1f} ms")
        print(f"    RMS: {result['rms_us']:.3f} μs")
    
    # First fit (cold - includes solver JIT)
    t0 = time.perf_counter()
    fit1 = session.fit_parameters(['F0', 'F1'], verbose=False)
    first_fit_time = time.perf_counter() - t0
    results['first_fit_ms'] = first_fit_time * 1000
    if verbose:
        print(f"  First fit: {first_fit_time*1000:.1f} ms")
        print(f"    Iterations: {fit1['iterations']}")
        print(f"    Final RMS: {fit1['final_rms']:.3f} μs")
    
    cold_total = import_time + session_time + first_residuals_time + first_fit_time
    results['cold_total_ms'] = cold_total * 1000
    
    # =========================================================================
    # PHASE 2: Interactive Use (after initial load)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("PHASE 2: Interactive Use (GUI-like repeated operations)")
        print("-" * 70)
    
    # Re-compute residuals (should use cache)
    t0 = time.perf_counter()
    result2 = session.compute_residuals(subtract_tzr=True)
    cached_residuals_time = time.perf_counter() - t0
    results['cached_residuals_ms'] = cached_residuals_time * 1000
    if verbose:
        print(f"  Cached residuals: {cached_residuals_time*1000:.1f} ms")
    
    # Re-fit (cache was invalidated by first fit, so need to recompute)
    t0 = time.perf_counter()
    _ = session.compute_residuals(subtract_tzr=False, force_recompute=True)
    recompute_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    fit2 = session.fit_parameters(['F0', 'F1'], verbose=False)
    second_fit_time = time.perf_counter() - t0
    results['second_fit_ms'] = (recompute_time + second_fit_time) * 1000
    if verbose:
        print(f"  Second fit (with recompute): {(recompute_time + second_fit_time)*1000:.1f} ms")
        print(f"    - Recompute residuals: {recompute_time*1000:.1f} ms")
        print(f"    - Fit: {second_fit_time*1000:.1f} ms")
    
    # Third fit (immediately after, should be fastest)
    # First, repopulate cache
    _ = session.compute_residuals(subtract_tzr=False, force_recompute=True)
    
    t0 = time.perf_counter()
    fit3 = session.fit_parameters(['F0', 'F1'], verbose=False)
    third_fit_time = time.perf_counter() - t0
    results['third_fit_ms'] = third_fit_time * 1000
    if verbose:
        print(f"  Third fit (optimal cache): {third_fit_time*1000:.1f} ms")
    
    # Different parameters fit
    _ = session.compute_residuals(subtract_tzr=False, force_recompute=True)
    t0 = time.perf_counter()
    fit4 = session.fit_parameters(['F0', 'F1', 'DM'], verbose=False)
    dm_fit_time = time.perf_counter() - t0
    results['dm_fit_ms'] = dm_fit_time * 1000
    if verbose:
        print(f"  Fit with DM: {dm_fit_time*1000:.1f} ms")
    
    # =========================================================================
    # PHASE 3: Fresh Session (simulates re-opening file)
    # =========================================================================
    if verbose:
        print("\n" + "-" * 70)
        print("PHASE 3: Warm JIT (new session, same process)")
        print("-" * 70)
    
    # Create new session
    t0 = time.perf_counter()
    session2 = TimingSession(par_file, tim_file, verbose=False)
    warm_session_time = time.perf_counter() - t0
    results['warm_session_ms'] = warm_session_time * 1000
    if verbose:
        print(f"  Create session: {warm_session_time*1000:.1f} ms")
    
    # Residuals (JIT is warm but Astropy needs recompute)
    t0 = time.perf_counter()
    result3 = session2.compute_residuals(subtract_tzr=False)
    warm_residuals_time = time.perf_counter() - t0
    results['warm_residuals_ms'] = warm_residuals_time * 1000
    if verbose:
        print(f"  Residuals: {warm_residuals_time*1000:.1f} ms")
    
    # Fit (JIT is warm)
    t0 = time.perf_counter()
    fit5 = session2.fit_parameters(['F0', 'F1'], verbose=False)
    warm_fit_time = time.perf_counter() - t0
    results['warm_fit_ms'] = warm_fit_time * 1000
    if verbose:
        print(f"  Fit: {warm_fit_time*1000:.1f} ms")
    
    warm_total = warm_session_time + warm_residuals_time + warm_fit_time
    results['warm_total_ms'] = warm_total * 1000
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<40} {'Time (ms)':>10}")
        print("-" * 52)
        print(f"{'Cold start total':<40} {cold_total*1000:>10.1f}")
        print(f"{'  - Import':<40} {import_time*1000:>10.1f}")
        print(f"{'  - Session creation':<40} {session_time*1000:>10.1f}")
        print(f"{'  - First residuals':<40} {first_residuals_time*1000:>10.1f}")
        print(f"{'  - First fit':<40} {first_fit_time*1000:>10.1f}")
        print()
        print(f"{'Warm interactive fit':<40} {third_fit_time*1000:>10.1f}")
        print(f"{'Warm session total':<40} {warm_total*1000:>10.1f}")
        print()
        print(f"{'Speedup (cold→warm fit)':<40} {first_fit_time/third_fit_time:>10.1f}x")
        print(f"{'Speedup (cold→warm session)':<40} {cold_total/warm_total:>10.1f}x")
        print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Workflow Benchmark'
    )
    parser.add_argument(
        '--par',
        type=Path,
        default=Path("data/pulsars/J1909-3744_tdb_wrong.par"),
        help='Path to .par file'
    )
    parser.add_argument(
        '--tim',
        type=Path,
        default=Path("data/pulsars/J1909-3744.tim"),
        help='Path to .tim file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    if not args.par.exists():
        print(f"ERROR: Par file not found: {args.par}")
        sys.exit(1)
    if not args.tim.exists():
        print(f"ERROR: Tim file not found: {args.tim}")
        sys.exit(1)
    
    results = run_interactive_benchmark(args.par, args.tim)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
