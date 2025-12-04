#!/usr/bin/env python3
"""
Level 1 Optimized Fitting - Cache dt_sec for 4x speedup
========================================================

This implements the simplest, safest optimization: compute expensive
delays (clock, bary, binary, DM) ONCE and reuse them for all iterations.

Expected speedup: 21s → 5.5s (4x faster)
"""

import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.utils.constants import SECS_PER_DAY


def compute_residuals_from_cached_dt_sec(
    dt_sec: np.ndarray,
    toas_mjd: np.ndarray,
    f0: float,
    f1: float,
    pepoch: float,
    weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Fast residual computation using pre-cached dt_sec.
    
    This matches the exact formula from simple_calculator.py line 494-628.
    
    Parameters
    ----------
    dt_sec : np.ndarray
        Time since PEPOCH in seconds, AFTER all delays removed.
        This is: TDB - PEPOCH - (clock + bary + binary + DM delays)
    toas_mjd : np.ndarray
        Original TOA times in MJD (not used, kept for interface compatibility)
    f0 : float
        Spin frequency (Hz)
    f1 : float
        Spin frequency derivative (Hz/s)
    pepoch : float
        Reference epoch (MJD) (not used, kept for interface compatibility)
    weights : np.ndarray
        TOA weights (1/error^2)
    
    Returns
    -------
    residuals_sec : np.ndarray
        Timing residuals in seconds
    rms_us : float
        Weighted RMS in microseconds
    """
    # Compute spin phase (EXACT same formula as simple_calculator.py line 494)
    f1_half = f1 / 2.0
    phase_cycles = dt_sec * (f0 + dt_sec * f1_half)
    
    # Wrap phase to nearest integer (line 624) - discard integer pulses
    phase_wrapped = phase_cycles - np.round(phase_cycles)
    
    # Convert phase to time residuals (line 628)
    # residuals = phase / f0 (in seconds)
    residuals_sec = phase_wrapped / f0
    
    # Subtract weighted mean (line 631-638)
    weighted_mean_sec = np.sum(residuals_sec * weights) / np.sum(weights)
    residuals_sec = residuals_sec - weighted_mean_sec
    
    # Compute weighted RMS (line 646)
    rms_sec = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights))
    rms_us = rms_sec * 1e6
    
    return residuals_sec, rms_us


def fit_f0_f1_optimized(
    par_file: Path,
    tim_file: Path,
    max_iter: int = 25,
    convergence_threshold: float = 1e-14
) -> dict:
    """
    Fit F0 and F1 using Level 1 optimization (cached dt_sec).
    
    This should give identical results to the original fitting code
    but run ~4x faster.
    
    Parameters
    ----------
    par_file : Path
        Path to .par file
    tim_file : Path
        Path to .tim file
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Convergence threshold on parameter changes
    
    Returns
    -------
    dict with keys:
        - iterations: number of iterations
        - total_time: total wall-clock time
        - cache_time: time to initialize cache
        - avg_iter_time: average iteration time
        - final_f0: fitted F0
        - final_f1: fitted F1
        - final_rms: final RMS in microseconds
        - converged: whether fitting converged
    """
    print("="*80)
    print("LEVEL 1 OPTIMIZED FITTING (Cached dt_sec)")
    print("="*80)
    
    total_start = time.time()
    
    # Parse files
    params = parse_par_file(par_file)
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract TOA data
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    
    pepoch = params['PEPOCH']
    f0_start = params['F0']
    f1_start = params['F1']
    
    print(f"\nStarting parameters:")
    print(f"  F0 = {f0_start:.20f} Hz")
    print(f"  F1 = {f1_start:.20e} Hz/s")
    print(f"  TOAs: {len(toas_mjd)}")
    
    # LEVEL 1 OPTIMIZATION: Compute dt_sec ONCE (expensive!)
    print(f"\nInitializing cache (computing dt_sec once)...")
    cache_start = time.time()
    
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = compute_residuals_simple(
            par_file,
            tim_file,
            clock_dir="data/clock",
            subtract_tzr=False  # DON'T wrap yet - we'll wrap with current F0/F1 each iteration
        )
    
    # This is the key: dt_sec includes ALL delays!
    dt_sec_cached = result['dt_sec']
    
    cache_time = time.time() - cache_start
    print(f"  Cache initialized in {cache_time:.3f}s")
    print(f"  dt_sec cached for {len(dt_sec_cached)} TOAs")
    print(f"  Ready for fast iterations!")
    
    # Fitting loop
    print(f"\nFitting F0 + F1 (using cached dt_sec)...")
    f0_curr = f0_start
    f1_curr = f1_start
    prev_delta_max = None
    iteration_times = []
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # FAST: Compute residuals from cached dt_sec
        residuals_sec, rms_us = compute_residuals_from_cached_dt_sec(
            dt_sec_cached,
            toas_mjd,
            f0_curr,
            f1_curr,
            pepoch,
            weights
        )
        
        # Compute derivatives (same as before)
        params_current = params.copy()
        params_current['F0'] = f0_curr
        params_current['F1'] = f1_curr
        
        derivs = compute_spin_derivatives(params_current, toas_mjd, ['F0', 'F1'])
        M = np.column_stack([derivs['F0'], derivs['F1']])
        
        # WLS solve (same as before)
        delta_params, cov, _ = wls_solve_svd(residuals_sec, errors_sec, M)
        
        # Update parameters (same as before)
        f0_curr += delta_params[0]
        f1_curr += delta_params[1]
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence (same as before)
        max_delta = max(abs(delta_params[0]), abs(delta_params[1]))
        
        if iteration < 3 or iteration >= max_iter - 1:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s")
        elif iteration == 3:
            print(f"  ...")
        
        if prev_delta_max is not None and abs(max_delta - prev_delta_max) < 1e-20:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        if max_delta < convergence_threshold:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        prev_delta_max = max_delta
    else:
        converged = False
        iterations = max_iter
    
    total_time = time.time() - total_start
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    print(f"\nConvergence:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    
    print(f"\nTiming:")
    print(f"  Cache initialization: {cache_time:.3f}s (one-time cost)")
    print(f"  Fitting iterations: {sum(iteration_times):.3f}s")
    print(f"  Average per iteration: {np.mean(iteration_times):.3f}s")
    print(f"  Total time: {total_time:.3f}s")
    
    print(f"\nFinal parameters:")
    print(f"  F0 = {f0_curr:.20f} Hz")
    print(f"  F1 = {f1_curr:.20e} Hz/s")
    print(f"  RMS = {rms_us:.6f} μs")
    
    print(f"\nUncertainties:")
    unc_f0 = np.sqrt(cov[0, 0])
    unc_f1 = np.sqrt(cov[1, 1])
    print(f"  σ(F0) = {unc_f0:.3e} Hz")
    print(f"  σ(F1) = {unc_f1:.3e} Hz/s")
    
    return {
        'iterations': iterations,
        'converged': converged,
        'total_time': total_time,
        'cache_time': cache_time,
        'avg_iter_time': np.mean(iteration_times),
        'final_f0': f0_curr,
        'final_f1': f1_curr,
        'final_rms': rms_us,
        'covariance': cov
    }


if __name__ == '__main__':
    par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    # Run optimized fitting
    results = fit_f0_f1_optimized(par_file, tim_file)
    
    # Compare to baseline
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    baseline_time = 21.15  # From previous benchmark
    speedup = baseline_time / results['total_time']
    
    print(f"\nBaseline (unoptimized): {baseline_time:.3f}s")
    print(f"Level 1 (cached dt_sec): {results['total_time']:.3f}s")
    print(f"\nSPEEDUP: {speedup:.2f}x faster! 🚀")
    
    if speedup >= 3.5:
        print("✅ SUCCESS: Achieved expected 4x speedup!")
    elif speedup >= 2.0:
        print("⚠️  Good speedup, but less than expected (target: 4x)")
    else:
        print("❌ Speedup less than expected - needs investigation")
