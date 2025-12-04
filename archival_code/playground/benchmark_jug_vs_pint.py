#!/usr/bin/env python3
"""
Comprehensive Benchmark: JUG vs PINT vs Tempo2
===============================================

Compares JUG, PINT, and Tempo2 timing on J1909-3744 with detailed breakdown
of where each tool spends its time.
"""

import time
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import os

# JUG imports
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized

# PINT imports
import pint.models
import pint.toa
import pint.fitter
import pint.residuals


def benchmark_jug_detailed(par_file, tim_file):
    """Benchmark JUG with detailed timing breakdown."""
    
    print("\n" + "="*80)
    print("JUG DETAILED BENCHMARK")
    print("="*80)
    
    times = {}
    
    # 1. File parsing
    t0 = time.time()
    params = parse_par_file(par_file)
    toas_data = parse_tim_file_mjds(tim_file)
    times['parse_files'] = time.time() - t0
    print(f"1. Parse files: {times['parse_files']:.3f}s")
    
    # 2. Residual computation (includes all delay calculations)
    print("\n2. Computing residuals (with detailed breakdown)...")
    t0 = time.time()
    result = compute_residuals_simple(par_file, tim_file, verbose=False)
    times['compute_residuals'] = time.time() - t0
    print(f"   Total residual computation: {times['compute_residuals']:.3f}s")
    
    # Get breakdown from result if available
    if 'timing_breakdown' in result:
        for key, val in result['timing_breakdown'].items():
            print(f"     - {key}: {val:.3f}s")
    
    # 3. Fitting (F0 and F1)
    print("\n3. Fitting F0 + F1...")
    t0 = time.time()
    fit_result = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=['F0', 'F1'],
        max_iter=15,
        verbose=False,
        device='cpu'
    )
    times['fitting'] = time.time() - t0
    
    # Breakdown of fitting
    times['cache_init'] = fit_result.get('cache_time', 0)
    times['jit_compile'] = fit_result.get('jit_time', 0)
    times['iterations'] = times['fitting'] - times['cache_init'] - times['jit_compile']
    
    print(f"   Cache initialization: {times['cache_init']:.3f}s")
    print(f"   JIT compilation: {times['jit_compile']:.3f}s")
    print(f"   Fitting iterations ({fit_result['iterations']} iters): {times['iterations']:.3f}s")
    print(f"   Total fitting: {times['fitting']:.3f}s")
    
    # 4. Total
    times['total'] = times['parse_files'] + times['compute_residuals'] + times['fitting']
    
    print(f"\n{'='*80}")
    print(f"JUG TOTAL TIME: {times['total']:.3f}s")
    print(f"{'='*80}")
    
    print(f"\nFit Results:")
    print(f"  Prefit RMS: {fit_result['prefit_rms']:.6f} μs")
    print(f"  Postfit RMS: {fit_result['final_rms']:.6f} μs")
    print(f"  Iterations: {fit_result['iterations']}")
    print(f"  Converged: {fit_result['converged']}")
    print(f"  Final F0: {fit_result['final_params']['F0']:.20e} Hz")
    print(f"  Final F1: {fit_result['final_params']['F1']:.20e} Hz/s")
    
    return times, fit_result


def benchmark_pint(par_file, tim_file):
    """Benchmark PINT with timing breakdown."""
    
    print("\n" + "="*80)
    print("PINT BENCHMARK")
    print("="*80)
    
    times = {}
    
    # 1. Load model and TOAs
    t0 = time.time()
    model = pint.models.get_model(str(par_file))
    toas = pint.toa.get_TOAs(str(tim_file), planets=True, ephem='DE440')
    times['load_model_toas'] = time.time() - t0
    print(f"1. Load model + TOAs: {times['load_model_toas']:.3f}s")
    
    # 2. Prefit residuals
    t0 = time.time()
    prefit_resids = pint.residuals.Residuals(toas, model)
    times['prefit_residuals'] = time.time() - t0
    print(f"2. Compute prefit residuals: {times['prefit_residuals']:.3f}s")
    
    prefit_rms = prefit_resids.rms_weighted().to_value('us')
    print(f"   Prefit RMS: {prefit_rms:.6f} μs")
    
    # 3. Fitting
    print(f"\n3. Fitting F0 + F1...")
    t0 = time.time()
    fitter = pint.fitter.WLSFitter(toas, model)
    fitter.model.free_params = ['F0', 'F1']
    fitter.fit_toas(maxiter=15)
    times['fitting'] = time.time() - t0
    print(f"   Fitting: {times['fitting']:.3f}s")
    
    # 4. Postfit residuals
    t0 = time.time()
    postfit_resids = pint.residuals.Residuals(toas, fitter.model)
    times['postfit_residuals'] = time.time() - t0
    print(f"4. Compute postfit residuals: {times['postfit_residuals']:.3f}s")
    
    postfit_rms = postfit_resids.rms_weighted().to_value('us')
    
    # Total
    times['total'] = (times['load_model_toas'] + times['prefit_residuals'] + 
                     times['fitting'] + times['postfit_residuals'])
    
    print(f"\n{'='*80}")
    print(f"PINT TOTAL TIME: {times['total']:.3f}s")
    print(f"{'='*80}")
    
    print(f"\nFit Results:")
    print(f"  Prefit RMS: {prefit_rms:.6f} μs")
    print(f"  Postfit RMS: {postfit_rms:.6f} μs")
    print(f"  Iterations: {fitter.fit_toas.niter if hasattr(fitter.fit_toas, 'niter') else 'N/A'}")
    print(f"  Final F0: {fitter.model.F0.value:.20e} Hz")
    print(f"  Final F1: {fitter.model.F1.value:.20e} Hz/s")
    
    return times, fitter


def benchmark_tempo2(par_file, tim_file):
    """Benchmark Tempo2."""
    
    print("\n" + "="*80)
    print("TEMPO2 BENCHMARK")
    print("="*80)
    
    times = {}
    
    # Tempo2 command: tempo2 -f par_file tim_file -fit F0 -fit F1
    cmd = [
        'tempo2',
        '-f', str(par_file),
        str(tim_file),
        '-fit', 'F0',
        '-fit', 'F1'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Time the complete execution
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )
    times['total'] = time.time() - t0
    
    print(f"\nTempo2 execution: {times['total']:.3f}s")
    
    # Parse output to get fit results
    output = result.stdout
    
    # Extract RMS
    prefit_rms = None
    postfit_rms = None
    for line in output.split('\n'):
        if 'Pre-fit RMS' in line or 'Prefit RMS' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'RMS' in part and i+1 < len(parts):
                    try:
                        prefit_rms = float(parts[i+1])
                    except:
                        pass
        if 'Post-fit RMS' in line or 'Postfit RMS' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'RMS' in part and i+1 < len(parts):
                    try:
                        postfit_rms = float(parts[i+1])
                    except:
                        pass
    
    # Extract fitted parameters
    f0_fit = None
    f1_fit = None
    in_params = False
    for line in output.split('\n'):
        if 'FITTED' in line or 'PARAMETERS' in line:
            in_params = True
        if in_params:
            if 'F0' in line and f0_fit is None:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'F0' and i+1 < len(parts):
                        try:
                            f0_fit = float(parts[i+1].replace('D', 'E'))
                        except:
                            pass
            if 'F1' in line and f1_fit is None:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'F1' and i+1 < len(parts):
                        try:
                            f1_fit = float(parts[i+1].replace('D', 'E'))
                        except:
                            pass
    
    print(f"\n{'='*80}")
    print(f"TEMPO2 TOTAL TIME: {times['total']:.3f}s")
    print(f"{'='*80}")
    
    if prefit_rms:
        print(f"\nFit Results:")
        print(f"  Prefit RMS: {prefit_rms:.6f} μs")
        if postfit_rms:
            print(f"  Postfit RMS: {postfit_rms:.6f} μs")
        if f0_fit:
            print(f"  Final F0: {f0_fit:.20e} Hz")
        if f1_fit:
            print(f"  Final F1: {f1_fit:.20e} Hz/s")
    
    return times, {'prefit_rms': prefit_rms, 'postfit_rms': postfit_rms, 
                   'F0': f0_fit, 'F1': f1_fit, 'output': output}


def print_comparison(jug_times, pint_times, tempo2_times):
    """Print comparison table including Tempo2."""
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Component':<30} {'JUG (s)':<12} {'PINT (s)':<12} {'Tempo2 (s)':<12}")
    print("-" * 90)
    
    # File loading
    jug_parse = jug_times['parse_files']
    pint_load = pint_times['load_model_toas']
    print(f"{'File parsing/loading':<30} {jug_parse:<12.3f} {pint_load:<12.3f} {'(included)':<12}")
    
    # Residual computation
    jug_resid = jug_times['compute_residuals']
    pint_resid = pint_times['prefit_residuals'] + pint_times['postfit_residuals']
    print(f"{'Residual computation':<30} {jug_resid:<12.3f} {pint_resid:<12.3f} {'(included)':<12}")
    
    # Fitting
    jug_fit = jug_times['fitting']
    pint_fit = pint_times['fitting']
    print(f"{'Fitting':<30} {jug_fit:<12.3f} {pint_fit:<12.3f} {'(included)':<12}")
    
    print("-" * 90)
    
    # Total
    jug_total = jug_times['total']
    pint_total = pint_times['total']
    tempo2_total = tempo2_times['total']
    
    print(f"{'TOTAL':<30} {jug_total:<12.3f} {pint_total:<12.3f} {tempo2_total:<12.3f}")
    
    print("\n" + "="*80)
    print("SPEEDUP COMPARISON (vs Tempo2)")
    print("="*80)
    
    jug_speedup = tempo2_total / jug_total if jug_total > 0 else float('inf')
    pint_speedup = tempo2_total / pint_total if pint_total > 0 else float('inf')
    
    print(f"  Tempo2: 1.00x (baseline)")
    print(f"  JUG:    {jug_speedup:.2f}x {'faster' if jug_speedup > 1 else 'slower'}")
    print(f"  PINT:   {pint_speedup:.2f}x {'faster' if pint_speedup > 1 else 'slower'}")
    
    print("\n" + "="*80)
    print("SPEEDUP COMPARISON (JUG vs PINT)")
    print("="*80)
    
    jug_vs_pint = pint_total / jug_total if jug_total > 0 else float('inf')
    print(f"  JUG is {jug_vs_pint:.2f}x faster than PINT")
    
    print("\n" + "="*80)
    print("JUG FITTING BREAKDOWN")
    print("="*80)
    print(f"  Cache initialization: {jug_times['cache_init']:.3f}s ({100*jug_times['cache_init']/jug_times['fitting']:.1f}%)")
    print(f"  JIT compilation:      {jug_times['jit_compile']:.3f}s ({100*jug_times['jit_compile']/jug_times['fitting']:.1f}%)")
    print(f"  Iterations:           {jug_times['iterations']:.3f}s ({100*jug_times['iterations']/jug_times['fitting']:.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall rankings:")
    
    results = [
        ('JUG', jug_total),
        ('Tempo2', tempo2_total),
        ('PINT', pint_total)
    ]
    results.sort(key=lambda x: x[1])
    
    print(f"  1. {results[0][0]}: {results[0][1]:.3f}s (fastest)")
    print(f"  2. {results[1][0]}: {results[1][1]:.3f}s ({results[1][1]/results[0][1]:.2f}x slower)")
    print(f"  3. {results[2][0]}: {results[2][1]:.3f}s ({results[2][1]/results[0][1]:.2f}x slower)")


def print_comparison_old(jug_times, pint_times):
    """Print comparison table."""
    
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    print(f"\n{'Component':<30} {'JUG (s)':<12} {'PINT (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    # File loading
    jug_parse = jug_times['parse_files']
    pint_load = pint_times['load_model_toas']
    speedup = pint_load / jug_parse if jug_parse > 0 else float('inf')
    print(f"{'File parsing/loading':<30} {jug_parse:<12.3f} {pint_load:<12.3f} {speedup:<10.2f}x")
    
    # Residual computation
    jug_resid = jug_times['compute_residuals']
    pint_resid = pint_times['prefit_residuals'] + pint_times['postfit_residuals']
    speedup = pint_resid / jug_resid if jug_resid > 0 else float('inf')
    print(f"{'Residual computation':<30} {jug_resid:<12.3f} {pint_resid:<12.3f} {speedup:<10.2f}x")
    
    # Fitting
    jug_fit = jug_times['fitting']
    pint_fit = pint_times['fitting']
    speedup = pint_fit / jug_fit if jug_fit > 0 else float('inf')
    print(f"{'Fitting':<30} {jug_fit:<12.3f} {pint_fit:<12.3f} {speedup:<10.2f}x")
    
    print("-" * 80)
    
    # Total
    jug_total = jug_times['total']
    pint_total = pint_times['total']
    speedup = pint_total / jug_total if jug_total > 0 else float('inf')
    print(f"{'TOTAL':<30} {jug_total:<12.3f} {pint_total:<12.3f} {speedup:<10.2f}x")
    
    print("\n" + "="*80)
    print("JUG FITTING BREAKDOWN")
    print("="*80)
    print(f"  Cache initialization: {jug_times['cache_init']:.3f}s ({100*jug_times['cache_init']/jug_times['fitting']:.1f}%)")
    print(f"  JIT compilation:      {jug_times['jit_compile']:.3f}s ({100*jug_times['jit_compile']/jug_times['fitting']:.1f}%)")
    print(f"  Iterations:           {jug_times['iterations']:.3f}s ({100*jug_times['iterations']/jug_times['fitting']:.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Overall speedup: {speedup:.2f}x")
    print(f"JUG is {speedup:.2f}x faster than PINT for this dataset")
    if speedup > 1:
        time_saved = pint_total - jug_total
        print(f"Time saved: {time_saved:.3f}s ({100*(1-1/speedup):.1f}% faster)")


def main():
    """Run comprehensive benchmark."""
    
    par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    if not par_file.exists():
        print(f"Error: Par file not found: {par_file}")
        return 1
    
    if not tim_file.exists():
        print(f"Error: Tim file not found: {tim_file}")
        return 1
    
    print("="*80)
    print("COMPREHENSIVE BENCHMARK: JUG vs PINT vs Tempo2")
    print("="*80)
    print(f"Dataset: {par_file.name}")
    print(f"TOAs: {tim_file.name}")
    print(f"Task: Compute residuals + Fit F0 and F1")
    
    # Run JUG benchmark
    jug_times, jug_result = benchmark_jug_detailed(par_file, tim_file)
    
    # Run PINT benchmark
    pint_times, pint_result = benchmark_pint(par_file, tim_file)
    
    # Run Tempo2 benchmark
    tempo2_times, tempo2_result = benchmark_tempo2(par_file, tim_file)
    
    # Print comparison
    print_comparison(jug_times, pint_times, tempo2_times)
    
    # Verify results match
    print("\n" + "="*80)
    print("RESULT VERIFICATION")
    print("="*80)
    
    # JUG vs PINT
    f0_diff_pint = abs(jug_result['final_params']['F0'] - pint_result.model.F0.value)
    f1_diff_pint = abs(jug_result['final_params']['F1'] - pint_result.model.F1.value)
    
    print(f"\nJUG vs PINT:")
    print(f"  F0 difference: {f0_diff_pint:.2e} Hz")
    print(f"  F1 difference: {f1_diff_pint:.2e} Hz/s")
    
    if f0_diff_pint < 1e-10 and f1_diff_pint < 1e-20:
        print("  ✅ Results match to high precision!")
    else:
        print("  ⚠️  Results differ slightly")
    
    # JUG vs Tempo2
    if tempo2_result['F0'] and tempo2_result['F1']:
        f0_diff_t2 = abs(jug_result['final_params']['F0'] - tempo2_result['F0'])
        f1_diff_t2 = abs(jug_result['final_params']['F1'] - tempo2_result['F1'])
        
        print(f"\nJUG vs Tempo2:")
        print(f"  F0 difference: {f0_diff_t2:.2e} Hz")
        print(f"  F1 difference: {f1_diff_t2:.2e} Hz/s")
        
        if f0_diff_t2 < 1e-10 and f1_diff_t2 < 1e-20:
            print("  ✅ Results match to high precision!")
        else:
            print("  ⚠️  Results differ slightly")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
