#!/usr/bin/env python3
"""
Complete Benchmark: Tempo2 vs PINT vs JUG
==========================================

Compares all three methods for F0+F1 fitting with:
- Prefit and postfit residual plots
- Speed measurements
- Weighted RMS comparisons
- Parameter accuracy

Based on Session 13-14 validated JUG fitting implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import tempfile
import os
import shutil
import sys
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
PAR_FILE = 'data/pulsars/J1909-3744_tdb_wrong.par'
TIM_FILE = 'data/pulsars/J1909-3744.tim'

# Check if files exist, otherwise try alternate location
if not os.path.exists(TIM_FILE):
    TIM_FILE = '/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim'

print("="*80)
print("PULSAR TIMING BENCHMARK: Tempo2 vs PINT vs JUG")
print("="*80)
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")
print()

# ============================================================================
# TEMPO2
# ============================================================================
print("\n" + "="*80)
print("TEMPO2")
print("="*80)

t2_prefit_res = None
t2_postfit_res = None
t2_errors = None
t2_prefit_wrms = None
t2_postfit_wrms = None
t2_f0 = None
t2_f1 = None
t2_time = None

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_par = os.path.join(tmpdir, 'pulsar.par')
        tmp_tim = os.path.join(tmpdir, 'pulsar.tim')
        shutil.copy(PAR_FILE, tmp_par)
        shutil.copy(TIM_FILE, tmp_tim)
        
        # Prefit residuals
        print("Computing prefit residuals...")
        result = subprocess.run(
            ['tempo2', '-f', tmp_par, tmp_tim, '-nofit', '-output', 'general2'],
            cwd=tmpdir, capture_output=True, text=True, timeout=60
        )
        
        # Find general2 output file
        gen2_files = [f for f in os.listdir(tmpdir) if 'general2' in f.lower()]
        if gen2_files:
            gen2_file = os.path.join(tmpdir, gen2_files[0])
            data_pre = np.loadtxt(gen2_file, usecols=(2, 3))
            t2_prefit_res = data_pre[:, 0]
            t2_errors = data_pre[:, 1]
            t2_prefit_wrms = np.sqrt(np.average(t2_prefit_res**2, weights=1.0/t2_errors**2))
            print(f"✓ Prefit WRMS: {t2_prefit_wrms:.6f} μs")
        
        # Fit F0 and F1
        print("Fitting F0 and F1...")
        start = time.time()
        result = subprocess.run(
            ['tempo2', '-f', tmp_par, tmp_tim, '-fit', 'F0,F1', '-output', 'general2'],
            cwd=tmpdir, capture_output=True, text=True, timeout=60
        )
        t2_time = time.time() - start
        
        # Postfit residuals
        gen2_files = [f for f in os.listdir(tmpdir) if 'general2' in f.lower()]
        if gen2_files:
            gen2_file = os.path.join(tmpdir, gen2_files[0])
            data_post = np.loadtxt(gen2_file, usecols=(2, 3))
            t2_postfit_res = data_post[:, 0]
            if t2_errors is None:
                t2_errors = data_post[:, 1]
            t2_postfit_wrms = np.sqrt(np.average(t2_postfit_res**2, weights=1.0/t2_errors**2))
            print(f"✓ Postfit WRMS: {t2_postfit_wrms:.6f} μs")
        
        # Get fitted values from new.par
        new_par = os.path.join(tmpdir, 'new.par')
        if os.path.exists(new_par):
            with open(new_par) as f:
                for line in f:
                    if line.strip().startswith('F0 '):
                        parts = line.split()
                        t2_f0 = float(parts[1])
                    elif line.strip().startswith('F1 '):
                        parts = line.split()
                        t2_f1 = float(parts[1])
        
        print(f"✓ Time: {t2_time:.3f}s")
        if t2_f0:
            print(f"✓ F0 = {t2_f0:.20f} Hz")
        if t2_f1:
            print(f"✓ F1 = {t2_f1:.20e} Hz/s")
            
except Exception as e:
    print(f"✗ Tempo2 failed: {e}")
    print("  (Continuing with PINT and JUG...)")

# ============================================================================
# PINT
# ============================================================================
print("\n" + "="*80)
print("PINT")
print("="*80)

try:
    import pint.models as pm
    import pint.toa as pt
    import pint.fitter as pf
    import pint.residuals
    
    start = time.time()
    
    # Load model and TOAs
    print("Loading model and TOAs...")
    model = pm.get_model(PAR_FILE)
    toas = pt.get_TOAs(TIM_FILE, model=model)
    
    # Prefit residuals
    print("Computing prefit residuals...")
    prefit = pint.residuals.Residuals(toas, model)
    pint_prefit_res = prefit.time_resids.to_value('us')
    pint_errors = toas.get_errors().to_value('us')
    pint_prefit_wrms = np.sqrt(np.average(pint_prefit_res**2, weights=1.0/pint_errors**2))
    print(f"✓ Prefit WRMS: {pint_prefit_wrms:.6f} μs")
    
    # Fit F0 and F1
    print("Fitting F0 and F1...")
    model.F0.frozen = False
    model.F1.frozen = False
    fitter = pf.WLSFitter(toas, model)
    fitter.fit_toas()
    
    # Postfit residuals
    pint_postfit_res = fitter.resids.time_resids.to_value('us')
    pint_postfit_wrms = np.sqrt(np.average(pint_postfit_res**2, weights=1.0/pint_errors**2))
    
    pint_time = time.time() - start
    pint_f0 = fitter.model.F0.quantity.value
    pint_f1 = fitter.model.F1.quantity.value
    pint_iters = len(fitter.chi2_history) if hasattr(fitter, 'chi2_history') else 'N/A'
    
    print(f"✓ Postfit WRMS: {pint_postfit_wrms:.6f} μs")
    print(f"✓ Time: {pint_time:.3f}s")
    print(f"✓ F0 = {pint_f0:.20f} Hz")
    print(f"✓ F1 = {pint_f1:.20e} Hz/s")
    print(f"✓ Iterations: {pint_iters}")
    
except Exception as e:
    print(f"✗ PINT failed: {e}")
    pint_prefit_res = None
    pint_postfit_res = None
    pint_errors = None
    pint_prefit_wrms = None
    pint_postfit_wrms = None
    pint_f0 = None
    pint_f1 = None
    pint_time = None
    pint_iters = None

# ============================================================================
# JUG
# ============================================================================
print("\n" + "="*80)
print("JUG (Session 13-14 Validated Implementation)")
print("="*80)

try:
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.fitting import fit_parameters_optimized
    from pathlib import Path
    
    start = time.time()
    
    # Prefit residuals
    print("Computing prefit residuals...")
    prefit_result = compute_residuals_simple(PAR_FILE, TIM_FILE)
    jug_prefit_res = prefit_result['residuals_us']
    jug_errors = prefit_result['errors_us']
    jug_prefit_wrms = np.sqrt(np.average(jug_prefit_res**2, weights=1.0/jug_errors**2))
    print(f"✓ Prefit WRMS: {jug_prefit_wrms:.6f} μs")
    
    # Fit F0 and F1 using optimized fitter
    print("Fitting F0 and F1...")
    fit_result = fit_parameters_optimized(
        par_file=Path(PAR_FILE),
        tim_file=Path(TIM_FILE),
        fit_params=['F0', 'F1'],
        max_iter=20,
        convergence_threshold=1e-13
    )
    
    # Get postfit residuals
    postfit_result = compute_residuals_simple(PAR_FILE, TIM_FILE)
    # But we need to update the par file with fitted params first
    # For now, use the RMS from the fitter
    
    jug_postfit_wrms = fit_result['final_rms']
    jug_time = time.time() - start
    jug_f0 = fit_result['final_params']['F0']
    jug_f1 = fit_result['final_params']['F1']
    jug_iters = fit_result['iterations']
    
    # Get initial values for comparison
    from jug.io.par_reader import parse_par_file
    params_init = parse_par_file(PAR_FILE)
    f0_init = params_init['F0']
    f1_init = params_init['F1']
    
    # Compute postfit residuals by writing temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
        with open(PAR_FILE) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    if parts[0] == 'F0':
                        tmp.write(f"F0 {jug_f0:.20e}\n")
                    elif parts[0] == 'F1':
                        tmp.write(f"F1 {jug_f1:.20e}\n")
                    else:
                        tmp.write(line)
                else:
                    tmp.write(line)
        tmp_par = tmp.name
    
    try:
        postfit_result = compute_residuals_simple(tmp_par, TIM_FILE)
        jug_postfit_res = postfit_result['residuals_us']
    finally:
        os.unlink(tmp_par)
    
    print(f"✓ Postfit WRMS: {jug_postfit_wrms:.6f} μs")
    print(f"✓ Time: {jug_time:.3f}s")
    print(f"✓ F0 = {jug_f0:.20f} Hz")
    print(f"✓ F1 = {jug_f1:.20e} Hz/s")
    print(f"✓ Iterations: {jug_iters}")
    print(f"✓ ΔF0 = {float(jug_f0) - float(f0_init):.15e} Hz")
    print(f"✓ ΔF1 = {float(jug_f1) - float(f1_init):.15e} Hz/s")
    
except Exception as e:
    print(f"✗ JUG failed: {e}")
    import traceback
    traceback.print_exc()
    jug_prefit_res = None
    jug_postfit_res = None
    jug_errors = None
    jug_prefit_wrms = None
    jug_postfit_wrms = None
    jug_f0 = None
    jug_f1 = None
    jug_time = None
    jug_iters = None

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Create summary table
print(f"\n{'Method':<10} {'Prefit (μs)':<15} {'Postfit (μs)':<15} {'Time (s)':<12} {'Iters':<8}")
print("-"*80)

if t2_postfit_wrms is not None:
    pre_str = f"{t2_prefit_wrms:.6f}" if t2_prefit_wrms else "N/A"
    post_str = f"{t2_postfit_wrms:.6f}"
    time_str = f"{t2_time:.3f}" if t2_time else "N/A"
    print(f"{'Tempo2':<10} {pre_str:<15} {post_str:<15} {time_str:<12} {'N/A':<8}")

if pint_postfit_wrms is not None:
    print(f"{'PINT':<10} {pint_prefit_wrms:<15.6f} {pint_postfit_wrms:<15.6f} {pint_time:<12.3f} {str(pint_iters):<8}")

if jug_postfit_wrms is not None:
    print(f"{'JUG':<10} {jug_prefit_wrms:<15.6f} {jug_postfit_wrms:<15.6f} {jug_time:<12.3f} {jug_iters:<8}")

# Parameter comparison
print(f"\n{'Method':<10} {'F0 (Hz)':<30} {'F1 (Hz/s)':<25}")
print("-"*80)

if t2_f0 is not None:
    print(f"{'Tempo2':<10} {t2_f0:.20f}  {t2_f1:.15e}")

if pint_f0 is not None:
    print(f"{'PINT':<10} {pint_f0:.20f}  {pint_f1:.15e}")

if jug_f0 is not None:
    print(f"{'JUG':<10} {jug_f0:.20f}  {jug_f1:.15e}")

# Speedup
if pint_time and jug_time:
    print(f"\nSpeedup JUG vs PINT: {pint_time/jug_time:.2f}x")
if t2_time and jug_time:
    print(f"Speedup JUG vs Tempo2: {t2_time/jug_time:.2f}x")

# Parameter agreement
if pint_f0 and jug_f0:
    print(f"\nParameter Agreement (JUG vs PINT):")
    print(f"  ΔF0 = {abs(float(jug_f0) - float(pint_f0)):.15e} Hz")
    print(f"  ΔF1 = {abs(float(jug_f1) - float(pint_f1)):.15e} Hz/s")

# ============================================================================
# PLOTS
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

methods = [
    ('Tempo2', t2_prefit_res, t2_postfit_res, t2_prefit_wrms, t2_postfit_wrms, t2_errors),
    ('PINT', pint_prefit_res, pint_postfit_res, pint_prefit_wrms, pint_postfit_wrms, pint_errors),
    ('JUG', jug_prefit_res, jug_postfit_res, jug_prefit_wrms, jug_postfit_wrms, jug_errors)
]

for idx, (name, prefit, postfit, pre_wrms, post_wrms, errors) in enumerate(methods):
    # Prefit
    ax_pre = axes[0, idx]
    if prefit is not None:
        toa_idx = np.arange(len(prefit))
        ax_pre.errorbar(toa_idx, prefit, yerr=errors, fmt='.', markersize=2, 
                       alpha=0.6, elinewidth=0.5, capsize=0)
        ax_pre.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_pre.set_ylabel('Residual (μs)', fontsize=10)
        ax_pre.set_title(f'{name} Prefit\nWRMS = {pre_wrms:.3f} μs', fontsize=11, fontweight='bold')
        ax_pre.grid(True, alpha=0.3, linestyle=':')
        ax_pre.set_xlabel('TOA Index', fontsize=10)
    else:
        ax_pre.text(0.5, 0.5, 'Not Available', ha='center', va='center', 
                   transform=ax_pre.transAxes, fontsize=12)
        ax_pre.set_title(f'{name} Prefit', fontsize=11)
    
    # Postfit
    ax_post = axes[1, idx]
    if postfit is not None:
        toa_idx = np.arange(len(postfit))
        ax_post.errorbar(toa_idx, postfit, yerr=errors, fmt='.', markersize=2,
                        alpha=0.6, elinewidth=0.5, capsize=0, color='red')
        ax_post.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_post.set_ylabel('Residual (μs)', fontsize=10)
        ax_post.set_title(f'{name} Postfit\nWRMS = {post_wrms:.3f} μs', fontsize=11, fontweight='bold')
        ax_post.grid(True, alpha=0.3, linestyle=':')
        ax_post.set_xlabel('TOA Index', fontsize=10)
    else:
        ax_post.text(0.5, 0.5, 'Not Available', ha='center', va='center',
                    transform=ax_post.transAxes, fontsize=12)
        ax_post.set_title(f'{name} Postfit', fontsize=11)

plt.tight_layout()
plt.savefig('benchmark_tempo2_pint_jug.png', dpi=200, bbox_inches='tight')
print("✓ Plot saved: benchmark_tempo2_pint_jug.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
with open('BENCHMARK_RESULTS.txt', 'w') as f:
    f.write("PULSAR TIMING BENCHMARK: Tempo2 vs PINT vs JUG\n")
    f.write("="*80 + "\n\n")
    f.write(f"Par file: {PAR_FILE}\n")
    f.write(f"Tim file: {TIM_FILE}\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Results table
    f.write(f"{'Method':<10} {'Prefit (μs)':<15} {'Postfit (μs)':<15} {'Time (s)':<12} {'Iters':<8}\n")
    f.write("-"*80 + "\n")
    
    if t2_postfit_wrms is not None:
        pre_str = f"{t2_prefit_wrms:.6f}" if t2_prefit_wrms else "N/A"
        post_str = f"{t2_postfit_wrms:.6f}"
        time_str = f"{t2_time:.3f}" if t2_time else "N/A"
        f.write(f"{'Tempo2':<10} {pre_str:<15} {post_str:<15} {time_str:<12} {'N/A':<8}\n")
    
    if pint_postfit_wrms is not None:
        f.write(f"{'PINT':<10} {pint_prefit_wrms:<15.6f} {pint_postfit_wrms:<15.6f} {pint_time:<12.3f} {str(pint_iters):<8}\n")
    
    if jug_postfit_wrms is not None:
        f.write(f"{'JUG':<10} {jug_prefit_wrms:<15.6f} {jug_postfit_wrms:<15.6f} {jug_time:<12.3f} {jug_iters:<8}\n")
    
    # Parameters
    f.write(f"\n{'Method':<10} {'F0 (Hz)':<30} {'F1 (Hz/s)':<25}\n")
    f.write("-"*80 + "\n")
    
    if t2_f0 is not None:
        f.write(f"{'Tempo2':<10} {t2_f0:.20f}  {t2_f1:.15e}\n")
    
    if pint_f0 is not None:
        f.write(f"{'PINT':<10} {pint_f0:.20f}  {pint_f1:.15e}\n")
    
    if jug_f0 is not None:
        f.write(f"{'JUG':<10} {jug_f0:.20f}  {jug_f1:.15e}\n")
    
    # Speedup
    if pint_time and jug_time:
        f.write(f"\nSpeedup JUG vs PINT: {pint_time/jug_time:.2f}x\n")
    if t2_time and jug_time:
        f.write(f"Speedup JUG vs Tempo2: {t2_time/jug_time:.2f}x\n")
    
    # Agreement
    if pint_f0 and jug_f0:
        f.write(f"\nParameter Agreement (JUG vs PINT):\n")
        f.write(f"  ΔF0 = {abs(float(jug_f0) - float(pint_f0)):.15e} Hz\n")
        f.write(f"  ΔF1 = {abs(float(jug_f1) - float(pint_f1)):.15e} Hz/s\n")

print("✓ Results saved: BENCHMARK_RESULTS.txt")
print("\nBenchmark complete!")
