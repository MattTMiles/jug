#!/usr/bin/env python3
"""
Complete F0+F1 fitting benchmark with prefit/postfit plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import tempfile
import os
import shutil

# Benchmark each method
par_file = 'data/pulsars/J1909-3744_tdb_wrong.par'
tim_file = 'data/pulsars/J1909-3744.tim'

print("="*70)
print("F0+F1 FITTING BENCHMARK: Tempo2 vs PINT vs JUG")
print("="*70)
print(f"Par file: {par_file}")
print(f"Tim file: {tim_file}")
print()

# ============================================================================
# TEMPO2
# ============================================================================
print("\n" + "="*70)
print("TEMPO2")
print("="*70)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_par = os.path.join(tmpdir, 'pulsar.par')
    tmp_tim = os.path.join(tmpdir, 'pulsar.tim')
    shutil.copy(par_file, tmp_par)
    shutil.copy(tim_file, tmp_tim)
    
    # Prefit
    result = subprocess.run(
        ['tempo2', '-f', tmp_par, tmp_tim, '-nofit', '-output', 'general2'],
        cwd=tmpdir, capture_output=True, text=True
    )
    
    # Check what files were created
    gen2_file = None
    for fname in os.listdir(tmpdir):
        if 'general2' in fname.lower():
            gen2_file = os.path.join(tmpdir, fname)
            break
    
    if gen2_file and os.path.exists(gen2_file):
        data_pre = np.loadtxt(gen2_file, usecols=(2, 3))
        t2_prefit_res = data_pre[:, 0]
        t2_errors = data_pre[:, 1]
        t2_prefit_wrms = np.sqrt(np.average(t2_prefit_res**2, weights=1.0/t2_errors**2))
    else:
        print("Warning: Tempo2 prefit output not found")
        print("Files created:", os.listdir(tmpdir))
        t2_prefit_res = None
        t2_prefit_wrms = None
        t2_errors = None
    
    # Fit with timing
    start = time.time()
    result = subprocess.run(
        ['tempo2', '-f', tmp_par, tmp_tim, '-fit', 'F0', '-fit', 'F1', '-output', 'general2'],
        cwd=tmpdir, capture_output=True, text=True
    )
    t2_time = time.time() - start
    
    # Postfit
    gen2_file = None
    for fname in os.listdir(tmpdir):
        if 'general2' in fname.lower():
            gen2_file = os.path.join(tmpdir, fname)
            break
    
    if gen2_file and os.path.exists(gen2_file):
        data_post = np.loadtxt(gen2_file, usecols=(2, 3))
        t2_postfit_res = data_post[:, 0]
        if t2_errors is None:
            t2_errors = data_post[:, 1]
        t2_postfit_wrms = np.sqrt(np.average(t2_postfit_res**2, weights=1.0/t2_errors**2))
    else:
        print("Warning: Tempo2 postfit output not found")
        t2_postfit_res = None
        t2_postfit_wrms = None
    
    # Get fitted values
    new_par = os.path.join(tmpdir, 'new.par')
    t2_f0 = t2_f1 = None
    if os.path.exists(new_par):
        with open(new_par) as f:
            for line in f:
                if line.startswith('F0'):
                    t2_f0 = float(line.split()[1])
                elif line.startswith('F1'):
                    t2_f1 = float(line.split()[1])

print(f"Time: {t2_time:.3f}s")
if t2_prefit_wrms:
    print(f"Prefit WRMS: {t2_prefit_wrms:.6f} μs")
if t2_postfit_wrms:
    print(f"Postfit WRMS: {t2_postfit_wrms:.6f} μs")
if t2_f0:
    print(f"F0: {t2_f0:.20f} Hz")
if t2_f1:
    print(f"F1: {t2_f1:.15e} Hz/s")

# ============================================================================
# PINT
# ============================================================================
print("\n" + "="*70)
print("PINT")
print("="*70)

import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
import pint.residuals

start = time.time()
model = pm.get_model(par_file)
toas = pt.get_TOAs(tim_file, model=model)

# Prefit
prefit = pint.residuals.Residuals(toas, model)
pint_prefit_res = prefit.time_resids.to_value('us')
pint_errors = toas.get_errors().to_value('us')
pint_prefit_wrms = np.sqrt(np.average(pint_prefit_res**2, weights=1.0/pint_errors**2))

# Fit
fitter = pf.WLSFitter(toas, model)
fitter.fit_toas()

# Postfit
pint_postfit_res = fitter.resids.time_resids.to_value('us')
pint_postfit_wrms = np.sqrt(np.average(pint_postfit_res**2, weights=1.0/pint_errors**2))

pint_time = time.time() - start
pint_f0 = fitter.model.F0.quantity.value
pint_f1 = fitter.model.F1.quantity.value
pint_iters = len(fitter.chi2_history) if hasattr(fitter, "chi2_history") else "N/A"

print(f"Time: {pint_time:.3f}s")
print(f"Prefit WRMS: {pint_prefit_wrms:.6f} μs")
print(f"Postfit WRMS: {pint_postfit_wrms:.6f} μs")
print(f"F0: {pint_f0:.20f} Hz")
print(f"F1: {pint_f1:.15e} Hz/s")
print(f"Iterations: {pint_iters}")

# ============================================================================
# JUG (Level 2 JAX)
# ============================================================================
print("\n" + "="*70)
print("JUG (Level 2 JAX Optimized)")
print("="*70)

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals
from jug.fitting.optimized_jax_fitter import fit_parameters_jax

start = time.time()

params = parse_par_file(par_file)
tim_data = parse_tim_file_mjds(tim_file)

# Prefit
prefit_result = compute_residuals(params, tim_data)
jug_prefit_res = prefit_result['residuals_us']
jug_errors = prefit_result['errors_us']
jug_prefit_wrms = np.sqrt(np.average(jug_prefit_res**2, weights=1.0/jug_errors**2))

# Fit
result = fit_parameters_jax(params, tim_data, ['F0', 'F1'], max_iter=20, tol=1e-9)

# Postfit
postfit_result = compute_residuals(result['params'], tim_data)
jug_postfit_res = postfit_result['residuals_us']
jug_postfit_wrms = np.sqrt(np.average(jug_postfit_res**2, weights=1.0/jug_errors**2))

jug_time = time.time() - start
jug_f0 = result['params']['F0']
jug_f1 = result['params']['F1']
jug_iters = result['iterations']

print(f"Time: {jug_time:.3f}s")
print(f"Prefit WRMS: {jug_prefit_wrms:.6f} μs")
print(f"Postfit WRMS: {jug_postfit_wrms:.6f} μs")
print(f"F0: {jug_f0:.20f} Hz")
print(f"F1: {jug_f1:.15e} Hz/s")
print(f"Iterations: {jug_iters}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Method':<20} {'Time (s)':<12} {'Postfit WRMS (μs)':<20} {'Iterations':<12}")
print("-"*70)
if t2_postfit_wrms:
    print(f"{'Tempo2':<20} {t2_time:<12.3f} {t2_postfit_wrms:<20.6f} {'N/A':<12}")
print(f"{'PINT':<20} {pint_time:<12.3f} {pint_postfit_wrms:<20.6f} {pint_iters:<12}")
print(f"{'JUG (Level 2 JAX)':<20} {jug_time:<12.3f} {jug_postfit_wrms:<20.6f} {jug_iters:<12}")
print()
print("Speedup:")
print(f"  JUG vs PINT: {pint_time/jug_time:.2f}x")
if t2_time:
    print(f"  JUG vs Tempo2: {t2_time/jug_time:.2f}x")

# ============================================================================
# PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

methods = [
    ('Tempo2', t2_prefit_res, t2_postfit_res, t2_prefit_wrms, t2_postfit_wrms),
    ('PINT', pint_prefit_res, pint_postfit_res, pint_prefit_wrms, pint_postfit_wrms),
    ('JUG (Level 2 JAX)', jug_prefit_res, jug_postfit_res, jug_prefit_wrms, jug_postfit_wrms)
]

for idx, (name, prefit, postfit, pre_wrms, post_wrms) in enumerate(methods):
    if prefit is not None:
        # Prefit
        axes[0, idx].scatter(range(len(prefit)), prefit, s=1, alpha=0.5, c='blue')
        axes[0, idx].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[0, idx].set_xlabel('TOA Index')
        axes[0, idx].set_ylabel('Residual (μs)')
        axes[0, idx].set_title(f'{name} Prefit\nWRMS={pre_wrms:.3f} μs')
        axes[0, idx].grid(True, alpha=0.3)
    
    if postfit is not None:
        # Postfit
        axes[1, idx].scatter(range(len(postfit)), postfit, s=1, alpha=0.5, c='red')
        axes[1, idx].axhline(0, color='k', linestyle='--', linewidth=0.5)
        axes[1, idx].set_xlabel('TOA Index')
        axes[1, idx].set_ylabel('Residual (μs)')
        axes[1, idx].set_title(f'{name} Postfit\nWRMS={post_wrms:.3f} μs')
        axes[1, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_f0_f1_final.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: benchmark_f0_f1_final.png")

# Save results
with open('BENCHMARK_F0_F1_FINAL.txt', 'w') as f:
    f.write("F0+F1 FITTING BENCHMARK RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Par file: {par_file}\n")
    f.write(f"Tim file: {tim_file}\n\n")
    
    if t2_postfit_wrms:
        f.write("TEMPO2:\n")
        f.write(f"  Time: {t2_time:.3f}s\n")
        f.write(f"  Prefit WRMS: {t2_prefit_wrms:.6f} μs\n")
        f.write(f"  Postfit WRMS: {t2_postfit_wrms:.6f} μs\n")
        f.write(f"  F0: {t2_f0:.20f} Hz\n")
        f.write(f"  F1: {t2_f1:.15e} Hz/s\n\n")
    
    f.write("PINT:\n")
    f.write(f"  Time: {pint_time:.3f}s\n")
    f.write(f"  Prefit WRMS: {pint_prefit_wrms:.6f} μs\n")
    f.write(f"  Postfit WRMS: {pint_postfit_wrms:.6f} μs\n")
    f.write(f"  F0: {pint_f0:.20f} Hz\n")
    f.write(f"  F1: {pint_f1:.15e} Hz/s\n")
    f.write(f"  Iterations: {pint_iters}\n\n")
    
    f.write("JUG (Level 2 JAX):\n")
    f.write(f"  Time: {jug_time:.3f}s\n")
    f.write(f"  Prefit WRMS: {jug_prefit_wrms:.6f} μs\n")
    f.write(f"  Postfit WRMS: {jug_postfit_wrms:.6f} μs\n")
    f.write(f"  F0: {jug_f0:.20f} Hz\n")
    f.write(f"  F1: {jug_f1:.15e} Hz/s\n")
    f.write(f"  Iterations: {jug_iters}\n\n")
    
    f.write("SPEEDUP:\n")
    f.write(f"  JUG vs PINT: {pint_time/jug_time:.2f}x\n")
    if t2_time:
        f.write(f"  JUG vs Tempo2: {t2_time/jug_time:.2f}x\n")

print("Results saved: BENCHMARK_F0_F1_FINAL.txt")
