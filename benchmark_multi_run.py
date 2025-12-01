#!/usr/bin/env python3
"""
Multi-run benchmark to eliminate timing variance
"""

import time
import numpy as np
from pathlib import Path
import subprocess
import sys

# Suppress warnings
import os
os.environ['JAX_LOG_COMPILES'] = '0'

print("="*80)
print("MULTI-RUN BENCHMARK: 10 REALIZATIONS")
print("="*80)

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

N_RUNS = 10

# ============================================================================
# JUG Benchmark
# ============================================================================
print("\n" + "="*80)
print("JUG BENCHMARK (10 runs)")
print("="*80)

from jug.fitting.optimized_fitter import fit_parameters_optimized

jug_times = []
for i in range(N_RUNS):
    print(f"\rRun {i+1}/{N_RUNS}...", end='', flush=True)
    
    t0 = time.time()
    result = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=['F0', 'F1'],
        max_iter=15,
        verbose=False,
        device='cpu'
    )
    elapsed = time.time() - t0
    jug_times.append(elapsed)

print()  # newline
jug_times = np.array(jug_times)

print(f"\nJUG Results (10 runs):")
print(f"  Mean:   {jug_times.mean():.3f}s")
print(f"  Median: {np.median(jug_times):.3f}s")
print(f"  Std:    {jug_times.std():.3f}s")
print(f"  Min:    {jug_times.min():.3f}s")
print(f"  Max:    {jug_times.max():.3f}s")
print(f"  All:    {', '.join(f'{t:.3f}' for t in jug_times)}s")

# ============================================================================
# PINT Benchmark
# ============================================================================
print("\n" + "="*80)
print("PINT BENCHMARK (10 runs)")
print("="*80)

import pint.models
import pint.toa
import pint.fitter

pint_times = []
for i in range(N_RUNS):
    print(f"\rRun {i+1}/{N_RUNS}...", end='', flush=True)
    
    t0 = time.time()
    
    # Load model and TOAs
    model = pint.models.get_model(str(par_file))
    toas = pint.toa.get_TOAs(str(tim_file), planets=True, ephem='DE440')
    
    # Fit
    fitter = pint.fitter.WLSFitter(toas, model)
    fitter.model.free_params = ['F0', 'F1']
    fitter.fit_toas(maxiter=15)
    
    elapsed = time.time() - t0
    pint_times.append(elapsed)

print()  # newline
pint_times = np.array(pint_times)

print(f"\nPINT Results (10 runs):")
print(f"  Mean:   {pint_times.mean():.3f}s")
print(f"  Median: {np.median(pint_times):.3f}s")
print(f"  Std:    {pint_times.std():.3f}s")
print(f"  Min:    {pint_times.min():.3f}s")
print(f"  Max:    {pint_times.max():.3f}s")
print(f"  All:    {', '.join(f'{t:.3f}' for t in pint_times)}s")

# ============================================================================
# Tempo2 Benchmark
# ============================================================================
print("\n" + "="*80)
print("TEMPO2 BENCHMARK (10 runs)")
print("="*80)

tempo2_times = []
for i in range(N_RUNS):
    print(f"\rRun {i+1}/{N_RUNS}...", end='', flush=True)
    
    cmd = [
        'tempo2',
        '-f', str(par_file),
        str(tim_file),
        '-fit', 'F0',
        '-fit', 'F1'
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    tempo2_times.append(elapsed)

print()  # newline
tempo2_times = np.array(tempo2_times)

print(f"\nTempo2 Results (10 runs):")
print(f"  Mean:   {tempo2_times.mean():.3f}s")
print(f"  Median: {np.median(tempo2_times):.3f}s")
print(f"  Std:    {tempo2_times.std():.3f}s")
print(f"  Min:    {tempo2_times.min():.3f}s")
print(f"  Max:    {tempo2_times.max():.3f}s")
print(f"  All:    {', '.join(f'{t:.3f}' for t in tempo2_times)}s")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY (Mean ± Std)")
print("="*80)

print(f"\nTempo2: {tempo2_times.mean():.3f} ± {tempo2_times.std():.3f}s")
print(f"JUG:    {jug_times.mean():.3f} ± {jug_times.std():.3f}s")
print(f"PINT:   {pint_times.mean():.3f} ± {pint_times.std():.3f}s")

print(f"\nSpeedup vs Tempo2:")
print(f"  JUG:  {tempo2_times.mean() / jug_times.mean():.2f}× {'faster' if tempo2_times.mean() > jug_times.mean() else 'slower'}")
print(f"  PINT: {tempo2_times.mean() / pint_times.mean():.2f}× {'faster' if tempo2_times.mean() > pint_times.mean() else 'slower'}")

print(f"\nJUG vs PINT:")
print(f"  JUG is {pint_times.mean() / jug_times.mean():.2f}× faster than PINT")

print("\n" + "="*80)
print("VARIANCE ANALYSIS")
print("="*80)
print(f"Tempo2 variance: {100*tempo2_times.std()/tempo2_times.mean():.1f}%")
print(f"JUG variance:    {100*jug_times.std()/jug_times.mean():.1f}%")
print(f"PINT variance:   {100*pint_times.std()/pint_times.mean():.1f}%")
