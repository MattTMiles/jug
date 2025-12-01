#!/usr/bin/env python3
"""
Fair Benchmark: Measure ONLY fitting time (not I/O or prefit/postfit)
"""

import time
import subprocess
import sys

PAR_FILE = 'data/pulsars/J1909-3744_tdb_wrong.par'
TIM_FILE = 'data/pulsars/J1909-3744.tim'

print("="*80)
print("FAIR BENCHMARK: Fitting Time Only")
print("="*80)
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")
print()

# ============================================================================
# PINT (fresh process)
# ============================================================================
print("Running PINT...")
start = time.time()
result = subprocess.run([
    sys.executable, '-c', f'''
import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
import time

# Load data
model = pm.get_model("{PAR_FILE}")
toas = pt.get_TOAs("{TIM_FILE}", model=model)

# Time JUST the fitting
model.F0.frozen = False
model.F1.frozen = False
fit_start = time.time()
fitter = pf.WLSFitter(toas, model)
fitter.fit_toas()
fit_time = time.time() - fit_start

print(f"FITTING_TIME={{fit_time:.3f}}")
print(f"F0={fitter.model.F0.quantity.value:.20f}")
print(f"RMS={fitter.resids.time_resids.std().to_value('us'):.6f}")
'''
], capture_output=True, text=True, env={'PYTHONWARNINGS': 'ignore'})

pint_total_time = time.time() - start

# Parse output
pint_fit_time = pint_f0 = pint_rms = None
for line in result.stdout.split('\n'):
    if line.startswith('FITTING_TIME='):
        pint_fit_time = float(line.split('=')[1])
    elif line.startswith('F0='):
        pint_f0 = float(line.split('=')[1])
    elif line.startswith('RMS='):
        pint_rms = float(line.split('=')[1])

print(f"✓ Fitting time: {pint_fit_time:.3f}s")
print(f"✓ Total time: {pint_total_time:.3f}s")
print(f"✓ F0 = {pint_f0:.20f}")
print(f"✓ RMS = {pint_rms:.3f} μs")
print()

# ============================================================================
# JUG (fresh process)
# ============================================================================
print("Running JUG...")
start = time.time()
result = subprocess.run([
    sys.executable, '-c', f'''
from jug.fitting import fit_parameters_optimized
from pathlib import Path

result = fit_parameters_optimized(
    par_file=Path("{PAR_FILE}"),
    tim_file=Path("{TIM_FILE}"),
    fit_params=["F0", "F1"],
    verbose=False
)

print(f"FITTING_TIME={{result['total_time']:.3f}}")
print(f"CACHE_TIME={{result['cache_time']:.3f}}")
print(f"JIT_TIME={{result['jit_time']:.3f}}")
print(f"F0={{result['final_params']['F0']:.20f}}")
print(f"RMS={{result['final_rms']:.6f}}")
print(f"ITERS={{result['iterations']}}")
'''
], capture_output=True, text=True)

jug_total_time = time.time() - start

# Parse output  
jug_fit_time = jug_cache_time = jug_jit_time = jug_f0 = jug_rms = jug_iters = None
for line in result.stdout.split('\n'):
    if line.startswith('FITTING_TIME='):
        jug_fit_time = float(line.split('=')[1])
    elif line.startswith('CACHE_TIME='):
        jug_cache_time = float(line.split('=')[1])
    elif line.startswith('JIT_TIME='):
        jug_jit_time = float(line.split('=')[1])
    elif line.startswith('F0='):
        jug_f0 = float(line.split('=')[1])
    elif line.startswith('RMS='):
        jug_rms = float(line.split('=')[1])
    elif line.startswith('ITERS='):
        jug_iters = int(line.split('=')[1])

jug_actual_fitting = jug_fit_time - jug_cache_time - jug_jit_time

print(f"✓ Total time: {jug_fit_time:.3f}s")
print(f"  - Cache init: {jug_cache_time:.3f}s")
print(f"  - JIT compile: {jug_jit_time:.3f}s")
print(f"  - Actual fitting: {jug_actual_fitting:.3f}s")
print(f"✓ F0 = {jug_f0:.20f}")
print(f"✓ RMS = {jug_rms:.3f} μs")
print(f"✓ Iterations: {jug_iters}")
print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*80)
print("RESULTS")
print("="*80)
print()
print("PINT:")
print(f"  Fitting only: {pint_fit_time:.3f}s")
print(f"  Total (with I/O): {pint_total_time:.3f}s")
print()
print("JUG:")
print(f"  Fitting only (including cache/JIT): {jug_fit_time:.3f}s")
print(f"  Actual iterations: {jug_actual_fitting:.3f}s")
print(f"  Total (with subprocess): {jug_total_time:.3f}s")
print()
print("Speedup:")
print(f"  JUG vs PINT (total fitting): {pint_fit_time/jug_fit_time:.2f}x")
print(f"  JUG vs PINT (iterations only): {pint_fit_time/jug_actual_fitting:.2f}x")
print()
print("Accuracy:")
print(f"  ΔF0 = {abs(jug_f0 - pint_f0):.2e} Hz")
print(f"  ΔRMS = {abs(jug_rms - pint_rms):.3f} μs")
print()

# Save results
with open('BENCHMARK_FITTING_ONLY.txt', 'w') as f:
    f.write("FAIR BENCHMARK: Fitting Time Only\n")
    f.write("="*80 + "\n\n")
    f.write(f"Par file: {PAR_FILE}\n")
    f.write(f"Tim file: {TIM_FILE}\n\n")
    
    f.write("PINT:\n")
    f.write(f"  Fitting only: {pint_fit_time:.3f}s\n")
    f.write(f"  F0: {pint_f0:.20f}\n")
    f.write(f"  RMS: {pint_rms:.3f} μs\n\n")
    
    f.write("JUG:\n")
    f.write(f"  Total fitting: {jug_fit_time:.3f}s\n")
    f.write(f"  - Cache init: {jug_cache_time:.3f}s (one-time)\n")
    f.write(f"  - JIT compile: {jug_jit_time:.3f}s (one-time)\n")
    f.write(f"  - Iterations: {jug_actual_fitting:.3f}s\n")
    f.write(f"  F0: {jug_f0:.20f}\n")
    f.write(f"  RMS: {jug_rms:.3f} μs\n")
    f.write(f"  Iterations: {jug_iters}\n\n")
    
    f.write("Speedup:\n")
    f.write(f"  JUG vs PINT (total): {pint_fit_time/jug_fit_time:.2f}x\n")
    f.write(f"  JUG vs PINT (iterations only): {pint_fit_time/jug_actual_fitting:.2f}x\n\n")
    
    f.write("Accuracy:\n")
    f.write(f"  ΔF0 = {abs(jug_f0 - pint_f0):.2e} Hz\n")
    f.write(f"  ΔRMS = {abs(jug_rms - pint_rms):.3f} μs\n")

print("✓ Results saved: BENCHMARK_FITTING_ONLY.txt")
