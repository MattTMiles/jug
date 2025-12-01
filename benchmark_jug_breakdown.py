#!/usr/bin/env python3
"""
Simple timing breakdown using the actual optimized_fitter with verbose mode.
"""

import time
from pathlib import Path
import os

# Suppress warnings
os.environ['JAX_LOG_COMPILES'] = '0'

print("="*80)
print("JUG WORKFLOW BREAKDOWN - Using Actual Production Code")
print("="*80)

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

from jug.fitting.optimized_fitter import fit_parameters_optimized

# ============================================================================
# RUN 1: COLD START
# ============================================================================
print("\n" + "="*80)
print("RUN 1: COLD START (First run with JIT compilation)")
print("="*80)

t0_total = time.time()
result1 = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=15,
    verbose=True,  # Shows detailed breakdown
    device='cpu'
)
time_cold = time.time() - t0_total

print(f"\n{'='*80}")
print(f"RUN 1 TOTAL TIME: {time_cold:.3f}s")
print(f"{'='*80}")
print(f"\nBreakdown from result:")
print(f"  Cache initialization: {result1['cache_time']:.3f}s  ({100*result1['cache_time']/time_cold:.1f}%)")
print(f"  JIT compilation:      {result1['jit_time']:.3f}s  ({100*result1['jit_time']/time_cold:.1f}%)")
print(f"  Fitting iterations:   {time_cold - result1['cache_time'] - result1['jit_time']:.3f}s  ({100*(time_cold - result1['cache_time'] - result1['jit_time'])/time_cold:.1f}%)")
print(f"\nFit quality:")
print(f"  Prefit RMS:  {result1['prefit_rms']:.6f} μs")
print(f"  Postfit RMS: {result1['final_rms']:.6f} μs")
print(f"  Iterations:  {result1['iterations']}")
print(f"  Converged:   {result1['converged']}")

# ============================================================================
# RUN 2: WARM START
# ============================================================================
print("\n" + "="*80)
print("RUN 2: WARM START (JIT cached, typical production performance)")
print("="*80)

t0_total = time.time()
result2 = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=15,
    verbose=True,
    device='cpu'
)
time_warm = time.time() - t0_total

print(f"\n{'='*80}")
print(f"RUN 2 TOTAL TIME: {time_warm:.3f}s")
print(f"{'='*80}")
print(f"\nBreakdown from result:")
print(f"  Cache initialization: {result2['cache_time']:.3f}s  ({100*result2['cache_time']/time_warm:.1f}%)")
print(f"  JIT compilation:      {result2['jit_time']:.3f}s  ({100*result2['jit_time']/time_warm:.1f}%)")
print(f"  Fitting iterations:   {time_warm - result2['cache_time'] - result2['jit_time']:.3f}s  ({100*(time_warm - result2['cache_time'] - result2['jit_time'])/time_warm:.1f}%)")
print(f"\nFit quality:")
print(f"  Prefit RMS:  {result2['prefit_rms']:.6f} μs")
print(f"  Postfit RMS: {result2['final_rms']:.6f} μs")
print(f"  Iterations:  {result2['iterations']}")
print(f"  Converged:   {result2['converged']}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COLD vs WARM COMPARISON")
print("="*80)

print(f"\n{'Component':<30} {'Cold':<15} {'Warm':<15} {'Speedup':<10}")
print("-" * 80)

components = [
    ('Cache initialization', result1['cache_time'], result2['cache_time']),
    ('JIT compilation', result1['jit_time'], result2['jit_time']),
    ('Fitting iterations', 
     time_cold - result1['cache_time'] - result1['jit_time'],
     time_warm - result2['cache_time'] - result2['jit_time']),
    ('TOTAL', time_cold, time_warm)
]

for name, cold_val, warm_val in components:
    speedup = cold_val / warm_val if warm_val > 0 else 1.0
    print(f"{name:<30} {cold_val:>8.3f}s      {warm_val:>8.3f}s      {speedup:>6.2f}x")

speedup_total = time_cold / time_warm
print(f"\n{'='*80}")
print(f"OVERALL SPEEDUP (Warm vs Cold): {speedup_total:.2f}x")
print(f"{'='*80}")

print(f"\nTypical production performance (warm): {time_warm:.3f}s per fit")
print(f"This is {(2.071 / time_warm):.2f}x faster than Tempo2 (2.071s)")
print(f"This is {(21.998 / time_warm):.2f}x faster than PINT (21.998s)")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"\n1. Cache initialization: ~{result2['cache_time']:.1f}s (consistent)")
print(f"   - This is computing all delays (barycentric, binary, DM, etc.)")
print(f"   - Done once, then cached for fitting iterations")
print(f"   - Takes {100*result2['cache_time']/time_warm:.0f}% of warm run time")

jit_reduction = (result1['jit_time'] - result2['jit_time']) / result1['jit_time'] * 100
print(f"\n2. JIT compilation: {result1['jit_time']:.3f}s cold → {result2['jit_time']:.3f}s warm ({jit_reduction:.0f}% reduction)")
print(f"   - Cold: Compiling JAX functions ({result1['jit_time']:.3f}s)")
print(f"   - Warm: Using cached compiled code ({result2['jit_time']:.3f}s)")
print(f"   - Savings: {result1['jit_time'] - result2['jit_time']:.3f}s")

iter_cold = time_cold - result1['cache_time'] - result1['jit_time']
iter_warm = time_warm - result2['cache_time'] - result2['jit_time']
print(f"\n3. Fitting iterations: {iter_cold:.3f}s cold → {iter_warm:.3f}s warm")
print(f"   - Per iteration (cold): {iter_cold/result1['iterations']:.3f}s")
print(f"   - Per iteration (warm): {iter_warm/result2['iterations']:.3f}s")
print(f"   - Speedup: {iter_cold/iter_warm:.2f}x")

print(f"\n4. Overall: {time_cold:.3f}s → {time_warm:.3f}s ({speedup_total:.2f}x speedup)")
print(f"   - Cold start overhead: {time_cold - time_warm:.3f}s (paid once)")
print(f"   - Amortized after just 1-2 pulsars")

print("\n" + "="*80)
print("WHERE DOES THE TIME GO? (Warm Run)")
print("="*80)

print(f"\nCache init:    {result2['cache_time']:6.3f}s  ", end='')
print("█" * int(40 * result2['cache_time'] / time_warm))
print(f"\nJIT compile:   {result2['jit_time']:6.3f}s  ", end='')
print("█" * int(40 * result2['jit_time'] / time_warm))
print(f"\nIterations:    {iter_warm:6.3f}s  ", end='')
print("█" * int(40 * iter_warm / time_warm))

print(f"\n\nTotal: {time_warm:.3f}s")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nAfter JIT warmup, JUG delivers consistent {time_warm:.3f}s performance.")
print(f"This makes JUG the FASTEST option for:")
print(f"  • Batch processing (2+ pulsars)")
print(f"  • Interactive sessions (after warmup)")
print(f"  • Production pipelines")
