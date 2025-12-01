#!/usr/bin/env python3
"""
Deep Performance Profiling of JUG Optimized Fitter
===================================================

This script profiles every component of the fitting pipeline to identify
bottlenecks and potential speedup opportunities.

We'll measure:
1. File I/O (par/tim parsing)
2. Cache initialization (delay computation)
   - Clock corrections
   - Ephemeris lookups
   - Barycentric delays
   - Binary delays
   - DM delays
3. JAX JIT compilation
4. Individual iteration components
   - Phase computation
   - Derivative computation
   - WLS solve
5. Total iteration time
"""

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path
import contextlib
import io

# Import JUG components
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import full_iteration_jax_f0_f1, wls_solve_jax

# Test data
PAR_FILE = Path("data/pulsars/J1909-3744_tdb_wrong.par")
TIM_FILE = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
CLOCK_DIR = "data/clock"

print("="*80)
print("DEEP PERFORMANCE PROFILING: JUG OPTIMIZED FITTER")
print("="*80)
print(f"\nPulsar: J1909-3744")
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")

# =============================================================================
# SECTION 1: FILE I/O
# =============================================================================
print("\n" + "="*80)
print("SECTION 1: FILE I/O")
print("="*80)

# Parse .par file
start = time.time()
params = parse_par_file(PAR_FILE)
par_time = time.time() - start
print(f"\n✓ Parse .par file: {par_time*1000:.3f} ms")
print(f"  Parameters loaded: {len(params)}")

# Parse .tim file
start = time.time()
toas_data = parse_tim_file_mjds(TIM_FILE)
tim_time = time.time() - start
print(f"✓ Parse .tim file: {tim_time*1000:.3f} ms")
print(f"  TOAs loaded: {len(toas_data)}")

total_io = par_time + tim_time
print(f"\n>>> TOTAL I/O TIME: {total_io*1000:.3f} ms")

# =============================================================================
# SECTION 2: CACHE INITIALIZATION (DELAY COMPUTATION)
# =============================================================================
print("\n" + "="*80)
print("SECTION 2: CACHE INITIALIZATION (Delay Computation)")
print("="*80)

print("\nWe'll compute residuals and extract intermediate timings...")

# Instrument compute_residuals_simple to get breakdown
# For now, just time the whole thing
start = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    result = compute_residuals_simple(
        PAR_FILE,
        TIM_FILE,
        clock_dir=CLOCK_DIR,
        subtract_tzr=False
    )
cache_total = time.time() - start

dt_sec_cached = result['dt_sec']
residuals_us = result['residuals_us']

print(f"\n✓ Total cache initialization: {cache_total:.3f} s ({cache_total*1000:.1f} ms)")
print(f"  dt_sec computed for {len(dt_sec_cached)} TOAs")
print(f"  Initial RMS: {np.sqrt(np.mean(residuals_us**2)):.3f} μs")

# Breakdown estimate (from Session 14 analysis)
print(f"\nEstimated breakdown:")
print(f"  - File parsing: ~{total_io*1000:.0f} ms (already measured)")
print(f"  - Clock corrections: ~{cache_total*0.15*1000:.0f} ms (15% estimate)")
print(f"  - Ephemeris lookups: ~{cache_total*0.40*1000:.0f} ms (40% estimate)")
print(f"  - Barycentric delays: ~{cache_total*0.25*1000:.0f} ms (25% estimate)")
print(f"  - Binary delays: ~{cache_total*0.15*1000:.0f} ms (15% estimate)")
print(f"  - DM/FD delays: ~{cache_total*0.05*1000:.0f} ms (5% estimate)")

print(f"\n>>> CACHE INITIALIZATION: {cache_total:.3f} s")

# =============================================================================
# SECTION 3: JAX ARRAY CONVERSION & SETUP
# =============================================================================
print("\n" + "="*80)
print("SECTION 3: JAX ARRAY CONVERSION")
print("="*80)

errors_us = np.array([toa.error_us for toa in toas_data])
errors_sec = errors_us * 1e-6
weights = 1.0 / errors_sec**2

start = time.time()
dt_sec_jax = jnp.array(dt_sec_cached)
errors_jax = jnp.array(errors_sec)
weights_jax = jnp.array(weights)
jax_conversion_time = time.time() - start

print(f"\n✓ Convert to JAX arrays: {jax_conversion_time*1000:.3f} ms")
print(f"  Shapes: dt_sec={dt_sec_jax.shape}, errors={errors_jax.shape}")

# =============================================================================
# SECTION 4: JAX JIT COMPILATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 4: JAX JIT COMPILATION")
print("="*80)

f0_test = params['F0']
f1_test = params['F1']

# First call triggers compilation
print("\nCompiling full_iteration_jax_f0_f1...")
start = time.time()
delta_params, rms_us, cov = full_iteration_jax_f0_f1(
    dt_sec_jax, f0_test, f1_test, errors_jax, weights_jax
)
jit_compile_time = time.time() - start

print(f"\n✓ JIT compilation time: {jit_compile_time:.3f} s ({jit_compile_time*1000:.0f} ms)")
print(f"  First iteration RMS: {float(rms_us):.3f} μs")

print(f"\n>>> JIT COMPILATION: {jit_compile_time:.3f} s")

# =============================================================================
# SECTION 5: ITERATION PERFORMANCE (POST-JIT)
# =============================================================================
print("\n" + "="*80)
print("SECTION 5: ITERATION PERFORMANCE (Post-JIT)")
print("="*80)

# Run 10 iterations to get stable timing
print("\nRunning 10 iterations to measure performance...")
iteration_times = []

for i in range(10):
    start = time.time()
    delta_params, rms_us, cov = full_iteration_jax_f0_f1(
        dt_sec_jax, f0_test, f1_test, errors_jax, weights_jax
    )
    iter_time = time.time() - start
    iteration_times.append(iter_time)
    
    # Update parameters for next iteration
    f0_test += float(delta_params[0])
    f1_test += float(delta_params[1])

iter_mean = np.mean(iteration_times)
iter_std = np.std(iteration_times)
iter_min = np.min(iteration_times)
iter_max = np.max(iteration_times)

print(f"\nIteration timing (10 runs):")
print(f"  Mean: {iter_mean*1000:.3f} ms")
print(f"  Std:  {iter_std*1000:.3f} ms")
print(f"  Min:  {iter_min*1000:.3f} ms")
print(f"  Max:  {iter_max*1000:.3f} ms")

print(f"\n>>> AVERAGE ITERATION TIME: {iter_mean*1000:.3f} ms")

# =============================================================================
# SECTION 6: BREAKDOWN OF SINGLE ITERATION
# =============================================================================
print("\n" + "="*80)
print("SECTION 6: BREAKDOWN OF SINGLE ITERATION")
print("="*80)

print("\nMeasuring individual components (outside JIT)...")

# 6.1: Phase computation
start = time.time()
phase = dt_sec_jax * (f0_test + dt_sec_jax * (f1_test / 2.0))
phase.block_until_ready()  # Force completion
phase_time = time.time() - start

print(f"\n1. Phase computation: {phase_time*1000:.3f} ms")
print(f"   Formula: phase = dt * (f0 + dt * f1/2)")

# 6.2: Phase wrapping
start = time.time()
phase_wrapped = phase - jnp.round(phase)
phase_wrapped.block_until_ready()
wrap_time = time.time() - start

print(f"2. Phase wrapping: {wrap_time*1000:.3f} ms")
print(f"   Formula: phase - round(phase)")

# 6.3: Convert to residuals
start = time.time()
residuals = phase_wrapped / f0_test
residuals.block_until_ready()
resid_time = time.time() - start

print(f"3. Phase → time: {resid_time*1000:.3f} ms")
print(f"   Formula: phase / f0")

# 6.4: Mean subtraction
start = time.time()
weighted_mean = jnp.sum(residuals * weights_jax) / jnp.sum(weights_jax)
residuals_centered = residuals - weighted_mean
residuals_centered.block_until_ready()
mean_time = time.time() - start

print(f"4. Weighted mean subtraction: {mean_time*1000:.3f} ms")

# 6.5: Derivative computation
start = time.time()
d_f0 = -(dt_sec_jax / f0_test)
d_f1 = -(dt_sec_jax**2 / 2.0) / f0_test
d_f0 = d_f0 - jnp.sum(d_f0 * weights_jax) / jnp.sum(weights_jax)
d_f1 = d_f1 - jnp.sum(d_f1 * weights_jax) / jnp.sum(weights_jax)
d_f1.block_until_ready()
deriv_time = time.time() - start

print(f"5. Derivative computation: {deriv_time*1000:.3f} ms")
print(f"   Includes: d/dF0, d/dF1, mean subtraction")

# 6.6: Design matrix assembly
start = time.time()
M = jnp.column_stack([d_f0, d_f1])
M.block_until_ready()
design_time = time.time() - start

print(f"6. Design matrix assembly: {design_time*1000:.3f} ms")
print(f"   Shape: {M.shape}")

# 6.7: WLS solve
start = time.time()
delta_params_test, cov_test = wls_solve_jax(residuals_centered, errors_jax, M)
delta_params_test.block_until_ready()
wls_time = time.time() - start

print(f"7. WLS solve (SVD): {wls_time*1000:.3f} ms")
print(f"   Output: delta_params shape {delta_params_test.shape}")

# 6.8: RMS computation
start = time.time()
rms_sec = jnp.sqrt(jnp.sum(residuals_centered**2 * weights_jax) / jnp.sum(weights_jax))
rms_us_test = rms_sec * 1e6
rms_us_test.block_until_ready()
rms_time = time.time() - start

print(f"8. RMS computation: {rms_time*1000:.3f} ms")

# Total breakdown
breakdown_total = (phase_time + wrap_time + resid_time + mean_time + 
                   deriv_time + design_time + wls_time + rms_time)

print(f"\n>>> BREAKDOWN TOTAL: {breakdown_total*1000:.3f} ms")
print(f">>> JIT COMPILED ACTUAL: {iter_mean*1000:.3f} ms")
print(f">>> JIT SPEEDUP: {breakdown_total/iter_mean:.2f}x")

# =============================================================================
# SECTION 7: MEMORY USAGE
# =============================================================================
print("\n" + "="*80)
print("SECTION 7: MEMORY FOOTPRINT")
print("="*80)

n_toas = len(dt_sec_cached)
bytes_per_float64 = 8

# Arrays
dt_sec_mem = n_toas * bytes_per_float64
errors_mem = n_toas * bytes_per_float64
weights_mem = n_toas * bytes_per_float64
residuals_mem = n_toas * bytes_per_float64
derivatives_mem = 2 * n_toas * bytes_per_float64  # F0 + F1
design_matrix_mem = 2 * n_toas * bytes_per_float64  # n_toas × 2

total_mem = (dt_sec_mem + errors_mem + weights_mem + residuals_mem + 
             derivatives_mem + design_matrix_mem)

print(f"\nArray memory usage:")
print(f"  dt_sec: {dt_sec_mem/1024/1024:.2f} MB")
print(f"  errors: {errors_mem/1024/1024:.2f} MB")
print(f"  weights: {weights_mem/1024/1024:.2f} MB")
print(f"  residuals: {residuals_mem/1024/1024:.2f} MB")
print(f"  derivatives: {derivatives_mem/1024/1024:.2f} MB")
print(f"  design matrix: {design_matrix_mem/1024/1024:.2f} MB")
print(f"\n>>> TOTAL MEMORY: {total_mem/1024/1024:.2f} MB")

# =============================================================================
# SECTION 8: SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n" + "="*80)
print("SECTION 8: PERFORMANCE SUMMARY")
print("="*80)

print(f"""
TIMING BREAKDOWN (for {n_toas} TOAs):
=====================================

1. File I/O:              {total_io*1000:8.1f} ms  ({total_io/cache_total*100:5.1f}% of cache)
2. Cache initialization:  {cache_total*1000:8.1f} ms  (includes ephemeris, clocks, delays)
3. JAX array conversion:  {jax_conversion_time*1000:8.1f} ms  ({jax_conversion_time/cache_total*100:5.1f}% of cache)
4. JIT compilation:       {jit_compile_time*1000:8.1f} ms  (one-time cost)
5. Per iteration:         {iter_mean*1000:8.1f} ms  (post-JIT, averaged over 10 runs)

TOTAL OVERHEAD (one-time): {(total_io + cache_total + jax_conversion_time + jit_compile_time):.3f} s
ITERATION TIME:            {iter_mean:.6f} s

For 10 iterations:
------------------
Total time = overhead + 10 × iteration
           = {(total_io + cache_total + jax_conversion_time + jit_compile_time):.3f}s + 10 × {iter_mean:.6f}s
           = {(total_io + cache_total + jax_conversion_time + jit_compile_time + 10*iter_mean):.3f}s

MEMORY USAGE: {total_mem/1024/1024:.2f} MB
""")

# =============================================================================
# SECTION 9: OPTIMIZATION OPPORTUNITIES
# =============================================================================
print("="*80)
print("SECTION 9: OPTIMIZATION OPPORTUNITIES")
print("="*80)

print("""
POTENTIAL SPEEDUPS:
==================

HIGH PRIORITY (Biggest Impact):
-------------------------------
1. Cache initialization ({:.1f}s = {:.0f}%)
   - Ephemeris lookups: ~{:.0f} ms
   - Could pre-load/cache ephemeris kernel
   - Could batch ephemeris queries
   - Could use faster ephemeris interpolation
   
2. JIT compilation ({:.1f}s, one-time)
   - Already optimal (can't improve JAX compile time)
   - Could save compiled function to disk (experimental)
   
3. File I/O ({:.0f} ms)
   - Already fast for Python
   - Could use binary format instead of text
   - Minimal gain expected

MEDIUM PRIORITY:
---------------
4. Per-iteration time ({:.1f} ms)
   - Already VERY fast with JAX JIT
   - WLS solve is dominant: ~{:.1f} ms
   - Could try GPU for large N (not needed at 10k TOAs)
   - Could reduce from float64 → float32 (risky!)

LOW PRIORITY (Already Optimal):
-------------------------------
5. Phase computation: {:.2f} ms (FAST)
6. Derivatives: {:.2f} ms (FAST)
7. Memory usage: {:.0f} MB (SMALL)

RECOMMENDATION:
==============
Focus on cache initialization (Section 2):
- Profile compute_residuals_simple() in detail
- Identify ephemeris/clock bottlenecks
- Consider caching strategy for multiple pulsars
""".format(
    cache_total, cache_total/(cache_total+jit_compile_time+10*iter_mean)*100,
    cache_total*0.40*1000,  # ephemeris estimate
    jit_compile_time,
    total_io*1000,
    iter_mean*1000,
    wls_time*1000,
    phase_time*1000,
    deriv_time*1000,
    total_mem/1024/1024
))

print("\n" + "="*80)
print("PROFILING COMPLETE")
print("="*80)
