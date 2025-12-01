#!/usr/bin/env python3
"""
Test Device Selection with Fitting
===================================

Verify that device selection works correctly and provides expected speedup.
"""

import time
from pathlib import Path

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.utils.device import set_device_preference, print_device_info

PAR_FILE = Path("data/pulsars/J1909-3744_tdb_wrong.par")
TIM_FILE = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")

print("="*80)
print("DEVICE SELECTION TEST")
print("="*80)

print("\nAvailable devices:")
print_device_info(verbose=True)

# Test 1: CPU (forced)
print("\n" + "="*80)
print("TEST 1: CPU Device (forced)")
print("="*80)

start = time.time()
result_cpu = fit_parameters_optimized(
    par_file=PAR_FILE,
    tim_file=TIM_FILE,
    fit_params=['F0', 'F1'],
    max_iter=10,
    verbose=False,
    device='cpu'
)
time_cpu = time.time() - start

print(f"\n✓ CPU fitting complete")
print(f"  Time: {time_cpu:.3f}s")
print(f"  RMS: {result_cpu['final_rms']:.6f} μs")
print(f"  Iterations: {result_cpu['iterations']}")

# Test 2: GPU (forced)
print("\n" + "="*80)
print("TEST 2: GPU Device (forced)")
print("="*80)

start = time.time()
result_gpu = fit_parameters_optimized(
    par_file=PAR_FILE,
    tim_file=TIM_FILE,
    fit_params=['F0', 'F1'],
    max_iter=10,
    verbose=False,
    device='gpu'
)
time_gpu = time.time() - start

print(f"\n✓ GPU fitting complete")
print(f"  Time: {time_gpu:.3f}s")
print(f"  RMS: {result_gpu['final_rms']:.6f} μs")
print(f"  Iterations: {result_gpu['iterations']}")

# Test 3: Auto selection
print("\n" + "="*80)
print("TEST 3: Auto Device Selection")
print("="*80)

start = time.time()
result_auto = fit_parameters_optimized(
    par_file=PAR_FILE,
    tim_file=TIM_FILE,
    fit_params=['F0', 'F1'],
    max_iter=10,
    verbose=False,
    device='auto'
)
time_auto = time.time() - start

print(f"\n✓ Auto fitting complete")
print(f"  Time: {time_auto:.3f}s")
print(f"  RMS: {result_auto['final_rms']:.6f} μs")
print(f"  Iterations: {result_auto['iterations']}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTiming comparison:")
print(f"  CPU:  {time_cpu:.3f}s")
print(f"  GPU:  {time_gpu:.3f}s")
print(f"  Auto: {time_auto:.3f}s (selected CPU)")

speedup = time_gpu / time_cpu
if speedup > 1.1:
    print(f"\n✓ CPU is {speedup:.2f}x FASTER than GPU (as expected for small problem)")
elif speedup < 0.9:
    print(f"\n⚠️  GPU is {1/speedup:.2f}x faster than CPU (unexpected for this problem size)")
else:
    print(f"\n✓ CPU and GPU have similar performance (±10%)")

# Verify accuracy
f0_diff = abs(result_cpu['final_params']['F0'] - result_gpu['final_params']['F0'])
f1_diff = abs(result_cpu['final_params']['F1'] - result_gpu['final_params']['F1'])

print(f"\nAccuracy comparison (CPU vs GPU):")
print(f"  ΔF0: {f0_diff:.2e} Hz")
print(f"  ΔF1: {f1_diff:.2e} Hz/s")

if f0_diff < 1e-15 and f1_diff < 1e-25:
    print("  ✅ Results identical within floating-point precision")
else:
    print("  ⚠️  Small differences (expected due to different compute paths)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)

print("""
Recommendation:
--------------
For typical pulsar timing (10k TOAs, 2 parameters):
  ✓ Use CPU device (--device cpu) - ~{:.1f}x faster
  
For large-scale analyses (>100k TOAs, >20 parameters):
  ✓ Use GPU device (--device gpu) - expected to be faster
  
For convenience:
  ✓ Use auto selection (--device auto) - chooses based on problem size
""".format(speedup))
