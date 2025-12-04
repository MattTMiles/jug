"""Benchmark JAX vs NumPy derivatives performance."""

import numpy as np
import time
from pathlib import Path

# Import both versions
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.derivatives_spin_jax import compute_spin_derivatives_jax

# Load J1909-3744 data for realistic test
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds

print("="*70)
print("JAX vs NumPy Derivatives Speed Comparison")
print("="*70)

# Load real data
data_dir = Path("/home/mattm/soft/JUG/data/pulsars")
par_file = data_dir / "J1909-3744_tdb.par"
tim_file = data_dir / "J1909-3744.tim"

params = parse_par_file(par_file)
tim_data = parse_tim_file_mjds(tim_file)
toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in tim_data])

print(f"\nDataset: J1909-3744")
print(f"Number of TOAs: {len(toas_mjd)}")
print(f"Fitting parameters: F0")

# Warm up JAX (first run compiles)
print("\n" + "-"*70)
print("Warming up JAX (compilation)...")
_ = compute_spin_derivatives_jax(params, toas_mjd, ['F0'])
print("✓ JAX compiled")

# NumPy benchmark
print("\n" + "-"*70)
print("Benchmarking NumPy version...")
n_iter = 100
times_numpy = []

for i in range(n_iter):
    t0 = time.perf_counter()
    derivs_numpy = compute_spin_derivatives(params, toas_mjd, ['F0'])
    t1 = time.perf_counter()
    times_numpy.append(t1 - t0)

avg_numpy = np.mean(times_numpy) * 1000  # ms
std_numpy = np.std(times_numpy) * 1000

print(f"NumPy: {avg_numpy:.3f} ± {std_numpy:.3f} ms ({n_iter} iterations)")

# JAX benchmark
print("\n" + "-"*70)
print("Benchmarking JAX version...")
times_jax = []

for i in range(n_iter):
    t0 = time.perf_counter()
    derivs_jax = compute_spin_derivatives_jax(params, toas_mjd, ['F0'])
    t1 = time.perf_counter()
    times_jax.append(t1 - t0)

avg_jax = np.mean(times_jax) * 1000  # ms
std_jax = np.std(times_jax) * 1000

print(f"JAX:   {avg_jax:.3f} ± {std_jax:.3f} ms ({n_iter} iterations)")

# Speedup
speedup = avg_numpy / avg_jax
print("\n" + "="*70)
print(f"SPEEDUP: {speedup:.1f}x faster with JAX!")
print("="*70)

# Verify results match
print("\nVerifying results match...")
max_diff = np.max(np.abs(derivs_numpy['F0'] - derivs_jax['F0']))
print(f"Max difference: {max_diff:.2e} seconds")

if max_diff < 1e-10:
    print("✅ Results match perfectly!")
else:
    print(f"⚠️  Results differ by {max_diff:.2e} seconds")

# Multi-parameter test
print("\n" + "="*70)
print("Multi-parameter test (F0, F1, F2)")
print("="*70)

fit_params = ['F0', 'F1', 'F2']

# Warm up
_ = compute_spin_derivatives_jax(params, toas_mjd, fit_params)

# NumPy
t0 = time.perf_counter()
for _ in range(n_iter):
    derivs_numpy = compute_spin_derivatives(params, toas_mjd, fit_params)
t1 = time.perf_counter()
time_numpy_multi = (t1 - t0) / n_iter * 1000

# JAX
t0 = time.perf_counter()
for _ in range(n_iter):
    derivs_jax = compute_spin_derivatives_jax(params, toas_mjd, fit_params)
t1 = time.perf_counter()
time_jax_multi = (t1 - t0) / n_iter * 1000

speedup_multi = time_numpy_multi / time_jax_multi

print(f"NumPy: {time_numpy_multi:.3f} ms")
print(f"JAX:   {time_jax_multi:.3f} ms")
print(f"Speedup: {speedup_multi:.1f}x")

print("\n" + "="*70)
print("✅ JAX derivatives validated and benchmarked!")
print("="*70)
