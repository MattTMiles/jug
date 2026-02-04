"""Diagnose why astrometric parameter fitting diverges on multiple iterations."""

import numpy as np
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized_cached

# Load test data
with open('data/pulsars/J1909-3744_tdb.par', 'r') as f:
    params = parse_par_file(f.read())
    
with open('data/pulsars/J1909-3744.tim', 'r') as f:
    toas = parse_tim_file(f.read())

# Compute initial residuals
result = compute_residuals_simple(params, toas)
print(f"Initial RMS: {result['rms_residual']:.6f} μs")
print()

# Try fitting astrometric parameters multiple times
fit_params = ['RAJ', 'DECJ', 'PX']

print("=" * 80)
print("ITERATION 1")
print("=" * 80)
result1 = fit_parameters_optimized_cached(
    params, toas, fit_params,
    max_iterations=1,
    tolerance=1e-10,
    verbose=True
)

print(f"\nPost-fit RMS: {result1['rms']:.6f} μs")
print("\nParameter changes:")
for param in fit_params:
    old_val = params[param]
    new_val = result1['fitted_params'][param]
    if isinstance(old_val, str):
        print(f"  {param}: {old_val} -> {new_val}")
    else:
        print(f"  {param}: {old_val:.12e} -> {new_val:.12e} (Δ = {new_val - old_val:.6e})")

# Update params for next iteration
params_iter2 = result1['fitted_params'].copy()

print("\n" + "=" * 80)
print("ITERATION 2")
print("=" * 80)
result2 = fit_parameters_optimized_cached(
    params_iter2, toas, fit_params,
    max_iterations=1,
    tolerance=1e-10,
    verbose=True
)

print(f"\nPost-fit RMS: {result2['rms']:.6f} μs")
print("\nParameter changes:")
for param in fit_params:
    old_val = params_iter2[param]
    new_val = result2['fitted_params'][param]
    if isinstance(old_val, str):
        print(f"  {param}: {old_val} -> {new_val}")
    else:
        print(f"  {param}: {old_val:.12e} -> {new_val:.12e} (Δ = {new_val - old_val:.6e})")

# Update params for next iteration
params_iter3 = result2['fitted_params'].copy()

print("\n" + "=" * 80)
print("ITERATION 3")
print("=" * 80)
result3 = fit_parameters_optimized_cached(
    params_iter3, toas, fit_params,
    max_iterations=1,
    tolerance=1e-10,
    verbose=True
)

print(f"\nPost-fit RMS: {result3['rms']:.6f} μs")
print("\nParameter changes:")
for param in fit_params:
    old_val = params_iter3[param]
    new_val = result3['fitted_params'][param]
    if isinstance(old_val, str):
        print(f"  {param}: {old_val} -> {new_val}")
    else:
        print(f"  {param}: {old_val:.12e} -> {new_val:.12e} (Δ = {new_val - old_val:.6e})")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Initial RMS: {result['rms_residual']:.6f} μs")
print(f"After iteration 1: {result1['rms']:.6f} μs")
print(f"After iteration 2: {result2['rms']:.6f} μs")
print(f"After iteration 3: {result3['rms']:.6f} μs")
