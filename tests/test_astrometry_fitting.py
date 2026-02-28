#!/usr/bin/env python3
"""Test astrometry parameter fitting integration."""

import numpy as np
import pytest
from pathlib import Path

# Test data
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

if not (par_file.exists() and tim_file.exists()):
    pytest.skip("J1909 test data not available", allow_module_level=True)

from jug.fitting.optimized_fitter import fit_parameters_optimized

clock_dir = None  # Skip clock corrections for testing

print("=" * 80)
print("Testing astrometry parameter fitting integration")
print("=" * 80)

# Test 1: Fit a single astrometry parameter (RAJ)
print("\n1. Testing RAJ fitting...")
result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'RAJ'],
    clock_dir=clock_dir,
    max_iter=1,
    convergence_threshold=1e-10,
    verbose=False
)
print(f"✓ RAJ fitting successful")
print(f"  Converged: {result['converged']}")
print(f"  Final RMS: {result['final_rms']:.3f} μs")

# Test 2: Fit multiple astrometry parameters
print("\n2. Testing multiple astrometry parameters (RAJ, DECJ, PMRA, PMDEC)...")
result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC'],
    clock_dir=clock_dir,
    max_iter=3,
    convergence_threshold=1e-10,
    verbose=False
)
print(f"✓ Multiple astrometry parameter fitting successful")
print(f"  Converged: {result['converged']}")
print(f"  Final RMS: {result['final_rms']:.3f} μs")

# Test 3: Fit all astrometry parameters (including PX)
print("\n3. Testing all astrometry parameters (RAJ, DECJ, PMRA, PMDEC, PX)...")
result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX'],
    clock_dir=clock_dir,
    max_iter=3,
    convergence_threshold=1e-10,
    verbose=False
)
print(f"✓ All astrometry parameter fitting successful")
print(f"  Converged: {result['converged']}")
print(f"  Final RMS: {result['final_rms']:.3f} μs")

# Test 4: Mixed fit (spin + DM + astrometry + binary)
print("\n4. Testing mixed parameter fit (spin + DM + astrometry + binary)...")
result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'DM', 'RAJ', 'DECJ', 'PB', 'A1', 'TASC'],
    clock_dir=clock_dir,
    max_iter=3,
    convergence_threshold=1e-10,
    verbose=False
)
print(f"✓ Mixed parameter fitting successful")
print(f"  Converged: {result['converged']}")
print(f"  Final RMS: {result['final_rms']:.3f} μs")

print("\n" + "=" * 80)
print("All astrometry fitting tests passed! ✓")
print("=" * 80)
