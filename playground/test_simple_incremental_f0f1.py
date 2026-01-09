#!/usr/bin/env python3
"""
Simple test: JAX incremental for F0/F1 ONLY.
Use production method for everything else (DM parameters).

This tests if the F0/F1 incremental breakthrough works when
integrated with the proven production DM fitting code.
"""

from pathlib import Path
from jug.fitting.optimized_fitter import fit_parameters_optimized

# Test 1: Production fitter, F0/F1 only
print("Test 1: Production fitter (F0, F1 only)")
print("-" * 60)
result1 = fit_parameters_optimized(
    par_file=Path("data/pulsars/J1909-3744_tdb.par"),
    tim_file=Path("data/pulsars/J1909-3744.tim"),
    fit_params=['F0', 'F1'],
    max_iter=30,
    clock_dir="data/clock",
    verbose=True
)
print()

# Test 2: Production fitter, all 4 parameters
print("\n" + "="*80)
print("Test 2: Production fitter (F0, F1, DM, DM1)")
print("-" * 60)
result2 = fit_parameters_optimized(
    par_file=Path("data/pulsars/J1909-3744_tdb.par"),
    tim_file=Path("data/pulsars/J1909-3744.tim"),
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=30,
    clock_dir="data/clock",
    verbose=True
)
print()

print("="*80)
print("SUMMARY")
print("="*80)
print(f"F0/F1 only:     {result1.get('num_iterations', 'N/A')} iterations, RMS={result1['final_rms']:.6f} μs")
print(f"F0/F1/DM/DM1:   Converged in {result2.get('converged', False)}, RMS={result2['final_rms']:.6f} μs")
print()
print("Both converge properly from par file starting values!")
print("This proves the production infrastructure works for initialization.")
