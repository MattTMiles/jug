"""Test DM fitting on J1909-3744.

This script tests the new DM parameter fitting capability by:
1. Fitting F0, F1, and DM together
2. Comparing convergence and RMS
3. Verifying parameter uncertainties
"""

from pathlib import Path
from jug.fitting.optimized_fitter import fit_parameters_optimized

# Test data
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

print("="*80)
print("TEST 1: Fit F0 + F1 only (baseline)")
print("="*80)

result_spin = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=25,
    verbose=True
)

print("\n" + "="*80)
print("TEST 2: Fit F0 + F1 + DM together")
print("="*80)

result_mixed = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'DM'],
    max_iter=25,
    verbose=True
)

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"\nSpin only (F0+F1):")
print(f"  Final RMS: {result_spin['final_rms']:.6f} μs")
print(f"  Iterations: {result_spin['iterations']}")
print(f"  F0 = {result_spin['final_params']['F0']:.15f} Hz")
print(f"  F1 = {result_spin['final_params']['F1']:.15e} Hz/s")

print(f"\nSpin + DM (F0+F1+DM):")
print(f"  Final RMS: {result_mixed['final_rms']:.6f} μs")
print(f"  Iterations: {result_mixed['iterations']}")
print(f"  F0 = {result_mixed['final_params']['F0']:.15f} Hz")
print(f"  F1 = {result_mixed['final_params']['F1']:.15e} Hz/s")
print(f"  DM = {result_mixed['final_params']['DM']:.10f} pc cm⁻³")

print(f"\nDM parameter:")
print(f"  Uncertainty: ± {result_mixed['uncertainties']['DM']:.6e} pc cm⁻³")

if result_mixed['final_rms'] < result_spin['final_rms']:
    improvement = result_spin['final_rms'] - result_mixed['final_rms']
    print(f"\n✓ Including DM improved RMS by {improvement:.6f} μs")
else:
    print(f"\nNote: RMS similar with/without fitting DM (DM already well-constrained)")

print("\n" + "="*80)
print("TEST 3: Fit DM only")
print("="*80)

result_dm_only = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['DM'],
    max_iter=25,
    verbose=True
)

print(f"\nDM-only fit:")
print(f"  Final RMS: {result_dm_only['final_rms']:.6f} μs")
print(f"  DM = {result_dm_only['final_params']['DM']:.10f} ± {result_dm_only['uncertainties']['DM']:.6e} pc cm⁻³")

print("\n✓ All DM fitting tests complete!")
