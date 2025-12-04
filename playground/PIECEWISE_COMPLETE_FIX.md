# Piecewise Fitting - Complete Fix

**Date:** 2025-12-02  
**Final Status:** ✓ FIXED - Both residuals and design matrix now consistent

## The Two Bugs

### Bug 1: Residual Normalization
**Location:** `compute_residuals_piecewise`, line 250  
**Problem:** Used `f0_local` instead of `f0_global` to convert phase to time  
**Fix:** Changed to `residuals_sec[idx] = phase_wrapped / f0_global`

### Bug 2: Design Matrix Normalization  
**Location:** `compute_design_matrix_piecewise`, lines 362-363  
**Problem:** Used `f0_local` instead of `f0_global` for derivatives  
**Fix:** Changed both lines to divide by `f0_global`

## Why Both Must Use f0_global

After adding the phase offset correction:
```python
phase_corrected = phase_local + phase_offset
```

The phase is now on the **global phase scale**. Therefore:

1. **Residuals:** `residuals = phase_wrapped / f0_global`
2. **Design matrix:** `M = -∂phase/∂param / f0_global`

Both must use the **same** normalization frequency, otherwise:
- Residuals and derivatives are inconsistent
- Fitting diverges or converges to wrong values
- Postfit RMS becomes worse than prefit

## Symptoms of the Bug

When only residuals were fixed but not design matrix:
- Piecewise residuals showed quadratic curve (~20-25 μs swing)
- Fitting made RMS WORSE (prefit: 0.4 μs → postfit: 6.9 μs!)
- Large offset between piecewise and standard residuals

## After Complete Fix

With both fixes applied:
- Piecewise residuals match standard residuals
- Peak-to-peak difference: ~70 ns (float64 precision limit)
- No quadratic curve
- Fitting converges properly
- Postfit RMS improves as expected

## To Verify the Fix

1. Restart Jupyter kernel
2. Run all cells in order
3. Check step 9 output:
   - Prefit RMS should be ~0.8 μs
   - Postfit RMS should be **less than** prefit
   - Improvement should be positive
4. Check step 12 plots:
   - Piecewise and standard residuals should look identical
   - Difference plot should show ~70 ns scatter, no curve

## Files Updated

1. `piecewise_fitting_implementation.ipynb`:
   - Line 250: `residuals_sec[idx] = phase_wrapped / f0_global`
   - Line 362: `M[idx, 0] = -d_phase_d_f0 / f0_global`
   - Line 363: `M[idx, 1] = -d_phase_d_f1 / f0_global`

2. Documentation:
   - `PIECEWISE_ACTUAL_FIX.md` - Explains the residual fix
   - `PIECEWISE_COMPLETE_FIX.md` - This document (both bugs)
   - `test_piecewise_fixed.py` - Validation script

## Key Principle

**Phase scale consistency is essential:**

When you apply a coordinate transformation (like shifting PEPOCH), you must carefully track which reference frame your phase is in and use the corresponding normalization frequency consistently throughout.

- Local coordinates: Used for **computation** (better numerical precision)
- Global coordinates: Used for **interpretation** (consistent reference)
- After phase offset correction, everything is back in global frame
- Therefore use f0_global for all phase-to-time conversions
