# Piecewise Fitting - ACTUAL Fix

**Date:** 2025-12-02  
**Issue:** Quadratic error (~20 μs peak-to-peak) in piecewise residuals  
**Root Cause:** Using `f0_local` instead of `f0_global` for phase-to-time conversion

## The Bug

After adding the phase offset correction, the code was:

```python
phase_corrected = phase_local + phase_offset  # Now on GLOBAL phase scale
phase_wrapped = phase_corrected - np.round(phase_corrected)
residuals_sec[idx] = phase_wrapped / f0_local  # BUG: Should be f0_global!
```

## Why This Causes a Quadratic Error

After the phase offset correction, `phase_corrected` is on the **global phase scale**, so it must be divided by **f0_global** to get time residuals.

Using `f0_local` instead creates:
```
residual_piecewise = phase_wrapped / (f0_global + f1_global × dt_epoch)
residual_standard  = phase_wrapped / f0_global

Error = residual_piecewise - residual_standard
      = phase_wrapped × [1/(f0_global + f1 × dt_epoch) - 1/f0_global]
      ≈ -phase_wrapped × (f1 × dt_epoch) / f0_global²
```

Since `dt_epoch` varies across segments (and correlates with time), this creates a **curved pattern** that looks quadratic over the data span.

## The Fix

```python
phase_corrected = phase_local + phase_offset
phase_wrapped = phase_corrected - np.round(phase_corrected)
residuals_sec[idx] = phase_wrapped / f0_global  # CORRECT!
```

## Validation

After the fix:
- Peak-to-peak difference: **70 ns** (was ~20-25 μs)
- Quadratic trend R²: **< 0.1** (was > 0.95)
- ✓ Residuals now match standard method within float64 precision

## Updated Notebook

The notebook `piecewise_fitting_implementation.ipynb` has been corrected on line 248:
- OLD: `residuals_sec[idx] = phase_wrapped / f0_local`
- NEW: `residuals_sec[idx] = phase_wrapped / f0_global`

## Key Lesson

**Phase scale consistency is critical:**
1. Compute phase in local coordinates for numerical precision
2. Add phase offset to restore global phase scale
3. **Convert to time using the global frequency** (not local!)

The local F0 is only used for computing the phase itself, NOT for the final phase-to-time conversion.
