# Piecewise Fitting Fix - Summary

**Date:** 2025-12-02  
**Issue:** Piecewise residuals were incorrect in initial implementation  
**Root Cause:** Missing phase offset correction when shifting PEPOCH

## The Problem

When implementing piecewise PEPOCH fitting, the initial formula was:

```python
dt_epoch = (local_pepoch_mjd - pepoch_global_mjd) * SECS_PER_DAY
f0_local = f0_global + f1_global * dt_epoch
dt_local = dt_sec_global - dt_epoch

phase = dt_local * (f0_local + dt_local * (f1_global / 2.0))
```

This gave residuals with RMS ~682 μs instead of ~0.8 μs!

## Mathematical Analysis

The global phase is:
```
φ_global = F0_global × dt_global + (F1_global/2) × dt_global²
```

The local phase with continuity constraints:
```
F0_local = F0_global + F1_global × dt_epoch
dt_local = dt_global - dt_epoch

φ_local = F0_local × dt_local + (F1_global/2) × dt_local²
```

Expanding the local phase:
```
φ_local = (F0_global + F1_global × dt_epoch) × (dt_global - dt_epoch)
          + (F1_global/2) × (dt_global - dt_epoch)²
          
        = F0_global × dt_global + (F1_global/2) × dt_global²
          - F0_global × dt_epoch - (F1_global/2) × dt_epoch²
          
        = φ_global - [F0_global × dt_epoch + (F1_global/2) × dt_epoch²]
```

**Key insight:** φ_local ≠ φ_global! There's a constant phase offset equal to the phase accumulated between the two PEPOCH choices.

## The Solution

To get correct residuals, we must add back the phase offset:

```python
dt_epoch = (local_pepoch_mjd - pepoch_global_mjd) * SECS_PER_DAY
f0_local = f0_global + f1_global * dt_epoch
dt_local = dt_sec_global - dt_epoch

# Compute local phase
phase_local = dt_local * (f0_local + dt_local * (f1_global / 2.0))

# Add phase offset to restore global phase reference
phase_offset = f0_global * dt_epoch + (f1_global / 2.0) * dt_epoch**2
phase_corrected = phase_local + phase_offset

# Wrap and convert to residuals
phase_wrapped = phase_corrected - np.round(phase_corrected)
residuals_sec = phase_wrapped / f0_local
```

## Validation

With the corrected formula:
- Global method: RMS = 0.817 μs
- Piecewise method: RMS = 0.817 μs  
- Difference: ~45 ns (float64 rounding error - acceptable)

The piecewise method now produces identical residuals to the global method, as expected mathematically.

## Implementation Notes

1. The phase offset correction **must** be applied before wrapping
2. Use `f0_local` (not `f0_global`) for converting phase to time residuals
3. The ~45 ns float64 error is negligible compared to typical TOA uncertainties (>100 ns)
4. For maximum precision on very long baselines, could use longdouble for phase_offset calculation

## Updated Notebook

The notebook `piecewise_fitting_implementation.ipynb` needs this correction in step 2 (compute_residuals_piecewise function).
