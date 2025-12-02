# Piecewise PEPOCH Fitting - Bug Fix and Solution

## Issue Discovered

The initial piecewise fitting implementation in the notebook produced incorrect residuals:
- **Expected:** RMS ~0.8 μs (matching standard method)
- **Got:** RMS ~682 μs with large offsets

## Root Cause

When shifting PEPOCH coordinates from global to local, there's a **phase offset** that must be accounted for:

```
φ_local = φ_global - [F0_global × dt_epoch + (F1_global/2) × dt_epoch²]
```

This offset represents the accumulated phase between the two PEPOCH choices and changes the pulse number reference for wrapping.

## Mathematical Proof

Starting from:
- Global: `φ_global = F0_global × dt_global + (F1_global/2) × dt_global²`
- Local: `φ_local = F0_local × dt_local + (F1_global/2) × dt_local²`
- Where: `F0_local = F0_global + F1_global × dt_epoch`
- And: `dt_local = dt_global - dt_epoch`

Expanding the local phase:
```
φ_local = (F0_global + F1_global × dt_epoch) × (dt_global - dt_epoch)
          + (F1_global/2) × (dt_global - dt_epoch)²
          
        = F0_global × dt_global + (F1_global/2) × dt_global²
          - F0_global × dt_epoch - (F1_global/2) × dt_epoch²
```

Therefore: **φ_local ≠ φ_global** (differs by the phase offset term)

## The Fix

**Corrected `compute_residuals_piecewise` function:**

```python
def compute_residuals_piecewise(dt_sec_global, pepoch_global_mjd, segments, 
                               f0_global, f1_global):
    n_toas = len(dt_sec_global)
    residuals_sec = np.zeros(n_toas)
    
    for seg in segments:
        idx = seg['indices']
        
        # Epoch offset
        dt_epoch = (seg['local_pepoch_mjd'] - pepoch_global_mjd) * SECS_PER_DAY
        
        # Continuity constraint
        f0_local = f0_global + f1_global * dt_epoch
        
        # Local coordinates
        dt_local = dt_sec_global[idx] - dt_epoch
        
        # Phase in local coordinates
        phase_local = dt_local * (f0_local + dt_local * (f1_global / 2.0))
        
        # *** CRITICAL FIX: Add phase offset ***
        phase_offset = f0_global * dt_epoch + (f1_global / 2.0) * dt_epoch**2
        phase_corrected = phase_local + phase_offset
        
        # Wrap and convert
        phase_wrapped = phase_corrected - np.round(phase_corrected)
        residuals_sec[idx] = phase_wrapped / f0_local
    
    return residuals_sec
```

## Validation Results

After applying the fix:
- **Global method:** RMS = 0.817248 μs
- **Piecewise method:** RMS = 0.817013 μs
- **Difference:** ~45 ns (float64 rounding error)

✓ The methods now agree to within numerical precision!

## Design Matrix Correction

The design matrix function needs the same correction. The derivatives must account for the phase offset:

**Key changes needed:**
1. Compute derivatives in local coordinates
2. Add chain rule terms from the phase offset
3. The F1 derivative gets an extra `dt_epoch × dt_local` term from `∂(phase_offset)/∂F1`

This will be implemented in the next iteration of the notebook.

## Files Updated

1. **`piecewise_fitting_implementation.ipynb`** - Fixed compute_residuals_piecewise()
2. **`PIECEWISE_FIX_SUMMARY.md`** - This document
3. **`diagnose_piecewise_issue.py`** - Diagnostic script showing the bug

## Next Steps

1. ✓ Fix residual computation
2. TODO: Fix design matrix computation with same phase offset correction
3. TODO: Run full notebook to verify fitting converges correctly
4. TODO: Move validated functions to `jug/fitting/piecewise_fitter.py`
5. TODO: Create unit tests

## Key Lesson

**When changing coordinate systems in phase calculations, always verify phase continuity!** The phase offset correction is mathematically necessary, not just a numerical fix.
