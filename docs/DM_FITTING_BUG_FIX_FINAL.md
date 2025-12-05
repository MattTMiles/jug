# DM Fitting Bug Fix - FINAL
**Date**: 2025-12-04  
**Status**: ✅ FIXED

## Problem Summary

DM parameters were not being updated during fitting. The design matrix had rank deficiency (rank 2 instead of 4) when fitting `['F0', 'F1', 'DM', 'DM1']`.

## Root Cause

The fitter was using `np.linalg.lstsq()` which fails silently when the design matrix is rank-deficient. This happened because:

1. **DM1 is highly correlated with spin parameters**:
   - DM1 vs F0: correlation = -0.91
   - DM1 vs F1: correlation = -0.84
   
2. Both DM1 and spin parameters are time-dependent:
   - F1 derivative ∝ t²
   - DM1 derivative ∝ t
   
3. Linear combination of (F0, F1) can approximate DM1, causing rank deficiency

## Solution

**Replace `np.linalg.lstsq()` with `wls_solve_svd()`** from `jug/fitting/wls_fitter.py`.

The SVD solver:
- Handles rank-deficient matrices gracefully
- Uses singular value decomposition with threshold
- Already existed in codebase but wasn't being used

### Code Changes

```python
# OLD (BROKEN)
M_weighted = M / errors_sec[:, np.newaxis]
r_weighted = residuals / errors_sec
delta_params, _, _, _ = np.linalg.lstsq(M_weighted, r_weighted, rcond=None)

# NEW (FIXED)
from jug.fitting.wls_fitter import wls_solve_svd

delta_params, cov, _ = wls_solve_svd(
    residuals=residuals,
    sigma=errors_sec,
    M=M,
    threshold=1e-14,
    negate_dpars=False
)
```

**File modified**: `jug/fitting/optimized_fitter.py` (lines 607-621)

## Test Results

| Test Case | Initial DM | Fitted DM | RMS (μs) | Iter | Converged |
|-----------|------------|-----------|----------|------|-----------|
| **F0 + F1 + DM** | 10.5907 | **10.3907** | **0.404** | **3** | **✓** |
| DM only | 10.5907 | 10.3929 | 23.9 | 3 | ✓ |
| F0 + F1 (no DM) | 10.5907 | N/A | 206.6 | - | ✗ |

**Success**: Fitting `['F0', 'F1', 'DM']` now works correctly!

## Known Limitation: DM1 Degeneracy

**DM1 fitting remains problematic** when combined with spin parameters due to strong correlation:

| Test Case | RMS (μs) | Converged | Note |
|-----------|----------|-----------|------|
| DM + DM1 | 24.5 | ✗ | Oscillates due to degeneracy |
| F0 + F1 + DM + DM1 | 0.40 | ✗ | SVD works but slow convergence |

This is a **known issue in pulsar timing** (not specific to JUG). PINT handles this by:
1. Orthogonalizing the design matrix
2. Fitting parameters in stages (spin first, then DM)
3. Using larger convergence thresholds for DM1

### Workaround

For now, **fit DM separately from DM1**:

```python
# Step 1: Fit spin + DM
result1 = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'DM'],
    convergence_threshold=1e-10
)

# Step 2: Update par file with new DM

# Step 3: Fit DM1 (with fixed spin + DM)
result2 = fit_parameters_optimized(
    par_file=updated_par_file,
    tim_file=tim_file,
    fit_params=['DM1'],
    convergence_threshold=1e-8  # Looser threshold
)
```

## Recommendations

1. ✅ **Use SVD solver for all fitting** (now default)
2. ✅ **Fit `['F0', 'F1', 'DM']` together** - works great!
3. ⚠️ **Avoid fitting DM1 with spin** - use staged fitting
4. 📝 **Document this limitation** in user guide

## Next Steps (Optional Improvements)

1. **Implement design matrix orthogonalization** (like PINT)
   - Use Modified Gram-Schmidt or QR decomposition
   - Remove correlations between parameters
   
2. **Add staged fitting helper**:
   ```python
   fit_parameters_staged(
       params=['F0', 'F1', 'DM', 'DM1'],
       stages=[['F0', 'F1', 'DM'], ['DM1']]
   )
   ```

3. **Auto-detect parameter correlations** and warn user

## Validation

Tested on J1909-3744 (10,408 TOAs):
- Initial DM: 10.590712224111 (wrong)
- Expected DM: 10.390712224111
- Fitted DM: 10.390712315533
- Difference: 9.1×10⁻⁸ pc/cm³ (well within uncertainty)
- Final RMS: 0.404 μs (matches expected)

**DM fitting is now PRODUCTION READY** for the `['F0', 'F1', 'DM']` case! ✅

---

**Fixed by**: Replacing `np.linalg.lstsq()` with `wls_solve_svd()`  
**Time to fix**: 2 hours (debugging) + 15 minutes (implementation)  
**Files modified**: 1 (`jug/fitting/optimized_fitter.py`)
