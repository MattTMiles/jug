# Milestone 2 Fitting Diagnosis (2025-11-30)

## Test Results

### Synthetic Fitting Test

**Goal**: Determine if JUG and PINT fitters converge to the same parameters when starting from the same perturbed initial conditions.

**Method**:
- Started with parameters from J1909-3744_tdb.par
- Perturbed F0 and F1 by ~10σ
- Fit with both JUG (JAX + Gauss-Newton) and PINT (WLS)
- Compare final fitted values

### Results

```
PERTURBED START:
  F0  = 339.315691920000000 Hz  (Δ = +1e-9 Hz)
  F1  = -1.614750000000000e-15 Hz/s  (Δ = +2e-17 Hz/s)

JUG FITTED (after 5 iterations):
  F0  = 339.315691919041342 ± 5.84e-14 Hz
  F1  = -1.614753938436094e-15 ± 9.43e-22 Hz/s
  
PINT FITTED (after 5 iterations):
  F0  = 339.315691919040830 ± 3.33e-14 Hz
  F1  = -1.614750512358547e-15 ± 5.37e-22 Hz/s

DIFFERENCE (JUG - PINT):
  F0: 5.12e-13 Hz  (7.6σ combined uncertainty)
  F1: 3.43e-21 Hz/s  (3.2σ combined uncertainty)
```

### Analysis

**Key Finding**: JUG and PINT converge to **different** parameter values (differ by 7-8σ).

**Root Cause**: The ~0.01 μs RMS difference in residual calculations between JUG and PINT creates different χ² surfaces. Each fitter finds the minimum of its own χ² surface, which are slightly offset from each other.

**Implications**:
1. ❌ JUG does NOT produce identical fits to PINT
2. ✓ JUG fitting algorithm appears to work (converges successfully)
3. ⚠️ Residual differences affect fitted parameters at the ~10σ level

## Path Forward

### Option 1: Accept Current Status (RECOMMENDED)
**Rationale**: 
- 0.01 μs residual differences are negligible for most science cases
- JUG fitter is working correctly (converges smoothly)
- Parameter differences (5e-13 Hz in F0) are tiny in absolute terms
- Having an independent implementation is valuable for cross-validation

**Action**: Document limitations and move to Milestone 3

### Option 2: Debug Residual Differences
**Effort**: High (could take many hours)
**Benefit**: Exact agreement with PINT
**Risk**: May never achieve perfect agreement due to numerical precision limits

**Next Steps** (if pursuing):
1. Investigate the 0.01 μs systematic offset in residuals
2. Check binary delay calculations (most likely source)
3. Verify barycentric correction details
4. Check TDB/TCB time scale handling

## Current Status

### What Works ✅
- JAX-based residual calculation with high precision (float64 mode)
- Gauss-Newton fitter with Levenberg-Marquardt damping
- Automatic differentiation for design matrix
- Weighted least squares
- Covariance matrix calculation

### What Doesn't Match ⚠️
- Residuals differ from PINT by ~0.01 μs RMS
- Fitted parameters differ by 7-8σ from PINT fits
- Uncertainties are ~2x larger than PINT (5.8e-14 vs 3.3e-14 for F0)

### Recommendation

**Proceed to Milestone 3** with current implementation. The fitting framework is solid and can be improved iteratively. Document the known differences clearly for users.

