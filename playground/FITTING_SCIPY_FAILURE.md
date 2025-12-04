# Fitting Test Results - scipy.optimize FAILED

**Date**: 2025-12-01  
**Test**: End-to-end fitting using scipy.optimize.least_squares
**Result**: ❌ **FAILED** - Optimizer converged to wrong values

## Test Setup

- **Pulsar**: J1909-3744 (10,408 TOAs)
- **Reference RMS**: 0.404 μs  
- **Parameters**: F0, F1, DM
- **Perturbation**: F0 +0.01%, F1 +1%, DM +0.1%
- **Perturbed RMS**: 854.9 μs
- **Optimizer**: scipy.optimize.least_squares (Levenberg-Marquardt)

## Results

### Convergence
- ✅ Optimizer reported convergence ("xtol satisfied")
- ❌ BUT converged to WRONG values

### RMS
- Reference: 0.404 μs
- Perturbed: 854.9 μs  
- **Fitted: 813.9 μs** ⚠️ Still 2000× worse than reference!

### Parameter Recovery

| Parameter | Reference | Fitted | Difference |
|-----------|-----------|--------|------------|
| F0 (Hz) | 3.39315692e+02 | 3.39315728e+02 | 3.64e-05 (107 ppb) |
| F1 (Hz/s) | -1.61475007e-15 | -6.61741949e-09 | **6.62e-09** (400 million %!) |
| DM (pc/cm³) | 10.3907122 | 10.3990295 | 8.32e-03 (800 ppm) |

**F1 is off by ~4 BILLION times the correct value!**

## Interpretation

### What Went Wrong

The optimizer thinks it converged, but it's at a completely wrong solution:
- F1 went from -1.6e-15 to -6.6e-09 (wrong by 4000×!)
- RMS is 813 μs instead of 0.4 μs (wrong by 2000×!)

This is **NOT** a precision issue - if it were, the fitted values would be close but slightly off. Instead, they're wildly wrong.

### Possible Causes

#### 1. Numerical Derivatives are Too Noisy
scipy.optimize uses finite differences:
```python
df/dx ≈ (f(x + h) - f(x)) / h
```

For F1 ~ 1e-15, choosing step size `h` is tricky:
- Too large: Misses the true derivative
- Too small: Float64 cancellation errors

This is EXACTLY the issue PINT encountered that made them use float128!

#### 2. Problem is Highly Non-linear
- Spin phase accumulates over 6+ years  
- Total phase ≈ 6.4×10¹⁰ cycles
- Small F1 error → huge phase error
- Numerical derivatives fail

#### 3. Multiple Local Minima
The optimizer may have gotten stuck in a local minimum where:
- RMS is "reasonably small" (813 μs)
- But parameters are completely wrong
- Gradient appears small due to numerical noise

## This Validates the Original Concern

**The conversation you showed me was RIGHT**:
> "Float64 might not be sufficient for GW-level precision without very careful numerics"

**We've confirmed**:
- ✅ JAX residual precision is excellent (0.02 μs vs PINT)
- ✅ Design matrix can be computed with JAX autodiff
- ❌ scipy.optimize with numerical derivatives FAILS
- ❌ Float64 alone is NOT sufficient without proper derivatives

## What This Means

### Option A Status: ⚠️ **PARTIALLY VALIDATED**

**What works**:
- JAX float64 residuals ✅ (0.02 μs precision)
- JAX autodiff for design matrix ✅ (tested on toy model)

**What doesn't work**:
- scipy.optimize with numerical derivatives ❌
- Fitting without analytical derivatives ❌

### The Real Options

We need to reconsider the options:

**Option A-Modified: JAX Autodiff (Required)**
```python
# MUST use JAX autodiff, not numerical derivatives
M = jax.jacfwd(residuals_jax)(params)  # Analytical!
```
- Float64 is fine IF we use analytical derivatives
- Cannot use numerical derivatives from scipy

**Option B: Hybrid longdouble/JAX**
- Delays in longdouble (high precision)
- Only fit spin params with JAX
- Limited to F0/F1/F2/F3 fitting

**Option C: Full analytical derivatives (Tempo2 approach)**
- Hand-code all derivatives
- Months of work

## Recommendation

**We MUST use JAX autodiff for the design matrix.**

The test shows that:
1. Float64 residuals are fine (0.4 μs achieved)
2. But numerical derivatives don't work
3. Need analytical derivatives via JAX autodiff

**Next step**: Implement full integration with JAX autodiff, NOT scipy's numerical derivatives.

This will require:
1. Making JUG residual calculator fully JAX-compatible
2. Using `jax.jacfwd` to compute design matrix
3. Using our WLS solver (already implemented)

**Timeline**: 4-6 hours to properly implement

## Files

- `test_scipy_fitting_validation.py` - Test script (shows failure)
- `FITTING_SCIPY_FAILURE.md` - This document

## Conclusion

**scipy.optimize alone is NOT sufficient.** We need JAX autodiff for analytical derivatives. Float64 is fine, but only with proper derivative computation, not finite differences.

The original conversation was correct about the challenge, just not about the solution (float128 vs analytical derivatives).
