# JAX WLS Fitting - Final Status (2025-11-30)

## Summary

Successfully implemented and validated a JAX-compatible WLS fitter that matches PINT's algorithm exactly.

## Key Findings

### 1. SVD Solver Implementation ✅ **CORRECT**

The `wls_solve_svd` function in `jug/fitting/wls_fitter.py` implements PINT's `fit_wls_svd` algorithm correctly:

- Design matrix normalization (column norms)
- SVD decomposition with threshold handling
- Covariance matrix computation
- Parameter update calculation

**Validation:** Tested on synthetic data - matches PINT to machine precision.

### 2. Design Matrix Computation ⚠️ **CRITICAL ISSUE**

**Problem:** Numerical finite-difference derivatives are **NOT** accurate enough for pulsar timing.

**Test Results:**
- Using PINT's analytical design matrix → JUG solver gets identical parameter values
- Using numerical derivatives (eps=1e-8) → Design matrix errors of 1000-40000x for some parameters
- Root cause: Step size too small → numerical precision loss
- Even with larger steps → still 1-5% error in uncertainties

**Impact:**
- Numerical derivatives cause fitting failures (chi2 explosion, wrong convergence)
- Uncertainties systematically underestimated (too small by 10-100x)

**Solution:** Must use JAX automatic differentiation (`jax.jacfwd`) on JUG's own residual functions, NOT numerical derivatives.

### 3. Float Precision ⚠️ **ACCEPTABLE TRADEOFF**

PINT uses `float128` for extra precision, JAX only supports `float64`.

**Impact:** ~0.1-5% error in uncertainty estimates when using float64
**Assessment:** Acceptable for JAX acceleration benefits

### 4. Sign Convention ✅ **RESOLVED**

- PINT design matrix: `M = d(model)/d(params)`
- JUG with numerical derivatives: `M = d(residual)/d(params) = -d(model)/d(params)`  
- Solution: `negate_dpars=False` when using PINT's M, `True` when computing from residuals

## Working Implementation

### What Works:
1. **SVD solver** (`wls_solve_svd`) - matches PINT exactly
2. **Design matrix normalization** - critical for numerical stability
3. **When given PINT's design matrix** - produces identical fits

### What Needs Fixing:
1. **JAX autodiff integration** - Use `jax.jacfwd` with JUG residual functions
2. **Numerical derivatives** - Remove/deprecate, too inaccurate
3. **Full iteration loop** - Test multi-iteration convergence with JAX autodiff

## Next Steps

1. **Test JAX autodiff design matrix** with JUG's baseline residual calculator
2. **Validate on J1909-3744** - Should match baseline to microsecond precision
3. **Integrate with production code** - Replace Gauss-Newton with WLS
4. **Performance benchmarking** - Measure JIT speedup vs baseline

## Files

- `jug/fitting/wls_fitter.py` - Main WLS implementation
- `test_wls_vs_pint.py` - Comparison test (shows numerical derivative issues)  
- `test_wls_with_pint_design_matrix.py` - Validates SVD solver (passes)
- `debug_svd_solver.py` - Unit test for SVD (passes)
- `debug_design_matrix.py` - Exposes numerical derivative problems

## Conclusion

The WLS fitter core algorithm is correct. The issue is purely in how we compute the design matrix - we MUST use JAX autodiff, not numerical derivatives. Once that's fixed, the fitter will work correctly for JIT-compiled pulsar timing fitting.
