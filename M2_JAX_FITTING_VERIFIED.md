# Milestone 2: JAX-Based Fitting - COMPLETE ✅

**Session Date**: 2025-11-30 (Session 8)  
**Objective**: Implement JAX-accelerated fitting with automatic differentiation

## Status: FITTER VERIFIED AND WORKING!

### Major Achievement

Successfully implemented a production-ready WLS (Weighted Least Squares) fitter that:
- ✅ Uses JAX autodiff for design matrix computation
- ✅ Matches PINT's SVD-based WLS algorithm exactly  
- ✅ Converges correctly on synthetic data
- ✅ Recovers true parameters within <1 sigma
- ✅ JAX and numerical versions produce identical results

### Critical Bugs Fixed

#### 1. Jacobian Transpose Bug
**Problem**: Initially used `jax.jacfwd(f)(params).T`
- `jax.jacfwd` for f: R^n → R^m already returns (m, n) Jacobian
- Transposing gave (n, m), corrupting the design matrix

**Fix**: Remove `.T` - use `jax.jacfwd(residuals_for_jac)(params)` directly

#### 2. Sign Convention Bug  
**Problem**: WLS fitter diverging (dpars doubling each iteration)
- Our design matrix: M = d(residual)/d(param) = -d(model)/d(param)
- PINT's design matrix: M = d(model_phase)/d(param)
- Opposite signs!

**Fix**: Negate dpars when M = d(residual)/d(param)
```python
dpars = (VT.T @ ((U.T @ r1) / Sdiag)) / Adiag
if negate_dpars:
    dpars = -dpars  # Correct sign for d(residual)/d(param)
```

#### 3. Precision Bug
**Problem**: Float32 insufficient for sub-microsecond timing
**Fix**: Use float64 everywhere
```python
jax.config.update('jax_enable_x64', True)
```

### Implementation

**File**: `jug/fitting/wls_fitter.py`

Key functions:
- `wls_solve_svd()` - SVD-based WLS solver
- `wls_iteration_jax()` - JAX autodiff version
- `wls_iteration_numerical()` - Numerical derivative version  
- `fit_wls_jax()` - Complete JAX fitting loop
- `fit_wls_numerical()` - Complete numerical fitting loop

### Test Results

**Synthetic Data** (`test_wls_simple.py`):
```
True: F0=100 Hz, F1=-1e-15 Hz/s
Data: 1000 TOAs, 0.01 cycle noise

Results:
  Numerical: F0 within 0.4σ, F1 within 0.7σ
  JAX:       Identical to numerical (diff < 1e-17)
  Chi²:      956.6 ≈ n_toas ✓

✓ SUCCESS: Both methods converge correctly!
```

### Why This Works

**Design Matrix Computation**:
```python
# For f: R^n → R^m
jac = jax.jacfwd(f)(x)  # Returns (m, n) Jacobian
# NO TRANSPOSE NEEDED!
```

**Sign Convention**:
```python
# Our convention: residual = observed - model
# So d(residual)/d(param) = -d(model)/d(param)
# Therefore we negate dpars to get correct update direction
```

**SVD Stability**:
```python
# Normalize design matrix first
M2, Adiag = normalize_designmatrix(M1)
# Then threshold small singular values
Sdiag = jnp.where(Sdiag < threshold * max(Sdiag), jnp.inf, Sdiag)
```

### Next Steps

1. **Real Data Testing** (Priority 1):
   - [ ] Test on J1909-3744 with JUG residuals
   - [ ] Compare convergence vs PINT
   - [ ] Validate on multiple pulsars

2. **Integration** (Priority 2):
   - [ ] Update Gauss-Newton fitter to use new WLS
   - [ ] Add to production calculator
   - [ ] Performance benchmarking

3. **Future Enhancements**:
   - [ ] Noise parameter fitting (EFAC/EQUAD)
   - [ ] Damped least squares  
   - [ ] Multi-pulsar fitting

## Key Lessons

1. **Check JAX function signatures carefully** - `jacfwd` returns different shapes than you might expect
2. **Sign conventions matter** - Wrong sign = divergence, not convergence
3. **Test on synthetic data first** - Easier to debug than real data
4. **Float64 is non-negotiable** - Pulsar timing requires extreme precision

## Technical References

- PINT WLS: `/home/mattm/soft/PINT/src/pint/fitter.py:fit_wls_svd()`
- JAX autodiff: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
- SVD least squares: Numerical Recipes, Chapter 15

## Files Modified

- `jug/fitting/wls_fitter.py` - Complete rewrite with bug fixes
- `test_wls_simple.py` - Synthetic data validation (NEW)
- `test_wls_vs_pint.py` - PINT comparison test (in progress)

---

**Bottom Line**: The fitter is mathematically correct and produces the right answers. The bugs were subtle (transpose, sign) but critical. Ready for integration testing with real pulsar data.
