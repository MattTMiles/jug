# Session 14 Quick Reference

**Date**: 2025-12-01  
**Duration**: ~2 hours  
**Achievement**: Multi-parameter fitting (F0 + F1 simultaneous)

---

## What We Accomplished

✅ Successfully implemented and validated **simultaneous multi-parameter fitting**
- F0 + F1 fitted together in one solve
- Converges to match Tempo2 with sub-nanoHertz precision
- Final RMS: 0.914 μs (from initial 24 μs)

---

## Critical Fixes Applied

### 1. Units Mismatch (MAJOR)
**Problem**: Residuals in microseconds, derivatives in seconds/Hz  
**Solution**: Convert residuals to seconds before WLS solver  
```python
res_s = res_us * 1e-6
errors_s = errors_us * 1e-6
delta_params, _, _ = wls_solve_svd(res_s, errors_s, M)
```

### 2. Function Argument Order
**Problem**: Called `wls_solve_svd(M, res, err)` instead of `(res, err, M)`  
**Solution**: Fix to match function signature  
```python
# WRONG
delta_params = wls_solve_svd(M, res_us, toa_errs_us)

# CORRECT
delta_params, _, _ = wls_solve_svd(res_us, toa_errs_us, M)
```

### 3. Return Value Unpacking
**Problem**: `wls_solve_svd` returns 3-tuple `(dpars, Sigma, Adiag)`  
**Solution**: Unpack properly  
```python
delta_params, _, _ = wls_solve_svd(res_s, errors_s, M)
```

---

## Test Results

### Input
- **Pulsar**: J1909-3744 (10,408 TOAs)
- **Wrong F0**: 339.315691919050039 Hz
- **Wrong F1**: -1.612750151003891e-15 Hz/s
- **Target F0**: 339.315691919040830 Hz (Tempo2 refit)
- **Target F1**: -1.614750151808481e-15 Hz/s (Tempo2 refit)

### Convergence
```
Iter 1: RMS=22.837 us, ΔF0=9.579e-12 Hz, ΔF1=-1.342e-18 Hz/s
Iter 2: RMS=10.271 us, ΔF0=-1.043e-11 Hz, ΔF1=-3.648e-19 Hz/s
Iter 3: RMS=4.645 us,  ΔF0=-4.660e-12 Hz, ΔF1=-1.626e-19 Hz/s
Iter 4: RMS=2.200 us,  ΔF0=-2.069e-12 Hz, ΔF1=-7.263e-20 Hz/s
Iter 5: RMS=1.230 us,  ΔF0=-9.420e-13 Hz, ΔF1=-3.226e-20 Hz/s
Converged in 5 iterations
```

### Final Results
```
Fitted F0: 339.315691919041569 Hz
Fitted F1: -1.614724289086383e-15 Hz/s
Final RMS: 0.914 μs

Comparison to Tempo2:
  ΔF0 = 7.390e-13 Hz  < 1e-12 Hz  ✓ PASS
  ΔF1 = 2.586e-20 Hz/s < 1e-19 Hz/s ✓ PASS
```

---

## How Multi-Parameter Fitting Works

```python
def fit_f0_f1(params, par_file, tim_file):
    """Fit F0 and F1 simultaneously."""
    f0, f1 = params['F0'], params['F1']
    
    for iteration in range(max_iter):
        # 1. Update parameters
        params_iter = params.copy()
        params_iter['F0'] = f0
        params_iter['F1'] = f1
        
        # 2. Compute residuals (with TZR recomputation)
        res_us, errors_us = compute_residuals_simple(...)
        res_s = res_us * 1e-6  # Convert to seconds
        errors_s = errors_us * 1e-6
        
        # 3. Compute derivatives
        derivs = compute_spin_derivatives(params_iter, toas_mjd, ['F0', 'F1'])
        M = np.column_stack([derivs['F0'], derivs['F1']])
        
        # 4. Solve WLS
        delta_params, _, _ = wls_solve_svd(res_s, errors_s, M)
        
        # 5. Update
        f0 += delta_params[0]
        f1 += delta_params[1]
        
        # 6. Check convergence
        if np.linalg.norm(delta_params) < 1e-12:
            break
    
    return f0, f1
```

---

## Files Created/Modified

### New Files
- `test_f0_f1_fitting.py` - Multi-parameter test
- `FITTING_SUCCESS_MULTI_PARAM.md` - Comprehensive report
- `QUICK_REFERENCE_SESSION_14.md` - This file

### Updated Files
- `JUG_PROGRESS_TRACKER.md` - Added multi-param success
- `JUG_implementation_guide.md` - Updated M2 status

### Unchanged (Already Working)
- `jug/fitting/derivatives_spin.py`
- `jug/fitting/wls_fitter.py`
- `jug/residuals/simple_calculator.py`

---

## Key Insights

1. **Units are critical**: Derivatives in seconds/Hz require residuals in seconds
2. **Function signatures matter**: Check argument order carefully
3. **TZR updates automatically**: Full residual recomputation handles it
4. **Design matrix assembly**: `np.column_stack()` for multiple parameters
5. **Convergence is fast**: 5 iterations typical for well-conditioned problems

---

## Next Steps (User Questions)

### Question 1: Can we use Gauss-Newton?
**Answer**: Not yet needed! WLS with SVD IS the Gauss-Newton method for linear problems (like spin parameters). We only need iterative solvers (Levenberg-Marquardt) for non-linear parameters (binary, astrometry).

### Question 2: Can we speed this up with JAX?
**Answer**: Potentially! Three optimization opportunities:
1. **JAX-accelerated WLS solver** (JIT compilation)
2. **JAX derivatives** (autodiff instead of analytical)  
3. **Batch residual computation** (vectorize over iterations)

However, the current bottleneck is residual computation (~8-10 sec/iteration), not the fitting itself (~100ms). JAX won't help much until we have:
- Many iterations (>100)
- Many parameters (>20)
- Real-time fitting requirements

For now, the numpy implementation is fast enough!

---

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| F0 fitting | ✅ COMPLETE | Exact match to Tempo2 |
| F0+F1 fitting | ✅ COMPLETE | Sub-nanoHertz precision |
| DM derivatives | ⏸️ NOT STARTED | Next priority |
| Astrometry derivatives | ⏸️ NOT STARTED | After DM |
| Binary derivatives | ⏸️ NOT STARTED | After astrometry |
| JAX acceleration | ⏸️ NOT STARTED | Low priority |

**Overall**: Milestone 2 is **COMPLETE** ✅

---

## Documentation

Full documentation in:
- `FITTING_SUCCESS_MULTI_PARAM.md` - Technical details
- `JUG_PROGRESS_TRACKER.md` - Project status
- `JUG_implementation_guide.md` - Implementation roadmap
- `test_f0_f1_fitting.py` - Working code example

---

**Recommendation**: Move to DM derivatives next (Milestone 3) to enable full timing model fitting.
