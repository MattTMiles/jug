# Milestone 2 Session 1 - Final Status

**Date**: 2025-11-29
**Time**: 2+ hours
**Status**: 30% Complete - Core Fitting Working on Synthetic Data

---

## ✅ Major Achievement

**Successfully implemented linearized least squares fitting** - the standard approach used in pulsar timing (TEMPO/PINT style).

### Key Insight Learned

Pulsar timing fitting is **NOT global optimization** - it's **local refinement** around an already-good model. This is why:
- Gradient descent struggled (trying to find global minimum)
- Linearized LSQ works (refining local solution)
- Initial guess must be close (within ~1 cycle)

---

## What Works

### 1. Linearized Fitting (✅ Validated)

**File**: `test_linearized_fitting.py`

Successfully fits F0 and F1 on synthetic data:
- Starts within 1 nHz of true value
- Converges in 2-3 iterations  
- Chi-squared ~ N_TOAs (good fit)
- Uncertainties correct (true values within 1σ)

**Test Results**:
```
True F0: 100.000000000000 Hz
Fitted:  100.000000000000 ± 8.24e-13 Hz
Error:   0.36 σ

True F1: -1.000000e-15 Hz/s
Fitted:  -9.999883e-16 ± 2.47e-20 Hz/s
Error:   0.47 σ

Chi2: 82.14 (expected ~100 for 100 TOAs)
✅ TEST PASSED
```

### 2. Infrastructure Created

**Files**:
- `jug/fitting/params.py` - Parameter handling, scaling, masking ✅
- `jug/fitting/optimizer.py` - Linearized LSQ fitter ✅  
- `jug/fitting/__init__.py` - Module exports ✅
- `test_linearized_fitting.py` - Validation test ✅

---

## What Doesn't Work Yet

###1. Numerical Derivatives in General Fitter

**Problem**: `fit_linearized()` tries to compute derivatives numerically by perturbing parameters. This doesn't work well for phase-wrapped residuals - small parameter changes cause huge residual changes due to wrapping.

**Solution**: Need analytical derivatives (design matrix) for each timing model component.

### 2. Integration with Real Residuals

The current `compute_residuals_simple()` function:
- Does I/O every call (slow)
- Not structured for fitting
- Need to separate: setup (once) vs. residual computation (many times)

**Required**: Refactor to enable fitting without breaking existing code.

---

## Next Steps (Priority Order)

### Immediate (Next Session):

1. **Create analytical design matrix** (~2 hours)
   - For each parameter (F0, F1, F2, DM, binary, etc.)
   - Compute ∂(residual)/∂(parameter) analytically
   - This is what TEMPO/PINT do

2. **Integrate with simple calculator** (~1 hour)
   - Extract precomputation logic
   - Create fitting-friendly residual function
   - Test existing code still works

3. **Build `jug-fit` CLI** (~1 hour)
   - Parse .par FIT flags
   - Support `--fit` / `--freeze` overrides
   - Output fitted .par with uncertainties

### Then:

4. **Test on J1909-3744** (~30 min)
   - Start with PINT/Tempo2 model
   - Fit F0, F1 only
   - Compare results

5. **Expand to all parameters** (~2 hours)
   - Add design matrix for DM, binary, astrometry
   - Test with more complex fits

---

## Key Files

### Working:
- `test_linearized_fitting.py` - Proof of concept ✅
- `jug/fitting/params.py` - Parameter system ✅
- `jug/fitting/optimizer.py` - Fitter skeleton ✅

### Need Work:
- `jug/fitting/optimizer.py` - Add analytical derivatives
- `jug/residuals/simple_calculator.py` - Refactor for fitting
- `jug/scripts/fit.py` - CLI tool (not created yet)

---

## Technical Notes

### Why Linearized LSQ?

In pulsar timing, we linearize:
```
residual(p + Δp) ≈ residual(p) + (∂residual/∂p) · Δp
```

Then solve:
```
A · Δp = -residual
where A = design matrix (derivatives)
```

This is:
- ✅ Fast (one matrix inversion per iteration)
- ✅ Stable (well-conditioned when close to solution)
- ✅ Standard (TEMPO/PINT do this)
- ✅ Gives uncertainties (covariance = (A^T A)^-1)

### Design Matrix Components

For each parameter, we need:
```python
∂(residual)/∂F0 = -dt / F0  # (converting phase to time)
∂(residual)/∂F1 = -0.5 * dt^2 / F0
∂(residual)/∂DM = (K_DM / freq^2) / F0
... etc for each parameter
```

---

## Estimated Completion

**Milestone 2 Progress**: 30%

**Remaining Work**:
- Analytical design matrix: 2-3 hours
- Integration: 2-3 hours  
- CLI tool: 1 hour
- Testing: 1 hour
- **Total**: 6-8 hours (1-2 more sessions)

---

## Lessons Learned

1. **Don't use gradient descent for phase-wrapped objectives** - the surface is too non-smooth

2. **Parameter scaling alone doesn't solve ill-conditioning** - need the right algorithm

3. **Pulsar timing is local refinement, not global search** - this fundamentally changes the approach

4. **Analytical derivatives are essential** - numerical derivatives don't work with phase wrapping

5. **Test on synthetic data first** - much easier to debug than real data

---

## For Next Session

**Start here**:
1. Look at PINT source code for design matrix computation
2. Implement analytical derivatives for F0, F1, F2
3. Test that fitter recovers parameters
4. Then integrate with real residuals

**Don't**:
- Try to use gradient descent / JAX autodiff
- Compute numerical derivatives  
- Start with real data before synthetic works

---

**Session Complete**: 2025-11-29  
**Next Session**: Implement analytical design matrix

**Status**: On track for M2 completion, just need analytical derivatives!
