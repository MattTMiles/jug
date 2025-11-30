# Milestone 2: Fitting Infrastructure - Session Status

**Date**: 2025-11-29
**Status**: In Progress (~20% complete)
**Time Invested**: ~1 hour

---

## What Was Completed

### 1. Module Structure Created ✅
```
jug/fitting/
  __init__.py          # Module exports
  params.py            # Parameter packing/unpacking/scaling (COMPLETE)
  chi2.py              # Chi-squared computation (PLACEHOLDER)
  core.py              # JAX residual function (STARTED)
```

### 2. Parameter Handling System ✅
- `extract_fittable_params()` - Extract parameters to fit based on FIT flags
- `pack_params()` - Convert dict → scaled array for optimization  
- `unpack_params()` - Convert scaled array → dict
- `PARAM_SCALES` - Scaling factors for all parameters (F0, F1, DM, binary, etc.)
- Support for CLI overrides (`--fit`, `--freeze`)

### 3. Synthetic Data Test ✅
- `test_fitting_synthetic.py` - Generate fake TOAs with known F0, F1
- Chi-squared function works
- JAX gradients compute correctly
- Optax optimizer integrated

---

## Current Problem

**Optimizer not converging** due to:
1. **Gradient scale mismatch**: F0 gradient ~ 1e12, F1 gradient ~ 1e20
2. **Parameter scales**: F0 ~ 100 Hz, F1 ~ 1e-15 Hz/s (15 orders of magnitude!)
3. **Learning rate**: Too small to make progress

**Solution needed**: Proper parameter scaling in optimization

---

## Next Steps

### Immediate (Next Session):

1. **Fix parameter scaling in synthetic test** (~30 min)
   ```python
   # Scale parameters before optimization
   x_scaled = pack_params({'F0': 100.0, 'F1': -1e-15}, ['F0', 'F1'])
   # x_scaled ≈ [1.0, -1.0] after scaling
   
   # Optimize in scaled space
   # Then unscale results
   ```

2. **Implement proper optimizer** (~30 min)
   - Add convergence checking (chi2 change < threshold)
   - Add progress printing (every N iterations)
   - Handle max iterations gracefully

3. **Validate on synthetic data** (~30 min)
   - Fit should recover F0, F1 to high precision
   - Chi-squared should be ~ N_toas
   - Uncertainties from Fisher matrix

### Medium Term:

4. **Integrate with real residuals** (~2-3 hours)
   - Refactor `compute_residuals_simple()` to separate:
     - `prepare_fitting_data()` - do I/O once
     - `compute_residuals_jax()` - pure JAX, can be differentiated
   - Test that existing code still works
   - Wire up to fitter

5. **Create CLI tool** (~1 hour)
   - `jug-fit` command
   - Parse FIT flags from .par file
   - Support `--fit` and `--freeze` overrides
   - Output fitted .par with uncertainties

6. **Test on J1909-3744** (~1 hour)
   - Fit F0, F1 only initially
   - Compare with PINT
   - Validate uncertainties

---

## Key Design Decisions Made

### 1. Parameter Scaling
All parameters scaled to O(1) for numerical stability:
```python
PARAM_SCALES = {
    'F0': 1e2,      # 100 Hz → 1.0
    'F1': 1e-15,    # 1e-15 Hz/s → 1.0
    'DM': 1e1,      # 10 pc/cm³ → 1.0
    ...
}
```

### 2. Parameter Masking
Three-level priority:
1. CLI `--fit` (highest priority, forces fit)
2. CLI `--freeze` (forces freeze)
3. .par FIT flags (default)

### 3. Optimization Strategy
- **Optimizer**: Optax Adam (adaptive learning rate)
- **Convergence**: Chi2 change < 1e-6 OR max iterations
- **Progress**: Print every 10-50 iterations

### 4. Two-Phase Approach
**Phase 1** (current): Synthetic data
- Validate optimizer works
- Debug scaling/convergence issues
- No dependency on existing code

**Phase 2** (next): Real data
- Refactor residual calculator
- Integrate with fitter
- Maintain backward compatibility

---

## Files Created

1. `jug/fitting/__init__.py` - Module exports
2. `jug/fitting/params.py` - Parameter handling (163 lines, COMPLETE)
3. `jug/fitting/chi2.py` - Chi2 computation (53 lines, PLACEHOLDER)
4. `jug/residuals/core.py` - JAX residual function (196 lines, STARTED)
5. `test_fitting_synthetic.py` - Synthetic test (155 lines, WORKING)

---

## Issues Encountered

### 1. Parameter Scaling Not Applied
The synthetic test doesn't use the scaling system yet. Parameters passed directly to optimizer, causing gradient mismatch.

**Fix**: Apply `pack_params()` before optimization, `unpack_params()` after.

### 2. Gradients Not Zero at Minimum
Even at true parameters, gradient is large (1e20). This is because:
- Weighted mean subtraction changes the chi2 surface
- Need to check if this is expected or a bug

**Investigation needed**: Is this correct behavior?

### 3. Learning Rate Too Small
Current LR = 1e-16 makes no progress. With proper scaling, LR should be ~1e-3 to 1e-2.

---

## Questions for Next Session

1. **Gradient behavior**: Is it normal for gradients to be non-zero at the minimum when using weighted mean subtraction?

2. **Convergence criteria**: Should I check:
   - Chi2 change < threshold?
   - Gradient norm < threshold?
   - Both?

3. **Fisher matrix**: Should uncertainties be computed from:
   - Hessian of chi2?
   - Covariance matrix (inverse Hessian)?
   - Standard errors (sqrt of diagonal of covariance)?

---

## Estimated Completion

- **Remaining for M2 fitting infrastructure**: 6-8 hours
- **Could complete in 1-2 more sessions** if focused

---

## Handoff Notes

**To continue**:
1. Fix parameter scaling in `test_fitting_synthetic.py`
2. Get optimizer converging on synthetic data
3. Then proceed with real data integration

**Key files to work on**:
- `test_fitting_synthetic.py` - Fix first
- `jug/fitting/optimizer.py` - Create next
- `jug/residuals/simple_calculator.py` - Refactor last

**Reference**: MK7 notebook doesn't have fitting code, so this is being built from scratch using JAX best practices.

---

**Session End**: 2025-11-29
**Next Session**: Continue with parameter scaling fix
