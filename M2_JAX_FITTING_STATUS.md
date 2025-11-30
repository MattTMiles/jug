# M2 JAX Fitting Status

**Date**: 2025-11-30  
**Session**: Post-DD model fixes

## Current Situation

### What Works ✅
1. **Binary models implemented**: DD, DDH, DDK, DDGR, ELL1, ELL1H all working
2. **Residual computation**: `compute_residuals_simple()` produces accurate residuals
3. **JAX infrastructure**: Design matrix computation with `jax.grad` implemented
4. **Gauss-Newton fitter**: `gauss_newton_fit_auto()` framework exists

### What Doesn't Work ❌
1. **Fitting infrastructure** - The fitter cannot converge because:
   - Residual function calls `compute_residuals_simple()` which does file I/O
   - Each residual evaluation takes ~2-3 seconds (parses files, loads clocks, etc.)
   - Design matrix needs 2N+1 evaluations per iteration (N=number of fit params)
   - For 2 parameters: 5 evaluations × 3 sec = 15 sec per iteration
   - Fit needs ~10-20 iterations → 2.5-5 minutes per fit
   - **Too slow for practical use**

2. **Design matrix mismatch** - Numerical derivatives don't match residual function:
   - `gauss_newton_fit_auto` uses JAX automatic differentiation
   - But residual function is not JAX-compatible (uses file I/O, external calls)
   - Derivatives are computed incorrectly
   - Result: Fitter rejects all steps and doesn't converge

## Root Cause

The fitting infrastructure expects a **pure JAX function**:
```python
def residuals_jax(params_array, toas_mjd, freq_mhz):
    """Pure JAX function - no file I/O, all JIT-compiled."""
    return residuals_sec  # shape: (n_toas,)
```

But we're providing a **file-based wrapper**:
```python
def compute_residuals_for_fit(params_dict):
    """Slow - writes temp file, parses, loads data."""
    write_temp_par_file(params_dict)
    result = compute_residuals_simple(temp_par, tim_file)
    return result['residuals_us'] / 1e6
```

## Solution Path

We need to create a **JAX-native residual function** that:
1. Takes parameters as JAX arrays (not files)
2. Has no file I/O inside
3. Can be JIT-compiled
4. Computes all delays in JAX

### Option A: Refactor existing code into JAX function ⭐ **RECOMMENDED**

Extract the core computation from `simple_calculator.py` into a pure JAX function:

```python
@jax.jit
def compute_residuals_jax(
    params_array,      # [F0, F1, DM, ...] - fitted parameters
    param_names,       # ['F0', 'F1', 'DM', ...] - which params
    fixed_params,      # dict of non-fitted params
    toas_tdb_mjd,      # Pre-computed TDB times
    freq_mhz,          # Observing frequencies
    # Pre-computed delays that don't depend on fit params:
    geometric_delay_sec,   # Roemer + Shapiro (from position)
    binary_delay_sec       # Binary delays (if not fitting binary params)
):
    # Build full params dict
    params = dict(fixed_params)
    for i, name in enumerate(param_names):
        params[name] = params_array[i]
    
    # Compute spin phase
    phase = spin_phase_jax(toas_tdb_mjd, params)
    
    # Compute DM delay
    dm_delay = dm_delay_jax(freq_mhz, params)
    
    # Total delay
    total_delay = geometric_delay_sec + binary_delay_sec + dm_delay
    
    # Residuals
    observed_phase = phase_offset + (toas_tdb_mjd - total_delay - PEPOCH) * F0 + ...
    model_phase = phase
    residuals = (observed_phase - model_phase) / F0
    
    return residuals
```

**Steps**:
1. Pre-compute TDB times (one-time, before fitting)
2. Pre-compute position-dependent delays (one-time)
3. Create JAX function that only computes spin/DM parts
4. Use this with existing `gauss_newton_fit_auto`

### Option B: Use PINT's fitter (temporary workaround)

For now, use PINT to fit parameters, then verify JUG residuals match PINT's fitted model.

**Pros**: Gets us fitting capability immediately  
**Cons**: Defeats purpose of PINT-free implementation

### Option C: Simplified fitting for testing

Create a minimal F0/F1 only fitter with analytic derivatives:

```python
# For phase residuals: r = (observed_phase - model_phase) / F0
# where model_phase = F0 * dt + 0.5 * F1 * dt^2
#
# Derivatives:
# dr/dF0 = -dt - r/F0
# dr/dF1 = -0.5 * dt^2 / F0
```

This would work immediately and prove the Gauss-Newton infrastructure.

## Recommendation

**Go with Option A** - Refactor into JAX function.

This is necessary for JUG to be a complete PINT replacement. The work is:

1. **Phase 1**: Extract TDB computation as separate pre-processing step
2. **Phase 2**: Extract geometric delays (Roemer+Shapiro) as pre-processing  
3. **Phase 3**: Create `residuals_jax_core()` that takes pre-computed delays
4. **Phase 4**: Integrate with `gauss_newton_fit_auto`
5. **Phase 5**: Test on J1909-3744

**Estimated effort**: 2-3 hours of focused refactoring

## Current Test Results

From `test_fitting_simple.py`:
- Initial RMS (true params): 0.417 μs ✅
- Perturbed RMS: 860 μs (expected - big perturbation)
- Fitted RMS: 860 μs ❌ (no improvement - fit failed)
- F0 recovery: 334696547σ off ❌
- F1 recovery: 987σ off ❌
- Convergence: Failed (rejected all steps) ❌

## Next Steps

1. ✅ Document current status (this file)
2. ⏸️ Decide on Option A, B, or C
3. ⏸️ Implement chosen approach
4. ⏸️ Validate on synthetic data
5. ⏸️ Test on J1909-3744
6. ⏸️ Update progress tracker

## References

- Design matrix implementation: `jug/fitting/design_matrix_jax.py`
- Gauss-Newton fitter: `jug/fitting/gauss_newton_jax.py`
- Current residual calculator: `jug/residuals/simple_calculator.py`
- Test script: `test_fitting_simple.py`
