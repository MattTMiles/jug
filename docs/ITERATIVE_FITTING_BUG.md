# Iterative Fitting Instability Bug

## Problem Statement

When fitting pulsar timing parameters in JUG's GUI, clicking "Fit" multiple times causes the fit to **deteriorate** instead of improve. The RMS increases, parameters diverge, and eventually the fit becomes unstable with NaN values.

**Expected behavior:** Like Tempo2, repeated fitting should converge to a better solution or remain stable at the optimal solution, with RMS either decreasing or remaining constant.

**Actual behavior:** 
- First fit: RMS ~0.406-0.407 μs
- Subsequent fits: RMS increases, parameters drift, eventually → NaN
- Tempo2 achieves: RMS ~0.403 μs and remains stable

## Reproduction Steps

1. Load test pulsar: `data/pulsars/J1909-3744_tdb.par` and `data/pulsars/J1909-3744.tim`
2. In GUI, select these parameters to fit:
   ```
   ['A1', 'DM', 'DM1', 'DM2', 'EPS1', 'EPS2', 'F0', 'F1', 'M2', 'PB', 'PBDOT', 'SINI', 'TASC']
   ```
3. Click "Fit" button
4. Observe RMS (should be ~0.406-0.407 μs)
5. Click "Fit" button again (without changing anything)
6. **BUG**: RMS increases instead of decreasing/staying constant
7. Continue clicking "Fit" → eventually diverges to NaN

## Context: What Changed

Recent changes that might be relevant:
1. Added binary parameter derivatives (`jug/fitting/derivatives_binary.py`)
2. Modified `GeneralFitSetup` to include `roemer_shapiro_sec` field
3. Modified `fit_parameters_optimized_cached` to compute binary derivatives
4. Added session parameter updates in `session.py` line 416

## Key Code Locations

### Entry Point (GUI)
```
jug/gui/workers/fit_worker.py:71
  → session.fit_parameters()
```

### Session Fitting
```
jug/engine/session.py:416
  def fit_parameters(self, params_to_fit):
      result = fit_parameters_optimized_cached(...)
      if result['success']:
          self.params.update(result['fitted_params'])  # ← Updates session params
```

### Main Fitting Logic
```
jug/fitting/optimized_fitter.py:1700
  fit_parameters_optimized_cached()
    → _run_general_fit_iterations()  (line 1339)
      → full_iteration_jax_general()  (line 437)
```

### Derivative Computation
```
jug/fitting/optimized_fitter.py:1339-1410 (_run_general_fit_iterations)
  - Lines 1362-1385: Compute spin derivatives
  - Lines 1387-1402: Compute DM derivatives  
  - Lines 1404-1410: Compute binary derivatives (NEW)
```

### Binary Derivatives (NEW CODE)
```
jug/fitting/derivatives_binary.py
  compute_binary_derivatives_ell1()
    → Computes derivatives for: PB, TASC, PBDOT, A1, EPS1, EPS2, M2, SINI
```

## Diagnostic Information

### What We Know
1. **First fit works correctly**:
   - RMS: ~0.406-0.407 μs
   - Parameters converge
   - Chi-squared reduces appropriately

2. **Session params ARE updated** after each fit (verified at line 420 in session.py)

3. **Residuals should be recomputed** from updated params, but something is wrong with how this happens

### What We Suspect

**Hypothesis 1: Cached data not invalidated**
- The `GeneralFitSetup` is created once and cached
- When parameters change, the setup's `roemer_shapiro_sec` array might not be recomputed
- Binary derivatives would then be computed against **stale** Roemer delays
- Location: `optimized_fitter.py:1679-1690` (setup creation)

**Hypothesis 2: Derivative sign issue**
- Binary derivatives might have wrong sign relative to residuals
- First iteration: moves in wrong direction but looks like it helps
- Second iteration: compounds the error → divergence
- Check: `derivatives_binary.py` all derivatives should match PINT/Tempo2 conventions

**Hypothesis 3: Residual computation path**
- Session might be using wrong residual computation between fits
- `session.py:272-287` has different code paths for residuals
- After fit, might be computing residuals from **original** par file instead of updated params

**Hypothesis 4: Delay caching in binary model**
- The binary model might cache intermediate results (orbital phase, etc.)
- When parameters change, these cached values aren't updated
- Check: `jug/model/binary/ell1.py` for any module-level caching

## Investigation Tasks

### Task 1: Verify residuals are recomputed correctly
```python
# Add debug output to session.py:420
print(f"Before fit: F0={self.params['F0']:.15f}, PB={self.params['PB']:.15f}")
result = fit_parameters_optimized_cached(...)
if result['success']:
    self.params.update(result['fitted_params'])
    print(f"After fit: F0={self.params['F0']:.15f}, PB={self.params['PB']:.15f}")
    
    # Compute residuals manually with new params
    test_resids = compute_residuals_simple(params=self.params, ...)
    print(f"New residuals RMS: {np.std(test_resids)*1e6:.3f} μs")
```

### Task 2: Check if GeneralFitSetup is stale
```python
# In optimized_fitter.py:1700, print setup data
setup = _prepare_general_fit_setup(...)
print(f"Setup F0: {setup.params['F0']:.15f}")
print(f"Setup PB: {setup.params['PB']:.15f}")
print(f"Setup roemer_shapiro_sec mean: {setup.roemer_shapiro_sec.mean()}")
```

### Task 3: Validate binary derivatives against PINT
```python
# Create test comparing JUG derivatives to PINT derivatives
# For SAME parameter values, check if derivatives match
# Run this BEFORE and AFTER a fit to see if they diverge
```

### Task 4: Check for any caching in binary model
```python
# Search for module-level variables or lru_cache in:
grep -r "@lru_cache" jug/model/binary/
grep -r "^[A-Z_].*=" jug/model/binary/  # module-level constants
```

### Task 5: Compare to non-binary fitting
```python
# Fit ONLY non-binary parameters: ['F0', 'F1', 'DM', 'DM1', 'DM2']
# Click "Fit" multiple times
# Does it diverge? Or stay stable?
# This isolates whether the bug is in binary derivatives or general fitting
```

## Test Data

Use the J1909-3744 dataset:
- Par: `/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par`
- Tim: `/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim`

Compare against:
- **Tempo2**: `tempo2 -f J1909-3744_tdb.par J1909-3744.tim -fit F0 F1 DM DM1 DM2 A1 PB ...`
- **PINT**: Use `pint_fit` or create design matrix comparison script

## Success Criteria

1. Clicking "Fit" multiple times should:
   - Keep RMS stable or decrease it slightly
   - Not cause parameters to diverge
   - Match Tempo2's final RMS (~0.403 μs)

2. Parameters should converge to same values as Tempo2/PINT

## Additional Notes

- JUG is JAX-based; minimize use of non-JAX code
- Binary derivatives were recently validated against Tempo2 and found to be **correct** (see `DERIVATIVE_PARITY.md`)
- The bug is NOT in the derivative formulas themselves, but in how they're used during iterative fitting
- Suspect the issue is in caching/state management between fit iterations

## Related Files

1. `jug/fitting/optimized_fitter.py` - Main fitting logic
2. `jug/engine/session.py` - Session management and parameter updates
3. `jug/fitting/derivatives_binary.py` - Binary parameter derivatives (NEW)
4. `jug/model/binary/ell1.py` - ELL1 binary model
5. `jug/gui/workers/fit_worker.py` - GUI interface to fitting
6. `docs/DERIVATIVE_PARITY.md` - Documentation of derivative validation

## Prompt for AI Agent

```
I'm debugging a pulsar timing software called JUG (JAX-based). There's a bug where 
repeated fitting makes the solution WORSE instead of better (RMS increases, parameters 
diverge to NaN).

Please read the full context in docs/ITERATIVE_FITTING_BUG.md, then:

1. Investigate the hypotheses listed there
2. Run the diagnostic tasks to isolate the bug
3. Fix the root cause with minimal code changes
4. Verify the fix works by:
   - Clicking "Fit" 5-10 times in a row
   - Confirming RMS stays stable or decreases
   - Confirming it matches Tempo2's final RMS (~0.403 μs)

Remember: JUG is JAX-based, so maintain JAX compatibility. The binary derivatives 
themselves are correct (validated against Tempo2), so the bug is likely in state 
management or caching between fit iterations.

Test data is in data/pulsars/J1909-3744_tdb.par and J1909-3744.tim.
```
