# Sign Convention Fix - December 1, 2025

**Status**: ✅ COMPLETED  
**Validation**: EXACT match with PINT maintained

---

## Problem Discovered

After Session 13's successful fitting validation, we discovered that the code worked due to **two sign errors canceling out**:

1. **First error**: `d_phase_d_F()` applied negative sign, then `compute_spin_derivatives()` applied another negative, resulting in **positive** design matrix
2. **Second error**: `wls_solve_svd()` had `negate_dpars=True` by default, negating the parameter updates
3. **Result**: Two wrongs made a right - fitting worked perfectly!

## PINT's Actual Convention

From PINT source code (`src/pint/models/timing_model.py` lines 2365-2368):

```python
# Line 2365: Apply negative to phase derivative
q = -self.d_phase_d_param(toas, delay, param)

# Line 2368: Divide by F0
M[:, ii] = q.to_value(the_unit) / F0.value
```

Where `d_phase_d_param()` returns **POSITIVE** phase derivatives, then:
- Negative sign applied in `designmatrix()`
- Result divided by F0
- Final design matrix: **NEGATIVE** values

## The Fix

### Changed Files

**1. `jug/fitting/derivatives_spin.py`**

**Before** (WRONG - double negative):
```python
def d_phase_d_F(...):
    derivative = taylor_horner(dt_sec, coeffs)
    derivative = -derivative  # ← Applied negative here
    return derivative

def compute_spin_derivatives(...):
    deriv_phase = d_phase_d_F(...)  # Already negative
    derivatives[param] = -deriv_phase / f0  # ← Another negative!
    # Result: -(-dt)/F0 = +dt/F0 (POSITIVE - wrong!)
```

**After** (CORRECT - matches PINT):
```python
def d_phase_d_F(...):
    derivative = taylor_horner(dt_sec, coeffs)
    return derivative  # ← Return POSITIVE (like PINT)

def compute_spin_derivatives(...):
    deriv_phase = d_phase_d_F(...)  # Positive
    derivatives[param] = -deriv_phase / f0  # ← Apply negative once
    # Result: -dt/F0 (NEGATIVE - correct!)
```

**2. `jug/fitting/wls_fitter.py`**

**Before**:
```python
def wls_solve_svd(..., negate_dpars: bool = True):
    # Misleading comment about convention difference
    dpars = ...
    if negate_dpars:
        dpars = -dpars  # ← This was compensating for wrong design matrix!
```

**After**:
```python
def wls_solve_svd(..., negate_dpars: bool = False):
    # Clear documentation about PINT convention
    # Design matrix M = -d(phase)/d(param) / F0 (NEGATIVE)
    # No negation needed for PINT-style design matrix
    dpars = ...
    if negate_dpars:
        dpars = -dpars  # Only for non-PINT conventions
```

### Updated Comments

All comments now clearly explain:
- `d_phase_d_F()` returns POSITIVE derivatives (matches PINT's `d_phase_d_F`)
- Negative sign applied in `compute_spin_derivatives()` (matches PINT's `designmatrix()`)
- `wls_solve_svd()` default is `negate_dpars=False` for PINT convention

## Validation

**Test**: Same J1909-3744 validation as Session 13

**Results**:
```
Design matrix mean:  -1.250e+05 s/Hz  ✅ (NEGATIVE, correct!)
Fitted F0:  339.31569191904083027111 Hz
Target F0:  339.31569191904083027111 Hz
Difference: 0.000e+00 Hz  ✅ EXACT MATCH!
```

**Before/After Sign Check**:

| Component | Before Fix | After Fix | Correct? |
|-----------|------------|-----------|----------|
| `d_phase_d_F()` output | -1000.0 | +1000.0 | ✅ |
| Design matrix M[F0] | +2.95 | -2.95 | ✅ |
| `negate_dpars` default | True | False | ✅ |
| Parameter updates | Correct (by luck) | Correct (by design) | ✅ |

## Why This Matters

**For next derivatives (DM, astrometry, binary)**:

1. **Clear convention**: All derivative functions return POSITIVE values
2. **Single negation point**: Only in `compute_*_derivatives()` functions
3. **No hidden compensations**: WLS solver doesn't need to fix sign errors
4. **Matches PINT exactly**: Easy to verify against PINT source code

## Code Pattern for Future Derivatives

```python
def d_<component>_d_<param>(inputs):
    """
    Compute derivative of <component> w.r.t. <param>.
    
    Returns: POSITIVE derivative (matches PINT convention)
    """
    # Compute derivative mathematically
    derivative = ...
    return derivative  # Return positive!

def compute_<component>_derivatives(params, toas, fit_params):
    """
    Build design matrix for <component> parameters.
    
    Returns: Design matrix with NEGATIVE derivatives (PINT convention)
    """
    derivatives = {}
    for param in fit_params:
        deriv = d_<component>_d_<param>(...)  # Get positive derivative
        derivatives[param] = -deriv  # Apply negative here!
    return derivatives
```

## Summary

**What changed**:
- Removed double-negative bug
- Matched PINT's convention exactly
- Updated all comments for clarity
- Changed `negate_dpars` default to False

**What stayed the same**:
- Fitting accuracy (still EXACT match!)
- Test results (validates perfectly)
- API (no breaking changes)

**Impact**:
- Code is now clearer and more maintainable
- Future derivatives will follow consistent pattern
- No hidden compensations or "magic" sign flips

---

**Bottom Line**: The code now does what it appears to do - no more relying on two wrongs to make a right! 🎯

---

## Additional Improvement: Adaptive Convergence Detection

**Date**: 2025-12-01 23:33 UTC

### Change Made

Updated `test_f0_fitting_tempo2_validation.py` to use **stagnation detection** instead of static threshold:

**Before** (static threshold):
```python
convergence_threshold = 1e-18  # Very tight, arbitrary
rel_change = abs(delta_params[0] / f0_curr)
if rel_change < convergence_threshold:
    converged = True
```

**After** (stagnation detection):
```python
min_iterations_for_stagnation = 3  # Need this many identical iterations
f0_history = []

# In loop:
f0_history.append(f0_new)
if len(f0_history) >= min_iterations_for_stagnation:
    recent_f0s = f0_history[-min_iterations_for_stagnation:]
    if all(f0 == recent_f0s[0] for f0 in recent_f0s):
        converged = True  # F0 literally stopped changing!
```

### Why This is Better

1. **Adaptive**: Detects when parameter stops changing at floating-point precision
2. **No arbitrary threshold**: Uses actual numerical stagnation
3. **Clearer**: "Converged when F0 unchanged for 3 iterations"
4. **More robust**: Works regardless of parameter scale or precision

### Results

Same test case (J1909-3744):
- **Iterations to convergence**: 9 (was 20 with threshold)
- **Final F0**: 339.31569191904083027111 Hz (EXACT match!)
- **Convergence message**: "F0 unchanged for 3 iterations" ✅

This pattern should be used for all future fitting tests!

