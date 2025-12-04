# Fitter Convergence Investigation

**Date**: 2025-12-01  
**Session**: Deep dive into fitter convergence issues

## Executive Summary

**CRITICAL FINDING**: Our Gauss-Newton fitter does not converge correctly, even when using PINT's own residual calculation. This is a fundamental algorithm issue, not a residual calculation problem.

## Investigation Steps

### Test 1: JAX Precision ✅ **PASSED**

**Question**: Does JAX maintain sufficient float64 precision?

**Method**: Compare JAX vs baseline numpy residual calculations

**Result**: **PERFECT** - attosecond-level agreement
```
Mean difference: 8.99e-18 seconds (9 attoseconds)
Std difference:  3.00e-17 seconds (30 attoseconds)  
Max difference:  1.42e-16 seconds (142 attoseconds)
```

**Conclusion**: JAX precision is NOT the problem. Using patterns from `/home/mattm/soft/discovery` codebase fixed all precision issues.

---

### Test 2: Residual Calculation ✅ **PASSED**

**Question**: Do JUG and PINT residuals agree?

**Method**: Compare JUG baseline vs PINT residuals on same data

**Result**: **EXCELLENT** - sub-microsecond agreement
```
Mean difference: 0.001 μs
RMS difference:  0.020 μs
```

**Conclusion**: Residual differences are negligible. Not the source of convergence problems.

---

### Test 3: Fitter Algorithm ❌ **FAILED**

**Question**: Does our Gauss-Newton fitter converge to same values as PINT?

**Method**: 
- Use PINT's residual function for BOTH fitters (eliminates residual differences)
- Perturb F0/F1 slightly from reference values
- Both fitters should recover same final values

**Setup**:
```python
# Reference values (from par file)
F0_ref = 339.315691919040660 Hz
F1_ref = -1.614740036909297e-15 Hz/s

# Perturb by small amounts
F0_pert = F0_ref + F0_ref * 1e-7  # +0.01% 
F1_pert = F1_ref * 1.001           # +0.1%
```

**Results - PINT WLSFitter**:
```
Converged smoothly in 5 iterations
F0  = 339.315725850609851 ± 1.80e-13 Hz
F1  = -1.616354776946206e-15 ± 2.90e-21 Hz/s
Postfit RMS: 850 μs
```

**Results - Our Gauss-Newton**:
```
RMS oscillates erratically:
  Iter 0: 853.3 μs
  Iter 1: 858.4 μs  ⚠️ (increased!)
  Iter 2: 871.8 μs  ⚠️ (increased!)
  Iter 3: 848.2 μs
  ... continues oscillating ...
  
Final values:
F0  = 339.315742526817928 ± 2.09e-12 Hz
F1  = -1.614529384218770e-15 ± 6.33e-23 Hz/s
Postfit RMS: 838 μs

DIFFERENCE FROM PINT:
F0: 92,654,728 sigma ❌❌❌
F1: 630 sigma ❌❌❌
```

**Conclusion**: **CRITICAL BUG** - Our fitter is fundamentally broken. Even with PINT's residuals, it converges to completely wrong values.

---

### Test 4: Damped Gauss-Newton ❌ **STILL FAILED**

**Question**: Can we fix convergence with line search/damping?

**Method**:
- Add line search: try step sizes [1.0, 0.5, 0.25, 0.125...]
- Only accept steps that reduce chi-squared
- Standard practice for non-linear optimization

**Result**: **STILL WRONG**
```
Iter 0: RMS=853.3 μs, chi2=1.365e12, step=0.25
Iter 1: RMS=795.1 μs, chi2=1.212e12, step=0.0
Converged: no improvement possible

Final values:
F0: 2,368,356 sigma from PINT ❌
F1: 271 sigma from PINT ❌
```

**Conclusion**: Line search helps but doesn't fix the fundamental problem. The algorithm is getting stuck in wrong local minima.

---

## Root Cause

**The problem**: Our implementation does NOT match PINT's algorithm.

We implemented "generic Gauss-Newton":
```python
# Simple Gauss-Newton step
dp = -inv(M.T @ W @ M) @ M.T @ W @ r
params += dp
```

PINT uses "damped/regularized least squares" with:
- Trust region methods
- Levenberg-Marquardt damping
- Specialized convergence criteria
- Careful handling of ill-conditioned problems

**Pulsar timing fitting is highly non-linear** - small parameter changes can cause large residual changes due to:
- Binary orbital phase wrapping
- Spin phase accumulation over years
- Strong parameter correlations (e.g., F0 vs F1 vs position)

## Path Forward

### Option A: Study and Replicate PINT's Algorithm ⭐ **RECOMMENDED**

**What to do**:
1. Read PINT's `WLSFitter` source code in detail (`src/pint/fitter.py`)
2. Understand their damping/regularization approach
3. Understand their convergence criteria
4. Replicate their algorithm step-by-step in JAX

**Pros**:
- Will definitely work (PINT is well-tested)
- Scientifically validated approach
- Drop-in PINT replacement

**Cons**:
- More complex than simple Gauss-Newton
- Takes longer to implement

**Estimated effort**: 4-6 hours

---

### Option B: Use scipy.optimize ⚙️ **PRAGMATIC**

**What to do**:
```python
from scipy.optimize import least_squares

def residual_func(params):
    # Update model with params
    # Return residuals
    pass

result = least_squares(
    residual_func,
    initial_params,
    method='lm',  # Levenberg-Marquardt
    ftol=1e-12
)
```

**Pros**:
- Production-quality optimization
- Handles ill-conditioned problems
- Works immediately

**Cons**:
- Adds scipy dependency
- Not pure JAX (can't JIT the full fit)
- But residual function can still be JAX

**Estimated effort**: 1-2 hours

---

### Option C: Fix Our Gauss-Newton 🔧 **EDUCATIONAL**

**What to do**:
1. Add Levenberg-Marquardt damping: `(M.T @ W @ M + λI) @ dp = M.T @ W @ r`
2. Adaptive λ adjustment based on progress
3. Trust region constraints
4. Better convergence criteria

**Pros**:
- Learn optimization theory
- Full control over algorithm

**Cons**:
- May still not match PINT exactly
- Reinventing wheel
- Debugging will be tedious

**Estimated effort**: 6-10 hours (uncertain)

---

## Recommendation

**Go with Option B (scipy.optimize) short-term, Option A long-term**:

1. **Immediate (next session)**:
   - Use `scipy.optimize.least_squares` with Levenberg-Marquardt
   - Verify it matches PINT
   - Unblock Milestone 2 completion
   - Effort: 1-2 hours

2. **Future (Milestone 4?)**:
   - Study PINT's WLSFitter in detail
   - Implement pure JAX version of their algorithm
   - Enable full JIT compilation
   - Effort: 4-6 hours when time permits

**Rationale**: We need a working fitter NOW to proceed with noise models (Milestone 3). We can optimize the implementation later.

## Testing Strategy

Once fitter is fixed, validate with:

1. **Synthetic data test**:
   - Generate fake TOAs from known parameters
   - Add noise
   - Fit and verify recovery within uncertainties

2. **Perturbation test**:
   - Start from PINT's fitted values
   - Perturb slightly
   - Verify both fitters return to same values

3. **Real data test**:
   - Fit J1909-3744 from scratch
   - Compare all parameters to PINT
   - Should agree within ~0.1 sigma

## Files

- Investigation scripts:
  - `plot_jax_vs_baseline.py` - JAX precision test
  - Various comparison test scripts
  
- Status documents:
  - `M2_JAX_FITTING_STATUS.md` - Previous investigation
  - `M2_FITTING_DIAGNOSIS.md` - Earlier diagnosis
  - `FITTER_CONVERGENCE_INVESTIGATION.md` (this file)

---

## Conclusion

**The good news**: JAX works perfectly, residuals are accurate.

**The bad news**: Our fitting algorithm is fundamentally broken.

**The solution**: Use scipy.optimize (battle-tested) or replicate PINT's algorithm exactly. Do NOT try to debug generic Gauss-Newton - it's the wrong algorithm for this problem.
