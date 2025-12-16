# JAX Gauss-Newton Prototype - Status

**Date**: 2025-12-05  
**Goal**: Replace manual derivatives + numpy WLS with JAX autodiff + jaxopt.GaussNewton

## What We're Trying to Do

Implement Gauss-Newton least-squares fitting using:
1. Pure JAX residual function (JIT-compilable)
2. Automatic differentiation for Jacobian (no manual derivatives!)
3. `jaxopt.GaussNewton` solver

**Benefits**:
- 10-100× faster iterations (JIT compilation)
- No manual derivative maintenance
- Automatically works for ANY parameter
- Clean, extensible architecture

## Current Status: 🔴 FUNDAMENTAL ARCHITECTURE ISSUE IDENTIFIED

### What's Working ✅
- JAX autodiff compiles successfully
- jaxopt.GaussNewton runs without errors
- JIT compilation works (~0.87s total time)

### What's NOT Working ❌
- **Both F0/F1 and DM fitting fail** (RMS gets worse)
- **Root cause identified**: `dt_sec` from `compute_residuals_simple` is already processed
- `dt_sec` = time from PEPOCH to emission, **with phase already computed**
- We can't recompute phase from this `dt_sec` - it's circular!
- The current JUG architecture separates "delay computation" from "phase computation"
- JAX GN needs them unified in one differentiable function

## The Core Issue - Architecture Mismatch

Current JUG architecture (what exists now):
```
1. Compute all delays (barycentric, binary, DM) → delay_sec
2. Compute dt_sec = TDB - PEPOCH - delay_sec  
3. Compute phase = F0*dt + F1*dt²/2 + ...
4. Compute residuals = (phase - int(phase)) / F0
```

What JAX GN needs:
```
residual_func(all_params) → residuals
```
Where `all_params` includes F0, F1, DM, etc., and the function computes EVERYTHING from scratch.

**The problem**: Step 2 (`dt_sec`) is what we're caching, but it's **already mixed with phase computation** in the current code. We can't extract a "pure" dt_sec that doesn't depend on fitted parameters.

**Why both tests failed**:
1. F0/F1 test: `dt_sec` was computed using F0/F1 from par file → circular
2. DM test: `dt_sec` represents "time to emission" which already has phase → we can't recompute phase from it

## What We Need To Do - REVISED

**The ONLY way forward**: Rewrite residual computation in pure JAX from TOAs.

This means implementing in JAX:
1. ✅ TDB computation (already have vectorized version)
2. ✅ Barycentric delays (Roemer + Shapiro) - can port from current code
3. ✅ Binary delays (ELL1/DD/BT) - already have JAX versions!
4. ✅ DM delay - trivial formula
5. ✅ Phase computation - simple Taylor series
6. ✅ Residual calculation - straightforward

**This is actually doable!** Most components already exist in JAX form in `jug/delays/combined.py`.

**Estimated effort**: 4-6 hours to wire everything together in pure JAX.

**Alternative (hybrid approach)**:
- Keep current fitter for "quick fits"
- Add JAX GN as "advanced fitter" for when you need speed or extensibility
- Document trade-offs clearly

## Files Created

1. **`jax_gauss_newton_prototype.py`** (418 lines)
   - Full implementation with F0/F1/F2/DM/DM1/DM2 support
   - Had JAX JIT issues with string parameters
   - Too complex for initial testing

2. **`jax_gn_minimal.py`** (150 lines) ✅ WORKS
   - Minimal F0+F1-only version
   - Successfully compiles and runs
   - Proves JAX autodiff works
   - But has RMS issue described above

## Next Steps

**Immediate** (to prove concept):
1. Test with perturbed F0 (e.g., F0 × 1.00001)
2. Verify GN corrects it back to optimal value
3. If yes → concept proven ✅

**Short-term** (if concept works):
4. Add DM parameter fitting (easier test case)
5. Compute uncertainties from Jacobian
6. Add Levenberg-Marquardt fallback
7. Benchmark speed

**Long-term** (production integration):
8. Rewrite delays in pure JAX (big project)
9. Or: use current fitter architecture with JAX-compiled iteration loop
10. Integrate into `jug/fitting/`

## Recommendation - UPDATED

**We have two paths forward:**

### Path A: Pure JAX Implementation (4-6 hours)
**Pros**:
- Clean, unified JAX codebase
- True autodiff for all parameters
- Maximum speed (full JIT compilation)
- Future-proof architecture

**Cons**:
- Significant upfront effort
- Need to port/rewrite delay calculations in pure JAX
- More complex initial implementation

### Path B: Hybrid Approach (1-2 hours)
**Pros**:
- Keep current optimized_fitter.py working
- Add JAX GN as optional advanced fitter
- Incremental path - can do Pure JAX later
- Lower risk

**Cons**:
- Two parallel implementations to maintain
- JAX version won't be as fast (can't cache as much)
- Doesn't fully leverage JAX advantages

## My Recommendation

Given that:
1. Current fitter already works and is reasonably fast (1.7s)
2. You're at limit and need results
3. Pure JAX would be 4-6 hours of work

**I recommend Path B for now**: Keep the current analytical derivative approach, add proper convergence detection, and defer full JAX rewrite to when you have more time.

**Immediate actions** (30 min):
1. Fix convergence detection in `optimized_fitter.py` (the oscillation issue)
2. Document current fitter performance  
3. Mark JAX GN as "future enhancement"

**Later** (when time permits):
4. Implement pure JAX residual function
5. Integrate jaxopt.GaussNewton
6. Benchmark and compare

This gets you a working, fast fitter TODAY while keeping the JAX path open for future optimization.

## Questions for User

1. Should we proceed with perturbation test to prove GN works?
2. Or should we focus on DM fitting (which might work already)?
3. Or do you want to see a different approach entirely?

The literature review strongly supports this direction, but we need to get the implementation details right.
