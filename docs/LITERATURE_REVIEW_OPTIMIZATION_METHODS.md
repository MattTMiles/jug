# Literature Review: Optimization Methods for Pulsar Timing

**Date**: 2025-12-05  
**Reviewed by**: External AI consultant  
**Purpose**: Inform JUG optimization strategy for Milestone 2 completion

---

## Executive Summary

This review examines optimization methods used in existing pulsar timing codes (TEMPO2, PINT) and evaluates options for JUG's JAX-based implementation. **Key finding**: All production timing codes use linearized WLS/GLS with SVD, not sophisticated optimizers. Moving to Gauss-Newton + JAX autodiff would be "next-gen" territory.

---

## 1. What Existing Pulsar-Timing Codes Actually Use

### TEMPO2
- **Core fitting**: Linear SVD weighted least squares on design matrix
- **Algorithm**: "Based on a linear singular-value decomposition, weighted least-squares algorithm"
- **Correlated noise**: GLS via Cholesky whitening, then WLS on whitened residuals
- **Derivatives**: Analytical, no AD
- **No Gauss-Newton/LM**: Just linearized timing model + SVD

### PINT
- **Main fitters**: 
  - `WLSFitter`: White-noise WLS
  - `GLSFitter`: GLS with full covariance (Cholesky whitening + WLS)
  - `DownhillFitter`: Combines WLS/GLS with scipy optimizers for tricky cases
- **Primary approach**: Linearized WLS (same as TEMPO2)
- **Derivatives**: Mostly analytic, finite differences for some components, no AD

### TempoNest / enterprise / PTA codes
- **TempoNest**: Wraps TEMPO2, replaces deterministic fit with nested sampling
- **enterprise**: Marginalizes timing parameters analytically (G-matrix/Cholesky projection), samples noise parameters
- **Vela.jl**: Same pattern - timing parameters marginalized, not fitted
- **Common theme**: Bayesian inference marginalizes timing params; no fancy optimizers

**Conclusion**: Production timing codes **overwhelmingly use linearized WLS/GLS with SVD/Cholesky**. Anything more sophisticated is for Bayesian inference, not deterministic fitting.

---

## 2. Gauss-Newton + Automatic Differentiation

### Is Anyone Doing This?

**No mainstream timing package uses GN + AD as of 2024-25.** TEMPO2, TEMPO, PINT all use hand-derived Jacobians + linear LS.

**JUG would be genuinely "next-gen" here.**

### JAX Libraries for Least-Squares

**Recommended for JUG:**

1. **JAXopt** (most mature)
   - `jaxopt.GaussNewton(residual_fun, ...)`
   - `jaxopt.LevenbergMarquardt(residual_fun, ...)`
   - Designed exactly for "minimize ½‖r(θ)‖²" problems
   - Automatic Jacobians via AD

2. **Optimistix** (JAX-native, clean API)
   - `optimistix.GaussNewton`
   - `optimistix.LevenbergMarquardt`
   - Good benchmarking results

3. **Optax** (NOT recommended)
   - Gradient descent/Adam: great for ML
   - Poor for deterministic least-squares (slower, less reliable)

### Expected Speed vs Manual Derivatives

**For JUG's regime (2-20 params, 10³-10⁵ TOAs):**

- Bottleneck: evaluating residuals & Jacobian, not AD overhead
- JAX advantages:
  - `jacfwd`/`jacrev`: compute full Jacobian in one fused pass
  - XLA JIT: fuse residual + Jacobian + GN step into single kernel
  - From benchmarks: GN/LM with AD competitive with hand-coded Jacobians

**Expected performance:**
- First call: JIT compile ~0.5-1s (one-time)
- Subsequent fits: Very fast (current 0.014s/iter, likely converge in fewer iterations)

**Verdict**: AD+GN not a magic 10× speedup, but buys developer speed, fewer bugs, cleaner JAX integration while staying within same runtime budget as TEMPO2.

---

## 3. Levenberg-Marquardt

### When LM Better Than Gauss-Newton?

LM regularizes GN step: `(J^T W J + λI) Δθ = -J^T W r`

Interpolates between:
- Steepest descent (large λ) 
- Gauss-Newton (small λ)

**Helps when:**
- Model nonlinear / far from solution (GN overshoots)
- Jacobian poorly conditioned / rank-deficient
- Fitting weird binary/DM models with poor linearization

**For PTA timing**: Model nearly linear, plain GN usually excellent. LM shines for:
- Wide prior searches
- Pathological pulsars (glitches, high eccentricity, strong covariances)

### Do Existing Codes Use LM?

**No.** TEMPO2/PINT/TempoNest rely on SVD/GLS, at most generic scipy "downhill" optimizers. **LM not part of standard timing lore yet.**

### JAX Implementations

- **JAXopt**: `LevenbergMarquardt` with Madsen-Nielsen stopping criteria (more mature)
- **Optimistix**: `LevenbergMarquardt` as regularized GaussNewton (more JAX-pure)

---

## 4. Other Optimization Methods

### L-BFGS / Quasi-Newton
- Efficient for general smooth minimization
- Doesn't exploit least-squares structure
- For 2-20 params: fine, but no advantage over GN/LM
- More useful for full log-likelihood minimization with robust loss

### Trust-Region Newton
- "Proper" GN with step control
- Very robust convergence, at cost of extra linear solves
- Available in JAXopt/Optimistix
- **Probably overkill for timing** - LM gives almost all benefits

### Specialized Pulsar-Timing Methods
- Robust fitting (M-estimators for outliers/non-Gaussian noise)
- Still done in ML/GLS framework, not exotic optimizers

**Bottom line**: Gauss-Newton + LM covers 99% of what JUG needs.

---

## 5. WLS vs GLS

### When Is GLS Necessary?

**WLS assumes**: Independent Gaussian errors with weights from TOA uncertainties

**Real MSP data has:**
- Strong low-frequency red noise
- ECORR/jitter
- DM variations
→ Correlated TOAs

**Literature**: Coles et al. (2011), van Haasteren & Levin (2013) show GLS (via Cholesky whitening) consistently superior to WLS for timing + noise parameter estimation with correlated noise.

**Recommendation:**
- Quick ephemeris fits / toy white-noise: WLS fine
- Production timing with real PTA noise: Need GLS/Cholesky

### Cost Scaling

**WLS**: 
- Build/solve normal equations in parameter space
- Cost: O(N_TOA × N_param²) + O(N_param³)

**GLS**:
- Need Cholesky of covariance C (N_TOA × N_TOA)
- Cost: O(N_TOA³) dominates

**In practice:**
- N_TOA ≤ few 10³: Cholesky cheap, widely used (TEMPO2, PINT)
- N_TOA = 10⁴-10⁵: Use low-rank + diagonal decompositions, Fourier bases (full PTA inference)

**For JUG now**: Reasonable to stick with WLS, defer GLS to noise modeling milestone.

---

## 6. Performance Strategies (JAX/JUG Specific)

### JIT & Caching

**Strategy:**
1. JIT the residual + Jacobian + solver together
2. Define `residual_fun(theta, data)` returning `(N_TOA,)` vector
3. Let JAXopt/Optimistix compute Jacobians via AD
4. JIT-compile entire `fit(theta0, data)` function

**Cache:**
- Design-matrix-like structures that don't change
- TOA metadata, observatory positions, etc.

**Expected**: Current 0.014s/iter likely maintained or beaten with fused JAX GN step

### Derivatives: Analytic vs AD vs Finite Differences

- **Analytic**: Slightly faster, high maintenance
- **Finite differences**: Slow and noisy
- **AD in JAX**: 
  - Near-analytic performance
  - No humans maintaining partials
  - Trivially extensible for new timing components

**Given JUG is all-in on JAX**: AD is the sweet spot.

### Parallelization

- **Over TOAs**: Residual evaluation naturally vectorized (JAX `vmap`)
- **Over pulsars**: When multi-pulsar, `vmap` over pulsars or `pmap` across devices

---

## 7. Convergence & Stability

### From Existing Codes

**TEMPO2**: 
- Stops when χ² or parameters stop changing (heuristic thresholds)

**PINT**: 
- `fit_toas(maxiter, threshold=...)` uses χ² improvement + parameter step thresholds
- `GLSFitter.ftest()`: test significance of parameters
- Reports degenerate parameters via SVD rank detection

### Recommended for JUG + GN/LM

**Stop when:**
- `‖Δθ‖₂ ≤ xtol × (‖θ‖₂ + xtol)` or
- `‖∇χ²‖_∞ ≤ gtol`

(Exactly as in JAXopt LM stopping criteria)

**Always:**
- Compute SVD of J (or J^T W J)
- Diagnose near-degenerate parameters
- Report to user

---

## 8. Ranked Recommendations for JUG

### Priority 1: "Modern WLS" with Gauss-Newton + AD ⭐

**Implement:**
- GN least-squares loop in JAX using AD for Jacobians
- Either use `jaxopt.GaussNewton` OR roll your own mini-GN (one linear solve per iteration) with JIT
- Keep SVD/rank-reveal for degeneracies
- Stay with WLS (diagonal W) for now

**Benefits:**
- TEMPO2-level speed (~0.3s per pulsar) after JIT warmup, likely below
- Simplifies derivative bookkeeping
- Sets up for GLS/Cholesky later

**This will give you:**
- Correctness ✅ (already have)
- Speed ✅ (match TEMPO2)
- Maintainability ✅ (no manual derivatives)
- Extensibility ✅ (any new parameter works automatically)

### Priority 2: Add LM Backend for Robustness

**Wire up:**
- `jaxopt.LevenbergMarquardt` OR `optimistix.LevenbergMarquardt`
- As drop-in alternative to GN

**Strategy:**
- Default: GN
- Fallback: LM if GN fails or for particularly nonlinear models

**Benefit**: Robustness to bad starting points and weird parameterizations with minimal extra cost.

### Priority 3: Plan for GLS/Cholesky

**Once moving to noise modeling:**
1. Implement covariance construction `C(η)` for noise params η
2. Cholesky factorization `C = LL^T`
3. Whitening of residuals and Jacobians

**Then**: Reuse same GN/LM infrastructure in whitened space (mirrors TEMPO2/PINT GLS)

---

## 9. Implementation Sketch (Conceptual)

### Basic JAX Implementation

```python
def residual_fun(theta, toas, model_state):
    """Returns residuals in seconds (shape: [N_toa])"""
    # build phases/delays, subtract integer turns, etc.
    return residuals

solver = jaxopt.GaussNewton(
    residual_fun=residual_fun,
    maxiter=10,
    tol=1e-12,
    implicit_diff=False,  # probably don't need it
)

@jax.jit
def fit(theta0, toas, model_state):
    result = solver.run(theta0, toas=toas, model_state=model_state)
    return result.params, result.state
```

### Later, for GLS

```python
def whiten(residuals, J, L):  # L = chol(C)
    y = jax.scipy.linalg.solve_triangular(L, residuals, lower=True)
    Jw = jax.scipy.linalg.solve_triangular(L, J, lower=True)
    return y, Jw

# Use residual_fun that returns whitened residuals,
# or wrap inside GN step
```

---

## 10. Expected Performance Improvement

### Current JUG Status
- ~0.014s / iteration for mixed parameter fits
- ~0.4s total fitting iterations
- ~1.3s cache building (one-off)

### With JIT-Fused GN/LM + AD

**Compile**: Similar or slightly more expensive (one-time)

**Per iteration**: Similar cost, but:
- Cut iteration count (GN/LM converges in handful of steps for nearly-linear timing)
- Remove Python overhead around loop

**Target**: <0.3s per fit after warmup for typical MSPs is **absolutely realistic**

**Headroom**: Still have room to optimize data structures/caching to beat TEMPO2 for large TOA sets

---

## 11. Key Insights

### What This Review Revealed

1. **JUG is pioneering**: No mainstream timing code uses GN+AD. You'd be first.

2. **TEMPO2/PINT are simpler than expected**: Just linearized WLS + SVD, no fancy optimization.

3. **GLS can wait**: Stick with WLS until noise modeling. This is what TEMPO2/PINT do by default.

4. **JAX AD is the right choice**: Near-analytic performance, zero maintenance, trivially extensible.

5. **LM is insurance**: Not needed for typical MSPs, but gives robustness for pathological cases.

6. **Convergence detection matters**: Current oscillation issue likely fixable with proper stopping criteria.

### What to Do Next

**Immediate (this session)**:
1. Implement Gauss-Newton + JAX autodiff
2. Use `jaxopt.GaussNewton` as starting point
3. JIT-compile the full fitting loop
4. Implement proper convergence criteria (from JAXopt)

**Short-term (next session)**:
5. Add `jaxopt.LevenbergMarquardt` as fallback
6. Benchmark against TEMPO2/PINT
7. Test on multiple pulsars

**Medium-term (Milestone 3)**:
8. Implement GLS/Cholesky whitening
9. Add noise models (EFAC/EQUAD/ECORR)
10. Extend to red noise (GP with FFT covariance)

---

## 12. References

This review synthesized information from:
- Edwards et al. (2006) - TEMPO2 paper
- Luo et al. (2020) - PINT paper
- Coles et al. (2011) - GLS for pulsar timing
- van Haasteren & Levin (2013) - Optimal filtering
- Blondel et al. (2021) - JAXopt paper
- Kidger (2021) - Optimistix
- TEMPO2/PINT GitHub repositories and documentation
- TempoNest, enterprise documentation

---

## Conclusion

**The path forward is clear**: Implement Gauss-Newton + JAX autodiff as Priority 1. This will:
- Match or beat TEMPO2 speed
- Eliminate manual derivative maintenance
- Set up clean architecture for GLS
- Position JUG as next-generation timing software

The literature review confirms this is the right technical choice, and JUG would be breaking new ground in the timing community.

**Recommendation**: Proceed with implementation immediately.

