# Performance Bottleneck Analysis for JUG Pulsar Timing

**Date**: 2025-12-05  
**Purpose**: Document constraints preventing further speedup via JAX GN  
**For**: Literature review on creative solutions

---

## Current Performance

**Target**: Match TEMPO2 (~0.3s per fit)  
**Current JUG**: ~1.1s per fit  
**Gap**: 3.8× slower than TEMPO2

### Time Breakdown (J1909-3744, 10,408 TOAs)

| Component | Time | % of Total | Optimization Status |
|-----------|------|------------|---------------------|
| **Cache building** | **0.6-1.3s** | **55-85%** | ⚠️ Bottleneck |
| - TDB computation | ~0.2s | 15% | Vectorized numpy |
| - Barycentric delays | ~0.5s | 35% | JPL ephemeris calls |
| - Binary delays | ~0.3s | 20% | ELL1/DD models |
| **Fitting iterations** | **~0.3s** | **20%** | ✅ Optimized |
| - 4 iterations × 0.075s | | | JAX derivatives + WLS |
| **Total** | **1.1s** | **100%** | |

**Key finding**: Cache building dominates (55-85% of time), not iterations.

---

## The Fundamental Constraint: Longdouble Precision

### Why Longdouble is Required

**Problem**: Float64 precision degrades with time span from PEPOCH

For pulsar timing phase calculation: `φ = F0 × dt + (F1/2) × dt²`

| Time Span | Phase (cycles) | Float64 Precision | Target | Status |
|-----------|----------------|-------------------|--------|--------|
| 10 years | 1.07×10¹¹ | 70 ns | <100 ns | ✅ OK |
| 20 years | 2.14×10¹¹ | 140 ns | <100 ns | ⚠️ Marginal |
| 30 years | 3.21×10¹¹ | 210 ns | <100 ns | ❌ Degraded |
| 40 years | 4.28×10¹¹ | 280 ns | <100 ns | ❌ Poor |
| 60 years | 6.42×10¹¹ | 420 ns | <100 ns | ❌ Unacceptable |

**Precision formula**: `ε ≈ |phase| × 2.22×10⁻¹⁶ / F0`

**Implication**: For NANOGrav/IPTA datasets (20-30+ year baselines), float64 is insufficient.

### Current Solution: Hybrid Longdouble/JAX

```python
# Phase calculation in numpy longdouble (80-bit, 18-digit precision)
f0_ld = np.longdouble(F0)
f1_ld = np.longdouble(F1)
dt_ld = np.longdouble(dt_sec)
phase_ld = f0_ld * dt_ld + 0.5 * f1_ld * dt_ld**2  # HIGH PRECISION

# Convert to float64 for JAX (derivatives, WLS)
residuals_f64 = np.float64(phase_wrapped / f0_ld)
J = jax_compute_derivatives(residuals_f64)  # JAX accelerated
```

**Performance impact**: Only 5% slower than pure float64, maintains <100 ns precision for any time span.

**References**: 
- `playground/LONGDOUBLE_SPIN_IMPLEMENTATION.md`
- `docs/JUG_PROGRESS_TRACKER.md` (Session 13, Longdouble Investigation)

---

## The JAX Limitation

**JAX does not support longdouble/float128.**

Maximum precision: `jax.numpy.float64` (same as numpy.float64)

This means:
- ✅ JAX can accelerate derivatives (don't need longdouble)
- ✅ JAX can accelerate WLS solve (doesn't need longdouble)
- ❌ JAX **cannot** compute phase with longdouble precision
- ❌ Pure JAX implementation would degrade precision for long baselines

### Why This Blocks "Pure JAX GN"

**Pure JAX Gauss-Newton** would require:
```python
def residual_func_jax(params, toa_data):
    # All in JAX (float64 only)
    dt_sec = compute_dt_jax(...)      # float64
    phase = F0 * dt_sec + ...          # float64 (PRECISION LOSS!)
    return residuals
```

**Result**: 
- 20-year dataset: 140 ns precision (marginal)
- 30-year dataset: 210 ns precision (degraded)
- 40-year dataset: 280 ns precision (poor)

**vs Current hybrid**:
- Any timespan: <100 ns precision ✅

---

## Investigated Approaches That Failed

### 1. Piecewise Fitting

**Idea**: Split data into segments with local PEPOCH to reduce dt

**Implementation**: 
- 3-year segments
- Local PEPOCH for each segment
- Smaller dt → better float64 precision

**Result**: ❌ Failed
- Coordinate transformations between segments introduce equivalent errors
- Quadratic drift: 20-25 μs spread increases with time
- **Lesson**: Can't improve precision by changing coordinates if transformation loses precision

**Reference**: `playground/PIECEWISE_FITTING_IMPLEMENTATION.md`

### 2. Two-Stage Fitting (Spin First, Then Others)

**Idea**: 
1. Fit F0/F1/F2 with longdouble (current method)
2. Fix spin parameters, fit DM/astro/binary with JAX GN

**Problem**: Parameter correlations
- F0 and DM are correlated (frequency-dependent delays)
- F1 and PMRA/PMDEC are correlated (proper motion)
- Fitting separately may not converge to global optimum

**Status**: Not implemented, questionable benefit

### 3. Residual Pre-Removal

**Idea**: Remove spin phase component first, fit other parameters

**Problem**: Assumes clean separation of spin from other effects
- DM delay depends on frequency
- Astrometric delay depends on position
- Binary delay depends on orbital phase
- All are coupled to spin phase via time coordinate

**Status**: Not viable

---

## Cache Building Bottleneck

**The real bottleneck is not the fitting iterations, but the cache.**

### What Gets Cached

```python
result = compute_residuals_simple(par_file, tim_file, clock_dir)
dt_sec = result['dt_sec']  # Time from PEPOCH to emission
```

This computes:
1. **TDB times** (~0.2s)
   - Read clock files (UTC→TAI→TT→TDB chain)
   - Apply clock corrections
   - Vectorized numpy, already fast

2. **Barycentric delays** (~0.5s) ⚠️ BOTTLENECK
   - Load JPL ephemeris (DE440s.bsp)
   - Compute Solar System barycenter position
   - Roemer delay: `R⃗·n̂/c`
   - Shapiro delay: `-2GM/c³ ln(...)`
   - Calls to JPL SPICE library (not JAX-compatible)

3. **Binary delays** (~0.3s)
   - ELL1/DD/BT orbital models
   - Already implemented in JAX (`jug/delays/combined.py`)
   - Could be faster if entire pipeline were JAX

4. **DM delays** (~0.05s)
   - Simple formula: `K_DM × DM / f²`
   - Already fast

### Why Cache Building is Slow

**JPL Ephemeris calls** are the bottleneck:
- Python → SPICE library → file I/O
- Not vectorized (called per TOA or in small batches)
- Not JAX-compatible (external C library)

TEMPO2 is faster because:
- Written in C/C++ (no Python overhead)
- Better ephemeris caching strategies
- Optimized SPICE usage

---

## The Catch-22

**To use JAX GN optimally, we need**:
1. Pure JAX residual function (end-to-end differentiable)
2. All computations in JAX (for JIT compilation)

**But we require**:
1. Longdouble precision for F0/F1/F2 (JAX doesn't support)
2. JPL ephemeris calls (not JAX-compatible)

**Result**: Can't have both precision AND full JAX acceleration.

---

## Current Hybrid Architecture (Optimal Given Constraints)

```
┌─────────────────────────────────────────────────────────┐
│  Cache Building (0.6-1.3s) - NUMPY/SPICE                │
│  - TDB computation (numpy)                              │
│  - Barycentric delays (SPICE library)                   │
│  - Binary delays (JAX possible)                         │
│  → dt_sec (time from PEPOCH to emission)               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Fitting Loop (0.3s) - HYBRID NUMPY/JAX                 │
│  - Phase calculation: NUMPY LONGDOUBLE (precision)     │
│  - Derivatives: JAX JIT (speed)                         │
│  - WLS solve: JAX JIT (speed)                          │
│  - Iterations: 4 (converges quickly)                   │
└─────────────────────────────────────────────────────────┘
```

**This is optimal because**:
- ✅ Uses longdouble where needed (phase)
- ✅ Uses JAX where possible (derivatives, WLS)
- ✅ Minimizes overhead (only 5% penalty vs full float64)
- ✅ Maintains precision (<100 ns for any timespan)

---

## Questions for Literature Review

### Primary Question

**Is there a way to achieve both longdouble precision AND full JAX acceleration?**

Specific sub-questions:

1. **JAX + High Precision**:
   - Are there JAX extensions/forks that support float128/longdouble?
   - Can we use JAX's custom dtype system to wrap longdouble?
   - Are there alternative autodiff frameworks that support high precision?

2. **Ephemeris Acceleration**:
   - Can we rewrite JPL ephemeris lookups in pure JAX/Python?
   - Are there faster ephemeris libraries (ChebyFit approximations)?
   - Can we precompute/cache ephemeris on a dense grid and interpolate?

3. **Precision Without Longdouble**:
   - Are there numerically stable algorithms for `F0 × dt` with large dt?
   - Can we use compensated summation (Kahan, etc.) with float64?
   - Can we reformulate the phase calculation to avoid precision loss?

4. **Hybrid Optimization**:
   - Can we JIT-compile the numpy longdouble phase calculation?
   - Can we use Numba/Cython for the cache building step?
   - Are there creative ways to separate concerns for parallel optimization?

5. **Alternative Architectures**:
   - Could we use GPU with higher precision (if available)?
   - Could we use interval arithmetic or arbitrary precision?
   - Are there domain-specific optimizations for pulsar timing?

### Comparison Points

**TEMPO2 (C++, ~0.3s)**:
- How does TEMPO2 achieve its speed?
- What precision does TEMPO2 use (float64? float80? float128?)?
- How does TEMPO2 handle long-baseline precision degradation?
- What ephemeris caching strategies does TEMPO2 use?

**PINT (Python+numpy, ~2.1s)**:
- What precision does PINT use?
- Why is PINT 2× slower than JUG?
- Does PINT have precision issues with long baselines?

---

## Current Performance Targets

| Metric | Current | TEMPO2 | Target | Gap |
|--------|---------|--------|--------|-----|
| Time per fit | 1.1s | 0.3s | <2s | ✅ Met |
| vs TEMPO2 | 3.8× slower | 1× | <5× | ✅ Good |
| vs PINT | 1.9× faster | - | >1× | ✅ Better |
| Precision | <100 ns | ? | <100 ns | ✅ Met |
| Iterations | 4 | ? | <10 | ✅ Excellent |

**Current status**: JUG is production-ready and faster than PINT. Further speedup would be nice but not critical.

---

## Files to Review

For detailed context:

1. **`playground/LONGDOUBLE_SPIN_IMPLEMENTATION.md`** - Why longdouble is necessary
2. **`playground/SESSION_FINAL_SUMMARY.md`** - Current performance analysis
3. **`docs/LITERATURE_REVIEW_OPTIMIZATION_METHODS.md`** - Previous lit review (JAX GN)
4. **`docs/JUG_PROGRESS_TRACKER.md`** - Full development history
5. **`jug/fitting/optimized_fitter.py`** - Current implementation

---

## Specific Request for Literature Review

**Please investigate**:

1. **Is JAX + longdouble possible?** (extensions, workarounds, alternatives)
2. **Can ephemeris calls be accelerated?** (pure Python/JAX implementations, caching strategies)
3. **Are there numerically stable float64 algorithms** for large F0×dt calculations?
4. **What does TEMPO2 actually do?** (precision, algorithms, architecture)
5. **Are there creative hybrid approaches** we haven't considered?

**Focus on**:
- Practical solutions (not just theoretical)
- Benchmarks/evidence from other fields (astrometry, geodesy, high-precision computing)
- Trade-offs (precision vs speed vs complexity)

**Timeline**: 1-2 hours for comprehensive review

---

## Bottom Line

**The constraint**: Need longdouble precision for F0/F1/F2 phase calculation, but JAX doesn't support it.

**Current solution**: Hybrid numpy longdouble (phase) + JAX float64 (derivatives/WLS) = 1.1s

**Goal**: Find creative ways to either:
1. Get longdouble working with JAX, or
2. Achieve longdouble-level precision without longdouble, or
3. Dramatically speed up the cache building step

**If none of these are possible**: Current implementation is optimal and Milestone 2 is complete.
