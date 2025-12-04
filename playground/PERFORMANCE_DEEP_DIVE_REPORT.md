# JUG Performance Deep Dive Report

**Date**: 2025-12-01  
**Pulsar**: J1909-3744 (10,408 TOAs)  
**Goal**: Profile optimized fitter and identify every possible speedup

---

## Executive Summary

**Current Performance** (10k TOAs):
- **Total fitting time**: ~3.1s (10 iterations)
- **Cache initialization**: 2.7s (88% of time) ⚠️ **BOTTLENECK**
- **JIT compilation**: 0.4s (one-time)
- **Per iteration**: 1.0ms (excellent!)

**Key Finding**: Cache initialization dominates. Within cache:
1. **Ephemeris lookups: 749ms (60%)** ← BIGGEST BOTTLENECK
2. **JAX array conversion: 352ms (28%)** ← UNEXPECTED!
3. **Clock file loading: 56ms (4.5%)**
4. **TDB computation: 42ms (3.3%)**

---

## Detailed Profiling Results

### Section 1: Full Pipeline Breakdown

```
Component                    Time        % of Total
================================================================
Cache initialization        2666 ms         88%
  ├─ Ephemeris (obs)         749 ms         25%  ← TARGET #1
  ├─ JAX array conversion    352 ms         12%  ← TARGET #2
  ├─ Clock file loading       56 ms          2%  ← TARGET #3
  ├─ TDB computation          42 ms          1%
  ├─ File I/O (tim)           29 ms          1%
  └─ Other delays             14 ms         <1%

JIT compilation              359 ms         12%
Per-iteration (×10)           10 ms         <1%  ✅ OPTIMAL
──────────────────────────────────────────────────
TOTAL                       3035 ms        100%
```

### Section 2: Iteration Performance (Post-JIT)

**Average iteration: 1.01 ms** (10 runs)
- Min: 0.92 ms
- Max: 1.33 ms
- Std: 0.12 ms

**JIT Speedup**: 376× faster than uncompiled (380ms → 1ms)

**Memory usage**: 0.64 MB (negligible)

### Section 3: Cache Initialization Deep Dive

```
Operation                           Time (ms)    % of Cache
================================================================
1. Ephemeris: Observatory pos/vel     748.7        60.0%  🔴
2. JAX array preparation               351.6        28.2%  🔴
3. Load MK clock file                   55.5         4.5%  🟡
4. Compute TDB (clock corrections)      41.8         3.3%  🟡
5. Parse .tim file                      29.0         2.3%  🟢
6. Planetary Shapiro (5 planets)        10.6         0.8%  🟢
7. Sun ephemeris                         2.4         0.2%  🟢
8. Other operations                      7.4         0.6%  🟢
──────────────────────────────────────────────────────────────
TOTAL                                 1247.0       100.0%
```

**Note**: Discrepancy between full run (2666ms) and instrumented run (1247ms) 
due to:
- Warmup JIT overhead in first call
- Additional operations (binary delays, spin phase, etc.)
- Python print overhead removed in instrumented version

---

## Optimization Opportunities

### Priority 1: Ephemeris Lookups (749ms → potential 200ms)

**Current bottleneck**: `compute_ssb_obs_pos_vel()` calls Astropy's JPL ephemeris for each TOA batch.

**Why it's slow**:
- DE440 ephemeris loaded from disk on first access
- Interpolation done in Python (not optimized)
- Earth position computed for 10k different times

**Potential speedups**:

#### Option A: Pre-load and Cache Ephemeris Kernel (Easiest)
```python
# Load once at session start
from jug.delays.ephemeris_cache import EphemerisCache
ephemeris = EphemerisCache('de440')  # Loads kernel once

# Reuse for all pulsars
for pulsar in pulsars:
    pos, vel = ephemeris.get_earth_posvel(tdb_times)  # Fast!
```
**Expected speedup**: 3-5× (749ms → ~150-250ms)  
**Effort**: 2-3 hours  
**Risk**: Low

#### Option B: Batch Multiple Pulsars
```python
# If fitting 10 pulsars, load ephemeris once, reuse 10 times
ephemeris = load_de440_once()
for pulsar in pulsars:
    # Reuse loaded kernel
```
**Expected speedup**: Amortized (749ms once, then ~50ms per pulsar)  
**Effort**: 1 hour (API design)  
**Risk**: Low

#### Option C: Use PINT's Cached Ephemeris (Most Compatible)
```python
# PINT already caches ephemeris internally
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
pos, vel = objPosVel_wrt_SSB('earth', tdb_times, ephemeris='de440')
```
**Expected speedup**: 2-3× (749ms → ~250-370ms)  
**Effort**: 30 minutes  
**Risk**: Low (adds PINT dependency for this component)

**RECOMMENDATION**: Try Option C first (quick win), then implement Option A for independence.

---

### Priority 2: JAX Array Conversion (352ms → potential 1ms)

**Surprise finding**: Converting numpy arrays to JAX is unexpectedly slow!

**Current code**:
```python
dt_sec_jax = jnp.array(dt_sec_cached)  # 352ms for 10k floats!
```

**Why it's slow**:
- Transfers data to GPU (if available)
- Creates new DeviceArray on GPU memory
- Synchronization overhead

**Potential speedups**:

#### Option A: Keep Arrays on CPU (If No GPU Needed)
```python
# Force CPU device
with jax.default_device(jax.devices('cpu')[0]):
    dt_sec_jax = jnp.array(dt_sec_cached)  # ~1ms
```
**Expected speedup**: 350× (352ms → 1ms!)  
**Effort**: 10 minutes  
**Risk**: Low (we're not using GPU anyway at 10k TOAs)

#### Option B: Pre-allocate JAX Arrays
```python
# Allocate once
dt_sec_jax = jnp.zeros(n_toas)
# Update in-place (if possible with JAX)
dt_sec_jax = dt_sec_jax.at[:].set(dt_sec_cached)
```
**Expected speedup**: ~2× (352ms → ~150ms)  
**Effort**: 1 hour  
**Risk**: Medium (JAX arrays are immutable)

#### Option C: Skip JAX Conversion Until Needed
```python
# Keep as numpy until fitting loop
dt_sec_np = dt_sec_cached  # No copy!

# Convert only when entering JIT function
def iteration_jax(dt_sec_np, ...):
    dt_sec_jax = jnp.asarray(dt_sec_np)  # Lazy conversion
    ...
```
**Expected speedup**: Moves cost to first iteration (amortized)  
**Effort**: 30 minutes  
**Risk**: Low

**RECOMMENDATION**: Try Option A first (force CPU). If no GPU is available, this is a huge win for free.

---

### Priority 3: Clock File Loading (56ms → potential 10ms)

**Current bottleneck**: `parse_clock_file()` reads and parses MK clock file (55ms for ~10k entries)

**Why it's slow**:
- Text file I/O
- Python string parsing line-by-line

**Potential speedups**:

#### Option A: Cache Parsed Clock Files
```python
_clock_cache = {}

def load_clock_cached(filename):
    if filename not in _clock_cache:
        _clock_cache[filename] = parse_clock_file(filename)
    return _clock_cache[filename]
```
**Expected speedup**: Infinite for subsequent pulsars (56ms → 0ms)  
**Effort**: 15 minutes  
**Risk**: None

#### Option B: Binary Clock File Format
```python
# Pre-process .clk files to .npy format
np.save('mk2utc.npy', clock_data)

# Load binary (10× faster)
clock_data = np.load('mk2utc.npy')  # ~5ms
```
**Expected speedup**: 10× (56ms → 5ms)  
**Effort**: 2 hours  
**Risk**: Low (one-time conversion)

**RECOMMENDATION**: Implement Option A immediately (trivial). Option B if processing many pulsars in batch.

---

### Priority 4: TDB Computation (42ms → potential 20ms)

**Current**: `compute_tdb_standalone_vectorized()` applies clock chain for 10k TOAs

**Why it's slow**:
- Linear interpolation for each TOA
- Multiple clock file lookups

**Potential speedups**:

#### Option A: Pre-compute Clock Splines
```python
# Fit cubic spline once
from scipy.interpolate import UnivariateSpline
mk_spline = UnivariateSpline(mk_clock['mjd'], mk_clock['correction'])

# Evaluate spline (vectorized, fast!)
corrections = mk_spline(toa_mjds)  # 10× faster than linear interp
```
**Expected speedup**: 2× (42ms → 20ms)  
**Effort**: 1-2 hours  
**Risk**: Low (may need to validate vs linear)

**RECOMMENDATION**: Worth trying, but lower priority than #1-3.

---

### Priority 5: File I/O (29ms → potential 10ms)

**Current**: `parse_tim_file_mjds()` reads 10k TOA lines from text file

**Potential speedup**:

#### Option A: Use Pandas for Faster Parsing
```python
import pandas as pd
df = pd.read_csv(tim_file, sep=r'\s+', comment='#')  # ~10ms
```
**Expected speedup**: 3× (29ms → 10ms)  
**Effort**: 1 hour  
**Risk**: Low (need to validate format handling)

**RECOMMENDATION**: Low priority (only 2.3% of time). Implement if batch-processing many pulsars.

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours total, 400ms savings)

1. **Force CPU for JAX arrays** (10 min) → Save 350ms ✅ HIGHEST ROI
2. **Cache parsed clock files** (15 min) → Save 56ms on subsequent pulsars
3. **Try PINT's cached ephemeris** (30 min) → Save 200-500ms

**Expected result**: 3.1s → 2.2s (30% faster)

### Phase 2: Ephemeris Optimization (2-4 hours, 500ms savings)

4. **Implement ephemeris cache class** (3 hours)
5. **Add batch API for multi-pulsar** (1 hour)

**Expected result**: 2.2s → 1.7s (45% faster than baseline)

### Phase 3: Polish (optional, 2-4 hours, 50ms savings)

6. **Binary clock file format** (2 hours)
7. **Clock spline interpolation** (2 hours)
8. **Pandas .tim parsing** (1 hour)

**Expected result**: 1.7s → 1.6s (50% faster than baseline)

---

## Conservative Estimates

| Optimization | Time | Speedup | Cumulative |
|--------------|------|---------|------------|
| **Baseline** | 3.1s | 1.0× | - |
| Force CPU JAX | 2.7s | 1.15× | 1.15× |
| Cache clocks | 2.7s | 1.15× | 1.15× |
| Cache ephemeris | 2.0s | 1.55× | 1.55× |
| Clock splines | 1.98s | 1.57× | 1.57× |
| **Best case** | 1.5s | 2.0× | 2.0× |

**Note**: "Best case" assumes all optimizations stack multiplicatively. Realistic target: **1.7-2.0s (1.5-1.8× speedup)**.

---

## What NOT to Optimize (Already Excellent)

✅ **Iteration time**: 1ms is blazing fast (376× JIT speedup)  
✅ **Memory usage**: 0.64 MB is negligible  
✅ **Phase computation**: 0.06ms (part of iteration)  
✅ **Derivatives**: 0.04ms (part of iteration)  
✅ **WLS solve**: 0.10ms (part of iteration, optimal for 2 params)  
✅ **File parsing**: 29ms is fast for Python I/O

**Leave these alone** - they're not bottlenecks!

---

## Benchmarking Next Steps

1. **Implement Force CPU JAX** (Priority 1, Quick Win)
   - Add `jax.default_device(jax.devices('cpu')[0])` context
   - Benchmark: should save ~350ms
   - Test: Verify accuracy unchanged

2. **Implement Clock File Caching** (Priority 2, Easy)
   - Add simple `_clock_cache` dict
   - Benchmark: saves 56ms on 2nd+ pulsar
   - Test: Fit 10 pulsars, measure total time

3. **Benchmark PINT Ephemeris** (Priority 3, Quick Test)
   - Replace `compute_ssb_obs_pos_vel()` with PINT's version
   - Benchmark: measure if faster
   - Test: Verify residuals match

4. **Profile Again** After Phase 1
   - Run `profile_optimized_fitter.py` again
   - Identify new bottlenecks
   - Iterate

---

## Questions for User

1. **GPU availability**: Do you have CUDA GPU? If not, forcing CPU is a huge win.
2. **Multi-pulsar fitting**: Will you fit many pulsars in one session? (Affects caching strategy)
3. **Precision requirements**: Can we use float32 for some operations? (Risky but could save time)
4. **PINT dependency**: OK to use PINT's ephemeris code? (Fastest option but adds dependency)

---

## Conclusion

**Current performance is excellent** for single-pulsar fitting (3.1s for 10k TOAs), but there's room for improvement:

1. **Quick wins available**: 30% speedup in 1-2 hours (JAX CPU + caching)
2. **Medium-term target**: 50% speedup in 4-8 hours (ephemeris optimization)
3. **Already optimal**: Iteration speed (1ms) is fantastic - don't touch!

**Next action**: Implement "Force CPU JAX" (10 minutes, 350ms savings, zero risk).

---

**Report generated**: 2025-12-01  
**Profiling scripts**: 
- `profile_optimized_fitter.py` (full pipeline)
- `profile_cache_initialization.py` (cache breakdown)
