# JUG Performance Improvements

## Overview

JUG has been extensively optimized for speed while maintaining **bit-for-bit identical** scientific results. This document summarizes all performance improvements.

---

## Quick Summary

| Optimization | Impact | Implementation Date |
|--------------|--------|---------------------|
| Cached Fitting | 257x faster subsequent fits | 2026-01 |
| Geometry Disk Cache | 4.5x faster warm session | 2026-01-29 |
| JAX Compilation Cache | Faster cold starts | 2026-01-29 |
| Session Caching | 5x faster fitting | 2026-01 |
| Astropy Config | No surprise downloads | 2026-01-29 |

---

## 1. Cached Fitting (257x Faster!)

### Before
```
Load GUI → Fit parameters → Wait 3 seconds → See results
Adjust parameter → Refit → Wait 3 seconds again → See results
```

### After
```
Load GUI → Fit parameters → Wait 0.8 seconds → See results
Adjust parameter → Refit → Wait 0.01 seconds → See results  ⚡
```

### How It Works
The optimization eliminates redundant work by caching expensive computations:

1. **First Time:** Parse files + compute delays (~0.8s one-time cost)
2. **Every Fit After:** Just run the iteration loop (~0.01s)

### Guarantee
✅ **Results are bit-for-bit identical** (proven by regression tests)

---

## 2. Geometry Disk Cache (4.5x Faster Warm Session)

### Problem
`compute_ssb_obs_pos_vel` took ~580ms every time (Astropy ephemeris + GCRS transforms).

### Solution
Cache geometry arrays to disk, keyed by:
- TDB times hash
- Observatory coordinates hash
- Ephemeris selection (e.g., "de440")
- Astropy/erfa versions

### Results
| Metric | Before | After |
|--------|--------|-------|
| compute_ssb_obs_pos_vel (miss) | 580ms | 580ms |
| compute_ssb_obs_pos_vel (hit) | N/A | **0.75ms** |
| Warm session total | 736ms | **162ms** |

### Files
- `jug/utils/geom_cache.py` - Disk cache implementation
- `jug/delays/barycentric.py` - Uses cache automatically

### Environment Variables
- `JUG_GEOM_CACHE_DIR`: Override cache location (default: `~/.cache/jug/geometry/`)
- `JUG_GEOM_CACHE_DISABLE=1`: Disable caching

---

## 3. JAX Compilation Cache (Faster Cold Starts)

### Problem
JAX JIT compilation takes ~600ms on first call per function.

### Solution
Enable JAX persistent compilation cache:
```python
from jug.utils.jax_cache import configure_jax_compilation_cache
configure_jax_compilation_cache()  # Called early in CLI/GUI entry points
```

### Environment Variables
- `JUG_JAX_CACHE_DIR`: Override cache location
- `JUG_JAX_EXPLAIN_CACHE_MISSES=1`: Debug cache behavior

---

## 4. Session Caching

### Before (v0.1)
Every `fit_parameters()` call re-parsed files and re-computed delays.

### After
`TimingSession` caches:
- Parsed parameters
- TDB times
- Frequency arrays
- Precomputed delays
- Geometry products

Subsequent fits skip all setup work.

---

## 5. Astropy Configuration

### Problem
Astropy can trigger IERS table downloads during coordinate transforms, causing unpredictable delays.

### Solution
Configure Astropy deterministically at startup:
```python
from jug.utils.astropy_config import configure_astropy
configure_astropy()  # Called early in CLI/GUI entry points
```

### Features
- Prevents surprise downloads during operations
- Optional prefetch: `python -m jug.scripts.download_data`
- Force offline: `JUG_ASTROPY_OFFLINE=1`

---

## Benchmark Results

### Interactive Workflow (10,408 TOAs)

```
Session creation:      0.030s  (one-time: parse files)
Cache population:      0.777s  (one-time: first residuals)
Fit time:              0.012s  (repeatable: uses cached arrays!)
Postfit residuals:     0.000s  (instant: fast evaluator)

Total for workflow:    0.818s
```

### Speedup Analysis

**For a Single Fit:**
- Old path: 3.017s
- New path (after cache): **0.012s**
- **Speedup: 257x faster!**

**For Complete GUI Workflow:**
1. Load data (one-time): 0.030s + 0.777s = 0.807s
2. Fit parameters (repeatable): 0.012s
3. View postfit (repeatable): 0.000s
4. Adjust and refit (repeatable): 0.012s each time

---

## Profiling & Debugging

### Geometry Call Profiling
```bash
JUG_PROFILE_GEOM=1 jug-fit pulsar.par pulsar.tim --fit F0 F1
```

Reports:
- Call count
- Total time
- Call sites (limited stack trace)

### Cache Status
```bash
python -m jug.scripts.download_data --status
```

Shows:
- IERS cache status
- JAX compilation cache
- Geometry disk cache (entry count, size)

---

## Testing

### Run All Performance-Related Tests
```bash
# Bit-for-bit regression test
python -c "from jug.tests.test_cached_fitting import run_tests; run_tests()"

# Geometry cache tests
python -c "from jug.tests.test_geom_cache import run_all_tests; run_all_tests()"

# Stats consistency tests
python -c "from jug.tests.test_stats import run_all_tests; run_all_tests()"

# Full equivalence suite
python -c "from jug.tests.test_equivalence import run_all_tests; run_all_tests()"

# Interactive benchmark
python -m jug.scripts.benchmark_interactive
```

---

## What Was NOT Changed

Following strict accuracy requirements:

1. ✓ **No math changes** - iteration loop is byte-for-byte identical
2. ✓ **No solver changes** - still uses WLS + SVD
3. ✓ **No derivative reordering** - same column-by-column design matrix build
4. ✓ **Backwards compatible** - existing CLI and file-based paths work unchanged
5. ✓ **Deterministic** - bit-for-bit identical output proven by tests

---

## Files Modified/Created

### New Files
- `jug/utils/jax_cache.py` - JAX compilation cache
- `jug/utils/geom_cache.py` - Geometry disk cache
- `jug/utils/astropy_config.py` - Astropy configuration
- `jug/engine/stats.py` - Canonical statistics
- `jug/scripts/download_data.py` - Data prefetch
- `jug/tests/test_geom_cache.py` - Cache tests
- `jug/tests/test_stats.py` - Stats tests

### Modified Files
- `jug/delays/barycentric.py` - Disk cache integration
- `jug/engine/session.py` - Session caching
- `jug/fitting/optimized_fitter.py` - Setup/iterate separation
- `jug/gui/main_window.py` - Canonical stats usage
- `jug/gui/main.py` - Early config calls
- `jug/scripts/*.py` - Early config calls

---

**Bottom line:** JUG is now extremely fast while maintaining exact scientific accuracy. Enjoy! 🎉
