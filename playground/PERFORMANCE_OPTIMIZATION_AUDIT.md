# Performance Optimization Audit

**Date**: 2025-12-01
**Status**: Initial implementation complete, deeper optimization needed

---

## What Was Implemented (Session 14)

Created `jug/fitting/cached_residuals.py` with:
- `CachedResidualCalculator` class
- `OptimizedFitter` class  
- Infrastructure for caching static components

**Result**: ~22s (no speedup yet - still doing full recomputation each iter)

---

## Why No Speedup Yet?

The caching infrastructure is in place, but we're still calling
`compute_residuals_simple()` which:
1. Parses par file
2. Loads clock files  
3. Computes ephemeris
4. Computes everything from scratch

**We need to go deeper!**

---

## Next Steps for Real Speedup

### Option A: Direct Residual Function (Fastest path to 4-5x)

Create a standalone residual function that takes pre-computed delays:

```python
@jax.jit
def compute_residuals_from_delays(
    t_emission_mjd,  # Pre-computed!
    f0, f1, f2,
    pepoch,
    freqs_mhz,
    dm_delay_cached  # Pre-computed!
):
    # Only recompute phase (fast!)
    dt_sec = (t_emission_mjd - pepoch) * SECS_PER_DAY
    phase = f0 * dt_sec + 0.5 * f1 * dt_sec**2 + (1/6) * f2 * dt_sec**3
    return phase / f0  # Residuals
```

Then cache:
- `t_emission_mjd` (includes all delays except DM if fitting DM)
- `dm_delay` (if not fitting DM)
- Everything static

**Expected**: 21s → 4-5s

### Option B: Modify `simple_calculator.py` (More work)

Add a `compute_residuals_cached()` function that accepts pre-computed components.

**Expected**: 21s → 5-6s

---

## Recommendation

**Implement Option A first** - it's cleaner and gives maximum speedup.

The key is to separate:
- **Static**: Clock, bary delays, binary delays (if not fitting binary)
- **Dynamic**: Only phase calculation

Time estimate: 2-3 hours for full implementation + testing

---

## Status

✅ Infrastructure created  
⏳ Deep caching not yet implemented  
📋 Ready for Session 15 implementation

