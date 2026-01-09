# Handoff Document: Piecewise/JAX Fitting Investigation

**Date:** 2026-01-09
**Status:** Testing hypothesis about chunk size vs precision
**Goal:** Enable JAX-based fitting with longdouble-equivalent precision

---

## Executive Summary

We're investigating whether a **piecewise method with small chunks** can achieve sub-nanosecond precision while being JAX-compatible (float64 only). The hypothesis is that smaller chunks (100 TOAs vs 500 days) will reduce precision errors.

**Ultimate Goal:** Pure JAX fitting with:
- ✅ Sub-ns precision (matches longdouble)
- ✅ 10× faster than numpy/longdouble
- ✅ Autodiff for derivatives (no manual implementation)
- ✅ No drift across timespan

---

## Current Status

### What We've Proven

1. **Longdouble single PEPOCH works** (0.18 ms, perfect precision)
2. **Numpy/longdouble hybrid with small chunks works** (0.001 ns error!)
3. **JAX hybrid attempts have 22.5 ns error** (1000× worse than longdouble)
4. **Piecewise with 500-day segments has 23 ns + 1.7× drift** (unacceptable)

### The Problem

**JAX doesn't support longdouble** - only float64. So methods that achieve sub-ns precision in longdouble degrade to ~22 ns in JAX float64.

### The Hypothesis (User's Insight)

**Smaller chunks should reduce precision loss!**

| Chunk Size | dt_epoch | Expected Precision |
|------------|----------|-------------------|
| 500 days (~2000 TOAs) | ~4.3×10^7 s | 23 ns + drift ❌ |
| 100 TOAs (~16 days) | ~1.4×10^6 s | ??? (testing now) |
| 25 TOAs (~4 days) | ~3.5×10^5 s | ??? (testing now) |
| 10 TOAs (~1.6 days) | ~1.4×10^5 s | ??? (testing now) |

**Key insight:** By using local PEPOCHs per chunk, we center the data and minimize dt_epoch. Smaller dt_epoch → smaller phase_offset values → less float64 precision loss.

---

## The Piecewise Method Explained

### Algorithm

```python
# For each chunk of TOAs:
for chunk in chunks:
    # 1. Set local PEPOCH at chunk center
    local_pepoch = mean(tdb_mjd[chunk])
    dt_epoch = (local_pepoch - global_pepoch) * 86400  # seconds

    # 2. Transform F0 to local frame (continuity constraint)
    f0_local = f0_global + f1_global * dt_epoch

    # 3. Compute local time coordinates (SMALL values!)
    dt_local = dt_sec[chunk] - dt_epoch

    # 4. Compute phase in local coordinates
    phase_local = dt_local * (f0_local + dt_local * (f1_global / 2))

    # 5. Add phase offset to restore global frame
    phase_offset = f0_global * dt_epoch + (f1_global / 2) * dt_epoch^2
    phase_corrected = phase_local + phase_offset

    # 6. Wrap and convert to residuals
    phase_wrapped = phase_corrected - round(phase_corrected)
    residuals[chunk] = phase_wrapped / f0_global
```

### Why This Might Work

**The phase_offset calculation is where precision is lost:**
```
phase_offset = F0 * dt_epoch + (F1/2) * dt_epoch^2
             = 339 Hz × dt_epoch + ...

For 500-day chunks:  dt_epoch ~ 4.3×10^7 s → phase_offset ~ 1.5×10^10 cycles
For 100-TOA chunks: dt_epoch ~ 1.4×10^6 s → phase_offset ~ 4.7×10^8 cycles
For 25-TOA chunks:  dt_epoch ~ 3.5×10^5 s → phase_offset ~ 1.2×10^8 cycles
```

**Float64 precision at different scales:**
- 10^10 cycles: ~100 ns precision loss
- 10^8 cycles: ~1 ns precision loss
- 10^6 cycles: ~0.01 ns precision loss

---

## Tests Completed

### Test 1: `test_piecewise_comparison_fresh.py`

**Result:**
- Piecewise (500-day): 23 ns max error, 1.7× spreading
- Hybrid (100-TOA chunks): 0.022 ns in longdouble ✅, 22.5 ns in JAX ❌

**Conclusion:** Hybrid chunking works in longdouble but not in JAX float64.

### Test 2: `test_hybrid_jax_compatibility.py`

**Result:**
- JAX hybrid (no wrapping): 22.5 ns
- JAX hybrid (with per-chunk wrapping): 22.5 ns (NO improvement)
- Autodiff works ✅
- 10× faster than longdouble ✅

**Conclusion:** Per-chunk wrapping doesn't help in float64 - the precision loss is in phase_offset calculation, not wrapping.

### Test 3: `test_hybrid_jax_NEW.py`

**Result:**
- Numpy/LD hybrid (old, no wrap): 0.001 ns ✅ (perfect!)
- Numpy/LD hybrid (NEW, with wrap): 11.242 ns (worse!)
- JAX hybrid (NEW, with wrap): 22.485 ns (no improvement)

**Conclusion:** The "NEW" wrapping method from notebook doesn't improve float64 precision.

---

## Current Hypothesis (TO TEST NOW)

**Smaller piecewise chunks will reduce errors:**

The user hypothesizes that if we use **100-TOA chunks** (instead of 500-day segments), the dt_epoch values will be ~30× smaller, reducing phase_offset precision loss.

**Test Plan:**
1. Implement piecewise method with various chunk sizes: 500 days, 100 TOAs, 50 TOAs, 25 TOAs, 10 TOAs
2. Measure precision vs chunk size
3. Check if spreading/drift decreases with smaller chunks
4. Test in both numpy/longdouble AND JAX float64
5. Plot: chunk size vs error, chunk size vs spreading ratio

**Expected outcome if hypothesis is correct:**
- Smaller chunks → smaller errors
- At some chunk size, errors approach sub-ns levels
- Spreading ratio approaches 1.0 (no drift)
- If this works in numpy/longdouble, proves the concept
- Then test if JAX float64 can achieve similar results

---

## Why This Matters for JAX

**If piecewise with small chunks achieves sub-ns precision:**

1. **Proves the concept** - small local coordinates work
2. **Tests if float64 is sufficient** - at what chunk size?
3. **Enables JAX implementation** - if 100-TOA chunks work in float64
4. **Opens door to pure JAX fitting:**
   - 10× faster iterations
   - Autodiff for derivatives
   - GPU acceleration potential
   - No manual derivative implementation

**The critical question:**
> At what chunk size does float64 piecewise match longdouble single PEPOCH precision?

If the answer is "100 TOAs" or "50 TOAs", we can use this in JAX!

---

## Files to Review

**Test Scripts (chronological):**
1. `test_piecewise_comparison_fresh.py` - Original comparison (500-day piecewise)
2. `test_hybrid_jax_compatibility.py` - JAX compatibility test
3. `test_hybrid_jax_NEW.py` - Per-chunk wrapping test
4. **NEXT:** `test_piecewise_chunk_sizes.py` - Testing user's hypothesis

**Documentation:**
1. `CHECKPOINT_PIECEWISE_INVESTIGATION.md` - Previous checkpoint
2. `HANDOFF_PIECEWISE_JAX_INVESTIGATION.md` - This document
3. `/docs/JUG_PROGRESS_TRACKER.md` - Overall project status

**Notebooks:**
1. `playground/piecewise_fitting_implementation.ipynb` - Original piecewise work
   - Section 14 has "NEW Hybrid" method with per-chunk wrapping
   - This is where the longdouble wrapping idea came from

**Key Code:**
1. `jug/fitting/optimized_fitter.py` - Current production fitter (uses longdouble)
2. `jug/residuals/simple_calculator.py` - Residual computation

---

## How to Continue

### Immediate Next Step

**Create test:** `test_piecewise_chunk_sizes.py`

This test should:
1. Load J1909-3744 data (10,408 TOAs, 6.33 year span)
2. Implement piecewise method with configurable chunk size
3. Test chunk sizes: [500 days, 200 TOAs, 100 TOAs, 50 TOAs, 25 TOAs, 10 TOAs]
4. For each chunk size, measure:
   - Max error vs longdouble baseline
   - Spreading ratio (early vs late data)
   - Computation time
5. Test in BOTH numpy/longdouble AND JAX float64
6. Generate plots:
   - Error vs chunk size
   - Spreading ratio vs chunk size
   - Residual difference plots for each method
   - Time-dependent spreading for each chunk size

### If Hypothesis is Confirmed

If smaller chunks → smaller errors in float64:

1. **Find optimal chunk size** - where error plateaus
2. **Test JAX implementation** with that chunk size
3. **Integrate into fitter** - replace longdouble mode
4. **Benchmark fitting convergence** - ensure it works identically
5. **Update documentation**

### If Hypothesis is Rejected

If errors remain constant regardless of chunk size:

**Options:**
1. Accept 22.5 ns precision with JAX (still good for science)
2. Stick with longdouble single PEPOCH (current production)
3. Use numpy/longdouble hybrid (0.001 ns) without JAX speedup
4. Investigate other approaches (different coordinate systems?)

---

## Key Equations

### Phase Calculation
```
φ = F0 × dt + (F1/2) × dt²
```

### Piecewise Coordinate Transformation
```
dt_epoch = (local_pepoch - global_pepoch) × 86400
F0_local = F0_global + F1_global × dt_epoch
dt_local = dt - dt_epoch

phase_local = dt_local × (F0_local + dt_local × F1/2)
phase_offset = F0_global × dt_epoch + (F1_global/2) × dt_epoch²
phase_total = phase_local + phase_offset
```

### Design Matrix (for fitting)
```
∂φ/∂F0 = dt_local + dt_epoch
∂φ/∂F1 = dt_epoch × dt_local + dt_local²/2 + dt_epoch²/2
M[:,0] = -∂φ/∂F0 / F0_global
M[:,1] = -∂φ/∂F1 / F0_global
```

---

## Success Criteria

**For piecewise method to be viable:**

| Criterion | Target | Why |
|-----------|--------|-----|
| Max error | <1 ns | Match longdouble precision |
| Spreading ratio | <1.5× | No systematic drift |
| Speed | <2× longdouble | Acceptable overhead |
| JAX compatible | Yes | Enable autodiff + GPU |
| Fitting convergence | Identical | Must not affect science |

**Minimum acceptable:**
- Max error <50 ns (better than TOA errors)
- Spreading ratio <2× (minimal drift)
- Works in JAX float64

---

## Questions to Answer

1. **Does chunk size affect precision?** (Testing now)
2. **Is there an optimal chunk size?** (e.g., 50 TOAs?)
3. **Can JAX float64 match longdouble at optimal chunk size?**
4. **What's the performance trade-off?** (smaller chunks = more overhead)
5. **Does fitting converge identically?** (must validate)

---

## Context for New Agent

**Read these first:**
1. This document (HANDOFF_PIECEWISE_JAX_INVESTIGATION.md)
2. `CHECKPOINT_PIECEWISE_INVESTIGATION.md` - background
3. Run `test_piecewise_chunk_sizes.py` (to be created)
4. Review plots: `piecewise_chunk_size_analysis.png`

**What we're trying to solve:**
- JUG currently uses longdouble for spin parameters (F0/F1/F2)
- This prevents pure JAX implementation (JAX only has float64)
- We need a method that achieves <1 ns precision in JAX float64
- User hypothesizes that piecewise with small chunks will work

**Why it matters:**
- Would enable 10-60× faster fitting
- Autodiff for derivatives (no manual implementation)
- GPU acceleration potential
- Simpler code (pure JAX instead of numpy/longdouble mix)

---

## Running the Tests

```bash
cd /home/mmiles/soft/jug
ml conda
mamba activate discotech

# Test chunk size hypothesis
python test_piecewise_chunk_sizes.py

# Previous tests (for reference)
python test_piecewise_comparison_fresh.py
python test_hybrid_jax_compatibility.py
python test_hybrid_jax_NEW.py
```

---

## Contact Information

**Original investigation:** User + Claude (2026-01-09)
**Files location:** `/home/mmiles/soft/jug/`
**Data:** `data/pulsars/J1909-3744_tdb.par` and `.tim`

---

**Status:** ⏳ AWAITING test_piecewise_chunk_sizes.py results
**Next milestone:** Determine if chunk size hypothesis is correct
**Decision point:** Choose final implementation based on results

---

## Quick Start for Resuming

```bash
# 1. Read this document
# 2. Review checkpoint
cat CHECKPOINT_PIECEWISE_INVESTIGATION.md

# 3. Run the chunk size test
python test_piecewise_chunk_sizes.py

# 4. Analyze results
# Look at: piecewise_chunk_size_analysis.png
# Check: Does error decrease with smaller chunks?
# Check: Does spreading ratio approach 1.0?

# 5. If hypothesis confirmed:
#    - Find optimal chunk size
#    - Test JAX implementation at that size
#    - Integrate into fitter

# 6. If hypothesis rejected:
#    - Decide on fallback approach
#    - Update documentation
```

---

**END OF HANDOFF DOCUMENT**
