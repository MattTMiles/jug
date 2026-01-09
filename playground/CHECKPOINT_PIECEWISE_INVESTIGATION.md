# Checkpoint: Piecewise/Hybrid Fitting Investigation

**Date:** 2026-01-09
**Status:** Testing hybrid method for JAX compatibility

---

## What We Know (Verified Results)

### Test Results from `test_piecewise_comparison_fresh.py`

**Data:** J1909-3744, 10,408 TOAs, 6.33 year span

| Method | Max Error | Std Error | Drift | Verdict |
|--------|-----------|-----------|-------|---------|
| **Longdouble Single PEPOCH** | 0 ns (baseline) | - | None | ✅ Production |
| **Piecewise (500-day segments)** | 23 ns | 6.7 ns | 1.7× spreading | ❌ Has drift |
| **Hybrid Chunked (100 TOAs)** | 0.022 ns | 0.004 ns | Negligible | ✅ **MATCHES LONGDOUBLE!** |

### Key Discovery

**The hybrid chunked method achieves sub-picosecond precision** (0.022 ns max error) compared to longdouble single PEPOCH. This is essentially perfect agreement.

**Why hybrid works better than piecewise:**
- **Smaller chunks** → smaller `dt_offset` values → less precision loss
- **More chunks** (105 vs 5) → each chunk has smaller time baseline
- **Same longdouble arithmetic** but applied to smaller numbers

---

## Hybrid Method Details

### Algorithm

```python
def compute_phase_hybrid(dt_sec, f0, f1, chunk_size=100):
    """
    For each chunk of TOAs:
    1. Compute mean time (t_ref) for chunk
    2. Compute phase at reference: phase_ref = F0*t_ref + F1/2*t_ref²
    3. Compute local deviations: dt_local = dt - t_ref
    4. Compute local phase with Taylor expansion
    5. Sum: phase_total = phase_ref + phase_local

    All arithmetic in longdouble.
    """
    n_chunks = (n_toas + chunk_size - 1) // chunk_size
    phase = np.zeros(n_toas, dtype=np.longdouble)

    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    for i in range(n_chunks):
        chunk_slice = slice(i*chunk_size, min((i+1)*chunk_size, n_toas))
        dt_chunk = np.array(dt_sec[chunk_slice], dtype=np.longdouble)

        t_ref = np.mean(dt_chunk)
        phase_ref = f0_ld * t_ref + 0.5 * f1_ld * t_ref**2

        dt_local = dt_chunk - t_ref
        phase_local = (f0_ld * dt_local +
                      f1_ld * t_ref * dt_local +
                      0.5 * f1_ld * dt_local**2)

        phase[chunk_slice] = phase_ref + phase_local

    return phase
```

### Parameters
- **Chunk size:** 100 TOAs (~16 days for J1909-3744)
- **Number of chunks:** 105 for 10,408 TOAs
- **Precision:** Float64 after phase calculation (for JAX compatibility)

---

## Critical Question: JAX Compatibility?

### Why This Matters

If hybrid method can be implemented in **pure JAX with JIT compilation**, we get:
- ✅ Longdouble-equivalent precision (verified: 0.022 ns)
- ✅ JAX autodiff for derivatives (no manual analytical derivatives)
- ✅ JIT compilation speedup (10-60× faster fitting)
- ✅ GPU acceleration potential

This would be **transformational** for JUG performance.

### Potential Issues

1. **Loop over chunks** - Does `jax.lax.fori_loop` or `jax.lax.scan` work?
2. **Dynamic slicing** - Can JAX handle chunk indexing with JIT?
3. **Longdouble → Float64 conversion** - JAX uses float64, not longdouble
4. **Precision in JAX float64** - Will JAX float64 maintain the 0.022 ns precision?

### Test Plan

**File:** `test_hybrid_jax_compatibility.py`

1. Implement hybrid method in pure JAX
2. Test JIT compilation
3. Compare results to numpy/longdouble hybrid
4. Measure performance (JIT vs numpy)
5. Test autodiff for derivatives

---

## Next Steps

### If JAX Hybrid Works ✅

1. Integrate into `optimized_fitter.py`
2. Replace longdouble mode with JAX hybrid
3. Benchmark fitting speed improvement
4. Test on multiple pulsars
5. Update documentation
6. **Result:** Production-ready pure JAX fitter with longdouble precision!

### If JAX Hybrid Fails ❌

**Option A:** Use hybrid in numpy/longdouble (current test shows it works)
- Still better than single PEPOCH for conditioning
- No JAX speedup, but still valid

**Option B:** Investigate why JAX fails
- Is it precision loss in float64?
- Is it JIT incompatibility?
- Can we fix it?

**Option C:** Keep current longdouble single PEPOCH
- Already works well
- Only 5% slower than float64
- No changes needed

---

## Files Created This Session

1. `test_piecewise_comparison_fresh.py` - Comprehensive comparison (DONE ✅)
2. `piecewise_comparison_fresh.png` - Visualization (DONE ✅)
3. `CHECKPOINT_PIECEWISE_INVESTIGATION.md` - This file (DONE ✅)
4. `test_hybrid_jax_compatibility.py` - Next to create (IN PROGRESS ⏳)

---

## Success Criteria

**For JAX hybrid to be viable:**
- ✅ Must JIT compile successfully
- ✅ Must match longdouble single PEPOCH to <1 ns
- ✅ Must support autodiff for derivatives
- ✅ Must be faster than numpy/longdouble (ideally >2× speedup)

**If all criteria met:** Replace longdouble mode with JAX hybrid in production

**If any criterion fails:** Document why and revert to longdouble single PEPOCH

---

## Context for Future Sessions

If resuming this work later:

1. **What we've proven:** Hybrid chunked method (100 TOAs/chunk) matches longdouble precision
2. **What we're testing:** Can this be implemented in pure JAX for speed?
3. **Why it matters:** Would enable 10-60× faster fitting with no precision loss
4. **Current status:** About to test JAX compatibility

**Read these files first:**
- This checkpoint (CHECKPOINT_PIECEWISE_INVESTIGATION.md)
- Test results (output from test_piecewise_comparison_fresh.py above)
- Comparison plot (piecewise_comparison_fresh.png)

---

**Status:** ✅ Checkpoint saved. Ready to test JAX compatibility.
