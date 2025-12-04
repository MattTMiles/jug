# Piecewise Fitting for Long Datasets - Project Specification

**Date:** 2025-12-02  
**Context:** Precision optimization for datasets >30 years from PEPOCH  
**Status:** Future work - Investigation phase  

---

## Executive Summary

This document specifies a potential enhancement to JUG for handling very long pulsar timing datasets (>30-60 year span) while maintaining float64 precision in JAX. The approach uses piecewise polynomial fitting with physical constraints to ensure F0 and F1 remain consistent across segments.

---

## Background: The Precision Problem

### Current JUG Implementation

**Two-tier precision architecture:**
- **Longdouble (18 digits):** Used for critical subtraction `dt = TDB - PEPOCH - delays`
- **Float64 (15 digits):** Used for everything else (JAX-compiled)

This works excellently for most datasets but has limitations for very long spans.

### Precision vs Dataset Span

**With PEPOCH centered at dataset middle:**

| Total Span | Max dt from PEPOCH | Float64 Precision | vs 100 ns Target |
|------------|-------------------|-------------------|------------------|
| 20 years | ±315M seconds | 70 ns | ✅ 1.4× better |
| 30 years | ±473M seconds | 105 ns | ✅ 1.05× better |
| 40 years | ±631M seconds | 140 ns | ⚠️ 1.4× worse |
| 50 years | ±789M seconds | 175 ns | ⚠️ 1.75× worse |
| 60 years | ±947M seconds | 210 ns | ⚠️ 2.1× worse |

**Precision degrades as:** `epsilon ≈ |dt| × 2.22e-16`

### Why Not Use Longdouble Throughout?

**JAX doesn't support longdouble!**
- ❌ No JIT compilation with longdouble
- ❌ No GPU acceleration
- ❌ 40-50% slower (loses JAX benefits)

**Therefore:** Need a float64-compatible solution for long datasets.

---

## Current Discussion: Session Context

### What We've Established

1. **Float64 is adequate for <20 year datasets** (empirical precision: 3 ns vs PINT)
2. **Centered PEPOCH gives 2× improvement** (standard practice)
3. **Multiple PEPOCHs are complex** (need to adjust F0/F1 per segment)
4. **No simple mathematical transform helps** (all tried, all failed)

### The Insight

**F0 and F1 are physical constants** for the pulsar's spin-down. They should be the same across the entire dataset, even if computed piecewise.

**Key idea:** Use piecewise computation for numerical precision, but enforce physical constraints to recover global F0/F1.

---

## Proposed Approach: Constrained Piecewise Fitting

### Core Concept

Split long dataset into chunks (e.g., thirds), compute phase/residuals locally within each chunk (good float64 precision), but **fit global F0/F1 that must work for all chunks simultaneously**.

### Physical Constraint

The spin-down is continuous:
```
f(t) = F0 + F1*t + F2*t²/2 + ...
```

If we split at times t₁, t₂:
```
Chunk 1: t ∈ [0, t₁]     → Local dt₁, but same global F0, F1
Chunk 2: t ∈ [t₁, t₂]    → Local dt₂, but same global F0, F1  
Chunk 3: t ∈ [t₂, tend]  → Local dt₃, but same global F0, F1
```

The phase must be continuous at boundaries.

---

## Approach 1: Sequential Thirds with Shared F0/F1

### Method

1. **Split dataset into thirds by time**
   ```
   PEPOCH = center of entire dataset (unchanged)
   Chunk 1: TOAs in years [0, 20]
   Chunk 2: TOAs in years [20, 40]
   Chunk 3: TOAs in years [40, 60]
   ```

2. **For each chunk, compute local dt**
   ```
   For TOA in chunk i:
     dt_local[i] = TDB - PEPOCH - delays
   ```
   Same PEPOCH, different TOA subsets → smaller dt range per chunk

3. **Fit global F0, F1 using ALL chunks simultaneously**
   ```python
   # Pseudo-code
   def compute_residuals_all_chunks(F0, F1):
       residuals = []
       for chunk in [chunk1, chunk2, chunk3]:
           dt_chunk = chunk['dt']  # Already computed, cached
           phase = F0 * dt_chunk + F1 * dt_chunk²/2
           phase_wrapped = phase - round(phase)
           res_chunk = phase_wrapped / F0
           residuals.append(res_chunk)
       return concatenate(residuals)
   
   # Fit F0, F1 to minimize residuals across ALL chunks
   F0_global, F1_global = fit_wls(residuals_all_chunks, errors_all)
   ```

### Precision Benefit

**60-year dataset example:**

Without chunking:
- dt ranges: -947M to +947M seconds
- Max |dt|: 947M → precision 210 ns ❌

With thirds (20-year chunks):
- Chunk 1 dt: -947M to -315M → range 631M → precision 140 ns
- Chunk 2 dt: -315M to +315M → range 631M → precision 140 ns
- Chunk 3 dt: +315M to +947M → range 631M → precision 140 ns

Average precision: ~140 ns (vs 210 ns) - **1.5× improvement!**

### Advantages

✅ Same global F0, F1 (physical interpretation preserved)  
✅ Better precision per chunk  
✅ Can still use JAX (float64)  
✅ Straightforward to implement  

### Disadvantages

⚠️ Still not perfect (140 ns vs 100 ns target)  
⚠️ More complex than current approach  
⚠️ Need to manage chunk boundaries  

---

## Approach 2: Piecewise Linear Offsets

### Method

1. **Define reference dt for each chunk**
   ```
   Chunk 1 reference: dt_ref1 = -630M (middle of chunk 1)
   Chunk 2 reference: dt_ref2 = 0       (middle of chunk 2, at PEPOCH)
   Chunk 3 reference: dt_ref3 = +630M  (middle of chunk 3)
   ```

2. **Compute offset from reference**
   ```
   For TOA in chunk i:
     dt_offset[i] = dt[i] - dt_ref[i]
   ```
   Now dt_offset ranges from -315M to +315M per chunk (70 ns precision!)

3. **Phase calculation with offset correction**
   ```
   For chunk i:
     phase[i] = F0*(dt_offset[i] + dt_ref[i]) 
              + F1*(dt_offset[i] + dt_ref[i])²/2
     
     # Expand:
     phase[i] = F0*dt_offset[i] + F0*dt_ref[i]
              + F1*dt_offset²[i]/2 + F1*dt_ref[i]*dt_offset[i] 
              + F1*dt_ref²[i]/2
   ```

4. **Fit global F0, F1 with offset terms**
   ```
   Design matrix includes:
     - dt_offset (high precision)
     - dt_ref (constant per chunk)
   ```

### Precision Benefit

Each chunk has dt_offset with max 315M seconds → **70 ns precision**!

This **meets the 100 ns target** even for 60-year datasets!

### Advantages

✅ Excellent precision (70 ns)  
✅ Same global F0, F1  
✅ Still uses JAX/float64  

### Disadvantages

⚠️ More complex design matrix  
⚠️ Need to handle chunk boundaries carefully  
⚠️ Phase wrapping complications  

---

## Approach 3: Hybrid Precision Mode

### Method

Detect dataset span and switch modes automatically:

```python
def choose_precision_mode(toas, pepoch):
    span_years = (max(toas) - min(toas)) / 365.25
    max_dt_from_pepoch = max(abs(toas - pepoch))
    
    if max_dt_from_pepoch < 15 * 365.25 * 86400:  # <15 years
        return 'float64'  # Standard mode, excellent precision
    elif max_dt_from_pepoch < 30 * 365.25 * 86400:  # 15-30 years
        return 'float64'  # Acceptable precision (70-140 ns)
    else:  # >30 years
        return 'piecewise_float64'  # Use chunking approach
```

**Advantages:**
- ✅ Automatic optimization
- ✅ Users don't need to think about it
- ✅ Best performance for typical datasets
- ✅ Graceful degradation for long datasets

---

## Implementation Considerations

### Chunk Size Selection

**Option A: Fixed time spans**
- 20-year chunks regardless of dataset length
- Simple, predictable

**Option B: Fixed number of chunks**
- Always use 3 chunks (thirds)
- Adapts to dataset length

**Option C: Adaptive**
- Choose chunk size to keep max |dt| < threshold
- Optimal precision, more complex

**Recommendation:** Start with fixed thirds (Option B) for simplicity.

### Boundary Handling

**Critical question:** What happens at chunk boundaries?

TOAs near boundaries might be closer to adjacent chunk's reference. Need to ensure:
1. Phase continuity across boundaries
2. No double-counting of TOAs
3. Derivatives computed correctly

**Potential solution:** Use overlapping windows with smooth weighting.

### Design Matrix Construction

Current (single chunk):
```
M = [d(phase)/d(F0), d(phase)/d(F1)]
  = [-dt/F0, -(dt²/2)/F0]
```

Piecewise (Approach 1):
```
M = concatenate([M_chunk1, M_chunk2, M_chunk3])
```
Same structure, just longer!

Piecewise (Approach 2):
```
M_chunk[i] = [-(dt_offset[i] + dt_ref[i])/F0,
              -((dt_offset[i] + dt_ref[i])²/2)/F0]
```
Need to handle offset terms explicitly.

### Caching Strategy

**Key insight:** dt computation is expensive (delays, barycentric corrections).

With piecewise:
1. Compute dt ONCE for all TOAs (same as current)
2. Split into chunks (cheap - just array slicing)
3. Compute offsets if using Approach 2 (cheap)
4. Cache everything for iterative fitting

**Performance impact:** Minimal! Chunking happens after expensive dt computation.

---

## Testing Strategy

### Phase 1: Validation

1. **Test on synthetic data**
   - Generate 60-year dataset with known F0, F1
   - Add realistic noise
   - Compare piecewise vs standard fitting
   - Should recover same F0, F1

2. **Test on real data with artificial extension**
   - Take existing 20-year dataset
   - Extrapolate to 60 years using known F0, F1
   - Fit with both methods
   - Compare results

3. **Precision measurement**
   - Compare with PINT on long datasets
   - Measure residual differences
   - Should achieve <100 ns precision

### Phase 2: Edge Cases

1. **Uneven TOA distribution**
   - What if one chunk has 10× more TOAs?
   - Should still work (weighted by uncertainties)

2. **Glitches at boundaries**
   - What if real glitch occurs at chunk boundary?
   - Need to distinguish from numerical artifacts

3. **Very long datasets**
   - Test with 100-year synthetic dataset
   - Ensure scales to arbitrary length

---

## Integration with Existing Code

### Minimal Changes Required

**Current fitting loop (simplified):**
```python
# Level 1: Cache dt_sec (expensive)
dt_sec = compute_dt_once(toas, params)

# Level 2: Iterative fitting (fast)
for iteration in range(max_iter):
    # Compute phase
    phase = F0 * dt_sec + F1 * dt_sec²/2
    
    # Compute residuals
    residuals = (phase - round(phase)) / F0
    
    # Compute derivatives
    M = compute_derivatives(dt_sec, F0)
    
    # WLS solve
    delta_params = wls_solve(residuals, errors, M)
    
    # Update
    F0 += delta_params[0]
    F1 += delta_params[1]
```

**Piecewise fitting (Approach 1):**
```python
# Level 1: Cache dt_sec (expensive, same as before)
dt_sec = compute_dt_once(toas, params)

# NEW: Split into chunks
chunks = split_into_thirds(dt_sec, toas, errors)

# Level 2: Iterative fitting (fast)
for iteration in range(max_iter):
    residuals_all = []
    M_all = []
    errors_all = []
    
    # Loop over chunks
    for chunk in chunks:
        # Same calculations as before, per chunk
        phase = F0 * chunk.dt + F1 * chunk.dt²/2
        residuals = (phase - round(phase)) / F0
        M = compute_derivatives(chunk.dt, F0)
        
        residuals_all.append(residuals)
        M_all.append(M)
        errors_all.append(chunk.errors)
    
    # Concatenate
    residuals_all = np.concatenate(residuals_all)
    M_all = np.concatenate(M_all)
    errors_all = np.concatenate(errors_all)
    
    # WLS solve (same as before)
    delta_params = wls_solve(residuals_all, errors_all, M_all)
    
    # Update (same as before)
    F0 += delta_params[0]
    F1 += delta_params[1]
```

**Changes needed:**
1. Add `split_into_chunks()` function
2. Add loop over chunks in fitting iteration
3. Concatenate results before WLS solve
4. That's it!

### File Modifications

**Minimal changes to:**
- `jug/fitting/optimized_fitter.py` - Add chunking logic
- `jug/fitting/derivatives_spin.py` - Already works per-array
- `jug/fitting/wls_fitter.py` - No changes needed!

**New files:**
- `jug/fitting/piecewise_utils.py` - Chunking utilities

---

## Success Criteria

### Must Have

1. ✅ Recovers same F0, F1 as single-chunk fitting (within uncertainties)
2. ✅ Improves precision for >30 year datasets
3. ✅ Maintains JAX/float64 (no longdouble dependency)
4. ✅ Performance degradation <20%
5. ✅ Passes all existing tests

### Nice to Have

1. ⭐ Automatic mode selection based on dataset span
2. ⭐ Configurable chunk size
3. ⭐ Works with future delay parameter fitting (DM, position, binary)
4. ⭐ Visualization showing chunk boundaries and precision

---

## Open Questions for Investigation

### Physics Questions

1. **Does piecewise fitting introduce systematic errors?**
   - Need to validate on known pulsars with long datasets
   - Compare with PINT's longdouble approach

2. **How does this interact with timing noise?**
   - Real pulsars have timing noise (phase jitter)
   - Could chunking mistake noise for signal?

3. **What about glitches?**
   - Need to ensure glitches aren't masked by chunk boundaries
   - Might need glitch detection first

4. **Does this work for all parameter types?**
   - **YES!** All parameters (F0, DM, RAJ, PB, etc.) are global constants
   - Piecewise is purely a numerical strategy for precision
   - We fit the SAME GLOBAL VALUE to all chunks simultaneously
   - Difference: Delay parameters (DM, RAJ, PB) might need dt recomputation
   - But they're still single global constants across the dataset!

### Numerical Questions

1. **What's the optimal chunk size?**
   - Too large: precision degrades
   - Too small: boundary effects dominate
   - Need empirical testing

2. **How to handle phase wrapping at boundaries?**
   - Phase must be continuous
   - But we wrap to [-0.5, 0.5] cycles
   - Need careful boundary treatment

3. **When to recompute dt for delay parameters?**
   - F0, F1: Never (don't affect delays)
   - DM, RAJ, PB, etc.: When they change significantly during fitting
   - Strategy: Recompute every N iterations, or when residuals stop improving
   - Note: All parameters remain global constants across chunks!

### Implementation Questions

1. **JAX compatibility?**
   - Can we JIT compile chunked operations?
   - Or does chunking force Python loops?

2. **Memory usage?**
   - Does storing multiple chunks increase memory?
   - Probably negligible (just slicing arrays)

3. **User interface?**
   - Should this be automatic or user-controlled?
   - How to document this feature?

---

## Recommended Development Path

### Phase 1: Proof of Concept (1-2 weeks)

1. Implement Approach 1 (simple thirds) in standalone script
2. Test on synthetic 60-year dataset
3. Verify F0, F1 recovery and precision improvement
4. Document results

**Deliverable:** Working proof-of-concept with validation results

### Phase 2: Integration (1-2 weeks)

1. Add chunking to `optimized_fitter.py`
2. Add mode detection (auto-enable for >30 year datasets)
3. Add tests
4. Documentation

**Deliverable:** Production-ready feature with tests

### Phase 3: Optimization (1 week)

1. Benchmark performance
2. Optimize chunking logic if needed
3. Add configuration options
4. User guide

**Deliverable:** Polished feature ready for users

### Phase 4: Advanced Features (optional)

1. Implement Approach 2 (offset-based)
2. Adaptive chunk sizing
3. Visualization tools
4. Extended validation

**Deliverable:** Research paper comparing approaches

---

## Alternative: Document the Limitation

**Simplest option:** Don't implement piecewise fitting at all!

Instead, document:

> "JUG is optimized for datasets spanning <30 years from PEPOCH (<60 year total span with centered PEPOCH), achieving <100 ns precision. For longer datasets, numerical precision degrades to ~150-250 ns due to float64 limits. This covers >99% of published pulsar timing datasets. For exceptionally long datasets (>60 years), use PINT or contact developers."

**Advantages:**
- ✅ Zero development time
- ✅ Zero maintenance burden
- ✅ Keeps code simple and fast
- ✅ Honest about limitations

**Disadvantages:**
- ❌ Doesn't push the envelope
- ❌ Might lose potential users with very long datasets

---

## References & Context

### Related Documents

- `PRECISION_AUDIT_REPORT.md` - Complete precision analysis
- `DELAY_PRECISION_FLOW.md` - How delays and dt are computed
- `EMPIRICAL_PRECISION_EXPLAINED.md` - 3 ns empirical precision meaning
- `SESSION_14_COMPLETE_SUMMARY.md` - F0/F1 fitting implementation

### Key Papers

- Edwards et al. (2006) - Tempo2 precision timing
- Hobbs et al. (2009) - TEMPO2: New timing software
- Luo et al. (2021) - PINT precision timing package
- NANOGrav 15-year data: Example of long-baseline datasets

### Contact

**Original implementer:** (to be filled in)  
**AI Assistant:** Claude (Anthropic)  
**Session date:** 2025-12-02

---

## Summary for Next AI Agent

**What we're trying to do:**
Extend JUG's float64-based fitting to handle very long (>60 year) pulsar timing datasets without sacrificing precision or JAX performance.

**Why it's hard:**
Float64 precision degrades with |dt| magnitude. At 60 years from PEPOCH, precision is ~210 ns (vs 100 ns target).

**The key insight:**
F0 and F1 are physical constants (pulsar spin-down rate). Even though we compute piecewise for numerical precision, the final fitted F0/F1 must be the same global values.

**What to explore:**
Split dataset into chunks (thirds), compute phase/residuals per chunk (good float64 precision), but fit global F0/F1 that satisfies all chunks simultaneously.

**Start here:**
Implement Approach 1 (simple thirds) as proof-of-concept. Validate on synthetic 60-year dataset. Measure precision improvement.

**Questions to answer:**
1. Does this actually improve precision? (target: <100 ns)
2. Does it recover correct F0/F1? (compare to single-chunk)
3. What's the performance cost? (target: <20% slower)
4. How to handle chunk boundaries? (phase continuity)

Good luck! 🚀
