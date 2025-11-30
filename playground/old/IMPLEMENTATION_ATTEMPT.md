# Implementation Attempt Summary

## What Was Attempted

I implemented a new `residuals_tempo2_style()` function based on the analysis that showed JUG's methodology was fundamentally different from Tempo2's.

**The approach:**
1. Created a time-based residual function that doesn't pre-apply delays to the time coordinate
2. Computes all delays separately (DM, barycentric, Shapiro, binary, clock)
3. Sums total delay and computes predicted arrival time
4. Computes phase at predicted time and extracts fractional component

**Result: Did NOT fix the 1000x offset**

## Why It Didn't Work

The investigation revealed something more complex:

1. **The time-based method gave different (but still wrong) residuals:**
   - Old method: ~850 μs RMS
   - New method: ~670 μs RMS  (only 0.4% improvement)
   - Tempo2: ~0.82 μs RMS
   - Still off by ~1000x!

2. **Discovery of a factor-of-2 discrepancy:**
   - Manual calculation of fractional phase: 0.244 cycles → 719.5 μs
   - Notebook result: 1414.6 μs (exactly 1.97x larger)
   - This suggests res_us_topo is computing something different than what the residual function defines

3. **The barycentric delay confusion:**
   - bary_delay_sec values are ~-374 seconds (huge!)
   - But time differences show only ~+0.38 seconds
   - This indicates the stored delays might be in a different frame or convention

##Root Issues Identified

1. **Unclear delay semantics:** The delays stored in `bary_delay_sec`, `dm_sec`, etc. may not be directly usable in the standard Tempo2 residual formula without additional context or sign corrections.

2. **Pre-correction incompatibility:** JUG pre-computes delays and applies them to time coordinates before computing residuals. This breaks the standard Tempo2 approach where delays are computed from delay functions.

3. **Wrapped phase state:** The `model.phase_offset_cycles` and how it interacts with the phase computation creates a non-standard residual that isn't directly comparable to Tempo2.

4. **Fractional phase extraction:** There may be additional processing (frequency dependence, profile evolution, etc.) that happens between the spin phase and the final residual value in the current notebook.

## What Would Be Needed to Truly Fix This

1. **Deep code review of current residual calculation:**
   - Trace through cells 19-20 to understand exactly where 1414 μs comes from
   - Check if there are multiple sources adding to the final residual
   - Verify all subtraction/addition operations

2. **Reverse-engineer Tempo2's exact approach:**
   - Get the Tempo2 source code for residual calculation
   - Match every step of the algorithm exactly
   - Consider Tempo2's treatment of: DM, phase wrapping, frequency dependence, observation coordinates

3. **Simplify from first principles:**
   - Start with a minimal example: single TOA, no binary, no DM
   - Compare JUG and Tempo2 step-by-step
   - Add complexity gradually

4. **Use PINT as intermediary:**
   - PINT has well-documented residual calculation
   - JUG could compute via PINT's method first
   - Then optimize/JAX-compile

## Recommendation

**Don't try to "fix" by swapping methods.** Instead:

1. **Identify what's in `res_us_topo`:** What calculation does it actually represent?
2. **Factor-of-2 investigation:** Why is the notebook's result 2x the phase-based calculation?
3. **Direct Tempo2 comparison:** Pick one TOA, compute residual in Tempo2 manually, trace exactly what it does
4. **Implement matching algorithm:** Once you understand Tempo2's exact steps, implement them in JAX

The 1000x error is a symptom of a deeper conceptual mismatch, not a parameter issue that can be fixed by rearranging the code.

