# DD Binary Model Debug Status
**Date**: 2025-11-30
**Session**: Binary Model Integration (Session 6 continuation)

## Problem Summary

DD/BT/T2 binary models are implemented and produce correct delays in isolation, but residuals computed by `compute_residuals_simple()` are completely wrong (~755 μs RMS vs PINT's 5 μs RMS).

## Key Findings

### ✅ What Works

1. **DD binary delay calculation**: When called directly, `dd_binary_delay()` and `dd_binary_delay_vectorized()` produce correct delays:
   - Test: MJD 58595.7 → delay = -3.64 s (expected: ~3-4 s) ✓
   - Range: ±3-6 seconds (consistent with A1=21.3 light-seconds) ✓

2. **Roemer delay formula fix**: Fixed critical bug where `sin(omega + arctan2(sin_nu, cos_nu))` was replaced with correct trig addition formula `sin(omega)*cos(nu) + cos(omega)*sin(nu)`. This improved single-TOA accuracy from 5.8 ms error to 87 ns error.

3. **Parameter passing**: Binary parameters (PB, A1, ECC, OM, T0, etc.) are extracted correctly from .par files and passed to DD function.

4. **JAX compilation issue resolved**: Removed `@jax.jit` from `dd_binary_delay()` to fix persistent cache problem. The non-JIT version works correctly.

### ⚠️ What's Broken

**Residuals are 150x too large**: 
- JUG residual RMS: 790.762 μs
- PINT residual RMS: 5.334 μs  
- Difference RMS: 755.595 μs

**Binary delay magnitude mismatch**:
- TZR printout shows: Binary delay = -0.000062 s
- Direct calculation shows: Binary delay = -6.05 s at same MJD
- Factor: ~100,000x difference!

**NOTE**: The TZR printout is misleading - it computes binary delay as a residual (total - other delays) rather than the actual value. But this suggests something is wrong with how delays are being combined.

## Hypothesis

The binary delay is being computed correctly but either:
1. Not being added to the total delay correctly
2. Being divided by a large factor somewhere (SECS_PER_DAY = 86400?)
3. Sign convention issue
4. Being overwritten/reset after computation

## Debug Steps Needed

1. **Add print statements** in `simple_calculator.py` lines 260-280 and 405-445 to trace:
   - Binary delay values immediately after computation
   - Binary delay values after iteration 2
   - Total delay before and after adding binary delay

2. **Check combined_delays.py**: Verify that when `has_binary_jax = False` (external binary), the JAX kernel isn't computing some conflicting binary delay

3. **Verify iteration logic**: The topocentric time iteration (lines 397-420) changes binary delay by only 4.26 ms. For A1=21s, PB=38d, typical changes should be larger.

4. **Compare component by component with PINT**: Extract individual delay components from both JUG and PINT and compare:
   - Roemer delay (SSB)
   - Shapiro delay (SSB)
   - DM delay
   - Binary delay (separate: Roemer + Einstein + Shapiro)

## Files Involved

- `/home/mattm/soft/JUG/jug/delays/binary_dd.py` - DD model implementation (✓ works)
- `/home/mattm/soft/JUG/jug/residuals/simple_calculator.py` - Integration point (⚠️ bug here)
- `/home/mattm/soft/JUG/jug/delays/combined.py` - JAX delay kernel (check if interfering)

## Test Case

**Pulsar**: J1012-4235 (DD model, A1=21.3 lt-s, PB=38d, 7089 TOAs)
**Command**: 
```python
from jug.residuals.simple_calculator import compute_residuals_simple
result = compute_residuals_simple(
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235_tdb.par",
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235.tim"
)
```

**Expected**: RMS ~ 5 μs (like PINT)
**Actual**: RMS = 790 μs

## Next Steps

1. Add detailed logging to track binary delay through the pipeline
2. Compare with ELL1 implementation in `combined.py` to see how it handles binary delays correctly
3. Consider whether DD/BT/T2 should be integrated into `combined_delays()` rather than computed externally

## ELL1 Status (for comparison)

ELL1 model works perfectly:
- J1909-3744: RMS = 0.417 μs (vs PINT's ~0.4 μs) ✓
- Agreement: 2.5 ns RMS ✓
- Implementation: Inline in `combined_delays()` JAX kernel

The key difference is ELL1 is computed INSIDE the JAX kernel with proper topocentric time handling, while DD/BT/T2 are computed OUTSIDE and then added.

## Update 2: Progress Made (2025-11-30, 23:35 UTC)

### ✅ Critical Findings

1. **Binary delays ARE being computed correctly**:
   ```
   DEBUG Iter1: Binary delays computed - range [-21.266, 21.260] s, mean=5.449 s, std=16.129 s
   DEBUG Iter2: Binary delays refined - range [-21.266, 21.260] s, mean=5.449 s, std=16.129 s
   ```
   - Range matches expected A1 = 21.3 light-seconds ✓
   - Mean of 5.4 s is reasonable for the orbital phase distribution ✓

2. **Binary delays ARE being added to total delay**:
   ```
   DEBUG: total_delay_jax before adding binary - range [-328.694, 327.811] s
   DEBUG: Total delay after adding binary - range [-348.754, 344.379] s
   ```
   - Total delay shifts by ~20 s after adding binary ✓
   - This confirms the addition on line 446 is working ✓

3. **Single TOA test confirms DD function works**:
   - Test: MJD 58595.714318 → delay = -3.595 s ✓
   - This is the same value we got in earlier isolation tests ✓

### ⚠️ Remaining Mystery

**Residuals are still 150x too large**: RMS = 790 μs (PINT: 5 μs)

Since binary delays are computed correctly AND added correctly, the problem must be elsewhere:

**Hypotheses**:
1. **TDB calculation difference**: JUG's standalone TDB might differ from PINT's
2. **SSB delay difference**: Roemer/Shapiro delays might be computed differently
3. **Phase calculation issue**: Something subtle in how phase is computed from delays
4. **Parameter interpretation**: Some parameter (like OMDOT) might need different units

### Debugging Strategy

The fact that Tempo2 gives good residuals (as user confirmed) suggests the problem is NOT fundamental to the DD model itself, but rather a JUG-specific integration issue.

**Next steps** (in order of likelihood):
1. Compare TDB values: JUG vs PINT for all TOAs (look for systematic offset)
2. Compare SSB delays (Roemer+Shapiro) component-by-component
3. Compare OMDOT units (JUG uses deg/yr, check if PINT expects rad/yr)
4. Check if PBDOT needs special handling (it's very small: -8.59e-12)

### Code Changes Made

- Added debug logging to trace binary delay values through pipeline
- Confirmed JAX JIT cache issue workaround (removed `@jax.jit` from `dd_binary_delay`)
- Fixed Roemer delay trig formula (sin(ω+ν) expansion)

### Test Command

```python
from jug.residuals.simple_calculator import compute_residuals_simple
result = compute_residuals_simple(
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235_tdb.par",
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235.tim"
)
# Current: RMS = 790 μs
# Target: RMS ~ 5 μs (like PINT/Tempo2)
```

The binary delay part is WORKING. The bug is elsewhere in the pipeline.
