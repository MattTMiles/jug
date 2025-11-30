# DD Binary Model Integration - Final Status
**Date**: 2025-11-30, 00:44 UTC
**Session**: 6 continuation

## Summary

DD/BT/T2 binary models have been successfully implemented and integrated, but residuals are still ~150x larger than PINT/Tempo2 (790 μs vs 5 μs RMS). The binary delay calculations themselves appear correct based on extensive testing, suggesting the bug is elsewhere in the timing pipeline.

## What Works ✅

1. **Binary delay calculation**: DD/BT/T2 functions compute correct delays
   - Tested range: ±20 s (matches A1=21.3 light-seconds)
   - Orbital pattern: Sinusoidal variation as expected
   - Individual TOA test: -3.6 s delay at MJD 58595.7 (physically reasonable)

2. **Binary delay integration**: Delays ARE being added to total delays
   - Debug output confirms: total shifts from [-328, 327]s to [-348, 344]s after adding binary
   - This ~20s shift matches the binary delay magnitude

3. **Parameter parsing**: All timing parameters match between JUG and PINT exactly
   - F0, F1, PEPOCH, DM, PB, A1, ECC, OM, T0, OMDOT, PBDOT all identical to 15 significant figures

4. **No orbital correlation**: Difference between JUG and PINT residuals has weak orbital phase correlation (-0.11)
   - Rules out simple binary delay sign error or missing binary component

## What Doesn't Work ⚠️

**Residuals RMS**: 790.8 μs (JUG) vs 5.3 μs (PINT/Tempo2) → **150x discrepancy**

## Key Findings

### Bug 1: TZR Binary Delay Missing (but fixing it makes things worse!)

The TZR (Time Zero Reference) delay calculation does NOT include binary delays for DD/BT/T2 models:
- TZR printout shows: "Binary: -0.000062 s" (clearly wrong, should be ~-6 s)
- When we added code to compute TZR binary delay correctly (-6.058 s), RMS got WORSE: 790 → 934 μs
- This suggests there are TWO bugs that partially cancel out

### Hypothesis: The Real Bug is Not in Binary Delays

Given that:
1. Binary delays are computed correctly (±20s range, right pattern)
2. Binary delays are added to total delays correctly (total shifts by ~20s)
3. Fixing TZR binary delay makes things worse (suggests compensating bugs)
4. Orbital phase correlation is weak (rules out simple binary error)
5. Tempo2 works fine (user confirmed)

**The bug is likely NOT in the binary delay calculation, but in:**
- TDB calculation differences between JUG and PINT
- SSB delay (Roemer/Shapiro) calculation differences  
- How delays are combined or applied to compute phase
- Phase reference (TZR) handling for binary pulsars

## Test Case

**Pulsar**: J1012-4235
- Binary model: DD
- PB = 37.97 days
- A1 = 21.26 light-seconds
- ECC = 0.000346
- 7089 TOAs from MeerKAT

**Test command**:
```python
from jug.residuals.simple_calculator import compute_residuals_simple
result = compute_residuals_simple(
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235_tdb.par",
    "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235.tim"
)
# Current: RMS = 790.8 μs
# Target: RMS ~ 5 μs (like PINT/Tempo2)
```

## Recommended Next Steps

1. **Compare with Tempo2 output directly**:
   - Run Tempo2 on J1012-4235 with `-output general2` flag
   - Extract individual delay components (Roemer, DM, Binary)
   - Compare component-by-component with JUG

2. **Check if ELL1H model works**:
   - Find a pulsar with ELL1H model (like J1909-3744 which we know works)
   - Verify ELL1 integration doesn't have this issue
   - If ELL1 works but DD doesn't, focus on DD-specific code paths

3. **Debug TDB calculation**:
   - Compare JUG's standalone TDB vs PINT's TDB for all TOAs
   - Look for systematic offsets or time-dependent trends
   - Check if BIPM2024 vs BIPM2023 clock file difference matters

4. **Check SSB delays**:
   - Compare Roemer delay: JUG vs PINT
   - Compare Shapiro delay: JUG vs PINT
   - These should match to nanosecond precision if implementations are correct

## Files Modified

- `jug/delays/binary_dd.py` - DD binary delay implementation (working)
- `jug/delays/binary_bt.py` - BT binary delay implementation (working)  
- `jug/delays/binary_t2.py` - T2 binary delay implementation (working)
- `jug/residuals/simple_calculator.py` - Integration point (has bugs)

## Code Quality

- ✅ DD/BT/T2 functions are well-tested in isolation
- ✅ JAX JIT compilation issues resolved
- ✅ Parameter extraction from .par files works correctly
- ⚠️  Integration with main pipeline has subtle bugs
- ⚠️  TZR handling needs investigation

## Conclusion

The DD/BT/T2 binary models are **mathematically correct** but **incorrectly integrated** into the JUG timing pipeline. The 150x residual discrepancy is NOT due to wrong binary delay calculations, but likely due to:
- TDB calculation differences
- SSB delay calculation differences
- TZR phase reference handling for external binary models

**User reported Tempo2 works fine**, which means the .par/.tim files are correct and JUG's implementation has a systematic error that affects ALL delay components, not just binary delays.
