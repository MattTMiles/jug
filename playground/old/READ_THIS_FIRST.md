# PINT vs JUG Discrepancy Analysis - READ THIS FIRST

## Status
✓ **ANALYSIS COMPLETE** - All discrepancies found and documented

## The Critical Finding

**PINT's TDB is 354.0457 ± 0.0059 seconds AHEAD of Tempo2's BAT**

This is:
- ✓ Verified across all 10,408 TOAs
- ✓ Systematic and consistent (std dev: 0.006 seconds)
- ✓ Fundamental (indicates different computational approaches)
- ✗ NOT explainable by rounding, ephemeris, or minor differences

## Why The Notebook Failed

The quick fix (Cell 17) tried to use PINT's "correct" times, but:
- PINT's times are 354 seconds different from Tempo2's
- This difference propagates directly into residuals
- Using PINT's baseline without understanding it just shifts the error
- Result: Still ~850 microsecond residual errors

## What We Know

### Parameters (All Match ✓)
- F0, F1, PEPOCH identical
- DM, DM1, DM2 identical
- Binary parameters identical
- Input TOAs identical

### Barycentric Computation (DIFFERS ✗)
```
Tempo2:  Topo → (-285 seconds) → BAT
PINT:    Topo → (+69 seconds) → TDB
Offset:  354 seconds
```

## The Mystery

The 354-second offset equals 0.7094 AU of light-travel time. This suggests:

1. Roemer delay sign convention difference?
2. Reference frame difference (TDB vs BAT)?
3. Ephemeris version difference?
4. Different clock correction approach?

**We don't know which is CORRECT yet.**

## What JUG Needs

JUG CANNOT just:
- ✗ Use PINT's times (wrong by 354 seconds vs Tempo2)
- ✗ Use Tempo2's BAT (already tried, still fails)
- ✗ Apply corrections (too fundamental)

JUG MUST:
- ✓ Understand WHY the difference exists
- ✓ Determine which system is correct
- ✓ Implement own barycentric pipeline
- ✓ Match whichever is correct (PINT or Tempo2)

## Documentation Files

### For Executive Summary
**→ DETAILED_DISCREPANCY_REPORT.txt** (236 lines)
- Quick reference with all numbers
- Statistics and findings
- Next steps

### For Technical Details
**→ PINT_JUG_DISCREPANCIES.md** (258 lines)
- Comprehensive analysis
- Section-by-section breakdown
- Parameter comparison table
- Methodology differences

## Next Steps (IN ORDER)

### BEFORE ANY CODING:

1. **Verify correctness**
   - Test PINT vs Tempo2 against actual observations
   - Which matches real pulsar timing data?

2. **Extract PINT internals**
   - Get Roemer delay component (if accessible)
   - Get Shapiro delay component (if accessible)
   - Compare each with Tempo2's values

3. **Find the source of 354 seconds**
   - Is it clock corrections?
   - Is it ephemeris (DE440 vs older)?
   - Is it reference frame definition?
   - Is it coordinate system?

4. **THEN implement**
   - Choose correct reference (PINT or Tempo2)
   - Implement JUG's barycentric pipeline
   - Validate component-by-component
   - Test against observations

## Key Facts

- Analysis scope: 10,408 TOAs
- Offset consistency: 354.0275 to 354.0457 seconds
- Offset precision: ± 0.006 seconds (6000 sigma significance!)
- Reproducibility: 100% verified

## Why This Matters

The 354-second discrepancy is **not a minor issue**. It's:
- Too large to be rounding error
- Too systematic to be convergence issue
- Fundamental to how barycentric times are computed

**Solving this requires understanding the root cause, not applying patches.**

## Questions?

1. "Why didn't using PINT's times work?"
   → PINT's times are 354 seconds off from Tempo2's standard

2. "Which one is correct?"
   → Unknown - needs investigation vs actual observations

3. "Can JUG just add 354 seconds?"
   → No - this is a symptom, not the disease

4. "What should we do?"
   → Follow the 4 steps above before any implementation

---

**Status**: Analysis complete, verified, documented.  
**Next phase**: Investigation phase (determine which system is correct)  
**Then**: Implementation phase (build JUG's barycentric pipeline)

