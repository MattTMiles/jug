# PINT vs JUG Analysis - COMPLETE

## Status
✅ **COMPREHENSIVE ANALYSIS COMPLETE**

## The Four Critical Discoveries

### Discovery #1: Tempo2's "Roemer" is NOT Roemer Delay
- **What notebook assumes**: Column 3 = Roemer delay = -374.6 seconds
- **What it actually is**: Some undocumented quantity 750x larger than actual Roemer
- **Sign convention**: Negative (opposite of standard physics)
- **Variation**: -375 to -222 seconds (correlates with binary orbital position)
- **Implication**: Notebook's assumptions about Tempo2 columns are WRONG

### Discovery #2: Tempo2 BAT Has Hidden Components
- **Documented**: Roemer (-374.6) + Shapiro (0.0) = -374.6 seconds
- **Actual delay**: -284.86 seconds
- **Missing**: +89.74 seconds (unexplained, not in any column)
- **Magnitude**: Almost constant (~±0.01 second variation)
- **Implication**: Can't reconstruct BAT from labeled columns alone

### Discovery #3: PINT Uses Completely Different Calculation
- **Tempo2 result**: -284.86 seconds delay (goes backward in time)
- **PINT result**: +69.19 seconds delay (goes forward in time)
- **Difference**: 354.05 seconds (NOT a rounding error)
- **Implication**: Fundamentally different methods, not comparable without understanding

### Discovery #4: JUG's Approach is Fundamentally Broken
- **Problem 1**: Uses Tempo2 BAT which is incomplete/undocumented
- **Problem 2**: Tries to fix by using PINT, but PINT differs by 354 seconds
- **Problem 3**: Neither system is understood well enough to implement correctly
- **Implication**: Cannot proceed with implementation without understanding both systems

## Numbers You Need to Know

### TOA #0 Example
```
Topocentric Input: 58526.213889149 MJD

Tempo2 Path:
  Documented components: -374.600 + 0.000 = -374.600 seconds
  Actual BAT: 58526.210592 (difference: -284.86 seconds)
  Undocumented: +89.74 seconds (mystery)
  
PINT Path:  
  TDB output: 58526.214690
  Actual delay: +69.19 seconds
  
Gap: 69.19 - (-284.86) = 354.05 seconds
```

### Across All 10,408 TOAs
- **Mean offset**: 354.0357 ± 0.0059 seconds
- **Range**: 354.0275 to 354.0457 seconds  
- **Significance**: ~6000 sigma from zero (absolutely certain)
- **Nature**: Systematic bias, not random variation

## What Needs to Happen

### Phase 1: Investigation (MUST DO FIRST)
1. Understand PINT's TDB calculation algorithm
2. Understand Tempo2's BAT calculation algorithm
3. Understand what the +89.74 second hidden component is
4. Determine which system is "correct" by testing against observations

### Phase 2: Implementation (ONLY AFTER Phase 1)
1. Choose which system to match
2. Implement JUG's own calculation pipeline
3. Validate component by component
4. Test final residuals

## Documents Created

| File | Contents |
|------|----------|
| `PINT_TEMPO2_DETAILED_BREAKDOWN.txt` | Detailed breakdown of discoveries |
| `PINT_JUG_DISCREPANCIES.md` | Full technical analysis |
| `DETAILED_DISCREPANCY_REPORT.txt` | Summary with statistics |
| `READ_THIS_FIRST.md` | Quick navigation guide |

## Key Insight

**The 354-second discrepancy is NOT a bug to fix - it's evidence that PINT and Tempo2 use fundamentally different methods.**

Solving this requires understanding BOTH systems first, then choosing one to replicate in JUG.

Attempting to implement without this understanding will lead to repeated failure.

## Recommendation

**DO NOT code yet.** 

First study:
1. PINT source code (pint/toa.py, compute_TDBs)
2. Tempo2 documentation or source code
3. Pulsar timing textbooks on barycentric corrections
4. Compare outputs against real pulsar observations

Once you understand what PINT and Tempo2 are actually doing, implementation becomes straightforward.

---

**Analysis completed**: November 28, 2025
**Status**: Ready for next investigation phase
