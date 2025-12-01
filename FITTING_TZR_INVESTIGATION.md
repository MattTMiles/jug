# TZR/Phase Offset Fitting Investigation

**Date**: 2025-12-01  
**Status**: BLOCKED - Need to understand PINT/tempo2 approach better

## Problem Statement

Cannot get fitting to work with JUG residuals. Three approaches tried, all failed.

## Attempts

### Attempt 1: TZR-Subtracted Residuals (Wrapped)

**Approach**: Use normal TZR-subtracted, wrapped residuals with derivatives d(phase)/d(F0) = dt

**Result**: ❌ FAILED
- Step size: 1.3e-15 Hz (1000× too small)
- Correlation between derivatives and residuals: -0.062 (nearly zero)
- **Reason**: TZR cancels out 99.9% of F0 error signal

### Attempt 2: Mean-Subtracted Residuals (Unwrapped)

**Approach**: Use `subtract_tzr=False`, subtract mean phase instead

**Result**: ❌ FAILED  
- F0 diverged wildly (339 → 328 Hz)
- RMS = 62 million μs
- **Reason**: Unwrapped phases are millions of cycles, numerically unstable

### Attempt 3: PHOFF Parameter (PINT-Style)

**Approach**: Fit F0 and PHOFF simultaneously, use mean-subtracted unwrapped residuals

**Result**: ❌ FAILED
- F0 diverged (339 → 319 Hz)  
- PHOFF grew to billions of cycles
- RMS = 61 million μs
- **Reason**: Same numerical instability as Attempt 2

## What We Know

**Works**:
- ✅ Residual calculation (0.4 μs RMS)
- ✅ Analytical derivatives (correct formulas)
- ✅ WLS solver (copied from PINT)

**Doesn't Work**:
- ❌ Any form of fitting with current approach

## The Mystery

**PINT and tempo2 both**:
1. Subtract phase offsets (TZR or mean)
2. Fit parameters successfully
3. Converge in ~5-10 iterations

**But we can't replicate this!**

## Questions for Next Session

1. Does PINT's `PhaseOffset.PHOFF` actually use wrapped or unwrapped residuals?
2. How exactly does tempo2's fitting loop handle TZR recomputation?
3. Is there a derivative chain rule we're missing?
   - `d(residual)/d(F0) = d(phase)/d(F0) - d(tzr_phase)/d(F0)`?
4. Do they use a different residual formulation during fitting vs display?

## Possible Next Steps

1. **Read PINT fitter source code carefully** - see how PhaseOffset is actually used
2. **Run PINT in debug mode** - see what residuals look like during fitting
3. **Contact PINT/tempo2 developers** - ask how they handle this
4. **Try numerical finite differences** - compute d(RMS)/d(F0) numerically as sanity check

## Time Spent

- Session 13: ~5 hours debugging this specific issue
- Total Milestone 2: ~15 hours

## Recommendation

**PAUSE fitting implementation** until we understand the PINT/tempo2 approach better.

**Alternative**: Use PINT's fitter directly, focus JUG on other features (residual calculation, models, etc.)

---

**Bottom Line**: We're missing something fundamental about how phase offsets are handled during fitting. The math/implementation we have should work in theory, but doesn't in practice. Need expert guidance or deeper code reading.
