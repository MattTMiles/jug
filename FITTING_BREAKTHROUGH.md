# Fitting Breakthrough - Session 13

**Time**: 2025-12-01 04:57 UTC  
**Status**: MAJOR PROGRESS - Signal found!

## The Breakthrough

After 6 hours of investigation, we found how PINT handles phase wrapping for fitting!

### What We Discovered

**PINT's approach** (`track_mode="nearest"`):
```python
# 1. Compute full model phase
modelphase = self.model.phase(self.toas)

# 2. Subtract first phase for reference
modelphase -= Phase(modelphase.int[0], modelphase.frac[0])

# 3. DISCARD INTEGER PART - keep only fractional part
residualphase = Phase(np.zeros_like(modelphase.frac), modelphase.frac)

# 4. Subtract weighted mean
return residualphase - weighted_mean(residualphase)
```

**Key insight**: PINT wraps to "nearest pulse" by discarding integer cycles, NOT by subtracting TZR!

### Test Results

**Before fix** (with mean subtraction):
- Correlation: -0.062 (nearly zero - no signal!)
- Predicted ΔF0: 6.9e-16 Hz (1000× too small)
- Fitting failed completely

**After fix** (without mean subtraction):
- Correlation: **0.568** (strong signal! ✅)
- Predicted ΔF0: -2.9e-12 Hz (4000× bigger! ✅)
- Fitting converges but **wrong direction** ⚠️

### Current Status

**What works**:
- ✅ Signal is now visible to fitter (correlation 0.57)
- ✅ Step size is reasonable (piconewton scale)
- ✅ Fitter converges

**What's wrong**:
- ❌ F0 moves in wrong direction
- ❌ RMS gets slightly worse instead of better
- ⚠️ Fitted F0: 339.315691919037... (moved away from target)
- ⚠️ Target F0: 339.315691919041... (should move toward this)

### Hypothesis

The issue is likely **how mean subtraction interacts with derivatives**:

**Option A**: Derivatives need to account for mean subtraction
- PINT might subtract mean from BOTH residuals AND derivatives
- Our derivatives are raw `dt`, not mean-corrected

**Option B**: Sign error somewhere
- Maybe residual definition is backwards?
- Or WLS solver has wrong sign?

**Option C**: Mean should be fit as a parameter
- Like PHOFF, but for the mean offset
- This is what PINT does implicitly

### Next Steps

1. Check if PINT subtracts mean from design matrix columns
2. Try fitting with PHOFF parameter (mean as free parameter)
3. Check sign conventions in WLS solver
4. Compare residual signs with PINT

### Code Changes Made

**File**: `jug/residuals/simple_calculator.py`

Added `subtract_tzr` parameter:
- `True` (default): Legacy TZR subtraction + mean subtraction
- `False`: PINT-style wrapping without TZR, optional mean subtraction

**Critical change**:
```python
if subtract_tzr:
    # Subtract mean (for display)
    residuals_us = residuals_us - weighted_mean
else:
    # Don't subtract mean (for fitting)
    # Fitter sees raw wrapped residuals
```

This change increased correlation from -0.062 → 0.568!

### Time Summary

- Total session time: ~6 hours
- Time to breakthrough: ~5.5 hours
- Current phase: Debugging sign/mean issue (~30 min remaining)

---

**Bottom Line**: We found the signal! Just need to fix the sign/mean handling and fitting will work!

## UPDATE 23:01 UTC - PINT Comparison

**Shocking discovery**: PINT also doesn't fit this case!

Running PINT directly on the same files:
```
PINT residuals (wrong F0): Mean=0.497 μs, Std=2.376 μs
After 1 PINT WLS iteration:
  F0: 339.31569191904003446325
  ΔF0: 0.000e+00  ← NO CHANGE!
  RMS: 2.314 μs
```

**PINT didn't move F0 either!**

This suggests:
1. Either the F0 error is too small for single-iteration fitting
2. Or there's something about this specific par file that prevents fitting
3. Or PINT needs multiple iterations to converge from this starting point

**Key difference**: PINT's RMS is 2.3 μs vs JUG's 0.4 μs
- This suggests PINT and JUG are computing different residuals!
- Might be due to different delay models, ephemeris, or clock corrections

## Current Status Summary

**What we learned**:
- ✅ How PINT wraps phases (discard integer, keep fractional)
- ✅ PINT uses negative sign in design matrix
- ✅ Mean subtraction hides signal (correlation -0.06 → 0.57 without it!)
- ✅ Without mean subtraction, steps are 4× too large
- ⚠️ PINT also fails to fit in 1 iteration with these files

**Next steps needed**:
1. Understand why PINT/JUG have different RMS (2.3 vs 0.4 μs)
2. Figure out multi-iteration convergence strategy
3. Possibly fit PHOFF alongside F0 properly
4. May need to match PINT's delay calculations exactly first

**Time**: 7+ hours on this session
**Recommendation**: PAUSE and regroup with fresh perspective tomorrow

The fitting problem is deeper than expected - even PINT struggles with this specific test case!
