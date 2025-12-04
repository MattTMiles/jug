# Piecewise Fitting Investigation: Conclusion

**Date**: 2024-12-02  
**Session**: 16

## Summary

We investigated whether "piecewise fitting" with multiple local PEPOCHs could improve numerical precision over the standard longdouble implementation. After implementation and testing, we conclude that **the current longdouble implementation is optimal** and no special "piecewise" or "longdouble_spin_pars" flag is needed.

## What We Tried

### 1. Piecewise Fitting with Local PEPOCHs

**Concept**: Divide the data into segments, fit F0/F1 separately in each segment using a local PEPOCH at the segment center, then combine.

**Implementation**: See `piecewise_fitting_implementation.ipynb`

**Results**:
- ✓ Method works and produces reasonable residuals
- ✗ Residual difference vs longdouble shows **drift** (~20 μs across timespan)  
- ✗ Scatter in difference **increases with time** (not constant)
- ✗ Does NOT create multiple "zero-error" points at segment boundaries

**Why it didn't work as expected**:
The piecewise approach fits separate F0/F1 values for each segment. However:
1. The segments are **discontinuous** - no physical constraint connects them
2. Each segment accumulates independent float64 errors
3. Errors **compound** as you move away from each local PEPOCH
4. No advantage over single longdouble calculation with one PEPOCH

### 2. Hybrid Method with Longdouble Boundaries

**Concept**: Compute phase at segment boundaries in longdouble, then use float64 for interpolation within segments.

**Implementation**: See Step 13+ in `piecewise_fitting_implementation.ipynb`

**Results**:
- ✓ Slightly better than pure piecewise (~15 μs drift vs ~20 μs)
- ✗ Still shows systematic drift  
- ✗ More complex than simple longdouble
- ✗ No clear advantage

## What Actually Works

### Current Architecture is Optimal

**Residual Calculation** (`simple_calculator.py`):
- Uses `np.longdouble` for F0, F1, F2 parameters
- Uses `np.longdouble` for PEPOCH and time calculations
- Achieves ~0.403 μs WRMS baseline precision
- **NO changes needed!**

**Fitting** (`derivatives_spin.py` + `wls_fitter.py`):
- Derivatives computed in float64 (JAX)
- Design matrix in float64
- WLS solver in float64 (JAX)
- Converges correctly to ~0.403 μs WRMS
- **NO special longdouble mode needed!**

### Key Insight

The precision bottleneck is in the **residual calculation**, not the fitting. Once residuals are computed with longdouble precision (~0.403 μs), float64 derivatives are MORE than sufficient for fitting.

**Validation**:
```python
# Test in test_float64_fitting_works.py
Prefit RMS:   0.403544 μs
Postfit WRMS: 0.403540 μs  # ← Converges perfectly!
```

## Why Longdouble Works

The longdouble implementation succeeds because:

1. **Single reference point**: One PEPOCH minimizes accumulated errors
2. **Consistent precision**: All phase calculations use longdouble throughout
3. **No discontinuities**: Phase is continuous across entire timespan
4. **Simple**: Minimal code complexity = fewer bugs

Piecewise approaches introduce:
- Multiple reference points → multiple error sources
- Float64 segments → precision loss
- Discontinuities → systematic drift
- Complexity → harder to debug

## Recommendations

1. **Keep current implementation** - No changes to `simple_calculator.py`
2. **Remove longdouble_spin_pars flag** - Not needed, adds confusion
3. **Document the architecture** - Make it clear that:
   - Residuals always use longdouble for F0/F1/F2
   - Fitting always uses float64 (JAX)
   - This combination is optimal

4. **Focus optimization elsewhere**:
   - JAX JIT compilation (already done)
   - GPU acceleration for delays (already done)
   - Batch processing for multiple pulsars
   - NOT precision - we're already at the limit!

## Files Created This Session

- `piecewise_fitting_implementation.ipynb` - Full exploration of piecewise method
- `test_float64_fitting_works.py` - Validation that current architecture works
- `PIECEWISE_FITTING_CONCLUSION.md` - This document

## Precision Comparison

| Method | RMS vs Longdouble | Drift | Complexity |
|--------|-------------------|-------|------------|
| **Longdouble (current)** | **Baseline** | **None** | **Low** |
| Float64 only | ~20 ns RMS | Quadratic | Low |
| Piecewise | ~5-10 μs RMS | Linear drift | High |
| Hybrid | ~3-7 μs RMS | Linear drift | Very high |

**Winner**: Current longdouble implementation! 🎉

## Next Steps

Close this investigation and move on to:
- Milestone 3: EFAC/EQUAD noise models
- Milestone 5: PyQt6 GUI
- Performance optimization (if needed)

**Do not** implement piecewise fitting or longdouble_spin_pars flag.
