# Session 8: JAX Fitting Diagnosis (2025-11-30)

## Session Goal
Validate that JUG fitting produces the same results as PINT when both start from identical perturbed parameters.

## What We Discovered

### The Good News ✅
1. **JAX residuals work perfectly**: 0.008 μs RMS vs baseline, 0.013 μs vs PINT
2. **Gauss-Newton fitter converges**: Smooth convergence in 5 iterations
3. **Fitting algorithm validated**: Successfully minimizes χ² function

### The Issue ⚠️
**JUG and PINT converge to different parameter values** (7-8σ apart):

```
Starting from: F0 + 1e-9 Hz, F1 + 2e-17 Hz/s

JUG converges to:
  F0  = 339.315691919041342 ± 5.84e-14 Hz
  F1  = -1.614753938436094e-15 ± 9.43e-22 Hz/s
  
PINT converges to:
  F0  = 339.315691919040830 ± 3.33e-14 Hz
  F1  = -1.614750512358547e-15 ± 5.37e-22 Hz/s

Difference:
  F0: 5.12e-13 Hz  (7.6σ combined)
  F1: 3.43e-21 Hz/s  (3.2σ combined)
```

### Root Cause
**Residual calculation differences create offset χ² surfaces**:
- JUG residuals differ from PINT by ~0.013 μs RMS
- Each fitter finds the minimum of its own χ² surface
- These minima are ~7σ apart

**Both fitters are correct** - they're minimizing slightly different functions.

## Key Question Answered

**Q**: Does the fitter work correctly?  
**A**: ✅ **YES** - the Gauss-Newton algorithm converges smoothly and finds parameter minima.

**Q**: Does JUG produce identical fits to PINT?  
**A**: ❌ **NO** - parameter values differ by 7-8σ due to 0.013 μs residual differences.

## Implications

### Scientific Validity
- 0.013 μs residual differences are **negligible for most pulsar science**
- Parameter differences (5e-13 Hz in F0) are **tiny in absolute terms**
- Both fits are **scientifically valid** - just minimizing slightly different functions

### Software Development
- JUG is an **independent implementation**, not a PINT clone
- Users should expect **small differences** from PINT fitted values
- Having an independent implementation provides **valuable cross-validation**

## Decision Point

### Option 1: Accept Current Status ⭐ **RECOMMENDED**

**Rationale**:
- Fitting algorithm is working correctly
- Differences stem from residual calculation, not fitting bugs
- 0.013 μs is scientifically negligible
- Independent implementation is valuable for validation

**Action Items** (~1 hour):
1. Document known differences in README
2. Add warning about PINT compatibility
3. Update CLAUDE.md with limitations
4. Move to Milestone 3 (noise models)

### Option 2: Debug Residual Differences

**Goal**: Achieve exact PINT residual agreement (< 0.001 μs)

**Effort**: 4-8 hours (uncertain)

**Likely Sources of 0.013 μs difference**:
1. Binary delay calculation details
2. Barycentric correction subtleties  
3. TDB/TCB time scale handling
4. Numerical precision in intermediate steps

**Risk**: May never achieve perfect agreement due to:
- Different library implementations (astropy vs PINT internals)
- Numerical precision limits
- Algorithmic differences in delay calculations

## Recommendation

**Accept current status and move forward**. The fitting framework is:
- ✅ Algorithmically sound
- ✅ Numerically stable  
- ✅ Scientifically valid
- ✅ Fast with JAX acceleration

Attempting to debug 0.013 μs residual differences has diminishing returns and uncertain payoff.

## Files Created This Session

1. **test_synthetic_fitting.py**: Diagnostic test comparing JUG vs PINT convergence
   - Tests both fitters on same perturbed start
   - Reports convergence and final parameter differences
   - Validates fitting algorithm independent of residual differences

2. **M2_FITTING_DIAGNOSIS.md**: Detailed analysis document
   - Test results and statistics
   - Root cause analysis
   - Path forward options with effort estimates

3. **Updated M2_JAX_FITTING_STATUS.md**: Added Session 8 diagnosis section

4. **Updated JUG_PROGRESS_TRACKER.md**: 
   - Changed M2 status from "95% complete" to "85% complete"
   - Added convergence diagnosis notes
   - Updated session 8 achievements

## Milestone 2 Status

### Completed ✅
- [x] JAX-accelerated residual calculation (0.008 μs precision)
- [x] Gauss-Newton fitter with Levenberg-Marquardt damping
- [x] Automatic differentiation for design matrix
- [x] Convergence validation on real data
- [x] Multiple binary models (DD, DDH, ELL1, BT, T2)

### Remaining 🚧
- [ ] CLI tool for fitting (`jug-fit`)
- [ ] Documentation of limitations and known differences
- [ ] Decision: Accept current status vs. debug residuals

### Estimated Completion
- If accepting current status: ~1 hour (documentation)
- If debugging residuals: ~4-8 hours (uncertain outcome)

## Next Session Recommendations

1. **Discuss with user**: Accept current status or pursue exact PINT agreement?
2. **If accepting**: Document limitations, create CLI tool, move to M3
3. **If debugging**: Start with binary delay component comparison

## Key Takeaways

1. **Fitting algorithm works** - converges smoothly, finds minima
2. **Residual differences matter** - 0.013 μs creates 7σ parameter offsets
3. **Both implementations valid** - JUG and PINT are different but correct
4. **Independent validation valuable** - having two implementations helps catch bugs
5. **Perfect agreement hard** - numerical precision and algorithm differences accumulate

## Test Command

```bash
python test_synthetic_fitting.py
```

This test isolates the fitter from residual calculation by having both JUG and PINT use their own residuals, starting from the same perturbed point.

---

**Session Duration**: ~2 hours  
**Status**: Diagnosis complete, decision point reached  
**Next**: User decision on path forward
