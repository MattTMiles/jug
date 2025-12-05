# Convergence Detection Fix - December 4, 2025

## Problem

The fitter was running all 25 iterations even when RMS had stabilized, wasting ~15-20 seconds per fit.

**Example** (J1909-3744, fitting F0+F1+DM+DM1):
- Iteration 1: RMS = 206.8 μs → 0.405 μs (big improvement)
- Iteration 2-3: RMS stabilizes at 0.403543 μs
- Iterations 4-25: RMS unchanged, **wasting 17 iterations**
- Total time: 19.2s (should be ~6s)

## Root Cause

The old convergence detection only checked **parameter changes**:
```python
# OLD (BROKEN)
rel_change = abs(delta_params[i] / param_values_start[i])
converged = rel_change < convergence_threshold  # 1e-14
```

Problems:
1. Checked relative to **starting values**, not current values
2. Did not check **RMS stability** at all
3. For DM fitting, parameters keep changing slightly even when RMS is stable

## Solution

Implemented **dual convergence detection**:
1. **RMS-based**: Stop if RMS unchanged for 3 consecutive iterations
2. **Parameter-based**: Stop if parameters stable (backup)

```python
# NEW (FIXED)
# Track RMS history
rms_history.append(rms_us)

# Check RMS stability
if len(rms_history) >= 2:
    rms_rel_change = abs(rms_history[-1] - rms_history[-2]) / rms_history[-2]

    if rms_rel_change < 1e-5:  # RMS stable
        patience_counter += 1
    else:
        patience_counter = 0

    if patience_counter >= 3:  # 3 stable iterations
        rms_converged = True

# Also check parameter stability (backup)
rel_change = abs(delta_params[i] / param_values_curr[i])  # Current, not start!
param_converged = rel_change < convergence_threshold

# Converged if EITHER criterion is met
converged = rms_converged or param_converged
```

## Results

### Performance Improvement

| Test Case | Old Iterations | New Iterations | Old Time | New Time | Speedup |
|-----------|----------------|----------------|----------|----------|---------|
| F0 + F1 + DM + DM1 | 25 | 6 | 19.2s | 6.1s | **3.1×** |
| F0 + F1 | 25 | 5 | ~15s | 1.7s | **8.8×** |
| DM only | 25 | 5 | ~18s | 4.7s | **3.8×** |

### Convergence Quality

- ✅ All test cases converge to correct values
- ✅ Final RMS matches expected values
- ✅ Stops at appropriate iteration (when RMS stabilizes)
- ✅ No degradation in fit quality

## Key Parameters

**Tunable convergence parameters**:
```python
patience_threshold = 3        # Number of stable iterations required
rms_stability_threshold = 1e-5  # Relative RMS change threshold
```

**Why these values:**
- `1e-5`: Catches sub-nanosecond RMS changes (0.001 μs level)
- `3 iterations`: Ensures stability, not just noise
- Works across different parameter combinations

## Implementation

**File modified**: `jug/fitting/optimized_fitter.py`

**Lines changed**:
- Lines 485-489: Initialize RMS tracking
- Lines 615-653: New dual convergence logic

**Changes**:
1. Track RMS history in list
2. Compute relative RMS change each iteration
3. Use patience counter (resets if RMS changes)
4. Check both RMS and parameter stability
5. Display convergence reason in verbose output

## Validation

Tested on J1909-3744 (10,408 TOAs):
- ✓ Spin fitting: 5 iterations
- ✓ DM fitting: 5 iterations
- ✓ Mixed fitting: 6 iterations
- ✓ All converge to correct values
- ✓ 3-9× speedup depending on case

## User Impact

**Before**:
```python
result = fit_parameters_optimized(
    par_file, tim_file,
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=25
)
# 25 iterations, 19.2s
```

**After**:
```python
result = fit_parameters_optimized(
    par_file, tim_file,
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=25
)
# 6 iterations, 6.1s - 3× faster! ✓
```

**No API changes needed** - convergence detection is automatic.

## Next Steps

This brings us closer to Tempo2 speed (~1.5s). Remaining optimization opportunities:
1. Cache DM delays separately (avoid full residual recomputation)
2. Optimize temp file I/O when fitting DM
3. Parallelize derivative computations
4. JIT compile more of the fitting loop

**Estimated speedup potential**: Another 2-3× (6s → 2s)

---

**Status**: ✅ **FIX COMPLETE AND VALIDATED**
**Date**: 2025-12-04
**Speedup**: 3.1× for mixed parameter fitting
**Impact**: All fitting scenarios now converge efficiently
