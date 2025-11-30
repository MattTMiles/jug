# JUG vs Tempo2: Quick Reference Summary

## The Problem
- **JUG residuals:** 850 μs RMS
- **Tempo2 residuals:** 0.82 μs RMS
- **Error factor:** ~1000x

## Root Cause (Not a Single Bug)

### Initial Finding ❌
"The phase reference epoch is wrong (using TZR instead of first TOA)"
- **Status**: Partially correct, but not the root cause
- **Result of fixing this**: RMS stays ~850 μs (only 0.9% improvement)

### Actual Root Cause ✓
**Fundamental methodology mismatch:**

| JUG | Tempo2 |
|-----|--------|
| `residual = frac(phase(t_corrected)) / F0` | `residual = t_obs - t_predicted` |
| Pre-corrects time by applying delays | Computes residual as time difference |
| Phase-based calculation | Time-based calculation |
| ~850 μs residuals | ~0.82 μs residuals |

## Why This Matters

When you pre-apply DM delay to the time coordinate:
```
t_corrected = t_obs - 52 ms (DM delay)
spin_phase(t_corrected) = spin_phase(t_obs) - F0 × 52ms - 17,628 cycles

When you extract frac(phase):
  frac(phase) = tiny value (~microseconds)
  residual = tiny_value / F0 = wrong!
```

## The Fix

**Use Tempo2's approach:**

```python
# 1. Keep times uncorrected
t_obs = 58526.211 MJD

# 2. Compute delays separately
dm_delay = compute_dm_delay(f_mhz)          # ~52 ms
bary_delay = compute_barycentric_delay()     # ~milliseconds
shapiro_delay = compute_shapiro_delay()      # ~microseconds
binary_delay = compute_binary_delay()        # ~microseconds
clock_delay = compute_clock_correction()     # ~microseconds

# 3. Total delay
total_delay = dm_delay + bary_delay + shapiro_delay + binary_delay + clock_delay

# 4. Compute phase at corrected time
t_pred = t_obs - total_delay
phi = spin_phase(t_pred)

# 5. Extract fractional phase
n_nearest = round(phi)
phi_frac = phi - n_nearest

# 6. Convert to time residual
residual = phi_frac / F0
```

## Key Findings from Analysis

1. **TZR Reference Bug** ✓ FOUND
   - Using TZRMJD (1153 days away) instead of first observation
   - Creates 33-billion-cycle offset
   - But NOT the root cause of magnitude error!

2. **Time Correction Issue** ✓ FOUND  
   - Pre-correcting time breaks phase-based residual calculation
   - DM delay of 52 ms kills the calculation
   - Same issue whether we use t_obs or t_inf

3. **Methodology Incompatibility** ✓ ROOT CAUSE
   - JUG's phase-based approach is fundamentally different from Tempo2's time-based approach
   - Fixing TZR reference doesn't solve this
   - Need to switch to time-based residual calculation

4. **Data Observation** (Interesting, not a bug!)
   - Your TOAs are spaced at ~integer multiples of pulse period
   - Expected fractional phases should be < 1 microsecond
   - Tempo2 shows this correctly (~0.8 μs RMS)
   - JUG shows ~850 μs due to methodology issue

## Validation Checklist

- [ ] Implement time-based residual calculation
- [ ] Compare against Tempo2 residuals (should match < 0.1 μs)
- [ ] Verify sign patterns match
- [ ] Test with multiple pulsars
- [ ] Add unit tests for each delay component

## Files Generated

1. **COMPARISON_WITH_TEMPO2.md** - Detailed logic comparison
2. **DETAILED_ANALYSIS.md** - Complete methodology analysis with code examples
3. **residual_maker_playground.ipynb** - Updated with diagnostic cells (cells 31-37)

## Next Steps

1. **Immediate**: Refactor residual calculation to use time-based approach
2. **Short-term**: Validate against Tempo2 output
3. **Medium-term**: Implement fitting framework
4. **Long-term**: Support multiple tracking modes (nearest, absolute, etc.)

## Key Insight

The problem is not "bugs" in your code, but a **design-level mismatch**. JUG chose a phase-centric approach while Tempo2/PINT use time-centric residuals. These are **semantically different** and cannot be made compatible by tweaking parameters—they require architectural changes.

The good news: This is now clear and fixable!

