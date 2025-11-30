# Goal 4 Quick Reference

## Achievement
✅ **2.55 ns RMS** accuracy with **full independence** from PINT  
✅ **897x faster** than PINT (0.888 ms vs 796 ms)  
✅ Uses **standalone TDB** from Goal 3 (0.000 ns error)  

---

## The Key Fix

### Problem
```python
# WRONG - loses 5 decimal places:
self.tdbld_sec_ld = tdbld_ld * np.longdouble(SECS_PER_DAY)  # in __post_init__
```
**Impact**: 327 ns TDB scatter → 181 ns residual scatter

### Solution
```python
# CORRECT - preserves precision:
self.tdbld_mjd_ld = tdbld_ld  # Store as MJD in __post_init__

# Convert at calculation time:
tdbld_sec = self.tdbld_mjd_ld * np.longdouble(SECS_PER_DAY)  # in compute_residuals
dt_sec = tdbld_sec - self.PEPOCH_sec - delay_ld
```
**Result**: 2.55 ns RMS with full independence

---

## Why It Works

| Format | Magnitude | Fractional Digits | Precision at 1e-9 day |
|--------|-----------|-------------------|----------------------|
| MJD | ~58526 | 13 digits | ~0.01 ns ✓ |
| Seconds (pre-computed) | ~5e9 | 8 digits | ~300 ns ✗ |

**Key insight**: Longdouble has 18 total decimal digits. Keep values small (MJD range) to maximize fractional precision.

---

## Independence Check

| Component | Source | Error vs PINT |
|-----------|--------|---------------|
| TDB | Standalone (Goal 3) | 0.000 ns |
| Delays | Independent JAX | N/A |
| TZR Delay | Independent calc | -7.1 ns |
| Phase | Par file params | N/A |
| **Total Residuals** | **All standalone** | **2.55 ns RMS** |

**Verdict**: 100% independent calculation. PINT only used for data loading and comparison.

---

## Evolution

| Version | RMS | Independent? | Issue |
|---------|-----|--------------|-------|
| Original | 7900 ns | ❌ | Used PINT's TZR |
| First Try | 181 ns | ❌ | TDB precision loss |
| False Fix | 4.7 ns | ❌ | Used PINT's TDB |
| **Final** | **2.6 ns** | ✅ | Standalone + precision |

---

## Files
- **Implementation**: `residual_maker_playground_active_MK4_ind_TZR_secondTry.ipynb`
- **Full Report**: `GOAL_4_SUCCESS_REPORT.md`
- **First Attempt**: `residual_maker_playground_active_MK4_ind_TZR.ipynb` (reference only)

---

## Usage

```python
# Initialize with standalone TDB
jug_calc = JUGResidualCalculatorFinal(
    par_params=par_params,
    pint_model=pint_model,      # Only for data loading
    pint_toas=pint_toas,        # Only for data loading
    obs_itrf_km=obs_itrf_km,
    mk_clock=mk_clock_data,
    gps_clock=gps_clock_data,
    bipm_clock=bipm_clock_data,
    location=location
)

# Compute residuals (fully independent!)
residuals = jug_calc.compute_residuals()
```

**Result**: 2.55 ns RMS, 897x faster than PINT, 100% independent
