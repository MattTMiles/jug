# Enhanced Fit Results Dialog

**Date**: 2026-01-27  
**Feature**: Show parameter changes in fit results

---

## What Changed

The fit results dialog now shows a more informative table with **5 columns** instead of 3:

### Old Format (3 columns)
| Parameter | Value | Uncertainty |
|-----------|-------|-------------|
| F0 | 339.315691918984044 Hz | 1.35e-14 |
| F1 | -1.615059e-15 Hz/s | 6.64e-22 |

### New Format (5 columns)
| Parameter | New Value | Previous Value | Change | Uncertainty |
|-----------|-----------|----------------|--------|-------------|
| F0 | 339.315691918984044 Hz | 339.315691919050039 Hz | -0.000000000065995 Hz | 1.35e-14 |
| F1 | -1.615059e-15 Hz/s | -1.612750e-15 Hz/s | -2.308654e-18 Hz/s | 6.64e-22 |
| F2 | 6.075158e-26 Hz/s | 0.000000e+00 Hz/s | 6.075158e-26 Hz/s | 1.06e-29 |

---

## What Each Column Means

### 1. Parameter
The parameter name (F0, F1, F2, DM, etc.)

### 2. New Value
The fitted value after optimization

### 3. Previous Value
The value from the original .par file
- **If parameter was in .par file**: Shows the original value
- **If parameter was NOT in .par file**: Shows 0.0

### 4. Change
The difference: `New Value - Previous Value`
- Positive: Parameter increased
- Negative: Parameter decreased
- Shows how much the fit changed the parameter

### 5. Uncertainty
The 1-sigma uncertainty on the fitted value (same as before)

---

## Example: Fitting F2 (not in .par file)

When you fit F2 using `--fit F2`:

```
Parameter | New Value         | Previous Value    | Change            | Uncertainty
----------+-------------------+-------------------+-------------------+------------
F2        | 6.075158e-26 Hz/s | 0.000000e+00 Hz/s | 6.075158e-26 Hz/s | 1.06e-29
```

**Interpretation:**
- **Previous Value = 0.0**: F2 wasn't in the original .par file
- **New Value**: The value the fitter determined from the data
- **Change = New Value**: Since we started from 0.0, the change equals the new value

---

## Example: Fitting Existing Parameters

When you fit F0 and F1 (already in .par file):

```
Parameter | New Value                | Previous Value           | Change                  | Uncertainty
----------+--------------------------+--------------------------+-------------------------+------------
F0        | 339.315691918984044 Hz   | 339.315691919050039 Hz   | -0.000000000065995 Hz   | 1.35e-14
F1        | -1.615059e-15 Hz/s       | -1.612750e-15 Hz/s       | -2.308654e-18 Hz/s      | 6.64e-22
```

**Interpretation:**
- **Previous Value**: Original values from .par file
- **New Value**: Optimized values after fitting
- **Change**: Shows how much each parameter shifted
  - F0 decreased by ~6.6e-11 Hz
  - F1 became more negative by ~2.3e-18 Hz/s

---

## Why This Is Useful

### 1. See What Changed
Quickly identify which parameters changed significantly vs. barely moved

### 2. Validate Fit Quality
- Small changes → .par file was already good
- Large changes → Initial parameters were wrong
- Change ≈ Uncertainty → Marginal improvement

### 3. Track Parameter Evolution
If you fit multiple times, you can see how parameters evolve

### 4. Identify New Parameters
Parameters with Previous Value = 0.0 are new (not in original .par file)

---

## Implementation Details

### Code Changes

**File**: `jug/gui/main_window.py`

**Added**:
1. `self.initial_params = {}` in `__init__()` to store original values
2. `_load_initial_parameter_values()` method to load values from .par file
3. Updated `_show_fit_results()` to show 5-column table

**Logic**:
```python
# Get previous value (0.0 if not in original par file)
prev_value = self.initial_params.get(param, 0.0)
change = new_value - prev_value
```

---

## Testing

### Test 1: Fit F2 (not in file)
```bash
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim --fit F2
```

Expected result:
- F0: Previous ≠ 0, Small change
- F1: Previous ≠ 0, Small change  
- F2: **Previous = 0**, Change = New Value

### Test 2: Fit existing parameters
```bash
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim
# Select F0, F1 in GUI
```

Expected result:
- Both show non-zero previous values
- Changes reflect difference between wrong and fitted values

---

## Example Session

```bash
# Fit with wrong par file
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim --fit F2

# After clicking "Run Fit", dialog shows:
# ┌───────────┬──────────────────────────┬──────────────────────────┬─────────────────────────┬─────────────┐
# │ Parameter │ New Value                │ Previous Value           │ Change                  │ Uncertainty │
# ├───────────┼──────────────────────────┼──────────────────────────┼─────────────────────────┼─────────────┤
# │ F0        │ 339.315691918984044 Hz   │ 339.315691919050039 Hz   │ -0.000000000065995 Hz   │ 1.35e-14    │
# │ F1        │ -1.615059e-15 Hz/s       │ -1.612750e-15 Hz/s       │ -2.308654e-18 Hz/s      │ 6.64e-22    │
# │ F2        │ 6.075158e-26 Hz/s        │ 0.000000e+00 Hz/s        │ 6.075158e-26 Hz/s       │ 1.06e-29    │
# └───────────┴──────────────────────────┴──────────────────────────┴─────────────────────────┴─────────────┘
```

You can now see:
- F0 changed by -6.6e-11 Hz (small correction)
- F1 changed by -2.3e-18 Hz/s (small correction)
- F2 started at 0.0, ended at 6.1e-26 Hz/s² (new parameter!)

---

## Benefits

✅ **More informative**: See what changed, not just final values  
✅ **Easy comparison**: Previous vs New side-by-side  
✅ **Change quantification**: Explicit change column  
✅ **New parameter indication**: Previous = 0.0 means new parameter  
✅ **Validation helper**: Compare changes to uncertainties  

---

**Status**: ✅ Feature complete and tested  
**Version**: JUG GUI Phase 2 (2026-01-27)
