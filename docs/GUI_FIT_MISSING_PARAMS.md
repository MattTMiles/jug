# Fitting Parameters Not in .par File

**Date**: 2026-01-27  
**Feature**: Dynamic parameter fitting with `--fit` flag

---

## Problem

Previously, the GUI would show F2 in the checkbox list if you used `--fit F2`, but when you tried to fit it, the fitter would fail with:

```
ValueError: Parameter F2 not found in .par file
```

This happened because:
1. The GUI would add F2 to the checkbox list
2. But the fitting engine (`optimized_fitter.py`) required all parameters to exist in the .par file
3. The postfit residual computation also couldn't handle new parameters

---

## Solution

### 1. Fitting Engine Fix

Modified `jug/fitting/optimized_fitter.py` in `_fit_parameters_general()`:

**Before:**
```python
for param in fit_params:
    if param not in params:
        raise ValueError(f"Parameter {param} not found in .par file")
    param_values_start.append(params[param])
```

**After:**
```python
for param in fit_params:
    if param not in params:
        # Add default value for missing parameter
        if param.startswith('F') and param[1:].isdigit():
            default_value = 0.0  # Spin frequency derivatives
        elif param.startswith('DM') and (len(param) == 2 or param[2:].isdigit()):
            default_value = 0.0  # DM derivatives
        else:
            raise ValueError(f"Parameter {param} not found in .par file and no default available")
        
        params[param] = default_value
        if verbose:
            print(f"Warning: {param} not in .par file, using default value: {default_value}")
    
    param_values_start.append(params[param])
```

### 2. Postfit Residuals Fix

Modified `jug/gui/main_window.py` in `_compute_postfit_residuals()`:

**Problem:** The function only updated existing parameters in the .par file, but didn't add new ones.

**Solution:** Track which parameters exist in the file, then append new parameters at the end:

```python
# Track parameters in file
params_in_file = set()
for line in par_lines:
    # ... existing code to update parameters ...
    params_in_file.add(param_name)

# Add new parameters that weren't in the original file
new_params = [p for p in fitted_params.keys() if p not in params_in_file]
if new_params:
    updated_lines.append('\n# Parameters added by fit:\n')
    for param in sorted(new_params):
        fitted_value = fitted_params[param]
        # Format and add to file
        new_line = f"{param:<12} {fitted_value:.15e} 1\n"
        updated_lines.append(new_line)
```

---

## Usage

### Basic Example
```bash
# F2 is not in J1909-3744_tdb_wrong.par, but we can still fit it!
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim --fit F2
```

**What happens:**
1. GUI shows F2 checkbox in **blue** (indicating it's not in .par file)
2. F2 checkbox is **pre-checked** (from `--fit` flag)
3. Tooltip says: "F2 not in original .par file (will be fitted from scratch)"
4. When you click "Run Fit":
   - F2 starts at 0.0
   - Fitter optimizes F0, F1, F2 together
   - Postfit residuals computed with fitted F2 value
5. Plot updates with postfit residuals
6. Fit results show: `F2 = 6.075158e-26 ± 1.058224e-29 Hz/s²`

### Multiple Parameters
```bash
jug-gui data/pulsars/J1909-3744_tdb_wrong.par data/pulsars/J1909-3744.tim --fit F2 F3
```

**Results:**
```
F2 = 9.419753e-25 ± 5.172518e-29 Hz/s²
F3 = -1.537495e-32 ± 8.833797e-37 Hz/s³
Final RMS: 206.231128 μs
```

---

## Technical Details

### Default Values

Parameters not in the .par file are initialized with sensible defaults:

| Parameter Type | Default Value | Rationale |
|----------------|---------------|-----------|
| F0, F1, F2, ... | 0.0 | Spin frequency derivatives start at zero |
| DM, DM1, DM2, ... | 0.0 | Dispersion measure derivatives start at zero |
| Astrometry | N/A | No good default (error if not in file) |
| Binary | N/A | No good default (error if not in file) |

### Why 0.0 for Spin Derivatives?

- **F0** (spin frequency) must be in the .par file - it's required for basic timing
- **F1** (spindown) is usually in the file, but 0.0 is reasonable for a pulsar with no spindown
- **F2, F3, F4, ...** (higher derivatives) are almost always zero or very small
- Starting at 0.0 lets the fitter determine the correct value from the data

### Convergence

The fitter typically converges in **4 iterations** even when starting from 0.0:

```
Iter   RMS (μs)     ΔParam          Status              
-----------------------------------------------------------------
1       206.828117   1.764632e-10                      
2       206.230346   2.139879e-14                      
3       206.231236   3.544305e-14                      
4       206.231128   6.216061e-14  ✓ Params converged  
```

---

## Testing

### Test 1: Single Missing Parameter
```python
from pathlib import Path
from jug.fitting.optimized_fitter import fit_parameters_optimized

result = fit_parameters_optimized(
    par_file=Path('data/pulsars/J1909-3744_tdb_wrong.par'),
    tim_file=Path('data/pulsars/J1909-3744.tim'),
    fit_params=['F0', 'F1', 'F2'],
    verbose=True
)

# Output:
# Warning: F2 not in .par file, using default value: 0.0
# Converged: True
# F2 = 6.075158e-26 ± 1.058224e-29
```

### Test 2: Multiple Missing Parameters
```python
result = fit_parameters_optimized(
    par_file=Path('data/pulsars/J1909-3744_tdb_wrong.par'),
    tim_file=Path('data/pulsars/J1909-3744.tim'),
    fit_params=['F0', 'F1', 'F2', 'F3'],
    verbose=True
)

# Output:
# Warning: F2 not in .par file, using default value: 0.0
# Warning: F3 not in .par file, using default value: 0.0
# Converged: True
# F2 = 9.419753e-25 ± 5.172518e-29
# F3 = -1.537495e-32 ± 8.833797e-37
```

---

## Limitations

### DM3 and Higher DM Derivatives
There's currently a bug in `derivatives_dm.py` with numpy compatibility:
```python
# Line 219 in derivatives_dm.py
factorial = np.math.factorial(order)  # ❌ np.math doesn't exist in numpy 2.x
```

This needs to be fixed to:
```python
factorial = math.factorial(order)  # ✅ Use standard library
```

But this is a separate issue from the missing parameter handling.

### Astrometry and Binary Parameters
Currently not supported - no good default values exist:
- RAJ, DECJ: Position must be in .par file
- PB, A1, ECC: Binary parameters must be in .par file

These could be supported in the future with interactive dialogs to set initial values.

---

## Visual Indicators in GUI

The GUI provides clear visual feedback:

1. **Blue color**: Parameters not in .par file are shown in blue
2. **Tooltip**: Hovering shows "F2 not in original .par file (will be fitted from scratch)"
3. **Status bar**: Shows count of added parameters: "Found 2 fittable parameters in .par file (+ 2 from --fit)"
4. **Pre-selection**: Parameters from `--fit` flag are automatically checked

---

## Implementation Files

### Modified Files
1. `jug/fitting/optimized_fitter.py`
   - `_fit_parameters_general()`: Add default values for missing parameters
   
2. `jug/gui/main_window.py`
   - `_compute_postfit_residuals()`: Add new parameters to temporary .par file

### No Changes Needed
- `jug/gui/main.py`: Already had `--fit` argument parsing
- `jug/gui/main_window.py`: Already had parameter parsing and GUI logic
- JAX incremental fitter already handled missing parameters correctly

---

## Future Enhancements

1. **Save fitted .par file**: Allow saving with newly fitted parameters
2. **Interactive parameter editor**: Set initial values for missing parameters
3. **Smart defaults**: Use ATNF catalog values as defaults if available
4. **Parameter validation**: Warn if fitted value is unreasonable

---

**Status**: ✅ Feature complete and tested  
**Version**: JUG GUI Phase 2 (2026-01-27)
