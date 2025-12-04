# JUG Code Cleanup Manifest

**Date**: 2025-12-04
**Purpose**: Archive experimental/unused code before implementing DM fitting

---

## Production Code (KEEP)

### Core Production Modules

These are actively used by CLI tools and documented API:

#### Entry Points
- ✅ `jug/scripts/fit_parameters.py` - CLI: `jug-fit`
- ✅ `jug/scripts/compute_residuals.py` - CLI: `jug-compute-residuals`
- ✅ `jug/fitting/optimized_fitter.py` - Main fitting API
- ✅ `jug/residuals/simple_calculator.py` - Main residuals API

#### Supporting Modules (Used by Production)
- ✅ `jug/io/par_reader.py` - Parse .par files
- ✅ `jug/io/tim_reader.py` - Parse .tim files
- ✅ `jug/io/clock.py` - Clock corrections
- ✅ `jug/delays/barycentric.py` - Barycentric delays
- ✅ `jug/delays/combined.py` - Combined delay calculations
- ✅ `jug/delays/binary_dispatch.py` - Binary model routing
- ✅ `jug/delays/binary_bt.py` - BT/DD binary model
- ✅ `jug/delays/binary_dd.py` - DD model variants
- ✅ `jug/delays/binary_t2.py` - T2 general binary model
- ✅ `jug/utils/constants.py` - Physical constants
- ✅ `jug/utils/device.py` - CPU/GPU selection
- ✅ `jug/fitting/derivatives_spin.py` - Spin parameter derivatives
- ✅ `jug/fitting/wls_fitter.py` - WLS solver
- ✅ `jug/fitting/design_matrix.py` - Design matrix construction

#### Infrastructure
- ✅ All `__init__.py` files
- ✅ `jug/tests/` - Test suite (keep all)
- ✅ `jug/models/` - Empty placeholder (keep for future)
- ✅ `jug/noise/` - Empty placeholder (keep for Milestone 3)
- ✅ `jug/gui/` - Empty placeholder (keep for Milestone 5)

---

## Experimental Code (ARCHIVE)

### Category 1: Alternative Fitter Implementations

These were experiments during development but superseded by `optimized_fitter.py`:

#### To Archive:
1. **`jug/fitting/optimizer.py`**
   - Reason: Early linearized fitter, superseded by optimized_fitter
   - Status: Exposed in __init__.py but never actually used
   - Archive as: `archival_code/fitting/optimizer.py`

2. **`jug/fitting/params.py`**
   - Reason: Parameter packing/unpacking utilities, not used by production
   - Status: Exposed in __init__.py but never called
   - Archive as: `archival_code/fitting/params.py`

3. **`jug/fitting/levenberg_marquardt.py`**
   - Reason: LM algorithm experiment, WLS proved sufficient
   - Status: Not imported anywhere
   - Archive as: `archival_code/fitting/levenberg_marquardt.py`

4. **`jug/fitting/gauss_newton.py`**
   - Reason: Gauss-Newton implementation, superseded
   - Status: Not used in production
   - Archive as: `archival_code/fitting/gauss_newton.py`

5. **`jug/fitting/gauss_newton_jax.py`**
   - Reason: JAX version of Gauss-Newton, superseded
   - Status: Not used in production
   - Archive as: `archival_code/fitting/gauss_newton_jax.py`

6. **`jug/fitting/longdouble_spin.py`**
   - Reason: Longdouble experiment, now handled in optimized_fitter
   - Status: Not imported anywhere
   - Archive as: `archival_code/fitting/longdouble_spin.py`

### Category 2: Alternative Derivative/Matrix Implementations

JAX versions that were experimental:

7. **`jug/fitting/derivatives_spin_jax.py`**
   - Reason: JAX derivatives experiment, `optimized_fitter` uses inline JAX now
   - Status: Superseded by inline implementation
   - Archive as: `archival_code/fitting/derivatives_spin_jax.py`

8. **`jug/fitting/design_matrix_jax.py`**
   - Reason: JAX design matrix, superseded by inline implementation
   - Status: Not used in production
   - Archive as: `archival_code/fitting/design_matrix_jax.py`

### Category 3: Caching/Performance Experiments

9. **`jug/fitting/cached_residuals.py`**
   - Reason: Caching experiment, now handled in optimized_fitter
   - Status: Not used in production
   - Archive as: `archival_code/fitting/cached_residuals.py`

10. **`jug/fitting/fast_residuals.py`**
    - Reason: Fast residuals experiment, superseded
    - Status: Not used in production
    - Archive as: `archival_code/fitting/fast_residuals.py`

11. **`jug/fitting/residuals_for_fitting.py`**
    - Reason: Residual computation for fitting, superseded
    - Status: Not used in production
    - Archive as: `archival_code/fitting/residuals_for_fitting.py`

12. **`jug/fitting/residual_wrapper.py`**
    - Reason: Wrapper experiment, not needed
    - Status: Not used in production
    - Archive as: `archival_code/fitting/residual_wrapper.py`

### Category 4: Statistics/Analysis Modules

13. **`jug/fitting/chi2.py`**
    - Reason: Chi-squared calculations, not used in current workflow
    - Status: Not imported anywhere
    - Archive as: `archival_code/fitting/chi2.py`

### Category 5: Alternative Residual Implementations

14. **`jug/residuals/core.py`**
    - Reason: Alternative residual calculator, `simple_calculator.py` is canonical
    - Status: Not imported anywhere
    - Archive as: `archival_code/residuals/core.py`

---

## Playground/Examples Code (ARCHIVE ALL)

### To Archive:
- **`playground/*.py`** - All experimental scripts (117 files)
- **`playground/*.md`** - Keep documentation for reference
- **`playground/*.ipynb`** - Keep notebooks for reference
- **`playground/*.txt`** - Keep analysis results
- **`playground/*.npz`** - Keep data files

Archive Python scripts to: `archival_code/playground/`

**Reason**: These are development/testing scripts that helped build JUG but aren't part of the production package. The documentation (MD files) and notebooks contain valuable insights and should be kept.

---

## Files to Update After Archival

### 1. `jug/fitting/__init__.py`

Remove from imports and `__all__`:
```python
# REMOVE these imports:
from jug.fitting.params import (...)
from jug.fitting.optimizer import fit_linearized

# REMOVE from __all__:
'extract_fittable_params',
'pack_params',
'unpack_params',
'get_param_scales',
'fit_linearized',
```

Keep only:
```python
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.fitting.optimized_fitter import fit_parameters_optimized

__all__ = [
    'compute_spin_derivatives',
    'wls_solve_svd',
    'fit_parameters_optimized'
]
```

---

## Archive Directory Structure

```
archival_code/
├── fitting/
│   ├── optimizer.py
│   ├── params.py
│   ├── levenberg_marquardt.py
│   ├── gauss_newton.py
│   ├── gauss_newton_jax.py
│   ├── longdouble_spin.py
│   ├── derivatives_spin_jax.py
│   ├── design_matrix_jax.py
│   ├── cached_residuals.py
│   ├── fast_residuals.py
│   ├── residuals_for_fitting.py
│   ├── residual_wrapper.py
│   └── chi2.py
├── residuals/
│   └── core.py
├── playground/
│   └── [117 .py files]
└── README.md  # This manifest with timestamps
```

---

## Summary

### Keep: 26 production modules + tests + infrastructure
### Archive: 14 fitting modules + 1 residuals module + 117 playground scripts = 132 files

**Impact**:
- No functionality loss (all archived code is superseded or experimental)
- Cleaner codebase for DM fitting implementation
- Clear production vs experimental boundary
- Easier maintenance and onboarding

---

## Verification Steps

After archival:
1. ✅ Run tests: `pytest jug/tests/`
2. ✅ Test CLI: `jug-fit --help` and `jug-compute-residuals --help`
3. ✅ Test fitting: `jug-fit data/pulsars/J1909-3744.par data/pulsars/J1909-3744.tim --fit F0 F1`
4. ✅ Run example notebook: `examples/full_walkthrough.ipynb`
5. ✅ Verify imports: `python -c "from jug.fitting import fit_parameters_optimized; print('OK')"`

---

## Next Steps After Cleanup

1. Implement DM derivatives in `jug/fitting/derivatives_dm.py`
2. Extend `optimized_fitter.py` to handle DM parameters
3. Add tests for DM fitting
4. Update QUICK_REFERENCE.md with DM fitting examples
