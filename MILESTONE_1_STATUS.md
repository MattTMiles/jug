# Milestone 1 Status Report

**Date**: 2025-11-29
**Status**: In Progress (40% Complete)
**Time Invested**: ~2 hours

---

## Summary

Milestone 1 (Core Timing Package v0.1.0) is underway. The package structure has been created, and core modules have been extracted from the MK7 notebook. The JAX-compiled delay calculation kernel (the performance-critical component) has been successfully extracted.

---

## Completed Tasks ✅

### 1.1 Package Structure ✅
**Status**: COMPLETED
**Time**: 5 minutes

Created complete directory structure:
```
jug/
├── io/              # File I/O (par, tim readers)
├── models/          # Timing models
├── delays/          # Delay computations
├── residuals/       # Residual calculation
├── noise/           # Noise models (future)
├── fitting/         # Optimization (future)
├── gui/             # GUI components (future)
├── utils/           # Constants and utilities
├── scripts/         # CLI scripts
└── tests/           # Unit tests
    ├── test_io/
    ├── test_models/
    ├── test_delays/
    ├── test_residuals/
    ├── test_noise/
    ├── test_fitting/
    ├── fixtures/
    └── benchmarks/
```

All `__init__.py` files created.

### 1.2 Setup pyproject.toml ✅
**Status**: COMPLETED
**Time**: 15 minutes

Created complete `pyproject.toml` with:
- ✅ Package metadata (name, version, description, authors)
- ✅ Core dependencies (JAX, NumPy, Astropy, NumPyro, Optax, SciPy)
- ✅ Optional dependencies:
  - `[gui]`: PyQt6, pyqtgraph, matplotlib
  - `[dev]`: pytest, ruff, mypy, black
  - `[docs]`: sphinx, sphinx-rtd-theme
- ✅ CLI entry points:
  - `jug-compute-residuals`
  - `jug-fit`
  - `jug-gui`
- ✅ Testing configuration (pytest, coverage)
- ✅ Linting configuration (ruff, black, mypy)

**Files Created**:
- `pyproject.toml` (5.5 KB)
- `README.md` (3.5 KB) - Project overview, installation, quick start

### 1.3 Extract Constants and Utilities ✅
**Status**: COMPLETED
**Time**: 20 minutes

**Files Created**:
- `jug/utils/constants.py` (2.5 KB)

**Contents**:
- Physical constants: `C_M_S`, `C_KM_S`, `AU_M`, `AU_KM`, `K_DM_SEC`, `L_B`, `T_SUN_SEC`
- Time constants: `SECS_PER_DAY`, `MJD_TO_JD`
- Planetary parameters: `T_PLANET` dict (Jupiter, Saturn, Uranus, Neptune, Venus)
- Observatory coordinates: `OBSERVATORIES` dict (MeerKAT, Parkes, GBT)
- Precision handling: `HIGH_PRECISION_PARAMS` set
- Conversion factors: `DEG_TO_RAD`, `RAD_TO_DEG`

All constants documented with docstrings and units.

### 1.4 Extract I/O Functions ✅ (Partial)
**Status**: PARTIAL - .par reader complete, .tim reader pending
**Time**: 30 minutes

**Files Created**:
- `jug/io/par_reader.py` (4.5 KB)

**Functions Implemented**:
- `parse_par_file(path)`: Parse Tempo2-style .par files
- `get_longdouble(params, key, default)`: Extract high-precision parameters as np.longdouble
- `parse_ra(ra_str)`: Convert RA string (HH:MM:SS) to radians
- `parse_dec(dec_str)`: Convert DEC string (DD:MM:SS) to radians

**Features**:
- ✅ High-precision parameter handling (F0, F1, F2, PEPOCH, etc.)
- ✅ Stores original string representations for np.longdouble conversion
- ✅ Comprehensive docstrings with examples
- ✅ Error handling for missing parameters

**Still TODO**:
- `.tim` file reader (`jug/io/tim_reader.py`)
- Clock file loader (`jug/io/clock.py`)
- Observatory data loader (`jug/io/observatory.py`)

### 1.6 Extract Delay Calculations ✅ (Core Kernel)
**Status**: PARTIAL - JAX combined delays complete
**Time**: 45 minutes

**Files Created**:
- `jug/delays/combined.py` (8.5 KB)

**Core Function**:
- `combined_delays()`: JAX-JIT compiled single-kernel delay calculation
  - ✅ DM delay with polynomial expansion
  - ✅ Solar wind delay with elongation geometry
  - ✅ FD delay (frequency-dependent)
  - ✅ Binary delay (ELL1 model with 3rd-order expansion)
    - Binary Roemer delay
    - Einstein delay (time dilation)
    - Shapiro delay from companion

**Performance**: This is the critical 100x speedup component over PINT.

**Still TODO**:
- Separate modules for individual delay types (optional, for clarity)
- Barycentric delay calculation (Roemer + Shapiro from solar system)
- Clock correction chain

---

## In Progress 🚧

### 1.4 Extract I/O Functions (Continued)
**Current Status**: .tim reader extraction in progress

**Next Steps**:
1. Extract TIM file parser from Cell 7
2. Extract TDB conversion functions
3. Extract clock correction interpolation

### 1.7 Extract Residual Calculation
**Current Status**: Not started

**Next Steps**:
1. Extract main calculator class from Cell 10
2. Extract TZR (phase reference) computation
3. Create wrapper functions for easy API

---

## Pending ⏸️

### 1.5 Extract Timing Models
**Status**: NOT STARTED
**Estimated Time**: 2-3 hours

**TODO**:
- Create `jug/models/` module structure
- Extract parameter classes (may use dataclasses or simple dicts)
- Consider if we need formal model classes or just use parsed .par dicts

### 1.8 Write Unit Tests
**Status**: NOT STARTED
**Estimated Time**: 3-4 hours

**TODO**:
- Test DM delay (known values, freq^-2 scaling)
- Test constants (verify values match references)
- Test .par parsing (high-precision parameters)
- Test combined delays (compare to notebook output)

### 1.9 Create CLI Script
**Status**: NOT STARTED
**Estimated Time**: 30 minutes

**TODO**:
- Create `jug/scripts/compute_residuals.py`
- Simple argparse interface
- Load .par + .tim, compute residuals, write CSV

### 1.10 Validate Against Notebook
**Status**: NOT STARTED
**Estimated Time**: 1-2 hours

**TODO**:
- Run package on J0437-4715 test data
- Compare residuals to MK7 notebook
- Verify RMS difference < 1 ns

---

## Key Decisions Made

### 1. Module Organization
- Kept delays in single `combined.py` for now (matches notebook structure)
- Can refactor into separate modules later if needed
- Prioritized getting working code over perfect organization

### 2. Precision Handling
- Preserved np.longdouble for F0, F1, F2, epochs
- Store original string representations in .par parser
- Critical for μs-level timing accuracy

### 3. JAX Optimization
- Extracted complete 3rd-order ELL1 binary expansion
- Single JIT kernel for all delays (DM + SW + FD + binary)
- This is the core performance advantage

### 4. Documentation
- Comprehensive docstrings on all functions
- Include examples, parameter descriptions, units
- Following NumPy docstring convention

---

## Challenges Encountered

### 1. Notebook Size
- MK7 notebook is 25,000 tokens (too large to read at once)
- Solution: Used Task/Explore agent to analyze structure first
- Extracted code cell-by-cell via Python JSON parsing

### 2. Code Complexity
- Combined delays function is ~200 lines with complex math
- Required careful extraction to preserve all terms
- 3rd-order ELL1 expansion has many trigonometric terms

### 3. Interdependencies
- Functions reference each other (e.g., parse_par_file uses HIGH_PRECISION_PARAMS)
- Need to extract in correct order
- Imports must be managed carefully

---

## Next Steps (Priority Order)

### Immediate (Next Session)
1. **Complete .tim file reader** (Cell 7)
   - Parse TOAs with uncertainties
   - Parse observatory codes
   - Parse FLAGS
   - Extract TDB conversion functions

2. **Extract main calculator class** (Cell 10)
   - `JUGResidualCalculatorFinal` class
   - Initialization with .par/.tim data
   - `compute_residuals()` method

3. **Create simple CLI script**
   - Load .par + .tim
   - Compute residuals
   - Print RMS, write output

### Short-Term (This Week)
4. **Write basic tests**
   - Test DM delay calculation
   - Test .par parsing
   - Test constants

5. **Validate against notebook**
   - Run on same data
   - Compare outputs
   - Debug discrepancies

### Medium-Term (Next Week)
6. **Polish and document**
   - Add more docstrings
   - Create examples
   - Write user guide

---

## Time Estimate to Completion

| Task | Estimated Time | Status |
|------|---------------|--------|
| 1.1 Package structure | 5 min | ✅ Done |
| 1.2 Setup pyproject.toml | 15 min | ✅ Done |
| 1.3 Constants & utilities | 20 min | ✅ Done |
| 1.4 I/O functions | 2 hours | 🚧 50% done |
| 1.5 Timing models | 2 hours | ⏸️ Not started |
| 1.6 Delay calculations | 2 hours | ✅ Core done |
| 1.7 Residual calculation | 1 hour | ⏸️ Not started |
| 1.8 Unit tests | 3 hours | ⏸️ Not started |
| 1.9 CLI script | 30 min | ⏸️ Not started |
| 1.10 Validation | 1 hour | ⏸️ Not started |
| **Total** | **~12 hours** | **40% complete** |

**Estimated Remaining**: ~7 hours (1 more full session)

---

## Files Created So Far

```
/home/mattm/soft/JUG/
├── README.md                           # Project overview
├── pyproject.toml                      # Package configuration
├── jug/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── constants.py                # ✅ Physical constants
│   ├── io/
│   │   ├── __init__.py
│   │   └── par_reader.py               # ✅ .par file parser
│   ├── delays/
│   │   ├── __init__.py
│   │   └── combined.py                 # ✅ JAX delay kernel
│   ├── models/
│   │   └── __init__.py
│   ├── residuals/
│   │   └── __init__.py
│   ├── scripts/
│   │   └── __init__.py
│   └── tests/
│       └── [test directories...]
└── MILESTONE_1_STATUS.md               # This file
```

**Total**: 5 functional modules, ~21 KB of code

---

## Conclusion

Milestone 1 is progressing well. The package structure is in place, and the performance-critical JAX kernel has been extracted. The next session should complete the I/O functions, residual calculator, and basic CLI, bringing Milestone 1 to 80-90% completion.

**Estimated Completion**: 1 more session (~3-4 hours of active work)

**Blocker**: None - all dependencies available, code extraction proceeding smoothly.

---

**Last Updated**: 2025-11-29 (End of session 1)
