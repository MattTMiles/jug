# JUG Debug and Validation Workflow

This document describes the testing, debugging, and validation workflow for JUG.

## Quick Reference

| Task | Command |
|------|---------|
| Run all tests | `python tests/run_all.py` |
| Run quick tests only | `python tests/run_all.py --quick` |
| List available tests | `python tests/run_all.py --list` |
| Check test data paths | `python tests/test_paths.py` |

---

## Directory Structure

```
JUG/
├── tests/              # Automated tests (run via run_all.py)
│   ├── run_all.py      # One-command test runner
│   ├── test_paths.py   # Test data path utilities
│   └── test_*.py       # Individual test scripts
├── playground/         # Debug/diagnostic scripts (not run automatically)
│   ├── debug_*.py      # Debugging scripts
│   ├── diagnose_*.py   # Diagnostic scripts
│   ├── compare_*.py    # Comparison scripts (JUG vs PINT/Tempo2)
│   └── *.md            # Session notes and status docs
└── docs/               # Documentation
```

---

## Running Tests

### One-Command Validation

```bash
# From repo root - run all tests
python tests/run_all.py

# Quick validation (skip slow tests)
python tests/run_all.py --quick

# Verbose output for debugging failures
python tests/run_all.py -v

# Run specific test(s)
python tests/run_all.py prebinary_cache timescale_validation
```

### Test Categories

- **Critical**: Must pass. Failures indicate broken core functionality.
  - `imports`: Core module imports
  - `prebinary_cache`: Regression guard for GUI crash

- **Standard**: Should pass. Failures indicate bugs but not complete breakage.
  - `timescale_validation`: TDB/TCB par file handling
  - `binary_patch`: Binary delay vs PINT
  - `astrometry_fitting`: Astrometry parameter fitting

- **Slow**: Take longer to run. Skipped with `--quick`.
  - `j2241_fit`: Full FB parameter fitting test

### CI/Portable Test Data

Tests use environment variables to locate test data:

```bash
# Base directory (tests look for J1713+0747_tdb.par, etc.)
export JUG_TEST_DATA_DIR=/path/to/test/data

# Or per-pulsar overrides
export JUG_TEST_J1713_PAR=/path/to/J1713+0747.par
export JUG_TEST_J1713_TIM=/path/to/J1713+0747.tim
export JUG_TEST_J2241_PAR=/path/to/J2241-5236.par
export JUG_TEST_J2241_TIM=/path/to/J2241-5236.tim
```

If files are missing, tests **SKIP** with a clear message rather than fail.

Check your configuration:
```bash
python tests/test_paths.py
```

---

## Debug Scripts (playground/)

The `playground/` directory contains debugging and diagnostic scripts. These are
**not run automatically** - use them for manual investigation.

### Debugging Binary Models
- `debug_dd_*.py` - DD binary model debugging
- `diagnose_binary_time*.py` - Binary timing diagnostics

### Comparing with PINT/Tempo2
- `compare_jug_pint.py` - Full JUG vs PINT comparison
- `compare_dd_delays.py` - Binary delay comparison

### Fitting Diagnostics
- `diagnose_fit_divergence.py` - Fitting convergence issues
- `diagnose_astrometry_fit*.py` - Astrometry fitting stability

### Running Debug Scripts

```bash
# From repo root
python playground/debug_dd_delay.py

# Or from playground/
cd playground
python debug_dd_delay.py
```

---

## Timescale Handling

JUG strictly enforces correct timescale handling for par files.

### Supported: UNITS=TDB

- Par files with `UNITS TDB` work normally
- All epochs (PEPOCH, T0, TASC, TZRMJD, etc.) are treated as TDB
- This is the standard for modern pulsar timing

### Not Supported: UNITS=TCB

TCB par files will **hard-fail** with a clear error message explaining:
1. TCB requires scaling spin parameters (F0, F1) by L_B
2. TCB requires epoch conversion to TDB
3. How to convert: use PINT or Tempo2 to produce TDB par files

```python
# This will raise NotImplementedError with guidance
compute_residuals_simple("tcb_pulsar.par", "pulsar.tim")
```

### TZRMJD Handling

The `tzrmjd_scale` parameter controls how TZRMJD is interpreted:

| Value | Behavior |
|-------|----------|
| `"AUTO"` (default) | Derive from par file `UNITS` keyword |
| `"TDB"` | Force TDB interpretation (no conversion) |
| `"UTC"` | Force UTC→TDB conversion (legacy, warns loudly) |

**Recommendation**: Use the default `"AUTO"` which derives the timescale from
the par file's `UNITS` keyword. This ensures consistency.

```python
# Recommended (default)
compute_residuals_simple(par, tim)  # tzrmjd_scale="AUTO"

# Override if you know better (rare)
compute_residuals_simple(par, tim, tzrmjd_scale="TDB")
```

---

## Prebinary Delay Logic

JUG uses `prebinary_delay_sec` for PINT/Tempo2-compatible binary evaluation.

### What is prebinary_delay_sec?

The time at which to evaluate binary orbital effects:

```
prebinary_delay_sec = roemer_delay + shapiro_delay + dm_delay + sw_delay + tropo_delay
```

This is the sum of all delays that occur **before** the binary system in the
signal path (from pulsar → binary orbit → SSB → observatory).

### Why Does This Matter?

PINT and Tempo2 evaluate binary models at this "prebinary time":

```python
# Correct (PINT-compatible)
t_binary = TDB - prebinary_delay_sec / 86400

# Incomplete (missing DM, SW, tropo)
t_binary = TDB - roemer_shapiro_sec / 86400  # ← causes ~μs errors
```

### Cache Fallback Warning

If you see this warning:

```
Cache missing 'prebinary_delay_sec' - falling back to roemer_shapiro_sec only.
IMPACT: Binary timing model will NOT be PINT-compatible (missing DM+SW+tropo corrections).
```

**Fix it by**:
1. Close the GUI and restart with fresh data, OR
2. In Python: `session.compute_residuals(force_recompute=True)`, OR
3. Delete `session._cached_result_by_mode` and recompute

This happens when cached data was produced by an older JUG version before
`prebinary_delay_sec` was added.

---

## Adding New Tests

1. Create `tests/test_yourtest.py`
2. Use `test_paths.py` for portable data paths:
   ```python
   from tests.test_paths import get_j1713_paths, skip_if_missing
   
   par, tim = get_j1713_paths()
   if not skip_if_missing(par, tim, "yourtest"):
       sys.exit(0)  # Skip gracefully
   ```
3. Add to `TESTS` list in `tests/run_all.py`
4. Test locally: `python tests/run_all.py yourtest`

---

## Troubleshooting

### Tests Skip Due to Missing Data

```
SKIP: PAR file not found: /home/mattm/...
```

Set environment variables or ensure default paths exist:
```bash
export JUG_TEST_DATA_DIR=/your/data/dir
python tests/run_all.py
```

### Import Errors

```
ImportError: No module named 'jug'
```

Install JUG in development mode:
```bash
pip install -e .
```

### GUI Tests Fail

GUI tests may fail if Qt/PySide6 is not properly configured:
```bash
# Install Qt dependencies
pip install PySide6 pyqtgraph

# For headless environments, use virtual display
export QT_QPA_PLATFORM=offscreen
```

---

## See Also

- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API quick reference
- [GUI_QUICK_START.md](GUI_QUICK_START.md) - GUI usage guide
