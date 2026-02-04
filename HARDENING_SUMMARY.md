# JUG Repository Hardening - Summary of Changes

This document summarizes the repository hardening improvements made to ensure
long-term maintainability and prevent testing/debug workflow rot.

## Changes Made

### 1. One-Command Test Runner ✓

**File**: `tests/run_all.py` (NEW)

A lightweight test aggregator that runs all tests in order with a clear PASS/FAIL
summary. Supports:
- One-command validation: `python tests/run_all.py`
- Quick mode: `python tests/run_all.py --quick` (skip slow tests)
- Specific tests: `python tests/run_all.py test1 test2`
- Verbose output: `python tests/run_all.py -v`
- Test listing: `python tests/run_all.py --list`

Features:
- Critical tests are marked and must pass
- Standard tests are checked but don't block
- Slow tests can be skipped for fast CI
- Script-style tests are wrapped and executed with timeout
- Inline "quick checks" (imports, session workflow)
- Exits with nonzero on any failure
- Shows execution time for each test
- Gracefully skips tests when data is unavailable

Tests included:
- `imports` [critical] - Core module imports
- `prebinary_cache` [critical] - Regression: prebinary_delay_sec cache path
- `timescale_validation` - TDB/TCB par file handling
- `binary_patch` - Binary delay vs PINT comparison
- `astrometry_fitting` - Astrometry parameter fitting
- `j2241_fit` [slow] - J2241-5236 FB parameter fitting

### 2. CI-Friendly Test Data Paths ✓

**File**: `tests/test_paths.py` (NEW)

A portable path resolution system for test data that supports:

Environment variables:
- `JUG_TEST_DATA_DIR` - Base directory for all test data
- `JUG_TEST_J1713_PAR`, `JUG_TEST_J1713_TIM` - Per-pulsar overrides
- `JUG_TEST_J2241_PAR`, `JUG_TEST_J2241_TIM`
- `JUG_TEST_J1909_PAR`, `JUG_TEST_J1909_TIM`
- `JUG_TEST_J1022_PAR`, `JUG_TEST_J1022_TIM`

Defaults:
- Falls back to Matt's local paths if env vars not set
- Maintains backward compatibility

Helper functions:
- `get_j1713_paths()`, `get_j2241_paths()`, etc.
- `skip_if_missing(par, tim)` - Graceful skip with clear message
- `require_files(par, tim)` - Hard requirement for "must run" tests
- `files_exist(par, tim)` - Check existence
- `get_available_datasets()` - Find all configured datasets

Usage in tests:
```python
from tests.test_paths import get_j1713_paths, skip_if_missing

par, tim = get_j1713_paths()
if not skip_if_missing(par, tim, "test_name"):
    sys.exit(0)  # Graceful skip
```

### 3. Updated Tests to Use Path Helper ✓

**Files Modified**:
- `tests/test_cache_prebinary_regression.py`
- `tests/test_timescale_validation.py`
- `tests/test_binary_patch.py`
- `tests/test_j2241_fit.py`

All updated to:
- Import from `test_paths`
- Use environment-friendly path resolution
- Support graceful skips when data unavailable
- Maintain compatibility with existing hardcoded paths

### 4. Playground Cleanup (.gitignore) ✓

**File**: `playground/.gitignore` (NEW)

Configured to:
- **Allow** (keep tracked): Debug scripts (`debug_*.py`, `diagnose_*.py`, etc.)
- **Ignore** (don't track): Generated outputs
  - `*.txt` - Text outputs
  - `*.log` - Log files
  - `*.par`, `*.tim` - Temporary par/tim files
  - `*.patch` - Patch files
  - `*.png`, `*.jpg` - Image outputs
  - `*.npz`, `*.pkl` - Data files
  - `__pycache__/` - Python cache
  - `.ipynb_checkpoints/` - Jupyter cache
  - Editor files, OS files, etc.

This allows playground to stay clean while keeping valuable scripts tracked.

### 5. Documentation ✓

**Files Created/Updated**:
- `docs/DEBUG_WORKFLOW.md` (NEW) - Comprehensive debug workflow guide
- `docs/TESTING.md` (NEW) - Quick testing reference

#### DEBUG_WORKFLOW.md
Covers:
- Quick reference commands
- Directory structure
- Running tests (all, quick, specific)
- CI/portable test data setup
- Debug scripts (where they live, what they do)
- Timescale handling (TDB support, TCB failure, TZRMJD)
- Prebinary delay logic (what it is, why it matters, cache fallback)
- Adding new tests
- Troubleshooting

#### TESTING.md
Quick reference for:
- One-command validation
- What each test checks
- CI setup
- Where debug scripts live

## Verification

All tests pass:
```
✓ imports: PASS (0.3s)
✓ prebinary_cache: PASS (2.0s)
✓ timescale_validation: PASS (1.4s)
✓ binary_patch: PASS (2.3s)
✓ astrometry_fitting: PASS (4.7s)
✓ j2241_fit: PASS (3.1s)

SUMMARY: 6 passed, 0 failed, 0 skipped, 0 errors
Total time: 13.7s
RESULT: PASS
```

## Usage

### Developers (Local)
```bash
# Quick validation
python tests/run_all.py --quick

# Full validation
python tests/run_all.py

# Debug workflow
python playground/debug_dd_delay.py
python tests/test_paths.py
```

### CI/Remote Machines
```bash
export JUG_TEST_DATA_DIR=/path/to/test/data
python tests/run_all.py --quick  # 11.8s
# or full test suite (13.7s)
python tests/run_all.py
```

### Documentation
```bash
# See full debug workflow
cat docs/DEBUG_WORKFLOW.md

# Quick reference
cat docs/TESTING.md
```

## Benefits

1. **One-command validation**: `python tests/run_all.py` from repo root
2. **CI-friendly**: Environment variables eliminate hardcoded paths
3. **Graceful skips**: Tests skip when data unavailable rather than fail
4. **Clean playground**: Debug scripts stay tracked, outputs are ignored
5. **Documented**: New developers can understand the workflow
6. **Resistant to rot**: Tests are easy to run so they stay maintained

## Files Added

```
tests/run_all.py                 - Test runner (executable)
tests/test_paths.py              - Test data path utilities
playground/.gitignore            - Cleanup configuration
docs/DEBUG_WORKFLOW.md           - Comprehensive workflow guide
docs/TESTING.md                  - Quick test reference
HARDENING_SUMMARY.md             - This file
```

## Files Modified

```
tests/test_cache_prebinary_regression.py - Use test_paths helper
tests/test_timescale_validation.py       - Use test_paths helper
tests/test_binary_patch.py               - Use test_paths helper
tests/test_j2241_fit.py                  - Use test_paths helper
```

## Backward Compatibility

All changes are backward compatible:
- Existing tests still work (now via test_paths)
- Hardcoded paths still work (used as defaults when env vars not set)
- New tests can adopt path helpers gradually
- No breaking changes to any APIs

## Next Steps (Optional)

1. **Pytest integration**: If desired, tests can be made pytest-discoverable
   - Current script-style is simpler and works well
   - Can be added without disrupting current runner

2. **Coverage tracking**: Add `--cov` option to run_all.py
   - Already installed: `pytest-cov`
   - Can wrap individual test execution

3. **GitHub Actions**: Use `python tests/run_all.py --quick` in CI
   - No Docker/container setup needed
   - ~12s to run quick tests, ~14s for full suite

4. **Pre-commit hooks**: Add test execution before commits
   - Keeps tests relevant
   - Prevents regressions

---

Created: February 3, 2026
Status: Complete and verified
