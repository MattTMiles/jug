# Testing JUG

Quick start for running tests and validating functionality.

## One-Command Test Validation

```bash
# From repo root, run all tests
python tests/run_all.py

# Quick validation (skip slow tests)
python tests/run_all.py --quick

# Verbose output for debugging
python tests/run_all.py -v

# Run specific tests
python tests/run_all.py imports prebinary_cache
```

## What the Tests Check

| Test | Purpose | Duration |
|------|---------|----------|
| `imports` | Core module imports | <1s |
| `prebinary_cache` | Cache path regression | ~2s |
| `timescale_validation` | TDB/TCB handling | ~2s |
| `binary_patch` | Binary delay correctness | ~3s |
| `astrometry_fitting` | Astrometry parameters | ~4s |
| `j2241_fit` | Full parameter fitting | ~3s |

Use `python tests/run_all.py --quick` to skip `j2241_fit`.

## CI/Portable Test Data

Tests auto-skip if data is missing. To run on CI/other machines, set:

```bash
export JUG_TEST_DATA_DIR=/path/to/data
python tests/run_all.py
```

Or per-pulsar:
```bash
export JUG_TEST_J1713_PAR=/path/to/J1713+0747.par
export JUG_TEST_J1713_TIM=/path/to/J1713+0747.tim
python tests/run_all.py
```

Check your setup:
```bash
python tests/test_paths.py
```

## Debug Scripts

Debug/diagnostic scripts are in `playground/`. Run manually as needed:

```bash
python playground/debug_dd_delay.py
python playground/compare_jug_pint.py
```

See [docs/DEBUG_WORKFLOW.md](DEBUG_WORKFLOW.md) for the full workflow guide.
