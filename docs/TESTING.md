# Testing JUG

Quick start for running tests and validating functionality.

## One-Command Test Validation

```bash
# From repo root, run all tests
python tests/run_all.py

# Quick validation (no external data, skip slow tests) - ~16s
python tests/run_all.py --quick

# Skip GUI tests (for headless CI)
python tests/run_all.py --quick --no-gui

# With PINT cross-validation (~22s)
python tests/run_all.py --quick --pint

# Verbose output for debugging
python tests/run_all.py -v

# Run specific tests
python tests/run_all.py imports prebinary_cache

# Run only a category
python tests/run_all.py -c api
python tests/run_all.py -c correctness
python tests/run_all.py -c cli

# List all available tests
python tests/run_all.py --list
```

## Test Categories

| Category | Description | Data Required | GUI Required |
|----------|-------------|---------------|--------------|
| `critical` | Must pass - core imports | No | No |
| `cli` | CLI smoke and integration tests | No | No |
| `api` | Python API workflow | No (uses mini) | No |
| `correctness` | Golden reference validation | No (uses mini) | No |
| `gui` | GUI initialization tests | No | Yes |
| `standard` | Standard validation tests | Yes | No |
| `slow` | Long-running tests | Yes | No |

## What the Tests Check

| Test | Category | Purpose | Duration |
|------|----------|---------|----------|
| `imports` | critical | Core module imports | <1s |
| `prebinary_cache` | critical | Cache path regression | ~2s |
| `ddk_not_implemented` | critical | DDK raises NotImplementedError | ~1s |
| `cli_smoke` | cli | CLI entry points respond to --help | ~3s |
| `cli_integration` | cli | CLI compute/fit end-to-end | ~5s |
| `api_workflow` | api | Python API with bundled data | ~2s |
| `correctness` | correctness | Residuals match golden values + checksum | ~2s |
| `fit_correctness` | correctness | Fit reduces RMS, deterministic, finite params | ~2s |
| `invariants` | correctness | Prebinary time, fit recovery, gradient sanity | ~3s |
| `gui_smoke` | gui | GUI initializes, computes, fits headless | ~3s |
| `timescale_validation` | standard | TDB/TCB handling | ~2s |
| `binary_patch` | standard | Binary delay correctness | ~3s |
| `astrometry_fitting` | standard | Astrometry parameters | ~4s |
| `j2241_fit` | slow | Full parameter fitting | ~3s |

## Quick Mode (CI-Friendly)

Use `--quick` for fast CI validation without external data:

```bash
python tests/run_all.py --quick
```

This runs (~16s total):
- Import tests (critical)
- Cache regression tests (critical)
- CLI smoke tests (help flags)
- CLI integration tests (compute/fit with mini data)
- API workflow tests (uses bundled mini data)
- Correctness tests (golden reference + checksum)
- Fit correctness tests (RMS reduction, determinism)
- GUI smoke tests (skipped if no DISPLAY)

Add `--no-gui` to explicitly skip GUI tests in headless environments.

## PINT Cross-Validation

The `--pint` flag adds optional PINT cross-validation:

```bash
python tests/run_all.py --quick --pint
```

This compares JUG residuals against PINT as an informational check. Note that
JUG and PINT may produce somewhat different residuals due to:
- Different ephemeris versions (JUG uses DE440, PINT defaults to DE421)
- Different clock correction handling
- Different binary delay algorithms

The test verifies both codes produce reasonable residual patterns (same
number of TOAs, similar RMS magnitude) rather than exact agreement.

## Bundled Test Data

The `tests/data_golden/` directory contains:
- `J1909_mini.par` - Simplified par file (20 TOAs, ELL1 binary, DM=10.39)
- `J1909_mini.tim` - Mini tim file (20 TOAs)
- `J1909_mini_golden.json` - Golden reference values with:
  - Expected RMS values (µs)
  - First 5 residuals (ns precision)
  - Residual checksum (first 10 rounded to 10ns)
  - Tolerances: `rms_rel_tol=1e-5`, `residual_abs_tol_ns=1.0`

These enable CI tests to run without external data dependencies.

**Note**: The mini dataset has nonzero DM and CORRECT_TROPOSPHERE=Y, ensuring
`prebinary_delay_sec` differs from `roemer_shapiro_sec` (required for invariant tests).

## Environment Variables

### DDK Override

JUG does not support the DDK binary model (requires Kopeikin terms not implemented).
By default, DDK par files raise `NotImplementedError`. For testing or comparison:

```bash
# Force DDK to be treated as DD (INCORRECT for high-parallax pulsars)
JUG_ALLOW_DDK_AS_DD=1 python -m jug.scripts.compute_residuals par tim

# Also works with Python API
JUG_ALLOW_DDK_AS_DD=1 python -c "from jug.residuals.simple_calculator import compute_residuals_simple; ..."
```

**Warning**: This override produces scientifically incorrect results for pulsars
where Kopeikin corrections are significant (e.g., J0437-4715). Use only for testing.

## CI/Portable Test Data

For external data tests, set environment variables:

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

## GitHub Actions

The `.github/workflows/tests.yml` workflow runs:

1. **Quick tests** - Every push, Python 3.10/3.11/3.12, no external data
2. **Full tests** - On commits containing `[full-tests]`, includes GUI
3. **PINT validation** - On PRs, cross-validates against PINT
4. **Lint** - Code quality checks with ruff/black

## Correctness Validation

JUG validates correctness by comparing computed residuals against:

1. **Golden reference** - Pre-computed values in `tests/data_golden/`
2. **PINT (optional)** - Cross-validation with `--pint` flag

To regenerate golden values after intentional changes:
```bash
python tests/generate_golden.py
```

## Debug Scripts

Debug/diagnostic scripts are in `playground/`. Run manually as needed:

```bash
python playground/debug_dd_delay.py
python playground/compare_jug_pint.py
```

See [docs/DEBUG_WORKFLOW.md](DEBUG_WORKFLOW.md) for the full workflow guide.
