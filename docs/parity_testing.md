# Parity & Regression Testing

This document explains how to run and maintain JUG's parity and regression
test infrastructure.

## Quick Start

```bash
cd /home/mattm/soft/JUG
conda activate discotech

# Run all tests (excluding Tempo2 integration)
pytest jug/tests/ -v -o "addopts=" -m "not integration"

# Run only parity/regression tests
pytest jug/tests/test_regression_parity.py jug/tests/test_parity_smoke.py -v -o "addopts="

# Run derivative validation
pytest jug/tests/test_derivative_validation.py -v -o "addopts="

# Run Tempo2 integration (requires tempo2 on PATH)
pytest jug/tests/test_parity_smoke.py -v -o "addopts=" -m integration
```

## Test Files

| File | Purpose | Markers |
|------|---------|---------|
| `test_regression_parity.py` | Eval-only vs fitter identity, dt\_sec precision, golden artifact integrity | — |
| `test_derivative_validation.py` | Analytic vs finite-difference derivatives for all fitted params | — |
| `test_parity_smoke.py` | Cross-pulsar sanity (WRMS, determinism, fingerprinting, fitting convergence) | `integration` for Tempo2 |
| `test_golden_regression.py` | Bit-for-bit regression against stored `.npy` golden files | — |

## Pulsars Covered

| Pulsar | Binary Model | Key Features | TOAs |
|--------|-------------|--------------|------|
| J1909-3744 | ELL1 | SINI+M2, DM1/DM2, FD1–FD9, NE\_SW, PBDOT, XDOT | 10,408 |
| J0614-3329 | DD | T0/ECC/OM, SINI+M2, DM1/DM2, PBDOT, FD1 | 10,099 |

## Derivative Validation

The derivative test compares **analytic** design-matrix columns (used in WLS
fitting) against **central finite-difference** derivatives of the forward model.

Two metrics are checked per parameter:

1. **Pearson correlation** > 0.99999
2. **Max relative error** < 1e-5 (default) or 1e-3 (for noisy params like PBDOT, XDOT)

Relative error is defined as `max|analytic - FD| / max|analytic|`.

Parameter families tested:
- **Spin**: F0, F1 — special FD: perturbs phase numerator only, keeps F0 fixed in denominator to match analytic convention `d(delay)/d(Fn) = -d(phase)/d(Fn) / F0`
- **DM**: DM, DM1, DM2
- **Astrometry**: RAJ, DECJ, PMRA, PMDEC, PX
- **Binary ELL1**: PB, A1, TASC, EPS1, EPS2, PBDOT, XDOT, SINI, M2
- **Binary DD**: PB, A1, T0, ECC, OM, SINI, M2, PBDOT
- **FD**: FD1–FD9
- **Solar wind**: NE\_SW — uses the same `K_DM_SEC` constant as the analytic code

## Tempo2 Parity Harness

The `tools/parity_harness.py` script runs both Tempo2 and JUG on the same
par+tim and produces a per-TOA comparison report.

```bash
# Default (J0125-2327)
python tools/parity_harness.py --assert-thresholds

# Custom pulsar
python tools/parity_harness.py \
    --par data/pulsars/J1909-3744_tdb.par \
    --tim data/pulsars/J1909-3744.tim \
    --assert-thresholds
```

Golden artifacts are saved to `tests/data_golden/<PSRJ>_parity.npz` and `.json`.

### Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Per-TOA max\|Δ\| | 50 ns | Clock-chain implementation differences (~16 ns systematic) |
| \|ΔWRMS\| | 1.0 ns | Sub-ns WRMS agreement despite per-TOA offsets |

### Known Systematic Offset

A ~16 ns systematic offset exists between JUG and Tempo2 residuals.
This is caused by independent clock correction implementations:
- JUG uses Astropy's TDB conversion chain
- Tempo2 uses its own native clock files

This offset is expected and stable. The 50 ns per-TOA threshold accommodates
it while still catching real regressions.

## Config Fingerprinting

The `jug.testing.fingerprint` module extracts critical configuration from par
files and validates JUG compatibility:

```python
from jug.testing.fingerprint import extract_fingerprint, validate_jug_compatible

# Check compatibility
ok, issues = validate_jug_compatible("pulsar.par")
assert ok, f"Not JUG-compatible: {issues}"

# Compare two par files
fp = extract_fingerprint("pulsar.par")
print(fp)
```

Required settings for JUG: `UNITS=TDB`, `EPHEM=DE440`, `CLK=TT(BIPM2024)`.

## Updating Golden Data

If you make an **intentional** change to the residuals engine:

1. Run the existing tests to confirm they fail as expected.
2. Regenerate golden files:
   ```bash
   python -m jug.tests.golden.generate_golden     # bit-for-bit goldens
   python tools/parity_harness.py                  # Tempo2 parity goldens
   ```
3. Review changes carefully — any shift should be explainable.
4. Commit the updated golden files.
