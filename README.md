# JUG — JAX-based Unified pulsar timinG (?)

Fast, independent pulsar timing software built on JAX with automatic GPU acceleration.

JUG provides a complete pulsar timing workflow: load par/tim files, compute residuals, fit timing models with correlated noise, and inspect results — all from a Python API or interactive GUI.

## Installation

### From source (recommended)

```bash
git clone https://github.com/MattTMiles/jug.git
cd jug
```

**Option A: conda (recommended for GPU support)**

```bash
conda env create -f environment.yml
conda activate jug
```

**Option B: pip**

```bash
pip install -e .
```

### GPU support

JUG automatically uses a GPU if one is available. For CUDA GPU acceleration, install the GPU version of JAX:

```bash
pip install --upgrade "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

## Quick start

### Python API

```python
from jug.engine.session import TimingSession

# Load pulsar data
session = TimingSession("J1909-3744.par", "J1909-3744.tim")

# Inspect the model and data
session.print_model()
session.print_toas()

# Compute pre-fit residuals
result = session.compute_residuals(subtract_tzr=True)

# Fit — automatically fits all parameters flagged in the par file
fit_result = session.fit_parameters(max_iter=5, verbose=True)

# Add extra parameters to the fit
session.set_free('F2')
fit_result = session.fit_parameters(fit_params=['F2'], max_iter=5)

# Inspect results
session.parameter_table(fit_result)
session.summary()

# Save fitted par/tim files
session.save_par("J1909_fitted.par", fit_result=fit_result)
session.save_tim("J1909_fitted.tim")
```

### Interactive GUI

```bash
jug-gui
```

Load par and tim files via **File > Open .par** and **File > Open .tim**, select parameters to fit, and click **Fit**. The GUI supports:

- Backend-coloured residual plots
- Box zoom and box delete (select regions with mouse)
- Noise process toggling and subtraction
- Saving fitted par/tim files via **File > Save .par / Save .tim**

To launch with a specific pulsar:

```bash
jug-gui --par J1909-3744.par --tim J1909-3744.tim
```

## Examples

See [`notebooks/jug_example_j1909.ipynb`](notebooks/jug_example_j1909.ipynb) for a complete walkthrough using J1909-3744 from the NANOGrav 15-year dataset, including:

- Loading and inspecting the timing model and TOAs
- Pre-fit and post-fit residual plots (coloured by backend)
- Fitting with noise processes (ECORR, EQUAD, EFAC, red noise)
- Noise realizations and whitened residuals
- Gaussianity testing (Anderson-Darling)
- Saving and round-trip verification of par/tim files

## Dependencies

- Python >= 3.10
- JAX >= 0.4.0
- NumPy, SciPy, Astropy
- PySide6 + pyqtgraph (GUI)
- matplotlib (plotting)

### Optional Tempo2 / libstempo testing

Tempo2-compatible tests use a subprocess sandbox around `libstempo` so crashes
in Tempo2 do not kill the pytest process. To run these tests, install Tempo2
and libstempo in the active environment and make sure Tempo2 runtime data are
available:

```bash
conda install -c conda-forge tempo2 libstempo
export TEMPO2=/path/to/T2runtime
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_*.py -q -o addopts=''
```

The curated fixtures live in `tests/data_tempo2/`. The manifest includes:

- Case A (TCB regression fixtures),
- Case B (NG5 equatorial TDB),
- Case C (NG5 ecliptic cross-engine TDB).

`tests/tempo2_fixtures.py` exposes helpers to select parity fixtures by case
for CI and local debugging.

### Compatibility modes and residual conventions

JUG exposes two compatibility families:

- `compatibility="pint"`: PINT-family runtime conventions and weighted residual
  mean subtraction.
- `compatibility="tempo2"`: tempo2-family runtime conventions and unweighted
  residual mean subtraction.

When evaluating Tempo2 parity, use **raw pre-fit residuals** only (no post-hoc
mean centering). This is the acceptance metric used by
`tests/test_tempo2_residual_parity.py`.

For notebook diagnostics, weighted-mean-centered deltas are useful only for
PINT-family-vs-PINT-family comparisons; tempo2-labeled comparisons should stay
raw.

**Nonlinear / autodiff / MetaPulsar:** green residual tests on curated fixtures
do **not** mean tempo2 mode is ready for JAX-traced likelihoods or IPTA-scale
workloads. See [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md) for policy and
[`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) for gap analysis, pytempo workflow, and usage guidance.

### Tempo2-native JAX fitting (hybrid path, 2026-07-07)

Production tempo2 `design_matrix_method="autodiff"` and `residual_delta_jax` use a
**host-frozen** native chain by default:

| Switch | Default | Role |
|--------|---------|------|
| `USE_JAX_TEMPO2_NATIVE_CHAIN` | `True` | Master switch for tempo2-native fitting / JAX residual deltas |
| `USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH` | `False` | Opt-in slow unified in-graph model (`JUG_TEMPO2_NATIVE_FULL_INGRAPH=1`) |

Requirements for MetaPulsar / `export_jax_timing_state`:

1. Call `session.compute_residuals(...)` (or `force_recompute=True` after upgrades) so
   the cache includes `term_diagnostics['tempo2_obs_state']`.
2. `_build_general_fit_setup_from_cache` must pass `term_diagnostics` and `toas` into
   `GeneralFitSetup.native_chain_static`.

IERS preflight: **warn** in general use; **strict fail** under pytest or `JUG_IERS_STRICT=1`.

Fast hybrid regression probes:

```bash
cd ref-packages/jug
JAX_ENABLE_X64=1 PYTHONPATH=.:tests python3 -m pytest \
  tests/test_tempo2_obs_state_export.py \
  tests/test_tempo2_native_staging_host_frozen.py \
  tests/test_tempo2_native_residual_delta_jax.py -q
```

See [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md) for the full parity table.

**Tempo2 parity status (2026-07-06):** Phase C TZR closed J0030 to **~4.7 ns RMS**.
wsrt167 remains **~16.4 ns RMS** (max **~110 ns** at idx 85 — spin-error tail). Phase D Steps 1–3 done/ruled out; Step 4 ruled out Taylor vs tempo2 ``phase2+phase3`` (0.02 ns fractional). Next: clock / ``model_mjd`` vs ``updateBatsAll``.
Details: [`TEMPO2_NATIVE_CLOCK_STATUS.md`](TEMPO2_NATIVE_CLOCK_STATUS.md).

Test-data policy, provenance, and fixture-size guidance live in
[`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md).

### Design-matrix unit convention

`compute_designmatrix()` exports columns as `d(residual)/d(param)` in seconds
per parameter unit using a PINT/Vela-compatible unit vocabulary
(`str(PINT param.units)` style). The returned `DesignMatrixResult.column_units`
are parseable Astropy unit strings.

When comparing against raw Tempo2/libstempo design matrices, apply the explicit
unit translation first (for example RAJ/DECJ are exported as hourangle/deg
convention at the API boundary).

## Hardware requirements
- JUG needs longdouble precision for some of its calculations. For that reason, it must be run on hardware that allows for this. This means that Apple Silicon chips can not run this software without hitting numerical precision errors.

## License

MIT
