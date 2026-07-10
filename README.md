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
# Use the installed T2 *runtime* tree (contains observatory/, clocks/, etc.),
# not the tempo2 source checkout.
export TEMPO2=/path/to/T2runtime
PYTHONPATH=.:tests TEMPO2=$TEMPO2 \
  pytest tests/test_tempo2_j0613_fast_gates.py \
         tests/test_tempo2_simulated_fixtures.py \
  -q -o addopts='' --no-cov -m 'not slow'
```

The default tempo2 parity fixtures live in `tests/data_tempo2_sim/`: small
libstempo-generated par/tim pairs with 5–12 TOAs. Real excerpts live in
`tests/data_tempo2/` for TIM-format edge cases and historical probes. Its manifest
includes:

- Case A (TCB regression fixtures),
- Case B (NG5 equatorial TDB),
- Case C (NG5 ecliptic cross-engine TDB).

`tests/tempo2_fixtures.py` exposes helpers to select parity fixtures by case
for CI and local debugging.

Avoid using `pytest tests/ -k "tempo2"` as a development loop: it selects hundreds of
oracle-heavy tests and can run for hours. Prefer the simulated fixtures, wsrt167, and
J0613 fast gates, then expand to a named real pulsar or fixture only when needed.
Full wsrt167 (167 TOAs), full J0613/IPTA, and NG5 625-TOA oracle tests are marked
`slow` and excluded by the command above. The tiny J0613 addsat regression is also
currently marked `slow` because its JUG path is several minutes despite only using
11 TOAs.

### Compatibility modes and residual conventions

JUG exposes two compatibility families:

- `compatibility="pint"`: PINT-family runtime conventions and weighted residual
  mean subtraction.
- `compatibility="tempo2"`: tempo2-family runtime conventions and unweighted
  residual mean subtraction.

> **On "picosecond agreement with PINT":** this holds for *host residuals at
> fixed parameters with identical ephemeris/clock/timescale inputs* (a shared
> phase-precision floor, ~5 ps; paper Fig. 7). It is **not** a claim of absolute
> (vs-nature) accuracy, and it is distinct from the internal JAX-vs-NumPy
> picosecond tests. With unmatched clock/ephemeris files (the CI default) the
> difference is tens of ns, dominated by a DC phase-offset convention plus
> clock-file drift — not a timing-model disagreement. See
> [`PARITY_THEORY.md`](PARITY_THEORY.md) §"What 'picosecond
> compatibility' means (and does not)".

When evaluating Tempo2 parity, use **raw pre-fit residuals** only (no post-hoc
mean centering). This is the acceptance metric used by
`tests/test_tempo2_residual_parity.py`.

For notebook diagnostics, weighted-mean-centered deltas are useful only for
PINT-family-vs-PINT-family comparisons; tempo2-labeled comparisons should stay
raw.

**Nonlinear / autodiff / notebook integrators:** green residual tests on curated fixtures
do **not** mean tempo2 mode is ready for JAX-traced likelihoods or IPTA-scale
workloads. See [`PARITY_THEORY.md`](PARITY_THEORY.md) for theory, policy, and JAX-traced
graph guarantees, and [`PARITY_ROADMAP.md`](PARITY_ROADMAP.md) for gap analysis,
pytempo workflow, and usage guidance.

### Tempo2-native JAX fitting (graph modes, 2026-07-07)

**Design matrices:** default WLS uses `design_matrix_method="analytic"` (PINT-style
simplified tangents, fast, independent of graph mode). Set
`design_matrix_method="autodiff"` for native `jacfwd(residual_delta)` through the
tempo2 JAX graph (NUTS / libstempo column parity).

Production tempo2 `residual_delta_jax` and native autodiff always use the tempo2-native
JAX graph. Select the graph with session kwargs:

```python
session = TimingSession(
    par, tim,
    compatibility="tempo2",
    tempo2_native="staged_bclt",  # fixed_state_bclt | fixed_state_stripped | full
    tempo2_jug_options={
        "iers_policy": "warn",       # or "strict"
        "bclt_fixed_iter": 12,
        "force_cache_refresh": False,
        "require_native_cache": True,
    },
)
```

| Mode | `tempo2_native` | Role |
|------|-----------------|------|
| `staged_bclt` | default (omit or explicit) | Freeze host ephemeris/clocks; recompute BCLT scan, formBats, Shklovskii in JAX |
| `fixed_state_bclt` | `"fixed_state_bclt"` | Freeze host state + reference BCLT `dt_ssb`; one-pass BCLT + full tail (envelope reference) |
| `fixed_state_stripped` | `"fixed_state_stripped"` | Same host freeze + BBAT lite kernel (fast NUTS target after validation) |
| `full` | `"full"` | Unified in-graph clocks/SPK/EOP/IFTE/tropo/BCLT (oracle/dev only) |

Requirements for notebook integrators / `export_jax_timing_state`:

1. Call `session.compute_residuals(...)` (or `force_recompute=True` after upgrades) so
   the cache includes `term_diagnostics['tempo2_obs_state']`.
2. `_build_general_fit_setup_from_cache` must pass `term_diagnostics` and `toas` into
   `GeneralFitSetup.native_chain_static`.

IERS preflight: **warn** in general use; **strict fail** under pytest (auto-detected) or
`tempo2_jug_options={"iers_policy": "strict"}`.

Fast hybrid regression probes:

```bash
cd ref-packages/jug
JAX_ENABLE_X64=1 PYTHONPATH=.:tests python3 -m pytest \
  tests/test_tempo2_obs_state_export.py \
  tests/test_tempo2_staging_host_frozen.py \
  tests/test_tempo2_residual_delta_jax.py -q
```

See [`jug/testing/DEV_ORACLE.md`](jug/testing/DEV_ORACLE.md) for the full parity table.

**Tempo2 parity status (2026-07-09):** Host residuals remain sub-ns to low-ns on
gated fixtures (NG5, EPTA J0613 full, wsrt167 TRACK −2). JAX native delay chain on
wsrt167 closes pytempo for formBats delay physics, JAX `bbat_mjd`, stripped lite BBAT,
and `torb_sec` (all < 1 ns RMS). Libstempo autodiff design-matrix gates cover
F0/RAJ/DECJ/DM on wsrt167 (staged/fixed/stripped), binary columns on `epta_j1909_t2`,
and F0 on `epta_j0613_addsat_min`. `full` graph mode has component parity CI (< 1 ns).
Remaining: `ppta_j1741_ell1` host debt (~5.5 ns), `J0900-3144` TDB probe, model-epoch
IFTE batCorr scalar (~272 ns, pinned), and `full` autodiff columns.
Details: [`PARITY_ROADMAP.md`](PARITY_ROADMAP.md).

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
