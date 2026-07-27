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

### Portable pint-only testing

This branch ships pint-family JAX/autodiff without tempo2 modules. Default CI:

```bash
cd ref-packages/jug-split-main
python -m pytest tests/ -m "not tempo2 and not dev_oracle" -q --no-cov
```

MPTA regression fixtures live under `tests/data_mpta/` (see `manifest.json`).
Golden PINT parity sets live under `tests/data_golden/`. Tests marked `slow` or
`dev_oracle` are excluded from the command above.

### Compatibility mode and residual conventions

This portable build supports a single compatibility family:

- `compatibility="pint"`: PINT-family runtime conventions and weighted residual
  mean subtraction.

> **On "picosecond agreement with PINT":** this holds for *host residuals at
> fixed parameters with identical ephemeris/clock/timescale inputs* (a shared
> phase-precision floor, ~5 ps; paper Fig. 7). It is **not** a claim of absolute
> (vs-nature) accuracy, and it is distinct from the internal JAX-vs-NumPy
> picosecond tests. With unmatched clock/ephemeris files (the CI default) the
> difference is tens of ns, dominated by a DC phase-offset convention plus
> clock-file drift — not a timing-model disagreement. See
> [`PARITY_THEORY.md`](PARITY_THEORY.md) §"What 'picosecond
> compatibility' means (and does not)".

**Nonlinear / autodiff / notebook integrators:** green residual tests on curated
fixtures validate the PINT-family Taylor `forward_delay` path and its JAX
residual-delta graph. The analytic fitter basis `M` (`compute_designmatrix`) and
the residual Jacobian `J` (`FrozenResidualModel.residual_jacobian_native`) are
different objects: `r(theta+delta) ~= r(theta) - M @ delta`, while
`J = jacfwd(residual_delta)(0)`. For notebook workflows that export a frozen
residual model, call `session.compute_residuals(...)` first, then
`export_frozen_residual_model` (Phase B on `tempo2-dev`; residual deltas are
already available via `make_residual_delta_jax_fn` on this branch).
Remaining: `ppta_j1741_ell1` host debt (~5.5 ns), `J0900-3144` TDB probe, model-epoch
IFTE batCorr scalar (~272 ns, pinned), and fuller residual-Jacobian coverage.
Details: [`PARITY_ROADMAP.md`](PARITY_ROADMAP.md).

Test-data policy, provenance, and fixture-size guidance live in
[`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md).

### Design-matrix unit convention

`compute_designmatrix()` returns the raw analytic fitter basis `M` with sign
contract `r(theta+delta) ~= r(theta) - M @ delta`, in seconds per parameter unit
using a PINT/Vela-compatible unit vocabulary (`str(PINT param.units)` style).
The returned `DesignMatrixResult.column_units` are parseable Astropy unit
strings. Residual Jacobians are never returned under a design-matrix name.

When comparing against raw Tempo2/libstempo design matrices, apply the explicit
unit translation first (for example RAJ/DECJ are exported as hourangle/deg
convention at the API boundary).

## Hardware requirements
- JUG needs longdouble precision for some of its calculations. For that reason, it must be run on hardware that allows for this. This means that Apple Silicon chips can not run this software without hitting numerical precision errors.

## License

MIT
