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

## License

MIT
