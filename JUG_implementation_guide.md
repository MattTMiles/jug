# JUG Implementation Guide

**Version**: 1.0
**Date**: 2025-11-29

This guide helps you convert the JUG notebook into a production-ready Python package. It provides recommendations on whether you should implement yourself or have Claude assist, along with detailed step-by-step instructions for both approaches.

---

## Should You Implement or Should Claude?

### Recommended Division of Labor

| Task | Best Done By | Reasoning |
|------|--------------|-----------|
| **Package structure creation** | You or Claude | Simple file/directory creation; Claude can do this quickly |
| **Code extraction from notebook** | Claude | Tedious, mechanical; Claude won't miss cells or functions |
| **Refactoring into modules** | Claude + Your Review | Claude handles boilerplate, you review for correctness |
| **Writing docstrings** | Claude | Time-consuming, Claude can infer from code |
| **Writing unit tests** | Claude + Your Review | Claude generates templates, you add physics validation |
| **JAX optimization** | You + Claude Support | You understand timing physics, Claude handles JAX syntax |
| **GUI development** | Start with Claude, iterate together | Claude builds skeleton, you refine UX |
| **Debugging** | Collaborative | You know expected behavior, Claude finds code issues |
| **Documentation** | Claude drafts, You edit | Claude writes verbose docs, you trim to essentials |

### Time Estimates

**If You Implement Alone**:
- Milestone 1 (core package): ~2-3 weeks
- Milestone 2 (fitting): ~2-3 weeks
- Milestone 3-4 (noise): ~3-4 weeks
- Milestone 5 (GUI): ~4-6 weeks
- **Total**: ~3-4 months part-time

**If Claude Assists**:
- Milestone 1: ~1 week (Claude does extraction, you review)
- Milestone 2: ~1-2 weeks (Claude writes optimizer, you validate)
- Milestone 3-4: ~2-3 weeks (Claude ports FFT covariance, you test)
- Milestone 5: ~3-4 weeks (Claude builds GUI, you iterate on design)
- **Total**: ~1.5-2 months part-time

**Recommendation**: **Collaborative approach**. Claude handles mechanical tasks (code extraction, boilerplate, docstrings), you focus on physics validation, design decisions, and testing.

---

## Milestone 1: Core Timing Package (v0.1.0)

**Goal**: Extract notebook code into a Python package with modules for I/O, models, delays, and residuals.

### Step 1.1: Create Package Structure

**Task**: Set up directory tree and files.

**Option A: You Do It**

```bash
cd /home/mattm/soft/JUG
mkdir -p jug/io jug/models jug/delays jug/residuals jug/utils jug/tests
touch jug/__init__.py
touch jug/io/__init__.py jug/models/__init__.py jug/delays/__init__.py
touch jug/residuals/__init__.py jug/utils/__init__.py jug/tests/__init__.py
touch pyproject.toml README.md
```

**Option B: Claude Does It**

*Instruction for Claude*:
"Create the package directory structure for JUG as specified in `JUG_package_architecture_flowcharts.md` Section 1. Include all subdirectories and `__init__.py` files. Create empty placeholder files for modules we'll populate later."

**Time**: 5 minutes (you) | 2 minutes (Claude)

---

### Step 1.2: Setup `pyproject.toml`

**Task**: Define package metadata, dependencies, and build configuration.

**Option A: You Do It**

Create `pyproject.toml` with:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "jug-timing"
version = "0.1.0"
description = "JAX-based pulsar timing software"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}  # or your preferred license
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "astropy>=5.3.0",
    "numpyro>=0.13.0",
    "optax>=0.1.7",
]

[project.optional-dependencies]
gui = ["PyQt6>=6.5.0", "pyqtgraph>=0.13.0"]
dev = ["pytest>=7.4.0", "ruff>=0.1.0", "mypy>=1.5.0"]
docs = ["sphinx>=7.0.0", "sphinx-rtd-theme>=1.3.0"]

[project.scripts]
jug-compute-residuals = "jug.scripts.compute_residuals:main"
jug-fit = "jug.scripts.fit:main"
jug-gui = "jug.scripts.gui:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["jug*"]

[tool.pytest.ini_options]
testpaths = ["jug/tests"]
python_files = "test_*.py"

[tool.ruff]
line-length = 100
target-version = "py310"
```

**Option B: Claude Does It**

*Instruction for Claude*:
"Create a `pyproject.toml` file for JUG following modern Python packaging standards. Include dependencies from the design doc: JAX, NumPyro, Optax, Astropy, SciPy. Add optional dependency groups for GUI (PyQt6) and dev (pytest, ruff). Set up entry points for CLI scripts."

**Time**: 15 minutes (you) | 3 minutes (Claude)

---

### Step 1.3: Extract Constants and Utilities

**Task**: Move constants (C, AU, K_DM, L_B) and helper functions to `jug/utils/`.

**Notebook Cells to Extract**:
- Constants: C_M_S, AU_M, SECS_PER_DAY, K_DM_SEC, L_B
- Time conversion functions (if any)
- Coordinate helpers

**Option A: You Do It**

1. Open notebook `residual_maker_playground_active_MK7.ipynb`
2. Find all constant definitions (search for `C_M_S`, `K_DM`, etc.)
3. Copy to `jug/utils/constants.py`:
   ```python
   """Physical and astronomical constants for pulsar timing."""

   C_M_S = 299792458.0  # Speed of light (m/s)
   AU_M = 149597870700.0  # Astronomical unit (m)
   SECS_PER_DAY = 86400.0  # Seconds per day
   K_DM_SEC = 4.148808e3  # DM constant (MHz^2 pc^-1 cm^3 s)
   L_B = 1.550519768e-8  # IAU TCB-TDB scaling factor
   ```
4. Create `jug/utils/time.py` for MJD/JD conversions (if any exist in notebook)

**Option B: Claude Does It**

*Instruction for Claude*:
"Read `residual_maker_playground_active_MK7.ipynb` and extract all physical constants (C, AU, K_DM, L_B, etc.) into `jug/utils/constants.py`. Add docstrings with units. Also extract any time conversion helper functions into `jug/utils/time.py`."

**Time**: 20 minutes (you) | 5 minutes (Claude)

---

### Step 1.4: Extract I/O Functions

**Task**: Move `.par` and `.tim` file parsing functions to `jug/io/`.

**Notebook Cells to Extract**:
- `.par` file parser (likely in cells 1-3)
- `.tim` file parser (FORMAT 1 and fallback)
- Clock file loading
- Observatory data loading

**Option A: You Do It**

1. Find `.par` parsing code in notebook (look for function parsing `F0`, `RA`, `DEC`, etc.)
2. Copy to `jug/io/par_reader.py`:
   ```python
   """Parse Tempo2-style .par files."""
   from pathlib import Path
   from typing import Dict, Any

   def parse_par_file(par_file: str | Path) -> Dict[str, Any]:
       """Parse a .par file and return dictionary of parameters.

       Parameters
       ----------
       par_file : str or Path
           Path to .par file

       Returns
       -------
       dict
           Dictionary mapping parameter names to values
       """
       # Copy parsing logic from notebook here
       pass
   ```
3. Repeat for `tim_reader.py`, `clock.py`, `observatory.py`

**Option B: Claude Does It**

*Instruction for Claude*:
"Extract the `.par` file parsing code from the notebook into `jug/io/par_reader.py`. Create a function `parse_par_file()` that returns a dictionary of parameters. Handle fit flags (the '1' suffix). Add proper type hints and docstrings. Do the same for `.tim` parsing in `jug/io/tim_reader.py`."

**Time**: 1-2 hours (you) | 15 minutes (Claude)

---

### Step 1.5: Extract Timing Models

**Task**: Create timing model classes in `jug/models/`.

**Notebook Cells to Extract**:
- Spin model dataclass or parameters
- DM polynomial model
- Binary model parameters (ELL1)
- Astrometry parameters

**Option A: You Do It**

Create `jug/models/spin.py`:
```python
"""Spin model for pulsar rotation."""
from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class SpinModel:
    """Spin frequency and derivatives.

    Attributes
    ----------
    F0 : float
        Spin frequency at PEPOCH (Hz)
    F1 : float
        First derivative of spin frequency (Hz/s)
    F2 : float
        Second derivative of spin frequency (Hz/s^2)
    PEPOCH : float
        Reference epoch for spin parameters (MJD)
    """
    F0: float
    F1: float = 0.0
    F2: float = 0.0
    PEPOCH: float = 55000.0

    def phase(self, t_mjd: jnp.ndarray) -> jnp.ndarray:
        """Compute rotational phase at given times.

        Parameters
        ----------
        t_mjd : jnp.ndarray
            Times in MJD (TDB)

        Returns
        -------
        jnp.ndarray
            Rotational phase (cycles)
        """
        dt_sec = (t_mjd - self.PEPOCH) * 86400.0
        # Horner's method: phase = F0*t + 0.5*F1*t^2 + (1/6)*F2*t^3
        phase = self.F0 * dt_sec + 0.5 * self.F1 * dt_sec**2 + (1.0/6.0) * self.F2 * dt_sec**3
        return phase
```

Repeat for `dm.py`, `binary_ell1.py`, `astrometry.py`.

**Option B: Claude Does It**

*Instruction for Claude*:
"Extract the timing model dataclass from the notebook (it might be called `SpinDMModel` or similar). Split it into separate model classes in `jug/models/`: `SpinModel` (F0, F1, F2, PEPOCH), `DMModel` (DM, DM1, DM2, DMEPOCH), `ELL1Model` (PB, A1, TASC, EPS1, EPS2, ...). Each should be a dataclass with methods for computing relevant quantities. Register as JAX pytrees where needed."

**Time**: 2-3 hours (you) | 30 minutes (Claude)

---

### Step 1.6: Extract Delay Calculations

**Task**: Move delay computation functions to `jug/delays/`.

**Notebook Cells to Extract**:
- Clock correction function
- Barycentric delay (Roemer + Einstein + Shapiro)
- Binary delay (ELL1 model)
- DM delay
- Combined delay function (JAX-compiled)

**Option A: You Do It**

Create `jug/delays/dm_delay.py`:
```python
"""Dispersion measure delay calculations."""
import jax
import jax.numpy as jnp
from jug.utils.constants import K_DM_SEC

@jax.jit
def dm_delay_sec(dm: float, freq_mhz: jnp.ndarray) -> jnp.ndarray:
    """Compute cold-plasma dispersion delay.

    Parameters
    ----------
    dm : float
        Dispersion measure (pc cm^-3)
    freq_mhz : jnp.ndarray
        Observing frequencies (MHz)

    Returns
    -------
    jnp.ndarray
        Delay in seconds (positive = signal arrives later)
    """
    return K_DM_SEC * dm / freq_mhz**2
```

Repeat for `barycentric.py`, `binary_delay.py`, `clock_correction.py`, `combined.py`.

**Option B: Claude Does It**

*Instruction for Claude*:
"Extract all delay computation functions from the notebook into `jug/delays/`. Create separate modules for each delay type: `clock_correction.py`, `barycentric.py`, `binary_delay.py`, `dm_delay.py`, `shapiro_planets.py`, `combined.py`. Ensure all functions use `@jax.jit` decorators where appropriate. Keep the JAX-compiled combined delay function intact."

**Time**: 2-3 hours (you) | 30 minutes (Claude)

---

### Step 1.7: Extract Residual Calculation

**Task**: Move residual computation to `jug/residuals/`.

**Notebook Cells to Extract**:
- Main residual calculation function
- Spin phase calculation (Horner's method)

**Option A: You Do It**

Create `jug/residuals/compute.py`:
```python
"""Core residual computation functions."""
import jax
import jax.numpy as jnp
from jug.models import SpinModel
from jug.delays import combined_delays

@jax.jit
def compute_residuals(
    t_toa_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    model: SpinModel,
    # ... other parameters
) -> jnp.ndarray:
    """Compute timing residuals.

    Parameters
    ----------
    t_toa_mjd : jnp.ndarray
        TOA times (MJD, topocentric)
    freq_mhz : jnp.ndarray
        Observing frequencies (MHz)
    model : SpinModel
        Timing model parameters

    Returns
    -------
    jnp.ndarray
        Residuals in seconds
    """
    # Apply delays
    total_delay = combined_delays(t_toa_mjd, freq_mhz, model)

    # Emission times
    t_emission = t_toa_mjd - total_delay

    # Compute phase
    phase = model.phase(t_emission)

    # Residuals (phase → time)
    phase_residual = phase - jnp.round(phase)
    time_residual = phase_residual / model.F0

    return time_residual
```

**Option B: Claude Does It**

*Instruction for Claude*:
"Extract the main residual calculation logic from the notebook into `jug/residuals/compute.py`. Create a JAX-JIT-compiled function `compute_residuals()` that takes TOAs, frequencies, and timing model, and returns residuals. Ensure it uses the combined delay function and spin phase calculation. Add comprehensive docstrings."

**Time**: 1 hour (you) | 15 minutes (Claude)

---

### Step 1.8: Write Unit Tests

**Task**: Create tests for all modules, focusing on delay calculations (critical for accuracy).

**Option A: You Do It**

Create `jug/tests/test_delays/test_dm_delay.py`:
```python
"""Tests for DM delay calculation."""
import pytest
import jax.numpy as jnp
from jug.delays.dm_delay import dm_delay_sec
from jug.utils.constants import K_DM_SEC

def test_dm_delay_known_values():
    """Test DM delay with known values."""
    dm = 30.0  # pc cm^-3
    freq_mhz = jnp.array([1400.0])

    expected_delay = K_DM_SEC * dm / freq_mhz**2
    computed_delay = dm_delay_sec(dm, freq_mhz)

    assert jnp.allclose(computed_delay, expected_delay, rtol=1e-10)

def test_dm_delay_frequency_scaling():
    """Test that DM delay scales as freq^-2."""
    dm = 30.0
    freq1 = jnp.array([1400.0])
    freq2 = jnp.array([700.0])  # Half the frequency

    delay1 = dm_delay_sec(dm, freq1)
    delay2 = dm_delay_sec(dm, freq2)

    # Delay at half frequency should be 4x larger
    assert jnp.allclose(delay2 / delay1, 4.0, rtol=1e-10)
```

Repeat for all delay functions, models, I/O functions.

**Option B: Claude Does It**

*Instruction for Claude*:
"Write comprehensive unit tests for all delay calculation functions in `jug/tests/test_delays/`. Test: (1) Known values (compare to Tempo2/hand calculations), (2) Correct scaling (e.g., DM delay ∝ freq^-2), (3) Edge cases (zero DM, high eccentricity binary). Aim for 100% coverage of `jug/delays/`. Use pytest fixtures for test data."

**Time**: 3-4 hours (you) | 1 hour (Claude)

---

### Step 1.9: Create CLI Script

**Task**: Create command-line script to compute residuals.

**Option A: You Do It**

Create `jug/scripts/compute_residuals.py`:
```python
"""CLI script to compute pulsar timing residuals."""
import argparse
from pathlib import Path
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file
from jug.residuals.compute import compute_residuals

def main():
    parser = argparse.ArgumentParser(description="Compute pulsar timing residuals")
    parser.add_argument("par_file", help=".par file with timing model")
    parser.add_argument("tim_file", help=".tim file with TOAs")
    parser.add_argument("--output", "-o", help="Output CSV file", default="residuals.csv")

    args = parser.parse_args()

    # Load data
    model = parse_par_file(args.par_file)
    toas = parse_tim_file(args.tim_file)

    # Compute residuals
    residuals = compute_residuals(toas['mjd'], toas['freq'], model)

    # Save output
    # ... (write CSV)

    print(f"Residuals saved to {args.output}")

if __name__ == "__main__":
    main()
```

**Option B: Claude Does It**

*Instruction for Claude*:
"Create a CLI script `jug/scripts/compute_residuals.py` that loads a `.par` and `.tim` file, computes residuals, and outputs to CSV. Use argparse for command-line arguments. Add `--output` flag for output file. Register this as `jug-compute-residuals` entry point in `pyproject.toml`."

**Time**: 30 minutes (you) | 10 minutes (Claude)

---

### Step 1.10: Validate Against Notebook

**Task**: Run the package version and compare to notebook output.

**Option A: You Do It**

```bash
# Install package in editable mode
pip install -e .

# Run on same data as notebook
jug-compute-residuals J0437-4715.par J0437-4715.tim --output residuals_package.csv

# Compare to notebook output
python -c "
import numpy as np
res_notebook = np.loadtxt('residuals_notebook.csv')  # from notebook
res_package = np.loadtxt('residuals_package.csv')
diff = res_notebook - res_package
print(f'Max difference: {np.max(np.abs(diff))*1e6:.3f} ns')
print(f'RMS difference: {np.std(diff)*1e6:.3f} ns')
"
```

**Option B: Claude Does It**

*Instruction for Claude*:
"After extracting all code, run the package CLI on the same test data used in the notebook. Compare residuals output to the notebook's computed residuals. Report RMS difference (should be <1 ns). Debug any discrepancies."

**Time**: 1 hour (you) | 30 minutes (Claude, if errors occur)

---

### Milestone 1 Summary

**Deliverables**:
- [x] Package structure with `jug/io/`, `jug/models/`, `jug/delays/`, `jug/residuals/`
- [x] Parsed `.par` and `.tim` files
- [x] JAX-compiled residual computation
- [x] Unit tests for delay calculations (>90% coverage)
- [x] CLI script `jug-compute-residuals`
- [x] Validation: Package matches notebook within 1 ns RMS

**Time Estimate**:
- You alone: 2-3 weeks
- Claude assists: 1 week (2-3 days active work, rest is review/validation)

**Recommendation**: Have Claude do steps 1.2-1.7 (extraction, boilerplate), you do 1.8-1.10 (testing, validation). Claude handles mechanical work, you ensure correctness.

---

## Milestone 2: Gradient-Based Fitting (v0.2.0)

**Goal**: Implement Gauss-Newton least squares fitting with analytical Jacobian for timing model parameters.

**Status**: 85% complete (as of Session 7, 2025-11-30)

**Key Decision**: After benchmarking multiple optimizers (scipy, JAX autodiff, Adam/Optax), we chose **Gauss-Newton with analytical derivatives** over NumPyro/SVI:
- 10-100x faster than gradient descent methods
- Analytical Jacobian more accurate than autodiff for this problem
- Standard least-squares formulation (weighted chi-squared)
- Levenberg-Marquardt damping for robustness

### Step 2.1: Research and Benchmark Optimizers ✅ COMPLETED

**Task**: Compare optimization methods for pulsar timing.

**What We Did**:
1. Benchmarked scipy.optimize methods (Levenberg-Marquardt, trust-region, BFGS)
2. Tested JAX autodiff + scipy optimizers
3. Tested Adam/Optax (gradient descent) - 10,000x slower!
4. Created comprehensive comparison documents

**Deliverables**:
- `OPTIMIZER_COMPARISON.md` - Detailed benchmark results
- `JAX_ACCELERATION_ANALYSIS.md` - Performance analysis
- Decision: Gauss-Newton with analytical Jacobian

**Time**: 1.5 hours (Session 4)

---

### Step 2.2: Implement Analytical Design Matrix ✅ COMPLETED

**Task**: Compute analytical derivatives (Jacobian) for timing model parameters.

**What We Did**:
Create `jug/fitting/design_matrix.py` (260 lines):
```python
"""Analytical design matrix for pulsar timing."""
import numpy as np

def compute_design_matrix(residuals_fn, params, param_names, toas, freqs):
    """Compute analytical Jacobian (design matrix).
    
    Parameters
    ----------
    residuals_fn : callable
        Function to compute residuals
    params : dict
        Parameter values
    param_names : list
        Parameters to fit
    toas : np.ndarray
        Times of arrival
    freqs : np.ndarray
        Observing frequencies
        
    Returns
    -------
    M : np.ndarray
        Design matrix, shape (n_toas, n_params)
    """
    # Analytical derivatives for each parameter
    # ∂residual/∂F0, ∂residual/∂F1, etc.
    ...
```

**Implemented Derivatives**:
- ✅ F0, F1, F2, F3 (spin frequency and derivatives)
- ✅ DM, DM1, DM2 (dispersion measure)
- ⏳ Binary parameters (PB, A1, TASC, EPS1, EPS2) - deferred to M2.7
- ⏳ Astrometry (RAJ, DECJ, PMRA, PMDEC, PX) - deferred to M2.8

**Time**: 1.5 hours (Session 4)

---

### Step 2.3: Implement Gauss-Newton Solver ✅ COMPLETED

**Task**: Implement Gauss-Newton least squares with Levenberg-Marquardt damping.

**What We Did**:
Create `jug/fitting/gauss_newton.py` (240 lines):
```python
"""Gauss-Newton least squares solver."""
import numpy as np

def gauss_newton_fit(residuals_fn, params, param_names, toas, freqs, errors,
                     max_iter=20, lambda_init=1e-3, convergence_threshold=1e-6):
    """Fit timing model using Gauss-Newton with Levenberg-Marquardt.
    
    Algorithm:
    1. Compute residuals: r = residuals_fn(params)
    2. Compute design matrix: M = ∂r/∂p (analytical)
    3. Solve: (M^T W M + λI) Δp = M^T W r
    4. Update: p_new = p_old - Δp
    5. If chi2 decreases: accept, reduce λ
       If chi2 increases: reject, increase λ
    6. Check convergence: |Δchi2| < threshold and |Δp| < threshold
    
    Returns
    -------
    fitted_params : dict
        Best-fit parameter values
    uncertainties : dict
        1-sigma uncertainties from covariance matrix
    info : dict
        Fitting statistics (chi2, iterations, convergence)
    """
    ...
```

**Features**:
- Levenberg-Marquardt damping for stability
- Trust-region-style step acceptance/rejection
- Convergence checking (chi2 change + parameter change)
- Covariance matrix → parameter uncertainties
- Progress reporting

**Time**: 1 hour (Session 4)

---

### Step 2.4: JAX Acceleration ⏳ IN PROGRESS (NEXT TASK)

**Task**: Port design matrix and Gauss-Newton to JAX for 10-60x speedup.

**Plan**:
1. Create `jug/fitting/design_matrix_jax.py`
   - Use `jax.jacfwd()` or `jax.jacrev()` for automatic differentiation
   - Or keep analytical derivatives, wrap in JAX
   
2. Create `jug/fitting/gauss_newton_jax.py`
   - JIT-compile matrix operations: `M^T W M`, `M^T W r`
   - Use `jax.scipy.linalg.solve` for linear system
   
3. Add hybrid backend selection:
   - NumPy for <500 TOAs (setup overhead too high)
   - JAX for ≥500 TOAs (amortizes compilation cost)

**Expected Speedup**: 10-60x for large datasets

**Assigned to**: Claude (next task)
**Estimated time**: 1-2 hours

---

### Step 2.5: Integration with Real Residuals ⏳ TO DO

**Task**: Refactor `simple_calculator.py` for fitting use.

**Plan**:
1. Separate setup (load data, compute ephemeris) from residual computation
2. Setup runs once, residual computation called many times during fitting
3. Test on J1909-3744 (clean ELL1 binary pulsar)
4. Compare fitted parameters with PINT

**Example**:
```python
# Setup (once)
setup_data = prepare_for_fitting(par_file, tim_file)

# Residual function (called many times)
def residuals_fn(params):
    return compute_residuals_simple(params, setup_data)

# Fit
fitted_params, uncertainties = gauss_newton_fit(
    residuals_fn, initial_params, ...
)
```

**Assigned to**: Claude
**Estimated time**: 1 hour

---

### Step 2.6: Fitting CLI Script ⏳ TO DO

**Task**: Create `jug-fit` command-line tool.

    Returns
    -------
    dict
        Parameter uncertainties (1-sigma)
    """
    # Compute Hessian using JAX autodiff
    hessian_fn = jax.hessian(loss_fn)
    hess = hessian_fn(params)

    # Fisher matrix = -Hessian of log-likelihood
    fisher = -hess

    # Covariance = inverse of Fisher matrix
    cov = jnp.linalg.inv(fisher)

    # Uncertainties = sqrt of diagonal
    uncertainties = jnp.sqrt(jnp.diag(cov))

    return uncertainties
```

**Option B: Claude Does It**

*Instruction for Claude*:
"Implement Fisher matrix uncertainty estimation in `jug/fitting/fisher.py`. Use `jax.hessian()` to compute second derivatives of the log-likelihood at the MAP estimate. Invert the Hessian to get covariance matrix, take sqrt of diagonal for 1-sigma uncertainties. Handle potential singular matrices gracefully."

**Time**: 1-2 hours (you) | 30 minutes (Claude)

---

### Step 2.5: Validate Against Tempo2/PINT

**Task**: Fit a test pulsar and compare parameter values to Tempo2/PINT.

**Option A: You Do It**

```bash
# Fit with JUG
jug-fit J0437-4715.par J0437-4715.tim --output fitted_jug.par

# Fit with PINT (for comparison)
# (assuming PINT is installed)
# ... run PINT fit ...

# Compare parameters
python compare_fits.py fitted_jug.par fitted_pint.par
```

**Option B: Claude Does It**

*Instruction for Claude*:
"After implementing the fitting module, run a fit on J0437-4715 test data. Compare fitted parameters (F0, F1, RA, DEC, PB, A1, etc.) to the values from a PINT fit (if available) or Tempo2 fit. Report differences in units of 1-sigma uncertainties. Debug if discrepancies >3-sigma."

**Time**: 2 hours (you) | 1 hour (Claude)

---

### Milestone 2 Summary (Updated 2025-11-30)

**Progress**: 85% complete (Sessions 4-7)

**Deliverables Completed** ✅:
- [x] Optimizer research and benchmarking → Chose **Gauss-Newton** over NumPyro/SVI
- [x] `jug/fitting/design_matrix.py` with analytical derivatives (F0-F3, DM, DM1, DM2)
- [x] `jug/fitting/gauss_newton.py` with Levenberg-Marquardt damping
- [x] Synthetic data validation - successfully recovers parameters
- [x] Multi-pulsar testing framework (`test_binary_models.py`)
- [x] Binary model expansion (DD, DDH, BT, T2 implemented and validated)
- [x] Performance audit (`PERFORMANCE_OPTIMIZATION_AUDIT.md`)
- [x] Fixed BT vectorization (10-100x speedup)

**Remaining Work** ⏳:
- [ ] JAX acceleration (design_matrix_jax.py, gauss_newton_jax.py) - **NEXT**
- [ ] Integration with real residuals computation
- [ ] CLI script `jug-fit`
- [ ] Binary parameter derivatives (deferred)
- [ ] Astrometry derivatives (deferred)

**Key Decision**: Gauss-Newton with analytical Jacobian is 10-100x faster than gradient descent methods (Adam, SGD) for this least-squares problem structure.

**Time Estimate**:
- Completed so far: ~8 hours (Sessions 4-7)
- Remaining: 2-3 hours
- **Total for M2**: ~10-11 hours

**Documentation Created**:
- `OPTIMIZER_COMPARISON.md` - Why Gauss-Newton over SVI
- `JAX_ACCELERATION_ANALYSIS.md` - Performance benchmarks
- `BINARY_MODEL_INTEGRATION_STATUS.md` - Binary model validation
- `PERFORMANCE_OPTIMIZATION_AUDIT.md` - Code optimization status

---

## Milestone 3-4: Noise Models (v0.3.0 - v0.4.0)

**Goal**: Implement white noise (EFAC/EQUAD/ECORR) and GP noise (FFT covariance).

### Step 3.1: Port FFT Covariance from Discovery

**Task**: Adapt the FFT covariance method from `/home/mattm/soft/discovery/src/discovery/signals.py`.

**Option A: You Do It**

1. Study `discovery/src/discovery/signals.py`, specifically:
   - `psd2cov()` function (lines ~864-890)
   - `makegp_fftcov()` (line 899)
   - `makegp_fftcov_dm()` (line 905)
   - `makegp_fftcov_chrom()` (line 910)
2. Extract core PSD→covariance conversion logic
3. Create `jug/noise/fft_covariance.py`
4. Adapt to work with JUG's JAX framework

**Option B: Claude Does It**

*Instruction for Claude*:
"Port the FFT covariance method from `/home/mattm/soft/discovery/src/discovery/signals.py` to `jug/noise/fft_covariance.py`. Extract `psd2cov()`, `makegp_fftcov()`, `makegp_fftcov_dm()`, and `makegp_fftcov_chrom()`. Ensure compatibility with JUG's JAX/NumPyro framework. Add docstrings explaining the math (PSD → time-domain covariance via inverse FFT)."

**Time**: 4-6 hours (you, understanding math + code) | 1-2 hours (Claude, direct port + testing)

---

### Step 3.2: Implement White Noise Models

**Task**: Create EFAC, EQUAD, ECORR classes.

**Recommendation**: **Claude handles this** (straightforward implementation).

*Instruction for Claude*:
"Create white noise models in `jug/noise/white.py`: EFAC (multiplicative error scaling), EQUAD (additive white noise), ECORR (epoch-correlated noise). Each should be a class with methods to modify TOA covariance matrix. Use JAX arrays for compatibility with JIT compilation."

**Time**: You: 2 hours | Claude: 30 minutes

---

### Step 3.3: Implement GP Noise Classes

**Task**: Create RedNoise, DMNoise, ChromaticNoise classes using FFT covariance.

**Recommendation**: **Collaborative** (Claude writes class structure, you validate physics).

*Instruction for Claude*:
"Create GP noise classes in `jug/noise/`: `RedNoise` (achromatic power-law), `DMNoise` (chromatic ∝ freq^-2), `ChromaticNoise` (scattering ∝ freq^-4). Each should use the FFT covariance framework from step 3.1. Add PSD functions: `powerlaw()` for red noise, etc. Ensure parameters (log10_A, gamma) are compatible with PTA conventions."

**Time**: You: 3-4 hours | Claude: 1-2 hours

---

### Step 3.4: Test Noise Models

**Task**: Write unit tests for noise likelihood evaluation.

**Recommendation**: **You write physics tests, Claude writes code coverage tests**.

*Instruction for you*:
- Test that red noise reduces χ² for simulated data with injected red noise
- Test that DM noise scales correctly with frequency
- Validate GP covariance matrix is positive definite

*Instruction for Claude*:
"Write unit tests for all noise model classes in `jug/tests/test_noise/`. Test: (1) Correct covariance matrix shape, (2) Positive definite matrices, (3) Likelihood computation doesn't NaN/Inf. Use synthetic residuals with known noise properties."

**Time**: You: 2-3 hours | Claude: 1 hour

---

### Milestone 3-4 Summary

**Deliverables**:
- [x] `jug/noise/white.py`: EFAC, EQUAD, ECORR
- [x] `jug/noise/fft_covariance.py`: PSD → covariance conversion
- [x] `jug/noise/gp.py`: RedNoise, DMNoise, ChromaticNoise
- [x] Tests for noise model likelihood

**Time Estimate**:
- You alone: 3-4 weeks
- Claude assists: 2-3 weeks

---

## Milestone 5: Desktop GUI (v0.5.0)

**Goal**: Build PyQt6 GUI with residual plot, parameter panel, and fit control.

### Recommendation: Start with Claude, Iterate Together

**GUI development is iterative**. Claude can quickly build a functional skeleton, then you refine the UX.

### Step 5.1: GUI Skeleton

*Instruction for Claude*:
"Create a PyQt6 GUI in `jug/gui/main_window.py`. Layout: (1) Top-left: Residual plot (pyqtgraph scatter plot), (2) Top-right: Parameter table (QTableWidget), (3) Bottom-left: Placeholder for noise diagnostics, (4) Bottom-right: Fit control buttons. Add file menu: Open .par, Open .tim, Save .par. Make it runnable via `jug-gui` CLI script."

**Claude builds**: Basic window, layout, widgets, file dialogs.
**You refine**: Colors, fonts, spacing, responsiveness.

**Time**: Claude 2-3 hours (initial build) + You 2-3 hours (refinement) = ~1 day

---

### Step 5.2: Real-Time Parameter Updates

*Instruction for Claude*:
"Add real-time residual updates to the GUI. When user edits a parameter value in the table, recompute residuals using `jug.residuals.compute()` and update the plot. Use Qt signals/slots to connect parameter changes to plot updates. Debounce updates (wait 300ms after last edit before recomputing)."

**You test**: Does it feel responsive? Is 300ms lag acceptable, or should it be instant?

**Time**: Claude 2 hours + You 1 hour testing = ~3 hours

---

### Step 5.3: Interactive Flagging

*Instruction for Claude*:
"Add click-to-flag functionality. When user clicks a TOA point in the residual plot, toggle its flag status (flagged = grayed out or red X). Store flags in a list. Add buttons: 'Flag Selected', 'Unflag Selected', 'Unflag All'. When saving .tim file, write `FLAG -toa` for flagged TOAs."

**You test**: Does click detection work reliably? Are visual indicators clear?

**Time**: Claude 2 hours + You 1 hour = ~3 hours

---

### Step 5.4: Fit Integration

*Instruction for Claude*:
"Add 'Fit Selected' button. When clicked, read parameter fit flags from table, call `jug.fitting.optimizer.run_fit()`, update parameter values and uncertainties in table, recompute residuals. Show progress bar during fit (update every batch). Add 'Reset to Initial' and 'Undo Last Fit' buttons."

**You test**: Does fit converge? Are uncertainties displayed correctly?

**Time**: Claude 3 hours + You 2 hours = ~5 hours

---

### Step 5.5: Noise Diagnostics Panel

*Instruction for Claude*:
"Add noise diagnostic plots in bottom-left panel: (1) Power spectrum (periodogram of residuals vs. fitted noise model), (2) ACF plot, (3) Residual histogram with Gaussian overlay. Use pyqtgraph for fast rendering. Add tabs to switch between plots."

**You refine**: Plot aesthetics, axis labels, legend positioning.

**Time**: Claude 3 hours + You 2 hours = ~5 hours

---

### GUI Iteration Strategy

1. **Claude builds feature skeleton** (functional but not polished)
2. **You test and provide feedback** ("button is too small", "plot needs grid lines")
3. **Claude refines** based on feedback
4. **Repeat** until UX feels good

**Estimated Total Time for Milestone 5**:
- Claude active coding: ~15-20 hours
- Your testing/refinement: ~10-15 hours
- **Total**: 3-4 weeks (part-time, iterative)

---

## Summary: Recommended Workflow

### Phase 1: Foundation (Milestones 1-2)

**Week 1-2**: Claude extracts notebook code into package structure
- You review each module as Claude completes it
- You write physics validation tests
- End of Week 2: Package can compute residuals, matches notebook

**Week 3-4**: Claude implements fitting, you validate
- Claude ports optimizer from reference repo
- You test on real pulsars, compare to Tempo2/PINT
- End of Week 4: Package can fit timing models

### Phase 2: Noise Models (Milestones 3-4)

**Week 5-7**: Claude ports FFT covariance, you validate physics
- Claude adapts discovery package code
- You ensure GP likelihoods are correct (inject synthetic noise, recover)
- End of Week 7: Package can fit timing + noise models

### Phase 3: GUI (Milestone 5)

**Week 8-11**: Iterative GUI development
- Claude builds feature, you test, Claude refines
- Rapid iteration cycles (1-2 features per week)
- End of Week 11: Functional GUI with core features

### Phase 4: Polish (Milestones 6-10)

**Week 12+**: Priors, advanced models, performance optimization, documentation
- Mix of Claude automation (docs, boilerplate) and your expertise (design decisions)
- Flexible based on priorities

---

## How to Work with Claude

### Effective Instructions

**Good**:
> "Extract the DM delay calculation from cell 8 of the notebook into `jug/delays/dm_delay.py`. The function should be JAX-JIT-compiled, take `dm` (float) and `freq_mhz` (jnp.ndarray) as inputs, and return delay in seconds. Add a docstring with parameter descriptions and units. Write a unit test in `jug/tests/test_delays/test_dm_delay.py` that checks the freq^-2 scaling."

**Bad**:
> "Move the DM stuff to a new file."

### Review Checklist

After Claude implements a module, check:
- [ ] **Correctness**: Does math match notebook? Run test cases.
- [ ] **Performance**: Are JAX decorators (`@jax.jit`) in place?
- [ ] **Documentation**: Are docstrings clear and complete?
- [ ] **Tests**: Do tests cover edge cases? (Not just happy path)
- [ ] **Code style**: Is it readable? (Claude usually writes clean code, but check)

### Iterative Refinement

If Claude's first attempt isn't perfect:
1. **Be specific about what's wrong**: "The function returns NaN when freq=0, add a check"
2. **Provide expected behavior**: "Should raise ValueError if freq <= 0"
3. **Give examples**: "For DM=30, freq=1400, expected delay is 2.1e-6 seconds"

Claude learns from feedback and improves on subsequent iterations.

---

## Conclusion

**Recommended Approach**: **Collaborative Development**

- **Claude handles**: Code extraction, boilerplate, docstrings, test templates, GUI skeleton
- **You handle**: Physics validation, design decisions, UX refinement, final testing

**Why this works**:
- Claude is fast at mechanical tasks (extracting 50 functions from a notebook)
- You ensure correctness (does the binary delay match Tempo2?)
- Together: Faster than you alone, more accurate than Claude alone

**Estimated Timeline**:
- Milestones 1-2 (core + fitting): ~4 weeks
- Milestones 3-4 (noise): ~3 weeks
- Milestone 5 (GUI): ~4 weeks
- **Total to functional v0.5**: ~11 weeks (~3 months part-time)
- **Total to polished v1.0**: ~4-6 months (including advanced features, docs, benchmarks)

**Next Step**: Decide on Milestone 1 approach. If you want Claude to proceed, say:
> "Claude, start Milestone 1. Create the package structure, extract code from the notebook into modules, and write unit tests for delay calculations. I'll review each module as you complete it."

If you prefer to start yourself:
> "I'll handle Milestone 1. I'll come back to you when I need help with Milestone 2 (fitting)."

**Either way, you'll have a production-ready pulsar timing package faster than solo development!**
