# JUG Master Design Philosophy

**Version**: 1.0
**Date**: 2025-11-29
**Status**: Living Document

---

## Mission Statement

JUG (JAX-based pulsar timing) is an independent, high-performance pulsar timing software that prioritizes **speed without sacrificing accuracy**. Built from the ground up using JAX for automatic differentiation and JIT compilation, JUG provides a complete timing pipeline with gradient-based fitting and modern noise modeling capabilities.

### Core Principles

1. **Speed First**: Performance is paramount. Every design decision must consider computational efficiency.
2. **JAX-Native**: Leverage JAX's JIT compilation, automatic differentiation, and vectorization throughout.
3. **Independence**: No dependencies on PINT or Tempo2 for core functionality (comparisons only).
4. **Accuracy**: Microsecond-level precision matching or exceeding Tempo2/PINT standards.
5. **Extensibility**: Easy for users to add custom timing models and noise processes.
6. **Modern UX**: Desktop GUI with real-time parameter updates and interactive diagnostics.

---

## I. Timing Model Coverage

### Phase 1: Core Models (PRIORITY - Current Focus)

**Spin Model**:
- F0, F1, F2, ..., F_N (arbitrary order frequency derivatives)
- PEPOCH (reference epoch for spin parameters)
- Proper handling of TCB/TDB units conversion

**Astrometry**:
- RA, DEC (J2000 coordinates)
- PMRA, PMDEC (proper motion)
- PX (parallax)
- POSEPOCH (reference epoch for astrometric parameters)

**Dispersion Measure (DM)**:
- DM, DM1, DM2, ..., DM_N (polynomial DM model)
- DMEPOCH (reference epoch for DM parameters)
- Frequency-dependent DM delay: K_DM * DM / freq^2

**Binary Models**:
- **ELL1/ELL1H**: Low-eccentricity orbits (PB, A1, TASC, EPS1, EPS2, GAMMA, PBDOT, XDOT, H3, H4)
- **BT**: Basic binary model (PB, A1, ECC, OM, T0)
- **DD/DDH/DDGR**: Relativistic binary models (includes Shapiro delay, aberration)
- **T2**: Tempo2 binary model extensions
- All models include: Roemer delay, Einstein delay, Shapiro delay from companion

### Phase 2: Advanced Models (PRIORITY)

**Glitches**:
- GLEP_N (glitch epoch)
- GLPH_N (phase jump)
- GLF0_N, GLF1_N, GLF2_N (permanent frequency changes)
- GLTD_N, GLF0D_N (exponential recovery parameters)

**Frequency-Dependent (FD) Parameters**:
- FD1, FD2, ..., FD_N (profile evolution with frequency)
- Handles frequency-dependent arrival time corrections

**Higher-Order Binary Effects**:
- Shapiro delay (SINI, M2 or SINI, H3/H4 parametrizations)
- Aberration delays
- Orbital period derivatives (PBDOT, OMDOT, XDOT, EDOT)
- Third-body effects (FB0, FB1, ...)

**Solar System Effects**:
- Solar wind dispersion (NE_SW parameter)
- Planet Shapiro delays (Jupiter, Saturn, Uranus, Neptune, Venus)

### Phase 3: Exotic Models (FUTURE - Lower Priority)

**Note**: Many Phase 3 effects are better handled via noise parameters rather than deterministic models.

- Wideband DM (simultaneous TOA + DM measurements)
- Chromatic timing (scattering, profile evolution)
- Post-Keplerian parameters for binary tests of GR
- Gravitational wave modeling (continuous waves, stochastic background)

---

## II. Noise Model Architecture

### Design Philosophy

Noise modeling in JUG follows these principles:
1. **Fourier-domain representation** for computational efficiency
2. **FFT covariance method** (inspired by `discovery` package) for fast GP likelihood evaluation
3. **Extensible framework** allowing users to add custom noise processes
4. **JAX-compatible** for gradient-based optimization

### Phase 1: White Noise (PRIORITY)

**EFAC**: Multiplicative error scaling per backend/receiver
- Format: `EFAC {backend_flag} {value}`
- Applied as: `sigma_eff = EFAC * sigma_toa`

**EQUAD**: Additive white noise per backend/receiver
- Format: `EQUAD {backend_flag} {value}`
- Applied as: `sigma_eff^2 = (EFAC * sigma_toa)^2 + EQUAD^2`

**ECORR**: Epoch-correlated white noise (same-epoch TOAs correlated)
- Format: `ECORR {backend_flag} {value}`
- Block-diagonal covariance for simultaneous TOAs

### Phase 2: Fourier-Domain Gaussian Processes (PRIORITY)

**Achromatic Red Noise**:
- Power-law spectrum: P(f) = A^2 / (12π^2) * (f/f_yr)^(-γ) * f_yr^(γ-3)
- Parameters: log10_A (amplitude), gamma (spectral index)
- Components: Sine/cosine Fourier basis over observing span
- Implementation: `makegp_fftcov()` from discovery framework

**Dispersion Measure Variations**:
- Power-law chromatic noise: DM(t) modeled as GP
- Frequency scaling: ∝ freq^(-2)
- Parameters: log10_A_dm, gamma_dm
- Components: Fourier basis with DM time interpolation
- Implementation: `makegp_fftcov_dm()` pattern

**Chromatic Scattering Noise**:
- Scattering timescale variations: τ(t) as GP
- Frequency scaling: ∝ freq^(-4) (or custom index)
- Parameters: log10_A_scat, gamma_scat, alpha_scat
- Components: Fourier basis with chromatic interpolation
- Implementation: `makegp_fftcov_chrom()` pattern

**Band-Specific Noise**:
- Per-frequency-band correlated noise (e.g., L-band vs S-band)
- Allows for receiver-specific systematic effects
- Implementation: `makegp_fftcov_band()` pattern

### FFT Covariance Method Details

**Key Concept**: Represent GP covariance in Fourier domain, use FFT for O(N log N) likelihood evaluation instead of O(N^3) matrix inversion.

**Implementation Steps**:
1. Define power spectral density (PSD) function: `P(f) = prior(f, θ)`
2. Convert PSD to time-domain covariance via inverse FFT with oversampling
3. Interpolate covariance to TOA times using time-interpolation basis
4. Construct design matrix: `F @ diag(phi) @ F.T` where F is Fourier basis
5. Likelihood: `log p(r|θ) = -0.5 * r.T @ (N + F @ diag(phi) @ F.T)^(-1) @ r`

**Advantages**:
- Fast: O(N log N) instead of O(N^3)
- Numerically stable for long timespans
- Easy to add new noise processes (just define new PSD)

**Reference**: See `/home/mattm/soft/discovery/src/discovery/signals.py` for implementation patterns.

### Phase 3: Additional Noise Models (FUTURE)

- **System noise**: Per-telescope systematic noise models
- **Scintillation**: Short-timescale stochastic variations
- **Profile evolution noise**: Frequency-dependent profile changes as GP
- **Solar wind noise**: Time-variable solar wind dispersion

### Extensibility Requirements

Users should be able to add custom noise processes by:
1. Defining a PSD function: `custom_psd(f, df, *params) -> phi`
2. Specifying frequency scaling (achromatic, chromatic, custom)
3. Registering the noise component with JUG's noise framework
4. JAX-compatible: auto-differentiation for gradient-based fitting

---

## III. Fitting and Optimization

### Gradient-Based Optimization

**Primary Method**: Use JAX automatic differentiation with NumPyro/Optax optimizers.

**Reference Implementation**: `/home/mattm/soft/pulsar-map-noise-estimates`

**Optimizer**: AdamW with warmup cosine decay learning rate
- Warmup steps: 500 (configurable)
- Peak learning rate: 0.01 (configurable)
- Decay to 1% of peak over max_epochs
- Gradient clipping: Optional, default disabled

**Training Loop**:
- Batch optimization using `jax.lax.scan` for speed
- Early stopping: Monitor loss, halt if no improvement for N batches
- Loss improvement threshold: Configurable (default: 1.0)
- Patience: 3 batches without improvement
- Max batches: 50 (configurable)

**Advantages Over Traditional Methods**:
- **Speed**: Gradient-based >> Powell/Nelder-Mead for high-dimensional problems
- **Scalability**: Handles 100+ parameters efficiently
- **JAX Integration**: Seamless with JIT-compiled residuals
- **Flexibility**: Easy to add priors, constraints

### Incremental Fitting (Tempo2/PINT-Style)

**Parameter Freezing**:
- `.par` file convention: `1` suffix indicates parameter is being fit
- Example: `F0 100.5 1` (fit F0), `F1 -1.5e-15 0` (freeze F1)
- GUI toggle: Click button to enable/disable fitting for each parameter

**Implementation**:
- Parse fit flags from `.par` file
- Construct parameter mask: `fit_mask = jnp.array([1, 0, 1, ...])`
- Apply mask in optimizer: only update parameters where `fit_mask == 1`
- Use `optax.masked()` wrapper to freeze parameters

**Sequential Fitting Workflow**:
1. Fit spin parameters (F0, F1, F2)
2. Fit astrometry (RA, DEC, PMRA, PMDEC, PX)
3. Fit binary parameters (PB, A1, ECC, OM, T0, ...)
4. Fit DM parameters (DM, DM1, DM2, ...)
5. Fit noise parameters (EFAC, EQUAD, red noise, ...)

Users can customize this order via GUI or config file.

### Bayesian Priors (CRITICAL FEATURE)

**Motivation**: Current pulsar timing software (Tempo2, PINT) lacks built-in support for Bayesian priors, leading to unphysical parameter values.

**Prior Specification**:
- **Normal priors**: `F0 ~ Normal(100.5, 0.01)` (mean, std)
- **Uniform priors**: `PB ~ Uniform(10.0, 11.0)` (min, max)
- **Log-uniform priors**: `A1 ~ LogUniform(1e-3, 1e-1)` for scale-invariant parameters
- **Truncated priors**: `SINI ~ TruncatedNormal(0.9, 0.1, low=0, high=1)`

**Prior Storage**:
- Option 1: Extend `.par` file with prior syntax:
  ```
  F0 100.5 1 prior=Normal(100.5, 0.01)
  PB 10.348 1 prior=Uniform(10.0, 11.0)
  ```
- Option 2: Separate `.prior` file with JSON/YAML format
- GUI: Interactive prior specification with live preview

**Prior Integration**:
- NumPyro model function includes prior distributions
- Posterior: `log p(θ|r) = log p(r|θ) + log p(θ) - log p(r)`
- MAP estimation: Maximize posterior (gradient ascent)
- Full Bayesian (optional): HMC/NUTS sampling for uncertainty quantification

### Uncertainty Estimation

**Phase 1: Fisher Matrix Covariances** (Fast, approximate)
- Compute Hessian of log-likelihood at MAP: `H = -∇∇ log p(r|θ_MAP)`
- Covariance: `Cov(θ) ≈ H^(-1)`
- JAX automatic: `jax.hessian()` computes second derivatives
- Speed: Milliseconds for ~50 parameters

**Phase 2: Full Bayesian Sampling** (Slow, exact) - Optional
- HMC/NUTS via NumPyro or BlackJAX
- Posterior samples: Full uncertainty quantification including correlations
- Speed: Minutes to hours depending on dimensionality
- Use case: Final publication-quality uncertainties

**Trade-off**: Let users choose. Fisher matrix for iterative fitting, MCMC for final results.

---

## IV. Graphical User Interface (GUI)

### Framework: Desktop Application

**Choice**: PyQt6 or PySide6 (Qt for Python)
- **Advantages**: Fast rendering, native feel, offline-first, cross-platform
- **Disadvantages**: Larger binary size, steeper learning curve than web frameworks

**Alternative Considered**: Plotly Dash / Streamlit (web-based)
- Rejected due to: Slower rendering, requires server, less responsive for real-time updates

### Design Philosophy

**Inspired by**: Tempo2's `plk` interface
**Improvements over plk**:
1. Modern, clean aesthetic (not cluttered)
2. Real-time parameter updates (see residuals change as you adjust F0)
3. Multi-panel views (residuals, DM, orbital phase, noise diagnostics)
4. Interactive flagging (click TOAs to flag/unflag)
5. Parameter uncertainty visualization (1σ, 2σ error bars on plots)
6. Noise diagnostics (power spectra, ACF, residual histograms)
7. Publication-quality exports (SVG, PDF, PNG with customizable styles)

### Core GUI Components

#### 1. Main Residual Plot Panel

**Features**:
- Scatter plot: TOA MJD (x-axis) vs. Residual (y-axis)
- Color-coded by: Backend, frequency band, observatory (user-selectable)
- Error bars: TOA uncertainties (can toggle on/off)
- Interactive zoom/pan (mouse wheel, click-drag)
- Click TOA to flag/unflag (visual indicator: grayed out or red X)
- Right-click context menu: Flag selected, unflag selected, info

**Real-Time Updates**:
- As user adjusts parameters (sliders, text boxes), residuals recompute
- JAX JIT compilation keeps updates fast (<100ms for 10k TOAs)
- Smooth transitions (no flickering)

**Multi-View Options**:
- Residuals vs. MJD (default)
- Residuals vs. Orbital Phase (for binaries)
- Residuals vs. Frequency (for chromatic effects)
- Pre-fit vs. Post-fit residuals (side-by-side comparison)

#### 2. Parameter Control Panel

**Layout**: Scrollable table with columns:
- Parameter Name (e.g., "F0")
- Current Value (editable text box)
- Uncertainty (display only, updated after fit)
- Fit Toggle (checkbox or button: ON/OFF)
- Prior Specification (dropdown: None, Normal, Uniform, ...)

**Interaction**:
- Double-click value to edit
- Click fit toggle to freeze/unfreeze parameter
- Right-click parameter name: Set prior, reset to initial, copy value

**Slider Controls** (for key parameters):
- F0, F1: Fine-tune sliders below text box
- Range: ±3σ around current value
- Live update: Residuals change as slider moves

**Parameter Groups** (collapsible sections):
- Spin Parameters (F0, F1, F2, ...)
- Astrometry (RA, DEC, PMRA, PMDEC, PX)
- Binary (PB, A1, ECC, OM, T0, ...)
- DM (DM, DM1, DM2, ...)
- Noise (EFAC, EQUAD, red noise, ...)

#### 3. Noise Diagnostic Panel

**Power Spectral Density Plot**:
- Log-log plot: Frequency (cycles/day) vs. Power
- Overlay: Data periodogram vs. Fitted noise model
- Color-coded by noise component (red noise, DM noise, etc.)

**Residual Histogram**:
- Histogram of post-fit residuals
- Overlay: Expected Gaussian (white noise only)
- Highlight non-Gaussian tails (outliers)

**Autocorrelation Function (ACF)**:
- Plot: Lag (days) vs. ACF
- Detect residual correlations (indicates missing noise model)

**Whiteness Test**:
- Reduced χ² statistic
- DW (Durbin-Watson) statistic for serial correlation
- Color indicator: Green (white), Yellow (marginal), Red (non-white)

#### 4. Fit Control Panel

**Fit Buttons**:
- **Fit Selected**: Fit only parameters with toggle ON
- **Fit All**: Fit all parameters (ignore toggles)
- **Reset to Initial**: Restore parameters to values from loaded `.par` file
- **Undo Last Fit**: Revert to previous parameter values

**Fit Settings**:
- Max Iterations (slider: 100 - 10,000)
- Learning Rate (text box: 0.001 - 0.1)
- Convergence Tolerance (text box: 1e-6 - 1e-3)
- Gradient Clipping (checkbox, value text box)

**Progress Indicator**:
- Progress bar showing fit progress (batch N / max_batches)
- Live loss plot: Iteration vs. Loss value
- Stop button: Halt fitting early

#### 5. File Menu

**Standard Operations**:
- Open `.par` file
- Open `.tim` file
- Save `.par` file (write updated parameters)
- Save Flagged `.tim` file (write with FLAG -toa for flagged TOAs)
- Export Residuals (CSV, ASCII)
- Export Figure (PNG, SVG, PDF)

**Session Management**:
- Save Session (`.jug` file: parameters + fit state + GUI settings)
- Load Session
- Recent Files list

#### 6. Advanced Features

**Comparison Mode**:
- Load second `.par` file (e.g., PINT fit result)
- Plot residuals for both models side-by-side
- Difference plot: JUG - PINT residuals

**Batch Processing**:
- Load multiple pulsars from directory
- Apply same fitting procedure to all
- Export summary table (RMS, χ², parameter values)

**Scripting Interface** (Stretch Goal):
- Embedded Python console (IPython)
- Access to JUG internals: `jug.model.F0`, `jug.residuals`
- Automate workflows: Load, fit, save loop

---

## V. File Format Compatibility

### Input Formats

**`.par` Files (Timing Model Parameters)**:
- Standard Tempo/Tempo2/PINT format
- Support for all common parameter names (see Phase 1/2 models)
- Extended syntax for priors (optional): `F0 100.5 1 prior=Normal(100.5, 0.01)`
- Fit flags: `1` = fit, `0` = freeze (or omit flag = freeze)

**`.tim` Files (Time-of-Arrival Data)**:
- FORMAT 1 (Tempo2 standard): Observatory, Frequency, MJD, TOA error, backend flags
- Fallback parsing for non-standard formats
- Support for: JUMP, PHASE, EFAC, EQUAD flags (inline flags)
- FLAG syntax: `-flag value` (e.g., `-sys EFF.EBPP.1360`, `-fe L-wide`)

**Clock Files**:
- Tempo2 format: `data/clock/*.clk`
- Chain: observatory → UTC → TAI → TT
- BIPM files: `tai2tt_bipm{year}.clk`

**Ephemeris Files**:
- JPL DE440s (default): `data/ephemeris/de440s.bsp`
- Support for DE421, DE438, DE440 (configurable)

**Observatory Data**:
- `data/observatory/observatories.dat`: X, Y, Z geocentric coordinates (meters)
- `data/observatory/tempo.aliases`: Observatory code mappings

**Earth Orientation Parameters**:
- `data/earth/eopc04_IAU2000.62-now`: IERS EOP data

### Output Formats

**`.par` Files**: Write updated parameters in standard Tempo2 format
- Preserve comments from original file
- Update parameter values to fitted values
- Include uncertainties as comments (e.g., `# F0_err = 1.234e-10`)

**`.tim` Files**: Write TOAs with updated flags
- Preserve original format
- Add `FLAG -toa` for flagged TOAs (Tempo2 convention)
- Optional: Write residuals as inline comments

**Residual Output**:
- ASCII table: MJD, Residual (μs), Uncertainty (μs), Backend, Frequency
- CSV format for easy import to plotting software

**Session Files** (`.jug` format):
- Custom binary format (pickle, HDF5, or Feather)
- Stores: Parameters, uncertainties, fit state, GUI settings, flagged TOAs

### Data Exchange (Stretch Goals)

**Enterprise Pulsar Objects** (for PTA noise analysis):
- Export as Feather file: Compatible with `enterprise`, `PTMCMCSampler`
- Includes: TOAs, residuals, design matrices, noise model parameters
- Use case: Fit timing model in JUG, export to Enterprise for stochastic GW analysis

**IPTA Data Formats**:
- Read/write IPTA data release formats (if applicable to future PTA features)

---

## VI. Performance Targets

### Typical Use Case

**Dataset**:
- ~10,000 TOAs (10x current test data)
- ~20 timing parameters
- ~5 noise components (EFAC, EQUAD, red noise, DM noise)

**Target Performance**:
- **Residual computation**: <10ms per evaluation (1000 TOAs/ms)
- **Single fit iteration**: <50ms
- **Full fit (100 iterations)**: <5 seconds
- **GUI responsiveness**: <100ms lag for real-time parameter updates

### Stress Test

**Dataset**:
- 100,000 TOAs (NANOGrav/IPTA scale)
- 100+ parameters (high-order spin, binary, glitches, FD)
- 50+ noise components (multi-backend EFAC/EQUAD, GP noise)

**Target Performance**:
- **Residual computation**: <100ms per evaluation
- **Single fit iteration**: <500ms
- **Full fit (100 iterations)**: <60 seconds
- **Faster than**: Tempo2 (baseline), PINT (comparison target)

### Optimization Strategies

**JAX JIT Compilation**:
- All hot-path functions: `@jax.jit` decorated
- Minimize Python overhead in residual loop

**Vectorization**:
- No per-TOA loops in delay calculations
- Batch ephemeris calls where possible
- Pre-compute time-independent quantities (e.g., sky position)

**FFT Covariance**:
- O(N log N) GP likelihood vs. O(N³) matrix inversion
- Critical for large datasets and many noise components

**Sparse Matrices** (if applicable):
- ECORR: Block-diagonal structure
- Use JAX sparse matrix ops or custom kernels

**GPU Acceleration** (Future):
- JAX supports transparent GPU execution
- For 100k+ TOA datasets, GPU could provide 10-100x speedup
- Requires: GPU-compatible libraries (cupy, jax[cuda])

---

## VII. Software Architecture

### Package Structure (High-Level)

```
jug/
├── jug/
│   ├── __init__.py
│   ├── io/                   # File I/O (par, tim, clock, ephemeris)
│   ├── models/               # Timing models (spin, binary, astrometry, DM)
│   ├── delays/               # Delay computations (clock, barycentric, binary, DM)
│   ├── noise/                # Noise models (white, red, GP, FFT covariance)
│   ├── fitting/              # Optimization (JAX, NumPyro, Optax)
│   ├── residuals/            # Residual calculation (JAX-compiled)
│   ├── gui/                  # PyQt6 GUI components
│   ├── utils/                # Helper functions (constants, conversions, time utils)
│   └── tests/                # Unit tests (pytest)
├── data/                      # Reference data (clock, ephemeris, observatory)
├── examples/                  # Example scripts and notebooks
├── docs/                      # Documentation (Sphinx)
├── pyproject.toml             # Package metadata and dependencies
└── README.md
```

### Dependency Management

**Core Dependencies**:
- `jax`, `jaxlib`: JAX framework
- `numpy`: Numerical arrays
- `astropy`: Coordinate transformations, time scales, ephemeris
- `scipy`: Interpolation (clock corrections)
- `numpyro`: Bayesian inference, SVI optimization
- `optax`: Gradient-based optimizers

**GUI Dependencies**:
- `PyQt6` or `PySide6`: Qt bindings
- `pyqtgraph`: Fast plotting (alternative to matplotlib for real-time)
- `matplotlib`: Publication-quality static plots

**Optional Dependencies**:
- `enterprise`: PTA data export (stretch goal)
- `h5py`: HDF5 session files
- `pyarrow`: Feather format export

**Development Dependencies**:
- `pytest`: Testing framework
- `ruff`: Linting and formatting
- `mypy`: Type checking
- `sphinx`: Documentation generation

### Configuration Management

**Config File** (`.jug/config.toml` or `~/.jugrc`):
```toml
[ephemeris]
default = "DE440s"
path = "/path/to/ephemeris/de440s.bsp"

[clock]
data_dir = "/path/to/clock/data/"

[fitting]
default_optimizer = "adamw"
default_lr = 0.01
max_iterations = 5000

[gui]
theme = "dark"  # or "light"
default_colormap = "viridis"
```

Users can override via command-line flags or GUI settings.

---

## VIII. Testing and Validation

### Unit Tests

**Coverage Targets**:
- **Delay calculations**: 100% (critical for accuracy)
- **Timing models**: >90%
- **Noise models**: >90%
- **Fitting routines**: >80%
- **I/O functions**: >80%

**Test Data**:
- Synthetic TOAs with known residuals (test precision)
- Real pulsar data: J0437-4715 (MSP), B1855+09 (binary)
- Edge cases: High-eccentricity binaries, glitches, chromatic noise

### Integration Tests

**End-to-End Workflows**:
1. Load `.par` + `.tim` → Compute residuals → Compare to Tempo2/PINT
2. Fit timing model → Validate parameter convergence
3. Add noise model → Fit → Check reduced χ²

**Accuracy Benchmarks**:
- Residual agreement with Tempo2: RMS difference <1 ns
- Residual agreement with PINT: RMS difference <1 ns
- Parameter agreement: Within 1σ uncertainties

### Continuous Integration (CI)

**GitHub Actions**:
- Run tests on: Linux, macOS, Windows
- Python versions: 3.10, 3.11, 3.12
- JAX backends: CPU, GPU (if runner supports)

**Performance Regression Tests**:
- Benchmark residual computation time (must stay <10ms for 1000 TOAs)
- Alert if performance degrades >10%

---

## IX. Documentation

### User Documentation

**Quickstart Guide**:
- Installation (pip, conda)
- Load example data
- Compute residuals
- Fit timing model
- GUI walkthrough

**Tutorials**:
1. Basic pulsar timing workflow (MSP example)
2. Binary pulsar timing (orbital parameters)
3. Noise modeling (white + red noise)
4. Advanced features (glitches, chromatic noise, priors)
5. Exporting results for publication

**API Reference**:
- Sphinx-generated API docs for all modules
- Docstrings: NumPy style with examples

**FAQ**:
- How does JUG compare to PINT/Tempo2?
- When should I use Bayesian priors?
- How do I add a custom noise model?
- GPU acceleration setup

### Developer Documentation

**Architecture Overview**:
- Package structure diagram
- Data flow from `.par/.tim` to residuals
- JAX JIT compilation strategy

**Contributing Guide**:
- Code style (PEP 8, Ruff config)
- Testing requirements (pytest, coverage)
- Pull request process

**Adding New Models**:
- Template for timing model class
- Registering with JUG's model framework
- Writing tests and validation

---

## X. Roadmap and Milestones

### Milestone 1: Core Timing Package (v0.1.0)

**Goal**: Extract notebook into Python package, reproduce current functionality.

**Deliverables**:
- [ ] Package structure: `jug/io/`, `jug/models/`, `jug/delays/`, `jug/residuals/`
- [ ] Parse `.par` and `.tim` files (Phase 1 models)
- [ ] Compute residuals (JAX-compiled)
- [ ] Unit tests for delay calculations
- [ ] CLI script: `jug-compute-residuals input.par input.tim`

**Timeline**: 2-3 weeks (if user implements, 1 week if Claude assists)

### Milestone 2: Gradient-Based Fitting (v0.2.0)

**Goal**: Implement JAX/NumPyro optimization for timing model fitting.

**Deliverables**:
- [ ] `jug/fitting/` module with AdamW optimizer
- [ ] Parameter freezing (incremental fitting)
- [ ] Fisher matrix uncertainties
- [ ] CLI: `jug-fit input.par input.tim --output fitted.par`
- [ ] Validation: Match Tempo2/PINT fit results within 1σ

**Timeline**: 2-3 weeks

### Milestone 3: White Noise Models (v0.3.0)

**Goal**: Add EFAC, EQUAD, ECORR support.

**Deliverables**:
- [ ] `jug/noise/white.py`: EFAC/EQUAD/ECORR classes
- [ ] Fit white noise parameters jointly with timing model
- [ ] Tests: Synthetic data with known white noise

**Timeline**: 1-2 weeks

### Milestone 4: GP Noise Models (v0.4.0)

**Goal**: Implement FFT covariance for red noise, DM noise, chromatic noise.

**Deliverables**:
- [ ] `jug/noise/gp.py`: FFT covariance framework (port from discovery)
- [ ] Power-law achromatic red noise
- [ ] Power-law DM noise
- [ ] Power-law chromatic scattering noise
- [ ] Extensibility: User-defined PSD functions

**Timeline**: 3-4 weeks

### Milestone 5: Desktop GUI (v0.5.0)

**Goal**: Build PyQt6 GUI with core features.

**Deliverables**:
- [ ] Main window: Residual plot + parameter panel
- [ ] Interactive TOA flagging (click to flag/unflag)
- [ ] Real-time parameter updates (sliders for F0, F1)
- [ ] Fit control panel (fit selected, reset, undo)
- [ ] File menu (open, save, export)

**Timeline**: 4-6 weeks

### Milestone 6: Bayesian Priors (v0.6.0)

**Goal**: Add prior specification and MAP estimation.

**Deliverables**:
- [ ] Prior syntax in `.par` files or separate `.prior` file
- [ ] GUI: Prior specification panel (dropdowns, text boxes)
- [ ] NumPyro model includes priors
- [ ] Posterior sampling (optional, slow mode)

**Timeline**: 2-3 weeks

### Milestone 7: Advanced Models (v0.7.0)

**Goal**: Phase 2 models (glitches, FD, higher-order binary).

**Deliverables**:
- [ ] Glitch model (GLEP, GLPH, GLF0, GLF1, GLTD, GLF0D)
- [ ] FD parameters (FD1, FD2, ...)
- [ ] Higher-order binary effects (PBDOT, OMDOT, XDOT, EDOT)

**Timeline**: 3-4 weeks

### Milestone 8: GUI Polish and Advanced Features (v0.8.0)

**Goal**: Noise diagnostics, multi-panel views, publication exports.

**Deliverables**:
- [ ] Noise diagnostic panel (PSD, ACF, whiteness tests)
- [ ] Multi-view residual plots (vs. MJD, orbital phase, frequency)
- [ ] Export publication-quality figures (SVG, PDF)
- [ ] Session save/load (`.jug` files)

**Timeline**: 3-4 weeks

### Milestone 9: Performance Optimization (v0.9.0)

**Goal**: Achieve stress test targets (100k TOAs, <60s fit).

**Deliverables**:
- [ ] Profile code, identify bottlenecks
- [ ] Optimize: Vectorize planet Shapiro, cache ephemeris calls
- [ ] Sparse matrix support for ECORR
- [ ] GPU acceleration (optional, if beneficial)

**Timeline**: 2-3 weeks

### Milestone 10: v1.0.0 Release

**Goal**: Production-ready, fully documented, tested pulsar timing software.

**Deliverables**:
- [ ] Complete documentation (user guide, API reference, tutorials)
- [ ] >90% test coverage
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Benchmark results: JUG vs. Tempo2 vs. PINT (speed, accuracy)
- [ ] Example datasets and scripts
- [ ] Publication: JOSS or A&A software announcement

**Timeline**: 1-2 weeks (polish, docs, release prep)

---

## XI. Open Questions and Future Considerations

### 1. GPU Acceleration

**Question**: When should we add GPU support?
- **If**: 100k TOAs on CPU already achieves <60s fit, GPU may be unnecessary for v1.0
- **If**: Stress test shows CPU struggles, prioritize GPU in v0.9.0

**Action**: Benchmark on CPU first, decide based on performance.

### 2. Multi-Pulsar / PTA Analysis

**Question**: How soon to add PTA features?
- **Option 1**: Wait until v2.0 (focus on single-pulsar excellence first)
- **Option 2**: Add minimal PTA support in v1.x (e.g., common red noise, GWB)

**Action**: Defer to v2.0. Single-pulsar optimization is sufficient for v1.0.

### 3. Web-Based GUI Alternative

**Question**: Should we offer a web GUI as alternative to desktop?
- **Pro**: Easier deployment, no installation
- **Con**: Slower rendering, requires server

**Action**: Desktop GUI for v1.0. Web GUI could be v1.5 or v2.0 if there's demand.

### 4. Uncertainty Estimation Trade-Off

**Question**: Fisher matrix (fast) vs. MCMC (slow but exact) - which default?
- **Proposal**: Fisher matrix default, MCMC opt-in via flag `--uncertainty mcmc`

**Action**: Implement Fisher first (v0.2.0), MCMC later (v0.6.0 or v0.8.0).

### 5. Feather / Enterprise Export

**Question**: How important is Enterprise compatibility?
- **User feedback**: Stretch goal, not critical for v1.0

**Action**: Add as optional feature in v0.8.0 or v1.1.0 if time permits.

### 6. Custom Noise Process API

**Question**: How should users add custom noise models?
- **Proposal**: Define PSD function, register with decorator:
  ```python
  @jug.noise.register_noise("my_custom_noise", chromatic=True)
  def my_custom_psd(f, df, log10_A, gamma, alpha):
      return ...
  ```

**Action**: Design API in Milestone 4 (v0.4.0), document in developer guide.

---

## XII. Success Metrics

### Technical Metrics

1. **Accuracy**: Residual agreement with Tempo2/PINT <1 ns RMS
2. **Speed**: Fit 10k TOAs in <5 seconds (faster than Tempo2/PINT)
3. **Coverage**: Support >90% of timing parameters in common use
4. **Reliability**: >90% test coverage, zero critical bugs in production

### Usability Metrics

1. **Learning Curve**: New user can fit a pulsar in <10 minutes (documented workflow)
2. **GUI Responsiveness**: <100ms lag for all interactions
3. **Documentation**: >80% of user questions answerable via docs/FAQ

### Adoption Metrics (Post-Release)

1. **Users**: >50 active users within 6 months of v1.0 release
2. **Citations**: >10 citations in pulsar timing papers within 1 year
3. **Contributions**: >5 external contributors add features or fix bugs

---

## XIII. Conclusion

JUG aims to be the **fastest, most flexible, and most user-friendly** pulsar timing software available. By leveraging JAX for gradient-based optimization, implementing efficient noise modeling with FFT covariance, and providing a modern desktop GUI, JUG will set a new standard for pulsar timing analysis.

This design document serves as a living roadmap. As development progresses, we will iterate on architectural decisions, incorporate user feedback, and adapt to new scientific requirements.

**Key Differentiators**:
- **Speed**: JAX JIT compilation + GPU support
- **Bayesian Priors**: First-class support (missing in Tempo2/PINT)
- **Extensibility**: Easy to add custom models and noise processes
- **Modern UX**: Real-time GUI with interactive diagnostics

**Next Steps**:
1. Create package architecture plan (flowchart)
2. Implement Milestone 1 (core timing package)
3. Validate against Tempo2/PINT on real pulsar data
4. Iterate based on performance and accuracy benchmarks

---

**Document Revision History**:
- v1.0 (2025-11-29): Initial design document based on user requirements and reference implementations
