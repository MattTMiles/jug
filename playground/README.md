# JUG: JAX-based Pulsar Timing

**Version**: 0.1.0 (Development)
**Status**: Alpha - Under Active Development

JUG is a modern, high-performance pulsar timing software built from the ground up using JAX for automatic differentiation and JIT compilation. It provides a complete timing pipeline with gradient-based fitting and advanced noise modeling.

## Key Features

- **Speed First**: JAX JIT compilation for microsecond-precision timing at millisecond computation speeds
- **Independent**: No dependencies on PINT or Tempo2 for core functionality
- **Gradient-Based Fitting**: NumPyro/Optax optimization with automatic differentiation
- **Bayesian Priors**: First-class support for prior specification (unique feature)
- **FFT Covariance**: Efficient O(N log N) Gaussian process noise modeling
- **Extensible**: Easy to add custom timing models and noise processes
- **Modern GUI**: PyQt6 desktop interface with real-time parameter updates (coming soon)

## Installation

### Development Installation

```bash
# Clone repository
cd /home/mattm/soft/JUG

# Install in editable mode with all dependencies
pip install -e ".[all]"

# Or minimal install (no GUI)
pip install -e .
```

### Dependencies

Core requirements:
- Python >=3.10
- JAX >=0.4.0
- NumPy >=1.24.0
- Astropy >=5.3.0
- NumPyro >=0.13.0
- Optax >=0.1.7
- SciPy >=1.10.0

## Quick Start

```python
from jug.io import parse_par_file, parse_tim_file
from jug.residuals import compute_residuals

# Load timing model and TOAs
model = parse_par_file("J0437-4715.par")
toas = parse_tim_file("J0437-4715.tim")

# Compute residuals
residuals = compute_residuals(toas['mjd'], toas['freq'], model)

print(f"RMS residual: {residuals.std() * 1e6:.2f} μs")
```

### Command-Line Interface

```bash
# Compute residuals
jug-compute-residuals J0437-4715.par J0437-4715.tim --output residuals.csv

# Fit timing model
jug-fit J0437-4715.par J0437-4715.tim --output fitted.par

# Launch GUI (coming in v0.5)
jug-gui J0437-4715.par J0437-4715.tim
```

## Current Status

**Milestone 1 (v0.1.0)**: Core Timing Package - In Progress
- [x] Package structure
- [x] Configuration (pyproject.toml)
- [ ] Code extraction from notebook
- [ ] Unit tests
- [ ] CLI tools

See `JUG_PROGRESS_TRACKER.md` for detailed status.

## Documentation

- **Design Philosophy**: See `JUG_master_design_philosophy.md`
- **Architecture**: See `JUG_package_architecture_flowcharts.md`
- **Implementation Guide**: See `JUG_implementation_guide.md`
- **Progress Tracker**: See `JUG_PROGRESS_TRACKER.md`

## Development Workflow

```bash
# Run tests
pytest jug/tests -v

# Check test coverage
pytest --cov=jug --cov-report=html

# Lint code
ruff check jug/

# Format code
black jug/
```

## Performance Targets

- **Typical Use**: <5 seconds for 10,000 TOAs with ~20 parameters
- **Stress Test**: <60 seconds for 100,000 TOAs with 100+ parameters
- **Faster than**: Tempo2 and PINT on equivalent hardware

## Roadmap

- **v0.1.0** (Current): Core timing package with residual computation
- **v0.2.0**: Gradient-based fitting with NumPyro/Optax
- **v0.3.0**: White noise models (EFAC, EQUAD, ECORR)
- **v0.4.0**: GP noise models with FFT covariance
- **v0.5.0**: PyQt6 desktop GUI
- **v0.6.0**: Bayesian priors
- **v1.0.0**: Production release

## Contributing

JUG is under active development. Contributions welcome!

## License

MIT License - See LICENSE file

## Citation

If you use JUG in your research, please cite:

```bibtex
@software{jug2025,
  author = {Miles, Matt},
  title = {JUG: JAX-based Pulsar Timing},
  year = {2025},
  url = {https://github.com/yourusername/jug}
}
```

## Acknowledgments

JUG builds on concepts from:
- PINT: Pulsar timing software
- Tempo2: Classic pulsar timing package
- Discovery: Fourier-domain GP noise framework
- Enterprise: PTA analysis toolkit

---

**Status**: Alpha software under active development. APIs may change.
