# JUG Installation Guide

## Quick Start (Recommended)

```bash
pip install jug-timing
```

**That's it!** This single command installs:
- ✅ Core timing engine with JAX acceleration
- ✅ CLI tools: `jug-compute-residuals`, `jug-fit`, `jug-gui`, `jugd`
- ✅ Qt GUI for interactive analysis
- ✅ REST API server for remote access

## What You Get

After installation, you have access to:

### 1. Command-Line Tools
```bash
# Compute residuals
jug-compute-residuals pulsar.par pulsar.tim

# Fit parameters
jug-fit pulsar.par pulsar.tim --params F0 F1

# Interactive Qt GUI
jug-gui pulsar.par pulsar.tim

# REST API server
jugd serve --port 8080
```

### 2. Python API
```python
from jug.engine import open_session

# Session-based API (with caching)
session = open_session('pulsar.par', 'pulsar.tim')
result = session.compute_residuals()
fit = session.fit_parameters(['F0', 'F1'])
```

### 3. Qt GUI Application
```bash
# Launch GUI
jug-gui

# Or load files directly
jug-gui data/pulsars/J1909.par data/pulsars/J1909.tim

# Fit additional parameters
jug-gui J1909.par J1909.tim --fit F2 DM1
```

### 4. Server for Remote Access
```bash
# Start server on cluster
jugd serve --port 8080

# Connect from local machine via SSH tunnel
ssh -L 8080:localhost:8080 user@cluster

# Access API at http://localhost:8080/docs
```

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64) | ✅ Full | Recommended for HPC clusters |
| macOS (Intel/ARM) | ✅ Full | Desktop GUI + server |
| Windows | ✅ Core + GUI | Server untested on Windows |

## Requirements

- **Python**: 3.10 or later
- **JAX**: CPU version included (GPU optional)
- **Qt**: PySide6 for GUI (included)
- **FastAPI**: For server (included)

## Optional: GPU Acceleration

If you have an NVIDIA GPU and want GPU-accelerated fitting:

```bash
# Install CUDA-enabled JAX (optional)
pip install --upgrade "jax[cuda12]"
```

**Note**: CPU is actually faster for typical pulsar timing (<100k TOAs).  
GPU only helps for very large datasets (>100k TOAs) or PTA analyses.

## Development Installation

For contributors:

```bash
# Clone repository
git clone https://github.com/yourusername/jug
cd jug

# Install in editable mode with dev tools
pip install -e .[dev]

# Run tests
pytest

# Run linter
ruff check jug/
```

## Troubleshooting

### Qt GUI won't start
```bash
# Install Qt dependencies (Linux)
conda install -c conda-forge pyside6

# Or use conda environment
conda env create -f environment.yml
conda activate jug
```

### Server fails to start
```bash
# Verify FastAPI is installed
python -c "import fastapi; print('FastAPI OK')"

# If not, reinstall
pip install --force-reinstall jug-timing
```

### Import errors
```bash
# Make sure you're using the right Python
which python
python --version  # Should be 3.10+

# Reinstall if needed
pip uninstall jug-timing
pip install jug-timing
```

## Upgrading

```bash
pip install --upgrade jug-timing
```

## Uninstalling

```bash
pip uninstall jug-timing
```

## What's Next?

After installation:
1. Try the quick start: `jug-gui` or `jug-compute-residuals --help`
2. Read the quick reference: `docs/QUICKSTART_ENGINE.md`
3. Check the full documentation: `docs/ENGINE_STEP1_COMPLETE.md`
4. Explore examples in the `examples/` directory

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory  
- **Issues**: https://github.com/yourusername/jug/issues
- **Discussions**: https://github.com/yourusername/jug/discussions

## Version Information

```bash
# Check installed version
pip show jug-timing

# Check available commands
jug-compute-residuals --help
jug-fit --help
jug-gui --help
jugd --help
```

---

**Ready to start timing pulsars? Let's go!** 🚀
