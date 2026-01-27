# JUG Installation Guide

## Recommended Installation (with GUI support)

The easiest way to install JUG with full GUI support is using conda/mamba:

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/jug.git
cd jug
```

### 2. Create conda environment from environment.yml
```bash
# Using mamba (faster, recommended)
mamba env create -f environment.yml

# Or using conda
conda env create -f environment.yml
```

### 3. Activate the environment
```bash
conda activate jug
```

### 4. Verify installation
```bash
# Test CLI tools
jug-fit --help

# Test GUI (will open window)
jug-gui
```

That's it! The GUI should work out of the box.

---

## Alternative Installation Methods

### Method 1: pip install (Core functionality only)

If you only need CLI tools and don't need the GUI:

```bash
pip install -e .
```

This installs JUG with core dependencies but **not** GUI support.

### Method 2: pip install with GUI (Advanced)

If you want to use pip for GUI dependencies, you'll need to ensure system Qt libraries are properly configured:

```bash
# Install JUG with GUI dependencies
pip install -e ".[gui]"

# On Linux, you may need to set library paths:
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Or install system Qt dependencies:
# Ubuntu/Debian:
sudo apt-get install libxcb-cursor0 qt6-base-dev

# Fedora/RHEL:
sudo dnf install xcb-util-cursor qt6-qtbase
```

**Note:** We recommend using conda (Method 1) for GUI support to avoid library path issues.

---

## Installation Options

### Core only (no GUI)
```bash
pip install -e .
```

### With GUI (recommended via conda)
```bash
mamba env create -f environment.yml
```

### With development tools
```bash
pip install -e ".[dev]"
```

### With documentation tools
```bash
pip install -e ".[docs]"
```

### Everything
```bash
pip install -e ".[all]"
```

---

## Troubleshooting

### GUI won't launch - Qt platform plugin error

**Error message:**
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
libxcb-cursor.so.0: cannot open shared object file
```

**Solution:** Use conda installation method instead of pip for GUI dependencies:

```bash
# Uninstall pip-installed PySide6
pip uninstall PySide6 PySide6-Essentials PySide6-Addons

# Install from conda-forge
mamba install pyside6 pyqtgraph -c conda-forge
```

### JAX GPU support

For GPU acceleration (optional):

```bash
# CUDA 12.x
pip install --upgrade "jax[cuda12]"

# Or CUDA 11.x
pip install --upgrade "jax[cuda11]"
```

### Test installation

```bash
# Test core functionality
python -c "from jug.fitting import fit_parameters_optimized; print('✓ JUG core installed')"

# Test GUI
python -c "from jug.gui.main import main; print('✓ GUI dependencies installed')"
```

---

## System Requirements

- **Python:** 3.10 or later
- **OS:** Linux, macOS, Windows
- **RAM:** 4GB minimum, 8GB+ recommended
- **Display:** For GUI, requires X11 (Linux), Cocoa (macOS), or Windows display system

---

## Quick Start

After installation, try:

```bash
# Compute residuals (CLI)
jug-compute-residuals data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim

# Fit parameters (CLI)
jug-fit data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim --fit F0 F1

# Launch GUI
jug-gui
```

See `docs/QUICK_REFERENCE.md` for full documentation.
