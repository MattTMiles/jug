# JUG GUI Changelog

## 2026-01-27 - Phase 2 Complete + Device Selection + Dynamic Parameters

### Fixed
- ✅ **Installation issues**: Switched from pip to conda for PySide6 (fixes xcb-cursor errors)
- ✅ **JAX version mismatch**: Upgraded JAX and CUDA plugins to 0.9.0
- ✅ **CUDA errors**: Fixed "No FFI handler registered" error
- ✅ **Parameter fitting**: Can now fit parameters not in original .par file (e.g., F2, F3)

### Added
- ✅ **Command-line device selection**: `--gpu` flag for GPU mode (CPU is default)
- ✅ **Dynamic parameter fitting**: `--fit` flag to specify additional parameters
  - Only shows parameters present in .par file by default
  - Use `--fit F2 F3` to add and fit parameters not in .par file
  - Parameters not in .par file are shown in blue with tooltip
  - Missing parameters start at default value (0.0 for spin/DM derivatives)
- ✅ **Enhanced fit results dialog**: Now shows 5 columns instead of 3
  - **New Value**: Fitted parameter value
  - **Previous Value**: Value from .par file (or 0.0 if not present)
  - **Change**: New - Previous
  - **Uncertainty**: 1-sigma error on fitted value
  - Makes it easy to see what changed and by how much
- ✅ **Phase 2 Features**:
  - Parameter selection checkboxes (dynamic based on .par file)
  - Background fit worker (non-blocking UI)
  - Fit results dialog with parameter table
  - Convergence statistics display
  - Reset to prefit functionality
  - Error handling and user feedback
  - Postfit residuals automatically recomputed and plotted

### Changed
- 🔧 **Default device**: CPU (faster for typical datasets <100k TOAs)
- 🔧 **GPU option**: Available via `--gpu` flag for large datasets

### Documentation
- 📝 Created `environment.yml` for conda installation
- 📝 Created `INSTALL.md` with comprehensive installation guide
- 📝 Created `docs/GUI_DEVICE_SELECTION.md` with performance guidelines
- 📝 Updated `docs/QUICK_REFERENCE.md` with GUI documentation
- 📝 Updated `GUI_QUICK_START.txt` with device selection info

---

## Usage

### Default (CPU mode - recommended)
```bash
jug-gui
```

### Load files on startup
```bash
jug-gui pulsar.par pulsar.tim
```

### Fit additional parameters (not in .par file)
```bash
# Fit F2 even if it's not in the .par file
jug-gui pulsar.par pulsar.tim --fit F2

# Fit multiple additional parameters
jug-gui pulsar.par pulsar.tim --fit F2 F3 DM3

# The GUI will show these parameters in blue and pre-select them
# Missing parameters start at 0.0 and are fitted from scratch
```

### GPU mode (for large datasets)
```bash
jug-gui --gpu
```

### Help
```bash
jug-gui --help
```

---

## Performance

**CPU is faster for typical pulsar timing!**

| Dataset | CPU Time | GPU Time | Winner |
|---------|----------|----------|--------|
| 10k TOAs (J1909-3744) | 1.7s | 2.8s | CPU ✅ |
| 100k TOAs (estimated) | ~15s | ~14s | Similar |
| 1M TOAs (estimated) | ~150s | ~60s | GPU ✅ |

**Recommendation:**
- Use default (`jug-gui`) for <100k TOAs
- Use `jug-gui --gpu` for >100k TOAs or PTAs

---

## Installation

### For GitHub users
```bash
git clone <repo>
cd jug
mamba env create -f environment.yml
conda activate jug
jug-gui
```

No more library path issues!

---

## What's Working

✅ Phase 1 (MVP):
- Load .par and .tim files
- Compute and plot residuals
- Interactive plot (zoom, pan)
- Menu bar and status bar

✅ Phase 2 (Fit Integration):
- Parameter selection
- Background fitting (non-blocking UI)
- Fit results display
- Statistics panel
- Error handling
- Device selection (CPU/GPU)

---

## Known Limitations

1. Postfit residuals not recomputed (shows statistics only)
2. Can't save fitted .par files yet
3. No parameter editing dialog yet
4. No command-line file arguments yet

These are planned for Phase 3 (optional enhancements).

---

## Next Steps (Optional)

### Phase 3: Parameter Editing
- Interactive parameter editor dialog
- Real-time residual updates
- Save modified .par files

### Phase 4: Advanced Features
- Prefit vs postfit plot comparison
- Plot export (PNG, PDF)
- Fit history tracking
- TOA flagging/exclusion
- Custom convergence settings

### Phase 5: Polish
- Custom themes/styling
- Application icon
- Keyboard shortcuts for all actions
- User preferences
- Device indicator in status bar

---

## Bug Reports

Report issues at: https://github.com/yourusername/jug/issues

Include:
- JUG version: `pip show jug-timing`
- Python version: `python --version`
- OS and version
- Error message and traceback
- Steps to reproduce
