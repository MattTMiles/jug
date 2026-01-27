# JUG Quick Start Guide - Updated for Engine Architecture

## Installation

```bash
# Core (CLI + engine)
pip install jug-timing

# With Qt GUI  
pip install jug-timing[qt]

# With server
pip install jug-timing[server]

# Everything
pip install jug-timing[all]
```

## Command-Line Tools

```bash
# Compute residuals
jug-compute-residuals pulsar.par pulsar.tim

# Fit parameters
jug-fit pulsar.par pulsar.tim --params F0 F1 DM

# Qt GUI
jug-gui pulsar.par pulsar.tim --fit F2

# Server (requires [server] install)
jugd serve --port 8080
```

## Python API - Session-Based (Recommended)

```python
from jug.engine import open_session

# Open session (parse files once)
session = open_session('pulsar.par', 'pulsar.tim')

# Compute residuals (fast with caching)
result = session.compute_residuals()
print(f"RMS: {result['rms_us']:.3f} μs")
print(f"TOAs: {len(result['residuals_us'])}")

# Fit parameters (reuses session)
fit = session.fit_parameters(['F0', 'F1'])
print(f"F0: {fit['final_params']['F0']:.15f} Hz")
print(f"Converged: {fit['converged']}")

# Compute again (even faster - uses cache!)
result2 = session.compute_residuals()
```

## Python API - Legacy (Still Works)

```python
from jug.engine import compute_residuals, fit_parameters

# One-shot operations
result = compute_residuals('pulsar.par', 'pulsar.tim')
fit = fit_parameters('pulsar.par', 'pulsar.tim', ['F0', 'F1'])
```

## Server Usage

```bash
# Start server on cluster
jugd serve --port 8080

# On local machine, create SSH tunnel
ssh -L 8080:localhost:8080 user@cluster

# Access API
curl http://localhost:8080/health
curl http://localhost:8080/version

# API docs at http://localhost:8080/docs
```

## Performance Tips

### Use Sessions for Multiple Operations
```python
# ❌ Slow (re-parses every time)
result1 = compute_residuals('pulsar.par', 'pulsar.tim')  # 2.4s
result2 = compute_residuals('pulsar.par', 'pulsar.tim')  # 2.4s

# ✅ Fast (parse once, cache results)
session = open_session('pulsar.par', 'pulsar.tim')
result1 = session.compute_residuals()  # 2.4s
result2 = session.compute_residuals()  # 0.0s (cached!)
```

### Device Selection (GPU vs CPU)
```python
# Auto-detect (uses GPU if available)
fit = session.fit_parameters(['F0', 'F1'], device=None)

# Force CPU (faster for <100k TOAs)
fit = session.fit_parameters(['F0', 'F1'], device='cpu')

# Force GPU (faster for >100k TOAs)
fit = session.fit_parameters(['F0', 'F1'], device='gpu')
```

## Next Steps

- **Qt GUI Refactoring**: Coming in Step 2
- **Tauri App**: Coming in Step 3
- **Advanced Features**: Parameter editing, plot export, etc.

## Documentation

- `docs/ENGINE_STEP1_COMPLETE.md` - Detailed architecture docs
- `docs/GUI_FIT_MISSING_PARAMS.md` - GUI parameter fitting
- `README.md` - Full documentation (to be updated)
