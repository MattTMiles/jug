# JUG Quick Reference Card

**Version**: 0.1.0 (Milestone 1 Complete)
**Date**: 2025-11-29

---

## Installation

```bash
cd /home/mattm/soft/JUG
pip install -e .
```

---

## CLI Usage

### Compute Residuals

```bash
# Basic
jug-compute-residuals pulsar.par pulsar.tim

# With plot
jug-compute-residuals --plot pulsar.par pulsar.tim

# Custom clock directory
jug-compute-residuals --clock-dir /path/to/clocks pulsar.par pulsar.tim

# Different observatory
jug-compute-residuals --observatory parkes pulsar.par pulsar.tim

# Specify output location
jug-compute-residuals --plot --output-dir ./plots pulsar.par pulsar.tim
```

### Get Help

```bash
jug-compute-residuals --help
```

---

## Python API

```python
from jug.residuals.simple_calculator import compute_residuals_simple

# Compute residuals
result = compute_residuals_simple(
    par_file="J1909-3744.par",
    tim_file="J1909-3744.tim",
    clock_dir="data/clock",
    observatory="meerkat"
)

# Access results
print(f"RMS: {result['rms_us']:.3f} μs")
print(f"Mean: {result['mean_us']:.3f} μs")
print(f"N TOAs: {result['n_toas']}")

# Get residual array
residuals = result['residuals_us']  # in microseconds
tdb_times = result['tdb_mjd']       # TDB times in MJD
```

---

## Output

### Console Output

```
============================================================
JUG Simple Residual Calculator
============================================================

1. Loading files...
   Loaded 10408 TOAs from J1909-3744.tim
   Loaded timing model from J1909-3744_tdb.par

2. Loading clock corrections...
   Loaded 3 clock files (using BIPM2024)

3. Computing TDB (standalone, no PINT)...
   Computed TDB for 10408 TOAs

4. Computing astrometric delays...
   Computing planetary Shapiro delays...

5. Running JAX delay kernel...

6. Computing phase residuals...

   Computing TZR phase at TZRMJD...
   TZR delay breakdown:
     Roemer+Shapiro: -46.802741244 s
     DM:             0.040724427 s
     Solar wind:     0.000000202 s
     FD:             -0.000000977 s
     Binary:         1.187806441 s
     TOTAL:          -45.574211152 s

============================================================
Results:
  RMS: 0.817 μs
  Mean: 0.052 μs
  Min: -7.520 μs
  Max: 8.386 μs
  N_TOAs: 10408
============================================================
```

### Plot Output

When using `--plot`:
- Creates `<pulsar>_residuals.png`
- Two panels: residuals vs time, histogram
- Error bars from TOA uncertainties
- 150 DPI publication quality

---

## Supported Features

### Timing Models

- ✅ Spin: F0, F1, F2 (frequency and derivatives)
- ✅ DM: Polynomial (DM, DM1, DM2, ...)
- ✅ Binary: ELL1 model (low eccentricity)
- ✅ Astrometry: RA, DEC, proper motion, parallax
- ✅ Shapiro delay: Both H3/STIG and M2/SINI

### Binary Models

- ✅ ELL1 (low eccentricity)
- ⏳ BT/DD (Milestone 2+)

### Delays Computed

- ✅ Clock corrections (observatory → TT)
- ✅ TDB conversion (TT → TDB/TCB)
- ✅ Roemer delay (geometric light travel)
- ✅ Solar Shapiro delay
- ✅ Planetary Shapiro delays (Jupiter, Saturn, etc.)
- ✅ Binary delays (Roemer, Einstein, Shapiro)
- ✅ DM delay (cold plasma)
- ✅ Solar wind delay
- ✅ Frequency-dependent delays

---

## Performance

**Tested on**: J1909-3744 (10,408 TOAs, challenging MSP binary)

- **Accuracy**: 0.817 μs RMS (matches PINT)
- **Precision**: 0.003 μs std vs PINT
- **Speed**: ~2-3 seconds for 10k TOAs
- **All delays**: JIT-compiled with JAX

---

## Required Files

### Data Directory Structure

```
data/
├── clock/
│   ├── mk2utc.clk              # MeerKAT clock
│   ├── gps2utc.clk             # GPS clock
│   └── tai2tt_bipm2024.clk     # BIPM 2024
└── (ephemeris files managed by Astropy)
```

### Input Files

- `.par` file: Timing model parameters
- `.tim` file: Time-of-arrival measurements

---

## Troubleshooting

### "Clock file not found"

```bash
# Make sure clock files exist
ls data/clock/

# Or specify custom directory
jug-compute-residuals --clock-dir /path/to/clocks pulsar.par pulsar.tim
```

### "JAX not using float64"

Check if JAX x64 is enabled:
```python
import jax
print(jax.config.jax_enable_x64)  # Should be True
```

JUG automatically enables x64 mode.

### "Matplotlib not available"

Install optional dependencies:
```bash
pip install jug-timing[gui]
```

Or install matplotlib directly:
```bash
pip install matplotlib
```

---

## Development

### Run Tests

```bash
cd /home/mattm/soft/JUG
pytest jug/tests/
```

### Build Documentation

```bash
cd docs/
make html
```

---

## Documentation

- `README.md` - Getting started
- `CLI_PLOT_GUIDE.md` - Plot feature details
- `CLAUDE.md` - Implementation notes
- `JUG_master_design_philosophy.md` - Design overview
- `MILESTONE_1_COMPLETION.md` - M1 completion report

---

## Support

**Issues**: Create GitHub issue (when repo is public)
**Email**: [Your email here]
**Notebook**: `playground/residual_maker_playground_active_MK7.ipynb`

---

## What's Next?

**Milestone 2**: Gradient-based fitting
- Parameter optimization
- Fisher matrix uncertainties
- `jug-fit` CLI command

See `MILESTONE_2_HANDOFF.md` for details.

---

**JUG v0.1.0** - JAX-based pulsar timing analysis
