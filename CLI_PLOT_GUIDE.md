# CLI Plot Feature - Quick Guide

## Usage

The `jug-compute-residuals` CLI now supports generating residual plots:

```bash
# Generate plot in current directory
python -m jug.scripts.compute_residuals --plot J1909-3744.par J1909-3744.tim

# Or if installed via pip
jug-compute-residuals --plot J1909-3744.par J1909-3744.tim

# Specify output directory
jug-compute-residuals --plot --output-dir ./plots J1909-3744.par J1909-3744.tim
```

## Output

Creates a PNG file named `<pulsar>_residuals.png` with:
- **Top panel**: Residuals vs MJD (scatter plot with error bars and zero line)
- **Bottom panel**: Residual distribution histogram with statistics

## Example Output

For J1909-3744:
```
Results for J1909-3744_tdb:
  RMS:        0.817 μs
  Mean:       0.052 μs
  N_TOAs:     10408
  Plot saved to: J1909-3744_tdb_residuals.png
```

## Plot Features

- **Scatter plot**: Blue dots showing residuals over time with error bars
- **Error bars**: TOA uncertainties from .tim file displayed on each point
- **Zero line**: Red dashed line for reference
- **Title**: Shows pulsar name and RMS
- **Histogram**: Distribution with mean and count
- **High quality**: 150 DPI for publication-quality output

## Dependencies

Requires matplotlib (already included in optional `gui` dependencies):
```bash
pip install jug-timing[gui]
```

If matplotlib is not available, the CLI will print a warning and continue without plotting.

## Quick Visual Verification

The plot lets you instantly verify:
- ✅ Residuals centered near zero (good fit)
- ✅ Even distribution across time (no systematics)
- ✅ Gaussian-like histogram (white noise)
- ⚠️ Outliers or trends (potential issues)

Date: 2025-11-29
