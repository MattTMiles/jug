#!/usr/bin/env python3
"""Plot JAX residuals vs simple_calculator baseline to diagnose differences."""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

# Compute both versions
print("Computing baseline (simple_calculator)...")
result_baseline = compute_residuals_simple(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim',
    clock_dir='data/clock',
    observatory='meerkat'
)

print("\nComputing JAX version...")
fixed_data = prepare_fixed_data(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim',
    clock_dir='data/clock',
    observatory='meerkat'
)

# Setup JAX computation
par_params = fixed_data['par_params']
param_names = ('F0', 'F1')
params_array = jnp.array([par_params['F0'], par_params['F1']])
fixed_params = {k: v for k, v in par_params.items() if k not in param_names}

residuals_jax_sec = compute_residuals_jax(
    params_array, param_names,
    fixed_data['tdb_mjd'],
    fixed_data['freq_mhz'],
    fixed_data['geometric_delay_sec'],
    fixed_data['other_delays_minus_dm_sec'],
    fixed_data['pepoch'],
    fixed_data['dm_epoch'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    fixed_params
)

# Convert to microseconds
residuals_baseline_us = result_baseline['residuals_us']
residuals_jax_us = np.array(residuals_jax_sec) * 1e6
tdb_mjd = result_baseline['tdb_mjd']

# Compute difference
diff_us = residuals_jax_us - residuals_baseline_us

# Statistics
print(f"\nStatistics:")
print(f"  Baseline RMS: {np.sqrt(np.mean(residuals_baseline_us**2)):.6f} μs")
print(f"  JAX RMS: {np.sqrt(np.mean(residuals_jax_us**2)):.6f} μs")
print(f"  Difference RMS: {np.sqrt(np.mean(diff_us**2)):.6f} μs")
print(f"  Difference mean: {np.mean(diff_us):.6f} μs")
print(f"  Difference std: {np.std(diff_us):.6f} μs")
print(f"  Min diff: {np.min(diff_us):.6f} μs")
print(f"  Max diff: {np.max(diff_us):.6f} μs")

# Create figure with multiple panels
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Both residuals vs time
ax = axes[0, 0]
ax.plot(tdb_mjd, residuals_baseline_us, 'o', alpha=0.3, ms=2, label='Baseline (simple_calc)')
ax.plot(tdb_mjd, residuals_jax_us, 'x', alpha=0.3, ms=2, label='JAX')
ax.set_xlabel('MJD (TDB)')
ax.set_ylabel('Residual (μs)')
ax.set_title('Residuals vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Difference vs time
ax = axes[0, 1]
ax.plot(tdb_mjd, diff_us, 'o', alpha=0.5, ms=2)
ax.axhline(np.mean(diff_us), color='r', ls='--', label=f'Mean = {np.mean(diff_us):.3f} μs')
ax.axhline(np.mean(diff_us) + np.std(diff_us), color='orange', ls=':', alpha=0.5)
ax.axhline(np.mean(diff_us) - np.std(diff_us), color='orange', ls=':', alpha=0.5)
ax.set_xlabel('MJD (TDB)')
ax.set_ylabel('JAX - Baseline (μs)')
ax.set_title(f'Difference vs Time (std={np.std(diff_us):.3f} μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Histogram of differences
ax = axes[1, 0]
ax.hist(diff_us, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(np.mean(diff_us), color='r', ls='--', linewidth=2, label=f'Mean = {np.mean(diff_us):.3f} μs')
ax.axvline(0, color='k', ls='-', linewidth=1, alpha=0.5)
ax.set_xlabel('JAX - Baseline (μs)')
ax.set_ylabel('Count')
ax.set_title('Histogram of Differences')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Scatter plot
ax = axes[1, 1]
ax.plot(residuals_baseline_us, residuals_jax_us, 'o', alpha=0.3, ms=2)
lim_min = min(residuals_baseline_us.min(), residuals_jax_us.min())
lim_max = max(residuals_baseline_us.max(), residuals_jax_us.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='1:1 line')
ax.set_xlabel('Baseline Residual (μs)')
ax.set_ylabel('JAX Residual (μs)')
ax.set_title('JAX vs Baseline Scatter')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('jax_vs_baseline_residuals.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: jax_vs_baseline_residuals.png")
plt.close()

# Also create a detailed zoomed plot of the difference
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Panel 1: Difference vs baseline residual (to see if error depends on residual size)
ax = axes[0]
ax.plot(residuals_baseline_us, diff_us, 'o', alpha=0.3, ms=2)
ax.axhline(np.mean(diff_us), color='r', ls='--', label=f'Mean = {np.mean(diff_us):.3f} μs')
ax.axhline(0, color='k', ls='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Baseline Residual (μs)')
ax.set_ylabel('JAX - Baseline (μs)')
ax.set_title('Difference vs Residual Magnitude')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Difference vs frequency (to see if error depends on frequency)
freq_mhz = np.array([fixed_data['freq_mhz'][i] for i in range(len(diff_us))])
ax = axes[1]
ax.plot(freq_mhz, diff_us, 'o', alpha=0.3, ms=2)
ax.axhline(np.mean(diff_us), color='r', ls='--', label=f'Mean = {np.mean(diff_us):.3f} μs')
ax.axhline(0, color='k', ls='-', linewidth=1, alpha=0.5)
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('JAX - Baseline (μs)')
ax.set_title('Difference vs Frequency (check for DM-related errors)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jax_vs_baseline_detailed.png', dpi=150, bbox_inches='tight')
print(f"Detail plot saved to: jax_vs_baseline_detailed.png")
