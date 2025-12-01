#!/usr/bin/env python3
"""Compare PINT, simple_calculator baseline, and JAX residuals."""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax, compute_residuals_jax_from_dt

# Try to import PINT
try:
    import pint
    from pint.models import get_model
    from pint.toa import get_TOAs
    from pint.residuals import Residuals
    has_pint = True
except ImportError:
    has_pint = False
    print("WARNING: PINT not available, will only compare baseline vs JAX")

print("="*60)
print("Computing residuals from all methods...")
print("="*60)

# Compute baseline (simple_calculator)
print("\n1. Computing baseline (simple_calculator)...")
result_baseline = compute_residuals_simple(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim',
    clock_dir='data/clock',
    observatory='meerkat'
)
residuals_baseline_us = result_baseline['residuals_us']
tdb_mjd = result_baseline['tdb_mjd']

# Compute JAX
print("\n2. Computing JAX version...")
fixed_data = prepare_fixed_data(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim',
    clock_dir='data/clock',
    observatory='meerkat'
)

par_params = fixed_data['par_params']
param_names = ('F0', 'F1')
params_array = jnp.array([par_params['F0'], par_params['F1']])
fixed_params = {k: v for k, v in par_params.items() if k not in param_names}

residuals_jax_sec = compute_residuals_jax_from_dt(
    params_array, param_names,
    fixed_data['dt_sec'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    fixed_params
)
residuals_jax_us = np.array(residuals_jax_sec) * 1e6

# Compute PINT residuals
if has_pint:
    print("\n3. Computing PINT residuals...")
    try:
        model = get_model('data/pulsars/J1909-3744_tdb.par')
        toas = get_TOAs('data/pulsars/J1909-3744.tim', ephem='DE440', planets=True, include_bipm=False)
        residuals_pint = Residuals(toas, model)
        residuals_pint_us = residuals_pint.time_resids.to_value('us')
        print(f"   Loaded {len(residuals_pint_us)} TOAs from PINT")
        
        # PINT might have different TOA ordering, match by MJD
        pint_mjds = toas.get_mjds().value
        
    except Exception as e:
        print(f"   Error loading PINT: {e}")
        has_pint = False
        
# Statistics
print("\n" + "="*60)
print("Statistics:")
print("="*60)
print(f"\nBaseline (simple_calculator):")
print(f"  RMS:  {np.sqrt(np.mean(residuals_baseline_us**2)):.6f} μs")
print(f"  Mean: {np.mean(residuals_baseline_us):.6f} μs")
print(f"  Std:  {np.std(residuals_baseline_us):.6f} μs")

print(f"\nJAX:")
print(f"  RMS:  {np.sqrt(np.mean(residuals_jax_us**2)):.6f} μs")
print(f"  Mean: {np.mean(residuals_jax_us):.6f} μs")
print(f"  Std:  {np.std(residuals_jax_us):.6f} μs")

if has_pint:
    print(f"\nPINT:")
    print(f"  RMS:  {np.sqrt(np.mean(residuals_pint_us**2)):.6f} μs")
    print(f"  Mean: {np.mean(residuals_pint_us):.6f} μs")
    print(f"  Std:  {np.std(residuals_pint_us):.6f} μs")

# Compute differences
diff_jax_baseline = residuals_jax_us - residuals_baseline_us

print(f"\nDifferences:")
print(f"\nJAX - Baseline:")
print(f"  RMS:  {np.sqrt(np.mean(diff_jax_baseline**2)):.6f} μs")
print(f"  Mean: {np.mean(diff_jax_baseline):.6f} μs")
print(f"  Std:  {np.std(diff_jax_baseline):.6f} μs")

if has_pint:
    # Need to match TOAs between PINT and our calculation
    # PINT and JUG should have same order if loaded from same file
    if len(residuals_pint_us) == len(residuals_baseline_us):
        diff_baseline_pint = residuals_baseline_us - residuals_pint_us
        diff_jax_pint = residuals_jax_us - residuals_pint_us
        
        print(f"\nBaseline - PINT:")
        print(f"  RMS:  {np.sqrt(np.mean(diff_baseline_pint**2)):.6f} μs")
        print(f"  Mean: {np.mean(diff_baseline_pint):.6f} μs")
        print(f"  Std:  {np.std(diff_baseline_pint):.6f} μs")
        
        print(f"\nJAX - PINT:")
        print(f"  RMS:  {np.sqrt(np.mean(diff_jax_pint**2)):.6f} μs")
        print(f"  Mean: {np.mean(diff_jax_pint):.6f} μs")
        print(f"  Std:  {np.std(diff_jax_pint):.6f} μs")
    else:
        print(f"\nWARNING: TOA count mismatch (PINT: {len(residuals_pint_us)}, JUG: {len(residuals_baseline_us)})")
        has_pint = False

# Create comparison plots
if has_pint:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
else:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = np.vstack([axes, [None, None]])

# Panel 1: All residuals vs time
ax = axes[0, 0]
ax.plot(tdb_mjd, residuals_baseline_us, 'o', alpha=0.3, ms=2, label='Baseline')
ax.plot(tdb_mjd, residuals_jax_us, 'x', alpha=0.3, ms=2, label='JAX')
if has_pint:
    ax.plot(pint_mjds, residuals_pint_us, '+', alpha=0.3, ms=2, label='PINT')
ax.set_xlabel('MJD (TDB)')
ax.set_ylabel('Residual (μs)')
ax.set_title('All Residuals vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: JAX - Baseline
ax = axes[0, 1]
ax.plot(tdb_mjd, diff_jax_baseline, 'o', alpha=0.5, ms=2)
ax.axhline(np.mean(diff_jax_baseline), color='r', ls='--', 
           label=f'Mean = {np.mean(diff_jax_baseline):.3f} μs')
ax.axhline(0, color='k', ls='-', linewidth=1, alpha=0.5)
ax.set_xlabel('MJD (TDB)')
ax.set_ylabel('JAX - Baseline (μs)')
ax.set_title(f'JAX - Baseline (std={np.std(diff_jax_baseline):.3f} μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Baseline - PINT (if available)
if has_pint:
    ax = axes[1, 0]
    ax.plot(tdb_mjd, diff_baseline_pint, 'o', alpha=0.5, ms=2, color='green')
    ax.axhline(np.mean(diff_baseline_pint), color='r', ls='--',
               label=f'Mean = {np.mean(diff_baseline_pint):.3f} μs')
    ax.axhline(0, color='k', ls='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('MJD (TDB)')
    ax.set_ylabel('Baseline - PINT (μs)')
    ax.set_title(f'Baseline - PINT (std={np.std(diff_baseline_pint):.3f} μs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    # Histogram of JAX - Baseline
    ax = axes[1, 0]
    ax.hist(diff_jax_baseline, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(diff_jax_baseline), color='r', ls='--', linewidth=2,
               label=f'Mean = {np.mean(diff_jax_baseline):.3f} μs')
    ax.axvline(0, color='k', ls='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('JAX - Baseline (μs)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram: JAX - Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Panel 4: JAX - PINT (if available)
if has_pint:
    ax = axes[1, 1]
    ax.plot(tdb_mjd, diff_jax_pint, 'o', alpha=0.5, ms=2, color='purple')
    ax.axhline(np.mean(diff_jax_pint), color='r', ls='--',
               label=f'Mean = {np.mean(diff_jax_pint):.3f} μs')
    ax.axhline(0, color='k', ls='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('MJD (TDB)')
    ax.set_ylabel('JAX - PINT (μs)')
    ax.set_title(f'JAX - PINT (std={np.std(diff_jax_pint):.3f} μs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    # Scatter: JAX vs Baseline
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

# Panel 5: Histogram comparison
if has_pint:
    ax = axes[2, 0]
    bins = np.linspace(-1, 1, 50)
    ax.hist(diff_jax_baseline, bins=bins, alpha=0.5, label='JAX-Baseline', edgecolor='black')
    ax.hist(diff_baseline_pint, bins=bins, alpha=0.5, label='Baseline-PINT', edgecolor='black')
    ax.hist(diff_jax_pint, bins=bins, alpha=0.5, label='JAX-PINT', edgecolor='black')
    ax.axvline(0, color='k', ls='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Difference (μs)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of All Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: 3-way scatter
    ax = axes[2, 1]
    # Plot all three against each other
    from mpl_toolkits.mplot3d import Axes3D
    # Actually just do 2D scatter of differences
    ax.plot(diff_baseline_pint, diff_jax_baseline, 'o', alpha=0.3, ms=2)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axvline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Baseline - PINT (μs)')
    ax.set_ylabel('JAX - Baseline (μs)')
    ax.set_title('Difference Correlation')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
filename = 'three_way_comparison.png' if has_pint else 'jax_vs_baseline_comparison.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {filename}")
