"""
Test the CORRECTED piecewise formula with f0_global for normalization.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from pathlib import Path
import matplotlib.pyplot as plt

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble

SECS_PER_DAY = 86400.0

par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

params = parse_par_file(par_file)
pepoch_mjd = float(get_longdouble(params, 'PEPOCH'))
f0 = float(params['F0'])
f1 = float(params['F1'])

result = compute_residuals_simple(par_file, tim_file, clock_dir="data/clock", subtract_tzr=True, verbose=False)
dt_sec = result['dt_sec']
tdb_mjd = result['tdb_mjd']
residuals_standard_us = result['residuals_us']

def create_time_segments(tdb_mjd, segment_duration_days=500.0):
    t_min = np.min(tdb_mjd)
    t_max = np.max(tdb_mjd)
    n_segments = max(1, int(np.ceil((t_max - t_min) / segment_duration_days)))
    
    segments = []
    for i in range(n_segments):
        seg_start = t_min + i * segment_duration_days
        seg_end = t_min + (i + 1) * segment_duration_days
        
        mask = (tdb_mjd >= seg_start) & (tdb_mjd < seg_end)
        if i == n_segments - 1:
            mask = (tdb_mjd >= seg_start) & (tdb_mjd <= seg_end)
        
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
            
        seg_times = tdb_mjd[indices]
        segments.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(seg_times)),
            'tmin_mjd': float(np.min(seg_times)),
            'tmax_mjd': float(np.max(seg_times)),
            'n_toas': len(indices)
        })
    
    return segments

segments = create_time_segments(tdb_mjd, segment_duration_days=500.0)

# CORRECTED piecewise computation
n_toas = len(dt_sec)
phase_piecewise = np.zeros(n_toas)

for seg in segments:
    idx = seg['indices']
    dt_epoch = (seg['local_pepoch_mjd'] - pepoch_mjd) * SECS_PER_DAY
    f0_local = f0 + f1 * dt_epoch
    dt_local = dt_sec[idx] - dt_epoch
    
    phase_local = dt_local * (f0_local + dt_local * (f1 / 2.0))
    
    # Phase offset with longdouble
    dt_epoch_ld = np.longdouble(dt_epoch)
    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)
    phase_offset = float(f0_ld * dt_epoch_ld + (f1_ld / 2.0) * dt_epoch_ld**2)
    
    phase_corrected = phase_local + phase_offset
    phase_wrapped = phase_corrected - np.round(phase_corrected)
    
    phase_piecewise[idx] = phase_wrapped

# Subtract TZR
tzr_phase_scalar = result['tzr_phase']
phase_piecewise_minus_tzr = phase_piecewise - tzr_phase_scalar

# Convert to residuals using F0_GLOBAL (not f0_local!)
residuals_piecewise_us = (phase_piecewise_minus_tzr / f0) * 1e6

print("="*80)
print("CORRECTED PIECEWISE FORMULA TEST")
print("="*80)
print(f"\nStandard residuals:")
print(f"  Mean: {np.mean(residuals_standard_us):.6f} μs")
print(f"  RMS:  {np.std(residuals_standard_us):.6f} μs")
print()

print(f"Piecewise residuals (CORRECTED):")
print(f"  Mean: {np.mean(residuals_piecewise_us):.6f} μs")
print(f"  RMS:  {np.std(residuals_piecewise_us):.6f} μs")
print()

diff_us = residuals_piecewise_us - residuals_standard_us

print(f"DIFFERENCE (piecewise - standard):")
print(f"  Mean: {np.mean(diff_us):.6f} μs")
print(f"  RMS:  {np.std(diff_us):.6f} μs")
print(f"  Range: [{np.min(diff_us):.6f}, {np.max(diff_us):.6f}] μs")
print(f"  Peak-to-peak: {np.max(diff_us) - np.min(diff_us):.6f} μs")
print()

# Check for quadratic trend
dt_days = tdb_mjd - pepoch_mjd
coeffs = np.polyfit(dt_days, diff_us, 2)
poly_fit = np.polyval(coeffs, dt_days)
ss_res = np.sum((diff_us - poly_fit)**2)
ss_tot = np.sum((diff_us - np.mean(diff_us))**2)
r_squared = 1 - (ss_res / ss_tot)

print(f"Quadratic fit test:")
print(f"  R² = {r_squared:.6f}")

if r_squared < 0.1:
    print(f"  ✓ NO quadratic trend (R² < 0.1) - FIX WORKS!")
else:
    print(f"  ✗ Still quadratic (R² = {r_squared:.3f})")
print()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(tdb_mjd, residuals_standard_us, 'b.', markersize=1, alpha=0.5)
axes[0].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[0].set_ylabel('Residual (μs)')
axes[0].set_title('Standard Residuals')
axes[0].grid(True, alpha=0.3)

axes[1].plot(tdb_mjd, residuals_piecewise_us, 'r.', markersize=1, alpha=0.5)
axes[1].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Residual (μs)')
axes[1].set_title('Piecewise Residuals (CORRECTED)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(tdb_mjd, diff_us * 1e3, 'g.', markersize=2, alpha=0.5, label='Difference')
axes[2].axhline(0, color='k', linestyle='-', linewidth=0.5)
axes[2].set_xlabel('MJD (TDB)')
axes[2].set_ylabel('Difference (ns)')
axes[2].set_title(f'Piecewise - Standard (Peak-to-peak: {(np.max(diff_us)-np.min(diff_us))*1e3:.1f} ns)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

for seg in segments[1:]:
    for ax in axes:
        ax.axvline(seg['tmin_mjd'], color='orange', linestyle=':', alpha=0.2, linewidth=1)

plt.tight_layout()
plt.savefig('piecewise_FIXED.png', dpi=150, bbox_inches='tight')
print("Saved: piecewise_FIXED.png")
