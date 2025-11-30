"""
Debug script to find source of time-correlated residual difference between JUG and PINT
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import pint.toa as toa
import pint.models as models

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple

# File paths
PAR_FILE = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
TIM_FILE = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

print("="*70)
print("DIAGNOSING TIME-CORRELATED JUG vs PINT DIFFERENCE")
print("="*70)

# Load with JUG
print("\nLoading with JUG...")
result = compute_residuals_simple(PAR_FILE, TIM_FILE)
res_jug = result['residuals_us']  # already in microseconds

# Load with PINT
print("Loading with PINT...")
toas_pint = toa.get_TOAs(TIM_FILE, planets=True, ephem='de440')  # Need planets=True for Shapiro delay
model_pint = models.get_model(PAR_FILE)
from pint.residuals import Residuals
res_obj = Residuals(toas_pint, model_pint)
res_pint = res_obj.time_resids.to(u.us).value

# Get MJDs
mjds = toas_pint.get_mjds().value

# Compute difference
diff = res_jug - res_pint

print(f"\nMJD range: {mjds.min():.2f} to {mjds.max():.2f}")
print(f"Difference RMS: {np.std(diff):.3f} μs")
print(f"Difference mean: {np.mean(diff):.3f} μs")

# Split into time bins to see trend
n_bins = 10
bin_edges = np.linspace(mjds.min(), mjds.max(), n_bins+1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_means = []
bin_stds = []

for i in range(n_bins):
    mask = (mjds >= bin_edges[i]) & (mjds < bin_edges[i+1])
    if np.sum(mask) > 0:
        bin_means.append(np.mean(diff[mask]))
        bin_stds.append(np.std(diff[mask]))
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)

print("\n" + "="*70)
print("TIME BINNED ANALYSIS:")
print("="*70)
for i, (center, mean, std) in enumerate(zip(bin_centers, bin_means, bin_stds)):
    print(f"Bin {i+1}: MJD {center:.1f} | Mean diff: {mean:+.3f} μs | Std: {std:.3f} μs")

# Check if there's a linear trend
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(mjds, dtype=float), np.array(diff, dtype=float))
print(f"\nLinear trend analysis:")
print(f"  Slope: {slope:.6f} μs/day")
print(f"  Total drift over {mjds.max()-mjds.min():.0f} days: {slope*(mjds.max()-mjds.min()):.3f} μs")
print(f"  R²: {r_value**2:.4f}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Top: residual difference vs time
ax = axes[0]
ax.scatter(mjds, diff, s=1, alpha=0.5, label='JUG - PINT')
ax.plot(mjds, slope*mjds + intercept, 'r-', lw=2, label=f'Linear fit (slope={slope:.6f} μs/day)')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('MJD')
ax.set_ylabel('Residual Difference (μs)')
ax.set_title(f'JUG - PINT: Time-correlated difference (RMS={np.std(diff):.3f} μs)')
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom: binned analysis
ax = axes[1]
ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, markersize=8)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('MJD')
ax.set_ylabel('Mean Difference (μs)')
ax.set_title('Binned Analysis: Mean ± Std in 10 time bins')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_time_trend.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Plot saved: debug_time_trend.png")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Check if EOP file coverage - TOAs may extend beyond EOP data")
print("2. Check clock correction files - may have different versions/coverage")
print("3. Check if PINT uses different interpolation for EOP parameters")
print("="*70)
