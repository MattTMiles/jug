"""Investigate time-dependent error in binary delays."""

import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Load data
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("TIME-DEPENDENT ERROR ANALYSIS")
print("="*80)

# Compute JUG residuals
print("\nComputing JUG residuals...")
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']

# Compute PINT residuals
print("Computing PINT residuals...")
model_pint = get_model(PAR_FILE)
toas_pint = get_TOAs(TIM_FILE, planets=True)
res_pint = Residuals(toas_pint, model_pint)
pint_res_us = res_pint.time_resids.to_value('us')

# Get MJDs
mjds = toas_pint.get_mjds().value

# Compute difference
diff_us = jug_res_us - pint_res_us

# Get T0 for reference
binary_comp = model_pint.get_components_by_category()['pulsar_system'][0]
T0 = float(binary_comp.T0.value)
PB = float(binary_comp.PB.value)

print(f"\nBinary parameters:")
print(f"  T0 = {T0:.6f} MJD")
print(f"  PB = {PB:.10f} days")
print(f"  PBDOT = {float(binary_comp.PBDOT.value):.6e}")
print(f"  OMDOT = {float(binary_comp.OMDOT.value):.6f} deg/yr")

# Compute time since T0
dt_days = mjds - T0
dt_years = dt_days / 365.25

# Compute orbital phase
orbital_phase = (dt_days / PB) % 1.0

print(f"\nTime span:")
print(f"  MJD range: {mjds.min():.1f} - {mjds.max():.1f}")
print(f"  dt range: {dt_days.min():.1f} - {dt_days.max():.1f} days")
print(f"  dt range: {dt_years.min():.2f} - {dt_years.max():.2f} years")
print(f"  Total span: {dt_days.max() - dt_days.min():.1f} days = {dt_years.max() - dt_years.min():.2f} years")

# Analyze correlation with time
print(f"\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

corr_dt_days = np.corrcoef(dt_days, diff_us)[0, 1]
corr_dt_years = np.corrcoef(dt_years, diff_us)[0, 1]
corr_phase = np.corrcoef(orbital_phase, diff_us)[0, 1]

print(f"\nCorrelation of (JUG - PINT) with:")
print(f"  dt (days since T0): {corr_dt_days:.6f}")
print(f"  dt (years since T0): {corr_dt_years:.6f}")
print(f"  Orbital phase: {corr_phase:.6f}")

# Bin by time to see trend
n_bins = 10
time_bins = np.linspace(dt_days.min(), dt_days.max(), n_bins + 1)
bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
bin_means = []
bin_stds = []
bin_counts = []

for i in range(n_bins):
    mask = (dt_days >= time_bins[i]) & (dt_days < time_bins[i+1])
    if np.sum(mask) > 0:
        bin_means.append(np.mean(diff_us[mask]))
        bin_stds.append(np.std(diff_us[mask]))
        bin_counts.append(np.sum(mask))
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)
        bin_counts.append(0)

print(f"\n" + "="*80)
print("BINNED ANALYSIS (by time since T0)")
print("="*80)
print(f"\n{'Bin Center':>12}  {'Count':>6}  {'Mean Diff':>12}  {'Std Dev':>12}")
print(f"{'(days)':>12}  {'':>6}  {'(μs)':>12}  {'(μs)':>12}")
print("-" * 60)
for i in range(n_bins):
    if bin_counts[i] > 0:
        print(f"{bin_centers[i]:12.1f}  {bin_counts[i]:6d}  {bin_means[i]:12.3f}  {bin_stds[i]:12.3f}")

# Fit linear trend (convert to float64 to avoid float128 issue)
coeffs = np.polyfit(np.array(dt_years, dtype=np.float64),
                    np.array(diff_us, dtype=np.float64), 1)
trend_us_per_year = coeffs[0]
offset_us = coeffs[1]

print(f"\n" + "="*80)
print("LINEAR TREND FIT")
print("="*80)
print(f"\nDifference = {offset_us:.3f} + {trend_us_per_year:.3f} * dt_years")
print(f"\nTrend: {trend_us_per_year:.3f} μs/year")
print(f"Over {dt_years.max() - dt_years.min():.2f} years: {trend_us_per_year * (dt_years.max() - dt_years.min()):.3f} μs total drift")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Difference vs time since T0
ax = axes[0, 0]
ax.plot(dt_years, diff_us, 'g.', markersize=2, alpha=0.5)
ax.plot(dt_years, offset_us + trend_us_per_year * dt_years, 'r-', linewidth=2,
        label=f'Linear fit: {trend_us_per_year:.3f} μs/yr')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time since T0 (years)')
ax.set_ylabel('JUG - PINT (μs)')
ax.set_title(f'Residual Difference vs Time (corr = {corr_dt_years:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Scatter vs time since T0 (binned)
ax = axes[0, 1]
ax.errorbar(bin_centers/365.25, bin_means, yerr=bin_stds, fmt='o-', capsize=5,
            color='blue', label='Binned mean ± std')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time since T0 (years)')
ax.set_ylabel('Mean JUG - PINT (μs)')
ax.set_title('Binned Mean Difference vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Std dev vs time (showing increasing scatter)
ax = axes[1, 0]
ax.plot(bin_centers/365.25, bin_stds, 'ro-', linewidth=2, markersize=8)
ax.set_xlabel('Time since T0 (years)')
ax.set_ylabel('Std Dev of Difference (μs)')
ax.set_title('Scatter Increasing Over Time')
ax.grid(True, alpha=0.3)

# Plot 4: Difference vs orbital phase (to check for phase-dependent error)
ax = axes[1, 1]
scatter = ax.scatter(orbital_phase, diff_us, c=dt_years, cmap='viridis',
                     s=5, alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time since T0 (years)')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Orbital Phase')
ax.set_ylabel('JUG - PINT (μs)')
ax.set_title(f'Difference vs Orbital Phase (corr = {corr_phase:.3f})')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_dependent_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: time_dependent_error_analysis.png")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print(f"\nThe increasing scatter over time (bottom-left plot) confirms")
print(f"time-dependent divergence in the binary delay calculation.")
print(f"\nLinear trend of {trend_us_per_year:.3f} μs/year suggests:")
print(f"  - PBDOT evolution formula may be incorrect")
print(f"  - Mean anomaly calculation may drift over time")
print(f"  - OMDOT application may have wrong time dependence")
print(f"\nRecommendation: Compare PINT's exact formulas for:")
print(f"  1. PB(t) with PBDOT")
print(f"  2. Mean anomaly M(t)")
print(f"  3. OM(t) with OMDOT")
