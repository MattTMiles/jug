"""Analyze the periodic component in JUG-PINT residual difference."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Load data
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("PERIODIC COMPONENT ANALYSIS")
print("="*80)

# Compute residuals
print("\nComputing JUG and PINT residuals...")
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']

model_pint = get_model(PAR_FILE)
toas_pint = get_TOAs(TIM_FILE, planets=True)
res_pint = Residuals(toas_pint, model_pint)
pint_res_us = res_pint.time_resids.to_value('us')

# Get times
mjds = toas_pint.get_mjds().value
binary_comp = model_pint.get_components_by_category()['pulsar_system'][0]
T0 = float(binary_comp.T0.value)
PB_days = float(binary_comp.PB.value)

dt_days = mjds - T0
dt_years = dt_days / 365.25

# Compute difference
diff_us = jug_res_us - pint_res_us

# Remove linear trend to isolate periodic component
coeffs = np.polyfit(dt_years.astype(np.float64), diff_us.astype(np.float64), 1)
linear_trend = coeffs[0] * dt_years + coeffs[1]
detrended = diff_us - linear_trend

print(f"\nLinear trend removed: {coeffs[0]:.3f} μs/year + {coeffs[1]:.3f} μs")
print(f"Detrended RMS: {np.std(detrended):.3f} μs")

# Compute periodogram
print(f"\n" + "="*80)
print("FOURIER ANALYSIS")
print("="*80)

# Sort by time for spectral analysis
sort_idx = np.argsort(dt_days)
t_sorted = dt_days[sort_idx]
y_sorted = detrended[sort_idx]

# Compute Lomb-Scargle periodogram (handles uneven sampling)
freq_min = 1.0 / (2 * 365.25)  # 2 years
freq_max = 1.0 / 10.0  # 10 days
frequencies = np.linspace(freq_min, freq_max, 10000)

pgram = signal.lombscargle(t_sorted, y_sorted, 2*np.pi*frequencies, normalize=True)

# Find peak
peak_idx = np.argmax(pgram)
peak_freq = frequencies[peak_idx]
peak_period_days = 1.0 / peak_freq
peak_power = pgram[peak_idx]

print(f"\nStrongest periodic signal:")
print(f"  Period: {peak_period_days:.2f} days = {peak_period_days/365.25:.3f} years")
print(f"  Frequency: {peak_freq:.6f} cycles/day")
print(f"  Power: {peak_power:.4f}")

# Compare to known periods
print(f"\nComparison to known periods:")
print(f"  Orbital period (PB): {PB_days:.2f} days")
print(f"  Annual: 365.25 days")
print(f"  Ratio to PB: {peak_period_days/PB_days:.2f}×")
print(f"  Ratio to year: {peak_period_days/365.25:.2f}×")

# Find other significant peaks
peak_threshold = 0.1 * peak_power
significant_peaks = np.where(pgram > peak_threshold)[0]
periods_significant = 1.0 / frequencies[significant_peaks]

print(f"\nOther significant periods (power > {peak_threshold:.3f}):")
for i, (idx, period) in enumerate(zip(significant_peaks, periods_significant)):
    if i < 10 and pgram[idx] < peak_power:  # Skip the main peak
        print(f"  {period:.2f} days ({period/PB_days:.2f}× PB, power={pgram[idx]:.4f})")

# Plot results
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Detrended difference vs time
ax = axes[0]
ax.plot(dt_years, detrended, 'g.', markersize=2, alpha=0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time since T0 (years)')
ax.set_ylabel('Detrended Difference (μs)')
ax.set_title(f'Residual Difference After Removing Linear Trend (RMS = {np.std(detrended):.3f} μs)')
ax.grid(True, alpha=0.3)

# Plot 2: Periodogram
ax = axes[1]
ax.plot(1.0/frequencies, pgram, 'b-', linewidth=0.5)
ax.axvline(peak_period_days, color='r', linestyle='--', linewidth=2,
           label=f'Peak: {peak_period_days:.1f} days')
ax.axvline(PB_days, color='orange', linestyle='--', linewidth=2,
           label=f'Orbital: {PB_days:.1f} days')
ax.axvline(365.25, color='green', linestyle='--', linewidth=2,
           label='Annual: 365.25 days')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Lomb-Scargle Power')
ax.set_title('Periodogram of Detrended Difference')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(10, 2*365)

# Plot 3: Phase-folded at peak period
ax = axes[2]
phase = (dt_days % peak_period_days) / peak_period_days
ax.plot(phase, detrended, 'g.', markersize=3, alpha=0.5)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel(f'Phase (P = {peak_period_days:.1f} days)')
ax.set_ylabel('Detrended Difference (μs)')
ax.set_title(f'Folded at Peak Period')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('oscillation_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: oscillation_analysis.png")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if abs(peak_period_days - PB_days) < 1.0:
    print(f"\n✓ Peak period matches orbital period (PB = {PB_days:.2f} days)")
    print("  This suggests an error in the binary delay calculation.")
elif abs(peak_period_days - 365.25) < 30:
    print(f"\n✓ Peak period matches annual period (365.25 days)")
    print("  This suggests an error in annual effects (parallax, solar wind, etc.)")
elif abs(peak_period_days - 2*PB_days) < 2.0:
    print(f"\n✓ Peak period is 2× orbital period")
    print("  This could be Shapiro delay (goes as sin²)")
else:
    print(f"\n? Peak period ({peak_period_days:.1f} days) doesn't match obvious candidates")
    print(f"  Investigate other possibilities...")
