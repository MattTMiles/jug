"""Compare JUG vs PINT binary delays for J1022+1001 to find the orbital-phase error."""

import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs

PAR = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001_tdb.par'
TIM = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001.tim'

print("Loading PINT model and TOAs...")
model = get_model(PAR)
toas = get_TOAs(TIM, planets=True, ephem='DE440', include_bipm=True, bipm_version='BIPM2024')

# For now, just analyze the residual difference pattern
print("Computing JUG residuals...")
jug_result = compute_residuals_simple(PAR, TIM, clock_dir='data/clock')
jug_res = jug_result['residuals_us']

from pint.residuals import Residuals
res_pint = Residuals(toas, model)
pint_res = res_pint.time_resids.to_value('us')

diff_us = jug_res - pint_res

# Get orbital phase for plotting
mjds = toas.get_mjds().value
pb = 7.8051347940537224349  # days
t0 = 51652.7584089621  # From par file
orbital_phase = ((mjds - t0) / pb) % 1.0

print(f"\nResidual difference vs PINT:")
print(f"  Mean: {np.mean(diff_us):.6f} μs")
print(f"  RMS:  {np.std(diff_us):.6f} μs")
print(f"  Max:  {np.max(np.abs(diff_us)):.6f} μs")

# Plot difference vs orbital phase
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Panel 1: Difference vs time
ax = axes[0]
ax.plot(mjds, diff_us * 1000, 'g.', markersize=2, alpha=0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('MJD')
ax.set_ylabel('JUG - PINT (ns)')
ax.set_title('Residual Difference vs Time')
ax.grid(True, alpha=0.3)

# Panel 2: Difference vs orbital phase
ax = axes[1]
ax.plot(orbital_phase, diff_us * 1000, 'b.', markersize=3, alpha=0.6)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Orbital Phase')
ax.set_ylabel('JUG - PINT (ns)')
ax.set_title('Residual Difference vs Orbital Phase (should be flat if correct)')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('j1022_orbital_phase_error.png', dpi=150)
print("\nPlot saved to: j1022_orbital_phase_error.png")

# Check if there's a systematic pattern
# Bin by orbital phase
n_bins = 50
phase_bins = np.linspace(0, 1, n_bins + 1)
binned_diff = []
bin_centers = []
for i in range(n_bins):
    mask = (orbital_phase >= phase_bins[i]) & (orbital_phase < phase_bins[i+1])
    if np.sum(mask) > 0:
        binned_diff.append(np.mean(diff_us[mask]))
        bin_centers.append(0.5 * (phase_bins[i] + phase_bins[i+1]))

binned_diff = np.array(binned_diff)
bin_centers = np.array(bin_centers)

print(f"\nOrbital phase binned analysis:")
print(f"  Peak-to-peak amplitude: {np.max(binned_diff) - np.min(binned_diff):.3f} μs")
print(f"  Amplitude: {0.5*(np.max(binned_diff) - np.min(binned_diff)):.3f} μs")

# Try to identify the harmonic
from scipy.signal import lombscargle
freqs = np.linspace(0.5, 3.0, 1000)
pgram = lombscargle(orbital_phase * 2 * np.pi, diff_us - np.mean(diff_us), freqs)
peak_freq = freqs[np.argmax(pgram)]
print(f"  Dominant frequency: {peak_freq:.3f} cycles/orbit (should be 1.0 for single orbit)")
print(f"  Period: {1.0/peak_freq:.3f} orbits")
