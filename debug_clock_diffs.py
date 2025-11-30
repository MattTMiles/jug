#!/usr/bin/env python3
"""Compare JUG and PINT clock corrections to debug time-correlated residual trends."""

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.core import TimingCalculator
import numpy as np
import matplotlib.pyplot as plt
import pint.models
import pint.toa
from astropy import units as u

# Load data
print("Loading data...")
params = parse_par_file('data/J1909-3744.par')
toa_list = parse_tim_file_mjds('data/J1909-3744.tim')
mjds = np.array([t.mjd_utc for t in toa_list])
obs_codes = np.array([t.obs_code for t in toa_list])
print(f"✓ Loaded {len(mjds)} TOAs")

# Get JUG clock corrections
print("Computing JUG clock corrections...")
calc = TimingCalculator(params)
jug_clock = calc.compute_clock_corrections(mjds, obs_codes)
print(f"✓ JUG: {len(jug_clock)} corrections")

# Get PINT clock corrections
print("Computing PINT clock corrections...")
pint_model = pint.models.get_model('data/J1909-3744.par')
pint_toas = pint.toa.get_TOAs('data/J1909-3744.tim')
pint_clock = pint_toas.get_clock_corrections().to(u.s).value
print(f"✓ PINT: {len(pint_clock)} corrections")

# Plot comparison
print("Generating plot...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
mjd = mjds

axes[0].plot(mjd, np.array(jug_clock)*1e6, 'o', label='JUG', alpha=0.6, ms=2)
axes[0].plot(mjd, pint_clock*1e6, 'x', label='PINT', alpha=0.6, ms=2)
axes[0].set_ylabel('Clock Correction (μs)')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_title('Observatory Clock Corrections: JUG vs PINT')

diff_ns = (np.array(jug_clock) - pint_clock) * 1e9
axes[1].plot(mjd, diff_ns, 'o', ms=2)
axes[1].set_ylabel('Difference (ns): JUG - PINT')
axes[1].set_xlabel('MJD')
axes[1].axhline(0, color='r', ls='--', alpha=0.5)
axes[1].grid(alpha=0.3)
axes[1].set_title(f'Clock Difference (RMS={np.std(diff_ns):.2f} ns, Max={np.max(np.abs(diff_ns)):.2f} ns)')

plt.tight_layout()
plt.savefig('clock_correction_comparison.png', dpi=150)

print("\n" + "="*60)
print("✓ Saved: clock_correction_comparison.png")
print("="*60)
print(f"Clock Difference Statistics:")
print(f"  Mean = {np.mean(diff_ns):.3f} ns")
print(f"  Std  = {np.std(diff_ns):.3f} ns")
print(f"  Min  = {np.min(diff_ns):.3f} ns")
print(f"  Max  = {np.max(diff_ns):.3f} ns")
print("="*60)
