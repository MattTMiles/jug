#!/usr/bin/env python3
"""Create detailed JUG vs PINT comparison matching original plot format."""

import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.tim_reader import parse_tim_file_mjds

import pint.models
import pint.toa
import pint.residuals
import astropy.units as u

# File paths
PAR_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
TIM_FILE = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("Computing JUG and PINT residuals...")

# Compute JUG residuals
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']
jug_rms = jug_result['rms_us']

# Compute PINT residuals  
pint_model = pint.models.get_model(PAR_FILE)
pint_toas = pint.toa.get_TOAs(TIM_FILE, model=pint_model)
pint_res = pint.residuals.Residuals(pint_toas, pint_model, use_weighted_mean=False)
pint_res_us = pint_res.time_resids.to(u.us).value
pint_rms = np.sqrt(np.mean(pint_res_us**2))

# Get MJDs
toas = parse_tim_file_mjds(TIM_FILE)
mjds = np.array([t.mjd_int + t.mjd_frac for t in toas])

# Compute difference
diff_us = jug_res_us - pint_res_us

# Create plot matching original format
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top panel: Both residuals
ax1.plot(mjds, jug_res_us, 'o', ms=2, alpha=0.6, label=f'JUG (RMS={jug_rms:.3f} μs)', color='blue')
ax1.plot(mjds, pint_res_us, 'x', ms=2, alpha=0.6, label=f'PINT (RMS={pint_rms:.3f} μs)', color='orange')
ax1.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax1.set_xlabel('MJD', fontsize=12)
ax1.set_ylabel('Residual (μs)', fontsize=12)
ax1.set_title('J1909-3744: JUG vs PINT Residuals (ELL1 Binary Model)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom panel: Difference
mean_diff = np.mean(diff_us)
std_diff = np.std(diff_us)
ax2.plot(mjds, diff_us * 1000, 'o', ms=2, alpha=0.6, color='red')  # Convert to ns
ax2.axhline(mean_diff * 1000, color='red', ls='--', lw=1, alpha=0.7, label=f'Mean={mean_diff*1000:.3f} ns')
ax2.axhspan((mean_diff - std_diff) * 1000, (mean_diff + std_diff) * 1000, 
            color='gray', alpha=0.2, label=f'±1σ={std_diff*1000:.3f} ns')
ax2.set_xlabel('MJD', fontsize=12)
ax2.set_ylabel('Residual Difference (JUG - PINT) [ns]', fontsize=12)
ax2.set_title('Residual Difference', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('J1909-3744_jug_vs_pint_CURRENT.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: J1909-3744_jug_vs_pint_CURRENT.png")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"JUG  RMS:  {jug_rms:.3f} μs")
print(f"PINT RMS:  {pint_rms:.3f} μs")
print(f"\nDifference (JUG - PINT):")
print(f"  Mean:  {mean_diff*1000:.3f} ns")
print(f"  Std:   {std_diff*1000:.3f} ns") 
print(f"  Range: {(np.max(diff_us) - np.min(diff_us))*1000:.3f} ns")
print(f"{'='*60}")
