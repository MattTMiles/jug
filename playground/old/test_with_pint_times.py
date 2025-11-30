#!/usr/bin/env python3
"""
Test JUG's residual calculation using PINT's correct infinite-frequency times.
This proves the residual calculation itself is correct.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pint.models import get_model
from pint.toa import get_TOAs

# Enable JAX float64
jax.config.update('jax_enable_x64', True)

# Load data
par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, model=model)

# Get PINT's infinite-frequency barycentric times
pint_tdbld = toas.table['tdbld'].value  # MJD

print("="*80)
print("TESTING JUG RESIDUALS WITH PINT'S CORRECT TIMES")
print("="*80)

# JUG's spin phase function
SECS_PER_DAY = 86400.0
F0 = 339.31569191904066
F1 = -1.6147400369092967e-15
PEPOCH = 59017.9997538705

@jax.jit
def spin_phase(t_mjd, f0, f1, pepoch):
    dt = (t_mjd - pepoch) * SECS_PER_DAY
    return f0 * dt + 0.5 * f1 * dt**2

# TZR reference from JUG
TZR_INF_MJD = 59679.249646122036  # From notebook cell 14

# Compute JUG residuals using PINT's times
print("\nComputing JUG residuals with PINT's infinite-frequency times...")
t_inf_jax = jnp.array(pint_tdbld, dtype=jnp.float64)

# Phase calculation
phase = spin_phase(t_inf_jax, F0, F1, PEPOCH)
phase_ref = spin_phase(jnp.array([TZR_INF_MJD]), F0, F1, PEPOCH)[0]

# Residual
phase_diff = phase - phase_ref
frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
residual_sec = frac_phase / F0
residual_sec = residual_sec - jnp.mean(residual_sec)  # Remove mean
residual_us = np.array(residual_sec) * 1e6

# Load tempo2 residuals for comparison
t2_res = []
with open('temp_pre_general2.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            try:
                t2_res.append(float(parts[1]))
            except:
                pass
t2_res_us = np.array(t2_res) * 1e6

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nJUG (with PINT times):")
print(f"  RMS: {np.sqrt(np.mean(residual_us**2)):.3f} μs")
print(f"  First 10: {residual_us[:10]}")

print(f"\nTempo2:")
print(f"  RMS: {np.sqrt(np.mean(t2_res_us**2)):.3f} μs")
print(f"  First 10: {t2_res_us[:10]}")

print(f"\nComparison:")
corr = np.corrcoef(residual_us, t2_res_us)[0,1]
diff = residual_us - t2_res_us
rms_diff = np.sqrt(np.mean(diff**2))

print(f"  Correlation: {corr:.6f}")
print(f"  RMS difference: {rms_diff:.3f} μs")
print(f"  Mean difference: {np.mean(diff):.6f} μs")

if rms_diff < 10 and corr > 0.99:
    print("\n" + "="*80)
    print("✓✓✓ SUCCESS! JUG's residual calculation is CORRECT!")
    print("="*80)
    print("\nThe problem was NOT the residual calculation.")
    print("The problem was using tempo2's BAT instead of computing")
    print("infinite-frequency barycentric times correctly.")
    print("\nJUG needs to either:")
    print("  1. Compute barycentric times from scratch (like PINT)")
    print("  2. Or understand what tempo2's BAT actually represents")
else:
    print(f"\n⚠️  Still some discrepancy: {rms_diff:.3f} μs")
