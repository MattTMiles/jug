#!/usr/bin/env python3
"""
Check what TZR time PINT actually uses.
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs

par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, model=model)

print("="*80)
print("CHECKING PINT'S TZR HANDLING")
print("="*80)

# Check if model has absolute phase component
if hasattr(model, 'TZRMJD'):
    print(f"\n✓ Model has TZRMJD: {model.TZRMJD.value}")
    print(f"  TZRFRQ: {model.TZRFRQ.value} MHz")
    print(f"  TZRSITE: {model.TZRSITE.value}")

print("\n" + "="*80)
print("CHECKING PHASE AT TZR")
print("="*80)

F0 = model.F0.value
F1 = model.F1.value
PEPOCH = model.PEPOCH.value
SECS_PER_DAY = 86400.0

# Find the TOA closest to TZRMJD
tzrmjd = model.TZRMJD.value
topo_mjds = toas.table['mjd_float']  # Use float version
idx_closest = np.argmin(np.abs(topo_mjds - tzrmjd))

print(f"\nTZRMJD from .par: {tzrmjd}")
print(f"Closest TOA (topocentric): {topo_mjds[idx_closest]}")
print(f"  Index: {idx_closest}")
print(f"  Difference: {(topo_mjds[idx_closest] - tzrmjd) * 86400:.6f} seconds")

# Get that TOA's barycentric time
tzr_tdbld = toas.table['tdbld'][idx_closest]
print(f"  Barycentric time (tdbld): {tzr_tdbld}")

# Compute phase at this time
dt = (tzr_tdbld - PEPOCH) * SECS_PER_DAY
phase = F0 * dt + 0.5 * F1 * dt**2
frac_phase = np.mod(phase + 0.5, 1.0) - 0.5

print(f"\nPhase at TZR (according to PINT):")
print(f"  Absolute phase: {phase:.6f} cycles")
print(f"  Fractional phase: {frac_phase:.9f} cycles")
print(f"  Time equivalent: {frac_phase / F0 * 1e6:.3f} μs")

# JUG's values
jug_tzr_inf = 59679.249646122036
jug_phase_offset = -0.09033203125

print(f"\n" + "="*80)
print("COMPARISON: PINT vs JUG TZR")
print("="*80)

print(f"\nTZR infinite-frequency time:")
print(f"  JUG:  {jug_tzr_inf}")
print(f"  PINT: {tzr_tdbld}")
print(f"  Diff: {(tzr_tdbld - jug_tzr_inf) * 86400:.9f} seconds")
print(f"        {(tzr_tdbld - jug_tzr_inf) * 86400 * 1e6:.3f} μs")

print(f"\nFractional phase at TZR:")
print(f"  JUG:  {jug_phase_offset:.9f} cycles")
print(f"  PINT: {frac_phase:.9f} cycles")
print(f"  Diff: {abs(frac_phase - jug_phase_offset):.9f} cycles")
print(f"        {abs(frac_phase - jug_phase_offset) / F0 * 1e6:.3f} μs")

if abs((tzr_tdbld - jug_tzr_inf) * 86400) > 1.0:
    print(f"\n✗ JUG's TZR barycentric time is WRONG by {(tzr_tdbld - jug_tzr_inf) * 86400:.3f} seconds!")
    print(f"  This is the root cause of the residual discrepancy.")
else:
    print(f"\n✓ TZR times match")
