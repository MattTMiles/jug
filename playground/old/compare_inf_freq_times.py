#!/usr/bin/env python3
"""
Compare JUG's infinite-frequency times with PINT's tdbld.
"""

import numpy as np
import pint
from pint.models import get_model
from pint.toa import get_TOAs

# Load with PINT
par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

print("Loading data...")
model = get_model(par_file)
toas = get_TOAs(tim_file, model=model)

# Get PINT's infinite-frequency barycentric times
pint_tdbld = toas.table['tdbld'].value  # MJD in TDB
pint_freq = toas.table['freq'].value  # MHz

print(f"✓ Loaded {len(pint_tdbld)} TOAs")

# Load JUG's computed infinite-frequency times
# These should be in the notebook as t_inf_mjd
# For now, let's load tempo2 BAT and JUG's delays from the notebook output

# From the notebook output cell 15:
# Binary delays: [0.38488819, 0.38488819, ...]
# DM delays: [0.05230413, 0.04997496, ...]

# Load tempo2 BAT
t2_bat = []
with open('temp_pre_components_next.out') as f:
    for line in f:
        if any(x in line for x in ['Starting', '[', 'This', 'Looking', 'under', 'conditions']):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                t2_bat.append(float(parts[1]))
            except:
                pass
t2_bat = np.array(t2_bat)

# Compute JUG's delays (replicate the calculation)
SECS_PER_DAY = 86400.0
K_DM_SEC = 4.148808e3

# Binary parameters
PB = 1.5334494508182372
A1 = 1.8979908298383135
TASC = 53630.72305223218
EPS1 = 1.4015229285046415e-08
EPS2 = -1.5014192207638522e-07
PBDOT = 5.0807149512732152081e-13

# Shapiro
T_sun = 4.925490947e-6
M2 = 0.2038397727534019566
SINI = 0.99807242781161617436
r_shapiro = T_sun * M2
s_shapiro = SINI

# DM parameters
DM = 10.39071222411148
DM1 = 3.2655074244434767e-05
DMEPOCH = 59017.9997538705

print("\nComputing JUG's binary delays...")
dt_orb = (t2_bat - TASC) * SECS_PER_DAY
n = 2.0 * np.pi / (PB * SECS_PER_DAY)

# Apply PBDOT
dn_n = -PBDOT / PB
n_corrected = n * (1.0 + dn_n * dt_orb / (PB * SECS_PER_DAY))
phi = n_corrected * dt_orb

sin_phi = np.sin(phi)
sin_2phi = np.sin(2*phi)
cos_2phi = np.cos(2*phi)

binary_roemer = A1 * (sin_phi + 0.5 * (EPS1 * sin_2phi - EPS2 * cos_2phi))
binary_shapiro = -2.0 * r_shapiro * np.log(1.0 - s_shapiro * sin_phi)
binary_total = binary_roemer + binary_shapiro

# Emission times
t_emission = t2_bat - binary_total / SECS_PER_DAY

print("Computing JUG's DM delays...")
dt_years = (t_emission - DMEPOCH) / 365.25
dm_eff = DM + DM1 * dt_years
dm_delay = K_DM_SEC * dm_eff / (pint_freq ** 2)

# Infinite-frequency times
jug_t_inf = t_emission - dm_delay / SECS_PER_DAY

print("\n" + "="*80)
print("COMPARING JUG vs PINT INFINITE-FREQUENCY TIMES")
print("="*80)

diff = (jug_t_inf - pint_tdbld) * SECS_PER_DAY  # Convert to seconds
diff_us = diff * 1e6  # Convert to microseconds

print(f"\nDifference (JUG - PINT):")
print(f"  Mean: {np.mean(diff):.9f} s = {np.mean(diff_us):.3f} μs")
print(f"  RMS: {np.std(diff):.9f} s = {np.std(diff_us):.3f} μs")
print(f"  Min/Max: {np.min(diff):.9f} / {np.max(diff):.9f} s")
print(f"  Range: {(np.max(diff) - np.min(diff)):.9f} s")

print(f"\nFirst 10 differences (μs):")
print(f"  {diff_us[:10]}")

if np.abs(np.mean(diff_us)) < 10 and np.std(diff_us) < 10:
    print("\n✓✓✓ JUG's infinite-frequency times match PINT within ~10 μs!")
    print("    The delay calculations are CORRECT!")
elif np.abs(np.mean(diff_us)) < 1000:
    print(f"\n⚠️  Close but systematic offset of {np.mean(diff_us):.3f} μs")
    print(f"   This is {np.mean(diff_us) * model.F0.value / 1e6:.6f} cycles")
else:
    print(f"\n✗ Large discrepancy! JUG differs from PINT by {np.mean(diff_us):.3f} μs")

# Now compare residuals
print("\n" + "="*80)
print("COMPARING RESIDUALS")
print("="*80)

# PINT residuals
from pint.residuals import Residuals
residuals_pint = Residuals(toas, model)
pint_res_us = residuals_pint.time_resids.to_value('us')

# Load tempo2 residuals
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

print(f"\nPINT residuals:")
print(f"  RMS: {np.sqrt(np.mean(pint_res_us**2)):.3f} μs")
print(f"  First 10: {pint_res_us[:10]}")

print(f"\nTempo2 residuals:")
print(f"  RMS: {np.sqrt(np.mean(t2_res_us**2)):.3f} μs")
print(f"  First 10: {t2_res_us[:10]}")

print(f"\nPINT vs Tempo2:")
diff_pint_t2 = pint_res_us - t2_res_us
print(f"  Mean diff: {np.mean(diff_pint_t2):.6f} μs")
print(f"  RMS diff: {np.sqrt(np.mean(diff_pint_t2**2)):.6f} μs")
print(f"  Correlation: {np.corrcoef(pint_res_us, t2_res_us)[0,1]:.6f}")

if np.sqrt(np.mean(diff_pint_t2**2)) < 1.0:
    print("\n✓ PINT matches Tempo2 perfectly! PINT is the correct reference.")
