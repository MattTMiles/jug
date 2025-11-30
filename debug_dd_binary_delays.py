#!/usr/bin/env python3
"""
Deep debugging of DD binary model - compare JUG vs PINT at the delay component level.

This script identifies exactly where JUG and PINT differ in their DD model calculations.
"""

import numpy as np
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.delays.binary_bt import bt_binary_delay

import pint.models
import pint.toa
import pint.residuals
from astropy import units as u
from astropy import time as atime

# J1012+5307 is a DD binary
PAR_FILE = "/home/mattm/soft/scripts/PTA_globalfit/BayesCave/data/pulsars/base_pulsars/ng12p5/par/J1012+5307.par"
TIM_FILE = "/home/mattm/soft/scripts/PTA_globalfit/BayesCave/data/pulsars/base_pulsars/ng12p5/tim/J1012+5307.tim"

print("="*80)
print("DD BINARY MODEL DEEP DEBUGGING: JUG vs PINT")
print("="*80)

# Parse par file
params = parse_par_file(PAR_FILE)

print("\n1. BINARY PARAMETERS:")
print("-" * 40)
for key in ['BINARY', 'PB', 'A1', 'ECC', 'OM', 'T0', 'OMDOT', 'GAMMA', 'PBDOT', 'XDOT', 'M2', 'SINI']:
    if key in params:
        print(f"  {key:12s} = {params[key]}")

# Compute JUG residuals
print("\n2. COMPUTING JUG RESIDUALS...")
print("-" * 40)
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock", return_details=True)
jug_res_us = jug_result['residuals_us']

print(f"  N_TOAs: {len(jug_res_us)}")
print(f"  Mean:   {np.mean(jug_res_us):10.3f} μs")
print(f"  Std:    {np.std(jug_res_us):10.3f} μs")
print(f"  RMS:    {np.sqrt(np.mean(jug_res_us**2)):10.3f} μs")

# Compute PINT residuals
print("\n3. COMPUTING PINT RESIDUALS...")
print("-" * 40)
pint_model = pint.models.get_model(PAR_FILE)
pint_toas = pint.toa.get_TOAs(TIM_FILE, model=pint_model)
pint_res = pint.residuals.Residuals(pint_toas, pint_model, use_weighted_mean=False)
pint_res_us = pint_res.time_resids.to(u.us).value

print(f"  N_TOAs: {len(pint_res_us)}")
print(f"  Mean:   {np.mean(pint_res_us):10.3f} μs")
print(f"  Std:    {np.std(pint_res_us):10.3f} μs")
print(f"  RMS:    {np.sqrt(np.mean(pint_res_us**2)):10.3f} μs")

# Difference
print("\n4. DIFFERENCE (JUG - PINT):")
print("-" * 40)
diff_us = jug_res_us - pint_res_us
print(f"  Mean:   {np.mean(diff_us):10.3f} μs")
print(f"  Std:    {np.std(diff_us):10.3f} μs")
print(f"  RMS:    {np.sqrt(np.mean(diff_us**2)):10.3f} μs")
print(f"  Max:    {np.max(np.abs(diff_us)):10.3f} μs")
print(f"  Min:    {np.min(diff_us):10.3f} μs")

# Check for orbital correlation
from scipy.stats import pearsonr
orbital_phase = (jug_result['toas_mjd'] - params['T0']) / params['PB']
orbital_phase = orbital_phase % 1.0
corr, pval = pearsonr(diff_us, orbital_phase)
print(f"\n  Correlation with orbital phase: {corr:.4f} (p={pval:.2e})")

if abs(corr) > 0.5:
    print("  ⚠️  STRONG ORBITAL CORRELATION - Bug is in binary delay calculation!")
else:
    print("  ✓ Weak orbital correlation - Bug may be elsewhere")

# Now examine a single TOA in detail
print("\n" + "="*80)
print("5. SINGLE TOA ANALYSIS (First TOA)")
print("="*80)

idx = 0

# Get PINT's barycentric time and binary delay
bary_mjd_pint = pint_toas.table['tdbld'][idx]
t_pint = atime.Time(bary_mjd_pint, format='mjd', scale='tdb')

# PINT binary model
binary_pint = pint_model.binary_instance
print(f"\nPINT binary model: {type(binary_pint).__name__}")
print(f"PINT binary name: {binary_pint.binary_name}")

# Get PINT binary delay
delay_pint_s = binary_pint.binary_delay(t_pint).to(u.s).value

print(f"\nPINT TOA[{idx}]:")
print(f"  Barycentric MJD (TDB): {bary_mjd_pint:.12f}")
print(f"  Binary delay:          {delay_pint_s:.9f} s")

# Get JUG's barycentric time and binary delay
bary_mjd_jug = jug_result['toas_mjd'][idx]

# Extract binary parameters
PB = params['PB']
A1 = params['A1']
ECC = params['ECC']
OM = params['OM']
T0 = params['T0']
OMDOT = params.get('OMDOT', 0.0)
GAMMA = params.get('GAMMA', 0.0)
PBDOT = params.get('PBDOT', 0.0)
M2 = params.get('M2', 0.0)
SINI = params.get('SINI', 0.0)
XDOT = params.get('XDOT', 0.0)

# Compute JUG binary delay
delay_jug = bt_binary_delay(
    jnp.array([bary_mjd_jug]),
    PB, A1, ECC, OM, T0, GAMMA, PBDOT, M2, SINI, OMDOT, XDOT
)
delay_jug_s = float(delay_jug[0])

print(f"\nJUG TOA[{idx}]:")
print(f"  Barycentric MJD (TDB): {bary_mjd_jug:.12f}")
print(f"  Binary delay:          {delay_jug_s:.9f} s")

# Compare
print(f"\n6. DIFFERENCES:")
print("-" * 40)

bary_diff_days = bary_mjd_jug - bary_mjd_pint
bary_diff_s = bary_diff_days * 86400
delay_diff_s = delay_jug_s - delay_pint_s
delay_diff_us = delay_diff_s * 1e6

print(f"  Barycentric MJD: {bary_diff_days:.12e} days")
print(f"                   {bary_diff_s:.9f} s")
print(f"                   {bary_diff_s * 1e6:.3f} μs")
print(f"\n  Binary delay:    {delay_diff_s:.9f} s")
print(f"                   {delay_diff_us:.3f} μs")

# Identify root cause
print(f"\n7. ROOT CAUSE ANALYSIS:")
print("-" * 40)

if abs(bary_diff_s) > 1e-6:  # > 1 μs difference in barycentric time
    print(f"  ⚠️  BARYCENTRIC TIMES DIFFER BY {bary_diff_s*1e6:.3f} μs")
    print("      Root cause: SSB correction (Roemer/Shapiro/Einstein delays)")
    print("      The binary delay calculation itself may be correct!")
else:
    print(f"  ✓ Barycentric times agree to < 1 μs")
    if abs(delay_diff_us) > 1.0:  # > 1 μs difference in binary delay
        print(f"  ⚠️  BINARY DELAYS DIFFER BY {delay_diff_us:.3f} μs")
        print("      Root cause: Binary orbital calculation (Kepler equation, delays)")
    else:
        print(f"  ✓ Binary delays agree to < 1 μs")
        print("  ? Bug must be elsewhere (phase model, DM, etc.)")

# Compute Kepler equation solution step-by-step
print("\n" + "="*80)
print("8. KEPLER EQUATION STEP-BY-STEP (JUG)")
print("="*80)

t_bary = bary_mjd_jug
t_days_since_t0 = t_bary - T0

# Mean motion
n_rad_per_day = 2.0 * np.pi / PB

# Apply PBDOT if present
if PBDOT != 0.0:
    t_yrs = t_days_since_t0 / 365.25
    pb_corrected = PB * (1.0 + PBDOT * t_yrs)
    n_corrected = 2.0 * np.pi / pb_corrected
    M = n_corrected * t_days_since_t0
    print(f"  PBDOT correction applied: PB = {pb_corrected:.9f} days")
else:
    M = n_rad_per_day * t_days_since_t0

print(f"\n  Time since T0:     {t_days_since_t0:.6f} days")
print(f"  Mean motion n:     {n_rad_per_day:.9f} rad/day")
print(f"  Mean anomaly M:    {M:.9f} rad = {np.degrees(M):.6f} deg")

# Solve Kepler equation
from jug.delays.binary_bt import solve_kepler
E = solve_kepler(M, ECC)
print(f"  Eccentric anom E:  {E:.9f} rad = {np.degrees(E):.6f} deg")

# True anomaly
nu = 2.0 * np.arctan2(
    np.sqrt(1.0 + ECC) * np.sin(E/2.0),
    np.sqrt(1.0 - ECC) * np.cos(E/2.0)
)
print(f"  True anomaly ν:    {nu:.9f} rad = {np.degrees(nu):.6f} deg")

# Apply OMDOT if present
omega_deg = OM
if OMDOT != 0.0:
    t_yrs = t_days_since_t0 / 365.25
    omega_deg = OM + OMDOT * t_yrs
    print(f"  OMDOT correction:  ω = {omega_deg:.6f} deg")
else:
    print(f"  Longitude of peri: ω = {omega_deg:.6f} deg")

omega_rad = np.radians(omega_deg)

# Apply XDOT if present
a1_corrected = A1
if XDOT != 0.0:
    t_yrs = t_days_since_t0 / 365.25
    a1_corrected = A1 + XDOT * t_yrs
    print(f"  XDOT correction:   A1 = {a1_corrected:.9f} lt-s")

# Compute delay components
print(f"\n9. DELAY COMPONENTS:")
print("-" * 40)

# Roemer delay
alpha = omega_rad + nu
roemer = a1_corrected * (np.cos(alpha) + ECC * np.cos(omega_rad))
print(f"  Roemer:   {roemer:.9f} s")

# Einstein delay
einstein = GAMMA * np.sin(E)
print(f"  Einstein: {einstein:.9f} s")

# Shapiro delay
T_SUN = 4.925490947e-6  # G*M_sun/c^3 in seconds
if M2 > 0 and SINI > 0:
    r = T_SUN * M2
    s = SINI
    shapiro = -2.0 * r * np.log(1.0 - s * np.sin(omega_rad + nu))
    print(f"  Shapiro:  {shapiro:.9f} s (r={r:.9e}, s={s:.6f})")
else:
    shapiro = 0.0
    print(f"  Shapiro:  {shapiro:.9f} s (not computed: M2={M2}, SINI={SINI})")

total_manual = roemer + einstein + shapiro
print(f"\n  Total (manual):      {total_manual:.9f} s")
print(f"  Total (JUG function): {delay_jug_s:.9f} s")
print(f"  Difference:          {(total_manual - delay_jug_s):.9e} s")

if abs(total_manual - delay_jug_s) > 1e-9:
    print("  ⚠️  Manual calculation doesn't match JUG function!")
else:
    print("  ✓ Manual calculation matches JUG function")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if abs(bary_diff_s) > 1e-6:
    print("\n🔴 PRIMARY ISSUE: Barycentric times differ")
    print("   → Debug SSB correction in simple_calculator.py")
    print("   → Check Roemer/Einstein/Shapiro delays to SSB")
elif abs(delay_diff_us) > 1.0:
    print("\n🔴 PRIMARY ISSUE: Binary delays differ")
    print("   → Debug Kepler solver or delay formulas in binary_bt.py")
    print("   → Compare with PINT's DD model source code")
else:
    print("\n🟢 Barycentric times and binary delays agree")
    print("   → Bug must be in phase model, DM, or something else")

print("\n" + "="*80)
