#!/usr/bin/env python3
"""
Trace a single TOA through Tempo2 vs JUG calculation step-by-step.

This will identify exactly where JUG diverges from Tempo2.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import math

jax.config.update('jax_enable_x64', True)

# Constants
SECS_PER_DAY = 86400.0
K_DM_SEC = 4.148808e3  # DM constant in seconds (when DM in pc/cm³, freq in MHz)

print("="*80)
print("TEMPO2 vs JUG: Single TOA Trace")
print("="*80)

# =============================================================================
# STEP 1: Load Tempo2 data for first TOA
# =============================================================================
print("\n" + "="*80)
print("STEP 1: Load Tempo2 Reference Data (First TOA)")
print("="*80)

# From temp_pre_components_next.out (line 15)
t_topo_tempo2 = 58526.213889148718145  # Topocentric MJD
bat_tempo2 = 58526.210592150969514      # Barycentric arrival time (TDB)
roemer_tempo2 = -374.60020656199253608  # Roemer delay (seconds)
shapiro_tempo2 = 1.3910568340310207382e-05  # Shapiro delay (seconds)

# From temp_pre_general2.out (first numeric line)
residual_tempo2 = -2.016075911598742077e-06  # Tempo2 residual (seconds)
residual_tempo2_us = residual_tempo2 * 1e6    # -2.016 μs

print(f"Tempo2 values for TOA #1:")
print(f"  t_topo    = {t_topo_tempo2:.15f} MJD")
print(f"  BAT       = {bat_tempo2:.15f} MJD")
print(f"  Roemer    = {roemer_tempo2:.10f} s")
print(f"  Shapiro   = {shapiro_tempo2:.10e} s")
print(f"  Residual  = {residual_tempo2_us:.6f} μs")

# Verify relationship: BAT = t_topo + delays
delay_total_tempo2 = (bat_tempo2 - t_topo_tempo2) * SECS_PER_DAY
print(f"\n  BAT - t_topo = {delay_total_tempo2:.10f} s")
print(f"  Should match Roemer + Shapiro + Einstein")
print(f"  Roemer + Shapiro = {roemer_tempo2 + shapiro_tempo2:.10f} s")

# =============================================================================
# STEP 2: Load .par file parameters
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Load Timing Parameters")
print("="*80)

def parse_par(path):
    params = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0].upper()
                try:
                    params[key] = float(parts[1])
                except ValueError:
                    params[key] = parts[1]
    return params

par = parse_par(Path('temp_model_tdb.par'))

# Spin parameters
F0 = par['F0']
F1 = par.get('F1', 0.0)
F2 = par.get('F2', 0.0)
PEPOCH = par['PEPOCH']

# DM parameters
DM = par['DM']
DM1 = par.get('DM1', 0.0)
DM2 = par.get('DM2', 0.0)
DMEPOCH = par.get('DMEPOCH', 59000.0)

# Binary parameters (ELL1)
PB = par['PB']          # Orbital period (days)
A1 = par['A1']          # Projected semi-major axis (light-seconds)
TASC = par['TASC']      # Ascending node time (MJD)
EPS1 = par['EPS1']      # Eccentricity parameter
EPS2 = par['EPS2']      # Eccentricity parameter
M2 = par.get('M2', 0.2062)     # Companion mass (solar masses)
SINI = par.get('SINI', 0.998)  # Sin of inclination

# TZR parameters
TZRMJD = par.get('TZRMJD', PEPOCH)
TZRFRQ = par.get('TZRFRQ', 1400.0)

print(f"Spin parameters:")
print(f"  F0     = {F0:.15f} Hz")
print(f"  F1     = {F1:.15e} Hz/s")
print(f"  PEPOCH = {PEPOCH} MJD")
print(f"\nBinary parameters (ELL1):")
print(f"  PB   = {PB} days")
print(f"  A1   = {A1} lt-s")
print(f"  TASC = {TASC} MJD")
print(f"  EPS1 = {EPS1:.10e}")
print(f"  EPS2 = {EPS2:.10e}")
print(f"\nDM parameters:")
print(f"  DM      = {DM} pc/cm³")
print(f"  DMEPOCH = {DMEPOCH} MJD")

# =============================================================================
# STEP 3: Compute binary delay at TOA
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Binary Delay Calculation (ELL1 Model)")
print("="*80)

# ELL1 model
dt_orb = (bat_tempo2 - TASC) * SECS_PER_DAY
PB_sec = PB * SECS_PER_DAY
n = 2.0 * np.pi / PB_sec  # Mean motion (rad/s)
M = n * dt_orb            # Mean anomaly (radians)

print(f"Orbital phase calculation:")
print(f"  Time since TASC = {dt_orb:.6f} s = {dt_orb/SECS_PER_DAY:.6f} days")
print(f"  Orbital period  = {PB_sec:.6f} s")
print(f"  Mean motion n   = {n:.15e} rad/s")
print(f"  Mean anomaly M  = {M:.10f} rad = {M/(2*np.pi):.6f} orbits")

# ELL1 Roemer delay
ecc = np.sqrt(EPS1**2 + EPS2**2)
omega = np.arctan2(EPS2, EPS1)
print(f"  Eccentricity e  = {ecc:.10e}")
print(f"  Omega (phase)   = {omega:.10f} rad")

# Phase angle
phi = M + omega
print(f"  Orbital phase   = {phi:.10f} rad")

# Roemer delay (low-eccentricity approximation for ELL1)
sin_phi = np.sin(phi)
cos_phi = np.cos(phi)
sin_2phi = np.sin(2*phi)
cos_2phi = np.cos(2*phi)

# Binary Roemer delay
delay_roemer_binary = A1 * (sin_phi + 0.5 * (EPS1 * sin_2phi - EPS2 * cos_2phi))

print(f"\nBinary delays:")
print(f"  Roemer (a1*sin(phi) + corrections) = {delay_roemer_binary:.10f} s")

# Einstein delay (gamma)
GAMMA = par.get('GAMMA', 0.0)
delay_einstein_binary = GAMMA * sin_phi if GAMMA != 0 else 0.0
print(f"  Einstein (γ*sin(phi))              = {delay_einstein_binary:.10e} s")

# Shapiro delay from companion
if M2 != 0 and SINI != 0:
    T_sun = 4.925490947e-6  # seconds (GM_sun/c^3)
    r = T_sun * M2  # Shapiro range parameter
    s = SINI        # Shapiro shape parameter
    arg = 1.0 - s * sin_phi
    if arg > 0:
        delay_shapiro_binary = -2.0 * r * np.log(arg)
    else:
        delay_shapiro_binary = 0.0
    print(f"  Shapiro (-2r*log(1-s*sin(phi)))   = {delay_shapiro_binary:.10e} s")
else:
    delay_shapiro_binary = 0.0

total_binary_delay = delay_roemer_binary + delay_einstein_binary + delay_shapiro_binary
print(f"  TOTAL binary delay                 = {total_binary_delay:.10f} s")

# =============================================================================
# STEP 4: Compute emission time
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Emission Time")
print("="*80)

t_emission_jug = bat_tempo2 - total_binary_delay / SECS_PER_DAY
print(f"t_emission = BAT - binary_delay")
print(f"           = {bat_tempo2:.15f} - {total_binary_delay/SECS_PER_DAY:.15f}")
print(f"           = {t_emission_jug:.15f} MJD")

# =============================================================================
# STEP 5: Get frequency for this TOA
# =============================================================================
print("\n" + "="*80)
print("STEP 5: Frequency")
print("="*80)

# Need to load from .tim file or components
# For now, use typical value for this pulsar
freq_mhz = 1029.0  # MHz (typical for MeerKAT L-band)
print(f"Frequency: {freq_mhz} MHz (assumed - need to verify from .tim file)")

# =============================================================================
# STEP 6: DM delay
# =============================================================================
print("\n" + "="*80)
print("STEP 6: DM Delay")
print("="*80)

# DM evolution
dt_dm_years = (t_emission_jug - DMEPOCH) / 365.25
DM_eff = DM + DM1 * dt_dm_years + DM2 * dt_dm_years**2

print(f"DM evolution:")
print(f"  Time since DMEPOCH = {dt_dm_years:.6f} years")
print(f"  DM(t) = DM + DM1*dt + DM2*dt²")
print(f"        = {DM} + {DM1}*{dt_dm_years:.6f} + {DM2}*{dt_dm_years:.6f}²")
print(f"        = {DM_eff:.10f} pc/cm³")

# DM delay
dm_delay_sec = K_DM_SEC * DM_eff / (freq_mhz**2)
print(f"\nDM delay:")
print(f"  K_DM = {K_DM_SEC} s (when DM in pc/cm³, freq in MHz)")
print(f"  delay = K_DM * DM / freq²")
print(f"        = {K_DM_SEC} * {DM_eff:.6f} / {freq_mhz}²")
print(f"        = {dm_delay_sec:.10f} s")

# =============================================================================
# STEP 7: Infinite-frequency time
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Infinite-Frequency Time")
print("="*80)

t_inf_jug = t_emission_jug - dm_delay_sec / SECS_PER_DAY
print(f"t_inf = t_emission - dm_delay")
print(f"      = {t_emission_jug:.15f} - {dm_delay_sec/SECS_PER_DAY:.15f}")
print(f"      = {t_inf_jug:.15f} MJD")

# =============================================================================
# STEP 8: Spin phase at infinite-frequency time
# =============================================================================
print("\n" + "="*80)
print("STEP 8: Spin Phase")
print("="*80)

dt_spin = (t_inf_jug - PEPOCH) * SECS_PER_DAY
phase_jug = F0 * dt_spin + 0.5 * F1 * dt_spin**2 + (1.0/6.0) * F2 * dt_spin**3

print(f"Spin phase calculation:")
print(f"  dt = t_inf - PEPOCH")
print(f"     = {t_inf_jug:.15f} - {PEPOCH}")
print(f"     = {dt_spin:.10f} s")
print(f"\n  phase = F0*dt + 0.5*F1*dt² + (1/6)*F2*dt³")
print(f"        = {F0}*{dt_spin:.6f}")
print(f"          + 0.5*{F1}*{dt_spin:.6f}²")
print(f"          + (1/6)*{F2}*{dt_spin:.6f}³")
print(f"        = {phase_jug:.10f} cycles")

# =============================================================================
# STEP 9: TZR phase reference
# =============================================================================
print("\n" + "="*80)
print("STEP 9: TZR Phase Reference")
print("="*80)

# Need to compute phase at TZR the same way
# This requires computing all delays at TZR epoch too
print(f"TZR parameters:")
print(f"  TZRMJD = {TZRMJD} MJD")
print(f"  TZRFRQ = {TZRFRQ} MHz")
print(f"\n⚠️  NOTE: TZR phase requires full delay calculation at TZR epoch")
print(f"    For now, using simple approximation...")

# Simple TZR phase (without delays - will be wrong!)
dt_tzr = (TZRMJD - PEPOCH) * SECS_PER_DAY
phase_tzr_simple = F0 * dt_tzr + 0.5 * F1 * dt_tzr**2
frac_tzr_simple = np.mod(phase_tzr_simple + 0.5, 1.0) - 0.5

print(f"\nSimplified TZR calculation (WRONG - missing delays):")
print(f"  dt = TZRMJD - PEPOCH = {dt_tzr:.6f} s")
print(f"  phase = {phase_tzr_simple:.10f} cycles")
print(f"  frac  = {frac_tzr_simple:.10f} cycles")

# =============================================================================
# STEP 10: Compute residual
# =============================================================================
print("\n" + "="*80)
print("STEP 10: Residual Calculation")
print("="*80)

# Method 1: No TZR (wrong)
phase_diff_no_tzr = phase_jug - np.floor(phase_jug + 0.5)
residual_no_tzr_sec = phase_diff_no_tzr / F0
residual_no_tzr_us = residual_no_tzr_sec * 1e6

print(f"Method 1: No TZR reference (simple mod)")
print(f"  frac_phase = mod(phase + 0.5, 1.0) - 0.5")
print(f"             = {phase_diff_no_tzr:.10f} cycles")
print(f"  residual   = frac_phase / F0")
print(f"             = {residual_no_tzr_us:.6f} μs")

# Method 2: With simple TZR (still wrong - TZR doesn't have delays)
phase_diff_with_tzr = phase_jug - phase_tzr_simple - frac_tzr_simple
frac_with_tzr = np.mod(phase_diff_with_tzr + 0.5, 1.0) - 0.5
residual_with_tzr_sec = frac_with_tzr / F0
residual_with_tzr_us = residual_with_tzr_sec * 1e6

print(f"\nMethod 2: With TZR reference (simplified)")
print(f"  phase_diff = phase - phase_tzr - frac_tzr")
print(f"             = {phase_jug:.6f} - {phase_tzr_simple:.6f} - {frac_tzr_simple:.6f}")
print(f"             = {phase_diff_with_tzr:.6f}")
print(f"  frac_phase = mod(phase_diff + 0.5, 1.0) - 0.5")
print(f"             = {frac_with_tzr:.10f} cycles")
print(f"  residual   = {residual_with_tzr_us:.6f} μs")

# =============================================================================
# STEP 11: Compare with Tempo2
# =============================================================================
print("\n" + "="*80)
print("STEP 11: Comparison with Tempo2")
print("="*80)

print(f"Tempo2 residual:      {residual_tempo2_us:.6f} μs")
print(f"JUG (no TZR):         {residual_no_tzr_us:.6f} μs")
print(f"JUG (simple TZR):     {residual_with_tzr_us:.6f} μs")

print(f"\nDifferences:")
print(f"  JUG (no TZR) - Tempo2:     {residual_no_tzr_us - residual_tempo2_us:.6f} μs")
print(f"  JUG (simple TZR) - Tempo2: {residual_with_tzr_us - residual_tempo2_us:.6f} μs")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY OF ISSUES FOUND")
print("="*80)

issues = []

# Check binary delay magnitude
if abs(total_binary_delay) < 1.0:
    issues.append("⚠️  Binary delay seems too small (< 1s). For J1909-3744, expect ~-374s to +1200s range")

# Check DM delay
expected_dm_delay = K_DM_SEC * DM / (freq_mhz**2)
if abs(dm_delay_sec - expected_dm_delay) / expected_dm_delay > 0.01:
    issues.append(f"⚠️  DM delay calculation may be wrong. Expected ~{expected_dm_delay:.6f}s, got {dm_delay_sec:.6f}s")

# Check residual magnitude
if abs(residual_no_tzr_us - residual_tempo2_us) > 1000:
    issues.append(f"✗ Large residual discrepancy ({abs(residual_no_tzr_us - residual_tempo2_us):.1f} μs)")
elif abs(residual_no_tzr_us - residual_tempo2_us) > 10:
    issues.append(f"⚠️  Moderate residual discrepancy ({abs(residual_no_tzr_us - residual_tempo2_us):.1f} μs) - likely TZR issue")
else:
    issues.append(f"✓ Residuals close to Tempo2 (within {abs(residual_no_tzr_us - residual_tempo2_us):.1f} μs)")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✓✓✓ No obvious issues found!")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Verify frequency from .tim file (assumed 1029 MHz)
2. Implement proper TZR calculation with all delays at TZR epoch
3. Check if binary delay formula matches Tempo2 exactly
4. Verify DM delay formula and K_DM constant
5. Compare intermediate values (BAT, t_emission, t_inf) with Tempo2 if available
""")
