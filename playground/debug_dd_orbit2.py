#!/usr/bin/env python
"""Compare orbital calculations between JUG and PINT DD model - fixed version."""

import numpy as np
from astropy import units as u

# PINT imports
import pint
from pint.models import get_model
from pint import toa as pint_toa

# JUG imports
from jug.delays.binary_dd import solve_kepler
from jug.utils.constants import SECS_PER_DAY
import jax.numpy as jnp

par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.tim"

print("Loading PINT model...")
model = get_model(par_file)
toas = pint_toa.get_TOAs(tim_file, model=model)

# Get parameters
PB = float(model.PB.value)  # days
T0 = float(model.T0.value)  # MJD
A1 = float(model.A1.value)  # light-seconds
OM = float(model.OM.value)  # degrees
ECC = float(model.ECC.value)
OMDOT = float(model.OMDOT.value) if model.OMDOT.value is not None else 0.0
PBDOT = float(model.PBDOT.value) if model.PBDOT.value is not None else 0.0

print(f"PB = {PB} days")
print(f"T0 = {T0} MJD")
print(f"ECC = {ECC}")
print(f"OM = {OM} deg")
print(f"OMDOT = {OMDOT} deg/yr")
print(f"PBDOT = {PBDOT}")

# Get TDB times
tdb_mjd = toas.get_mjds()
tdb_float = np.array([t.value for t in tdb_mjd])

# Get PINT's binary model instance
binary = model.binary_instance
print(f"\nPINT binary model type: {type(binary)}")

# Call binary delay to populate internal state
_ = model.binarymodel_delay(toas, None)

print("\n" + "=" * 70)
print("Comparing orbital quantities at first TOA")
print("=" * 70)

t = tdb_float[0]
print(f"\nTDB time: {t:.15f} MJD")
print(f"T0: {T0:.15f} MJD")

# ========================================
# JUG calculation
# ========================================
print("\n--- JUG Calculation ---")

dt_days_jug = t - T0
dt_sec_jug = dt_days_jug * SECS_PER_DAY
print(f"dt_days = {dt_days_jug:.15f}")
print(f"dt_sec = {dt_sec_jug:.10f}")

# Periastron advance
dt_years_jug = dt_days_jug / 365.25
omega_current_jug = OM + OMDOT * dt_years_jug
omega_rad_jug = np.deg2rad(omega_current_jug)
print(f"omega at t = {omega_current_jug:.15f} deg = {omega_rad_jug:.15f} rad")

# Mean anomaly (JUG style)
pb_sec = PB * SECS_PER_DAY
orbits_jug = dt_sec_jug / pb_sec - 0.5 * PBDOT * (dt_sec_jug / pb_sec)**2
norbits_jug = np.floor(orbits_jug)
frac_orbits_jug = orbits_jug - norbits_jug
mean_anomaly_jug = frac_orbits_jug * 2.0 * np.pi
print(f"orbits = {orbits_jug:.15f}")
print(f"frac_orbits = {frac_orbits_jug:.15f}")
print(f"mean_anomaly = {mean_anomaly_jug:.15f} rad = {np.rad2deg(mean_anomaly_jug):.10f} deg")

# Eccentric anomaly
E_jug = float(solve_kepler(jnp.array(mean_anomaly_jug), ECC))
print(f"eccentric_anomaly E = {E_jug:.15f} rad = {np.rad2deg(E_jug):.10f} deg")

# ========================================
# PINT calculation (accessing internal values)
# ========================================
print("\n--- PINT Calculation ---")

# Access PINT values properly using .to().value or just .value
try:
    # Mean anomaly - PINT stores as attribute
    M_pint = binary.M()
    if hasattr(M_pint, 'to'):
        M_pint_val = M_pint[0].to(u.rad).value
    else:
        M_pint_val = float(M_pint[0])
    print(f"PINT M (first TOA) = {M_pint_val:.15f} rad = {np.rad2deg(M_pint_val):.10f} deg")

    # Eccentric anomaly
    E_pint = binary.E()
    if hasattr(E_pint, 'to'):
        E_pint_val = E_pint[0].to(u.rad).value
    else:
        E_pint_val = float(E_pint[0])
    print(f"PINT E (first TOA) = {E_pint_val:.15f} rad = {np.rad2deg(E_pint_val):.10f} deg")

    # Omega
    omega_pint = binary.omega()
    if hasattr(omega_pint, 'to'):
        omega_pint_val = omega_pint[0].to(u.rad).value
    else:
        omega_pint_val = float(omega_pint[0])
    print(f"PINT omega (first TOA) = {omega_pint_val:.15f} rad = {np.rad2deg(omega_pint_val):.10f} deg")

    # Time since T0
    tt0_pint = binary.tt0
    if hasattr(tt0_pint, 'to'):
        tt0_pint_val = tt0_pint[0].to(u.s).value
    else:
        tt0_pint_val = float(tt0_pint[0])
    print(f"PINT tt0 (first TOA) = {tt0_pint_val:.10f} s")

    # a1
    a1_pint = binary.a1()
    if hasattr(a1_pint, 'to'):
        a1_pint_val = a1_pint.to(u.s).value  # light-seconds
    else:
        a1_pint_val = float(a1_pint)
    print(f"PINT a1 = {a1_pint_val:.15f} lt-s")

    # ecc
    ecc_pint = binary.ecc()
    if hasattr(ecc_pint, 'to'):
        ecc_pint_val = ecc_pint[0].to(u.dimensionless_unscaled).value
    else:
        ecc_pint_val = float(ecc_pint[0]) if hasattr(ecc_pint, '__len__') else float(ecc_pint)
    print(f"PINT ecc = {ecc_pint_val:.15e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Direct comparison
# ========================================
print("\n" + "=" * 70)
print("Direct comparison")
print("=" * 70)

try:
    print(f"\nMean anomaly M:")
    print(f"  JUG:  {mean_anomaly_jug:.15f} rad ({np.rad2deg(mean_anomaly_jug):.10f} deg)")
    print(f"  PINT: {M_pint_val:.15f} rad ({np.rad2deg(M_pint_val):.10f} deg)")
    M_diff = mean_anomaly_jug - M_pint_val
    print(f"  Diff: {M_diff:.15e} rad ({np.rad2deg(M_diff):.10e} deg)")

    # If M differs by ~2π, they're equivalent
    if abs(M_diff) > np.pi:
        M_diff_wrapped = M_diff - np.sign(M_diff) * 2 * np.pi
        print(f"  Diff (wrapped): {M_diff_wrapped:.15e} rad ({np.rad2deg(M_diff_wrapped):.10e} deg)")

    print(f"\nEccentric anomaly E:")
    print(f"  JUG:  {E_jug:.15f} rad ({np.rad2deg(E_jug):.10f} deg)")
    print(f"  PINT: {E_pint_val:.15f} rad ({np.rad2deg(E_pint_val):.10f} deg)")
    E_diff = E_jug - E_pint_val
    print(f"  Diff: {E_diff:.15e} rad ({np.rad2deg(E_diff):.10e} deg)")

    print(f"\nOmega:")
    print(f"  JUG:  {omega_rad_jug:.15f} rad ({omega_current_jug:.10f} deg)")
    print(f"  PINT: {omega_pint_val:.15f} rad ({np.rad2deg(omega_pint_val):.10f} deg)")
    omega_diff = omega_rad_jug - omega_pint_val
    print(f"  Diff: {omega_diff:.15e} rad ({np.rad2deg(omega_diff):.10e} deg)")

    print(f"\nTime since T0:")
    print(f"  JUG:  {dt_sec_jug:.10f} s")
    print(f"  PINT: {tt0_pint_val:.10f} s")
    print(f"  Diff: {dt_sec_jug - tt0_pint_val:.10e} s")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Compute Roemer with both sets of values
# ========================================
print("\n" + "=" * 70)
print("Computing Roemer delay with both sets of values")
print("=" * 70)

try:
    # JUG values
    sinE_jug = np.sin(E_jug)
    cosE_jug = np.cos(E_jug)
    sinOm_jug = np.sin(omega_rad_jug)
    cosOm_jug = np.cos(omega_rad_jug)
    alpha_jug = A1 * sinOm_jug
    beta_jug = A1 * np.sqrt(1.0 - ECC**2) * cosOm_jug
    roemer_jug = alpha_jug * (cosE_jug - ECC) + beta_jug * sinE_jug

    # PINT values (using D&D formula)
    sinE_pint = np.sin(E_pint_val)
    cosE_pint = np.cos(E_pint_val)
    sinOm_pint = np.sin(omega_pint_val)
    cosOm_pint = np.cos(omega_pint_val)
    alpha_pint = a1_pint_val * sinOm_pint
    beta_pint = a1_pint_val * np.sqrt(1.0 - ecc_pint_val**2) * cosOm_pint
    roemer_pint_calc = alpha_pint * (cosE_pint - ecc_pint_val) + beta_pint * sinE_pint

    print(f"\nRoemer delay (D&D formula):")
    print(f"  Using JUG orbital elements: {roemer_jug * 1e6:.6f} μs")
    print(f"  Using PINT orbital elements: {roemer_pint_calc * 1e6:.6f} μs")
    print(f"  Difference: {(roemer_jug - roemer_pint_calc) * 1e6:.6f} μs")

    # Get actual PINT binary delay
    pint_delays = model.binarymodel_delay(toas, None)
    pint_delay_first = pint_delays[0].to(u.s).value
    print(f"\nActual PINT binary delay: {pint_delay_first * 1e6:.6f} μs")
    print(f"  (includes Shapiro, Einstein, etc.)")

    # The difference should tell us if the formula is the issue or the orbital elements
    print(f"\nDifference between PINT actual and PINT-elements-D&D-formula:")
    print(f"  {(pint_delay_first - roemer_pint_calc) * 1e6:.6f} μs")
    print(f"  (This is Shapiro + Einstein + inverse delay correction)")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Check PINT's actual Roemer calculation
# ========================================
print("\n" + "=" * 70)
print("PINT's actual Roemer delay calculation")
print("=" * 70)

try:
    # PINT DD model has delayR() for Roemer delay
    if hasattr(binary, 'delayR'):
        roemer_pint_actual = binary.delayR()
        roemer_pint_first = roemer_pint_actual[0].to(u.s).value
        print(f"PINT delayR() first TOA: {roemer_pint_first * 1e6:.6f} μs")

        print(f"\nComparison:")
        print(f"  JUG Roemer (D&D formula): {roemer_jug * 1e6:.6f} μs")
        print(f"  PINT Roemer (delayR):     {roemer_pint_first * 1e6:.6f} μs")
        print(f"  Difference: {(roemer_jug - roemer_pint_first) * 1e6:.6f} μs")

        # This difference is what we need to explain!
        # If it's large, the issue is in the Roemer formula or orbital elements
        # If it's small, the issue is elsewhere (Shapiro, Einstein, etc.)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Check what makes M different
# ========================================
print("\n" + "=" * 70)
print("Investigating M difference")
print("=" * 70)

try:
    # Check if PINT uses different wrapping
    # JUG wraps to [0, 2π)
    # PINT might not wrap at all, or wrap to [-π, π]

    # Also check if the issue is in how PBDOT is applied
    # JUG formula: orbits = tt0/PB_sec - 0.5*PBDOT*(tt0/PB_sec)^2
    # where PBDOT is dimensionless dP/dt

    # Let's compute M with different formulas and see which matches PINT

    tt0 = dt_sec_jug
    PB_sec = PB * SECS_PER_DAY

    # Formula 1: JUG current formula
    orbits_1 = tt0 / PB_sec - 0.5 * PBDOT * (tt0 / PB_sec)**2
    M_1 = (orbits_1 % 1.0) * 2 * np.pi

    # Formula 2: No wrapping
    orbits_2 = tt0 / PB_sec - 0.5 * PBDOT * (tt0 / PB_sec)**2
    M_2 = orbits_2 * 2 * np.pi

    # Formula 3: Different PBDOT normalization
    # Maybe PBDOT in JUG should be PBDOT/PB?
    orbits_3 = tt0 / PB_sec - 0.5 * (PBDOT/PB) * (tt0 / PB_sec)**2
    M_3 = (orbits_3 % 1.0) * 2 * np.pi

    # Formula 4: Using days instead of seconds
    orbits_4 = dt_days_jug / PB - 0.5 * PBDOT * (dt_days_jug / PB)**2
    M_4 = (orbits_4 % 1.0) * 2 * np.pi

    print(f"Different M calculation formulas (first TOA):")
    print(f"  PINT M:                   {M_pint_val:.15f} rad ({np.rad2deg(M_pint_val):.10f} deg)")
    print(f"  JUG (current, wrapped):   {M_1:.15f} rad ({np.rad2deg(M_1):.10f} deg)")
    print(f"  JUG (no wrap):            {M_2:.15f} rad ({np.rad2deg(M_2):.10f} deg)")
    print(f"  JUG (PBDOT/PB):           {M_3:.15f} rad ({np.rad2deg(M_3):.10f} deg)")
    print(f"  JUG (days):               {M_4:.15f} rad ({np.rad2deg(M_4):.10f} deg)")

    # Check wrapping
    M_2_wrapped = M_2 % (2 * np.pi)
    print(f"\n  JUG no-wrap mod 2π:       {M_2_wrapped:.15f} rad ({np.rad2deg(M_2_wrapped):.10f} deg)")

    # PINT's M might need to be wrapped for comparison
    M_pint_wrapped = M_pint_val % (2 * np.pi)
    print(f"  PINT M mod 2π:            {M_pint_wrapped:.15f} rad ({np.rad2deg(M_pint_wrapped):.10f} deg)")

    print(f"\nDifferences from PINT M:")
    print(f"  JUG current:    {(M_1 - M_pint_val):.15e} rad")
    print(f"  JUG no wrap:    {(M_2_wrapped - M_pint_wrapped):.15e} rad")
    print(f"  JUG PBDOT/PB:   {(M_3 - M_pint_val):.15e} rad")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
