#!/usr/bin/env python
"""Compare orbital calculations between JUG and PINT DD model."""

import numpy as np

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

# PINT DD model computes various orbital quantities
# Let's get them for the first TOA
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

# Roemer delay components
sinE_jug = np.sin(E_jug)
cosE_jug = np.cos(E_jug)
sinOm_jug = np.sin(omega_rad_jug)
cosOm_jug = np.cos(omega_rad_jug)

alpha_jug = A1 * sinOm_jug
beta_jug = A1 * np.sqrt(1.0 - ECC**2) * cosOm_jug

roemer_jug = alpha_jug * (cosE_jug - ECC) + beta_jug * sinE_jug
print(f"alpha = {alpha_jug:.15f}")
print(f"beta = {beta_jug:.15f}")
print(f"Roemer delay = {roemer_jug:.15f} s = {roemer_jug * 1e6:.6f} μs")

# ========================================
# PINT calculation (accessing internal values)
# ========================================
print("\n--- PINT Calculation ---")

# PINT's DDmodel computes these values when you call it
# We need to call the binary model's delay calculation to populate these

# First, let's call binarymodel_delay to populate the internal state
_ = model.binarymodel_delay(toas, None)

# Now try to access PINT's internal orbital calculations
# The binary instance should have computed values
try:
    # PINT uses different variable names
    # Let's see what's available
    print(f"\nPINT binary instance attributes:")

    # tt0 = time since T0/TASC
    if hasattr(binary, 'tt0'):
        tt0_pint = binary.tt0()
        print(f"tt0 (first TOA) = {float(tt0_pint[0]):.10f} s")

    # orbits_per_second = orbital frequency
    if hasattr(binary, 'pb'):
        pb_pint = binary.pb()
        print(f"pb = {float(pb_pint):.15f}")

    # Mean anomaly
    if hasattr(binary, 'M'):
        M_pint = binary.M()
        print(f"Mean anomaly M (first TOA) = {float(M_pint[0]):.15f} rad = {np.rad2deg(float(M_pint[0])):.10f} deg")

    # Eccentric anomaly
    if hasattr(binary, 'E'):
        E_pint = binary.E()
        print(f"Eccentric anomaly E (first TOA) = {float(E_pint[0]):.15f} rad = {np.rad2deg(float(E_pint[0])):.10f} deg")

    # omega (longitude of periastron)
    if hasattr(binary, 'omega'):
        omega_pint = binary.omega()
        print(f"omega (first TOA) = {float(omega_pint[0]):.15f} rad = {np.rad2deg(float(omega_pint[0])):.10f} deg")

    # a1 (semi-major axis)
    if hasattr(binary, 'a1'):
        a1_pint = binary.a1()
        print(f"a1 = {float(a1_pint):.15f} lt-s")

    # ecc
    if hasattr(binary, 'ecc'):
        ecc_pint = binary.ecc()
        print(f"ecc (first TOA) = {float(ecc_pint[0]):.15e}")

except Exception as e:
    print(f"Error accessing PINT internals: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Direct comparison
# ========================================
print("\n" + "=" * 70)
print("Direct comparison")
print("=" * 70)

try:
    M_pint_val = float(binary.M()[0])
    E_pint_val = float(binary.E()[0])
    omega_pint_val = float(binary.omega()[0])

    print(f"\nMean anomaly:")
    print(f"  JUG:  {mean_anomaly_jug:.15f} rad ({np.rad2deg(mean_anomaly_jug):.10f} deg)")
    print(f"  PINT: {M_pint_val:.15f} rad ({np.rad2deg(M_pint_val):.10f} deg)")
    print(f"  Diff: {(mean_anomaly_jug - M_pint_val):.15e} rad ({np.rad2deg(mean_anomaly_jug - M_pint_val):.10e} deg)")

    print(f"\nEccentric anomaly:")
    print(f"  JUG:  {E_jug:.15f} rad ({np.rad2deg(E_jug):.10f} deg)")
    print(f"  PINT: {E_pint_val:.15f} rad ({np.rad2deg(E_pint_val):.10f} deg)")
    print(f"  Diff: {(E_jug - E_pint_val):.15e} rad ({np.rad2deg(E_jug - E_pint_val):.10e} deg)")

    print(f"\nOmega (longitude of periastron):")
    print(f"  JUG:  {omega_rad_jug:.15f} rad ({omega_current_jug:.10f} deg)")
    print(f"  PINT: {omega_pint_val:.15f} rad ({np.rad2deg(omega_pint_val):.10f} deg)")
    print(f"  Diff: {(omega_rad_jug - omega_pint_val):.15e} rad ({np.rad2deg(omega_rad_jug - omega_pint_val):.10e} deg)")

    # Check if PINT wraps mean anomaly differently
    print(f"\n--- Checking wrapping ---")
    print(f"JUG frac_orbits: {frac_orbits_jug:.15f}")
    print(f"PINT M / 2π:     {M_pint_val / (2*np.pi):.15f}")

    # The difference in mean anomaly might explain the Roemer delay difference
    # dR/dM ≈ x * cos(ω + ν) at low e
    # At ν ≈ M (low e), dR/dM ≈ x * cos(ω + M)
    dRdM_approx = A1 * np.cos(omega_rad_jug + mean_anomaly_jug)
    M_diff = mean_anomaly_jug - M_pint_val
    roemer_diff_from_M = dRdM_approx * M_diff
    print(f"\nEstimated Roemer diff from M diff: {roemer_diff_from_M * 1e6:.6f} μs")

except Exception as e:
    print(f"Error in comparison: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# Check PINT's M calculation formula
# ========================================
print("\n" + "=" * 70)
print("Checking PINT M calculation")
print("=" * 70)

# Look at PINT source code approach
# PINT computes: orbits = tt0/PB - 0.5*(PBDOT/PB)*(tt0/PB)**2
#                M = 2*pi * orbits

# JUG computes: orbits = tt0/pb_sec - 0.5*PBDOT*(tt0/pb_sec)**2
#               M = 2*pi * frac(orbits)

# The difference: JUG uses PBDOT directly, PINT uses PBDOT/PB

# Let's check what PBDOT means
print(f"\nPBDOT interpretation:")
print(f"  PBDOT from par file: {PBDOT:.15e}")
print(f"  If PBDOT is dP/dt: PBDOT/PB = {PBDOT/PB:.15e} per day")

# Let's compute M both ways
tt0_sec = dt_sec_jug
pb_sec = PB * SECS_PER_DAY

# JUG formula
orbits_jug_formula = tt0_sec / pb_sec - 0.5 * PBDOT * (tt0_sec / pb_sec)**2
M_jug_formula = (orbits_jug_formula % 1.0) * 2 * np.pi

# PINT formula (PBDOT/PB correction)
orbits_pint_formula = tt0_sec / pb_sec - 0.5 * (PBDOT / PB) * (tt0_sec / pb_sec)**2

# Wait, that doesn't make sense dimensionally either
# Let me check the actual PINT code

# Actually, PINT uses:
# orbits_per_sec = 1/PB (in sec^-1)
# phase = orbits_per_sec * tt0 * (1 - 0.5*(PBDOT/PB)*tt0/PB)
# This simplifies to: tt0/PB_sec - 0.5*(PBDOT/PB)*(tt0/PB_sec)^2

# But wait, the par file PBDOT has no units shown
# In pulsar timing, PBDOT is typically dP/dt, which is dimensionless
# So if P is in days and t is in days, PBDOT = dP/dt is dimensionless

# Let me check units more carefully
print(f"\nPBDOT unit check:")
print(f"  If PBDOT = dP/dt (dimensionless):")
print(f"    Term: 0.5 * PBDOT * (tt0/PB)^2 = 0.5 * {PBDOT:.6e} * {(tt0_sec/pb_sec)**2:.6e}")
print(f"         = {0.5 * PBDOT * (tt0_sec/pb_sec)**2:.15e}")
print(f"  This is a very small correction")

# The issue might be something else entirely
# Let me check if PINT wraps M differently

print(f"\nWrapping check:")
print(f"  JUG orbits (raw): {orbits_jug:.15f}")
print(f"  JUG orbits (frac): {frac_orbits_jug:.15f}")

# PINT might not wrap to [0, 2π)
# Let's check what range PINT's M is in
try:
    M_all = binary.M()
    print(f"  PINT M range: [{float(np.min(M_all)):.6f}, {float(np.max(M_all)):.6f}] rad")
    print(f"  PINT M range: [{np.rad2deg(float(np.min(M_all))):.3f}, {np.rad2deg(float(np.max(M_all))):.3f}] deg")
except:
    pass

# Check the actual difference in M causes the Roemer difference
# For a more accurate calculation:
print("\n" + "=" * 70)
print("Computing Roemer delay with PINT's M and E")
print("=" * 70)

try:
    E_pint_val = float(binary.E()[0])
    omega_pint_val = float(binary.omega()[0])
    ecc_pint_val = float(binary.ecc()[0])
    a1_pint_val = float(binary.a1())

    sinE_pint = np.sin(E_pint_val)
    cosE_pint = np.cos(E_pint_val)
    sinOm_pint = np.sin(omega_pint_val)
    cosOm_pint = np.cos(omega_pint_val)

    alpha_pint = a1_pint_val * sinOm_pint
    beta_pint = a1_pint_val * np.sqrt(1.0 - ecc_pint_val**2) * cosOm_pint

    roemer_pint_formula = alpha_pint * (cosE_pint - ecc_pint_val) + beta_pint * sinE_pint

    print(f"\nUsing PINT's orbital elements:")
    print(f"  E = {E_pint_val:.15f} rad")
    print(f"  ω = {omega_pint_val:.15f} rad")
    print(f"  e = {ecc_pint_val:.15e}")
    print(f"  a1 = {a1_pint_val:.15f} lt-s")
    print(f"  Roemer (D&D formula) = {roemer_pint_formula:.15f} s = {roemer_pint_formula * 1e6:.6f} μs")

    print(f"\nJUG Roemer: {roemer_jug * 1e6:.6f} μs")
    print(f"PINT formula Roemer: {roemer_pint_formula * 1e6:.6f} μs")
    print(f"Difference: {(roemer_jug - roemer_pint_formula) * 1e6:.6f} μs")

    # Now compare with actual PINT binary delay
    pint_delay = model.binarymodel_delay(toas, None)
    pint_delay_first = float(pint_delay[0].to('s').value)
    print(f"\nActual PINT binary delay: {pint_delay_first * 1e6:.6f} μs")
    print(f"(includes Shapiro and Einstein)")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
