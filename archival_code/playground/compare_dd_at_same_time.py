"""Detailed step-by-step comparison of JUG vs PINT DD model calculations."""

import numpy as np
import jax.numpy as jnp
from jug.io.par_reader import parse_par_file
from pint.models import get_model
from pint.toa import get_TOAs
import sys

PAR = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001_tdb.par'
TIM = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001.tim'

print("="*80)
print("DETAILED JUG vs PINT DD MODEL COMPARISON")
print("="*80)

# Load parameters
print("\n1. Loading parameters...")
params = parse_par_file(PAR)
model = get_model(PAR)
toas = get_TOAs(TIM, planets=True, ephem='DE440', include_bipm=True, bipm_version='BIPM2024')

# Get parameters (ensure exact same values)
PB = float(params['PB'])
A1 = float(params['A1'])
ECC = float(params['ECC'])
OM = float(params['OM'])
T0 = float(params['T0'])
PBDOT = float(params.get('PBDOT', 0.0))
OMDOT = float(params.get('OMDOT', 0.0))
XDOT = float(params.get('XDOT', 0.0))
EDOT = float(params.get('EDOT', 0.0))
GAMMA = float(params.get('GAMMA', 0.0))
H3 = float(params['H3'])
STIG = float(params['STIG'])

print(f"  PB = {PB} days")
print(f"  A1 = {A1} lt-s")
print(f"  ECC = {ECC}")
print(f"  OM = {OM} deg")
print(f"  T0 = {T0} MJD")
print(f"  PBDOT = {PBDOT}")
print(f"  OMDOT = {OMDOT} deg/yr")

# Pick a single time for detailed comparison
mjds = toas.get_mjds().value
idx = len(mjds) // 2
t_test = mjds[idx]

print(f"\n2. Test time: MJD {t_test}")

# Get PINT's binary model instance
binary = model.components['BinaryDDH'].binary_instance
binary.t = toas.table['tdbld'][idx:idx+1]

print("\n" + "="*80)
print("STEP-BY-STEP COMPARISON")
print("="*80)

# Step 1: Time since T0
print("\n--- STEP 1: Time since T0 ---")
dt_days_jug = t_test - T0
dt_sec_jug = dt_days_jug * 86400.0
dt_years_jug = dt_days_jug / 365.25

# PINT's tt0
pint_tt0 = binary.tt0.to_value('s')[0]

print(f"JUG:")
print(f"  dt = {dt_days_jug:.15e} days")
print(f"  dt = {dt_sec_jug:.15e} s")
print(f"  dt = {dt_years_jug:.15e} years")
print(f"PINT:")
print(f"  tt0 = {pint_tt0:.15e} s")
print(f"DIFF:")
print(f"  Δtt0 = {dt_sec_jug - pint_tt0:.3e} s")

# Step 2: Orbital period
print("\n--- STEP 2: Orbital period ---")
pb_sec_jug = PB * 86400.0
pint_pb_sec = binary.pb().to_value('s')[0]

print(f"JUG:")
print(f"  PB = {pb_sec_jug:.15e} s")
print(f"PINT:")
print(f"  PB = {pint_pb_sec:.15e} s")
print(f"DIFF:")
print(f"  ΔPB = {pb_sec_jug - pint_pb_sec:.3e} s")

# Step 3: Number of orbits
print("\n--- STEP 3: Number of orbits ---")
orbits_jug = dt_sec_jug / pb_sec_jug - 0.5 * PBDOT * (dt_sec_jug / pb_sec_jug)**2

# Get PINT's orbits
pint_orbits = binary.orbits()[0]

print(f"JUG:")
print(f"  orbits = tt0/PB - 0.5*PBDOT*(tt0/PB)^2")
print(f"  orbits = {orbits_jug:.15e}")
print(f"PINT:")
print(f"  orbits = {pint_orbits:.15e}")
print(f"DIFF:")
print(f"  Δorbits = {orbits_jug - pint_orbits:.3e}")

# Step 4: Mean anomaly (CRITICAL: Check wrapping)
print("\n--- STEP 4: Mean anomaly ---")

# JUG method 1: Direct (wrong - what we had before)
M_jug_unwrapped = orbits_jug * 2.0 * np.pi

# JUG method 2: With wrapping (what we should have now)
norbits_jug = np.floor(orbits_jug)
frac_orbits_jug = orbits_jug - norbits_jug
M_jug_wrapped = frac_orbits_jug * 2.0 * np.pi

# PINT's M
pint_M = binary.M().value[0]

print(f"JUG (unwrapped - OLD METHOD):")
print(f"  M = orbits * 2π = {M_jug_unwrapped:.15e} rad")
print(f"JUG (wrapped - NEW METHOD):")
print(f"  norbits = floor(orbits) = {norbits_jug}")
print(f"  frac_orbits = orbits - norbits = {frac_orbits_jug:.15e}")
print(f"  M = frac_orbits * 2π = {M_jug_wrapped:.15e} rad")
print(f"PINT:")
print(f"  M = {pint_M:.15e} rad")
print(f"DIFF (unwrapped):")
print(f"  ΔM = {M_jug_unwrapped - pint_M:.3e} rad")
print(f"DIFF (wrapped):")
print(f"  ΔM = {M_jug_wrapped - pint_M:.3e} rad")

# Step 5: Omega evolution
print("\n--- STEP 5: Omega (periastron longitude) ---")
omega_jug_deg = OM + OMDOT * dt_years_jug
omega_jug_rad = np.deg2rad(omega_jug_deg)

pint_omega = binary.omega().value[0]

print(f"JUG:")
print(f"  omega = OM + OMDOT * dt_years")
print(f"  omega = {omega_jug_deg:.15e} deg")
print(f"  omega = {omega_jug_rad:.15e} rad")
print(f"PINT:")
print(f"  omega = {pint_omega:.15e} rad")
print(f"  omega = {np.rad2deg(pint_omega):.15e} deg")
print(f"DIFF:")
print(f"  Δomega = {omega_jug_rad - pint_omega:.3e} rad")

# Step 6: Eccentricity evolution
print("\n--- STEP 6: Eccentricity ---")
ecc_jug = ECC + EDOT * dt_sec_jug
pint_ecc = binary.ecc()[0]

print(f"JUG:")
print(f"  e = ECC + EDOT * tt0")
print(f"  e = {ecc_jug:.15e}")
print(f"PINT:")
print(f"  e = {pint_ecc:.15e}")
print(f"DIFF:")
print(f"  Δe = {ecc_jug - pint_ecc:.3e}")

# Step 7: A1 evolution
print("\n--- STEP 7: Semi-major axis ---")
a1_jug = A1 + XDOT * dt_sec_jug
pint_a1 = binary.a1()[0]

print(f"JUG:")
print(f"  a1 = A1 + XDOT * tt0")
print(f"  a1 = {a1_jug:.15e} lt-s")
print(f"PINT:")
print(f"  a1 = {pint_a1:.15e} lt-s")
print(f"DIFF:")
print(f"  Δa1 = {a1_jug - pint_a1:.3e} lt-s")

# Step 8: Eccentric anomaly (Kepler's equation)
print("\n--- STEP 8: Eccentric anomaly (Kepler's equation) ---")

# For JUG, we need to import and use the actual solver
from jug.delays.binary_dd import solve_kepler
E_jug_from_unwrapped = float(solve_kepler(jnp.array(M_jug_unwrapped), ecc_jug))
E_jug_from_wrapped = float(solve_kepler(jnp.array(M_jug_wrapped), ecc_jug))

pint_E = binary.E().value[0]

print(f"JUG (from unwrapped M):")
print(f"  E = {E_jug_from_unwrapped:.15e} rad")
print(f"  sin(E) = {np.sin(E_jug_from_unwrapped):.15e}")
print(f"  cos(E) = {np.cos(E_jug_from_unwrapped):.15e}")
print(f"JUG (from wrapped M):")
print(f"  E = {E_jug_from_wrapped:.15e} rad")
print(f"  sin(E) = {np.sin(E_jug_from_wrapped):.15e}")
print(f"  cos(E) = {np.cos(E_jug_from_wrapped):.15e}")
print(f"PINT:")
print(f"  E = {pint_E:.15e} rad")
print(f"  sin(E) = {np.sin(pint_E):.15e}")
print(f"  cos(E) = {np.cos(pint_E):.15e}")
print(f"DIFF (from unwrapped):")
print(f"  ΔE = {E_jug_from_unwrapped - pint_E:.3e} rad")
print(f"  Δsin(E) = {np.sin(E_jug_from_unwrapped) - np.sin(pint_E):.3e}")
print(f"  Δcos(E) = {np.cos(E_jug_from_unwrapped) - np.cos(pint_E):.3e}")
print(f"DIFF (from wrapped):")
print(f"  ΔE = {E_jug_from_wrapped - pint_E:.3e} rad")
print(f"  Δsin(E) = {np.sin(E_jug_from_wrapped) - np.sin(pint_E):.3e}")
print(f"  Δcos(E) = {np.cos(E_jug_from_wrapped) - np.cos(pint_E):.3e}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if abs(M_jug_wrapped - pint_M) < 1e-6:
    print("✓ Mean anomaly wrapping is working correctly in the source code")
    print(f"  JUG M (wrapped) = {M_jug_wrapped:.12f} rad")
    print(f"  PINT M          = {pint_M:.12f} rad")
    print(f"  Difference      = {abs(M_jug_wrapped - pint_M):.3e} rad")
else:
    print("✗ Mean anomaly wrapping may not be applied correctly")
    print(f"  Expected difference < 1e-6 rad")
    print(f"  Actual difference = {abs(M_jug_wrapped - pint_M):.3e} rad")

if abs(np.sin(E_jug_from_wrapped) - np.sin(pint_E)) < 1e-10:
    print("\n✓ Eccentric anomaly sin(E) matches PINT")
else:
    print(f"\n✗ Eccentric anomaly sin(E) differs by {abs(np.sin(E_jug_from_wrapped) - np.sin(pint_E)):.3e}")

if abs(np.cos(E_jug_from_wrapped) - np.cos(pint_E)) < 1e-10:
    print("✓ Eccentric anomaly cos(E) matches PINT")
else:
    print(f"✗ Eccentric anomaly cos(E) differs by {abs(np.cos(E_jug_from_wrapped) - np.cos(pint_E)):.3e}")
