"""Compare JUG vs PINT intermediate values for DD binary model on J1022+1001."""

import numpy as np
import jax.numpy as jnp
from jug.io.par_reader import parse_par_file
from jug.delays.binary_dd import dd_binary_delay, solve_kepler
from pint.models import get_model
from pint.toa import get_TOAs
import matplotlib.pyplot as plt

PAR = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001_tdb.par'
TIM = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001.tim'

print("Loading data...")
params = parse_par_file(PAR)
model = get_model(PAR)
toas = get_TOAs(TIM, planets=True, ephem='DE440', include_bipm=True, bipm_version='BIPM2024')

# Get parameters
PB = float(params['PB'])
A1 = float(params['A1'])
ECC = float(params['ECC'])
OM = float(params['OM'])
T0 = float(params['T0'])
GAMMA = float(params.get('GAMMA', 0.0))
PBDOT = float(params.get('PBDOT', 0.0))
OMDOT = float(params.get('OMDOT', 0.0))
H3 = float(params['H3'])
STIG = float(params['STIG'])

T_SUN = 4.925490947e-6
SINI = 2.0 * STIG / (1.0 + STIG**2)
M2 = H3 / (STIG**3 * T_SUN)

print(f"\nBinary parameters:")
print(f"  PB:    {PB} days")
print(f"  A1:    {A1} lt-s")
print(f"  ECC:   {ECC}")
print(f"  OM:    {OM} deg")
print(f"  T0:    {T0} MJD")
print(f"  OMDOT: {OMDOT} deg/yr")
print(f"  H3:    {H3} s")
print(f"  STIG:  {STIG}")
print(f"  SINI:  {SINI} (computed)")
print(f"  M2:    {M2} Msun (computed)")

# Pick a single time to test
mjds = toas.get_mjds().value
t_test = mjds[len(mjds)//2]  # Middle of dataset

print(f"\nTest time: MJD {t_test}")

# Compute JUG's intermediate values
dt_days = t_test - T0
dt_sec = dt_days * 86400.0
dt_years = dt_days / 365.25

# Mean anomaly
pb_sec = PB * 86400.0
orbits = dt_sec / pb_sec - 0.5 * PBDOT * (dt_sec / pb_sec)**2
M = orbits * 2.0 * np.pi

# Omega evolution
omega_current = OM + OMDOT * dt_years
omega_rad = np.deg2rad(omega_current)

# Solve Kepler's equation
E = float(solve_kepler(jnp.array(M), ECC))

print(f"\nJUG intermediate values:")
print(f"  dt:        {dt_days:.6f} days")
print(f"  orbits:    {orbits:.6f}")
print(f"  M:         {M:.12f} rad")
print(f"  omega:     {omega_current:.12f} deg")
print(f"  E (Kepler):{E:.12f} rad")
print(f"  sin(E):    {np.sin(E):.15e}")
print(f"  cos(E):    {np.cos(E):.15e}")

# Compute Shapiro delay argument
sinE = np.sin(E)
cosE = np.cos(E)
sinOm = np.sin(omega_rad)
cosOm = np.cos(omega_rad)

shapiro_arg_jug = (
    1.0
    - ECC * cosE
    - SINI * (sinOm * (cosE - ECC) + np.sqrt(1.0 - ECC**2) * cosOm * sinE)
)

shapiro_delay_jug = -2.0 * T_SUN * M2 * np.log(shapiro_arg_jug)

print(f"\nShapiro delay calculation (JUG):")
print(f"  sin(omega): {sinOm:.15e}")
print(f"  cos(omega): {cosOm:.15e}")
print(f"  Shapiro arg: {shapiro_arg_jug:.15e}")
print(f"  log(arg):    {np.log(shapiro_arg_jug):.15e}")
print(f"  Shapiro delay: {shapiro_delay_jug:.15e} s")
print(f"                 {shapiro_delay_jug * 1e6:.9f} μs")

# Now get PINT's values
print("\n" + "="*60)
print("PINT intermediate values:")

# Get PINT's binary model
binary = model.components['BinaryDDH'].binary_instance

# Set the time
binary.t = toas.table['tdbld'][len(mjds)//2:len(mjds)//2+1]

# Get PINT's computed values
pint_M = binary.M().value[0]
pint_E = binary.E().value[0]
pint_omega = binary.omega().value[0]

print(f"  M (PINT):     {pint_M:.12f} rad")
print(f"  E (PINT):     {pint_E:.12f} rad")
print(f"  omega (PINT): {np.rad2deg(pint_omega):.12f} deg")
print(f"  sin(E):       {np.sin(pint_E):.15e}")
print(f"  cos(E):       {np.cos(pint_E):.15e}")

# Compute PINT's Shapiro delay manually with their values
sinE_pint = np.sin(pint_E)
cosE_pint = np.cos(pint_E)
sinOm_pint = np.sin(pint_omega)
cosOm_pint = np.cos(pint_omega)

shapiro_arg_pint = (
    1.0
    - ECC * cosE_pint
    - SINI * (sinOm_pint * (cosE_pint - ECC) + np.sqrt(1.0 - ECC**2) * cosOm_pint * sinE_pint)
)

shapiro_delay_pint = -2.0 * T_SUN * M2 * np.log(shapiro_arg_pint)

print(f"\nShapiro delay calculation (PINT):")
print(f"  sin(omega): {sinOm_pint:.15e}")
print(f"  cos(omega): {cosOm_pint:.15e}")
print(f"  Shapiro arg: {shapiro_arg_pint:.15e}")
print(f"  log(arg):    {np.log(shapiro_arg_pint):.15e}")
print(f"  Shapiro delay: {shapiro_delay_pint:.15e} s")
print(f"                 {shapiro_delay_pint * 1e6:.9f} μs")

# Compare
print("\n" + "="*60)
print("DIFFERENCES:")
print(f"  ΔM:           {(M - pint_M):.3e} rad")
print(f"  ΔE:           {(E - pint_E):.3e} rad")
print(f"  Δomega:       {(omega_current - np.rad2deg(pint_omega)):.3e} deg")
print(f"  Δsin(E):      {(np.sin(E) - sinE_pint):.3e}")
print(f"  Δcos(E):      {(np.cos(E) - cosE_pint):.3e}")
print(f"  ΔShapiro arg: {(shapiro_arg_jug - shapiro_arg_pint):.3e}")
print(f"  ΔShapiro delay: {(shapiro_delay_jug - shapiro_delay_pint):.3e} s")
print(f"                  {(shapiro_delay_jug - shapiro_delay_pint)*1e9:.3f} ns")

# Check if the difference is orbital phase dependent
print("\n" + "="*60)
print("Testing orbital phase dependence...")

# Compute for several phases
phases = np.linspace(0, 1, 20)
jug_delays = []
pint_delays = []

for phase in phases:
    # Time at this phase
    t = T0 + phase * PB
    
    # JUG calculation
    dt_days = t - T0
    dt_sec = dt_days * 86400.0
    dt_years = dt_days / 365.25
    orbits = dt_sec / pb_sec - 0.5 * PBDOT * (dt_sec / pb_sec)**2
    M = orbits * 2.0 * np.pi
    omega_current = OM + OMDOT * dt_years
    omega_rad = np.deg2rad(omega_current)
    E = float(solve_kepler(jnp.array(M), ECC))
    
    sinE = np.sin(E)
    cosE = np.cos(E)
    sinOm = np.sin(omega_rad)
    cosOm = np.cos(omega_rad)
    
    shapiro_arg = (
        1.0 - ECC * cosE
        - SINI * (sinOm * (cosE - ECC) + np.sqrt(1.0 - ECC**2) * cosOm * sinE)
    )
    jug_delay = -2.0 * T_SUN * M2 * np.log(shapiro_arg)
    jug_delays.append(jug_delay)

# Get PINT delays (we'll approximate since setting individual times is complex)
# For now, just note the pattern
print(f"JUG Shapiro delay range: {np.min(jug_delays)*1e6:.6f} to {np.max(jug_delays)*1e6:.6f} μs")
print(f"Peak-to-peak: {(np.max(jug_delays) - np.min(jug_delays))*1e6:.6f} μs")
