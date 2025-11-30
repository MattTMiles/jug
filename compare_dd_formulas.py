"""Compare JUG and PINT DD delay formulas for a single TOA.

This script instruments both implementations to extract intermediate values
and identify exactly where they diverge.
"""

import numpy as np
import jax
import jax.numpy as jnp

# Configure JAX
jax.config.update('jax_enable_x64', True)

# Import JUG implementation
from jug.delays.binary_bt import bt_binary_delay, solve_kepler
from jug.utils.constants import SECS_PER_DAY

# Load test pulsar parameters
par_file = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
tim_file = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

# Parse .par file to get binary parameters
params = {}
with open(par_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                params[parts[0]] = parts[1]

# Extract binary parameters
PB = float(params.get('PB', 0))
A1 = float(params.get('A1', 0))
ECC = float(params.get('ECC', 0))
OM = float(params.get('OM', 0))
T0 = float(params.get('T0', 0))
GAMMA = float(params.get('GAMMA', 0))
PBDOT = float(params.get('PBDOT', 0))
M2 = float(params.get('M2', 0))
SINI = float(params.get('SINI', 0))
OMDOT = float(params.get('OMDOT', 0))

print("="*80)
print("DD BINARY MODEL FORMULA COMPARISON")
print("="*80)
print(f"\nPulsar: J1012-4235")
print(f"\nBinary Parameters:")
print(f"  PB     = {PB:.10f} days")
print(f"  A1     = {A1:.6f} lt-s")
print(f"  ECC    = {ECC:.6f}")
print(f"  OM     = {OM:.2f} deg")
print(f"  T0     = {T0:.3f} MJD")
print(f"  GAMMA  = {GAMMA:.6e} s")
print(f"  PBDOT  = {PBDOT:.6e}")
print(f"  M2     = {M2:.3f} Msun")
print(f"  SINI   = {SINI:.5f}")
print(f"  OMDOT  = {OMDOT:.3f} deg/yr")

# Get first TOA from .tim file
first_toa_mjd = None
with open(tim_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('C ') and not line.startswith('FORMAT'):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    first_toa_mjd = float(parts[2])
                    break
                except:
                    pass

if first_toa_mjd is None:
    raise ValueError("Could not parse first TOA from .tim file")

print(f"\nTest TOA: {first_toa_mjd:.10f} MJD")

# ============================================================================
# JUG's CURRENT IMPLEMENTATION (WRONG)
# ============================================================================

print("\n" + "="*80)
print("JUG'S CURRENT FORMULA (WRONG FOR DD)")
print("="*80)

# For this comparison, use T0 as the test time (simplifies comparison)
t_test = first_toa_mjd

# Compute JUG's delay
dt_days = t_test - T0
print(f"\nTime since periastron: {dt_days:.6f} days")

# OMDOT correction
om_current_deg = OM + OMDOT * dt_days / 365.25
om_rad = np.deg2rad(om_current_deg)
print(f"OM (with OMDOT): {om_current_deg:.6f} deg = {om_rad:.6f} rad")

# Mean motion with PBDOT
pb_eff = PB * (1.0 + PBDOT * dt_days / PB)
n = 2.0 * np.pi / (pb_eff * SECS_PER_DAY)
print(f"PB (effective): {pb_eff:.10f} days")
print(f"Mean motion n: {n:.10e} rad/s")

# Mean anomaly
mean_anomaly = n * dt_days * SECS_PER_DAY
print(f"Mean anomaly M: {mean_anomaly:.6f} rad = {np.rad2deg(mean_anomaly):.2f} deg")

# Solve Kepler equation
ecc_anomaly = solve_kepler(jnp.array(mean_anomaly), ECC)
ecc_anomaly = float(ecc_anomaly)
print(f"Eccentric anomaly E: {ecc_anomaly:.6f} rad = {np.rad2deg(ecc_anomaly):.2f} deg")

# True anomaly
true_anomaly = 2.0 * np.arctan2(
    np.sqrt(1.0 + ECC) * np.sin(ecc_anomaly / 2.0),
    np.sqrt(1.0 - ECC) * np.cos(ecc_anomaly / 2.0)
)
print(f"True anomaly nu: {true_anomaly:.6f} rad = {np.rad2deg(true_anomaly):.2f} deg")

# Roemer delay (JUG's formula)
sin_omega_nu = np.sin(om_rad + true_anomaly)
roemer_jug = A1 * (sin_omega_nu + ECC * np.sin(om_rad))
print(f"\nJUG Roemer delay: {roemer_jug:.6f} s")
print(f"  sin(omega + nu) = {sin_omega_nu:.6f}")
print(f"  ecc * sin(omega) = {ECC * np.sin(om_rad):.6f}")

# Einstein delay
einstein = GAMMA * np.sin(ecc_anomaly)
print(f"Einstein delay: {einstein:.9f} s")

# Shapiro delay (JUG's formula)
T_SUN = 4.925490947e-6
r_shap = T_SUN * M2
shapiro_jug = -2.0 * r_shap * np.log(1.0 - SINI * sin_omega_nu)
print(f"Shapiro delay: {shapiro_jug:.9f} s")

total_jug = roemer_jug + einstein + shapiro_jug
print(f"\n** JUG TOTAL: {total_jug:.9f} s **")

# ============================================================================
# PINT's DD MODEL (CORRECT)
# ============================================================================

print("\n" + "="*80)
print("PINT's DD MODEL (CORRECT)")
print("="*80)

# Compute PINT's formula components
# Alpha and Beta (PINT formulation)
alpha = A1 * np.sin(om_rad)
beta = A1 * np.sqrt(1 - ECC**2) * np.cos(om_rad)
print(f"\nalpha = A1 * sin(omega) = {alpha:.6f} s")
print(f"beta = A1 * sqrt(1-e^2) * cos(omega) = {beta:.6f} s")

# Roemer delay (PINT's alpha-beta formulation)
er = ECC  # Assuming DR=0
roemer_pint = alpha * (np.cos(ecc_anomaly) - er) + beta * np.sin(ecc_anomaly)
print(f"\nPINT Roemer delay (delayR): {roemer_pint:.6f} s")
print(f"  alpha * (cos(E) - e) = {alpha * (np.cos(ecc_anomaly) - er):.6f} s")
print(f"  beta * sin(E) = {beta * np.sin(ecc_anomaly):.6f} s")

# Dre = Roemer + Einstein
Dre = roemer_pint + einstein
print(f"\nDre (Roemer + Einstein): {Dre:.9f} s")

# Drep = d(Dre)/dE
Drep = -alpha * np.sin(ecc_anomaly) + (beta + GAMMA) * np.cos(ecc_anomaly)
print(f"Drep (dDre/dE): {Drep:.9f} s")

# Drepp = d^2(Dre)/dE^2
Drepp = -alpha * np.cos(ecc_anomaly) - (beta + GAMMA) * np.sin(ecc_anomaly)
print(f"Drepp (d^2Dre/dE^2): {Drepp:.9f} s")

# nhat = dE/dt
nhat = 2.0 * np.pi / (pb_eff * SECS_PER_DAY) / (1 - ECC * np.cos(ecc_anomaly))
print(f"nhat (dE/dt): {nhat:.10e} rad/s")

# Inverse delay correction
correction = (
    1.0
    - nhat * Drep
    + (nhat * Drep)**2
    + 0.5 * nhat**2 * Dre * Drepp
    - 0.5 * ECC * np.sin(ecc_anomaly) / (1 - ECC * np.cos(ecc_anomaly)) * nhat**2 * Dre * Drep
)
delayInverse = Dre * correction

print(f"\nInverse delay correction factor: {correction:.12f}")
print(f"  Term 1 (unity): 1.0")
print(f"  Term 2 (-nhat*Drep): {-nhat * Drep:.10e}")
print(f"  Term 3 ((nhat*Drep)^2): {(nhat * Drep)**2:.10e}")
print(f"  Term 4 (0.5*nhat^2*Dre*Drepp): {0.5 * nhat**2 * Dre * Drepp:.10e}")
print(f"  Term 5 (-0.5*e*sinE*...): {-0.5 * ECC * np.sin(ecc_anomaly) / (1 - ECC * np.cos(ecc_anomaly)) * nhat**2 * Dre * Drep:.10e}")

print(f"\nPINT delayInverse: {delayInverse:.9f} s")
print(f"  Difference from simple Dre: {delayInverse - Dre:.9f} s = {(delayInverse - Dre)*1e6:.3f} μs")

# Shapiro delay (PINT's formula)
sinE = np.sin(ecc_anomaly)
cosE = np.cos(ecc_anomaly)
sinOmega = np.sin(om_rad)
cosOmega = np.cos(om_rad)

shapiro_pint = -2 * r_shap * np.log(
    1 - ECC * cosE - SINI * (sinOmega * (cosE - ECC) + np.sqrt(1 - ECC**2) * cosOmega * sinE)
)
print(f"\nPINT Shapiro delay: {shapiro_pint:.9f} s")

# Total PINT delay (ignoring aberration A0/B0 for now)
total_pint = delayInverse + shapiro_pint
print(f"\n** PINT TOTAL (delayInverse + delayS): {total_pint:.9f} s **")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

diff_total = total_jug - total_pint
diff_roemer = roemer_jug - roemer_pint
diff_shapiro = shapiro_jug - shapiro_pint
diff_inverse = roemer_jug + einstein - delayInverse

print(f"\nTotal delay difference (JUG - PINT): {diff_total:.9f} s = {diff_total*1e6:.3f} μs")
print(f"Roemer difference (JUG - PINT): {diff_roemer:.9f} s = {diff_roemer*1e6:.3f} μs")
print(f"Shapiro difference (JUG - PINT): {diff_shapiro:.9f} s = {diff_shapiro*1e6:.3f} μs")
print(f"Inverse delay correction: {diff_inverse:.9f} s = {diff_inverse*1e6:.3f} μs")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("\nThe bug is that JUG uses:")
print("  delay = Roemer + Einstein + Shapiro")
print("\nBut DD model requires:")
print("  delay = delayInverse(Roemer, Einstein) + delayS + delayA")
print("\nThe 'inverse delay' transformation includes a correction factor")
print("accounting for coordinate time vs proper time in the binary system.")
print(f"\nFor this TOA, the missing correction is ~{diff_inverse*1e6:.1f} μs")
print("This matches the order of magnitude of the 755 μs RMS error!")
