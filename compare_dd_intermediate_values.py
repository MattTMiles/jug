"""Compare intermediate DD binary delay values between JUG and PINT for single TOA."""

import numpy as np
import jax
import jax.numpy as jnp

# Configure JAX
jax.config.update('jax_enable_x64', True)

# Import JUG DD model
from jug.delays.binary_dd import dd_binary_delay
from jug.utils.constants import SECS_PER_DAY

# Import PINT
from pint.models import get_model
from pint.toa import get_TOAs

# Test files
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("DETAILED DD INTERMEDIATE VALUE COMPARISON")
print("="*80)

# Load PINT model and TOAs
print("\nLoading PINT model and TOAs...")
model = get_model(PAR_FILE)
toas = get_TOAs(TIM_FILE, planets=True)

# Get barycentric TOAs
tdb_mjd_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Get binary component
binary_comp = model.get_components_by_category()['pulsar_system'][0]
bm = binary_comp.binary_instance

# Update binary model with TOAs
binary_comp.update_binary_object(toas, None)

# Use first TOA for detailed comparison
idx = 0
t_bary_mjd = tdb_mjd_pint[idx]

print(f"\n" + "="*80)
print(f"TEST TOA #{idx}: {t_bary_mjd:.12f} MJD")
print("="*80)

# Get binary parameters
PB = float(binary_comp.PB.value)
A1 = float(binary_comp.A1.value)
ECC = float(binary_comp.ECC.value)
OM = float(binary_comp.OM.value)
T0 = float(binary_comp.T0.value)
GAMMA = float(binary_comp.GAMMA.value) if binary_comp.GAMMA.value is not None else 0.0
PBDOT = float(binary_comp.PBDOT.value)
OMDOT = float(binary_comp.OMDOT.value)
M2 = float(binary_comp.M2.value)
SINI = float(binary_comp.SINI.value)

print(f"\nBinary Parameters:")
print(f"  PB     = {PB:.15f} days")
print(f"  A1     = {A1:.15f} lt-s")
print(f"  ECC    = {ECC:.15e}")
print(f"  OM     = {OM:.15f} deg")
print(f"  T0     = {T0:.15f} MJD")
print(f"  GAMMA  = {GAMMA:.15e} s")
print(f"  PBDOT  = {PBDOT:.15e}")
print(f"  OMDOT  = {OMDOT:.15e} deg/yr")
print(f"  M2     = {M2:.15f} Msun")
print(f"  SINI   = {SINI:.15e}")

# Compute JUG binary delay
print(f"\n" + "="*80)
print("JUG CALCULATION")
print("="*80)

dt_days = t_bary_mjd - T0
dt_sec = dt_days * SECS_PER_DAY
dt_years = dt_days / 365.25

print(f"\nTime since T0:")
print(f"  dt = {dt_days:.10f} days = {dt_years:.10f} years")

# OMDOT
omega_current_deg_jug = OM + OMDOT * dt_years
omega_rad_jug = np.deg2rad(omega_current_deg_jug)
print(f"\nOMDOT correction:")
print(f"  OM(t) = {OM:.10f} + {OMDOT:.10e} * {dt_years:.10f}")
print(f"  OM(t) = {omega_current_deg_jug:.10f} deg = {omega_rad_jug:.10f} rad")

# Mean anomaly
pb_sec = PB * SECS_PER_DAY
orbits_jug = dt_sec / pb_sec - 0.5 * PBDOT * (dt_sec / pb_sec)**2
mean_anomaly_jug = orbits_jug * 2.0 * np.pi

print(f"\nMean anomaly (PINT formula):")
print(f"  PB = {PB:.10f} days = {pb_sec:.10f} s")
print(f"  orbits = {orbits_jug:.15f}")
print(f"  M = {mean_anomaly_jug:.15f} rad = {np.rad2deg(mean_anomaly_jug):.10f} deg")

# Compute full JUG delay
jug_delay = dd_binary_delay(
    t_bary_mjd, PB, A1, ECC, OM, T0, GAMMA, PBDOT,
    OMDOT, 0.0, 0.0, SINI, M2
)
jug_delay = float(jug_delay)
print(f"\nJUG total binary delay: {jug_delay:.15f} s")

# Compute PINT binary delay
print(f"\n" + "="*80)
print("PINT CALCULATION")
print("="*80)

# Get PINT's intermediate values
print(f"\nPINT intermediate values:")
tt0_pint_qty = bm.tt0[idx]
pb_pint_qty = bm.pb()[idx]
M_pint_qty = bm.M()[idx]
omega_pint_qty = bm.omega()[idx]
E_pint_qty = bm.E()[idx]

print(f"  tt0 (time since T0): {float(tt0_pint_qty.to_value('s')):.15f} s")
print(f"  pb(): {float(pb_pint_qty.to_value('day')):.15f} days")
print(f"  M() (mean anomaly): {float(M_pint_qty.to_value('rad')):.15f} rad")
print(f"  omega(): {float(omega_pint_qty.to_value('rad')):.15f} rad = {float(omega_pint_qty.to_value('deg')):.10f} deg")
print(f"  E() (eccentric anomaly): {float(E_pint_qty.to_value('rad')):.15f} rad")

# Get delay components
pint_delay_total = float(bm.DDdelay()[idx].to_value('s'))
pint_delay_inverse = float(bm.delayInverse()[idx].to_value('s'))
pint_delay_shapiro = float(bm.delayS()[idx].to_value('s'))
pint_delay_aberr = float(bm.delayA()[idx].to_value('s'))

print(f"\nPINT delay components:")
print(f"  delayInverse: {pint_delay_inverse:.15f} s")
print(f"  delayS (Shapiro): {pint_delay_shapiro:.15f} s")
print(f"  delayA (Aberration): {pint_delay_aberr:.15f} s")
print(f"  TOTAL: {pint_delay_total:.15f} s")

# Compare
print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)

diff_total = jug_delay - pint_delay_total
print(f"\nTotal delay difference (JUG - PINT):")
print(f"  {diff_total:.15f} s = {diff_total*1e6:.6f} μs = {diff_total*1e9:.3f} ns")

# Check if omega matches
omega_pint = float(omega_pint_qty.to_value('rad'))
diff_omega = omega_rad_jug - omega_pint
print(f"\nOmega difference (JUG - PINT):")
print(f"  {diff_omega:.15e} rad = {np.rad2deg(diff_omega):.10e} deg")

# Check if mean anomaly matches
M_pint = float(M_pint_qty.to_value('rad'))
diff_M = mean_anomaly_jug - M_pint
print(f"\nMean anomaly difference (JUG - PINT):")
print(f"  {diff_M:.15e} rad = {np.rad2deg(diff_M):.10e} deg")

# Check orbits calculation
print(f"\nOrbits calculation check:")
tt0_pint = float(tt0_pint_qty.to_value('s'))
pb_pint_sec = float(pb_pint_qty.to_value('s'))
orbits_pint = float(bm.orbits()[idx])
print(f"  PINT: tt0={tt0_pint:.10f} s, pb={pb_pint_sec:.10f} s")
print(f"  PINT: orbits={orbits_pint:.15f}")
print(f"  JUG:  orbits={orbits_jug:.15f}")
print(f"  Difference: {orbits_jug - orbits_pint:.15e}")

print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if abs(diff_total * 1e9) < 50:
    print(f"\n✅ SUCCESS! Difference ({abs(diff_total)*1e9:.1f} ns) < 50 ns")
else:
    print(f"\n❌ Difference ({abs(diff_total)*1e9:.1f} ns) still > 50 ns")

    if abs(diff_omega) > 1e-10:
        print(f"  → Omega calculation differs!")
    if abs(diff_M) > 1e-10:
        print(f"  → Mean anomaly calculation differs!")
    if abs(orbits_jug - orbits_pint) > 1e-12:
        print(f"  → Orbits calculation differs!")

print(f"\nNext steps:")
print(f"  1. Check OMDOT time reference (years vs days)")
print(f"  2. Verify PBDOT is applied consistently")
print(f"  3. Check if there are any unit conversion issues")
