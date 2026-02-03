"""Debug DD delay components step by step."""

import numpy as np
import jax.numpy as jnp
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.utils.constants import SECS_PER_DAY

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

print("="*80)
print("STEP-BY-STEP DD DELAY DEBUG")
print("="*80)

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

binary_comp = model.components.get('BinaryDD', None)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

# Single TOA for comparison
idx = 500
t_bary = bary_mjd_pint[idx]

# JUG parameters
params = parse_par_file(par_file)
pb = float(params.get('PB', 0.0))
a1 = float(params.get('A1', 0.0))
ecc = float(params.get('ECC', 0.0))
om = float(params.get('OM', 0.0))
t0 = float(params.get('T0', 0.0))
gamma = float(params.get('GAMMA', 0.0))
pbdot = float(params.get('PBDOT', 0.0))
omdot = float(params.get('OMDOT', 0.0))
xdot = float(params.get('XDOT', 0.0))
edot = float(params.get('EDOT', 0.0))
sini = float(params.get('SINI', 0.0)) if not isinstance(params.get('SINI', 0.0), str) else 0.0
m2 = float(params.get('M2', 0.0))

# ===== JUG CALCULATION (step by step) =====
print("\n[JUG] Step-by-step calculation:")

# Time since periastron
dt_days = t_bary - t0
dt_sec = dt_days * SECS_PER_DAY
tt0 = dt_sec

# Omega evolution
dt_years = dt_days / 365.25
omega_current_deg = om + omdot * dt_years
omega_rad = np.deg2rad(omega_current_deg)

# Mean anomaly
pb_sec = pb * SECS_PER_DAY
orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
norbits = np.floor(orbits)
frac_orbits = orbits - norbits
M = frac_orbits * 2.0 * np.pi

# Solve Kepler's equation (simple iteration)
E = M
for _ in range(50):
    E = M + ecc * np.sin(E)

# Secular changes
a1_current = a1 + xdot * dt_sec
ecc_current = ecc + edot * dt_sec

# Trig functions
sinE = np.sin(E)
cosE = np.cos(E)
sinOm = np.sin(omega_rad)
cosOm = np.cos(omega_rad)

# Alpha, Beta
alpha = a1_current * sinOm
beta = a1_current * np.sqrt(1.0 - ecc_current**2) * cosOm

# Dre components
delayR = alpha * (cosE - ecc_current) + beta * sinE
delayE_jug = gamma * sinE
Dre_jug = delayR + delayE_jug

# Derivatives
Drep_jug = -alpha * sinE + (beta + gamma) * cosE
Drepp_jug = -alpha * cosE - (beta + gamma) * sinE

# nhat
pb_prime_sec = pb_sec + pbdot * tt0
nhat_jug = (2.0 * np.pi / pb_prime_sec) / (1.0 - ecc_current * cosE)

# Correction factor
correction_factor = (
    1.0
    - nhat_jug * Drep_jug
    + (nhat_jug * Drep_jug)**2
    + 0.5 * nhat_jug**2 * Dre_jug * Drepp_jug
    - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat_jug**2 * Dre_jug * Drep_jug
)

delayInverse_jug = Dre_jug * correction_factor

# Shapiro
T_SUN = 4.925490947e-6
shapiro_arg = (
    1.0
    - ecc_current * cosE
    - sini * (sinOm * (cosE - ecc_current) + np.sqrt(1.0 - ecc_current**2) * cosOm * sinE)
)
delayS_jug = -2.0 * T_SUN * m2 * np.log(max(shapiro_arg, 1e-30))

total_jug = delayInverse_jug + delayS_jug

# ===== PINT CALCULATION =====
print("\n[PINT] Getting internal values:")

# Get PINT's internal values (strip units)
E_pint_raw = bo.E()[idx]
E_pint = float(E_pint_raw.to(u.rad).value) if hasattr(E_pint_raw, 'to') else float(E_pint_raw)
sinE_pint = np.sin(E_pint)
cosE_pint = np.cos(E_pint)

# Get omega
omega_pint_raw = bo.omega()[idx]
omega_pint_rad = float(omega_pint_raw.to(u.rad).value) if hasattr(omega_pint_raw, 'to') else float(omega_pint_raw)
sinOm_pint = np.sin(omega_pint_rad)
cosOm_pint = np.cos(omega_pint_rad)

Dre_pint_raw = bo.Dre()[idx]
Dre_pint = float(Dre_pint_raw.to(u.s).value) if hasattr(Dre_pint_raw, 'to') else float(Dre_pint_raw)

Drep_pint_raw = bo.Drep()[idx]
Drep_pint = float(Drep_pint_raw.decompose().value) if hasattr(Drep_pint_raw, 'decompose') else float(Drep_pint_raw)

Drepp_pint_raw = bo.Drepp()[idx]
Drepp_pint = float(Drepp_pint_raw.decompose().value) if hasattr(Drepp_pint_raw, 'decompose') else float(Drepp_pint_raw)

nhat_pint_raw = bo.nhat()[idx]
nhat_pint = float(nhat_pint_raw.to(1/u.s).value) if hasattr(nhat_pint_raw, 'to') else float(nhat_pint_raw)

delayI_pint_raw = bo.delayInverse()[idx]
delayI_pint = float(delayI_pint_raw.to(u.s).value) if hasattr(delayI_pint_raw, 'to') else float(delayI_pint_raw)

delayS_pint_raw = bo.delayS()[idx]
delayS_pint = float(delayS_pint_raw.to(u.s).value) if hasattr(delayS_pint_raw, 'to') else float(delayS_pint_raw)

total_pint = binary_comp.binarymodel_delay(toas, None).to_value(u.s)[idx]

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nEccentric anomaly E:")
print(f"  JUG:  {E:.9f} rad ({np.degrees(E):.6f} deg)")
print(f"  PINT: {E_pint:.9f} rad ({np.degrees(E_pint):.6f} deg)")
print(f"  Diff: {(E - E_pint)*1e9:.3f} nrad")

print(f"\nOmega:")
print(f"  JUG:  {omega_rad:.9f} rad ({omega_current_deg:.6f} deg)")
print(f"  PINT: {omega_pint_rad:.9f} rad ({np.degrees(omega_pint_rad):.6f} deg)")
print(f"  Diff: {(omega_rad - omega_pint_rad)*1e9:.3f} nrad")

print(f"\nalpha, beta:")
print(f"  JUG alpha:  {alpha:.12f}")
print(f"  JUG beta:   {beta:.12f}")

print(f"\nDre (Roemer + Einstein):")
print(f"  JUG:  {Dre_jug*1e6:.9f} μs")
print(f"  PINT: {Dre_pint*1e6:.9f} μs")
print(f"  Diff: {(Dre_jug - Dre_pint)*1e6:.9f} μs")

print(f"\nDrep:")
print(f"  JUG:  {Drep_jug:.12f}")
print(f"  PINT: {Drep_pint:.12f}")
print(f"  Diff: {Drep_jug - Drep_pint:.12e}")

print(f"\nDrepp:")
print(f"  JUG:  {Drepp_jug:.12f}")
print(f"  PINT: {Drepp_pint:.12f}")
print(f"  Diff: {Drepp_jug - Drepp_pint:.12e}")

print(f"\nnhat:")
print(f"  JUG:  {nhat_jug:.12e}")
print(f"  PINT: {nhat_pint:.12e}")
print(f"  Diff: {(nhat_jug - nhat_pint):.12e}")

print(f"\ndelayInverse:")
print(f"  JUG:  {delayInverse_jug*1e6:.9f} μs")
print(f"  PINT: {delayI_pint*1e6:.9f} μs")
print(f"  Diff: {(delayInverse_jug - delayI_pint)*1e6:.9f} μs")

print(f"\nShapiro delay:")
print(f"  JUG:  {delayS_jug*1e6:.9f} μs")
print(f"  PINT: {delayS_pint*1e6:.9f} μs")
print(f"  Diff: {(delayS_jug - delayS_pint)*1e6:.9f} μs")

print(f"\nTotal delay:")
print(f"  JUG:  {total_jug*1e6:.9f} μs")
print(f"  PINT: {total_pint*1e6:.9f} μs")
print(f"  Diff: {(total_jug - total_pint)*1e6:.9f} μs")

# ===== Compare with actual JUG dd_binary_delay function =====
from jug.delays.binary_dd import dd_binary_delay

jug_fn_delay = float(dd_binary_delay(
    t_bary, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
    sini, m2, 0.0, 0.0, 0.0
))

print(f"\n" + "="*80)
print("COMPARISON WITH ACTUAL JUG DD_BINARY_DELAY FUNCTION")
print("="*80)
print(f"\nJUG step-by-step: {total_jug*1e6:.9f} μs")
print(f"JUG function:     {jug_fn_delay*1e6:.9f} μs")
print(f"PINT:             {total_pint*1e6:.9f} μs")
print(f"")
print(f"Diff (step vs fn): {(total_jug - jug_fn_delay)*1e6:.3f} μs")
print(f"Diff (fn vs PINT): {(jug_fn_delay - total_pint)*1e6:.3f} μs")
