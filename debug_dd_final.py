"""Final debug - trace actual dd_binary_delay execution."""

import numpy as np
import jax.numpy as jnp

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.utils.constants import SECS_PER_DAY
from jug.delays.binary_dd import dd_binary_delay, solve_kepler

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')
binary_comp = model.components.get('BinaryDD', None)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

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

idx = 500
t_bary = bary_mjd_pint[idx]

print("="*80)
print(f"Parameters passed to dd_binary_delay:")
print("="*80)
print(f"  t_bary_mjd = {t_bary}")
print(f"  pb_days = {pb}")
print(f"  a1_lt_sec = {a1}")
print(f"  ecc = {ecc}")
print(f"  omega_deg = {om}")
print(f"  t0_mjd = {t0}")
print(f"  gamma_sec = {gamma}")
print(f"  pbdot = {pbdot}")
print(f"  omdot_deg_yr = {omdot}")
print(f"  xdot = {xdot}")
print(f"  edot = {edot}")
print(f"  sini = {sini}")
print(f"  m2_msun = {m2}")
print(f"  h3_sec = 0.0")
print(f"  h4_sec = 0.0")
print(f"  stig = 0.0")

# Call the function
result = float(dd_binary_delay(
    t_bary, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
    sini, m2, 0.0, 0.0, 0.0
))
print(f"\ndd_binary_delay result: {result*1e6:.9f} μs")

# PINT result
pint_total = binary_comp.binarymodel_delay(toas, None).to_value(u.s)[idx]
print(f"PINT result:            {pint_total*1e6:.9f} μs")
print(f"Difference:             {(result - pint_total)*1e6:.9f} μs")

# Now let's manually trace through the calculation with the SAME parameters
print("\n" + "="*80)
print("Manual calculation with same parameters:")
print("="*80)

dt_days = t_bary - t0
dt_sec = dt_days * SECS_PER_DAY
tt0 = dt_sec

dt_years = dt_days / 365.25
omega_current_deg = om + omdot * dt_years
omega_rad = np.deg2rad(omega_current_deg)

pb_sec = pb * SECS_PER_DAY
orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
norbits = np.floor(orbits)
frac_orbits = orbits - norbits
M = frac_orbits * 2.0 * np.pi

# Use JAX's solve_kepler to match the function
E_jax = float(solve_kepler(jnp.array(M), ecc))

# Also use Python Kepler solver for comparison
E_py = M
for _ in range(50):
    E_py = M + ecc * np.sin(E_py)

print(f"\n  dt_days = {dt_days}")
print(f"  orbits = {orbits}")
print(f"  frac_orbits = {frac_orbits}")
print(f"  M = {M}")
print(f"  E (JAX) = {E_jax}")
print(f"  E (Py) = {E_py}")

# Compare with PINT's E
E_pint_raw = bo.E()[idx]
E_pint = float(E_pint_raw.to(u.rad).value) if hasattr(E_pint_raw, 'to') else float(E_pint_raw)
print(f"  E (PINT) = {E_pint}")

# Continue manual calculation
a1_current = a1 + xdot * dt_sec
ecc_current = ecc + edot * dt_sec

sinE = np.sin(E_py)
cosE = np.cos(E_py)
sinOm = np.sin(omega_rad)
cosOm = np.cos(omega_rad)

alpha = a1_current * sinOm
beta = a1_current * np.sqrt(1.0 - ecc_current**2) * cosOm

delayR = alpha * (cosE - ecc_current) + beta * sinE
Dre = delayR + gamma * sinE

Drep = -alpha * sinE + (beta + gamma) * cosE
Drepp = -alpha * cosE - (beta + gamma) * sinE

pb_prime_sec = pb_sec + pbdot * tt0
nhat = (2.0 * np.pi / pb_prime_sec) / (1.0 - ecc_current * cosE)

correction = (
    1.0
    - nhat * Drep
    + (nhat * Drep)**2
    + 0.5 * nhat**2 * Dre * Drepp
    - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat**2 * Dre * Drep
)
delayI = Dre * correction

T_SUN = 4.925490947e-6
shapiro_arg = (
    1.0
    - ecc_current * cosE
    - sini * (sinOm * (cosE - ecc_current) + np.sqrt(1.0 - ecc_current**2) * cosOm * sinE)
)
delayS = -2.0 * T_SUN * m2 * np.log(max(shapiro_arg, 1e-30))

total_manual = delayI + delayS

print(f"\n  Manual result: {total_manual*1e6:.9f} μs")
print(f"  Function result: {result*1e6:.9f} μs")
print(f"  PINT result: {pint_total*1e6:.9f} μs")
print(f"\n  Manual - PINT: {(total_manual - pint_total)*1e6:.9f} μs")
print(f"  Function - PINT: {(result - pint_total)*1e6:.9f} μs")
print(f"  Manual - Function: {(total_manual - result)*1e6:.9f} μs")
