"""Trace DD delay calculation to find discrepancy."""

import numpy as np
import jax.numpy as jnp
from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.utils.constants import SECS_PER_DAY

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')
binary_comp = model.components.get('BinaryDD', None)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

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

idx = 500
t_bary = bary_mjd_pint[idx]

print("="*80)
print(f"TRACING DD DELAY CALCULATION FOR TOA {idx}")
print("="*80)

# ===== JAX VERSION (mimicking dd_binary_delay) =====
print("\n[JAX version]")

dt_days = t_bary - t0
dt_sec = dt_days * SECS_PER_DAY
tt0 = dt_sec

dt_years = dt_days / 365.25
omega_current_deg = om + omdot * dt_years
omega_rad_jax = jnp.deg2rad(omega_current_deg)

pb_sec = pb * SECS_PER_DAY
orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
norbits = jnp.floor(orbits)
frac_orbits = orbits - norbits
M_jax = frac_orbits * 2.0 * jnp.pi

# Kepler solver (JIT version)
from jug.delays.binary_dd import solve_kepler
E_jax = solve_kepler(M_jax, ecc)

a1_current = a1 + xdot * dt_sec
ecc_current = ecc + edot * dt_sec

sinE_jax = jnp.sin(E_jax)
cosE_jax = jnp.cos(E_jax)
sinOm_jax = jnp.sin(omega_rad_jax)
cosOm_jax = jnp.cos(omega_rad_jax)

alpha_jax = a1_current * sinOm_jax
beta_jax = a1_current * jnp.sqrt(1.0 - ecc_current**2) * cosOm_jax

delayR_jax = alpha_jax * (cosE_jax - ecc_current) + beta_jax * sinE_jax
Dre_jax = delayR_jax + gamma * sinE_jax

# Derivatives
Drep_jax = -alpha_jax * sinE_jax + (beta_jax + gamma) * cosE_jax
Drepp_jax = -alpha_jax * cosE_jax - (beta_jax + gamma) * sinE_jax

pb_prime_sec = pb_sec + pbdot * tt0
nhat_jax = (2.0 * jnp.pi / pb_prime_sec) / (1.0 - ecc_current * cosE_jax)

correction_jax = (
    1.0
    - nhat_jax * Drep_jax
    + (nhat_jax * Drep_jax)**2
    + 0.5 * nhat_jax**2 * Dre_jax * Drepp_jax
    - 0.5 * ecc_current * sinE_jax / (1.0 - ecc_current * cosE_jax) * nhat_jax**2 * Dre_jax * Drep_jax
)
delayI_jax = Dre_jax * correction_jax

T_SUN = 4.925490947e-6
shapiro_arg = (
    1.0
    - ecc_current * cosE_jax
    - sini * (sinOm_jax * (cosE_jax - ecc_current) + jnp.sqrt(1.0 - ecc_current**2) * cosOm_jax * sinE_jax)
)
delayS_jax = jnp.where(
    (m2 > 0.0) & (sini > 0.0),
    -2.0 * T_SUN * m2 * jnp.log(jnp.maximum(shapiro_arg, 1e-30)),
    0.0
)
total_jax = delayI_jax + delayS_jax

print(f"  dt_days: {dt_days}")
print(f"  orbits: {float(orbits):.12f}")
print(f"  frac_orbits: {float(frac_orbits):.12f}")
print(f"  M: {float(M_jax):.12f} rad")
print(f"  E: {float(E_jax):.12f} rad")
print(f"  omega_rad: {float(omega_rad_jax):.12f}")
print(f"  Dre: {float(Dre_jax)*1e6:.9f} μs")
print(f"  delayI: {float(delayI_jax)*1e6:.9f} μs")
print(f"  delayS: {float(delayS_jax)*1e6:.9f} μs")
print(f"  total: {float(total_jax)*1e6:.9f} μs")

# ===== NUMPY VERSION =====
print("\n[NumPy version]")

omega_rad_np = np.deg2rad(omega_current_deg)
M_np = float(frac_orbits) * 2.0 * np.pi

# Kepler solver (Python)
E_np = M_np
for _ in range(50):
    E_np = M_np + ecc * np.sin(E_np)

sinE_np = np.sin(E_np)
cosE_np = np.cos(E_np)
sinOm_np = np.sin(omega_rad_np)
cosOm_np = np.cos(omega_rad_np)

alpha_np = a1_current * sinOm_np
beta_np = a1_current * np.sqrt(1.0 - ecc_current**2) * cosOm_np

delayR_np = alpha_np * (cosE_np - ecc_current) + beta_np * sinE_np
Dre_np = delayR_np + gamma * sinE_np

Drep_np = -alpha_np * sinE_np + (beta_np + gamma) * cosE_np
Drepp_np = -alpha_np * cosE_np - (beta_np + gamma) * sinE_np

nhat_np = (2.0 * np.pi / pb_prime_sec) / (1.0 - ecc_current * cosE_np)

correction_np = (
    1.0
    - nhat_np * Drep_np
    + (nhat_np * Drep_np)**2
    + 0.5 * nhat_np**2 * Dre_np * Drepp_np
    - 0.5 * ecc_current * sinE_np / (1.0 - ecc_current * cosE_np) * nhat_np**2 * Dre_np * Drep_np
)
delayI_np = Dre_np * correction_np

shapiro_arg_np = (
    1.0
    - ecc_current * cosE_np
    - sini * (sinOm_np * (cosE_np - ecc_current) + np.sqrt(1.0 - ecc_current**2) * cosOm_np * sinE_np)
)
delayS_np = -2.0 * T_SUN * m2 * np.log(max(shapiro_arg_np, 1e-30)) if m2 > 0 and sini > 0 else 0.0
total_np = delayI_np + delayS_np

print(f"  M: {M_np:.12f} rad")
print(f"  E: {E_np:.12f} rad")
print(f"  omega_rad: {omega_rad_np:.12f}")
print(f"  Dre: {Dre_np*1e6:.9f} μs")
print(f"  delayI: {delayI_np*1e6:.9f} μs")
print(f"  delayS: {delayS_np*1e6:.9f} μs")
print(f"  total: {total_np*1e6:.9f} μs")

# ===== ACTUAL DD_BINARY_DELAY FUNCTION =====
print("\n[Actual dd_binary_delay function]")
from jug.delays.binary_dd import dd_binary_delay
total_fn = float(dd_binary_delay(
    t_bary, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
    sini, m2, 0.0, 0.0, 0.0
))
print(f"  total: {total_fn*1e6:.9f} μs")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nJAX version:    {float(total_jax)*1e6:.9f} μs")
print(f"NumPy version:  {total_np*1e6:.9f} μs")
print(f"Function call:  {total_fn*1e6:.9f} μs")
print(f"")
print(f"JAX vs NumPy:   {(float(total_jax) - total_np)*1e6:.9f} μs")
print(f"Func vs NumPy:  {(total_fn - total_np)*1e6:.3f} μs")
