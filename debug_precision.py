"""Debug precision issues in DD delay calculation."""

import numpy as np
import jax
import jax.numpy as jnp

# Ensure JAX uses 64-bit
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

from pint.models import get_model
from pint.toa import get_TOAs
import astropy.units as u

from jug.io.par_reader import parse_par_file
from jug.utils.constants import SECS_PER_DAY

print(f"JAX x64 enabled: {jax.config.x64_enabled}")

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')
binary_comp = model.components.get('BinaryDD', None)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)
binary_comp.update_binary_object(toas, None)
bo = binary_comp.binary_instance

# JUG parameters
params = parse_par_file(par_file)
pb = float(params.get('PB', 0.0))
t0 = float(params.get('T0', 0.0))
pbdot = float(params.get('PBDOT', 0.0))

idx = 500
t_bary = bary_mjd_pint[idx]

print(f"\nt_bary = {t_bary}")
print(f"t0 = {t0}")
print(f"pb = {pb}")
print(f"pbdot = {pbdot}")

dt_days = t_bary - t0
dt_sec = dt_days * SECS_PER_DAY
pb_sec = pb * SECS_PER_DAY

print(f"\ndt_days = {dt_days}")
print(f"dt_sec = {dt_sec}")
print(f"pb_sec = {pb_sec}")

# NumPy calculation
orbits_np = dt_sec / pb_sec - 0.5 * pbdot * (dt_sec / pb_sec)**2
norbits_np = np.floor(orbits_np)
frac_np = orbits_np - norbits_np
M_np = frac_np * 2.0 * np.pi

print(f"\n[NumPy]")
print(f"  orbits = {orbits_np}")
print(f"  norbits = {norbits_np}")
print(f"  frac_orbits = {frac_np}")
print(f"  M = {M_np}")

# JAX calculation (with arrays)
dt_sec_jax = jnp.array(dt_sec)
pb_sec_jax = jnp.array(pb_sec)
pbdot_jax = jnp.array(pbdot)

orbits_jax = dt_sec_jax / pb_sec_jax - 0.5 * pbdot_jax * (dt_sec_jax / pb_sec_jax)**2
norbits_jax = jnp.floor(orbits_jax)
frac_jax = orbits_jax - norbits_jax
M_jax = frac_jax * 2.0 * jnp.pi

print(f"\n[JAX (arrays)]")
print(f"  orbits = {float(orbits_jax)}")
print(f"  norbits = {float(norbits_jax)}")
print(f"  frac_orbits = {float(frac_jax)}")
print(f"  M = {float(M_jax)}")

# PINT calculation
M_pint_raw = bo.M()[idx]
M_pint = float(M_pint_raw.to(u.rad).value) if hasattr(M_pint_raw, 'to') else float(M_pint_raw)

orbits_pint_raw = bo.orbits()[idx]
orbits_pint = float(orbits_pint_raw) if not hasattr(orbits_pint_raw, 'value') else float(orbits_pint_raw.value)

print(f"\n[PINT]")
print(f"  orbits = {orbits_pint}")
print(f"  M = {M_pint}")

# Check dd_binary_delay function
print(f"\n[dd_binary_delay function internals]")
from jug.delays.binary_dd import dd_binary_delay

# Add debugging by importing and calling the internals
t_bary_mjd = t_bary
t0_mjd = t0
pb_days = pb
omdot_deg_yr = float(params.get('OMDOT', 0.0))
omega_deg = float(params.get('OM', 0.0))

dt_days_fn = t_bary_mjd - t0_mjd
dt_sec_fn = dt_days_fn * SECS_PER_DAY
tt0_fn = dt_sec_fn

dt_years_fn = dt_days_fn / 365.25
omega_current_deg_fn = omega_deg + omdot_deg_yr * dt_years_fn
omega_rad_fn = jnp.deg2rad(omega_current_deg_fn)

pb_sec_fn = pb_days * SECS_PER_DAY
orbits_fn = tt0_fn / pb_sec_fn - 0.5 * pbdot * (tt0_fn / pb_sec_fn)**2

# Using jnp for floor
norbits_fn = jnp.floor(orbits_fn)
frac_orbits_fn = orbits_fn - norbits_fn
M_fn = frac_orbits_fn * 2.0 * jnp.pi

print(f"  tt0 = {tt0_fn}")
print(f"  orbits = {float(orbits_fn)}")
print(f"  norbits = {float(norbits_fn)}")
print(f"  frac_orbits = {float(frac_orbits_fn)}")
print(f"  M = {float(M_fn)}")

print(f"\nDifferences:")
print(f"  M(JAX) - M(PINT) = {(float(M_jax) - M_pint)*1e9:.3f} nrad")
print(f"  M(fn) - M(PINT) = {(float(M_fn) - M_pint)*1e9:.3f} nrad")
