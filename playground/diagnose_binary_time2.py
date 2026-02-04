"""Diagnose binary evaluation time differences - with detailed sign checks."""

import numpy as np
from pathlib import Path
from pint.models import get_model
from pint.toa import get_TOAs
import pint.utils

from jug.io.par_reader import parse_par_file, parse_ra, parse_dec
from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
from jug.io.clock import parse_clock_file
from jug.delays.barycentric import (
    compute_ssb_obs_pos_vel,
    compute_pulsar_direction,
    compute_roemer_delay,
    compute_shapiro_delay,
)
from jug.utils.constants import SECS_PER_DAY, T_SUN_SEC, OBSERVATORIES

from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel, EarthLocation
from astropy.time import Time
from astropy import units as u

# Test case: J1713+0747
par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'
clock_dir = Path('/home/mattm/soft/JUG/data/clock')

print("="*80)
print("DETAILED SIGN CONVENTION CHECK")
print("="*80)

# ===== PINT SIDE =====
print("\n[PINT] Loading with DE440...")
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

# Get individual delay components from PINT
print("\n[PINT] Extracting individual delay components...")

# Get the astrometry component
astro_comp = model.components.get('AstrometryEquatorial', None) or model.components.get('AstrometryEcliptic', None)

# Get Roemer delay from PINT
roemer_pint_sec = astro_comp.solar_system_geometric_delay(toas).to_value(u.s)

# Get Shapiro delay from PINT
shapiro_comp = model.components.get('SolarSystemShapiro', None)
if shapiro_comp:
    shapiro_pint_sec = shapiro_comp.solar_system_shapiro_delay(toas).to_value(u.s)
else:
    shapiro_pint_sec = np.zeros(len(toas))

# Get Einstein delay from PINT (if present)
einstein_comp = model.components.get('SolarSystemEinstein', None)
if einstein_comp:
    einstein_pint_sec = einstein_comp.solar_system_einstein_delay(toas).to_value(u.s)
    print(f"  Einstein delay: mean={np.mean(einstein_pint_sec)*1e6:.3f} μs")
else:
    einstein_pint_sec = np.zeros(len(toas))
    print("  Einstein component NOT present in model")

print(f"  Roemer delay: mean={np.mean(roemer_pint_sec):.6f} s, range=[{np.min(roemer_pint_sec):.6f}, {np.max(roemer_pint_sec):.6f}] s")
print(f"  Shapiro delay: mean={np.mean(shapiro_pint_sec)*1e6:.3f} μs")

# ===== JUG SIDE =====
print("\n[JUG] Computing delays with DE440...")

# Parse files
params = parse_par_file(par_file)
toas_jug = parse_tim_file_mjds(tim_file)

# Clock files
mk_clock = parse_clock_file(clock_dir / "mk2utc.clk")
gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")
bipm_clock = parse_clock_file(clock_dir / "tai2tt_bipm2024.clk")

# Observatory
obs_itrf_km = OBSERVATORIES['meerkat']
location = EarthLocation.from_geocentric(obs_itrf_km[0]*u.km, obs_itrf_km[1]*u.km, obs_itrf_km[2]*u.km)

# Compute TDB
mjd_ints = [toa.mjd_int for toa in toas_jug]
mjd_fracs = [toa.mjd_frac for toa in toas_jug]
tdb_jug = compute_tdb_standalone_vectorized(mjd_ints, mjd_fracs, mk_clock, gps_clock, bipm_clock, location)

# Astrometry
ra_rad = parse_ra(params['RAJ'])
dec_rad = parse_dec(params['DECJ'])
pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
posepoch = params.get('POSEPOCH', params['PEPOCH'])
parallax_mas = params.get('PX', 0.0)

ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_jug, obs_itrf_km)
L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_jug)

# Roemer delay (JUG)
roemer_jug_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)

# Shapiro delay (JUG) using DE440
times = Time(tdb_jug, format='mjd', scale='tdb')
with solar_system_ephemeris.set('de440'):
    sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
obs_sun_pos_km = sun_pos - ssb_obs_pos_km
shapiro_jug_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)

print(f"  Roemer delay: mean={np.mean(roemer_jug_sec):.6f} s, range=[{np.min(roemer_jug_sec):.6f}, {np.max(roemer_jug_sec):.6f}] s")
print(f"  Shapiro delay: mean={np.mean(shapiro_jug_sec)*1e6:.3f} μs")

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPONENT-BY-COMPONENT COMPARISON")
print("="*80)

# Roemer comparison
roemer_diff = (roemer_pint_sec - roemer_jug_sec) * 1e6
print(f"\nRoemer delay (PINT - JUG):")
print(f"  Mean: {np.mean(roemer_diff):.3f} μs")
print(f"  RMS:  {np.std(roemer_diff):.3f} μs")
print(f"  Range: [{np.min(roemer_diff):.3f}, {np.max(roemer_diff):.3f}] μs")

# Check if it's a sign flip
roemer_sum = roemer_pint_sec + roemer_jug_sec
print(f"\nSign flip check (PINT + JUG):")
print(f"  Mean: {np.mean(roemer_sum):.6f} s")
print(f"  If mean ≈ 0, there's likely a sign flip")

# Shapiro comparison
shapiro_diff = (shapiro_pint_sec - shapiro_jug_sec) * 1e6
print(f"\nShapiro delay (PINT - JUG):")
print(f"  Mean: {np.mean(shapiro_diff):.3f} μs")
print(f"  RMS:  {np.std(shapiro_diff):.3f} μs")

# Check observer position
print("\n" + "="*80)
print("OBSERVER POSITION CHECK")
print("="*80)

# Get PINT's SSB obs pos
pint_ssb_obs_pos = toas.table['ssb_obs_pos'].to_value(u.km)

# Compare positions
pos_diff_km = pint_ssb_obs_pos - ssb_obs_pos_km
print(f"\nSSB observer position difference (PINT - JUG):")
print(f"  X: mean={np.mean(pos_diff_km[:,0]):.3f} km, RMS={np.std(pos_diff_km[:,0]):.3f} km")
print(f"  Y: mean={np.mean(pos_diff_km[:,1]):.3f} km, RMS={np.std(pos_diff_km[:,1]):.3f} km")
print(f"  Z: mean={np.mean(pos_diff_km[:,2]):.3f} km, RMS={np.std(pos_diff_km[:,2]):.3f} km")
print(f"  Total: mean={np.mean(np.linalg.norm(pos_diff_km, axis=1)):.3f} km")

# Check pulsar direction
print("\n" + "="*80)
print("PULSAR DIRECTION CHECK")
print("="*80)

# Get PINT's pulsar direction
pint_L_hat = astro_comp.ssb_to_psrdir(toas)  # This should return unit vectors

print(f"\nJUG L_hat[0]: {L_hat[0]}")
print(f"PINT L_hat[0]: {pint_L_hat[0]}")

L_hat_diff = pint_L_hat - L_hat
print(f"\nPulsar direction difference (PINT - JUG):")
print(f"  X: mean={np.mean(L_hat_diff[:,0])*1e9:.3f} × 10⁻⁹")
print(f"  Y: mean={np.mean(L_hat_diff[:,1])*1e9:.3f} × 10⁻⁹")
print(f"  Z: mean={np.mean(L_hat_diff[:,2])*1e9:.3f} × 10⁻⁹")

# ===== FINAL DIAGNOSIS =====
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if np.abs(np.mean(roemer_sum)) < 1.0:  # If sum is near zero, there's a sign flip
    print("\n🚨 SIGN FLIP DETECTED in Roemer delay!")
    print("   JUG's Roemer delay has the OPPOSITE SIGN compared to PINT.")
    print("   This means JUG is ADDING the Roemer delay when it should SUBTRACT, or vice versa.")
else:
    print(f"\n✓ No sign flip detected (sum mean = {np.mean(roemer_sum):.3f} s)")

if np.abs(np.mean(roemer_diff)) > 1000:  # > 1 ms
    print(f"\n⚠️  Large systematic offset in Roemer delay: {np.mean(roemer_diff)/1000:.3f} ms")
