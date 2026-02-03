"""Diagnose binary evaluation time differences between JUG and PINT."""

import numpy as np
from pathlib import Path
from pint.models import get_model
from pint.toa import get_TOAs

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
print("DIAGNOSING BINARY EVALUATION TIME DISCREPANCY")
print("="*80)

# ===== PINT SIDE =====
print("\n[PINT] Loading model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True)

# Get TDB times
tdb_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Get barycentric time (what PINT feeds to binary)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

# Delay before binary (in seconds)
delay_before_binary_pint = (tdb_pint - bary_mjd_pint) * SECS_PER_DAY

print(f"  Number of TOAs: {len(tdb_pint)}")
print(f"  TDB MJD range: {tdb_pint.min():.6f} - {tdb_pint.max():.6f}")
print(f"  Delay before binary: mean={np.mean(delay_before_binary_pint):.6f} s, range=[{np.min(delay_before_binary_pint):.6f}, {np.max(delay_before_binary_pint):.6f}] s")

# ===== JUG SIDE =====
print("\n[JUG] Computing delays...")

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

# Roemer delay
roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)

# Shapiro delay (Sun)
times = Time(tdb_jug, format='mjd', scale='tdb')
with solar_system_ephemeris.set('de440'):
    sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
obs_sun_pos_km = sun_pos - ssb_obs_pos_km
sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)

# Total delay before binary (JUG style)
roemer_shapiro_jug = roemer_sec + sun_shapiro_sec

print(f"  Number of TOAs: {len(tdb_jug)}")
print(f"  TDB MJD range: {np.min(tdb_jug):.6f} - {np.max(tdb_jug):.6f}")
print(f"  Roemer delay: mean={np.mean(roemer_sec):.6f} s, range=[{np.min(roemer_sec):.6f}, {np.max(roemer_sec):.6f}] s")
print(f"  Shapiro delay: mean={np.mean(sun_shapiro_sec):.6f} s, range=[{np.min(sun_shapiro_sec):.6f}, {np.max(sun_shapiro_sec):.6f}] s")
print(f"  Roemer+Shapiro: mean={np.mean(roemer_shapiro_jug):.6f} s, range=[{np.min(roemer_shapiro_jug):.6f}, {np.max(roemer_shapiro_jug):.6f}] s")

# ===== COMPARISON =====
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# TDB difference
tdb_diff = (tdb_pint - tdb_jug) * SECS_PER_DAY * 1e6
print(f"\nTDB difference (PINT - JUG):")
print(f"  Mean: {np.mean(tdb_diff):.3f} μs")
print(f"  RMS:  {np.std(tdb_diff):.3f} μs")
print(f"  Range: [{np.min(tdb_diff):.3f}, {np.max(tdb_diff):.3f}] μs")

# Delay before binary difference
# PINT uses: barycentric_time = tdbld - delay_before_binary
# So delay_before_binary_pint = tdb - barycentric
# JUG uses: t_ssb = tdbld - roemer_shapiro / SECS_PER_DAY
# So roemer_shapiro_jug is the delay before binary

delay_diff = (delay_before_binary_pint - roemer_shapiro_jug) * 1e6
print(f"\nDelay before binary difference (PINT - JUG):")
print(f"  Mean: {np.mean(delay_diff):.3f} μs")
print(f"  RMS:  {np.std(delay_diff):.3f} μs")
print(f"  Range: [{np.min(delay_diff):.3f}, {np.max(delay_diff):.3f}] μs")

# This difference should be mostly the Einstein delay (missing from JUG)
print("\n[NOTE] The difference above should be approximately the Einstein delay")
print("       which JUG is NOT computing but PINT includes.")

# Binary evaluation time comparison
bary_jug = tdb_jug - roemer_shapiro_jug / SECS_PER_DAY
bary_diff = (bary_mjd_pint - bary_jug) * SECS_PER_DAY * 1e6

print(f"\nBinary evaluation time difference (PINT - JUG):")
print(f"  Mean: {np.mean(bary_diff):.3f} μs")
print(f"  RMS:  {np.std(bary_diff):.3f} μs")
print(f"  Range: [{np.min(bary_diff):.3f}, {np.max(bary_diff):.3f}] μs")

# Compute orbital phase error
pb_days = float(params.get('PB', 67.825))  # J1713+0747 orbital period
phase_error_orbits = bary_diff / (pb_days * SECS_PER_DAY * 1e6)
print(f"\nOrbital phase error:")
print(f"  Mean: {np.mean(phase_error_orbits):.3e} orbits")
print(f"  Range: [{np.min(phase_error_orbits):.3e}, {np.max(phase_error_orbits):.3e}] orbits")

# Convert to radians and estimate effect on binary delay
# For DD model, the main term is a1 * sin(omega + E) where E is eccentric anomaly
# The derivative is roughly a1 * n0 * cos(...)
a1_sec = float(params.get('A1', 32.34))  # J1713+0747 projected semi-major axis
n0 = 2 * np.pi / (pb_days * SECS_PER_DAY)  # orbital frequency in rad/s

estimated_binary_error_us = np.abs(bary_diff) * n0 * a1_sec
print(f"\nEstimated binary delay error (from time error):")
print(f"  Mean: {np.mean(estimated_binary_error_us):.3f} μs")
print(f"  Max:  {np.max(estimated_binary_error_us):.3f} μs")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nThe binary evaluation time differs by {np.std(bary_diff):.1f} μs RMS")
print(f"This is likely due to MISSING Einstein delay in JUG.")
print(f"For J1713+0747 (PB={pb_days:.1f}d, A1={a1_sec:.1f}s), this causes ~{np.max(estimated_binary_error_us):.1f} μs binary delay error.")
