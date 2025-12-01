#!/usr/bin/env python3
"""
Deep Profile of Cache Initialization (compute_residuals_simple)
===============================================================

This script instruments compute_residuals_simple() to find exactly
where the 2.7 seconds is spent.

Target breakdown:
- Clock corrections: ~15%
- Ephemeris lookups: ~40%
- Barycentric delays: ~25%
- Binary delays: ~15%
- DM/FD delays: ~5%
"""

import time
import numpy as np
from pathlib import Path
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from astropy import units as u
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Import JUG components
from jug.io.par_reader import parse_par_file, parse_ra, parse_dec
from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
from jug.io.clock import parse_clock_file, check_clock_files
from jug.delays.barycentric import (
    compute_ssb_obs_pos_vel,
    compute_pulsar_direction,
    compute_roemer_delay,
    compute_shapiro_delay,
    compute_barycentric_freq
)
from jug.utils.constants import SECS_PER_DAY, T_SUN_SEC, T_PLANET, OBSERVATORIES

# Test data
PAR_FILE = Path("data/pulsars/J1909-3744_tdb_wrong.par")
TIM_FILE = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
CLOCK_DIR = Path("data/clock")

print("="*80)
print("DEEP PROFILE: CACHE INITIALIZATION")
print("="*80)
print(f"\nPulsar: J1909-3744")
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")

timings = {}

# =============================================================================
# STEP 1: File Parsing
# =============================================================================
print("\n" + "="*80)
print("STEP 1: FILE PARSING")
print("="*80)

start = time.time()
params = parse_par_file(PAR_FILE)
timings['parse_par'] = time.time() - start
print(f"✓ Parse .par: {timings['parse_par']*1000:.1f} ms")

start = time.time()
toas = parse_tim_file_mjds(TIM_FILE)
timings['parse_tim'] = time.time() - start
print(f"✓ Parse .tim: {timings['parse_tim']*1000:.1f} ms ({len(toas)} TOAs)")

# =============================================================================
# STEP 2: Load Clock Files
# =============================================================================
print("\n" + "="*80)
print("STEP 2: LOAD CLOCK FILES")
print("="*80)

start = time.time()
mk_clock = parse_clock_file(CLOCK_DIR / "mk2utc.clk")
timings['load_mk_clock'] = time.time() - start
print(f"✓ Load mk2utc.clk: {timings['load_mk_clock']*1000:.1f} ms ({len(mk_clock)} entries)")

start = time.time()
gps_clock = parse_clock_file(CLOCK_DIR / "gps2utc.clk")
timings['load_gps_clock'] = time.time() - start
print(f"✓ Load gps2utc.clk: {timings['load_gps_clock']*1000:.1f} ms ({len(gps_clock)} entries)")

start = time.time()
bipm_clock = parse_clock_file(CLOCK_DIR / "tai2tt_bipm2024.clk")
timings['load_bipm_clock'] = time.time() - start
print(f"✓ Load tai2tt_bipm2024.clk: {timings['load_bipm_clock']*1000:.1f} ms ({len(bipm_clock)} entries)")

# =============================================================================
# STEP 3: Clock Validation
# =============================================================================
print("\n" + "="*80)
print("STEP 3: CLOCK VALIDATION")
print("="*80)

mjd_utc = np.array([toa.mjd_int + toa.mjd_frac for toa in toas])
mjd_start = np.min(mjd_utc)
mjd_end = np.max(mjd_utc)

start = time.time()
clock_ok = check_clock_files(mjd_start, mjd_end, mk_clock, gps_clock, bipm_clock, verbose=False)
timings['clock_validation'] = time.time() - start
print(f"✓ Validate clock coverage: {timings['clock_validation']*1000:.1f} ms")

# =============================================================================
# STEP 4: Observatory Setup
# =============================================================================
print("\n" + "="*80)
print("STEP 4: OBSERVATORY SETUP")
print("="*80)

start = time.time()
obs_itrf_km = OBSERVATORIES.get('meerkat')
location = EarthLocation.from_geocentric(
    obs_itrf_km[0] * u.km,
    obs_itrf_km[1] * u.km,
    obs_itrf_km[2] * u.km
)
timings['observatory_setup'] = time.time() - start
print(f"✓ Setup observatory: {timings['observatory_setup']*1000:.1f} ms")

# =============================================================================
# STEP 5: TDB Computation (Clock Corrections Applied Here!)
# =============================================================================
print("\n" + "="*80)
print("STEP 5: TDB COMPUTATION (includes clock corrections)")
print("="*80)

mjd_ints = [toa.mjd_int for toa in toas]
mjd_fracs = [toa.mjd_frac for toa in toas]

start = time.time()
tdb_mjd = compute_tdb_standalone_vectorized(
    mjd_ints, mjd_fracs,
    mk_clock, gps_clock, bipm_clock,
    location
)
timings['compute_tdb'] = time.time() - start
print(f"✓ Compute TDB (with clocks): {timings['compute_tdb']*1000:.1f} ms")
print(f"  This includes UTC→GPS→TAI→TT→TDB chain")

# =============================================================================
# STEP 6: Astrometric Parameters
# =============================================================================
print("\n" + "="*80)
print("STEP 6: ASTROMETRIC PARAMETERS")
print("="*80)

start = time.time()
ra_rad = parse_ra(params['RAJ'])
dec_rad = parse_dec(params['DECJ'])
pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
posepoch = params.get('POSEPOCH', params['PEPOCH'])
parallax_mas = params.get('PX', 0.0)
timings['parse_astrometry'] = time.time() - start
print(f"✓ Parse astrometry: {timings['parse_astrometry']*1000:.1f} ms")

# =============================================================================
# STEP 7: EPHEMERIS - SSB Observatory Position/Velocity
# =============================================================================
print("\n" + "="*80)
print("STEP 7: EPHEMERIS - Observatory Position/Velocity")
print("="*80)

start = time.time()
ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km)
timings['ephemeris_obs'] = time.time() - start
print(f"✓ Ephemeris lookup (obs): {timings['ephemeris_obs']*1000:.1f} ms")
print(f"  This queries JPL DE440 ephemeris for Earth position")

# =============================================================================
# STEP 8: Pulsar Direction
# =============================================================================
print("\n" + "="*80)
print("STEP 8: PULSAR DIRECTION (with proper motion)")
print("="*80)

start = time.time()
L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_mjd)
timings['pulsar_direction'] = time.time() - start
print(f"✓ Compute pulsar direction: {timings['pulsar_direction']*1000:.1f} ms")

# =============================================================================
# STEP 9: Roemer Delay
# =============================================================================
print("\n" + "="*80)
print("STEP 9: ROEMER DELAY")
print("="*80)

start = time.time()
roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)
timings['roemer_delay'] = time.time() - start
print(f"✓ Roemer delay: {timings['roemer_delay']*1000:.1f} ms")

# =============================================================================
# STEP 10: EPHEMERIS - Sun Position
# =============================================================================
print("\n" + "="*80)
print("STEP 10: EPHEMERIS - Sun Position")
print("="*80)

times = Time(tdb_mjd, format='mjd', scale='tdb')

start = time.time()
with solar_system_ephemeris.set('de440'):
    sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
timings['ephemeris_sun'] = time.time() - start
print(f"✓ Ephemeris lookup (sun): {timings['ephemeris_sun']*1000:.1f} ms")

obs_sun_pos_km = sun_pos - ssb_obs_pos_km

# =============================================================================
# STEP 11: Solar Shapiro Delay
# =============================================================================
print("\n" + "="*80)
print("STEP 11: SOLAR SHAPIRO DELAY")
print("="*80)

start = time.time()
sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)
timings['sun_shapiro'] = time.time() - start
print(f"✓ Sun Shapiro delay: {timings['sun_shapiro']*1000:.1f} ms")

# =============================================================================
# STEP 12: Planetary Shapiro Delays (if enabled)
# =============================================================================
print("\n" + "="*80)
print("STEP 12: PLANETARY SHAPIRO DELAYS")
print("="*80)

planet_shapiro_enabled = str(params.get('PLANET_SHAPIRO', 'N')).upper() in ('Y', 'YES', 'TRUE', '1')
planet_shapiro_sec = np.zeros(len(tdb_mjd))

if planet_shapiro_enabled:
    planet_times = {}
    with solar_system_ephemeris.set('de440'):
        for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
            start = time.time()
            planet_pos = get_body_barycentric_posvel(planet, times)[0].xyz.to(u.km).value.T
            obs_planet_km = planet_pos - ssb_obs_pos_km
            planet_shapiro_sec += compute_shapiro_delay(obs_planet_km, L_hat, T_PLANET[planet])
            planet_times[planet] = time.time() - start
            print(f"  ✓ {planet.capitalize()}: {planet_times[planet]*1000:.1f} ms")
    
    timings['planet_shapiro'] = sum(planet_times.values())
    print(f"✓ Total planetary Shapiro: {timings['planet_shapiro']*1000:.1f} ms")
else:
    timings['planet_shapiro'] = 0.0
    print("  Planetary Shapiro disabled")

# =============================================================================
# STEP 13: Barycentric Frequency
# =============================================================================
print("\n" + "="*80)
print("STEP 13: BARYCENTRIC FREQUENCY")
print("="*80)

freq_mhz = np.array([toa.freq_mhz for toa in toas])

start = time.time()
freq_bary_mhz = compute_barycentric_freq(freq_mhz, ssb_obs_vel_km_s, L_hat)
timings['barycentric_freq'] = time.time() - start
print(f"✓ Barycentric frequency: {timings['barycentric_freq']*1000:.1f} ms")

# =============================================================================
# STEP 14: JAX Array Preparation
# =============================================================================
print("\n" + "="*80)
print("STEP 14: JAX ARRAY PREPARATION")
print("="*80)

start = time.time()
tdb_jax = jnp.array(tdb_mjd, dtype=jnp.float64)
freq_bary_jax = jnp.array(freq_bary_mhz, dtype=jnp.float64)
obs_sun_jax = jnp.array(obs_sun_pos_km, dtype=jnp.float64)
L_hat_jax = jnp.array(L_hat, dtype=jnp.float64)
roemer_shapiro = roemer_sec + sun_shapiro_sec + planet_shapiro_sec
roemer_shapiro_jax = jnp.array(roemer_shapiro, dtype=jnp.float64)
timings['jax_array_prep'] = time.time() - start
print(f"✓ JAX array preparation: {timings['jax_array_prep']*1000:.1f} ms")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("TIMING SUMMARY")
print("="*80)

# Group by category
categories = {
    'File I/O': [
        ('Parse .par', timings['parse_par']),
        ('Parse .tim', timings['parse_tim']),
    ],
    'Clock System': [
        ('Load MK clock', timings['load_mk_clock']),
        ('Load GPS clock', timings['load_gps_clock']),
        ('Load BIPM clock', timings['load_bipm_clock']),
        ('Validate clocks', timings['clock_validation']),
        ('Compute TDB (with corrections)', timings['compute_tdb']),
    ],
    'Ephemeris Lookups': [
        ('Observatory pos/vel', timings['ephemeris_obs']),
        ('Sun position', timings['ephemeris_sun']),
    ],
    'Barycentric Delays': [
        ('Pulsar direction', timings['pulsar_direction']),
        ('Roemer delay', timings['roemer_delay']),
        ('Sun Shapiro', timings['sun_shapiro']),
        ('Planetary Shapiro', timings['planet_shapiro']),
        ('Barycentric frequency', timings['barycentric_freq']),
    ],
    'Setup & Prep': [
        ('Observatory setup', timings['observatory_setup']),
        ('Parse astrometry', timings['parse_astrometry']),
        ('JAX array prep', timings['jax_array_prep']),
    ]
}

print("")
total_time = 0
for category, items in categories.items():
    cat_time = sum(t for _, t in items)
    total_time += cat_time
    print(f"{category}:")
    print(f"  TOTAL: {cat_time*1000:8.1f} ms ({cat_time/total_time*100:5.1f}%)")
    for name, t in items:
        print(f"    - {name:30s} {t*1000:8.1f} ms")
    print("")

print("="*80)
print(f"TOTAL CACHE INITIALIZATION: {total_time:.3f} s ({total_time*1000:.0f} ms)")
print("="*80)

# =============================================================================
# OPTIMIZATION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("OPTIMIZATION OPPORTUNITIES")
print("="*80)

# Find top 5 slowest operations
all_ops = []
for category, items in categories.items():
    for name, t in items:
        all_ops.append((name, t, category))

all_ops.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 slowest operations:")
for i, (name, t, cat) in enumerate(all_ops[:5], 1):
    print(f"{i}. {name:30s} {t*1000:8.1f} ms ({t/total_time*100:5.1f}%) - {cat}")

print("\nRecommendations:")
print("-" * 80)

if timings['compute_tdb'] > 0.5:
    print(f"1. TDB computation ({timings['compute_tdb']*1000:.0f} ms) could be optimized:")
    print(f"   - Vectorize clock interpolation more efficiently")
    print(f"   - Pre-compute clock correction splines")
    print(f"   - Cache clock lookups if fitting multiple pulsars")

if timings['ephemeris_obs'] > 0.5:
    print(f"2. Ephemeris lookups ({timings['ephemeris_obs']*1000:.0f} ms) could be cached:")
    print(f"   - Pre-load DE440 kernel once per session")
    print(f"   - Batch ephemeris queries for multiple pulsars")
    print(f"   - Consider using PINT's cached ephemeris")

if timings['ephemeris_sun'] > 0.3:
    print(f"3. Sun position lookup ({timings['ephemeris_sun']*1000:.0f} ms):")
    print(f"   - Could be batched with other ephemeris lookups")
    print(f"   - Already uses Astropy's optimized code")

if timings['planet_shapiro'] > 0.5:
    print(f"4. Planetary Shapiro ({timings['planet_shapiro']*1000:.0f} ms):")
    print(f"   - Dominant if PLANET_SHAPIRO=Y")
    print(f"   - Could batch all planet lookups")
    print(f"   - Consider skipping if residuals < 100 ns precision needed")

print("\n" + "="*80)
print("PROFILING COMPLETE")
print("="*80)
