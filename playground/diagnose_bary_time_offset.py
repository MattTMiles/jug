"""Diagnose the binary evaluation time offset and its effect on residuals."""

import numpy as np
import matplotlib.pyplot as plt
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
import astropy.units as u

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
from jug.residuals.simple_calculator import compute_residuals_simple

from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel, EarthLocation
from astropy.time import Time
from pathlib import Path

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'
clock_dir = Path('/home/mattm/soft/JUG/data/clock')

print("="*80)
print("BINARY EVALUATION TIME OFFSET ANALYSIS")
print("="*80)

# Load PINT
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

# Get PINT's barycentric time
bary_toas_pint = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas_pint], dtype=np.float64)

# Get TDB times
tdb_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Compute PINT residuals
residuals_pint = Residuals(toas, model)
resid_us_pint = residuals_pint.time_resids.to_value('us')

# ===== JUG CALCULATION =====
params = parse_par_file(par_file)
toas_jug = parse_tim_file_mjds(tim_file)

mk_clock = parse_clock_file(clock_dir / "mk2utc.clk")
gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")
bipm_clock = parse_clock_file(clock_dir / "tai2tt_bipm2024.clk")

obs_itrf_km = OBSERVATORIES['meerkat']
location = EarthLocation.from_geocentric(obs_itrf_km[0]*u.km, obs_itrf_km[1]*u.km, obs_itrf_km[2]*u.km)

mjd_ints = [toa.mjd_int for toa in toas_jug]
mjd_fracs = [toa.mjd_frac for toa in toas_jug]
tdb_jug = compute_tdb_standalone_vectorized(mjd_ints, mjd_fracs, mk_clock, gps_clock, bipm_clock, location)

ra_rad = parse_ra(params['RAJ'])
dec_rad = parse_dec(params['DECJ'])
pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
posepoch = params.get('POSEPOCH', params['PEPOCH'])
parallax_mas = params.get('PX', 0.0)

ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_jug, obs_itrf_km)
L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_jug)

# JUG's roemer + shapiro
roemer_jug = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)
times = Time(tdb_jug, format='mjd', scale='tdb')
with solar_system_ephemeris.set('de440'):
    sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
obs_sun_pos_km = sun_pos - ssb_obs_pos_km
shapiro_jug = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)
roemer_shapiro_jug = roemer_jug + shapiro_jug

# JUG's barycentric time
bary_mjd_jug = tdb_jug - roemer_shapiro_jug / SECS_PER_DAY

# Get JUG residuals
result_jug = compute_residuals_simple(par_file, tim_file, verbose=False)
resid_us_jug = result_jug['residuals_us']

# ===== ANALYSIS =====
print("\n[COMPARISON]")

# Time difference
bary_diff_us = (bary_mjd_jug - bary_mjd_pint) * SECS_PER_DAY * 1e6
print(f"\nBarycentric time difference (JUG - PINT):")
print(f"  Mean: {np.mean(bary_diff_us):.3f} μs")
print(f"  RMS:  {np.std(bary_diff_us):.3f} μs")
print(f"  Range: [{np.min(bary_diff_us):.3f}, {np.max(bary_diff_us):.3f}] μs")

# Residual difference
resid_diff = resid_us_jug - (resid_us_pint - np.mean(resid_us_pint))
print(f"\nResidual difference (JUG - PINT):")
print(f"  Mean: {np.mean(resid_diff):.3f} μs")
print(f"  RMS:  {np.std(resid_diff):.3f} μs")

# Orbital phase
pb = float(params.get('PB', 0.0))
t0 = float(params.get('T0', 0.0))
orbital_phase = ((bary_mjd_pint - t0) / pb) % 1.0

# Correlation analysis
print(f"\n[CORRELATION ANALYSIS]")
print(f"Correlation between bary_time_diff and resid_diff: {np.corrcoef(bary_diff_us, resid_diff)[0,1]:.4f}")

# Check if the time difference has an annual pattern (would indicate ephemeris/Einstein issues)
# or an orbital period pattern (would indicate binary calculation issues)
from scipy.fft import fft, fftfreq

# Sort by time for FFT
sort_idx = np.argsort(tdb_pint)
dt_days = np.diff(tdb_pint[sort_idx])
if len(dt_days) > 0:
    median_dt = np.median(dt_days)
    print(f"\nMedian time spacing: {median_dt:.3f} days")

# Check periodicity in bary_diff
print(f"\n[PERIODICITY CHECK]")
# Bin by orbital phase
n_bins = 20
bins = np.linspace(0, 1, n_bins + 1)
phase_bin_idx = np.digitize(orbital_phase, bins) - 1

print(f"\nBarycentric time diff binned by orbital phase:")
print(f"  Phase     | Mean bary_diff (μs)")
print(f"  ----------|--------------------")
bary_diff_by_phase = []
for i in range(n_bins):
    mask = phase_bin_idx == i
    if np.sum(mask) > 0:
        mean_diff = np.mean(bary_diff_us[mask])
        bary_diff_by_phase.append(mean_diff)
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}   |  {mean_diff:+.3f}")
    else:
        bary_diff_by_phase.append(np.nan)

# Estimate binary delay derivative
# For small eccentricity: d(binary_delay)/dt ≈ 2πa1/PB * cos(2π*phase)
a1 = float(params.get('A1', 0.0))
pb_sec = pb * SECS_PER_DAY
n0 = 2 * np.pi / pb_sec

print(f"\n[ESTIMATED EFFECT]")
print(f"  a1 = {a1:.6f} s")
print(f"  n0 = {n0:.9e} rad/s")
print(f"  Max d(binary)/dt ≈ a1 * n0 = {a1 * n0:.6e} s/s")

# If there's a time offset dt, the binary delay error is approximately:
# delta_binary ≈ dt * a1 * n0 * cos(phase)
# So for dt=1ms, delta_binary ≈ 1e-3 * 32 * 1.07e-6 ≈ 34 ns
# For the observed ~1 μs error, we'd need dt ≈ 1μs / (32 * 1.07e-6) ≈ 30 ms

mean_bary_diff_s = np.mean(bary_diff_us) * 1e-6
expected_binary_error = mean_bary_diff_s * a1 * n0 * 1e6  # in μs
print(f"\n  Mean bary time diff: {np.mean(bary_diff_us):.3f} μs = {mean_bary_diff_s*1e3:.6f} ms")
print(f"  Expected binary error from this offset: ~{expected_binary_error:.3f} μs peak-to-peak")

# Also check the RMS of bary_diff vs expected effect
rms_bary_diff_s = np.std(bary_diff_us) * 1e-6
expected_rms_error = rms_bary_diff_s * a1 * n0 * 1e6
print(f"  RMS bary time diff: {np.std(bary_diff_us):.3f} μs = {rms_bary_diff_s*1e3:.6f} ms")
print(f"  Expected binary error from RMS offset: ~{expected_rms_error:.3f} μs")

# The actual residual RMS
print(f"\n  Actual residual diff RMS: {np.std(resid_diff):.3f} μs")
