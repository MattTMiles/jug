"""Compare DD binary delays between JUG and PINT at the same evaluation time."""

import numpy as np
from pathlib import Path
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

from jug.io.par_reader import parse_par_file, parse_ra, parse_dec
from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
from jug.io.clock import parse_clock_file
from jug.delays.barycentric import (
    compute_ssb_obs_pos_vel,
    compute_pulsar_direction,
    compute_roemer_delay,
    compute_shapiro_delay,
)
from jug.delays.binary_dd import dd_binary_delay
from jug.utils.constants import SECS_PER_DAY, T_SUN_SEC, OBSERVATORIES

from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel, EarthLocation
from astropy.time import Time
from astropy import units as u

par_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747_tdb.par'
tim_file = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J1713+0747.tim'
clock_dir = Path('/home/mattm/soft/JUG/data/clock')

print("="*80)
print("COMPARING DD BINARY DELAYS AT SAME EVALUATION TIME")
print("="*80)

# ===== Load PINT model and TOAs =====
print("\n[PINT] Loading model and TOAs...")
model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, ephem='de440')

# Get binary component
binary_comp = model.components.get('BinaryDDK', None) or model.components.get('BinaryDD', None)
if binary_comp is None:
    print("ERROR: No DD/DDK binary component found")
    exit(1)
print(f"  Binary model: {type(binary_comp).__name__}")

# Get PINT's barycentric time (what it uses for binary evaluation)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t.to_value('day')) for t in bary_toas], dtype=np.float64)

# Get PINT's binary delay
binary_comp.update_binary_object(toas, None)
pint_binary_delay_sec = binary_comp.binarymodel_delay(toas, None).to_value(u.s)

print(f"  Binary delay: mean={np.mean(pint_binary_delay_sec)*1e6:.3f} μs")
print(f"  Range: [{np.min(pint_binary_delay_sec)*1e6:.3f}, {np.max(pint_binary_delay_sec)*1e6:.3f}] μs")

# ===== JUG SIDE =====
print("\n[JUG] Computing binary delay at PINT's barycentric time...")

# Parse par file for binary parameters
params = parse_par_file(par_file)

# DD Parameters
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
h3 = float(params.get('H3', 0.0))
h4 = float(params.get('H4', 0.0))
stig = float(params.get('STIG', 0.0))

print(f"\n  Binary parameters:")
print(f"    PB={pb:.6f} d, A1={a1:.6f} s, ECC={ecc:.9f}")
print(f"    OM={om:.6f} deg, T0={t0:.6f}")
print(f"    GAMMA={gamma:.6e} s, PBDOT={pbdot:.6e}")
print(f"    OMDOT={omdot:.6f} deg/yr, SINI={sini:.6f}, M2={m2:.6f} M_sun")

# Compute JUG binary delay using PINT's barycentric time
jug_binary_delay_sec = np.array([
    float(dd_binary_delay(
        t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
        sini, m2, h3, h4, stig
    ))
    for t in bary_mjd_pint
])

print(f"\n  JUG binary delay (at PINT time): mean={np.mean(jug_binary_delay_sec)*1e6:.3f} μs")
print(f"  Range: [{np.min(jug_binary_delay_sec)*1e6:.3f}, {np.max(jug_binary_delay_sec)*1e6:.3f}] μs")

# ===== Compare =====
print("\n" + "="*80)
print("COMPARISON (JUG vs PINT at same time)")
print("="*80)

delay_diff_us = (jug_binary_delay_sec - pint_binary_delay_sec) * 1e6
print(f"\nBinary delay difference (JUG - PINT):")
print(f"  Mean: {np.mean(delay_diff_us):.3f} μs")
print(f"  RMS:  {np.std(delay_diff_us):.3f} μs")
print(f"  Range: [{np.min(delay_diff_us):.3f}, {np.max(delay_diff_us):.3f}] μs")

# ===== Now compute JUG binary delay at JUG's computed barycentric time =====
print("\n" + "="*80)
print("NOW CHECK JUG's BARYCENTRIC TIME")
print("="*80)

# Compute JUG's roemer+shapiro
tdb_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Get JUG's roemer/shapiro from PINT (since we verified they match)
astro_comp = model.components.get('AstrometryEquatorial', None)
roemer_jug_sec = astro_comp.solar_system_geometric_delay(toas).to_value(u.s)

shapiro_comp = model.components.get('SolarSystemShapiro', None)
shapiro_jug_sec = shapiro_comp.solar_system_shapiro_delay(toas).to_value(u.s) if shapiro_comp else np.zeros(len(toas))

roemer_shapiro_jug = roemer_jug_sec + shapiro_jug_sec

# JUG's barycentric time
bary_mjd_jug = tdb_pint - roemer_shapiro_jug / SECS_PER_DAY

# Difference in barycentric times
bary_diff_us = (bary_mjd_jug - bary_mjd_pint) * SECS_PER_DAY * 1e6
print(f"\nBarycentric time difference (JUG - PINT):")
print(f"  Mean: {np.mean(bary_diff_us):.3f} μs")
print(f"  RMS:  {np.std(bary_diff_us):.3f} μs")
print(f"  Range: [{np.min(bary_diff_us):.3f}, {np.max(bary_diff_us):.3f}] μs")

# Now compute JUG binary delay at JUG's barycentric time
jug_binary_delay_jug_time_sec = np.array([
    float(dd_binary_delay(
        t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
        sini, m2, h3, h4, stig
    ))
    for t in bary_mjd_jug
])

# Compare with PINT binary delay
delay_diff_jug_time_us = (jug_binary_delay_jug_time_sec - pint_binary_delay_sec) * 1e6
print(f"\nBinary delay difference using JUG's bary time (JUG - PINT):")
print(f"  Mean: {np.mean(delay_diff_jug_time_us):.3f} μs")
print(f"  RMS:  {np.std(delay_diff_jug_time_us):.3f} μs")
print(f"  Range: [{np.min(delay_diff_jug_time_us):.3f}, {np.max(delay_diff_jug_time_us):.3f}] μs")

# Check if the delay difference correlates with orbital phase
print("\n" + "="*80)
print("ORBITAL PHASE ANALYSIS")
print("="*80)

# Compute orbital phase
orbital_phase = ((bary_mjd_pint - t0) / pb) % 1.0

# Bin by phase
n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
phase_bin_idx = np.digitize(orbital_phase, bins) - 1

print(f"\nPhase-binned binary delay difference (JUG@JUG_time - PINT):")
print(f"  Phase     | Mean diff (μs) | Count")
print(f"  ----------|----------------|------")
for i in range(n_bins):
    mask = phase_bin_idx == i
    if np.sum(mask) > 0:
        mean_diff = np.mean(delay_diff_jug_time_us[mask])
        count = np.sum(mask)
        bar = "*" * int(abs(mean_diff) / 0.2) if abs(mean_diff) > 0.05 else ""
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}   |  {mean_diff:+.3f}          | {count:4d}  {bar}")
