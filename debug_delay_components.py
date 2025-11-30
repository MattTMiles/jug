#!/usr/bin/env python3
"""
Compare individual delay components between JUG and PINT to isolate trend source.
"""
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# PINT imports
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# JUG imports - need to extract intermediate delays
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file
from jug.time.clock_chain import clock_correction_seconds
from jug.delays.bary import roemer_delay_seconds, shapiro_delay_seconds, einstein_delay_TT_to_TDB_seconds
from jug.delays.combined import ell1_delay_with_shapiro
import jax.numpy as jnp

# File paths
par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("=" * 80)
print("COMPONENT-LEVEL DELAY COMPARISON: JUG vs PINT")
print("=" * 80)

# ======================
# PINT COMPONENT EXTRACTION
# ======================
print("\n1. Loading PINT components...")
pint_model = get_model(par_file)
pint_toas = get_TOAs(tim_file, planets=True, ephem='DE440', bipm_version='BIPM2024')

# Get delay components from PINT
# Clock corrections: toas.table['delta_pulse_number'] has clock corrections applied
pint_clock = (pint_toas.table['tdb'] - pint_toas.table['mjd']).to_value('s')  # TDB - UTC includes clock

# Barycentric delays
pint_delays = pint_model.delay(pint_toas)
pint_roemer = pint_delays.to_value('s')  # Total SSB delay

# Get individual components if possible
try:
    # PINT stores components internally in delay calculation
    from pint.models.timing_model import Component
    # This is tricky - PINT doesn't expose components easily
    # We'll compute total delay and compare
    print("   (Note: PINT doesn't easily expose individual components)")
except:
    pass

pint_mjd_tdb = pint_toas.table['tdbld'].to_value('mjd')[:, 0] + pint_toas.table['tdbld'].to_value('mjd')[:, 1]

print(f"   Loaded {len(pint_mjd_tdb)} TOAs")

# ======================
# JUG COMPONENT EXTRACTION
# ======================
print("\n2. Computing JUG components...")

# Read files
params = parse_par_file(par_file)
tim_data = parse_tim_file(tim_file)

mjd_utc = tim_data['mjd']
freq_mhz = tim_data['freq_mhz']
obs_name = tim_data['obs']
errors_us = tim_data['error_us']

print(f"   Loaded {len(mjd_utc)} TOAs")

# Step 1: Clock corrections
jug_clock = np.array([
    clock_correction_seconds(mjd, obs) 
    for mjd, obs in zip(mjd_utc, obs_name)
])

mjd_tt = mjd_utc + jug_clock / 86400.0

print(f"   ✓ Clock corrections computed")

# Step 2: Load ephemeris and observatory data
from jug.ephemeris.kernel import KernelEphemeris
from jug.observatory.positions import get_observatory_posvel
import astropy.units as u
from astropy.time import Time

eph = KernelEphemeris('/home/mattm/soft/JUG/data/ephemeris/de440s.bsp')

# Compute observer positions in SSB frame
obs_pos_ssb = []
obs_vel_ssb = []
for mjd, obs in zip(mjd_tt, obs_name):
    pos, vel = get_observatory_posvel(obs, Time(mjd, format='mjd', scale='tt'), eph)
    obs_pos_ssb.append(pos)
    obs_vel_ssb.append(vel)

obs_pos_ssb = np.array(obs_pos_ssb)  # meters
obs_vel_ssb = np.array(obs_vel_ssb)  # m/s

print(f"   ✓ Observatory positions computed")

# Step 3: Sky position
from astropy.coordinates import SkyCoord
ra_deg = float(params['RAJ'])
dec_deg = float(params['DECJ'])
coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
ra_rad = coord.ra.radian
dec_rad = coord.dec.radian

# Unit vector to pulsar
n_hat = np.array([
    np.cos(dec_rad) * np.cos(ra_rad),
    np.cos(dec_rad) * np.sin(ra_rad),
    np.sin(dec_rad)
])

# Step 4: Roemer delay (light travel time)
from jug.delays.bary import roemer_delay_seconds
jug_roemer = np.array([
    roemer_delay_seconds(pos, n_hat) 
    for pos in obs_pos_ssb
])

print(f"   ✓ Roemer delay computed")

# Step 5: Einstein delay (TT -> TDB)
from jug.delays.bary import einstein_delay_TT_to_TDB_seconds
jug_einstein = np.array([
    einstein_delay_TT_to_TDB_seconds(mjd, pos, vel, eph)
    for mjd, pos, vel in zip(mjd_tt, obs_pos_ssb, obs_vel_ssb)
])

print(f"   ✓ Einstein delay computed")

# Step 6: Shapiro delay (solar gravitational)
from jug.delays.bary import shapiro_delay_seconds
jug_shapiro = np.array([
    shapiro_delay_seconds(mjd, pos, n_hat, eph)
    for mjd, pos in zip(mjd_tt, obs_pos_ssb)
])

print(f"   ✓ Shapiro delay computed")

# Step 7: Binary delays (ELL1)
from jug.delays.combined import ell1_delay_with_shapiro

# Extract binary parameters
PB = float(params['PB'])
A1 = float(params['A1'])
TASC = float(params['TASC'])
EPS1 = float(params.get('EPS1', 0.0))
EPS2 = float(params.get('EPS2', 0.0))
GAMMA = float(params.get('GAMMA', 0.0))
PBDOT = float(params.get('PBDOT', 0.0))
XDOT = float(params.get('XDOT', 0.0))

# Shapiro parameters
if 'M2' in params and 'SINI' in params:
    T_SUN_SEC = 4.925490947e-6
    r_shap = T_SUN_SEC * float(params['M2'])
    s_shap = float(params['SINI'])
else:
    r_shap = 0.0
    s_shap = 0.0

# Compute TDB times for binary calculation
mjd_tdb = mjd_tt + (jug_roemer + jug_einstein + jug_shapiro) / 86400.0

# Binary delays
jug_binary = []
for mjd in mjd_tdb:
    delay_sec = ell1_delay_with_shapiro(
        float(mjd), PB, A1, TASC, EPS1, EPS2,
        GAMMA, PBDOT, XDOT, r_shap, s_shap
    )
    jug_binary.append(float(delay_sec))
jug_binary = np.array(jug_binary)

print(f"   ✓ Binary delay computed")

# ======================
# COMPUTE DIFFERENCES
# ======================
print("\n3. Computing component differences...")

# Total barycentric correction in JUG
jug_bary_total = jug_roemer + jug_einstein + jug_shapiro

# For PINT, we only have total delay
# Let's compare what we can

# Clock difference (hard to extract from PINT cleanly)
# Binary difference (hard to extract from PINT cleanly)

# Instead: compute total delay difference
# JUG: clock + bary + binary
# PINT: pint_delays

# Actually, let's just look at trends in JUG components themselves
print("\n4. Analyzing trends in JUG components...")

late_mask = mjd_tdb > 60500
late_mjd = mjd_tdb[late_mask]

components = {
    'Clock': jug_clock[late_mask] * 1e9,  # ns
    'Roemer': jug_roemer[late_mask] * 1e9,
    'Einstein': jug_einstein[late_mask] * 1e9,
    'Shapiro': jug_shapiro[late_mask] * 1e9,
    'Binary': jug_binary[late_mask] * 1e9,
}

print(f"\n{'Component':<15} {'Mean (ns)':<15} {'Std (ns)':<15} {'Trend (ns/day)':<20}")
print("-" * 70)

for name, values in components.items():
    slope, intercept, r_value, p_value, std_err = stats.linregress(late_mjd, values)
    print(f"{name:<15} {np.mean(values):<15.3f} {np.std(values):<15.3f} {slope:<20.6f}")

# Plot
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

for ax, (name, values) in zip(axes, components.items()):
    ax.plot(late_mjd, values, 'o', markersize=2, alpha=0.6)
    
    # Fit linear trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(late_mjd, values)
    fit_line = slope * late_mjd + intercept
    ax.plot(late_mjd, fit_line, 'r-', alpha=0.7, linewidth=2,
            label=f'Trend: {slope:.6f} ns/day')
    
    ax.set_ylabel(f'{name}\n(ns)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f'{name} Delay (MJD > 60500)')

axes[-1].set_xlabel('MJD (TDB)')
plt.tight_layout()
plt.savefig('jug_component_trends.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 80)
print("Plot saved: jug_component_trends.png")
print("=" * 80)
