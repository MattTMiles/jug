"""Compare individual delay components between JUG and PINT."""

import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Use J1909-3744 for simpler binary model (ELL1)
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim"

print("="*80)
print("DELAY COMPONENT COMPARISON: JUG vs PINT")
print("="*80)

# Load PINT model
print("\nLoading PINT model...")
model = get_model(PAR_FILE)
toas = get_TOAs(TIM_FILE, planets=True)

# Get times
mjds = toas.get_mjds().value
tdb_pint = np.array([float(t) for t in toas.table['tdbld']], dtype=np.float64)

# Get PINT delay components
print("Computing PINT delays...")

# Get barycentric TOAs (includes Roemer + Einstein + Shapiro)
bary_toas = model.get_barycentric_toas(toas)
bary_mjd_pint = np.array([float(t) for t in bary_toas], dtype=np.float64)

# Total delay before binary (in days)
delay_before_binary_days = bary_mjd_pint - tdb_pint
delay_before_binary_us = delay_before_binary_days * 86400 * 1e6

# Get individual PINT delay components if possible
# This is tricky - PINT doesn't expose all components separately
# But we can get the total barycentric correction

print(f"  PINT delay before binary: {np.mean(delay_before_binary_us):.3f} ± {np.std(delay_before_binary_us):.3f} μs")

# Now let's try to get planetary Shapiro from PINT
# This requires digging into PINT's internal calculations
from pint.models.timing_model import Component
from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time
from astropy import units as u
import astropy.constants as const

print("\nComputing planetary Shapiro delays (PINT method)...")

# Get Sun and planet positions
times = Time(tdb_pint, format='mjd', scale='tdb')

# Get observatory positions (SSB frame)
from pint.solar_system_ephemerides import objPosVel_wrt_SSB
obs_pos_pint = np.array([objPosVel_wrt_SSB(p, times[i])[0].xyz.to(u.km).value
                         for i, p in enumerate(toas.get_obss())])

# Pulsar direction
from pint.utils import PosVel
ra_rad = float(model.RAJ.quantity.to(u.rad).value)
dec_rad = float(model.DECJ.quantity.to(u.rad).value)
psr_dir = np.array([
    np.cos(dec_rad) * np.cos(ra_rad),
    np.cos(dec_rad) * np.sin(ra_rad),
    np.sin(dec_rad)
])

# Compute planetary Shapiro manually using PINT's positions
from pint.solar_system_ephemerides import get_tdb_tt_ephem_geocenter
import astropy.coordinates as coords

sun_shapiro_pint = []
planet_shapiro_pint = []

T_SUN = 4.925490947e-6  # seconds
T_PLANETS = {
    'jupiter': 9.546487e-8,
    'saturn': 2.856799e-8,
    'venus': 2.437086e-9,
    'uranus': 4.365086e-9,
    'neptune': 5.163768e-9,
}

print("  Computing for first 100 TOAs...")
n_test = min(100, len(toas))

for i in range(n_test):
    # Sun
    sun_pos = get_body_barycentric_posvel('sun', times[i])[0].xyz.to(u.km).value
    obs_sun = sun_pos - obs_pos_pint[i]
    r_sun = np.linalg.norm(obs_sun)
    cos_theta_sun = np.dot(obs_sun / r_sun, psr_dir)
    sun_shap = -2.0 * T_SUN * np.log((r_sun/1000 + np.dot(obs_sun/1000, psr_dir)))  # Convert km to lt-s
    sun_shapiro_pint.append(sun_shap)

    # Planets
    planet_total = 0.0
    for planet, T_P in T_PLANETS.items():
        p_pos = get_body_barycentric_posvel(planet, times[i])[0].xyz.to(u.km).value
        obs_p = p_pos - obs_pos_pint[i]
        r_p = np.linalg.norm(obs_p)
        p_shap = -2.0 * T_P * np.log((r_p/1000 + np.dot(obs_p/1000, psr_dir)))
        planet_total += p_shap
    planet_shapiro_pint.append(planet_total)

sun_shapiro_pint = np.array(sun_shapiro_pint)
planet_shapiro_pint = np.array(planet_shapiro_pint)

print(f"\n  Sun Shapiro (manual PINT calc):")
print(f"    Mean: {np.mean(sun_shapiro_pint)*1e6:.6f} μs")
print(f"    Range: [{np.min(sun_shapiro_pint)*1e6:.6f}, {np.max(sun_shapiro_pint)*1e6:.6f}] μs")

print(f"\n  Planet Shapiro (manual PINT calc):")
print(f"    Mean: {np.mean(planet_shapiro_pint)*1e6:.6f} μs")
print(f"    Range: [{np.min(planet_shapiro_pint)*1e6:.6f}, {np.max(planet_shapiro_pint)*1e6:.6f}] μs")

# Time evolution
dt_years = (mjds[:n_test] - mjds[0]) / 365.25

# Analyze trends
coeffs_sun = np.polyfit(dt_years, sun_shapiro_pint*1e6, 1)
coeffs_planet = np.polyfit(dt_years, planet_shapiro_pint*1e6, 1)

print(f"\n  Sun Shapiro trend: {coeffs_sun[0]:.6f} μs/year")
print(f"  Planet Shapiro trend: {coeffs_planet[0]:.6f} μs/year")
print(f"  Total Shapiro trend: {coeffs_sun[0] + coeffs_planet[0]:.6f} μs/year")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

ax = axes[0]
ax.plot(dt_years, sun_shapiro_pint * 1e6, 'b.', markersize=3, alpha=0.6, label='Sun Shapiro')
ax.set_ylabel('Sun Shapiro (μs)')
ax.set_title('Solar Shapiro Delay vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(dt_years, planet_shapiro_pint * 1e6, 'r.', markersize=3, alpha=0.6, label='Planet Shapiro')
ax.set_ylabel('Planet Shapiro (μs)')
ax.set_title('Planetary Shapiro Delay vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2]
total_shapiro = (sun_shapiro_pint + planet_shapiro_pint) * 1e6
ax.plot(dt_years, total_shapiro, 'g.', markersize=3, alpha=0.6, label='Total Shapiro')
ax.set_xlabel('Time since first TOA (years)')
ax.set_ylabel('Total Shapiro (μs)')
ax.set_title('Total Shapiro Delay vs Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shapiro_delay_components.png', dpi=150)
print(f"\nPlot saved to: shapiro_delay_components.png")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThe 1 μs/year drift we observe in residual differences could be due to:")
print("  1. Planetary Shapiro delays computed incorrectly in JUG")
print("  2. Solar Shapiro delay computed incorrectly")
print("  3. Different ephemeris (DE440 vs DE421)")
print("  4. Einstein delay calculation error")
