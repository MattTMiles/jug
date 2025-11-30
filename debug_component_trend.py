"""
Systematic component-by-component comparison: JUG vs PINT
Goal: Find which delay term causes the linear trend in late data
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
from jug.residuals.simple_calculator import compute_residuals

# Load data
parfile = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
timfile = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

print("Loading PINT model and TOAs...")
model_pint = get_model(parfile)
toas_pint = get_TOAs(timfile, ephem="DE440", bipm_version="BIPM2024", include_bipm=True, planets=True)

# Get PINT residuals
res_pint = Residuals(toas_pint, model_pint)
pint_toas_mjd = toas_pint.get_mjds().value
pint_residuals = res_pint.time_resids.to(u.us).value

print("\nComputing JUG residuals...")
jug_res, jug_toas = compute_residuals(parfile, timfile, bipm_version='BIPM2024')

# Match TOAs by MJD
jug_mjds = jug_res['mjd']
jug_residuals = jug_res['residual_us']

# Find common TOAs
mask_pint = np.isin(pint_toas_mjd, jug_mjds)
mask_jug = np.isin(jug_mjds, pint_toas_mjd)

common_mjd = pint_toas_mjd[mask_pint]
pint_res_common = pint_residuals[mask_pint]
jug_res_common = jug_residuals[mask_jug]

# Compute difference
diff = jug_res_common - pint_res_common

# Focus on late data (MJD > 60000)
late_mask = common_mjd > 60000
late_mjd = common_mjd[late_mask]
late_diff = diff[late_mask]

# Fit linear trend to late data
if len(late_diff) > 0:
    coeffs = np.polyfit(late_mjd, late_diff, 1)
    trend = np.poly1d(coeffs)
    print(f"\nLate data (MJD > 60000) linear trend:")
    print(f"  Slope: {coeffs[0]*1000:.6f} ns/day")
    print(f"  Intercept: {coeffs[1]:.3f} ns")
    print(f"  Total drift over {late_mjd.max()-late_mjd.min():.1f} days: {coeffs[0]*(late_mjd.max()-late_mjd.min())*1000:.1f} ns")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Full difference
axes[0].plot(common_mjd, diff, 'o', ms=2, alpha=0.5)
axes[0].axhline(0, color='red', ls='--', lw=1)
axes[0].axvline(60000, color='orange', ls='--', lw=1, label='MJD 60000')
axes[0].set_xlabel('MJD')
axes[0].set_ylabel('JUG - PINT Residual (ns)')
axes[0].set_title('Full Residual Difference')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Late data with trend line
axes[1].plot(late_mjd, late_diff, 'o', ms=2, alpha=0.5, label='Data')
if len(late_diff) > 0:
    axes[1].plot(late_mjd, trend(late_mjd), 'r-', lw=2, label=f'Linear fit: {coeffs[0]*1000:.3f} ns/day')
axes[1].axhline(0, color='gray', ls='--', lw=1)
axes[1].set_xlabel('MJD')
axes[1].set_ylabel('JUG - PINT Residual (ns)')
axes[1].set_title('Late Data (MJD > 60000) - Linear Trend Analysis')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_time_trend.png', dpi=150, bbox_inches='tight')
print("\nSaved plot: debug_time_trend.png")

print("\n" + "="*60)
print("NEXT STEP: Compare individual delay components")
print("="*60)
print("\nTo isolate the source of the trend, I'll need to compare:")
print("  1. Clock corrections (observatory → TT)")
print("  2. Geometric (Roemer) delays")
print("  3. Einstein delays")
print("  4. Shapiro delays")
print("  5. Binary delays")
print("  6. DM delays")

