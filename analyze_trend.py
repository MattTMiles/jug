"""Analyze the time-correlated trend between JUG and PINT residuals."""

import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple
from astropy import units as u
from pint.models import get_model
from pint.toa import get_TOAs
import pint.logging

pint.logging.setup(level="WARNING")

# Load data
from pathlib import Path
data_dir = Path("/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb")
parfile = data_dir / "J1909-3744_tdb.par"
timfile = data_dir / "J1909-3744.tim"
clock_dir = Path("/home/mattm/soft/JUG") / "data" / "clock"

print("Computing JUG residuals...")
jug_result = compute_residuals_simple(str(parfile), str(timfile), clock_dir=str(clock_dir))
jug_res_us = jug_result['residuals_us']
jug_mjds = jug_result['tdb_mjd']

print("\nComputing PINT residuals with BIPM2024...")
model = get_model(str(parfile))
model.CLOCK.value = "TT(BIPM2024)"
toas = get_TOAs(str(timfile), model=model, planets=True, ephem="DE440")
from pint.residuals import Residuals
pint_residuals = Residuals(toas, model, use_weighted_mean=False)
pint_res = pint_residuals.time_resids.to(u.us).value
pint_mjds = toas.get_mjds().value

# Compute difference (convert to float64 for polyfit)
diff = np.array(jug_res_us, dtype=np.float64) - np.array(pint_res, dtype=np.float64)

# Focus on late data (MJD > 60000)
late_mask = np.array(jug_mjds, dtype=np.float64) > 60000
diff_late = diff[late_mask]
mjd_late = np.array(jug_mjds[late_mask], dtype=np.float64)

# Fit linear trend
coeffs = np.polyfit(mjd_late, diff_late, 1)
trend = np.poly1d(coeffs)
trend_line = trend(mjd_late)

print(f"\n{'='*60}")
print("LINEAR TREND IN LATE DATA (MJD > 60000):")
print(f"{'='*60}")
print(f"  Slope: {coeffs[0]:.6f} μs/day")
print(f"  Slope: {coeffs[0]*365.25:.6f} μs/year") 
print(f"  Range: {diff_late.min():.3f} to {diff_late.max():.3f} μs")
print(f"  Span: {mjd_late.max() - mjd_late.min():.1f} days")
print(f"  Total drift: {coeffs[0]*(mjd_late.max()-mjd_late.min()):.3f} μs over span")
print(f"{'='*60}\n")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Full dataset
ax1.scatter(jug_mjds, jug_res_us, s=1, alpha=0.3, label='JUG', color='blue')
ax1.scatter(pint_mjds, pint_res, s=1, alpha=0.3, label='PINT', color='orange')
ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('MJD')
ax1.set_ylabel('Residual (μs)')
ax1.set_title('JUG vs PINT Residuals (BIPM2024)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Difference with trend
ax2.scatter(jug_mjds, diff, s=1, alpha=0.3, label='JUG - PINT', color='purple')
ax2.plot(mjd_late, trend_line, 'r-', linewidth=2, 
         label=f'Linear fit (MJD>60000): {coeffs[0]:.6f} μs/day')
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.axvline(60000, color='green', linestyle=':', linewidth=1, alpha=0.5, label='MJD 60000')
ax2.set_xlabel('MJD')
ax2.set_ylabel('Residual Difference (μs)')
ax2.set_title('JUG - PINT: Time-Correlated Trend')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('jug_pint_trend_analysis.png', dpi=150, bbox_inches='tight')
print("📊 Saved: jug_pint_trend_analysis.png\n")

# Let's also check if the trend exists in earlier data
early_mask = jug_mjds < 60000
if np.sum(early_mask) > 100:
    diff_early = diff[early_mask]
    mjd_early = jug_mjds[early_mask]
    coeffs_early = np.polyfit(mjd_early, diff_early, 1)
    print(f"EARLY DATA (MJD < 60000) TREND:")
    print(f"  Slope: {coeffs_early[0]:.6f} μs/day")
    print(f"  Slope: {coeffs_early[0]*365.25:.6f} μs/year")
    print(f"  Compare to late data: {coeffs[0]/coeffs_early[0]:.1f}x larger\n")
