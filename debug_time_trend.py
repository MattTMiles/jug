import numpy as np
import matplotlib.pyplot as plt
from jug.residuals.simple_calculator import compute_residuals_simple

# Load J1909-3744 data
par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

# Compute JUG residuals
result = compute_residuals_simple(par_file, tim_file)
mjd = np.array(result['tdb_mjd'], dtype=np.float64)
res_us = np.array(result['residuals_us'], dtype=np.float64)

# Also get PINT residuals for comparison
import pint.models
import pint.toa

model = pint.models.get_model(par_file)
toas = pint.toa.get_TOAs(tim_file, planets=True, include_bipm=True, ephem='DE440')
pint_res = np.array(model.phase(toas).frac * (1.0 / model.F0.quantity.value) * 1e6, dtype=np.float64)

# Focus on late data (MJD > 60500)
mask = mjd > 60500
mjd_late = mjd[mask]
res_late = res_us[mask]
pint_res_late = pint_res[mask]

# Fit linear trend to difference
diff_late = res_late - pint_res_late
p = np.polyfit(mjd_late, diff_late, 1)
print(f"\n{'='*70}")
print(f"LINEAR TREND ANALYSIS IN LATE DATA (MJD > 60500)")
print(f"{'='*70}")
print(f"Linear trend: {p[0]:.6f} μs/day = {p[0]*365.25:.3f} μs/year")
print(f"Offset at MJD 60500: {np.polyval(p, 60500):.3f} μs")
print(f"Offset at MJD {mjd_late[-1]:.1f}: {np.polyval(p, mjd_late[-1]):.3f} μs")
print(f"Total span: {np.polyval(p, mjd_late[-1]) - np.polyval(p, 60500):.3f} μs over {mjd_late[-1] - 60500:.1f} days")
print(f"Detrended RMS: {np.std(diff_late - np.polyval(p, mjd_late)):.4f} μs")
print(f"{'='*70}\n")

# Plot components
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Top: JUG vs PINT residuals in late data
axes[0].plot(mjd_late, res_late, 'o', label='JUG', alpha=0.6, ms=3)
axes[0].plot(mjd_late, pint_res_late, 'x', label='PINT', alpha=0.6, ms=3)
axes[0].set_ylabel('Residual (μs)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Late Data Residuals (MJD > 60500)')

# Middle: Difference with linear fit
axes[1].plot(mjd_late, diff_late, 'o', label='JUG - PINT', alpha=0.6, ms=3)
axes[1].plot(mjd_late, np.polyval(p, mjd_late), 'r-', label=f'Linear fit: {p[0]:.6f} μs/day', lw=2)
axes[1].axhline(0, color='k', ls='--', alpha=0.3)
axes[1].set_ylabel('Difference (μs)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_title(f'JUG - PINT Difference (slope = {p[0]*365.25:.3f} μs/yr)')

# Bottom: Detrended difference
detrended = diff_late - np.polyval(p, mjd_late)
axes[2].plot(mjd_late, detrended, 'o', alpha=0.6, ms=3)
axes[2].axhline(0, color='k', ls='--', alpha=0.3)
axes[2].set_xlabel('MJD (TDB)')
axes[2].set_ylabel('Detrended Diff (μs)')
axes[2].set_title(f'After removing linear trend (RMS={np.std(detrended):.4f} μs)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_time_trend.png', dpi=150, bbox_inches='tight')
print("📊 Plot saved: debug_time_trend.png")
