import numpy as np
import matplotlib.pyplot as plt
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
from jug.residuals.simple_calculator import compute_residuals_simple

# Load data
par_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par"
tim_file = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim"

print("Loading JUG residuals...")
jug_results = compute_residuals_simple(par_file, tim_file)
jug_res = jug_results['residuals_us']
mjds = jug_results['tdb_mjd']

print("\nLoading PINT residuals...")
pint_model = get_model(par_file)
pint_toas = get_TOAs(tim_file, planets=True, ephem='DE440')
pint_resids = Residuals(pint_toas, pint_model)
pint_res = pint_resids.time_resids.to_value('us')

# Difference
diff = jug_res - pint_res

print(f"\n=== Difference Analysis ===")
print(f"Mean: {np.mean(diff*1e3):.3f} ns")
print(f"Std: {np.std(diff*1e3):.3f} ns")
print(f"Max abs: {np.max(np.abs(diff*1e3)):.3f} ns")
print(f"Min: {np.min(diff*1e3):.3f} ns")
print(f"Max: {np.max(diff*1e3):.3f} ns")

# Plot the trend
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Residuals
axes[0].plot(mjds, jug_res, 'o', alpha=0.5, ms=2, label='JUG')
axes[0].plot(mjds, pint_res, 'x', alpha=0.5, ms=2, label='PINT')
axes[0].set_ylabel('Residual (μs)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title('JUG vs PINT Residuals')

# Difference vs time
axes[1].plot(mjds, diff*1e3, 'o', ms=2)
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1].set_ylabel('JUG - PINT (ns)')
axes[1].set_xlabel('MJD (TDB)')
axes[1].grid(True, alpha=0.3)
axes[1].set_title(f'Difference (mean={np.mean(diff*1e3):.2f} ns, std={np.std(diff*1e3):.2f} ns)')

# Check if it's related to binary phase
pb_days = float(pint_model.PB.value)
tasc_mjd = float(pint_model.TASC.value)

# Compute binary phase
binary_phase = np.mod((mjds - tasc_mjd) / pb_days, 1.0)

axes[2].scatter(binary_phase, diff*1e3, c=mjds, s=2, cmap='viridis')
axes[2].set_xlabel('Binary Phase')
axes[2].set_ylabel('JUG - PINT (ns)')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color='red', linestyle='--', alpha=0.5)
cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
cbar.set_label('MJD (TDB)')
axes[2].set_title('Difference vs Binary Phase')

plt.tight_layout()
plt.savefig('debug_time_trend.png', dpi=150, bbox_inches='tight')
print("\nSaved debug_time_trend.png")

# Check correlations
print(f"\n=== Correlation Tests ===")
print(f"Corr(diff, MJD): {np.corrcoef(diff, mjds)[0,1]:.4f}")
print(f"Corr(diff, binary_phase): {np.corrcoef(diff, binary_phase)[0,1]:.4f}")

# Look for any time-dependent pattern
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(mjds, diff*1e3)
print(f"\n=== Linear Fit (Difference vs Time) ===")
print(f"Slope: {slope:.6f} ns/day")
print(f"R²: {r_value**2:.6f}")
print(f"Total drift over {(mjds.max() - mjds.min()):.0f} days: {slope * (mjds.max() - mjds.min()):.2f} ns")
