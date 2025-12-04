"""Plot JUG vs PINT residuals for visual comparison."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import JUG
from jug.residuals.simple_calculator import compute_residuals_simple

# Import PINT
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Test pulsar
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("PLOTTING JUG vs PINT RESIDUALS")
print("="*80)

# Compute JUG residuals
print("\nComputing JUG residuals...")
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']
print(f"JUG: {len(jug_res_us)} TOAs, RMS = {jug_result['unweighted_rms_us']:.3f} μs")

# Compute PINT residuals
print("\nComputing PINT residuals...")
model_pint = get_model(PAR_FILE)
toas_pint = get_TOAs(TIM_FILE, planets=True)
res_pint = Residuals(toas_pint, model_pint)
pint_res_us = res_pint.time_resids.to_value('us')
print(f"PINT: {len(pint_res_us)} TOAs, RMS = {np.std(pint_res_us):.3f} μs")

# Get MJDs for x-axis
mjds = toas_pint.get_mjds().value

# Compute difference
diff_us = jug_res_us - pint_res_us

print("\n" + "="*80)
print("Creating plots...")
print("="*80)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: JUG residuals vs time
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(mjds, jug_res_us, 'b.', markersize=2, alpha=0.6)
ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax1.set_xlabel('MJD')
ax1.set_ylabel('Residual (μs)')
ax1.set_title(f'JUG Residuals (RMS = {np.std(jug_res_us):.3f} μs)')
ax1.grid(True, alpha=0.3)

# Plot 2: PINT residuals vs time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(mjds, pint_res_us, 'r.', markersize=2, alpha=0.6)
ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax2.set_xlabel('MJD')
ax2.set_ylabel('Residual (μs)')
ax2.set_title(f'PINT Residuals (RMS = {np.std(pint_res_us):.3f} μs)')
ax2.grid(True, alpha=0.3)

# Plot 3: Difference (JUG - PINT) vs time
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(mjds, diff_us, 'g.', markersize=2, alpha=0.6)
ax3.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax3.axhline(np.mean(diff_us), color='orange', linestyle='--', linewidth=1,
            label=f'Mean = {np.mean(diff_us):.3f} μs')
ax3.fill_between([mjds.min(), mjds.max()],
                  -0.05, 0.05, alpha=0.2, color='gray',
                  label='Target: ±50 ns')
ax3.set_xlabel('MJD')
ax3.set_ylabel('JUG - PINT (μs)')
ax3.set_title(f'Residual Difference (RMS = {np.std(diff_us):.3f} μs)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Plot 4: Histogram of differences
ax4 = fig.add_subplot(gs[2, 0])
n, bins, patches = ax4.hist(diff_us, bins=50, alpha=0.7, color='green', edgecolor='black')
ax4.axvline(0, color='k', linestyle='--', linewidth=1)
ax4.axvline(np.mean(diff_us), color='orange', linestyle='--', linewidth=1,
            label=f'Mean = {np.mean(diff_us):.3f} μs')
ax4.axvline(np.median(diff_us), color='red', linestyle='--', linewidth=1,
            label=f'Median = {np.median(diff_us):.3f} μs')
ax4.axvspan(-0.05, 0.05, alpha=0.2, color='gray', label='Target: ±50 ns')
ax4.set_xlabel('JUG - PINT (μs)')
ax4.set_ylabel('Count')
ax4.set_title('Histogram of Residual Differences')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Plot 5: Overlay of JUG and PINT residuals
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(mjds, pint_res_us, 'r.', markersize=3, alpha=0.4, label='PINT')
ax5.plot(mjds, jug_res_us, 'b.', markersize=2, alpha=0.4, label='JUG')
ax5.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax5.set_xlabel('MJD')
ax5.set_ylabel('Residual (μs)')
ax5.set_title('JUG vs PINT Residuals (Overlaid)')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# Add overall title
fig.suptitle(f'J1012-4235 DD Binary: JUG vs PINT Comparison\n' +
             f'Target: < 50 ns RMS difference | Actual: {np.std(diff_us)*1000:.1f} ns RMS ' +
             f'({(np.std(diff_us)*1000)/50:.1f}× too large)',
             fontsize=14, fontweight='bold')

# Save figure
output_file = 'jug_pint_residuals_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Also create a zoomed-in plot focused on the difference
fig2, (ax_zoom, ax_hist_zoom) = plt.subplots(2, 1, figsize=(16, 10))

# Zoomed difference plot
ax_zoom.plot(mjds, diff_us, 'g.', markersize=3, alpha=0.6)
ax_zoom.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax_zoom.axhline(np.mean(diff_us), color='orange', linestyle='--', linewidth=1,
                label=f'Mean = {np.mean(diff_us):.3f} μs')
ax_zoom.fill_between([mjds.min(), mjds.max()],
                      -0.05, 0.05, alpha=0.3, color='gray',
                      label='Target: ±50 ns')
ax_zoom.set_xlabel('MJD', fontsize=12)
ax_zoom.set_ylabel('JUG - PINT (μs)', fontsize=12)
ax_zoom.set_title(f'Residual Difference vs Time (RMS = {np.std(diff_us):.3f} μs = {np.std(diff_us)*1000:.1f} ns)',
                  fontsize=14)
ax_zoom.legend(loc='upper right', fontsize=11)
ax_zoom.grid(True, alpha=0.3)

# Calculate percentiles for annotations
p1, p99 = np.percentile(diff_us, [1, 99])
ax_zoom.text(0.02, 0.98, f'1st percentile: {p1:.3f} μs\n99th percentile: {p99:.3f} μs',
             transform=ax_zoom.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

# Detailed histogram
ax_hist_zoom.hist(diff_us, bins=100, alpha=0.7, color='green', edgecolor='black')
ax_hist_zoom.axvline(0, color='k', linestyle='--', linewidth=1, label='Zero')
ax_hist_zoom.axvline(np.mean(diff_us), color='orange', linestyle='--', linewidth=2,
                      label=f'Mean = {np.mean(diff_us):.3f} μs')
ax_hist_zoom.axvline(np.median(diff_us), color='red', linestyle='--', linewidth=2,
                      label=f'Median = {np.median(diff_us):.3f} μs')
ax_hist_zoom.axvspan(-0.05, 0.05, alpha=0.3, color='gray', label='Target: ±50 ns')
ax_hist_zoom.set_xlabel('JUG - PINT (μs)', fontsize=12)
ax_hist_zoom.set_ylabel('Count', fontsize=12)
ax_hist_zoom.set_title('Distribution of Residual Differences', fontsize=14)
ax_hist_zoom.legend(loc='upper right', fontsize=11)
ax_hist_zoom.grid(True, alpha=0.3, axis='y')

# Add statistics box
stats_text = (f'Statistics:\n'
              f'  Mean: {np.mean(diff_us):.3f} μs\n'
              f'  Median: {np.median(diff_us):.3f} μs\n'
              f'  Std Dev: {np.std(diff_us):.3f} μs\n'
              f'  Min: {np.min(diff_us):.3f} μs\n'
              f'  Max: {np.max(diff_us):.3f} μs')
ax_hist_zoom.text(0.98, 0.98, stats_text,
                  transform=ax_hist_zoom.transAxes,
                  verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                  fontsize=10, family='monospace')

fig2.suptitle(f'J1012-4235: Detailed Difference Analysis\n' +
              f'Target: < 50 ns RMS | Actual: {np.std(diff_us)*1000:.1f} ns RMS',
              fontsize=14, fontweight='bold')

plt.tight_layout()

output_file2 = 'jug_pint_difference_detailed.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"Detailed plot saved to: {output_file2}")

print("\n" + "="*80)
print("DONE")
print("="*80)
print(f"\nPlots saved:")
print(f"  1. {output_file}")
print(f"  2. {output_file2}")
print(f"\nSummary:")
print(f"  JUG RMS: {np.std(jug_res_us):.3f} μs")
print(f"  PINT RMS: {np.std(pint_res_us):.3f} μs")
print(f"  Difference RMS: {np.std(diff_us):.3f} μs = {np.std(diff_us)*1000:.1f} ns")
print(f"  Target: < 50 ns")
print(f"  Factor over target: {(np.std(diff_us)*1000)/50:.1f}×")
