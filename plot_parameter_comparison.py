#!/usr/bin/env python3
"""
Compare fitted parameters (F0, F1) and their uncertainties from JUG, PINT, and Tempo2.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Suppress warnings
import os
os.environ['JAX_LOG_COMPILES'] = '0'

print("="*80)
print("PARAMETER COMPARISON: JUG vs PINT vs Tempo2")
print("="*80)

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")
tempo2_par = Path("data/pulsars/J1909-3744_tdb_refit_F0_F1.par")

# ============================================================================
# Get JUG values
# ============================================================================
print("\nRunning JUG fit...")
from jug.fitting.optimized_fitter import fit_parameters_optimized

jug_result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=25,
    verbose=False,
    device='cpu'
)

# Check convergence
if not jug_result['converged']:
    print(f"  ⚠️  Warning: JUG did not converge after {jug_result['iterations']} iterations")
else:
    print(f"  ✓ Converged after {jug_result['iterations']} iterations")

jug_f0 = jug_result['final_params']['F0']
jug_f1 = jug_result['final_params']['F1']
jug_f0_err = jug_result['uncertainties']['F0']
jug_f1_err = jug_result['uncertainties']['F1']

print(f"JUG Results:")
print(f"  F0 = {jug_f0:.20e} ± {jug_f0_err:.2e} Hz")
print(f"  F1 = {jug_f1:.20e} ± {jug_f1_err:.2e} Hz/s")

# ============================================================================
# Get PINT values
# ============================================================================
print("\nRunning PINT fit...")
import pint.models
import pint.toa
import pint.fitter

model = pint.models.get_model(str(par_file))
toas = pint.toa.get_TOAs(str(tim_file), planets=True, ephem='DE440')

fitter = pint.fitter.WLSFitter(toas, model)
fitter.model.free_params = ['F0', 'F1']
fitter.fit_toas(maxiter=25)

# Check convergence
print(f"  PINT chi2: {fitter.resids.chi2:.2f}")
print(f"  PINT reduced chi2: {fitter.resids.chi2_reduced:.2f}")

pint_f0 = fitter.model.F0.value
pint_f1 = fitter.model.F1.value
pint_f0_err = fitter.model.F0.uncertainty.value
pint_f1_err = fitter.model.F1.uncertainty.value

print(f"PINT Results:")
print(f"  F0 = {pint_f0:.20e} ± {pint_f0_err:.2e} Hz")
print(f"  F1 = {pint_f1:.20e} ± {pint_f1_err:.2e} Hz/s")

# ============================================================================
# Get Tempo2 values
# ============================================================================
print("\nReading Tempo2 results...")

# Parse Tempo2 par file directly to get uncertainties
tempo2_f0 = None
tempo2_f1 = None
tempo2_f0_err = None
tempo2_f1_err = None

with open(tempo2_par, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == 'F0':
            tempo2_f0 = float(parts[1])
            tempo2_f0_err = float(parts[3])
        elif len(parts) >= 4 and parts[0] == 'F1':
            tempo2_f1 = float(parts[1])
            tempo2_f1_err = float(parts[3])

print(f"Tempo2 Results:")
print(f"  F0 = {tempo2_f0:.20e} ± {tempo2_f0_err:.2e} Hz")
print(f"  F1 = {tempo2_f1:.20e} ± {tempo2_f1_err:.2e} Hz/s")

# ============================================================================
# Create comparison plot with error bars
# ============================================================================
print("\nCreating comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Parameter Comparison: JUG vs PINT vs Tempo2\n(J1909-3744, F0+F1 Fit)', 
             fontsize=16, fontweight='bold')

tools = ['JUG', 'PINT', 'Tempo2']
colors = ['#2E86AB', '#A23B72', '#F18F01']

# Prepare data
f0_values = np.array([jug_f0, pint_f0, tempo2_f0])
f0_errors = np.array([jug_f0_err, pint_f0_err, tempo2_f0_err])
f1_values = np.array([jug_f1, pint_f1, tempo2_f1])
f1_errors = np.array([jug_f1_err, pint_f1_err, tempo2_f1_err])

# Use reference value (mean) for offsets
f0_ref = np.mean(f0_values)
f1_ref = np.mean(f1_values)

f0_offsets = (f0_values - f0_ref) * 1e14  # in units of 1e-14 Hz
f0_errors_scaled = f0_errors * 1e14

f1_offsets = (f1_values - f1_ref) * 1e22  # in units of 1e-22 Hz/s
f1_errors_scaled = f1_errors * 1e22

# ============================================================================
# Plot 1: F0 with error bars
# ============================================================================
ax = axes[0]
x_pos = np.arange(len(tools))

# Plot error bars and points
for i, (x, y, yerr, color, label) in enumerate(zip(x_pos, f0_offsets, f0_errors_scaled, colors, tools)):
    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=12, capsize=8, capthick=2,
                color=color, ecolor=color, label=label, linewidth=2, alpha=0.8)

ax.set_ylabel('F0 Offset [×10⁻¹⁴ Hz]', fontsize=13, fontweight='bold')
ax.set_title(f'F0 Parameter\n(Reference: {f0_ref:.20f} Hz)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(tools, fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')

# Add value labels above points
for i, (x, y, yerr) in enumerate(zip(x_pos, f0_offsets, f0_errors_scaled)):
    ax.text(x, y + yerr + 0.1, f'{y:.2f}\n±{yerr:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# Plot 2: F1 with error bars
# ============================================================================
ax = axes[1]

# Plot error bars and points
for i, (x, y, yerr, color, label) in enumerate(zip(x_pos, f1_offsets, f1_errors_scaled, colors, tools)):
    ax.errorbar(x, y, yerr=yerr, fmt='o', markersize=12, capsize=8, capthick=2,
                color=color, ecolor=color, label=label, linewidth=2, alpha=0.8)

ax.set_ylabel('F1 Offset [×10⁻²² Hz/s]', fontsize=13, fontweight='bold')
ax.set_title(f'F1 Parameter\n(Reference: {f1_ref:.20e} Hz/s)', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(tools, fontsize=12, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=2, alpha=0.5, zorder=0)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='best')

# Add value labels above points
for i, (x, y, yerr) in enumerate(zip(x_pos, f1_offsets, f1_errors_scaled)):
    ax.text(x, y + yerr + 0.15, f'{y:.2f}\n±{yerr:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# Save figure
output_file = Path("parameter_comparison_jug_pint_tempo2.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_file}")

plt.show()

# ============================================================================
# Statistical comparison
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL COMPARISON")
print("="*80)

print("\nF0 Parameter:")
print(f"  Mean:   {f0_mean:.20e} Hz")
print(f"  Std:    {np.std(f0_values):.2e} Hz")
print(f"  Range:  {max(f0_values) - min(f0_values):.2e} Hz")
print(f"\n  Differences from mean:")
for tool, val in zip(tools, f0_values):
    print(f"    {tool:8s}: {(val - f0_mean):.2e} Hz ({(val-f0_mean)/f0_mean*1e15:.3f} ppq)")

print("\nF1 Parameter:")
print(f"  Mean:   {f1_mean:.20e} Hz/s")
print(f"  Std:    {np.std(f1_values):.2e} Hz/s")
print(f"  Range:  {abs(max(f1_values) - min(f1_values)):.2e} Hz/s")
print(f"\n  Differences from mean:")
for tool, val in zip(tools, f1_values):
    print(f"    {tool:8s}: {(val - f1_mean):.2e} Hz/s ({(val-f1_mean)/abs(f1_mean)*1e6:.3f} ppm)")

print("\nF0 Uncertainties:")
for tool, err in zip(tools, f0_errors):
    print(f"  {tool:8s}: {err:.2e} Hz")

print("\nF1 Uncertainties:")
for tool, err in zip(tools, f1_errors):
    print(f"  {tool:8s}: {err:.2e} Hz/s")

# Agreement assessment
print("\n" + "="*80)
print("AGREEMENT ASSESSMENT")
print("="*80)

# Check if values agree within errors
f0_max_diff = max(f0_values) - min(f0_values)
f0_avg_err = np.mean(f0_errors)
f0_agreement = f0_max_diff / f0_avg_err

f1_max_diff = abs(max(f1_values) - min(f1_values))
f1_avg_err = np.mean(f1_errors)
f1_agreement = f1_max_diff / f1_avg_err

print(f"\nF0 Agreement:")
print(f"  Maximum difference: {f0_max_diff:.2e} Hz")
print(f"  Average uncertainty: {f0_avg_err:.2e} Hz")
print(f"  Ratio (diff/error): {f0_agreement:.2f}σ")
if f0_agreement < 1:
    print(f"  ✅ All values agree within 1σ!")
elif f0_agreement < 3:
    print(f"  ✅ All values agree within 3σ")
else:
    print(f"  ⚠️  Some disagreement beyond 3σ")

print(f"\nF1 Agreement:")
print(f"  Maximum difference: {f1_max_diff:.2e} Hz/s")
print(f"  Average uncertainty: {f1_avg_err:.2e} Hz/s")
print(f"  Ratio (diff/error): {f1_agreement:.2f}σ")
if f1_agreement < 1:
    print(f"  ✅ All values agree within 1σ!")
elif f1_agreement < 3:
    print(f"  ✅ All values agree within 3σ")
else:
    print(f"  ⚠️  Some disagreement beyond 3σ")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if f0_agreement < 3 and f1_agreement < 3:
    print("\n✅ Excellent agreement! All three tools produce consistent results.")
    print("   JUG, PINT, and Tempo2 can be used interchangeably for this analysis.")
else:
    print("\n⚠️  Some differences detected. This may be due to:")
    print("   - Different convergence criteria")
    print("   - Numerical precision differences")
    print("   - Slightly different implementations")
