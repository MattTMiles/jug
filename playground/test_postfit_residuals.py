#!/usr/bin/env python3
"""
Test postfit residual computation.
"""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from pathlib import Path
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized
import tempfile

# Test with wrong par file
par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

print("Testing postfit residual computation...")
print(f"PAR: {par_file}")
print(f"TIM: {tim_file}")
print()

# Compute prefit residuals
print("Computing prefit residuals...")
prefit = compute_residuals_simple(par_file, tim_file, verbose=False)
print(f"  Prefit RMS: {prefit['rms_us']:.6f} μs")

# Fit F0 and F1
print("\nFitting F0, F1...")
fit_result = fit_parameters_optimized(
    par_file, tim_file, ['F0', 'F1'], verbose=False, device='cpu'
)
print(f"  Fitted F0: {fit_result['final_params']['F0']:.15f} Hz")
print(f"  Fitted F1: {fit_result['final_params']['F1']:.6e} Hz/s")
print(f"  Fit RMS: {fit_result['final_rms']:.6f} μs")

# Update par file with fitted parameters
print("\nUpdating par file with fitted parameters...")
with open(par_file, 'r') as f:
    par_lines = f.readlines()

updated_lines = []
fitted_params = fit_result['final_params']

for line in par_lines:
    line_stripped = line.strip()
    if not line_stripped or line_stripped.startswith('#'):
        updated_lines.append(line)
        continue

    parts = line_stripped.split()
    if parts:
        param_name = parts[0]
        if param_name in fitted_params:
            fitted_value = fitted_params[param_name]
            if param_name == 'F0':
                new_line = f"{param_name:<12} {fitted_value:.15f} 1"
            else:
                new_line = f"{param_name:<12} {fitted_value:.15e} 1"
            updated_lines.append(new_line + '\n')
        else:
            updated_lines.append(line)
    else:
        updated_lines.append(line)

# Write temporary par file
with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
    tmp.writelines(updated_lines)
    tmp_par_path = Path(tmp.name)

try:
    # Compute postfit residuals
    print("Computing postfit residuals...")
    postfit = compute_residuals_simple(tmp_par_path, tim_file, verbose=False)
    print(f"  Postfit RMS: {postfit['rms_us']:.6f} μs")

    print("\nComparison:")
    print(f"  Prefit RMS:  {prefit['rms_us']:.6f} μs")
    print(f"  Postfit RMS: {postfit['rms_us']:.6f} μs")
    print(f"  Improvement: {prefit['rms_us'] - postfit['rms_us']:.6f} μs")
    print(f"  Factor: {prefit['rms_us'] / postfit['rms_us']:.2f}×")

    if abs(postfit['rms_us'] - fit_result['final_rms']) < 0.001:
        print("\n✓ Postfit RMS matches fit result!")
    else:
        print(f"\n✗ Postfit RMS mismatch: {postfit['rms_us']:.6f} vs {fit_result['final_rms']:.6f}")

finally:
    tmp_par_path.unlink()

print("\nTest complete!")
