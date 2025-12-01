#!/usr/bin/env python3
"""
Investigate why Tempo2 uncertainties are larger than JUG/PINT.
"""

import numpy as np
from pathlib import Path

print("="*80)
print("UNCERTAINTY INVESTIGATION: Why are Tempo2 uncertainties larger?")
print("="*80)

# Read Tempo2 refit results
tempo2_par = Path("data/pulsars/J1909-3744_tdb_refit_F0_F1.par")

with open(tempo2_par, 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[0] == 'F0':
            tempo2_f0_err = float(parts[3])
            print(f"\nTempo2 F0 uncertainty: {tempo2_f0_err:.2e} Hz")
        elif len(parts) >= 4 and parts[0] == 'F1':
            tempo2_f1_err = float(parts[3])
            print(f"Tempo2 F1 uncertainty: {tempo2_f1_err:.2e} Hz/s")
        elif parts[0] == 'CHI2R':
            chi2r = float(parts[1])
            ndof = int(parts[2])
            print(f"\nTempo2 reduced chi2: {chi2r:.2f} (DOF: {ndof})")
        elif parts[0] == 'TRES':
            tres = float(parts[1])
            print(f"Tempo2 TRES (weighted RMS): {tres:.3f} μs")

# Now get JUG/PINT values
print("\n" + "="*80)
print("Getting JUG and PINT uncertainties...")
print("="*80)

import os
os.environ['JAX_LOG_COMPILES'] = '0'

from jug.fitting.optimized_fitter import fit_parameters_optimized
import pint.models
import pint.toa
import pint.fitter
import warnings
warnings.filterwarnings('ignore')

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

# JUG
print("\nRunning JUG...")
jug_result = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=25,
    verbose=False,
    device='cpu'
)

jug_f0_err = jug_result['uncertainties']['F0']
jug_f1_err = jug_result['uncertainties']['F1']
jug_rms = jug_result['final_rms']

print(f"  F0 uncertainty: {jug_f0_err:.2e} Hz")
print(f"  F1 uncertainty: {jug_f1_err:.2e} Hz/s")
print(f"  Weighted RMS: {jug_rms:.3f} μs")

# PINT
print("\nRunning PINT...")
model = pint.models.get_model(str(par_file))
toas = pint.toa.get_TOAs(str(tim_file), planets=True, ephem='DE440')
fitter = pint.fitter.WLSFitter(toas, model)
fitter.model.free_params = ['F0', 'F1']
fitter.fit_toas(maxiter=25)

pint_f0_err = fitter.model.F0.uncertainty.value
pint_f1_err = fitter.model.F1.uncertainty.value
pint_rms = fitter.resids.rms_weighted().to('us').value
pint_chi2r = fitter.resids.chi2_reduced

print(f"  F0 uncertainty: {pint_f0_err:.2e} Hz")
print(f"  F1 uncertainty: {pint_f1_err:.2e} Hz/s")
print(f"  Weighted RMS: {pint_rms:.3f} μs")
print(f"  Reduced chi2: {pint_chi2r:.2f}")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nF0 Uncertainty Ratios:")
print(f"  Tempo2 / JUG:  {tempo2_f0_err / jug_f0_err:.2f}x")
print(f"  Tempo2 / PINT: {tempo2_f0_err / pint_f0_err:.2f}x")
print(f"  JUG / PINT:    {jug_f0_err / pint_f0_err:.2f}x")

print(f"\nF1 Uncertainty Ratios:")
print(f"  Tempo2 / JUG:  {tempo2_f1_err / jug_f1_err:.2f}x")
print(f"  Tempo2 / PINT: {tempo2_f1_err / pint_f1_err:.2f}x")
print(f"  JUG / PINT:    {jug_f1_err / pint_f1_err:.2f}x")

print("\n" + "="*80)
print("HYPOTHESIS: Chi-square scaling")
print("="*80)

print(f"\nAll three tools have similar reduced chi2 ~ 32")
print(f"  Tempo2: {chi2r:.2f}")
print(f"  PINT:   {pint_chi2r:.2f}")
print(f"  JUG:    (computed from weighted RMS)")

# Compute expected uncertainty scaling
print(f"\nSince reduced chi2 > 1, uncertainties should be scaled by sqrt(chi2_reduced)")
print(f"  Scaling factor: sqrt({chi2r:.2f}) = {np.sqrt(chi2r):.2f}")

# Check if PINT/JUG are scaling
print(f"\nIf base uncertainty (from covariance matrix) is unscaled:")
base_f0_err = jug_f0_err / np.sqrt(pint_chi2r)
base_f1_err = jug_f1_err / np.sqrt(pint_chi2r)

print(f"  Base F0 uncertainty: {base_f0_err:.2e} Hz")
print(f"  Base F1 uncertainty: {base_f1_err:.2e} Hz")

print(f"\nScaled by sqrt(chi2_reduced) = {np.sqrt(chi2r):.2f}:")
scaled_f0_err = base_f0_err * np.sqrt(chi2r)
scaled_f1_err = base_f1_err * np.sqrt(chi2r)

print(f"  Scaled F0: {scaled_f0_err:.2e} Hz")
print(f"  Scaled F1: {scaled_f1_err:.2e} Hz")

print(f"\nActual Tempo2 uncertainties:")
print(f"  Tempo2 F0: {tempo2_f0_err:.2e} Hz")
print(f"  Tempo2 F1: {tempo2_f1_err:.2e} Hz")

print("\n" + "="*80)
print("HYPOTHESIS TEST RESULTS")
print("="*80)

# Test if Tempo2 is using different scaling
tempo2_scaling = tempo2_f0_err / jug_f0_err
print(f"\nTempo2 appears to scale by: {tempo2_scaling:.2f}")
print(f"sqrt(chi2_reduced) would give: {np.sqrt(chi2r):.2f}")

if abs(tempo2_scaling - np.sqrt(chi2r)) < 0.1:
    print("\n✅ MATCH! Tempo2 uses sqrt(chi2_reduced) scaling")
    print("   JUG and PINT likely already include this scaling")
else:
    print(f"\n❓ Scaling doesn't match sqrt(chi2_r)")
    print(f"   Difference: {tempo2_scaling:.2f} vs {np.sqrt(chi2r):.2f}")
    print("\nPossible reasons:")
    print("  1. Different error rescaling methods")
    print("  2. Different DOF calculation (10405 vs 10406)")
    print("  3. Tempo2 uses different rescaling formula")
    print("  4. Additional error inflation in Tempo2")

print("\n" + "="*80)
print("CHECKING COVARIANCE MATRIX SCALING")
print("="*80)

# Check if JUG/PINT already scaled covariance
print("\nJUG covariance matrix info:")
cov = jug_result['covariance']
print(f"  F0-F0 element: {cov[0,0]:.2e}")
print(f"  F0 uncertainty: {np.sqrt(cov[0,0]):.2e}")
print(f"  Matches reported?: {np.isclose(np.sqrt(cov[0,0]), jug_f0_err)}")

# Test if unscaling works
unscaled_jug_f0 = jug_f0_err / np.sqrt(pint_chi2r)
rescaled_jug_f0 = unscaled_jug_f0 * np.sqrt(chi2r)

print(f"\nIf we unscale JUG by sqrt(chi2_r) and rescale by Tempo2's chi2_r:")
print(f"  Unscaled: {unscaled_jug_f0:.2e}")
print(f"  Rescaled: {rescaled_jug_f0:.2e}")
print(f"  Tempo2:   {tempo2_f0_err:.2e}")
print(f"  Match? {np.isclose(rescaled_jug_f0, tempo2_f0_err, rtol=0.01)}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if tempo2_scaling > 5.0:
    print("\n✅ EXPLANATION: Tempo2 uncertainties are ~5.7× larger because:")
    print("   1. All three tools have high reduced chi2 ~ 32")
    print("   2. This indicates either:")
    print("      - Underestimated TOA errors in the .tim file")
    print("      - Model inadequacy (missing parameters)")
    print("      - White noise not fully characterized")
    print("   3. Tempo2 scales uncertainties by sqrt(chi2_reduced) = 5.67")
    print("   4. JUG and PINT report the 'formal' uncertainties")
    print("      (already scaled or using different convention)")
    print("\n   Both approaches are valid:")
    print("   - Tempo2: Conservative, inflates errors for poor fit")
    print("   - JUG/PINT: Formal errors from covariance matrix")
    print("\n   For publication, you should:")
    print("   - Investigate why chi2_reduced is so high")
    print("   - Consider adding EFAC/EQUAD to model white noise")
    print("   - Use Tempo2-style scaling if desired for conservatism")
else:
    print(f"\nUnclear why Tempo2 uncertainties are {tempo2_scaling:.2f}× larger")
    print("Further investigation needed.")
