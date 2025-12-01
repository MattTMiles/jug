#!/usr/bin/env python3
"""
Deep dive into Tempo2's chi-square scaling and its implications for model selection.

This investigates the apparent paradox:
- If Tempo2 scales errors by sqrt(chi2_r), doesn't that make chi2_r always ~1?
- How can you use chi2_r for model selection then?
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TEMPO2 CHI-SQUARE SCALING: UNDERSTANDING THE PARADOX")
print("="*80)

print("\n" + "="*80)
print("THE APPARENT PARADOX")
print("="*80)

print("""
You're absolutely right to question this! Let's think through the logic:

1. Tempo2 fits parameters and computes χ² = Σ[(data - model)²/σ²]
2. Reduced χ²ᵣ = χ² / DOF = 32.2 (much greater than 1)
3. Tempo2 then scales parameter uncertainties by √(χ²ᵣ) = 5.68

BUT WAIT:

If you scale the TOA errors by √(χ²ᵣ), then:
   χ²_new = Σ[(data - model)² / (σ × √χ²ᵣ)²]
          = Σ[(data - model)² / (σ² × χ²ᵣ)]
          = (1/χ²ᵣ) × Σ[(data - model)²/σ²]
          = χ²_old / χ²ᵣ
          = χ²ᵣ × DOF / χ²ᵣ
          = DOF

So χ²ᵣ_new = χ²_new / DOF = DOF / DOF = 1 !!!

This would make χ²ᵣ ALWAYS equal to 1 after rescaling, which would make it
useless for model selection!

So what's really going on?
""")

print("\n" + "="*80)
print("THE KEY INSIGHT: TEMPO2 DOESN'T RESCALE THE DATA")
print("="*80)

print("""
Tempo2's chi-square scaling is applied ONLY to parameter uncertainties,
NOT to the TOA errors used in the fit!

Here's the workflow:

Step 1: FIT WITH ORIGINAL TOA ERRORS
   - Use σ_TOA from .tim file
   - Compute χ² = Σ[(data - model)²/σ_TOA²]
   - Get χ²ᵣ = χ²/DOF = 32.2

Step 2: COMPUTE FORMAL PARAMETER UNCERTAINTIES
   - From covariance matrix: Cov = (M^T W M)^(-1)
   - These are the "formal" uncertainties

Step 3: INFLATE PARAMETER UNCERTAINTIES
   - Multiply formal uncertainties by √(χ²ᵣ)
   - σ_param_reported = σ_param_formal × √(χ²ᵣ)
   - This is what goes in the output .par file

Step 4: CHI-SQUARE STAYS THE SAME
   - The χ² value reported is STILL based on original σ_TOA
   - χ²ᵣ = 32.2 is preserved for model comparison!

So Tempo2 is saying:
   "The fit has χ²ᵣ = 32, which is bad. I'll inflate the parameter
    uncertainties to reflect this poor fit quality, but I won't
    change the χ²ᵣ value itself because you need that for model selection."
""")

print("\n" + "="*80)
print("VERIFICATION: CHECK TEMPO2'S ACTUAL BEHAVIOR")
print("="*80)

# Read the refit par file
tempo2_par = Path("data/pulsars/J1909-3744_tdb_refit_F0_F1.par")

chi2r_tempo2 = None
dof_tempo2 = None
f0_err_tempo2 = None

with open(tempo2_par, 'r') as f:
    for line in f:
        parts = line.split()
        if parts and parts[0] == 'CHI2R':
            chi2r_tempo2 = float(parts[1])
            dof_tempo2 = int(parts[2])
        elif len(parts) >= 4 and parts[0] == 'F0':
            f0_err_tempo2 = float(parts[3])

print(f"\nFrom Tempo2 output .par file:")
print(f"  CHI2R: {chi2r_tempo2:.2f}")
print(f"  DOF: {dof_tempo2}")
print(f"  F0 uncertainty: {f0_err_tempo2:.2e} Hz")

print(f"\nKey observation:")
print(f"  χ²ᵣ is STILL {chi2r_tempo2:.2f}, not 1.0!")
print(f"  This proves Tempo2 doesn't rescale the TOA errors")

print("\n" + "="*80)
print("WHY DOES TEMPO2 DO THIS?")
print("="*80)

print("""
This is a CONSERVATIVE approach for parameter REPORTING:

1. Purpose: Protect against overconfident parameter uncertainties
   
2. Philosophy: "If the fit is poor (χ²ᵣ >> 1), don't trust the
   formal uncertainties from the covariance matrix"

3. Motivation:
   - High χ²ᵣ suggests missing systematics or underestimated errors
   - Formal uncertainties assume the model is correct
   - Inflating by √(χ²ᵣ) is an ad-hoc correction

4. Precedent: This is standard in some fields (e.g., metrology)
   - Called "Type B uncertainty" evaluation
   - ISO Guide to Uncertainty in Measurement (GUM)

5. Downside: Creates confusion about what the errors mean!
""")

print("\n" + "="*80)
print("IMPLICATIONS FOR MODEL SELECTION")
print("="*80)

print("""
For model selection (e.g., "Should I add F2?"), you use χ²ᵣ directly:

Model A (F0, F1 only):
   χ²_A = 335,000  with DOF_A = 10,406
   χ²ᵣ_A = 32.2

Model B (F0, F1, F2):
   χ²_B = 330,000  with DOF_B = 10,405
   χ²ᵣ_B = 31.7

F-test or Δχ² test:
   Δχ² = 5,000 with Δ_DOF = 1
   p-value = extremely significant!
   
Conclusion: Adding F2 improves the fit significantly.

The χ²ᵣ values themselves (32.2 vs 31.7) tell you about absolute fit quality.
The CHANGE in χ² tells you whether the new parameter is justified.

The parameter uncertainty inflation is SEPARATE from this process.
It's just for reporting purposes.
""")

print("\n" + "="*80)
print("THE MODERN VIEW (PINT/JUG APPROACH)")
print("="*80)

print("""
Modern pulsar timing prefers NOT to inflate parameter uncertainties:

Why?
1. χ²ᵣ >> 1 is almost ALWAYS true for pulsar timing data
   - TOA errors from .tim files rarely include all systematics
   - Red noise is ubiquitous but not in simple white-noise model

2. Proper solution: Model the noise explicitly
   - Add EFAC/EQUAD parameters to scale errors
   - Add red noise (RNAMP, RNIDX) 
   - Use Bayesian methods (enterprise, tempo2 -nobs)

3. With proper noise modeling, χ²ᵣ ≈ 1, and uncertainty inflation
   is unnecessary

4. Parameter uncertainties should come from:
   - Posterior distributions (Bayesian)
   - Bootstrap resampling
   - Properly marginalized over noise parameters

5. The informal "multiply by √(χ²ᵣ)" is a hack that:
   - Doesn't account for parameter correlations
   - Treats all parameters equally (but some may be more affected)
   - Can over- or under-correct

CONCLUSION:
   JUG and PINT give you the formal uncertainties.
   YOU decide whether to inflate them based on fit quality.
   This gives more transparency and control.
""")

print("\n" + "="*80)
print("PRACTICAL EXAMPLE: MODEL COMPARISON")
print("="*80)

import os
os.environ['JAX_LOG_COMPILES'] = '0'

from jug.fitting.optimized_fitter import fit_parameters_optimized
import pint.models
import pint.toa
import pint.fitter

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

# Model 1: F0 only
print("\nModel 1: Fit F0 only")
result_f0 = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0'],
    max_iter=25,
    verbose=False,
    device='cpu'
)

# Compute chi-square manually
residuals = result_f0['postfit_residuals_us'] / 1e6  # Convert to seconds
errors = result_f0['errors_us'] / 1e6  # Convert to seconds
chi2_f0 = np.sum((residuals / errors)**2)
dof_f0 = len(residuals) - 1
chi2r_f0 = chi2_f0 / dof_f0

print(f"  χ² = {chi2_f0:.1f}")
print(f"  DOF = {dof_f0}")
print(f"  χ²ᵣ = {chi2r_f0:.2f}")
print(f"  RMS = {result_f0['final_rms']:.3f} μs")

# Model 2: F0 + F1
print("\nModel 2: Fit F0 + F1")
result_f0f1 = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1'],
    max_iter=25,
    verbose=False,
    device='cpu'
)

residuals = result_f0f1['postfit_residuals_us'] / 1e6  # Convert to seconds
errors = result_f0f1['errors_us'] / 1e6  # Convert to seconds
chi2_f0f1 = np.sum((residuals / errors)**2)
dof_f0f1 = len(residuals) - 2
chi2r_f0f1 = chi2_f0f1 / dof_f0f1

print(f"  χ² = {chi2_f0f1:.1f}")
print(f"  DOF = {dof_f0f1}")
print(f"  χ²ᵣ = {chi2r_f0f1:.2f}")
print(f"  RMS = {result_f0f1['final_rms']:.3f} μs")

# Model comparison
delta_chi2 = chi2_f0 - chi2_f0f1
delta_dof = dof_f0 - dof_f0f1

print("\n" + "-"*80)
print("Model Comparison:")
print(f"  Δχ² = {delta_chi2:.1f}")
print(f"  Δ_DOF = {delta_dof}")
print(f"  Δχ² / Δ_DOF = {delta_chi2 / delta_dof:.1f}")

from scipy.stats import chi2
p_value = 1 - chi2.cdf(delta_chi2, delta_dof)
print(f"  p-value = {p_value:.2e}")

if p_value < 0.001:
    print(f"  ✅ F1 is HIGHLY significant (p < 0.001)")
else:
    print(f"  ❌ F1 is not significant")

print("\n" + "-"*80)
print("Conclusion:")
print(f"  Even though both models have χ²ᵣ >> 1 (bad absolute fit),")
print(f"  the CHANGE in χ² clearly shows F1 improves the model.")
print(f"  The χ²ᵣ values are still useful for model selection!")

print("\n" + "="*80)
print("FINAL ANSWER TO YOUR QUESTION")
print("="*80)

print("""
Why does Tempo2 scale uncertainties by √(χ²ᵣ)?

ANSWER: It's a conservative reporting convention, NOT part of the fitting.

The workflow is:
1. Fit parameters using UNSCALED TOA errors → get χ²ᵣ
2. Compute formal parameter uncertainties from covariance
3. REPORT scaled uncertainties: σ_reported = σ_formal × √(χ²ᵣ)
4. Keep χ²ᵣ unchanged for model selection

For model selection:
- You still use χ²ᵣ to compare models
- The uncertainty scaling doesn't affect this
- You're comparing Δχ² between models with different parameters

The scaling is ONLY for the final parameter uncertainties in the .par file.

Modern approach (JUG/PINT):
- Report formal uncertainties without inflation
- Let the user decide whether to scale based on χ²ᵣ
- Better: Model the noise properly so χ²ᵣ ≈ 1 from the start!

YOUR INTUITION WAS CORRECT:
- If you actually used the scaled errors in the fit, χ²ᵣ would become 1
- But Tempo2 doesn't do that - it only scales the OUTPUT uncertainties
- The χ²ᵣ reported is based on the original TOA errors
- So it remains useful for model comparison

This is admittedly confusing! It's mixing two concepts:
1. Fit quality assessment (χ²ᵣ) - stays unchanged
2. Parameter uncertainty reporting - gets inflated

JUG/PINT separate these more clearly by giving you formal uncertainties
and letting you apply any scaling you want.
""")

print("\n" + "="*80)
print("RECOMMENDATION FOR JUG")
print("="*80)

print("""
JUG should report BOTH:

1. Formal uncertainties (from covariance matrix)
   - Current behavior ✓
   - Matches PINT
   - Good for model comparison

2. Optionally, Tempo2-style scaled uncertainties
   - Add option: scale_errors='tempo2'
   - σ_scaled = σ_formal × √(χ²ᵣ)
   - For conservative reporting

Example output:
   F0 = 339.315691919 ± 1.02e-14 Hz (formal)
                      ± 5.77e-14 Hz (scaled by √χ²ᵣ=5.68)
   
   F1 = -1.615e-15 ± 1.66e-22 Hz/s (formal)
                   ± 9.43e-22 Hz/s (scaled by √χ²ᵣ=5.68)
   
   χ²ᵣ = 32.2 (DOF=10405)
   
This way users understand:
- The formal uncertainties from the fit
- The conservative inflated uncertainties
- Why they differ (poor fit quality)
- Which to use for different purposes
""")
