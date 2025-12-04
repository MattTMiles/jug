#!/usr/bin/env python3
"""Compare JUG's LM fitter with PINT's WLS fitter.

This test fits the same data with both fitters and compares:
- Fitted parameter values
- Parameter uncertainties
- Final chi-squared and RMS
- Convergence behavior
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import pint.models
import pint.toa
import pint.fitter
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax
from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt

print("="*80)
print("Comparing JUG LM Fitter vs PINT WLS Fitter")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# ============================================================================
# Part 1: Get PINT reference fit
# ============================================================================
print("\n" + "="*80)
print("Part 1: Fitting with PINT")
print("="*80)

# Load PINT model and TOAs
print("\n1. Loading PINT model and TOAs...")
pint_model = pint.models.get_model(par_file)
pint_toas = pint.toa.get_TOAs(tim_file, ephem='DE440', include_bipm=True, planets=True)

print(f"   Loaded {len(pint_toas)} TOAs")

# Get initial parameter values
f0_init = pint_model.F0.value
f1_init = pint_model.F1.value
dm_init = pint_model.DM.value

print(f"\n2. Initial parameters from .par file:")
print(f"   F0: {f0_init:.15f} Hz")
print(f"   F1: {f1_init:.15e} Hz/s")
print(f"   DM: {dm_init:.15f} pc/cm^3")

# Perturb parameters to give fitter something to do
perturb_f0 = 1e-10  # Small perturbation
perturb_f1 = f1_init * 0.001
perturb_dm = dm_init * 0.0001

pint_model.F0.value = f0_init + perturb_f0
pint_model.F1.value = f1_init + perturb_f1
pint_model.DM.value = dm_init + perturb_dm

print(f"\n3. Perturbed starting parameters:")
print(f"   F0: {pint_model.F0.value:.15f} Hz (Δ = {perturb_f0:.2e})")
print(f"   F1: {pint_model.F1.value:.15e} Hz/s (Δ = {perturb_f1:.2e})")
print(f"   DM: {pint_model.DM.value:.15f} pc/cm^3 (Δ = {perturb_dm:.2e})")

# Set parameters to fit
pint_model.free_params = ['F0', 'F1', 'DM']

# Fit with PINT
print(f"\n4. Fitting with PINT WLS fitter...")
pint_fitter = pint.fitter.WLSFitter(pint_toas, pint_model)
pint_fitter.fit_toas(maxiter=20)

pint_f0_fitted = pint_fitter.model.F0.value
pint_f1_fitted = pint_fitter.model.F1.value
pint_dm_fitted = pint_fitter.model.DM.value

pint_f0_unc = pint_fitter.model.F0.uncertainty.value
pint_f1_unc = pint_fitter.model.F1.uncertainty.value
pint_dm_unc = pint_fitter.model.DM.uncertainty.value

pint_chi2 = pint_fitter.resids.chi2
pint_wrms = pint_fitter.resids.time_resids.std().to_value('us')

print(f"\n5. PINT fit results:")
print(f"   F0: {pint_f0_fitted:.15f} ± {pint_f0_unc:.3e} Hz")
print(f"   F1: {pint_f1_fitted:.15e} ± {pint_f1_unc:.3e} Hz/s")
print(f"   DM: {pint_dm_fitted:.15f} ± {pint_dm_unc:.3e} pc/cm^3")
print(f"   Chi²: {pint_chi2:.2f}")
print(f"   WRMS: {pint_wrms:.3f} μs")

# ============================================================================
# Part 2: Fit with JUG LM fitter
# ============================================================================
print("\n" + "="*80)
print("Part 2: Fitting with JUG LM Fitter")
print("="*80)

print("\n1. Preparing fixed data...")
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

# Create residual function
def residual_func(toas, freqs, params_dict):
    """Residual function for JUG fitter."""
    fit_params = tuple(sorted(params_dict.keys()))
    params_array = jnp.array([params_dict[name] for name in fit_params])
    fixed_params = {k: v for k, v in par_params.items() if k not in params_dict}

    residuals_sec = compute_residuals_jax(
        params_array, fit_params,
        fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
        fixed_data['geometric_delay_sec'],
        fixed_data['other_delays_minus_dm_sec'],
        fixed_data['pepoch'], fixed_data['dm_epoch'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )

    return residuals_sec

# Use same perturbed starting values as PINT
initial_params = {
    'F0': f0_init + perturb_f0,
    'F1': f1_init + perturb_f1,
    'DM': dm_init + perturb_dm
}

print(f"\n2. Starting parameters (same as PINT):")
print(f"   F0: {initial_params['F0']:.15f} Hz")
print(f"   F1: {initial_params['F1']:.15e} Hz/s")
print(f"   DM: {initial_params['DM']:.15f} pc/cm^3")

# Fit with JUG LM
print(f"\n3. Fitting with JUG LM fitter...")
print("-"*80)

fitted_params = fit_levenberg_marquardt(
    residual_func,
    initial_params,
    fixed_data['tdb_mjd'],
    fixed_data['freq_mhz'],
    fixed_data['uncertainties_us'],
    max_iterations=20,
    tolerance=1e-10,
    initial_damping=1e-3,
    verbose=True
)

print("-"*80)

# Compute final residuals and chi2
res_final = residual_func(None, None, fitted_params)
res_final_us = np.array(res_final) * 1e6
errors_us = np.array(fixed_data['uncertainties_us'])

weights = 1.0 / (errors_us ** 2)
jug_chi2 = np.sum(weights * res_final_us**2)
jug_wrms = np.sqrt(np.sum(weights * res_final_us**2) / np.sum(weights))

print(f"\n4. JUG fit results:")
print(f"   F0: {fitted_params['F0']:.15f} Hz")
print(f"   F1: {fitted_params['F1']:.15e} Hz/s")
print(f"   DM: {fitted_params['DM']:.15f} pc/cm^3")
print(f"   Chi²: {jug_chi2:.2f}")
print(f"   WRMS: {jug_wrms:.3f} μs")

# ============================================================================
# Part 3: Comparison
# ============================================================================
print("\n" + "="*80)
print("Part 3: Comparison")
print("="*80)

print(f"\n{'Parameter':<10} {'PINT Fitted':<25} {'JUG Fitted':<25} {'Difference':<20} {'Diff (σ)'}")
print("-"*100)

# F0
f0_diff = fitted_params['F0'] - pint_f0_fitted
f0_sigma = abs(f0_diff / pint_f0_unc) if pint_f0_unc > 0 else np.inf
print(f"{'F0':<10} {pint_f0_fitted:.15f} {fitted_params['F0']:.15f} {f0_diff:+.3e} {f0_sigma:.2f}σ")

# F1
f1_diff = fitted_params['F1'] - pint_f1_fitted
f1_sigma = abs(f1_diff / pint_f1_unc) if pint_f1_unc > 0 else np.inf
print(f"{'F1':<10} {pint_f1_fitted:.15e} {fitted_params['F1']:.15e} {f1_diff:+.3e} {f1_sigma:.2f}σ")

# DM
dm_diff = fitted_params['DM'] - pint_dm_fitted
dm_sigma = abs(dm_diff / pint_dm_unc) if pint_dm_unc > 0 else np.inf
print(f"{'DM':<10} {pint_dm_fitted:.15f} {fitted_params['DM']:.15f} {dm_diff:+.3e} {dm_sigma:.2f}σ")

print(f"\n{'Statistic':<15} {'PINT':<20} {'JUG':<20} {'Difference'}")
print("-"*75)
print(f"{'Chi²':<15} {pint_chi2:<20.2f} {jug_chi2:<20.2f} {jug_chi2 - pint_chi2:+.2f}")
print(f"{'WRMS (μs)':<15} {pint_wrms:<20.3f} {jug_wrms:<20.3f} {jug_wrms - pint_wrms:+.3f}")

# ============================================================================
# Success Criteria
# ============================================================================
print("\n" + "="*80)
print("Success Criteria")
print("="*80)

# Parameters should agree within 3σ
f0_ok = f0_sigma < 3.0
f1_ok = f1_sigma < 3.0
dm_ok = dm_sigma < 3.0

# Chi2 should be similar (within 1%)
chi2_diff_pct = abs(jug_chi2 - pint_chi2) / pint_chi2 * 100
chi2_ok = chi2_diff_pct < 1.0

# WRMS should be similar (within 10%)
wrms_diff_pct = abs(jug_wrms - pint_wrms) / pint_wrms * 100
wrms_ok = wrms_diff_pct < 10.0

print(f"\nParameter agreement:")
print(f"  F0 within 3σ: {f0_ok} ({f0_sigma:.2f}σ)")
print(f"  F1 within 3σ: {f1_ok} ({f1_sigma:.2f}σ)")
print(f"  DM within 3σ: {dm_ok} ({dm_sigma:.2f}σ)")

print(f"\nStatistics agreement:")
print(f"  Chi² within 1%: {chi2_ok} ({chi2_diff_pct:.3f}%)")
print(f"  WRMS within 10%: {wrms_ok} ({wrms_diff_pct:.3f}%)")

all_ok = f0_ok and f1_ok and dm_ok and chi2_ok and wrms_ok

if all_ok:
    print(f"\n{'='*80}")
    print("✅ ✅ ✅  SUCCESS! JUG matches PINT to excellent precision! ✅ ✅ ✅")
    print(f"{'='*80}")
else:
    print(f"\n⚠️  Some criteria not met, but results may still be acceptable")

print(f"\n{'='*80}")
print("Comparison Complete")
print(f"{'='*80}")
