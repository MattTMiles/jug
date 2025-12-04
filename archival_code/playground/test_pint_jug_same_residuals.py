#!/usr/bin/env python3
"""Test if PINT and JUG converge to same parameters when fitting identical residuals.

Strategy:
1. Compute JUG baseline residuals
2. Feed the SAME residuals to both fitters by adjusting PINT's TOA times
3. Compare fitted parameters
"""

# CRITICAL: Enable JAX float64 BEFORE imports
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import pint.models
import pint.toa
import pint.residuals
from pint.fitter import WLSFitter
import astropy.units as u
from astropy.time import TimeDelta

from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt
from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax

print("=" * 80)
print("Test: Do PINT and JUG converge to same fitted parameters?")
print("=" * 80)
print()

parfile = 'data/pulsars/J1909-3744_tdb.par'
timfile = 'data/pulsars/J1909-3744.tim'

# ============================================================================
# 1. Compute JUG baseline residuals
# ============================================================================
print("1. Computing JUG baseline residuals...")

fixed_data = prepare_fixed_data(parfile, timfile)
par_params = fixed_data['par_params']

print(f"   Reference F0: {par_params['F0']:.15f} Hz")
print(f"   Reference F1: {par_params['F1']:.15e} Hz/s")

# Compute residuals
fit_params = ['F0', 'F1']
params_array = jnp.array([par_params['F0'], par_params['F1']])

residuals_jug_sec = compute_residuals_jax_from_dt(
    params_array,
    tuple(fit_params),
    fixed_data['dt_sec'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    {k: v for k, v in par_params.items() if k not in fit_params}
)

residuals_jug_sec = np.array(residuals_jug_sec)
print(f"   JUG baseline RMS: {np.std(residuals_jug_sec)*1e6:.3f} μs")
print(f"   JUG baseline mean: {np.mean(residuals_jug_sec)*1e6:.3f} μs")
print()

# ============================================================================
# 2. Load PINT and replace its residuals with JUG's
# ============================================================================
print("2. Adjusting PINT TOAs to match JUG residuals...")

model_pint = pint.models.get_model(parfile)
toas_pint = pint.toa.get_TOAs(timfile, model=model_pint)

# Compute PINT's original residuals
res_pint_original = pint.residuals.Residuals(toas_pint, model_pint)
residuals_pint_original = res_pint_original.time_resids.to_value('s')

print(f"   PINT original RMS: {np.std(residuals_pint_original)*1e6:.3f} μs")
print(f"   JUG-PINT difference: {np.std(residuals_jug_sec - residuals_pint_original)*1e6:.3f} μs")

# Adjust TOAs: move by (residual_jug - residual_pint)
delta_residuals = residuals_jug_sec - residuals_pint_original
toas_pint.adjust_TOAs(TimeDelta(delta_residuals * u.s))

# Verify adjustment worked
res_check = pint.residuals.Residuals(toas_pint, model_pint)
residuals_check = res_check.time_resids.to_value('s')
match_rms = np.std(residuals_check - residuals_jug_sec) * 1e6

print(f"   After adjustment: {np.std(residuals_check)*1e6:.3f} μs RMS")
print(f"   Match to JUG: {match_rms:.6f} μs")

if match_rms > 1e-6:
    print("   ⚠️  WARNING: Residuals don't match perfectly after adjustment!")
print()

# ============================================================================
# 3. Fit with PINT
# ============================================================================
print("3. Fitting with PINT on JUG residuals...")

fitter_pint = WLSFitter(toas_pint, model_pint)
fitter_pint.fit_toas()

pint_f0 = model_pint.F0.value
pint_f1 = model_pint.F1.value
pint_chi2 = fitter_pint.resids.chi2
pint_reduced_chi2 = fitter_pint.resids.reduced_chi2

print(f"   PINT fitted F0: {pint_f0:.15f} Hz")
print(f"   PINT fitted F1: {pint_f1:.15e} Hz/s")
print(f"   PINT chi2: {pint_chi2:,.1f}")
print(f"   PINT reduced chi2: {pint_reduced_chi2:.2f}")
print()

# ============================================================================
# 4. Fit with JUG
# ============================================================================
print("4. Fitting with JUG on same residuals...")

# Create residual function
def residuals_fn(params):
    """Compute residuals given parameter dict (returns μs)."""
    fit_params_list = ['F0', 'F1']
    params_array = jnp.array([params['F0'], params['F1']])
    fixed_params_dict = {k: v for k, v in par_params.items() if k not in fit_params_list}
    
    residuals_sec = compute_residuals_jax_from_dt(
        params_array,
        tuple(fit_params_list),
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params_dict
    )
    
    return residuals_sec * 1e6  # Convert to μs

# Create design matrix function
def design_matrix_fn(params, toas_mjd, freq_mhz, errors_us, fit_params_list):
    """Compute design matrix using JAX autodiff."""
    def residuals_for_grad(param_values):
        params_dict = par_params.copy()
        for i, name in enumerate(fit_params_list):
            params_dict[name] = param_values[i]
        return residuals_fn(params_dict)
    
    param_values = jnp.array([params[name] for name in fit_params_list])
    jacobian_fn = jax.jacfwd(residuals_for_grad)
    jacobian_us = jacobian_fn(param_values)
    jacobian_sec = np.array(jacobian_us) * 1e-6
    
    errors_sec = errors_us * 1e-6
    M_weighted = jacobian_sec / errors_sec[:, np.newaxis]
    
    return M_weighted

# Fit with Gauss-Newton
fitted_params, uncertainties, info = gauss_newton_fit_jax(
    residuals_fn=residuals_fn,
    params=par_params.copy(),
    fit_params=['F0', 'F1'],
    design_matrix_fn=design_matrix_fn,
    toas_mjd=np.array(fixed_data['tdb_mjd']),
    freq_mhz=np.array(fixed_data['freq_mhz']),
    errors_us=np.array(fixed_data['uncertainties_us']),
    max_iter=20,
    lambda_init=1e-3,
    convergence_threshold=1e-12,
    verbose=True
)

jug_f0 = fitted_params['F0']
jug_f1 = fitted_params['F1']
jug_chi2 = info['final_chi2']
jug_reduced_chi2 = info['final_reduced_chi2']

print(f"   JUG fitted F0: {jug_f0:.15f} Hz")
print(f"   JUG fitted F1: {jug_f1:.15e} Hz/s")
print(f"   JUG chi2: {jug_chi2:,.1f}")
print(f"   JUG reduced chi2: {jug_reduced_chi2:.2f}")
print()

# ============================================================================
# 5. Compare results
# ============================================================================
print("=" * 80)
print("RESULTS: Comparing PINT and JUG fits to the same residuals")
print("=" * 80)
print()

print(f"F0 (Hz):")
print(f"  Reference:   {par_params['F0']:.15f}")
print(f"  PINT fitted: {pint_f0:.15f}")
print(f"  JUG fitted:  {jug_f0:.15f}")
print(f"  JUG - PINT:  {jug_f0 - pint_f0:.3e} Hz")
print()

print(f"F1 (Hz/s):")
print(f"  Reference:   {par_params['F1']:.15e}")
print(f"  PINT fitted: {pint_f1:.15e}")
print(f"  JUG fitted:  {jug_f1:.15e}")
print(f"  JUG - PINT:  {jug_f1 - pint_f1:.3e} Hz/s")
print()

print(f"Chi2:")
print(f"  PINT: {pint_chi2:,.1f}")
print(f"  JUG:  {jug_chi2:,.1f}")
print(f"  Diff: {jug_chi2 - pint_chi2:,.1f}")
print()

print(f"Reduced Chi2:")
print(f"  PINT: {pint_reduced_chi2:.2f}")
print(f"  JUG:  {jug_reduced_chi2:.2f}")
print()

# Interpret results
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()

tol_f0 = 1e-12  # 1 pHz tolerance
tol_f1 = 1e-20  # 1e-20 Hz/s tolerance
tol_chi2 = 100   # 100 chi2 units tolerance

f0_match = abs(jug_f0 - pint_f0) < tol_f0
f1_match = abs(jug_f1 - pint_f1) < tol_f1
chi2_match = abs(jug_chi2 - pint_chi2) < tol_chi2

if f0_match and f1_match and chi2_match:
    print("✅ SUCCESS: JUG and PINT converge to identical parameters!")
    print()
    print("This confirms that:")
    print("  • Both fitters minimize chi2 correctly")
    print("  • Both compute gradients/Jacobians correctly")
    print("  • Both apply weights correctly")
    print("  • The fitters are mathematically equivalent")
    print()
    print("The earlier ~5e-13 Hz difference was due to the ~3 ns systematic")
    print("difference in baseline residuals, NOT a problem with the fitter!")
else:
    print("⚠️  CAUTION: JUG and PINT find slightly different parameters")
    print()
    if not f0_match:
        print(f"  F0 difference: {abs(jug_f0 - pint_f0):.3e} Hz (tol: {tol_f0:.3e})")
    if not f1_match:
        print(f"  F1 difference: {abs(jug_f1 - pint_f1):.3e} Hz/s (tol: {tol_f1:.3e})")
    if not chi2_match:
        print(f"  Chi2 difference: {abs(jug_chi2 - pint_chi2):.1f} (tol: {tol_chi2:.1f})")
    print()
    print("Possible causes:")
    print("  • Different convergence criteria")
    print("  • Different stopping tolerances")
    print("  • Numerical precision differences")
    print("  • Different number of iterations")
