"""Test JUG's Gauss-Newton fitter using PINT's residual calculation.

This isolates whether differences in fit results come from the fitter
itself or from differences in residual calculations.
"""

import numpy as np
import pint.fitter
import pint.models
import pint.toa
import pint.residuals
from astropy import units as u
from jug.fitting.gauss_newton import fit_gauss_newton


def pint_residuals_wrapper(params, data):
    """Compute residuals using PINT.
    
    This wrapper updates the PINT model with new parameters,
    then returns residuals in microseconds.
    """
    model = data['pint_model']
    toas = data['pint_toas']
    
    # Update PINT model with new parameters
    for name, value in params.items():
        if hasattr(model, name):
            param_obj = getattr(model, name)
            param_obj.value = value
    
    # Compute residuals
    residuals = pint.residuals.Residuals(toas, model)
    return np.array(residuals.time_resids.to_value('us'), dtype=np.float64)


def pint_design_matrix_wrapper(params, data, fit_params):
    """Compute design matrix using PINT's numerical derivatives.
    
    This uses finite differences to compute derivatives of residuals
    with respect to parameters.
    """
    model = data['pint_model']
    errors_us = data['errors_us']
    
    # Compute residuals at current point
    res_0 = pint_residuals_wrapper(params, data)
    
    # Compute numerical derivatives
    n_toas = len(res_0)
    n_params = len(fit_params)
    M = np.zeros((n_toas, n_params), dtype=np.float64)
    
    for i, param_name in enumerate(fit_params):
        # Determine step size (relative to parameter value)
        param_val = params[param_name]
        if param_val != 0:
            step = abs(param_val) * 1e-8
        else:
            step = 1e-8
        
        # Perturb parameter
        params_perturbed = params.copy()
        params_perturbed[param_name] = param_val + step
        
        # Compute residuals with perturbed parameter
        res_plus = pint_residuals_wrapper(params_perturbed, data)
        
        # Numerical derivative: dr/dp
        M[:, i] = (res_plus - res_0) / step
        
        # Weight by uncertainty (same as M.T @ M weighting)
        M[:, i] /= errors_us
    
    return M


def test_jug_fitter_with_pint():
    """Test JUG's fitter using PINT's residual calculation."""
    
    # Load data
    par_file = 'data/pulsars/J1909-3744_tdb.par'
    tim_file = 'data/pulsars/J1909-3744.tim'
    
    print("Loading PINT model and TOAs...")
    model = pint.models.get_model(par_file)
    toas = pint.toa.get_TOAs(tim_file, ephem='DE440', include_bipm=True, planets=True)
    
    # Get initial parameters
    params_init = {
        'F0': model.F0.value,
        'F1': model.F1.value,
        'RAJ': model.RAJ.value,
        'DECJ': model.DECJ.value
    }
    
    # Parameters to fit
    fit_params = ['F0', 'F1']
    
    # Perturb initial values (smaller perturbations)
    params_perturbed = params_init.copy()
    params_perturbed['F0'] *= 1.000001  # 0.0001% perturbation
    params_perturbed['F1'] *= 1.00001   # 0.001% perturbation
    
    # Prepare data dictionary for JUG fitter
    errors_us = np.array(toas.get_errors().to_value('us'))
    data = {
        'pint_model': model,
        'pint_toas': toas,
        'errors_us': errors_us
    }
    
    print("\n" + "="*70)
    print("Testing JUG's Gauss-Newton fitter with PINT's residuals")
    print("="*70)
    print("\nInitial (true) parameters:")
    for name in fit_params:
        print(f"  {name:10s} = {params_init[name]:.12e}")
    print("\nPerturbed starting parameters:")
    for name in fit_params:
        print(f"  {name:10s} = {params_perturbed[name]:.12e}")
    
    # Fit using JUG's Gauss-Newton with PINT's residuals
    result = fit_gauss_newton(
        compute_residuals_func=pint_residuals_wrapper,
        compute_design_matrix_func=pint_design_matrix_wrapper,
        params_init=params_perturbed,
        fit_params=fit_params,
        data=data,
        max_iter=20,
        verbose=True
    )
    
    # Compare with PINT's own fitter
    print("\n" + "="*70)
    print("Running PINT's own fitter for comparison")
    print("="*70)
    
    # Reset model to perturbed values
    for name, value in params_perturbed.items():
        if hasattr(model, name):
            getattr(model, name).value = value
    
    model.free_params = fit_params
    pint_fitter = pint.fitter.WLSFitter(toas, model)
    pint_fitter.fit_toas()
    
    # Compare results
    print("\n" + "="*70)
    print("Comparison: JUG Fitter vs PINT Fitter")
    print("="*70)
    print("\n{:15s} {:20s} {:20s} {:20s}".format(
        "Parameter", "True Value", "JUG Fitted", "PINT Fitted"
    ))
    print("-"*80)
    
    for name in fit_params:
        true_val = params_init[name]
        jug_val = result['params'][name]
        pint_val = getattr(pint_fitter.model, name).value
        
        jug_diff = jug_val - true_val
        pint_diff = pint_val - true_val
        
        print(f"{name:15s} {true_val:20.12e} {jug_val:20.12e} {pint_val:20.12e}")
        print(f"{'Difference:':<15s} {'':<20s} {jug_diff:20.2e} {pint_diff:20.2e}")
    
    print("\n" + "="*70)
    print("Fit Statistics Comparison")
    print("="*70)
    print(f"JUG:  Chi2 = {result['chi2']:.2f}, Reduced chi2 = {result['reduced_chi2']:.3f}")
    print(f"      RMS = {result['rms_us']:.3f} μs, WRMS = {result['weighted_rms_us']:.3f} μs")
    print(f"PINT: Chi2 = {pint_fitter.resids.chi2:.2f}, Reduced chi2 = {pint_fitter.resids.chi2_reduced:.3f}")
    print(f"      RMS = {pint_fitter.resids.time_resids.std().to_value('us'):.3f} μs")
    
    # Check if they agree
    print("\n" + "="*70)
    print("Conclusion")
    print("="*70)
    
    max_param_diff = 0.0
    for name in fit_params:
        jug_val = result['params'][name]
        pint_val = getattr(pint_fitter.model, name).value
        diff = abs(jug_val - pint_val)
        rel_diff = diff / abs(pint_val) if pint_val != 0 else diff
        max_param_diff = max(max_param_diff, rel_diff)
    
    print(f"Maximum relative parameter difference: {max_param_diff:.2e}")
    
    if max_param_diff < 1e-3:
        print("✅ JUG's fitter agrees with PINT to good precision (< 0.1%)!")
        print("   Any differences in real JUG fits are primarily due to residual calculation differences.")
    elif max_param_diff < 1e-2:
        print("✔️  JUG's fitter agrees with PINT reasonably well (< 1%).")
        print("   Differences may be due to numerical precision or slight algorithmic differences.")
    else:
        print("⚠️  JUG's fitter produces significantly different results from PINT")
        print("   This suggests a problem with the fitter itself.")


if __name__ == '__main__':
    test_jug_fitter_with_pint()
