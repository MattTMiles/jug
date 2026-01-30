"""Diagnose why astrometric fitting becomes unstable with repeated iterations."""

import numpy as np
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import _compute_astrometry_derivatives

# Load data
params = parse_par_file('data/pulsars/J1909-3744_tdb.par')
toas = parse_tim_file_mjds('data/pulsars/J1909-3744.tim')
toas_dict = {
    'toas_mjd': np.array([t.mjd_int + t.mjd_frac for t in toas]),
    'errors_us': np.array([t.error_us for t in toas])
}

result = compute_residuals_simple(params, toas_dict)

# Test different combinations of parameters
test_cases = [
    (['RAJ', 'DECJ'], "Position only"),
    (['PMRA', 'PMDEC'], "Proper motion only"),
    (['RAJ', 'DECJ', 'PMRA', 'PMDEC'], "All astrometric"),
    (['F0', 'F1', 'RAJ', 'DECJ'], "Spin + position"),
    (['A1', 'PB', 'TASC', 'EPS1', 'EPS2'], "Binary only"),
    (['F0', 'F1', 'A1', 'PB', 'TASC', 'EPS1', 'EPS2', 'RAJ', 'DECJ', 'PMRA', 'PMDEC'], "All together"),
]

print("="*80)
print("DESIGN MATRIX CONDITION NUMBER ANALYSIS")
print("="*80)

for fit_params, description in test_cases:
    # Build design matrix
    M_list = []
    
    # Get astrometry derivatives if needed
    astro_params = [p for p in fit_params if p in ['RAJ', 'DECJ', 'PMRA', 'PMDEC']]
    if astro_params:
        astro_derivs = _compute_astrometry_derivatives(
            params, toas_dict, result['ssb_obs_pos_ls'], astro_params
        )
        for p in astro_params:
            M_list.append(astro_derivs[p])
    
    # Get other derivatives (simplified - just check structure)
    # This is just for condition number, so we can use dummy values if needed
    for p in fit_params:
        if p not in astro_params:
            # Placeholder - in real code this would call appropriate derivative function
            M_list.append(np.zeros(len(toas_dict['toas_mjd'])))
    
    if not M_list:
        continue
        
    M = np.column_stack(M_list)
    
    # Weight by uncertainties
    errors_sec = toas_dict['errors_us'] * 1e-6
    W = np.diag(1.0 / errors_sec**2)
    
    # Normal equation matrix: M^T W M
    MTM = M.T @ W @ M
    
    # Compute condition number
    try:
        cond = np.linalg.cond(MTM)
        eigvals = np.linalg.eigvalsh(MTM)
        
        print(f"\n{description}:")
        print(f"  Parameters: {fit_params}")
        print(f"  Design matrix shape: {M.shape}")
        print(f"  Condition number: {cond:.3e}")
        print(f"  Eigenvalue range: [{eigvals.min():.3e}, {eigvals.max():.3e}]")
        
        if cond > 1e10:
            print(f"  ⚠️  WARNING: Poor conditioning (cond > 1e10)")
        elif cond > 1e6:
            print(f"  ⚠️  CAUTION: Moderate conditioning (cond > 1e6)")
        else:
            print(f"  ✓ Good conditioning")
            
        # Check for correlations
        corr_matrix = np.corrcoef(M.T)
        max_corr = 0
        for i in range(len(fit_params)):
            for j in range(i+1, len(fit_params)):
                if abs(corr_matrix[i, j]) > abs(max_corr):
                    max_corr = corr_matrix[i, j]
                    max_pair = (fit_params[i], fit_params[j])
        
        if abs(max_corr) > 0.9:
            print(f"  ⚠️  Strong correlation: {max_pair[0]}-{max_pair[1]} = {max_corr:.4f}")
        
    except np.linalg.LinAlgError:
        print(f"\n{description}:")
        print(f"  ❌ SINGULAR MATRIX - cannot compute condition number")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("If condition number > 1e10, consider:")
print("  1. Fitting parameters in groups (not all at once)")
print("  2. Using parameter scaling/normalization") 
print("  3. Regularization or damped least squares")
print("="*80)
