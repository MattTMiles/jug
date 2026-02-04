#!/usr/bin/env python3
"""Test astrometry derivatives against PINT"""

import numpy as np
from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, parse_ra, parse_dec
import pint.toa as toa
import pint.models as models

# Use the simpler approach - get data from PINT itself for testing
parfile = 'data/pulsars/J1909-3744_tdb.par'
timfile = 'data/pulsars/J1909-3744.tim'

# PINT setup
print("Loading PINT data...")
m = models.get_model(parfile)
t = toa.get_TOAs(timfile, model=m)
M_pint = m.designmatrix(t, incfrozen=False, incoffset=True)

# JUG setup - use JUG's own data loading for consistency
print("\nLoading JUG data...")
result = compute_residuals_simple(parfile, timfile, verbose=False)
toas_mjd_tdb = result['tdb_mjd']
ssb_obs_pos_lt_sec = result['ssb_obs_pos_ls']

# Build JUG params dict
params = parse_par_file(parfile)
params_dict = dict(params)
params_dict['RAJ'] = parse_ra(params['RAJ'])
params_dict['DECJ'] = parse_dec(params['DECJ'])
# PMRA, PMDEC, PX should stay in par file units (mas/yr, mas)

# Test each astrometry parameter
astro_params = ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']
available = [p for p in astro_params if p in M_pint[1]]  # M_pint is (matrix, labels, units)

print(f"\nTesting {len(available)} astrometry parameters")
print("Note: Both PINT and JUG design matrices use seconds")
print("=" * 80)

for param in available:
    # Get PINT derivative (in seconds/param_unit)
    idx = M_pint[1].index(param)  # M_pint is (matrix, labels, units)
    pint_deriv = M_pint[0][:, idx]  # In seconds/param_unit
    
    # Get JUG derivative (also in seconds/param_unit)
    jug_derivs = compute_astrometry_derivatives(
        params_dict, 
        toas_mjd_tdb, 
        ssb_obs_pos_lt_sec, 
        [param]
    )
    jug_deriv = np.array(jug_derivs[param])  # In seconds/param_unit
    
    # Compare directly (both in seconds)
    ratio = jug_deriv / pint_deriv
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    max_abs_diff = np.max(np.abs(jug_deriv - pint_deriv))
    
    print(f"\n{param}:")
    print(f"  Mean ratio (JUG/PINT): {mean_ratio:.10f}")
    print(f"  Std ratio: {std_ratio:.2e}")
    print(f"  Max abs diff: {max_abs_diff:.2e} seconds")
    print(f"  PINT range: [{pint_deriv.min():.6e}, {pint_deriv.max():.6e}] s")
    print(f"  JUG range: [{jug_deriv.min():.6e}, {jug_deriv.max():.6e}] s")
    
    if abs(mean_ratio - 1.0) < 1e-8:
        print(f"  ✓ Perfect match!")
    elif abs(mean_ratio - 1.0) < 1e-6:
        print(f"  ✓ Excellent match (sub-μs level)")
    elif abs(mean_ratio - 1.0) < 1e-3:
        print(f"  ✓ Good match")
    else:
        print(f"  ✗ Mismatch!")

print("\n" + "=" * 80)
print("Done!")
