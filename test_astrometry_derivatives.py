#!/usr/bin/env python3
"""Test astrometry derivatives against PINT"""

import numpy as np
from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives
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

# Extract data from PINT for JUG testing
print("\nExtracting data for JUG...")
toas_mjd_tdb = t.table['tdbld'].value  # TDB times in MJD

# Get SSB positions from PINT (in meters, convert to light-seconds)
C_M_S = 299792458.0
ssb_obs_pos_m = t.table['ssb_obs_pos'].value  # Shape (n_toas, 3) in meters
ssb_obs_pos_lt_sec = ssb_obs_pos_m / C_M_S  # Convert to light-seconds

# Build JUG params dict from PINT model
params_dict = {}
for p in m.params:
    val = getattr(m, p).value
    if val is not None:
        params_dict[p] = val
        
# Convert angular parameters from astropy units to radians
from astropy import units as u
if 'RAJ' in params_dict:
    params_dict['RAJ'] = m.RAJ.quantity.to(u.rad).value
if 'DECJ' in params_dict:
    params_dict['DECJ'] = m.DECJ.quantity.to(u.rad).value
if 'PMRA' in params_dict:
    params_dict['PMRA'] = m.PMRA.quantity.to(u.rad/u.yr).value
if 'PMDEC' in params_dict:
    params_dict['PMDEC'] = m.PMDEC.quantity.to(u.rad/u.yr).value
if 'PX' in params_dict:
    params_dict['PX'] = m.PX.quantity.to(u.rad).value  # Convert mas to rad
if 'POSEPOCH' in params_dict:
    params_dict['POSEPOCH'] = m.POSEPOCH.value  # Already in MJD

# Test each astrometry parameter
astro_params = ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']
available = [p for p in astro_params if p in M_pint[1]]  # M_pint is (matrix, labels, units)

print(f"\nTesting {len(available)} astrometry parameters")
print("Note: PINT's design matrix uses milliseconds, JUG uses seconds internally")
print("=" * 80)

for param in available:
    # Get PINT derivative
    # PINT's design matrix is in milliseconds (see PINT source)
    idx = M_pint[1].index(param)  # M_pint is (matrix, labels, units)
    pint_deriv_ms = M_pint[0][:, idx]  # In milliseconds/param_unit
    
    # Get JUG derivative (in seconds/param_unit)
    jug_derivs = compute_astrometry_derivatives(
        params_dict, 
        toas_mjd_tdb, 
        ssb_obs_pos_lt_sec, 
        [param]
    )
    jug_deriv_sec = jug_derivs[param]  # In seconds/param_unit
    
    # Convert JUG to milliseconds for comparison
    jug_deriv_ms = jug_deriv_sec * 1000  # Convert to ms/param_unit
    
    # Compare 
    ratio = jug_deriv_ms / pint_deriv_ms
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    max_abs_diff = np.max(np.abs(jug_deriv_ms - pint_deriv_ms))
    
    print(f"\n{param}:")
    print(f"  Mean ratio (JUG/PINT): {mean_ratio:.10f}")
    print(f"  Std ratio: {std_ratio:.2e}")
    print(f"  Max abs diff: {max_abs_diff:.2e} milliseconds")
    print(f"  PINT range: [{pint_deriv_ms.min():.6e}, {pint_deriv_ms.max():.6e}] ms")
    print(f"  JUG range: [{jug_deriv_ms.min():.6e}, {jug_deriv_ms.max():.6e}] ms")
    
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
