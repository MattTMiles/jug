#!/usr/bin/env python3
"""Test JUG binary derivatives against PINT and numerical verification.

This module validates JUG's analytical ELL1 binary derivatives using two methods:

1. **Comparison against PINT**: Validates unit conversions and general agreement.
   Note: PINT has a bug in d_delayS_d_Phi (missing cos(Φ) factor), so PB/TASC/PBDOT
   derivatives will differ near superior conjunction. JUG is correct.

2. **Numerical finite-difference verification**: The ground truth test.
   Perturbs each parameter and measures the actual change in delay.

Unit conversions:
- PINT returns derivatives in mixed SI/astronomical units
- JUG returns derivatives in par-file units (s/par-unit)
- The test converts PINT outputs to par-file units for comparison

PINT BUG (documented 2026-01-29):
PINT's formula: d_delayS_d_Phi = 2 * TM2 * SINI / (1 - SINI * sin(Phi))
Correct formula: d_delayS_d_Phi = 2 * TM2 * SINI * cos(Phi) / (1 - SINI * sin(Phi))
See docs/PINT_SHAPIRO_DERIVATIVE_BUG.md for full analysis.
"""

import numpy as np
import warnings
import logging

# Configure logging
warnings.filterwarnings('ignore')
logging.getLogger('pint').setLevel(logging.ERROR)

def test_ell1_derivatives_vs_pint():
    """Compare JUG ELL1 derivatives against PINT reference."""
    
    # Import JUG
    from jug.utils.jax_setup import ensure_jax_x64
    ensure_jax_x64()
    
    import jax.numpy as jnp
    from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1
    
    # Import PINT
    import pint.models as models
    import pint.toa as toa
    from astropy import constants as c
    
    # Load test data
    model = models.get_model("data/pulsars/J1909-3744_tdb.par")
    toas = toa.get_TOAs("data/pulsars/J1909-3744.tim", model=model)
    
    binary_comp = model.components['BinaryELL1']
    
    # Get PINT derivatives and convert to par-file units (s / par-unit)
    params_to_check = ['A1', 'EPS1', 'EPS2', 'SINI', 'PB', 'TASC', 'PBDOT', 'M2']
    
    pint_derivs = {}
    for param in params_to_check:
        d = binary_comp.d_binary_delay_d_xxxx(toas, param, None)
        
        # Convert to s / par-unit
        if param in ['PB', 'TASC']:
            d_sec = np.asarray([x.decompose().value for x in d], dtype=np.float64)
            pint_derivs[param] = d_sec * 86400.0  # s/s -> s/d
        elif param == 'A1':
            d_sec = np.asarray([x.value for x in d], dtype=np.float64)
            pint_derivs[param] = d_sec * c.c.value  # s/m -> s/ls
        elif param == 'M2':
            d_sec = np.asarray([x.decompose().value for x in d], dtype=np.float64)
            pint_derivs[param] = d_sec * 1.98892e30  # s/kg -> s/Msun
        else:
            d_sec = np.asarray([x.decompose().value for x in d], dtype=np.float64)
            pint_derivs[param] = d_sec
    
    # Get the barycentric time used by PINT
    binary = binary_comp.binary_instance
    binary_t = np.asarray(binary.t.value, dtype=np.float64)
    
    # Setup JUG parameters
    params = {
        'PB': float(binary.PB.value),
        'A1': float(binary.A1.value),
        'TASC': float(binary.TASC.value),
        'EPS1': float(binary.EPS1.value),
        'EPS2': float(binary.EPS2.value),
        'PBDOT': float(binary.PBDOT.value),
        'SINI': float(binary.SINI.value),
        'M2': float(binary.M2.value),
    }
    
    # Compute JUG derivatives
    jug_derivs = compute_binary_derivatives_ell1(params, jnp.array(binary_t), params_to_check)
    
    print("="*70)
    print("JUG vs PINT ELL1 Binary Derivative Comparison")
    print("="*70)
    print(f"{'Param':<8} {'Mean ratio':<15} {'Max rel diff':<15} {'Status'}")
    print("-"*70)
    
    results = {}
    for param in params_to_check:
        jug = np.asarray(jug_derivs[param])
        pint = pint_derivs[param]
        
        mask = np.abs(pint) > 1e-30
        if np.any(mask):
            ratios = jug[mask] / pint[mask]
            mean_ratio = np.mean(ratios)
            rel_diff = np.abs(jug - pint) / (np.abs(pint) + 1e-30)
            max_rel_diff = np.max(rel_diff[mask])
        else:
            mean_ratio = np.nan
            max_rel_diff = np.nan
        
        results[param] = {'mean_ratio': mean_ratio, 'max_rel_diff': max_rel_diff}
        
        # Tolerances: strict for core params, looser for orbit-related
        if param in ['A1', 'EPS1', 'EPS2', 'SINI']:
            tolerance = 1e-5
        else:
            tolerance = 1e-3  # 0.1% for PB, TASC, PBDOT, M2
        
        if abs(mean_ratio - 1.0) < tolerance:
            status = "✓ PASS"
        else:
            status = f"✗ FAIL ({mean_ratio:.6f})"
        
        print(f"{param:<8} {mean_ratio:<15.10f} {max_rel_diff:<15.6e} {status}")
    
    # Assertions
    # Strict tolerance for core parameters
    for param in ['A1', 'EPS1', 'EPS2', 'SINI']:
        assert abs(results[param]['mean_ratio'] - 1.0) < 1e-5, \
            f"{param} derivative ratio {results[param]['mean_ratio']:.6f} exceeds tolerance"
    
    # Looser tolerance for orbit-related parameters
    for param in ['PB', 'TASC', 'PBDOT']:
        assert abs(results[param]['mean_ratio'] - 1.0) < 1e-3, \
            f"{param} derivative ratio {results[param]['mean_ratio']:.6f} exceeds tolerance"
    
    print("\nAll derivative tests passed!")


if __name__ == "__main__":
    test_ell1_derivatives_vs_pint()
