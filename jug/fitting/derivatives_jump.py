"""Analytical derivatives for JUMP parameters using JAX.

JUMP parameters represent timing offsets between different backends/receivers.
The derivative d(residual)/d(JUMP_i) = -1 for matching TOAs (JUMP is subtracted from dt_sec).
The design matrix column M = -d(r)/d(JUMP) = +1 for affected TOAs, 0 otherwise.

JUMP formats in par files:
- JUMP -fe L-wide 0.0001234  -> applies to TOAs with flag -fe L-wide
- JUMP -sys meerkat 0.0001234 -> applies to TOAs with flag -sys meerkat
- JUMP MJD 58000 59000 0.0001234 -> applies to TOAs in MJD range

Reference: PINT src/pint/models/jump.py

Uses JAX for consistency with JUG's JAX-first architecture.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Any


def compute_jump_derivatives(
    params: Dict,
    toas_mjd: jnp.ndarray,
    fit_params: List[str],
    toa_flags: Optional[List[Dict[str, str]]] = None,
    jump_masks: Optional[Dict[str, jnp.ndarray]] = None,
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """Compute design matrix columns for JUMP parameters using JAX.
    
    The design matrix column M = +1 for TOAs that the JUMP applies to,
    and 0 for all others. This follows from JUG's convention where JUMP
    is subtracted from dt_sec: d(r)/d(JUMP) = -1, so M = -d(r)/d(JUMP) = +1.
    
    Parameters
    ----------
    params : dict
        Timing model parameters (JUMPs included)
    toas_mjd : jnp.ndarray
        TOA times in MJD, shape (n_toas,)
    fit_params : list of str
        Parameters to fit (e.g., ['JUMP1', 'JUMP2'])
    toa_flags : list of dict, optional
        List of flag dictionaries for each TOA (from TIM file parsing)
    jump_masks : dict of jnp.ndarray, optional
        Pre-computed boolean masks for each JUMP parameter
        Keys are JUMP names, values are boolean arrays of shape (n_toas,)
        
    Returns
    -------
    derivatives : dict
        Dictionary mapping JUMP parameter name to design matrix column
        Each value is jnp.ndarray of shape (n_toas,) with values 0.0 or 1.0
    """
    from jug.model.parameter_spec import is_jump_param
    
    n_toas = len(toas_mjd)
    derivatives = {}
    
    # Filter to only JUMP parameters
    jump_fit_params = [p for p in fit_params if is_jump_param(p)]
    
    if not jump_fit_params:
        return derivatives
    
    # Use pre-computed masks (preferred)
    if jump_masks is not None:
        for param in jump_fit_params:
            if param in jump_masks:
                mask = jump_masks[param]
                derivatives[param] = jnp.where(mask, 1.0, 0.0)
            else:
                derivatives[param] = jnp.ones(n_toas, dtype=jnp.float64)
        return derivatives
    
    # Fallback: assume each JUMP applies to all TOAs
    for param in jump_fit_params:
        derivatives[param] = jnp.ones(n_toas, dtype=jnp.float64)
    
    return derivatives


def create_jump_mask_from_mjd_range(
    toas_mjd: jnp.ndarray,
    mjd_start: float,
    mjd_end: float
) -> jnp.ndarray:
    """Create a boolean mask for TOAs in an MJD range using JAX.
    
    Parameters
    ----------
    toas_mjd : jnp.ndarray
        TOA times in MJD
    mjd_start : float
        Start of MJD range (inclusive)
    mjd_end : float
        End of MJD range (inclusive)
        
    Returns
    -------
    mask : jnp.ndarray of bool
        True for TOAs in range, False otherwise
    """
    return (toas_mjd >= mjd_start) & (toas_mjd <= mjd_end)


def create_jump_mask_from_flags(
    toa_flags: List[Dict[str, str]],
    flag_name: str,
    flag_value: str
) -> np.ndarray:
    """Create a boolean mask for TOAs matching a flag.
    
    Note: This uses numpy since flag matching involves string comparisons
    which aren't JIT-compilable. Convert to jnp.array after creation.
    
    Parameters
    ----------
    toa_flags : list of dict
        List of flag dictionaries for each TOA
    flag_name : str
        Flag name to match (e.g., 'fe', 'sys')
    flag_value : str
        Flag value to match (e.g., 'L-wide', 'meerkat')
        
    Returns
    -------
    mask : np.ndarray of bool
        True for TOAs that match, False otherwise
    """
    mask = np.zeros(len(toa_flags), dtype=bool)
    for i, flags in enumerate(toa_flags):
        if flags.get(flag_name) == flag_value:
            mask[i] = True
    return mask


def parse_jump_from_par_line(line: str) -> Dict[str, Any]:
    """Parse a JUMP line from a par file.
    
    Formats:
    - JUMP -fe L-wide 0.0001234
    - JUMP -sys meerkat 0.0001234  
    - JUMP MJD 58000 59000 0.0001234
    
    Parameters
    ----------
    line : str
        Line from par file starting with JUMP
        
    Returns
    -------
    jump_info : dict
        Dictionary with keys: 'type', 'value', and type-specific fields
    """
    parts = line.split()
    
    if len(parts) < 2:
        return {'type': 'unknown', 'value': 0.0}
    
    # Remove 'JUMP' keyword
    parts = parts[1:]
    
    # Check for flag-based JUMP: JUMP -fe L-wide value
    if parts[0].startswith('-') and len(parts) >= 3:
        flag_name = parts[0][1:]  # Remove leading '-'
        flag_value = parts[1]
        value = float(parts[2])
        return {
            'type': 'flag',
            'flag_name': flag_name,
            'flag_value': flag_value,
            'value': value,
        }
    
    # Check for MJD-based JUMP: JUMP MJD start end value
    if parts[0].upper() == 'MJD' and len(parts) >= 4:
        mjd_start = float(parts[1])
        mjd_end = float(parts[2])
        value = float(parts[3])
        return {
            'type': 'mjd',
            'mjd_start': mjd_start,
            'mjd_end': mjd_end,
            'value': value,
        }
    
    # Fallback
    try:
        value = float(parts[-1])
        return {'type': 'unknown', 'value': value}
    except ValueError:
        return {'type': 'unknown', 'value': 0.0}


if __name__ == '__main__':
    print("Testing JAX-based JUMP derivatives...")
    
    # Test with pre-computed masks
    jump_masks = {
        'JUMP1': jnp.array([True, True, False, False, False]),
        'JUMP2': jnp.array([False, False, True, True, False]),
    }
    
    toas = jnp.array([58000, 58001, 58002, 58003, 58004], dtype=jnp.float64)
    
    derivs = compute_jump_derivatives(
        params={'JUMP1': 0.001, 'JUMP2': 0.002},
        toas_mjd=toas,
        fit_params=['JUMP1', 'JUMP2', 'F0'],
        jump_masks=jump_masks,
    )
    
    print(f"JUMP1 derivatives: {derivs['JUMP1']}")
    print(f"JUMP2 derivatives: {derivs['JUMP2']}")
    
    assert 'F0' not in derivs, "F0 should not be in JUMP derivatives"
    assert jnp.allclose(derivs['JUMP1'], jnp.array([1., 1., 0., 0., 0.])), "JUMP1 mismatch"
    assert jnp.allclose(derivs['JUMP2'], jnp.array([0., 0., 1., 1., 0.])), "JUMP2 mismatch"
    
    # Test MJD range mask
    mjd_mask = create_jump_mask_from_mjd_range(toas, 58001, 58003)
    expected = jnp.array([False, True, True, True, False])
    assert jnp.array_equal(mjd_mask, expected), "MJD mask mismatch"
    print(f"MJD range [58001, 58003] mask: {mjd_mask}")
    
    print("\n✓ JAX-based JUMP derivatives module ready!")
