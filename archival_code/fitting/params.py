"""Parameter handling for fitting.

Handles packing/unpacking parameters, scaling, and masking based on FIT flags.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Parameter scaling factors for numerical stability
PARAM_SCALES = {
    # Spin parameters
    'F0': 1e2,      # ~100 Hz
    'F1': 1e-15,    # ~1e-15 Hz/s
    'F2': 1e-25,    # ~1e-25 Hz/s²
    
    # Astrometry
    'RAJ': 1.0,     # radians
    'DECJ': 1.0,    # radians
    'PMRA': 1e-3,   # ~mas/yr in rad/s
    'PMDEC': 1e-3,
    'PX': 1e-3,     # ~mas in radians
    
    # DM
    'DM': 1e1,      # ~10 pc/cm³
    'DM1': 1e-3,
    'DM2': 1e-4,
    
    # Binary (ELL1)
    'PB': 1.0,      # ~1 day
    'A1': 1.0,      # ~1 lt-s
    'TASC': 1e5,    # MJD ~50000-60000
    'EPS1': 1e-5,   # Small eccentricity
    'EPS2': 1e-5,
    'PBDOT': 1e-12,
    'XDOT': 1e-14,
    'M2': 1.0,      # solar masses
    'SINI': 1.0,    # dimensionless
}


def get_param_scales() -> Dict[str, float]:
    """Get parameter scaling factors."""
    return PARAM_SCALES.copy()


def extract_fittable_params(
    params: Dict,
    fit_flags: Optional[Dict[str, int]] = None,
    force_fit: Optional[List[str]] = None,
    force_freeze: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """Extract parameters to fit based on FIT flags.
    
    Parameters
    ----------
    params : dict
        All timing model parameters from .par file
    fit_flags : dict, optional
        FIT flags from .par file (1=fit, 0=freeze)
        If None, attempts to extract from params dict
    force_fit : list of str, optional
        Parameter names to force fit (overrides FIT flags)
    force_freeze : list of str, optional
        Parameter names to force freeze (overrides FIT flags)
        
    Returns
    -------
    param_names : list of str
        Names of parameters to fit
    initial_values : dict
        Initial values of fittable parameters
    fixed_params : dict
        Parameters not being fitted
    """
    # Start with all parameters as fixed
    fixed_params = params.copy()
    fittable = {}
    
    # Determine which parameters to fit
    for name, value in params.items():
        # Skip non-numeric parameters
        if not isinstance(value, (int, float, np.number)):
            continue
            
        # Check FIT flags
        should_fit = False
        
        if force_fit and name in force_fit:
            should_fit = True
        elif force_freeze and name in force_freeze:
            should_fit = False
        elif fit_flags and name in fit_flags:
            should_fit = fit_flags[name] == 1
        elif isinstance(value, dict) and 'fit' in value:
            # Handle case where params are stored as dicts with 'value' and 'fit' keys
            should_fit = value.get('fit', 0) == 1
        
        if should_fit:
            fittable[name] = float(value)
            del fixed_params[name]
    
    # Get ordered list of parameter names
    param_names = sorted(fittable.keys())
    initial_values = {name: fittable[name] for name in param_names}
    
    return param_names, initial_values, fixed_params


def pack_params(param_dict: Dict[str, float], param_names: List[str]) -> np.ndarray:
    """Pack parameter dict into array for optimization.
    
    Parameters
    ----------
    param_dict : dict
        Parameter values
    param_names : list of str
        Ordered list of parameter names
        
    Returns
    -------
    array : ndarray
        Parameter values in array form (scaled)
    """
    scales = get_param_scales()
    values = []
    for name in param_names:
        value = param_dict[name]
        scale = scales.get(name, 1.0)
        values.append(value / scale)
    return np.array(values)


def unpack_params(
    param_array: np.ndarray,
    param_names: List[str],
    fixed_params: Dict[str, float]
) -> Dict[str, float]:
    """Unpack parameter array into dict with all parameters.
    
    Parameters
    ----------
    param_array : ndarray
        Scaled parameter values being optimized
    param_names : list of str
        Ordered list of parameter names
    fixed_params : dict
        Parameters not being fitted
        
    Returns
    -------
    all_params : dict
        Complete parameter dict (fitted + fixed)
    """
    scales = get_param_scales()
    all_params = fixed_params.copy()
    
    for i, name in enumerate(param_names):
        scale = scales.get(name, 1.0)
        all_params[name] = float(param_array[i]) * scale
    
    return all_params
