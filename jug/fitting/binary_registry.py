"""
Binary Model Registry - Extensible dispatch for binary model computations.

This module provides a registry pattern for binary models, allowing new models
to be added without modifying dispatch logic in multiple places.

Usage
-----
To add a new binary model:

1. Create a module with delay and derivatives functions:
   - compute_XXX_binary_delay(toas_bary_mjd, params) -> array
   - compute_binary_derivatives_XXX(params, toas_bary_mjd, param_list) -> dict

2. Register the model in this file:
   register_binary_model('XXX', compute_XXX_binary_delay, compute_binary_derivatives_XXX)

3. The model will automatically work everywhere in JUG.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

# Type aliases for clarity
DelayFunc = Callable[[np.ndarray, Dict], np.ndarray]
DerivativesFunc = Callable[[Dict, np.ndarray, List[str]], Dict[str, np.ndarray]]

# The registry: maps model names to (delay_func, derivatives_func)
_BINARY_MODEL_REGISTRY: Dict[str, Tuple[DelayFunc, DerivativesFunc]] = {}


def register_binary_model(
    model_names: str | List[str],
    delay_func: DelayFunc,
    derivatives_func: DerivativesFunc
) -> None:
    """
    Register a binary model with its delay and derivatives functions.

    Parameters
    ----------
    model_names : str or list of str
        Model name(s) as they appear in BINARY parameter (e.g., 'DD', 'ELL1').
        Multiple names can be registered for aliases (e.g., ['DD', 'DDK', 'DDS']).
    delay_func : callable
        Function with signature: (toas_bary_mjd: array, params: dict) -> array
        Returns binary delay in seconds.
    derivatives_func : callable
        Function with signature: (params: dict, toas_bary_mjd: array, param_list: list) -> dict
        Returns dict mapping parameter names to derivative arrays.

    Examples
    --------
    >>> from jug.fitting.derivatives_dd import compute_dd_binary_delay, compute_binary_derivatives_dd
    >>> register_binary_model('DD', compute_dd_binary_delay, compute_binary_derivatives_dd)
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    for name in model_names:
        _BINARY_MODEL_REGISTRY[name.upper()] = (delay_func, derivatives_func)


def get_binary_delay_func(binary_model: str) -> Optional[DelayFunc]:
    """
    Get the delay function for a binary model.

    Parameters
    ----------
    binary_model : str
        Binary model name (e.g., 'DD', 'ELL1')

    Returns
    -------
    delay_func : callable or None
        The delay computation function, or None if model not registered.
    """
    entry = _BINARY_MODEL_REGISTRY.get(binary_model.upper())
    return entry[0] if entry else None


def get_binary_derivatives_func(binary_model: str) -> Optional[DerivativesFunc]:
    """
    Get the derivatives function for a binary model.

    Parameters
    ----------
    binary_model : str
        Binary model name (e.g., 'DD', 'ELL1')

    Returns
    -------
    derivatives_func : callable or None
        The derivatives computation function, or None if model not registered.
    """
    entry = _BINARY_MODEL_REGISTRY.get(binary_model.upper())
    return entry[1] if entry else None


def compute_binary_delay(toas_bary: np.ndarray, params: Dict) -> np.ndarray:
    """
    Compute binary delay using the correct model from params['BINARY'].

    This is the main entry point for binary delay computation throughout JUG.
    It automatically routes to the correct model implementation.

    Parameters
    ----------
    toas_bary : np.ndarray
        Barycentric TOA times in MJD
    params : dict
        Parameter dictionary (checks 'BINARY' key for model name)

    Returns
    -------
    delay : np.ndarray
        Binary delay in seconds. Returns zeros if no binary model.

    Raises
    ------
    ValueError
        If the binary model is not registered.
    """
    binary_model = params.get('BINARY', '').upper().strip()

    # No binary model - return zeros
    if not binary_model:
        return np.zeros_like(toas_bary)

    delay_func = get_binary_delay_func(binary_model)
    if delay_func is None:
        registered = list(_BINARY_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown binary model: '{binary_model}'. "
            f"Registered models: {registered}. "
            f"To add support, use register_binary_model() in binary_registry.py."
        )

    return np.asarray(delay_func(toas_bary, params))


def compute_binary_derivatives(
    params: Dict,
    toas_bary: np.ndarray,
    param_list: List[str],
    obs_pos_ls: np.ndarray = None
) -> Dict[str, np.ndarray]:
    """
    Compute binary parameter derivatives using the correct model.

    Parameters
    ----------
    params : dict
        Parameter dictionary (checks 'BINARY' key for model name)
    toas_bary : np.ndarray
        Barycentric TOA times in MJD
    param_list : list of str
        Binary parameters to compute derivatives for
    obs_pos_ls : np.ndarray, optional
        Observer position in light-seconds relative to SSB, shape (N, 3).
        Required for DDK Kopeikin 1995 parallax corrections.

    Returns
    -------
    derivatives : dict
        Maps parameter names to derivative arrays.

    Raises
    ------
    ValueError
        If the binary model is not registered.
    """
    binary_model = params.get('BINARY', '').upper().strip()

    if not binary_model:
        raise ValueError("Cannot compute binary derivatives: no BINARY model specified")

    deriv_func = get_binary_derivatives_func(binary_model)
    if deriv_func is None:
        registered = list(_BINARY_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown binary model: '{binary_model}'. "
            f"Registered models: {registered}."
        )

    # For DDK, pass obs_pos_ls for Kopeikin parallax corrections
    if binary_model == 'DDK':
        return deriv_func(params, toas_bary, param_list, obs_pos_ls=obs_pos_ls)
    else:
        return deriv_func(params, toas_bary, param_list)


def list_registered_models() -> List[str]:
    """Return list of all registered binary model names."""
    return list(_BINARY_MODEL_REGISTRY.keys())


def is_model_registered(binary_model: str) -> bool:
    """Check if a binary model is registered."""
    return binary_model.upper() in _BINARY_MODEL_REGISTRY


# =============================================================================
# Register built-in binary models
# =============================================================================

def _register_builtin_models():
    """Register all built-in binary models."""

    # DD family (Damour-Deruelle and variants)
    # All use the same core DD computation
    from jug.fitting.derivatives_dd import (
        compute_dd_binary_delay,
        compute_binary_derivatives_dd,
        compute_binary_derivatives_ddk
    )
    # DD and variants without KIN/KOM
    register_binary_model(
        ['DD', 'DDS', 'DDH', 'DDGR'],
        compute_dd_binary_delay,
        compute_binary_derivatives_dd
    )
    
    # DDK uses dedicated derivatives function with KIN/KOM support
    register_binary_model(
        ['DDK'],
        compute_dd_binary_delay,
        compute_binary_derivatives_ddk
    )

    # BT (Blandford-Teukolsky) - uses DD-style parameterization
    # Register separately in case it needs different handling later
    register_binary_model(
        ['BT', 'BTX'],
        compute_dd_binary_delay,
        compute_binary_derivatives_dd
    )

    # ELL1 (low-eccentricity model)
    from jug.fitting.derivatives_binary import (
        compute_ell1_binary_delay,
        compute_binary_derivatives_ell1
    )
    register_binary_model(
        ['ELL1', 'ELL1H', 'ELL1K'],
        compute_ell1_binary_delay,
        compute_binary_derivatives_ell1
    )


# Auto-register builtin models on import
_register_builtin_models()
