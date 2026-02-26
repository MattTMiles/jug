"""
JAX Device Management for JUG
==============================

Handles CPU/GPU device selection with smart defaults and command-line control.

Design Philosophy
-----------------
- Default to CPU for typical pulsar timing (fastest for <50k TOAs, <20 params)
- Allow GPU override for large-scale analyses (PTAs, MCMC, batch fitting)
- Provide automatic selection based on problem size
- Enable easy command-line control via environment variable or flag

Usage Examples
--------------
>>> from jug.utils.device import get_device, set_device_preference
>>>
>>> # Method 1: Explicit device selection
>>> device = get_device(prefer='cpu')
>>> with jax.default_device(device):
>>>     # All JAX operations use CPU
>>>     result = fit_pulsar(...)
>>>
>>> # Method 2: Automatic selection
>>> device = get_device(n_toas=10000, n_params=2)  # Returns CPU
>>> device = get_device(n_toas=100000, n_params=50)  # Returns GPU
>>>
>>> # Method 3: Global preference (for CLI)
>>> set_device_preference('gpu')  # All subsequent calls use GPU
>>> device = get_device()  # Returns GPU

Environment Variables
--------------------
JUG_DEVICE : str
    Override device selection ('cpu', 'gpu', or 'auto')
    Example: export JUG_DEVICE=gpu
"""

import os
import jax
from typing import Optional, Literal

# Global device preference (set by CLI or environment)
_DEVICE_PREFERENCE: Optional[str] = None


def set_device_preference(device: Literal['cpu', 'gpu', 'auto'] = 'auto'):
    """
    Set global device preference for all JUG operations.
    
    This affects all subsequent get_device() calls unless overridden.
    
    Parameters
    ----------
    device : {'cpu', 'gpu', 'auto'}
        Device preference:
        - 'cpu': Force CPU for all operations
        - 'gpu': Force GPU (if available, else fall back to CPU)
        - 'auto': Smart selection based on problem size
        
    Examples
    --------
    >>> set_device_preference('cpu')  # Force CPU
    >>> set_device_preference('gpu')  # Force GPU
    >>> set_device_preference('auto')  # Smart selection (default)
    """
    global _DEVICE_PREFERENCE
    
    if device not in ('cpu', 'gpu', 'auto'):
        raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'gpu', or 'auto'")
    
    _DEVICE_PREFERENCE = device


def get_device_preference() -> str:
    """
    Get current device preference.
    
    Priority order:
    1. JUG_DEVICE environment variable
    2. Global preference set by set_device_preference()
    3. Default: 'auto'
    
    Returns
    -------
    preference : str
        One of 'cpu', 'gpu', or 'auto'
    """
    # Check environment variable first
    env_device = os.environ.get('JUG_DEVICE', '').lower()
    if env_device in ('cpu', 'gpu', 'auto'):
        return env_device
    
    # Then check global preference
    if _DEVICE_PREFERENCE is not None:
        return _DEVICE_PREFERENCE
    
    # Default to auto
    return 'auto'


def estimate_computation_cost(n_toas: int, n_params: int) -> float:
    """
    Estimate computational cost (FLOPs) for a fitting problem.
    
    Used for automatic CPU/GPU selection.
    
    Parameters
    ----------
    n_toas : int
        Number of TOAs
    n_params : int
        Number of fit parameters
        
    Returns
    -------
    flops : float
        Estimated floating point operations (rough order of magnitude)
        
    Notes
    -----
    Cost breakdown per iteration:
    - Phase computation: O(n_toas * n_params)
    - Derivative computation: O(n_toas * n_params)
    - Design matrix assembly: O(n_toas * n_params)
    - WLS solve (SVD): O(n_toas * n_params^2 + n_params^3)
    
    For typical pulsar timing:
    - Small: 10k TOAs, 2 params -> ~0.4M FLOPs
    - Medium: 50k TOAs, 10 params -> ~50M FLOPs
    - Large: 100k TOAs, 50 params -> ~2.5B FLOPs
    """
    # Rough FLOP estimate (order of magnitude)
    phase_flops = n_toas * n_params
    deriv_flops = n_toas * n_params
    design_flops = n_toas * n_params
    wls_flops = n_toas * n_params**2 + n_params**3
    
    total_flops = phase_flops + deriv_flops + design_flops + wls_flops
    return total_flops


def should_use_gpu(n_toas: int, n_params: int, threshold_flops: float = 1e8) -> bool:
    """
    Decide whether to use GPU based on problem size.
    
    GPU becomes beneficial when computation time exceeds data transfer overhead.
    For JAX, this happens around 100M FLOPs (empirically determined).
    
    Parameters
    ----------
    n_toas : int
        Number of TOAs
    n_params : int
        Number of fit parameters
    threshold_flops : float
        FLOP threshold for GPU usage (default: 100M)
        
    Returns
    -------
    use_gpu : bool
        True if GPU is recommended, False otherwise
        
    Examples
    --------
    >>> should_use_gpu(10000, 2)  # Small problem
    False
    >>> should_use_gpu(100000, 50)  # Large problem
    True
    """
    flops = estimate_computation_cost(n_toas, n_params)
    return flops >= threshold_flops


def get_device(
    prefer: Optional[Literal['cpu', 'gpu', 'auto']] = None,
    n_toas: Optional[int] = None,
    n_params: Optional[int] = None
) -> jax.Device:
    """
    Get JAX device for computation.
    
    Smart device selection based on:
    1. Explicit preference (if provided)
    2. Global preference (if set)
    3. Problem size (if n_toas and n_params provided)
    4. Default: CPU (safest for typical pulsar timing)
    
    Parameters
    ----------
    prefer : {'cpu', 'gpu', 'auto'}, optional
        Explicit device preference (overrides global setting)
    n_toas : int, optional
        Number of TOAs (for automatic selection)
    n_params : int, optional
        Number of fit parameters (for automatic selection)
        
    Returns
    -------
    device : jax.Device
        JAX device object (CPU or GPU)
        
    Examples
    --------
    >>> # Explicit CPU
    >>> device = get_device(prefer='cpu')
    >>> print(device)  # TFRT_CPU_0
    >>>
    >>> # Automatic selection
    >>> device = get_device(n_toas=10000, n_params=2)
    >>> print(device)  # TFRT_CPU_0 (small problem)
    >>>
    >>> device = get_device(n_toas=100000, n_params=50)
    >>> print(device)  # cuda:0 (large problem, if GPU available)
    """
    # Determine preference
    if prefer is None:
        prefer = get_device_preference()
    
    # Handle 'auto' mode
    if prefer == 'auto':
        if n_toas is not None and n_params is not None:
            # Smart selection based on problem size
            prefer = 'gpu' if should_use_gpu(n_toas, n_params) else 'cpu'
        else:
            # Default to CPU (safest for typical use)
            prefer = 'cpu'
    
    # Get requested device
    if prefer == 'cpu':
        return jax.devices('cpu')[0]
    elif prefer == 'gpu':
        # Try GPU, fall back to CPU if not available
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                return gpu_devices[0]
        except:
            pass
        # GPU not available, fall back to CPU
        return jax.devices('cpu')[0]
    else:
        raise ValueError(f"Invalid preference: {prefer}")


def list_available_devices() -> dict:
    """
    List all available JAX devices.
    
    Useful for debugging and user information.
    
    Returns
    -------
    devices : dict
        Dictionary with keys 'cpu' and 'gpu' listing available devices
        
    Examples
    --------
    >>> devices = list_available_devices()
    >>> print(devices)
    {'cpu': [TFRT_CPU_0], 'gpu': [cuda:0, cuda:1]}
    """
    devices = {
        'cpu': [],
        'gpu': []
    }
    
    try:
        devices['cpu'] = jax.devices('cpu')
    except:
        pass
    
    try:
        devices['gpu'] = jax.devices('gpu')
    except:
        pass
    
    return devices


def print_device_info(verbose: bool = True):
    """
    Print information about available devices and current preference.
    
    Parameters
    ----------
    verbose : bool
        If True, print detailed device information
        
    Examples
    --------
    >>> print_device_info()
    JUG Device Configuration:
      Preference: auto
      CPU devices: 1 available
      GPU devices: 2 available (cuda:0, cuda:1)
      Current selection: CPU (TFRT_CPU_0)
    """
    devices = list_available_devices()
    preference = get_device_preference()
    current = get_device()
    
    if verbose:
        print("JUG Device Configuration:")
        print(f"  Preference: {preference}")
        print(f"  CPU devices: {len(devices['cpu'])} available")
        if devices['cpu']:
            print(f"    {devices['cpu']}")
        print(f"  GPU devices: {len(devices['gpu'])} available")
        if devices['gpu']:
            print(f"    {devices['gpu']}")
        print(f"  Current selection: {current}")
    else:
        device_type = 'CPU' if 'cpu' in str(current).lower() else 'GPU'
        print(f"Using {device_type} ({current})")


# Initialize with environment variable if set
_env_pref = os.environ.get('JUG_DEVICE', '').lower()
if _env_pref in ('cpu', 'gpu', 'auto'):
    _DEVICE_PREFERENCE = _env_pref
