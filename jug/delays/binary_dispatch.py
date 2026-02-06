"""Binary model dispatcher - routes to appropriate binary delay calculator.

This module provides a clean dispatch system for selecting binary models.
Each model has an optimized JAX-compiled implementation.

Supported Models
----------------
- ELL1/ELL1H : Low-eccentricity (Lange et al. 2001)
- BT/BTX    : Blandford-Teukolsky (1976)  
- DD/DDH/DDGR/DDK : Damour-Deruelle (1985)
- T2        : Tempo2 general model (Edwards et al. 2006)

Adding New Models
-----------------
To add support for a new binary model:

1. Create a new file in jug/delays/binary_MODELNAME.py
2. Implement a @jax.jit function with signature:
   
   def modelname_binary_delay(t_topo_tdb, param1, param2, ...) -> float

3. Add entry to BINARY_MODELS dict in this file
4. Add parameter extraction logic to extract_binary_params()
5. Update combined.py to call dispatch_binary_delay()

Examples
--------
>>> # Dispatch automatically selects the right model
>>> delay = dispatch_binary_delay('DD', t_topo_tdb, params)
"""

import jax.numpy as jnp
from jug.delays.binary_bt import bt_binary_delay
from jug.delays.binary_dd import dd_binary_delay  
from jug.delays.binary_t2 import t2_binary_delay


def dispatch_binary_delay(model_name, t_topo_tdb, params):
    """Dispatch to the appropriate binary delay calculator.
    
    Parameters
    ----------
    model_name : str
        Binary model name (e.g., 'ELL1', 'DD', 'BT', 'T2')
    t_topo_tdb : float or jnp.ndarray
        Topocentric TDB time (MJD) after removing non-binary delays
    params : dict
        Dictionary of binary parameters (extracted from .par file)
        
    Returns
    -------
    float or jnp.ndarray
        Binary delay in seconds (Roemer + Einstein + Shapiro)
        
    Raises
    ------
    ValueError
        If model_name is not supported
        
    Notes
    -----
    The ELL1 model has special treatment for performance - it's computed
    inline in combined_delays() to maximize JIT optimization. This dispatcher
    is for all other models.
    
    Examples
    --------
    >>> params = {
    ...     'PB': 1.0, 'A1': 10.0, 'ECC': 0.1, 'OM': 45.0, 'T0': 55000.0,
    ...     'GAMMA': 0.0, 'PBDOT': 0.0, 'OMDOT': 0.0, 'XDOT': 0.0, 'EDOT': 0.0,
    ...     'M2': 0.3, 'SINI': 0.9
    ... }
    >>> delay = dispatch_binary_delay('DD', 58000.0, params)
    """
    model = model_name.upper()
    
    # ELL1 models use inline code in combined_delays() for maximum performance
    if model in ('ELL1', 'ELL1H'):
        raise ValueError(
            "ELL1 model is computed inline in combined_delays() for performance. "
            "This dispatcher should not be called for ELL1 models."
        )
    
    # BT and its variants
    elif model in ('BT', 'BTX'):
        return bt_binary_delay(
            t_topo_tdb,
            pb=params['PB'],
            a1=params['A1'],
            ecc=params['ECC'],
            om=params['OM'],
            t0=params['T0'],
            gamma=params.get('GAMMA', 0.0),
            pbdot=params.get('PBDOT', 0.0),
            xdot=params.get('XDOT', 0.0),
            omdot=params.get('OMDOT', 0.0),
            edot=params.get('EDOT', 0.0),
            m2=params.get('M2', 0.0),
            sini=params.get('SINI', 0.0)
        )
    
    # DDK uses combined.py:branch_ddk() which handles Kopeikin corrections.
    # This dispatcher cannot compute DDK correctly (needs observer positions).
    elif model == 'DDK':
        raise ValueError(
            "DDK binary model requires Kopeikin corrections that need observer "
            "positions (obs_pos_ls). Use combined.py:branch_ddk() for DDK delays. "
            "This dispatcher only handles models that don't need per-TOA geometry."
        )

    # DD and its variants (DDH, DDGR)
    elif model in ('DD', 'DDH', 'DDGR'):
        return dd_binary_delay(
            t_topo_tdb,
            pb_days=params['PB'],
            a1_lt_sec=params['A1'],
            ecc=params['ECC'],
            omega_deg=params['OM'],
            t0_mjd=params['T0'],
            gamma_sec=params.get('GAMMA', 0.0),
            pbdot=params.get('PBDOT', 0.0),
            xdot=params.get('XDOT', 0.0),
            omdot_deg_yr=params.get('OMDOT', 0.0),
            edot=params.get('EDOT', 0.0),
            m2_msun=params.get('M2', 0.0),
            sini=params.get('SINI', 0.0)
        )
    
    # T2 general model
    elif model == 'T2':
        return t2_binary_delay(
            t_topo_tdb,
            pb=params['PB'],
            a1=params['A1'],
            ecc=params['ECC'],
            om=params['OM'],
            t0=params['T0'],
            gamma=params.get('GAMMA', 0.0),
            pbdot=params.get('PBDOT', 0.0),
            xdot=params.get('XDOT', 0.0),
            edot=params.get('EDOT', 0.0),
            omdot=params.get('OMDOT', 0.0),
            m2=params.get('M2', 0.0),
            sini=params.get('SINI', 0.0),
            kin=params.get('KIN', 0.0),
            kom=params.get('KOM', 0.0)
        )
    
    else:
        raise ValueError(
            f"Unsupported binary model: {model}. "
            f"Supported models: ELL1, ELL1H, BT, BTX, DD, DDH, DDGR, DDK, T2"
        )


# Binary model registry (for documentation and validation)
BINARY_MODELS = {
    'ELL1': {
        'name': 'ELL1 (Low-eccentricity)',
        'required_params': ['PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'optional_params': ['PBDOT', 'XDOT', 'GAMMA', 'M2', 'SINI', 'H3', 'STIG'],
        'inline': True  # Computed inline in combined_delays()
    },
    'ELL1H': {
        'name': 'ELL1H (Low-eccentricity with H3/H4)',
        'required_params': ['PB', 'A1', 'TASC', 'EPS1', 'EPS2'],
        'optional_params': ['PBDOT', 'XDOT', 'GAMMA', 'H3', 'H4'],
        'inline': True
    },
    'BT': {
        'name': 'BT (Blandford-Teukolsky)',
        'required_params': ['PB', 'A1', 'ECC', 'OM', 'T0'],
        'optional_params': ['GAMMA', 'PBDOT', 'XDOT', 'OMDOT', 'EDOT', 'M2', 'SINI'],
        'inline': False
    },
    'DD': {
        'name': 'DD (Damour-Deruelle)',
        'required_params': ['PB', 'A1', 'ECC', 'OM', 'T0'],
        'optional_params': ['GAMMA', 'PBDOT', 'XDOT', 'OMDOT', 'EDOT', 'M2', 'SINI'],
        'inline': False
    },
    'T2': {
        'name': 'T2 (Tempo2 general)',
        'required_params': ['PB', 'A1', 'ECC', 'OM', 'T0'],
        'optional_params': ['GAMMA', 'PBDOT', 'XDOT', 'EDOT', 'OMDOT', 'M2', 'SINI', 'KIN', 'KOM'],
        'inline': False
    },
    'DDK': {
        'name': 'DDK (DD + Kopeikin 1995/1996)',
        'required_params': ['PB', 'A1', 'ECC', 'OM', 'T0', 'KIN', 'KOM'],
        'optional_params': ['GAMMA', 'PBDOT', 'XDOT', 'OMDOT', 'EDOT', 'M2', 'SINI',
                            'PX', 'PMRA', 'PMDEC', 'K96'],
        'inline': False  # Forward model in combined.py:branch_ddk()
    }
}
