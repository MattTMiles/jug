"""Timescale conversion utilities for TCB <-> TDB parameter transformation.

This module implements TCB to TDB conversion following the Irwin & Fukushima (1999)
convention, matching the approach used in PINT and Tempo2.

Key concepts:
- TCB (Barycentric Coordinate Time): relativistically correct coordinate time
- TDB (Barycentric Dynamical Time): approximation to TCB with rate 1-L_B slower
- IFTE_K = 1 + 1.55051979176e-8: the scale factor between TCB and TDB rates

TCB metadata (tcb_scaling_dim, is_epoch) is defined in the ParameterSpec registry
(parameter_spec.py) as the single source of truth. This module derives its
conversion lists from that registry, supplemented by indexed/pattern parameters
(glitch, exponential dip, multi-companion) that are not individually registered.

Reference: Irwin & Fukushima (1999), A&A 348, 642-652
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

__all__ = [
    'IFTE_MJD0',
    'IFTE_KM1', 
    'IFTE_K',
    'parse_timescale',
    'convert_tcb_epoch_to_tdb',
    'convert_tdb_epoch_to_tcb',
    'scale_parameter_tcb_to_tdb',
    'scale_parameter_tdb_to_tcb',
    'convert_par_params_to_tdb',
    'convert_par_params_to_tcb',
]

# Constants from Irwin & Fukushima 1999
# These are the same as used in PINT and Tempo2 (as of Feb 2023)
IFTE_MJD0 = np.longdouble("43144.0003725")  # Reference epoch MJD
IFTE_KM1 = np.longdouble("1.55051979176e-8")  # L_B: rate difference
IFTE_K = np.longdouble(1.0) + IFTE_KM1  # Scale factor ~= 1.0000000155051979176

# ---------------------------------------------------------------------------
# Build EPOCH_PARAMETERS and SCALED_PARAMETERS from the ParameterSpec registry
# supplemented by indexed/pattern params not in the registry.
# ---------------------------------------------------------------------------
from jug.model.parameter_spec import (
    get_scaled_parameters as _registry_scaled,
    get_epoch_parameters as _registry_epochs,
)

# Indexed epoch parameters (glitch, exponential dip, multi-companion)
_INDEXED_EPOCH_PARAMS = {
    'START', 'FINISH', 'PBEPOCH',
    *(f'GLEP_{i}' for i in range(1, 6)),
    'T0_2', 'TASC_2', 'T0_3', 'TASC_3',
    *(f'EXPEP_{i}' for i in range(1, 11)),
    *(f'GAUSEP_{i}' for i in range(1, 6)),
}

# Indexed scaled parameters (glitch, exponential dip, multi-companion, high F-deriv)
_INDEXED_SCALED_PARAMS = [
    # Higher spin frequency derivatives (F4-F12)
    *((f'F{i}', -(i + 1)) for i in range(4, 13)),
    # DM3 (not in registry)
    ('DM3', -4),
    # Glitch frequency parameters
    *((f'GLF0_{i}', -1) for i in range(1, 6)),
    *((f'GLF1_{i}', -2) for i in range(1, 6)),
    *((f'GLF0D_{i}', -1) for i in range(1, 6)),
    # Multi-companion binary
    ('A1_2', 1), ('PB_2', 1), ('FB0_2', -1), ('FB1_2', -2),
    ('GAMMA_2', 1), ('OMDOT_2', -1),
    ('ECC_2', 0), ('OM_2', 0), ('SINI_2', 0), ('M2_2', 0),
    ('EPS1_2', 0), ('EPS2_2', 0),
    ('A1_3', 1), ('PB_3', 1),
    # Exponential dip parameters
    *((f'EXPPH_{i}', 1) for i in range(1, 11)),
    *((f'EXPTAU_{i}', 1) for i in range(1, 11)),
]

# Combine registry + indexed params
EPOCH_PARAMETERS = _registry_epochs() | _INDEXED_EPOCH_PARAMS

_registry_scaled_list = _registry_scaled()
_all_scaled_names = {p[0] for p in _registry_scaled_list}
# Append indexed params, skipping any already covered by the registry
SCALED_PARAMETERS = _registry_scaled_list + [
    (name, dim) for name, dim in _INDEXED_SCALED_PARAMS
    if name not in _all_scaled_names
]

# Parameters NOT converted (per PINT convention)
NO_CONVERSION_PARAMETERS = {
    'TZRMJD', 'TZRFRQ',  # TZR reference
    'DMJUMP',  # DM jumps
    'EFAC', 'EQUAD', 'ECORR',  # Noise scaling factors
    'DMEFAC',  # DM noise scaling
    'RNAMP', 'RNIDX', 'TRNAMP', 'TRNIDX',  # Red noise
    'TNREDAMP', 'TNREDGAM', 'TNREDC',  # Red noise
}

# FD parameters (frequency-dependent delays) are also not converted
# They are identified by pattern matching in convert_par_params_to_tdb


def parse_timescale(params: Dict[str, Any]) -> str:
    """Extract timescale from params dict.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary (from parse_par_file)
        
    Returns
    -------
    str
        Timescale: 'TDB', 'TCB', or 'TT'
    """
    return params.get('_par_timescale', 'TDB')


def convert_tcb_epoch_to_tdb(mjd_tcb: np.longdouble) -> np.longdouble:
    """Convert a single MJD from TCB to TDB.
    
    Formula: t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0
    
    Parameters
    ----------
    mjd_tcb : np.longdouble
        MJD in TCB
        
    Returns
    -------
    np.longdouble
        MJD in TDB
    """
    return (mjd_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0


def convert_tdb_epoch_to_tcb(mjd_tdb: np.longdouble) -> np.longdouble:
    """Convert a single MJD from TDB to TCB.
    
    Formula: t_tcb = (t_tdb - IFTE_MJD0) * IFTE_K + IFTE_MJD0
    
    Parameters
    ----------
    mjd_tdb : np.longdouble
        MJD in TDB
        
    Returns
    -------
    np.longdouble
        MJD in TCB
    """
    return (mjd_tdb - IFTE_MJD0) * IFTE_K + IFTE_MJD0


def scale_parameter_tcb_to_tdb(value: float, effective_dimensionality: int) -> float:
    """Scale a parameter from TCB to TDB based on effective dimensionality.
    
    Formula: x_tdb = x_tcb * IFTE_K^(-n)
    where n = effective_dimensionality
    
    Examples:
    - F0 (frequency): n = -1 -> F0_tdb = F0_tcb * IFTE_K^1
    - A1 (time): n = 1 -> A1_tdb = A1_tcb * IFTE_K^(-1)
    - PBDOT (dimensionless): n = 0 -> no scaling
    
    Parameters
    ----------
    value : float
        Parameter value in TCB
    effective_dimensionality : int
        Power of time dimension in parameter
        
    Returns
    -------
    float
        Parameter value in TDB
    """
    if effective_dimensionality == 0:
        return value
    
    # x_tdb = x_tcb * IFTE_K^(-n)
    factor = IFTE_K ** (-effective_dimensionality)
    return value * float(factor)


def scale_parameter_tdb_to_tcb(value: float, effective_dimensionality: int) -> float:
    """Scale a parameter from TDB to TCB based on effective dimensionality.
    
    Formula: x_tcb = x_tdb * IFTE_K^n
    
    Parameters
    ----------
    value : float
        Parameter value in TDB
    effective_dimensionality : int
        Power of time dimension in parameter
        
    Returns
    -------
    float
        Parameter value in TCB
    """
    if effective_dimensionality == 0:
        return value
    
    factor = IFTE_K ** effective_dimensionality
    return value * float(factor)


def convert_par_params_to_tdb(params: Dict[str, Any], 
                               verbose: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Convert all TCB parameters to TDB in place.
    
    This is the main entry point for TCB -> TDB conversion. It:
    1. Converts all MJD epoch parameters
    2. Scales frequency, DM, binary parameters appropriately
    3. Handles prefix parameters (DMX_*, DMXR1_*, DMXR2_*)
    4. Sets metadata about the conversion
    5. Returns a log of changes for verbose output
    
    Parameters
    ----------
    params : dict
        Parameter dictionary (modified in place)
    verbose : bool
        If True, print conversion details
        
    Returns
    -------
    params : dict
        Modified parameter dictionary (same object)
    conversion_log : list of str
        List of conversion messages
    """
    log = []
    
    # Check if already TDB
    timescale = parse_timescale(params)
    if timescale == 'TDB':
        log.append("Parameters already in TDB, no conversion needed")
        return params, log
    
    if timescale != 'TCB':
        log.append(f"Warning: Unknown timescale '{timescale}', assuming TDB")
        return params, log
    
    log.append(f"Converting parameters from TCB to TDB")
    
    # Store original timescale
    params['_timescale_in'] = 'TCB'
    
    # Get the high-precision string dict (used by get_longdouble for full precision)
    hp = params.get('_high_precision', {})
    
    # Convert epoch parameters
    epoch_converted = []
    for param_name in EPOCH_PARAMETERS:
        if param_name in params:
            old_val = params[param_name]
            if old_val is not None:
                try:
                    # Use high-precision string if available for maximum precision
                    if param_name in hp:
                        old_ld = np.longdouble(hp[param_name].replace('D', 'E').replace('d', 'e'))
                    else:
                        old_ld = np.longdouble(old_val)
                    new_ld = convert_tcb_epoch_to_tdb(old_ld)
                    params[param_name] = float(new_ld)
                    # Update _high_precision with converted value string
                    if hp is not None:
                        hp[param_name] = f"{new_ld:.20g}"
                    epoch_converted.append(f"{param_name}: {old_val:.9f} -> {float(new_ld):.9f}")
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values
    
    # Convert prefix epoch parameters (DMXR1_*, DMXR2_*, etc.)
    for key in list(params.keys()):
        if key.startswith('DMXR1_') or key.startswith('DMXR2_'):
            old_val = params[key]
            if old_val is not None:
                try:
                    if key in hp:
                        old_ld = np.longdouble(hp[key].replace('D', 'E').replace('d', 'e'))
                    else:
                        old_ld = np.longdouble(old_val)
                    new_ld = convert_tcb_epoch_to_tdb(old_ld)
                    params[key] = float(new_ld)
                    if hp is not None:
                        hp[key] = f"{new_ld:.20g}"
                    epoch_converted.append(f"{key}: {old_val:.6f} -> {float(new_ld):.6f}")
                except (ValueError, TypeError):
                    pass
    
    if epoch_converted and verbose:
        log.append(f"Converted {len(epoch_converted)} epoch parameters:")
        log.extend([f"  {msg}" for msg in epoch_converted])
    
    # Scale parameters based on effective dimensionality
    scaled_converted = []
    for param_pattern, eff_dim in SCALED_PARAMETERS:
        if param_pattern in params:
            old_val = params[param_pattern]
            if old_val is not None and eff_dim != 0:
                try:
                    # Use high-precision string if available
                    if param_pattern in hp:
                        old_ld = np.longdouble(hp[param_pattern].replace('D', 'E').replace('d', 'e'))
                        # Scale in longdouble to preserve full precision
                        factor_ld = IFTE_K ** (-eff_dim)
                        new_ld = old_ld * factor_ld
                        new_val = float(new_ld)
                    else:
                        new_val = scale_parameter_tcb_to_tdb(float(old_val), eff_dim)
                    params[param_pattern] = new_val
                    # Update _high_precision with converted value
                    if hp is not None and param_pattern in hp:
                        hp[param_pattern] = f"{new_ld:.20g}"
                    # Only log significant changes
                    rel_change = abs((new_val - old_val) / old_val) if old_val != 0 else 0
                    if rel_change > 1e-10:
                        scaled_converted.append(
                            f"{param_pattern}: {old_val:.12e} -> {new_val:.12e} "
                            f"(eff_dim={eff_dim})"
                        )
                except (ValueError, TypeError):
                    pass
    
    # Convert prefix scaled parameters (DMX_* values)
    for key in list(params.keys()):
        if key.startswith('DMX_'):
            old_val = params[key]
            if old_val is not None:
                try:
                    # DMX has same dimensionality as DM (eff_dim = -1)
                    if key in hp:
                        old_ld = np.longdouble(hp[key].replace('D', 'E').replace('d', 'e'))
                        # Scale in longdouble to preserve full precision
                        factor_ld = IFTE_K  # eff_dim=-1, so factor = IFTE_K^1
                        new_ld = old_ld * factor_ld
                        new_val = float(new_ld)
                    else:
                        new_val = scale_parameter_tcb_to_tdb(float(old_val), -1)
                    params[key] = new_val
                    if hp is not None and key in hp:
                        hp[key] = f"{new_ld:.20g}"
                    rel_change = abs((new_val - old_val) / old_val) if old_val != 0 else 0
                    if rel_change > 1e-10:
                        scaled_converted.append(f"{key}: {old_val:.9e} -> {new_val:.9e}")
                except (ValueError, TypeError):
                    pass
    
    if scaled_converted and verbose:
        log.append(f"Scaled {len(scaled_converted)} parameters:")
        log.extend([f"  {msg}" for msg in scaled_converted])
    
    # Scale JUMP values (time dimension, eff_dim=1)
    # JUMPs are stored as raw par-file lines in _jump_lines and also as
    # numbered JUMP1, JUMP2, ... entries parsed from those lines.
    jump_lines = params.get('_jump_lines', [])
    jump_count = 0
    if jump_lines:
        import re
        scale_factor = float(IFTE_K ** (-1))  # 1/(1+LB) for time dimension
        new_jump_lines = []
        for line in jump_lines:
            parts = line.split()
            # JUMP lines: "JUMP -flag value [fit] [err]" or "JUMP MJD start end value ..."
            # The numeric JUMP value is at a known position
            scaled_line = line
            if len(parts) >= 4 and parts[1].startswith('-'):
                # Flag-based: JUMP -flag flagval value [fit] [err]
                try:
                    old_val = float(parts[3])
                    new_val = old_val * scale_factor
                    parts[3] = f"{new_val:.15g}"
                    scaled_line = ' '.join(parts)
                    jump_count += 1
                except (ValueError, IndexError):
                    pass
            elif len(parts) >= 5 and parts[1].upper() == 'MJD':
                # MJD-based: JUMP MJD start end value [fit] [err]
                try:
                    old_val = float(parts[4])
                    new_val = old_val * scale_factor
                    parts[4] = f"{new_val:.15g}"
                    scaled_line = ' '.join(parts)
                    jump_count += 1
                except (ValueError, IndexError):
                    pass
            new_jump_lines.append(scaled_line)
        params['_jump_lines'] = new_jump_lines
        
        # Also scale the numbered JUMP1, JUMP2, ... values
        for key in list(params.keys()):
            if re.match(r'^JUMP\d+$', key):
                try:
                    old_val = float(params[key])
                    params[key] = old_val * scale_factor
                except (ValueError, TypeError):
                    pass
    
    if jump_count > 0:
        log.append(f"Scaled {jump_count} JUMP values (time dimension)")
    
    # Update metadata
    params['UNITS'] = 'TDB'
    params['_par_timescale'] = 'TDB'
    params['_tcb_converted'] = True
    
    log.append(f"Conversion complete: {len(epoch_converted)} epochs, {len(scaled_converted)} scaled params")
    
    return params, log


def convert_par_params_to_tcb(params: Dict[str, Any],
                               verbose: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Convert all TDB parameters to TCB in place (inverse conversion).
    
    This is mainly for testing/verification purposes.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary (modified in place)
    verbose : bool
        If True, print conversion details
        
    Returns
    -------
    params : dict
        Modified parameter dictionary
    conversion_log : list of str
        List of conversion messages
    """
    log = []
    
    timescale = parse_timescale(params)
    if timescale == 'TCB':
        log.append("Parameters already in TCB, no conversion needed")
        return params, log
    
    log.append(f"Converting parameters from TDB to TCB")
    
    # Convert epoch parameters
    for param_name in EPOCH_PARAMETERS:
        if param_name in params and params[param_name] is not None:
            try:
                old_ld = np.longdouble(params[param_name])
                new_ld = convert_tdb_epoch_to_tcb(old_ld)
                params[param_name] = float(new_ld)
            except (ValueError, TypeError):
                pass
    
    # Convert prefix epoch parameters
    for key in list(params.keys()):
        if key.startswith('DMXR1_') or key.startswith('DMXR2_'):
            if params[key] is not None:
                try:
                    old_ld = np.longdouble(params[key])
                    new_ld = convert_tdb_epoch_to_tcb(old_ld)
                    params[key] = float(new_ld)
                except (ValueError, TypeError):
                    pass
    
    # Scale parameters (inverse)
    for param_pattern, eff_dim in SCALED_PARAMETERS:
        if param_pattern in params and params[param_pattern] is not None and eff_dim != 0:
            try:
                params[param_pattern] = scale_parameter_tdb_to_tcb(
                    float(params[param_pattern]), eff_dim
                )
            except (ValueError, TypeError):
                pass
    
    # Convert DMX_* values
    for key in list(params.keys()):
        if key.startswith('DMX_') and params[key] is not None:
            try:
                params[key] = scale_parameter_tdb_to_tcb(float(params[key]), -1)
            except (ValueError, TypeError):
                pass
    
    # Update metadata
    params['UNITS'] = 'TCB'
    params['_par_timescale'] = 'TCB'
    params['_tcb_converted'] = False
    
    log.append("TDB -> TCB conversion complete")
    
    return params, log
