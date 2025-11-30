"""Wrapper to integrate residual computation with fitting.

This module provides functions to create residual computation functions
compatible with the Gauss-Newton fitters.
"""

from pathlib import Path
from typing import Dict, List, Callable, Tuple
import numpy as np
from spiceypy import furnsh, spkgeo
import jax.numpy as jnp

from ..io.par_reader import parse_par_file
from ..io.tim_reader import parse_tim_file
from ..time.clock_correction import clock_correction_seconds
from ..delays.combined import (
    compute_geometric_delay,
    compute_einstein_delay,
    compute_shapiro_delay
)
from ..delays.binary_ell1 import compute_binary_delay_ell1
from ..delays.binary_dd import compute_binary_delay_dd_general
from ..residuals.core import (
    spin_phase,
    dm_delay_sec,
    convert_tcb_to_tdb_spin_params
)


def create_residual_function(
    par_file: Path | str,
    tim_file: Path | str,
    ephemeris_path: Path | str = "data/ephemeris/de440s.bsp",
    clock_dir: Path | str = "data/clock",
    observatory: str = "meerkat",
    use_jax: bool = True
) -> Tuple[Callable, Dict, List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Create a residual computation function for fitting.
    
    This function loads all the necessary data once (TOAs, ephemeris, etc.)
    and returns a closure that can compute residuals with updated parameters.
    
    Parameters
    ----------
    par_file : Path or str
        Path to .par file with timing model parameters
    tim_file : Path or str
        Path to .tim file with TOAs
    ephemeris_path : Path or str, optional
        Path to JPL ephemeris file (default: "data/ephemeris/de440s.bsp")
    clock_dir : Path or str, optional
        Directory containing clock files (default: "data/clock")
    observatory : str, optional
        Observatory name (default: "meerkat")
    use_jax : bool, optional
        Use JAX arrays for computation (default: True)
        
    Returns
    -------
    residuals_fn : Callable
        Function that takes parameter dict and returns residuals in seconds
    initial_params : Dict[str, float]
        Initial parameter values from .par file
    fit_params : List[str]
        List of parameters marked with FIT flags
    toas_mjd : np.ndarray
        TOA times in MJD
    freq_mhz : np.ndarray
        Observing frequencies in MHz
    errors_us : np.ndarray
        TOA uncertainties in microseconds
    """
    par_file = Path(par_file)
    tim_file = Path(tim_file)
    ephemeris_path = Path(ephemeris_path)
    clock_dir = Path(clock_dir)
    
    # Parse files
    params = parse_par_file(par_file)
    toa_data = parse_tim_file(tim_file)
    
    toas_mjd = toa_data['mjd']
    freq_mhz = toa_data['freq_mhz']
    errors_us = toa_data['error_us']
    n_toas = len(toas_mjd)
    
    # Extract FIT flags
    fit_params = []
    for key, value in params.items():
        if key.startswith('FIT_') and value == 1:
            param_name = key[4:]  # Remove 'FIT_' prefix
            if param_name in params:
                fit_params.append(param_name)
    
    # If no FIT flags, default to common parameters
    if not fit_params:
        fit_params = []
        if 'F0' in params:
            fit_params.append('F0')
        if 'F1' in params:
            fit_params.append('F1')
        if 'DM' in params:
            fit_params.append('DM')
    
    # Load ephemeris
    furnsh(str(ephemeris_path))
    
    # Get observatory position (simplified - assumes geocentric)
    obs_xyz_m = np.zeros(3)  # Geocentric for now
    
    # Handle TCB/TDB conversion if needed
    units = params.get('UNITS', 'TDB')
    if units == 'TCB':
        params = convert_tcb_to_tdb_spin_params(params)
    
    # Extract timing model parameters
    f0 = params.get('F0', 0.0)
    f1 = params.get('F1', 0.0)
    f2 = params.get('F2', 0.0)
    f3 = params.get('F3', 0.0)
    pepoch = params.get('PEPOCH', toas_mjd[0])
    
    dm = params.get('DM', 0.0)
    dm1 = params.get('DM1', 0.0)
    dm2 = params.get('DM2', 0.0)
    dmepoch = params.get('DMEPOCH', pepoch)
    
    # Sky position (radians)
    raj_rad = params.get('RAJ', 0.0)
    decj_rad = params.get('DECJ', 0.0)
    
    # Proper motion (rad/s)
    pmra_rad_s = params.get('PMRA', 0.0)
    pmdec_rad_s = params.get('PMDEC', 0.0)
    
    # Parallax (arcsec)
    px_arcsec = params.get('PX', 0.0)
    
    # Binary parameters (if present)
    binary_model = params.get('BINARY', None)
    
    # Convert arrays to JAX if requested
    array_lib = jnp if use_jax else np
    toas_jax = array_lib.array(toas_mjd)
    freq_jax = array_lib.array(freq_mhz)
    
    # Pre-compute clock corrections (only done once)
    clock_corrections_sec = np.zeros(n_toas)
    for i, mjd in enumerate(toas_mjd):
        clock_corrections_sec[i] = clock_correction_seconds(
            mjd, observatory, clock_dir
        )
    
    # TT times (UTC + clock corrections)
    tt_mjd = toas_mjd + clock_corrections_sec / 86400.0
    tt_jax = array_lib.array(tt_mjd)
    
    def compute_residuals(updated_params: Dict[str, float]) -> np.ndarray:
        """Compute residuals with updated parameters.
        
        Parameters
        ----------
        updated_params : Dict[str, float]
            Updated parameter values
            
        Returns
        -------
        residuals_sec : np.ndarray
            Timing residuals in seconds
        """
        # Merge updated parameters with base parameters
        current_params = params.copy()
        current_params.update(updated_params)
        
        # Extract current values
        f0_curr = current_params.get('F0', f0)
        f1_curr = current_params.get('F1', f1)
        f2_curr = current_params.get('F2', f2)
        f3_curr = current_params.get('F3', f3)
        
        dm_curr = current_params.get('DM', dm)
        dm1_curr = current_params.get('DM1', dm1)
        dm2_curr = current_params.get('DM2', dm2)
        
        # Barycentric correction (simplified - using SSB only)
        # In full version, would compute geometric + Einstein + Shapiro delays
        # For now, just use TT times
        tdb_mjd = tt_mjd.copy()
        
        # DM delay
        dm_delay = dm_delay_sec(
            array_lib.array(dm_curr),
            array_lib.array(dm1_curr),
            array_lib.array(dm2_curr),
            toas_jax,
            array_lib.array(dmepoch),
            freq_jax
        )
        
        # Infinite frequency TOA
        t_inf = tdb_mjd - np.array(dm_delay) / 86400.0
        
        # Binary delay (if applicable)
        if binary_model:
            if binary_model in ['ELL1', 'ELL1H']:
                binary_delay_sec = compute_binary_delay_ell1(
                    array_lib.array(t_inf),
                    current_params
                )
            elif binary_model in ['DD', 'DDH', 'DDK', 'DDGR', 'BT']:
                binary_delay_sec = compute_binary_delay_dd_general(
                    array_lib.array(t_inf),
                    current_params
                )
            else:
                binary_delay_sec = array_lib.zeros(n_toas)
            
            # Emission time
            t_emit = t_inf - np.array(binary_delay_sec) / 86400.0
        else:
            t_emit = t_inf
        
        # Spin phase
        phase = spin_phase(
            array_lib.array(t_emit),
            array_lib.array(f0_curr),
            array_lib.array(f1_curr),
            array_lib.array(f2_curr),
            array_lib.array(f3_curr),
            array_lib.array(pepoch)
        )
        
        # Phase residuals (in cycles)
        phase_residual_cycles = phase - array_lib.round(phase)
        
        # Time residuals (in seconds)
        time_residual_sec = phase_residual_cycles / f0_curr
        
        return np.array(time_residual_sec)
    
    return (
        compute_residuals,
        params,
        fit_params,
        toas_mjd,
        freq_mhz,
        errors_us
    )


def fit_pulsar(
    par_file: Path | str,
    tim_file: Path | str,
    output_par: Path | str | None = None,
    ephemeris_path: Path | str = "data/ephemeris/de440s.bsp",
    clock_dir: Path | str = "data/clock",
    observatory: str = "meerkat",
    max_iter: int = 20,
    verbose: bool = True
) -> Dict:
    """Fit pulsar timing parameters.
    
    High-level function that handles the complete fitting workflow:
    loading data, fitting, and optionally writing output.
    
    Parameters
    ----------
    par_file : Path or str
        Input .par file
    tim_file : Path or str
        Input .tim file
    output_par : Path or str, optional
        Output .par file path (if None, don't write)
    ephemeris_path : Path or str, optional
        JPL ephemeris path
    clock_dir : Path or str, optional
        Clock file directory
    observatory : str, optional
        Observatory name
    max_iter : int, optional
        Maximum fitting iterations
    verbose : bool, optional
        Print progress
        
    Returns
    -------
    results : Dict
        Dictionary with fitted parameters, uncertainties, and fit info
    """
    from .gauss_newton_jax import gauss_newton_fit_auto
    from .design_matrix_jax import compute_design_matrix_jax_wrapper
    
    # Create residual function
    residuals_fn, initial_params, fit_params, toas_mjd, freq_mhz, errors_us = \
        create_residual_function(
            par_file, tim_file, ephemeris_path, clock_dir, observatory
        )
    
    if verbose:
        print(f"Fitting {len(fit_params)} parameters: {', '.join(fit_params)}")
        print(f"Dataset: {len(toas_mjd)} TOAs")
    
    # Fit
    fitted_params, uncertainties, info = gauss_newton_fit_auto(
        residuals_fn,
        initial_params,
        fit_params,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # Write output if requested
    if output_par:
        write_par_file(output_par, fitted_params, uncertainties)
    
    return {
        'fitted_params': fitted_params,
        'uncertainties': uncertainties,
        'fit_info': info,
        'initial_params': initial_params,
        'fit_params': fit_params
    }


def write_par_file(
    output_path: Path | str,
    params: Dict[str, float],
    uncertainties: Dict[str, float]
) -> None:
    """Write fitted parameters to .par file.
    
    Parameters
    ----------
    output_path : Path or str
        Output file path
    params : Dict[str, float]
        Parameter values
    uncertainties : Dict[str, float]
        Parameter uncertainties
    """
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("# Fitted timing model\n")
        f.write(f"# Generated by JUG\n\n")
        
        # Write parameters with uncertainties
        for key, value in sorted(params.items()):
            if key.startswith('FIT_'):
                continue  # Skip FIT flags
            
            unc = uncertainties.get(key, 0.0)
            
            # Format based on parameter type
            if key in ['F0', 'F1', 'F2', 'F3']:
                f.write(f"{key:20s} {value:25.16e}")
                if unc > 0:
                    f.write(f"  {unc:12.5e}")
                f.write("\n")
            elif key in ['DM', 'DM1', 'DM2']:
                f.write(f"{key:20s} {value:20.10f}")
                if unc > 0:
                    f.write(f"  {unc:12.5e}")
                f.write("\n")
            else:
                f.write(f"{key:20s} {value}\n")
