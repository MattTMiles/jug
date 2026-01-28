"""
Timing Session - Cached State for Fast Repeated Operations
===========================================================

The TimingSession class caches expensive setup operations:
- Par file parsing
- Tim file parsing  
- Clock file loading
- Ephemeris/TDB computation
- Barycentric delays (dt_sec)

This allows fast repeated operations without re-parsing files.

Performance Impact:
- First compute: ~2.5s (parse + compute)
- Subsequent computes: ~0.1s (just compute, reuse cache)
- Fitting: Much faster with cached delays
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from astropy.time import Time

# Import existing JUG modules
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import (
    fit_parameters_optimized,
    _build_general_fit_setup_from_cache,
    fit_parameters_optimized_cached
)


class TimingSession:
    """
    A cached timing session for fast repeated operations.
    
    This class parses files once and caches expensive computations,
    allowing fast repeated residual computation and parameter fitting.
    
    Attributes
    ----------
    par_file : Path
        Path to .par file
    tim_file : Path
        Path to .tim file
    params : dict
        Parsed timing parameters from .par file
    toas_data : list
        Parsed TOA data from .tim file
    clock_dir : str or None
        Directory containing clock files
    
    Cached State
    ------------
    _cached_result : dict or None
        Last full residuals result (includes dt_sec, tdb_mjd, etc.)
    _initial_params : dict
        Original parameters from .par file (for comparison)
    """
    
    def __init__(
        self,
        par_file: Path | str,
        tim_file: Path | str,
        clock_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize a timing session.
        
        Parameters
        ----------
        par_file : Path or str
            Path to .par file
        tim_file : Path or str
            Path to .tim file
        clock_dir : str, optional
            Directory containing clock files
        verbose : bool, default False
            Print status messages
        """
        self.par_file = Path(par_file)
        self.tim_file = Path(tim_file)
        self.clock_dir = clock_dir
        self.verbose = verbose
        
        # Parse files once and cache
        if self.verbose:
            print(f"Opening session: {self.par_file.name} + {self.tim_file.name}")
        
        self.params = parse_par_file(self.par_file)
        self.toas_data = parse_tim_file_mjds(self.tim_file)
        self._initial_params = dict(self.params)  # Copy for comparison
        
        # Cache for expensive computations
        # Key change: cache separately by subtract_tzr mode for correctness
        self._cached_result_by_mode: Dict[bool, Dict[str, Any]] = {}  # {subtract_tzr: result}
        self._cached_delays: Optional[Dict[str, Any]] = None  # For fast postfit
        self._cached_toa_data: Optional[Dict[str, Any]] = None  # For ultra-fast postfit
        
        if self.verbose:
            print(f"  Loaded {len(self.toas_data)} TOAs")
            print(f"  Session ready")
    
    def compute_residuals(
        self,
        params: Optional[Dict[str, float]] = None,
        subtract_tzr: bool = True,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute timing residuals.
        
        Parameters
        ----------
        params : dict, optional
            Parameter overrides. If None, uses original .par file values.
            If provided, only specified parameters are updated.
        subtract_tzr : bool, default True
            Whether to subtract TZR offset
        force_recompute : bool, default False
            Force recomputation even if cache is valid
        
        Returns
        -------
        result : dict
            Residuals result with keys:
            - 'residuals_us': Residuals in microseconds
            - 'rms_us': RMS in microseconds
            - 'tdb_mjd': TDB times
            - 'dt_sec': Total delay in seconds
            - 'errors_us': TOA uncertainties
            - etc.
        
        Notes
        -----
        This method uses caching:
        - If params unchanged and cache exists, returns cached result
        - If params changed, creates temporary par file and recomputes
        """
        # Check if we can use cache (must match subtract_tzr mode!)
        use_cache = (
            subtract_tzr in self._cached_result_by_mode
            and params is None 
            and not force_recompute
        )
        
        if use_cache:
            if self.verbose:
                print(f"  Using cached residuals (subtract_tzr={subtract_tzr})")
            return self._cached_result_by_mode[subtract_tzr]
        
        # If params provided, use fast evaluation if we have cached data
        if params is not None:
            # FAST PATH: Reuse parsed TOAs and clock data
            # This is 30x faster than creating temp file and reparsing everything
            if self._cached_toa_data is not None:
                from jug.residuals.fast_evaluator import compute_residuals_fast_v2
                
                if self.verbose:
                    print("  Using fast postfit evaluation (cached TOA data)")
                
                # Fast residual compute (reuses TOAs, clocks, corrections)
                try:
                    residuals_us, rms_us = compute_residuals_fast_v2(
                        toa_data=self._cached_toa_data,
                        params={**self.params, **params},  # Merge with new params
                        subtract_mean=True
                    )
                    
                    # Build result using cached metadata
                    result = {
                        'residuals_us': residuals_us,
                        'rms_us': rms_us,
                        'mean_us': 0.0,  # Already subtracted
                        'tdb_mjd': self._cached_toa_data['tdb_mjd'],
                        'dt_sec': self._cached_toa_data['dt_sec'],
                        'errors_us': self._cached_toa_data.get('errors_us'),
                        'n_toas': len(residuals_us),
                    }
                    
                    return result
                except Exception as e:
                    if self.verbose:
                        print(f"  Fast path failed: {e}, falling back to slow path")
            
            # SLOW PATH: Create temporary par file (no cached data yet)
            import tempfile
            from pathlib import Path
            
            if self.verbose:
                print("  Using temp par file (no cached data yet)")
            
            # Read original par file
            with open(self.par_file, 'r') as f:
                par_lines = f.readlines()
            
            # Update parameters
            updated_lines = []
            updated_params = set()
            
            for line in par_lines:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith('#'):
                    updated_lines.append(line)
                    continue
                
                parts = line_stripped.split()
                if parts:
                    param_name = parts[0]
                    if param_name in params:
                        # Update this parameter
                        new_value = params[param_name]
                        # Format appropriately
                        if param_name == 'F0':
                            new_line = f"{param_name:<12} {new_value:.15f}"
                        elif param_name.startswith('F') and param_name[1:].isdigit():
                            new_line = f"{param_name:<12} {new_value:.15e}"
                        elif param_name.startswith('DM'):
                            new_line = f"{param_name:<12} {new_value:.15f}"
                        else:
                            new_line = f"{param_name:<12} {new_value:.15e}"
                        
                        # Preserve flags if present
                        if len(parts) > 2:
                            flags = ' '.join(parts[2:])
                            new_line += f" {flags}"
                        elif len(parts) > 1 and parts[1] != str(new_value):
                            # Has a flag
                            new_line += f" {parts[1]}"
                        
                        updated_lines.append(new_line + '\n')
                        updated_params.add(param_name)
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            
            # Add any new parameters not in original file
            for param_name, value in params.items():
                if param_name not in updated_params:
                    if param_name == 'F0':
                        new_line = f"{param_name:<12} {value:.15f} 1\n"
                    elif param_name.startswith('F') and param_name[1:].isdigit():
                        new_line = f"{param_name:<12} {value:.15e} 1\n"
                    elif param_name.startswith('DM'):
                        new_line = f"{param_name:<12} {value:.15f} 1\n"
                    else:
                        new_line = f"{param_name:<12} {value:.15e} 1\n"
                    updated_lines.append(new_line)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
                tmp.writelines(updated_lines)
                tmp_par_path = Path(tmp.name)
            
            try:
                # Compute with updated par file
                result = compute_residuals_simple(
                    par_file=tmp_par_path,
                    tim_file=self.tim_file,
                    clock_dir=self.clock_dir,
                    subtract_tzr=subtract_tzr,
                    verbose=False
                )
            finally:
                # Clean up temp file
                tmp_par_path.unlink()
            
            return result
        
        # No params override - use original par file
        if self.verbose:
            print("  Computing residuals...")
        
        result = compute_residuals_simple(
            par_file=self.par_file,
            tim_file=self.tim_file,
            clock_dir=self.clock_dir,
            subtract_tzr=subtract_tzr,
            verbose=False
        )
        
        # Cache the result (only for original params), keyed by subtract_tzr
        self._cached_result_by_mode[subtract_tzr] = result
        
        # Cache TOA data for fast postfit (enables 30x speedup!)
        if 'dt_sec' in result and 'tdb_mjd' in result:
            self._cached_toa_data = {
                'dt_sec': result['dt_sec'],
                'tdb_mjd': result['tdb_mjd'],
                'errors_us': result.get('errors_us'),
                'freq_mhz': result.get('freq_bary_mhz'),  # Need for DM delay recalculation
                'original_dm_params': {  # Cache original DM params for fast DM fitting
                    'DM': self.params.get('DM', 0.0),
                    'DM1': self.params.get('DM1', 0.0),
                    'DM2': self.params.get('DM2', 0.0),
                }
            }
            
            if self.verbose:
                print("  Cached TOA data for fast postfit evaluation")
        
        return result
    
    def fit_parameters(
        self,
        fit_params: List[str],
        max_iter: int = 25,
        convergence_threshold: float = 1e-14,
        device: Optional[str] = None,
        verbose: Optional[bool] = None,
        toa_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit timing model parameters.
        
        Parameters
        ----------
        fit_params : list of str
            Parameters to fit (e.g., ['F0', 'F1', 'DM'])
        max_iter : int, default 25
            Maximum iterations
        convergence_threshold : float, default 1e-14
            Convergence threshold
        device : str, optional
            'cpu', 'gpu', or None (auto-detect)
        verbose : bool, optional
            Print fitting progress (uses session verbose if not specified)
        toa_mask : ndarray of bool, optional
            Boolean mask indicating which TOAs to include in fit (True = include).
            If None, all TOAs are used. This allows fitting on a subset of data.
        
        Returns
        -------
        result : dict
            Fit results with keys:
            - 'final_params': Fitted parameter values
            - 'uncertainties': Parameter uncertainties
            - 'final_rms': Final RMS in μs
            - 'iterations': Number of iterations
            - 'converged': Whether fit converged
            - etc.
        
        Notes
        -----
        This method uses cached arrays when available for maximum performance.
        On first fit, it computes residuals with subtract_tzr=False to populate cache.
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            n_toas = len(self.toas_data)
            if toa_mask is not None:
                n_used = np.sum(toa_mask)
                print(f"  Fitting {len(fit_params)} parameters: {', '.join(fit_params)} ({n_used}/{n_toas} TOAs)")
            else:
                print(f"  Fitting {len(fit_params)} parameters: {', '.join(fit_params)}")
        
        # FAST PATH: Use cached arrays if available
        # Ensure we have cached residuals with subtract_tzr=False (needed for fitting)
        if False not in self._cached_result_by_mode:
            if verbose:
                print("  Computing residuals (subtract_tzr=False) for fitting cache...")
            # Populate cache with subtract_tzr=False
            self.compute_residuals(subtract_tzr=False, force_recompute=False)
        
        # Check if we have all required cached data
        cached_result = self._cached_result_by_mode.get(False)
        has_required_cache = (
            cached_result is not None
            and 'dt_sec' in cached_result
            and 'tdb_mjd' in cached_result
            and 'freq_bary_mhz' in cached_result
        )
        
        if has_required_cache:
            # CACHED PATH: Build setup from cached arrays (fast!)
            if verbose:
                print("  Using cached arrays for fitting (fast path)")
            
            # Prepare cached data for setup builder
            toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in self.toas_data])
            errors_us = np.array([toa.error_us for toa in self.toas_data])
            
            session_cached_data = {
                'dt_sec': cached_result['dt_sec'],
                'tdb_mjd': cached_result['tdb_mjd'],
                'freq_bary_mhz': cached_result['freq_bary_mhz'],
                'toas_mjd': toas_mjd,
                'errors_us': errors_us
            }
            
            # Build setup from cache (with optional TOA mask)
            setup = _build_general_fit_setup_from_cache(
                session_cached_data,
                self.params,
                fit_params,
                toa_mask=toa_mask
            )
            
            # Run cached fit
            result = fit_parameters_optimized_cached(
                setup,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                verbose=verbose
            )
        else:
            # FALLBACK PATH: Use file-based fitting (slower but always works)
            if verbose:
                print("  Cache incomplete, using file-based fitting")
            
            result = fit_parameters_optimized(
                par_file=self.par_file,
                tim_file=self.tim_file,
                fit_params=fit_params,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                clock_dir=self.clock_dir,
                device=device,
                verbose=verbose
            )
        
        # Invalidate residuals cache since parameters changed
        self._cached_result_by_mode.clear()
        
        return result
    
    def get_initial_params(self) -> Dict[str, float]:
        """
        Get the original parameters from the .par file.
        
        Returns
        -------
        params : dict
            Copy of initial parameters
        """
        return dict(self._initial_params)
    
    def get_toa_count(self) -> int:
        """Get number of TOAs in the session."""
        return len(self.toas_data)
    
    def __repr__(self) -> str:
        return (
            f"TimingSession("
            f"par='{self.par_file.name}', "
            f"tim='{self.tim_file.name}', "
            f"ntoas={len(self.toas_data)})"
        )
