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
        
        # Convert TCB->TDB at ingest time so all downstream code sees TDB params
        from jug.io.par_reader import validate_par_timescale
        validate_par_timescale(self.params, context="TimingSession", verbose=verbose)
        
        # Tempo2's T2 model uses IAU convention for KIN/KOM.
        # JUG's DDK code (from PINT) uses DT92 convention.
        # Convert: KIN_DT92 = 180 - KIN_IAU, KOM_DT92 = 90 - KOM_IAU
        binary = self.params.get('BINARY', '').upper()
        if binary == 'T2' and ('KIN' in self.params or 'KOM' in self.params):
            if 'KIN' in self.params:
                self.params['KIN'] = 180.0 - float(self.params['KIN'])
            if 'KOM' in self.params:
                self.params['KOM'] = 90.0 - float(self.params['KOM'])
        
        self.toas_data = parse_tim_file_mjds(self.tim_file)
        self._initial_params = dict(self.params)  # Copy for comparison
        
        # Cache for expensive computations
        # Key change: cache separately by subtract_tzr mode for correctness
        self._cached_result_by_mode: Dict[bool, Dict[str, Any]] = {}  # {subtract_tzr: result}
        self._cached_delays: Optional[Dict[str, Any]] = None  # For fast postfit
        self._cached_toa_data: Optional[Dict[str, Any]] = None  # For ultra-fast postfit
        # Clock / EOP issues collected during first residual computation (for GUI)
        self.clock_issues: list = []
        
        if self.verbose:
            print(f"  Loaded {len(self.toas_data)} TOAs")
            print(f"  Session ready")

    def _compute_residuals_with_params(
        self,
        params: Dict[str, Any],
        subtract_tzr: bool
    ) -> Dict[str, Any]:
        """
        Compute residuals using specified parameters (creates temp par file).

        This is an internal helper for computing residuals with arbitrary params.
        Used when self.params differs from the original par file.
        """
        import tempfile

        def format_param_value(param_name: str, value: Any) -> str:
            """Format parameter value for par file."""
            # String values (like BINARY, EPHEM, etc.) - keep as-is
            if isinstance(value, str):
                return f"{param_name:<12} {value}"
            # RAJ/DECJ: convert radians back to sexagesimal for par file
            if param_name == 'RAJ' and isinstance(value, (int, float)):
                from jug.io.par_reader import format_ra
                return f"{param_name:<12} {format_ra(float(value))}"
            if param_name == 'DECJ' and isinstance(value, (int, float)):
                from jug.io.par_reader import format_dec
                return f"{param_name:<12} {format_dec(float(value))}"
            # Numeric values - format appropriately
            if param_name == 'F0':
                return f"{param_name:<12} {value:.15f}"
            elif param_name.startswith('F') and param_name[1:].isdigit():
                return f"{param_name:<12} {value:.15e}"
            elif param_name.startswith('DM'):
                return f"{param_name:<12} {value:.15f}"
            elif isinstance(value, float):
                return f"{param_name:<12} {value:.15e}"
            else:
                return f"{param_name:<12} {value}"

        # For ecliptic par files, convert fitted RAJ/DECJ back to LAMBDA/BETA
        # so the temp par file has consistent ecliptic coordinates
        if params.get('_ecliptic_coords'):
            from jug.io.par_reader import parse_ra, parse_dec, convert_equatorial_to_ecliptic
            ra_val = params.get('RAJ')
            dec_val = params.get('DECJ')
            if ra_val is not None and dec_val is not None:
                ra_rad = parse_ra(ra_val) if isinstance(ra_val, str) else float(ra_val)
                dec_rad = parse_dec(dec_val) if isinstance(dec_val, str) else float(dec_val)
                ecl_frame = params.get('_ecliptic_frame', 'IERS2010')
                ecl = convert_equatorial_to_ecliptic(
                    ra_rad, dec_rad,
                    pmra=params.get('PMRA', 0.0),
                    pmdec=params.get('PMDEC', 0.0),
                    ecl_frame=ecl_frame,
                )
                params['LAMBDA'] = ecl['LAMBDA']
                params['BETA'] = ecl['BETA']
                if 'PMLAMBDA' in params:
                    params['PMLAMBDA'] = ecl['PMLAMBDA']
                if 'PMBETA' in params:
                    params['PMBETA'] = ecl['PMBETA']

        # Build temp par file with updated params
        # Convert KIN/KOM from DT92 back to IAU convention for par file
        # (compute_residuals_simple will apply IAU->DT92 again when reading)
        binary = params.get('BINARY', '').upper()
        if binary == 'T2' and ('KIN' in params or 'KOM' in params):
            params = dict(params)  # avoid mutating caller's dict
            if 'KIN' in params:
                params['KIN'] = 180.0 - float(params['KIN'])
            if 'KOM' in params:
                params['KOM'] = 90.0 - float(params['KOM'])

        with open(self.par_file, 'r') as f:
            original_lines = f.readlines()

        updated_lines = []
        updated_params = set()

        for line in original_lines:
            line_stripped = line.strip()
            if line_stripped.startswith('#') or not line_stripped:
                updated_lines.append(line)
                continue

            parts = line_stripped.split()
            if parts:
                param_name = parts[0]
                if param_name in params:
                    new_value = params[param_name]
                    new_line = format_param_value(param_name, new_value)

                    # Preserve flags if present (for numeric params with fit flags)
                    if len(parts) > 2 and not isinstance(new_value, str):
                        flags = ' '.join(parts[2:])
                        new_line += f" {flags}"

                    updated_lines.append(new_line + '\n')
                    updated_params.add(param_name)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Add any new parameters not in original file (skip internal keys)
        for param_name, value in params.items():
            if param_name not in updated_params and not param_name.startswith('_'):
                if isinstance(value, (int, float, str)):
                    new_line = format_param_value(param_name, value) + '\n'
                    updated_lines.append(new_line)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
            tmp.writelines(updated_lines)
            tmp_par_path = Path(tmp.name)

        try:
            result = compute_residuals_simple(
                par_file=tmp_par_path,
                tim_file=self.tim_file,
                clock_dir=self.clock_dir,
                subtract_tzr=subtract_tzr,
                verbose=False
            )
        finally:
            tmp_par_path.unlink()

        return result

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
        
        # If params provided, recompute residuals with updated parameters
        if params is not None:
            # Create temporary par file with updated parameters
            import tempfile
            from pathlib import Path
            
            if self.verbose:
                print("  Using temp par file (no cached data yet)")
            
            # Read original par file
            with open(self.par_file, 'r') as f:
                par_lines = f.readlines()
            
            # For ecliptic pulsars, convert fitted RAJ/DECJ back to LAMBDA/BETA
            if params.get('_ecliptic_coords'):
                from jug.io.par_reader import (
                    parse_ra, parse_dec, convert_equatorial_to_ecliptic
                )
                ra_val = params.get('RAJ')
                dec_val = params.get('DECJ')
                if ra_val is not None and dec_val is not None:
                    ra_rad = parse_ra(ra_val) if isinstance(ra_val, str) else float(ra_val)
                    dec_rad = parse_dec(dec_val) if isinstance(dec_val, str) else float(dec_val)
                    ecl_frame = params.get('_ecliptic_frame', 'IERS2010')
                    ecl = convert_equatorial_to_ecliptic(
                        ra_rad, dec_rad,
                        pmra=params.get('PMRA', 0.0),
                        pmdec=params.get('PMDEC', 0.0),
                        ecl_frame=ecl_frame,
                    )
                    params['LAMBDA'] = ecl['LAMBDA']
                    params['BETA'] = ecl['BETA']
                    if 'PMLAMBDA' in params:
                        params['PMLAMBDA'] = ecl['PMLAMBDA']
                    if 'PMBETA' in params:
                        params['PMBETA'] = ecl['PMBETA']

            # Convert KIN/KOM from DT92 back to IAU convention for par file
            # (compute_residuals_simple will apply IAU->DT92 again when reading)
            binary = params.get('BINARY', '').upper()
            if binary == 'T2' and ('KIN' in params or 'KOM' in params):
                params = dict(params)  # avoid mutating caller's dict
                if 'KIN' in params:
                    params['KIN'] = 180.0 - float(params['KIN'])
                if 'KOM' in params:
                    params['KOM'] = 90.0 - float(params['KOM'])

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
                        # Format appropriately based on parameter type
                        if isinstance(new_value, str):
                            # Already a string (sexagesimal or other)
                            new_line = f"{param_name:<12} {new_value}"
                        elif param_name == 'RAJ':
                            # RAJ in radians - convert to sexagesimal
                            from jug.model.codecs import RAJCodec
                            new_line = f"{param_name:<12} {RAJCodec().encode(new_value)}"
                        elif param_name == 'DECJ':
                            # DECJ in radians - convert to sexagesimal
                            from jug.model.codecs import DECJCodec
                            new_line = f"{param_name:<12} {DECJCodec().encode(new_value)}"
                        elif param_name == 'F0':
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
                    if isinstance(value, str):
                        new_line = f"{param_name:<12} {value} 1\n"
                    elif param_name == 'RAJ':
                        from jug.model.codecs import RAJCodec
                        new_line = f"{param_name:<12} {RAJCodec().encode(value)} 1\n"
                    elif param_name == 'DECJ':
                        from jug.model.codecs import DECJCodec
                        new_line = f"{param_name:<12} {DECJCodec().encode(value)} 1\n"
                    elif param_name == 'F0':
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
        
        # No params override - use current self.params (not original par file!)
        # This is CRITICAL: after fitting, self.params has fitted values.
        # If we use the original par file, the cache would have stale DM delays
        # causing incorrect gradients in subsequent fits.
        if self.verbose:
            print("  Computing residuals...")

        # Check if self.params differs from initial - if so, use self.params
        params_changed = (self.params != self._initial_params)

        if params_changed:
            # Use current self.params (creates temp file internally)
            result = self._compute_residuals_with_params(self.params, subtract_tzr)
        else:
            # Use original par file (faster, no temp file needed)
            result = compute_residuals_simple(
                par_file=self.par_file,
                tim_file=self.tim_file,
                clock_dir=self.clock_dir,
                subtract_tzr=subtract_tzr,
                verbose=False
            )
        
        # Cache the result (only for original params), keyed by subtract_tzr
        self._cached_result_by_mode[subtract_tzr] = result
        # Capture clock issues from first load for the GUI
        if not self.clock_issues and result.get('clock_issues'):
            self.clock_issues = result['clock_issues']
        
        # Cache TOA data for fast postfit (enables 30x speedup!)
        if 'dt_sec' in result and 'tdb_mjd' in result:
            self._cached_toa_data = {
                'dt_sec': result['dt_sec'],
                'tdb_mjd': result['tdb_mjd'],
                'errors_us': result.get('errors_us'),
                'freq_mhz': result.get('freq_bary_mhz'),  # Need for DM delay recalculation
                'roemer_shapiro_sec': result.get('roemer_shapiro_sec'),  # Need for binary fitting
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
        fit_params: Optional[List[str]] = None,
        max_iter: int = 100,
        convergence_threshold: float = 1e-14,
        device: Optional[str] = None,
        verbose: Optional[bool] = None,
        toa_mask: Optional[np.ndarray] = None,
        solver_mode: str = "exact",
        noise_config: Optional[object] = None,
        subtract_noise_sec: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit timing model parameters.

        Parameters
        ----------
        fit_params : list of str, optional
            Additional parameters to fit on top of those already flagged
            in the par file (free_params). For example, passing ['F2']
            will fit all par-file free params plus F2.
            If None, fits only the par-file free params.
        max_iter : int, default 100
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
        solver_mode : str, default "exact"
            Solver mode: "exact" (SVD, bit-for-bit reproducible) or
            "fast" (QR/lstsq, faster but may differ slightly).
        noise_config : NoiseConfig, optional
            Override noise configuration. If None, auto-detected from par file.
        subtract_noise_sec : ndarray of float, optional
            Per-TOA noise realization (in seconds) to subtract from dt_sec
            before fitting. This implements Tempo2-style noise subtraction:
            after subtracting a noise realization from the displayed residuals,
            the fitter should work on the cleaned data. If None, no noise
            subtraction is applied.

        Returns
        -------
        result : dict
            Fit results with keys:
            - 'final_params': Fitted parameter values
            - 'uncertainties': Parameter uncertainties
            - 'final_rms': Final RMS in mus
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

        # Normalize solver_mode
        solver_mode = solver_mode.lower().strip() if solver_mode else "exact"
        if solver_mode not in ("exact", "fast"):
            solver_mode = "exact"

        # Build final fit param list: par-file free params + any extras
        base_params = self.free_params or []
        if fit_params is not None:
            # Merge extras, avoiding duplicates, preserving order
            extra = [p for p in fit_params if p not in base_params]
            fit_params = base_params + extra
        else:
            fit_params = base_params

        if not fit_params:
            raise ValueError(
                "No parameters to fit. Set fit flags in the par file "
                "or pass fit_params=['F0', 'F1', ...]."
            )

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
            toa_flags = [toa.flags for toa in self.toas_data]
            
            session_cached_data = {
                'dt_sec': cached_result['dt_sec'],
                'dt_sec_ld': cached_result.get('dt_sec_ld'),
                'tdb_mjd': cached_result['tdb_mjd'],
                'freq_bary_mhz': cached_result['freq_bary_mhz'],
                'toas_mjd': toas_mjd,
                'errors_us': errors_us,
                'toa_flags': toa_flags,
                'roemer_shapiro_sec': cached_result.get('roemer_shapiro_sec'),
                'prebinary_delay_sec': cached_result.get('prebinary_delay_sec'),
                'ssb_obs_pos_ls': cached_result.get('ssb_obs_pos_ls'),
                'sw_geometry_pc': cached_result.get('sw_geometry_pc'),
                'jump_phase': cached_result.get('jump_phase'),
                'tzr_phase': cached_result.get('tzr_phase'),
            }
            
            # Build setup from cache (with optional TOA mask)
            setup = _build_general_fit_setup_from_cache(
                session_cached_data,
                self.params,
                fit_params,
                toa_mask=toa_mask,
                noise_config=noise_config,
                subtract_noise_sec=subtract_noise_sec
            )
            
            # Run cached fit
            result = fit_parameters_optimized_cached(
                setup,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                verbose=verbose,
                solver_mode=solver_mode
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
        
        # Update session params with fitted values (CRITICAL for iterative fitting!)
        # Without this, subsequent fits would use old params and diverge.
        if result.get('success', True) and 'final_params' in result:
            from jug.io.par_reader import format_ra, format_dec
            updated_params = result['final_params'].copy()
            # Convert RAJ/DECJ from radians back to string format for consistency
            if 'RAJ' in updated_params:
                updated_params['RAJ'] = format_ra(updated_params['RAJ'])
            if 'DECJ' in updated_params:
                updated_params['DECJ'] = format_dec(updated_params['DECJ'])
            # For ecliptic pulsars, also update LAMBDA/BETA from fitted RAJ/DECJ
            if self.params.get('_ecliptic_coords') and 'RAJ' in updated_params:
                from jug.io.par_reader import (
                    parse_ra, parse_dec, convert_equatorial_to_ecliptic
                )
                ra_rad = parse_ra(updated_params['RAJ'])
                dec_rad = parse_dec(updated_params['DECJ'])
                ecl_frame = self.params.get('_ecliptic_frame', 'IERS2010')
                ecl = convert_equatorial_to_ecliptic(
                    ra_rad, dec_rad,
                    pmra=updated_params.get('PMRA', self.params.get('PMRA', 0.0)),
                    pmdec=updated_params.get('PMDEC', self.params.get('PMDEC', 0.0)),
                    ecl_frame=ecl_frame,
                )
                updated_params['LAMBDA'] = ecl['LAMBDA']
                updated_params['BETA'] = ecl['BETA']
                if 'PMLAMBDA' in self.params:
                    updated_params['PMLAMBDA'] = ecl['PMLAMBDA']
                if 'PMBETA' in self.params:
                    updated_params['PMBETA'] = ecl['PMBETA']
            self.params.update(updated_params)

        # Invalidate residuals cache since parameters changed
        self._cached_result_by_mode.clear()
        # Also invalidate cached TOA data -- the fast evaluator only handles
        # spin/DM changes, so stale dt_sec causes wrong postfit residuals
        # when binary, astrometric, FD, or SW parameters were fitted.
        self._cached_toa_data = None

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

    @property
    def free_params(self) -> List[str]:
        """Parameters with fit flag = 1 in the par file.

        These are the default parameters to fit, matching the par file
        convention used by PINT and Tempo2.
        """
        import re
        all_flags = sorted(self.params.get('_fit_flags', {}).keys())
        # Filter out parametric families handled automatically by the fitter
        # (DMX_nnnn, JUMP1..N are auto-detected from the par file)
        return [p for p in all_flags
                if not re.match(r'^DMX_\d+$', p) and not re.match(r'^JUMP\d+$', p)]

    def set_free(self, *param_names):
        """Turn on the fit flag for one or more parameters.

        Accepts individual names or a list:
            session.set_free('F0', 'F1')
            session.set_free(['F0', 'F1'])
        """
        flags = self.params.setdefault('_fit_flags', {})
        # If a single list/tuple was passed, unpack it
        if len(param_names) == 1 and isinstance(param_names[0], (list, tuple)):
            param_names = param_names[0]
        for name in param_names:
            flags[name] = True

    def set_frozen(self, *param_names):
        """Turn off the fit flag for one or more parameters.

        Accepts individual names or a list:
            session.set_frozen('F0', 'F1')
            session.set_frozen(['F0', 'F1'])
        """
        flags = self.params.get('_fit_flags', {})
        # If a single list/tuple was passed, unpack it
        if len(param_names) == 1 and isinstance(param_names[0], (list, tuple)):
            param_names = param_names[0]
        for name in param_names:
            flags.pop(name, None)

    def print_model(self, include_dmx: bool = False):
        """Print the timing model: all parameters, values, and fit status.

        Groups parameters by category (metadata, position, spin, DM,
        binary, noise, etc.) similar to PINT's model display.

        Parameters
        ----------
        include_dmx : bool, default False
            If True, print individual DMX bins. Otherwise show a
            summary count.
        """
        p = self.params
        fit_flags = p.get('_fit_flags', {})

        def _row(key, val, show_fit=True):
            flag_str = ''
            if show_fit:
                if key in fit_flags:
                    flag_str = '  [fit]'
                else:
                    flag_str = '  [frozen]'
            if isinstance(val, float):
                hp = p.get('_high_precision', {})
                if key in hp:
                    val_str = hp[key]
                else:
                    val_str = repr(val)
            else:
                val_str = str(val)
            return f"  {key:<20s} {val_str}{flag_str}"

        lines = [f"=== Timing Model: {p.get('PSR', 'Unknown')} ===", ""]

        # Metadata
        lines.append("-- Metadata --")
        for k in ['PSR', 'EPHEM', 'CLOCK', 'UNITS', 'TIMEEPH', 'T2CMETHOD',
                   'DILATEFREQ', 'DMDATA', 'NTOA', 'START', 'FINISH']:
            if k in p:
                lines.append(_row(k, p[k], show_fit=False))

        # Position
        lines.append("")
        lines.append("-- Position --")
        if p.get('_ecliptic_coords'):
            lines.append(_row('ECL', p.get('ECL', p.get('_ecliptic_frame', '')), show_fit=False))
            for k in ['ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'PX']:
                if k in p:
                    lines.append(_row(k, p[k]))
        else:
            for k in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
                if k in p:
                    lines.append(_row(k, p[k]))
        if 'POSEPOCH' in p:
            lines.append(_row('POSEPOCH', p['POSEPOCH'], show_fit=False))

        # Spin
        lines.append("")
        lines.append("-- Spin --")
        for k in ['F0', 'F1', 'F2', 'F3']:
            if k in p:
                lines.append(_row(k, p[k]))
        if 'PEPOCH' in p:
            lines.append(_row('PEPOCH', p['PEPOCH'], show_fit=False))

        # DM
        lines.append("")
        lines.append("-- Dispersion --")
        for k in ['DM', 'DM1', 'DM2', 'DM3']:
            if k in p:
                lines.append(_row(k, p[k]))
        if 'DMEPOCH' in p:
            lines.append(_row('DMEPOCH', p['DMEPOCH'], show_fit=False))

        # DMX summary or detail
        dmx_keys = sorted([k for k in p if k.startswith('DMX_') and k[4:].isdigit()])
        n_dmx = len(dmx_keys)
        if n_dmx > 0:
            n_dmx_fit = sum(1 for k in dmx_keys if k in fit_flags)
            if include_dmx:
                lines.append(f"  -- DMX ({n_dmx} bins, {n_dmx_fit} fit) --")
                for k in dmx_keys:
                    lines.append(_row(k, p[k]))
            else:
                lines.append(f"  DMX                {n_dmx} bins ({n_dmx_fit} fit)  [use print_model(include_dmx=True) to show]")

        # Binary
        if 'BINARY' in p:
            lines.append("")
            lines.append(f"-- Binary ({p['BINARY']}) --")
            binary_keys = ['PB', 'A1', 'ECC', 'E', 'T0', 'OM', 'OMDOT', 'PBDOT',
                           'GAMMA', 'M2', 'SINI', 'KIN', 'KOM',
                           'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT', 'TASC',
                           'FB0', 'FB1', 'A1DOT', 'XDOT', 'EDOT',
                           'DR', 'DTH', 'A0', 'B0', 'SHAPMAX',
                           'H3', 'H4', 'STIGMA', 'K96']
            for k in binary_keys:
                if k in p:
                    lines.append(_row(k, p[k]))

        # FD
        fd_keys = sorted([k for k in p if k.startswith('FD') and k[2:].isdigit()],
                         key=lambda k: int(k[2:]))
        if fd_keys:
            lines.append("")
            lines.append("-- Frequency-dependent delays --")
            for k in fd_keys:
                lines.append(_row(k, p[k]))

        # JUMPs
        jump_lines = p.get('_jump_lines', [])
        if jump_lines:
            lines.append("")
            lines.append(f"-- JUMPs ({len(jump_lines)}) --")
            for idx, jl in enumerate(jump_lines):
                jkey = f'JUMP{idx + 1}'
                parts = jl.strip().split()
                if parts[1].upper() == 'MJD':
                    label = f"MJD {parts[2]}-{parts[3]}"
                else:
                    label = f"{parts[1]} {parts[2]}"
                flag_str = '  [fit]' if jkey in fit_flags else '  [frozen]'
                lines.append(f"  {jkey:<20s} {p.get(jkey, 0.0):<22}  ({label}){flag_str}")

        # Noise
        noise_lines = p.get('_noise_lines', [])
        if noise_lines:
            lines.append("")
            lines.append(f"-- White Noise ({len(noise_lines)} entries) --")
            for nl in noise_lines:
                parts = nl.strip().split()
                kind = parts[0]
                flag_val = parts[2] if len(parts) >= 3 else ''
                value = parts[3] if len(parts) >= 4 else ''
                lines.append(f"  {kind:<12s} {flag_val:<20s} {value}")

        # Red noise
        if 'RNAMP' in p or any(k.upper() in ('TNREDAMP',) for k in p):
            lines.append("")
            lines.append("-- Red Noise --")
            if 'RNAMP' in p:
                import math
                _SEC_PER_YR = 365.25 * 86400.0
                log10_A = math.log10(2.0 * math.pi * math.sqrt(3.0) / (_SEC_PER_YR * 1e6) * float(p['RNAMP']))
                gamma = -float(p.get('RNIDX', 0))
                lines.append(f"  {'TNRedAmp':<20s} {log10_A}")
                lines.append(f"  {'TNRedGam':<20s} {gamma}")

        # Misc
        misc_keys = ['NE_SW', 'SWM', 'PLANET_SHAPIRO', 'CORRECT_TROPOSPHERE']
        misc_present = [k for k in misc_keys if k in p]
        if misc_present:
            lines.append("")
            lines.append("-- Miscellaneous --")
            for k in misc_present:
                lines.append(_row(k, p[k], show_fit=False))

        # TZR
        lines.append("")
        lines.append("-- Reference TOA --")
        for k in ['TZRMJD', 'TZRSITE', 'TZRFRQ']:
            if k in p:
                lines.append(_row(k, p[k], show_fit=False))

        print('\n'.join(lines))

    def print_toas(self, n: int = 10):
        """Print a summary of the TOA dataset.

        Parameters
        ----------
        n : int, default 10
            Number of sample TOAs to display at the start and end.
        """
        toas = self.toas_data
        n_toas = len(toas)
        errors = np.array([t.error_us for t in toas])
        freqs = np.array([t.freq_mhz for t in toas])
        mjds = np.array([t.mjd_int + t.mjd_frac for t in toas])

        # Backend counts
        from jug.engine.flag_mapping import resolve_backends
        backends = resolve_backends([t.flags for t in toas])
        from collections import Counter
        backend_counts = Counter(backends)

        # Observatory counts
        obs_counts = Counter(t.observatory for t in toas)

        lines = [
            f"=== TOA Summary: {self.params.get('PSR', 'Unknown')} ===",
            f"  Total TOAs:     {n_toas}",
            f"  MJD range:      {mjds.min():.3f} - {mjds.max():.3f}",
            f"  Timespan:       {(mjds.max() - mjds.min()) / 365.25:.2f} years",
            f"  Freq range:     {freqs.min():.1f} - {freqs.max():.1f} MHz",
            f"  Error range:    {errors.min():.4f} - {errors.max():.4f} us",
            f"  Median error:   {np.median(errors):.4f} us",
            "",
            "  Backends:",
        ]
        for backend, count in sorted(backend_counts.items()):
            lines.append(f"    {backend:<25s} {count:>6d} TOAs")

        lines.append("")
        lines.append("  Observatories:")
        for obs, count in sorted(obs_counts.items()):
            lines.append(f"    {obs:<25s} {count:>6d} TOAs")

        # Sample TOAs
        lines.append("")
        show_n = min(n, n_toas)
        lines.append(f"  First {show_n} TOAs:")
        lines.append(f"    {'MJD':>18s}  {'Freq (MHz)':>10s}  {'Error (us)':>10s}  {'Backend':>15s}  {'Obs':>6s}")
        for t in toas[:show_n]:
            b = resolve_backends([t.flags])[0]
            lines.append(f"    {t.mjd_int + t.mjd_frac:>18.10f}  {t.freq_mhz:>10.1f}  {t.error_us:>10.3f}  {b:>15s}  {t.observatory:>6s}")

        if n_toas > 2 * show_n:
            lines.append(f"    ... ({n_toas - 2*show_n} more) ...")
            lines.append(f"  Last {show_n} TOAs:")
            for t in toas[-show_n:]:
                b = resolve_backends([t.flags])[0]
                lines.append(f"    {t.mjd_int + t.mjd_frac:>18.10f}  {t.freq_mhz:>10.1f}  {t.error_us:>10.3f}  {b:>15s}  {t.observatory:>6s}")

        print('\n'.join(lines))
    
    def summary(self) -> str:
        """Print a summary of the current session state.

        Shows pulsar name, TOA count, parameter count, binary model,
        noise configuration, and fit status.
        """
        p = self.params
        lines = []
        lines.append(f"=== {p.get('PSR', 'Unknown')} ===")
        lines.append(f"  TOAs:       {len(self.toas_data)}")
        lines.append(f"  Par file:   {self.par_file.name}")
        lines.append(f"  Tim file:   {self.tim_file.name}")

        # Coordinate system
        if p.get('_ecliptic_coords'):
            lines.append(f"  Coords:     Ecliptic ({p.get('_ecliptic_frame', 'IERS2010')})")
        else:
            lines.append(f"  Coords:     Equatorial (RAJ/DECJ)")

        # Binary
        binary = p.get('BINARY')
        if binary:
            lines.append(f"  Binary:     {binary}")

        # DMX
        n_dmx = len([k for k in p if k.startswith('DMX_') and k[4:].isdigit()])
        if n_dmx:
            lines.append(f"  DMX bins:   {n_dmx}")

        # Noise
        noise_lines = p.get('_noise_lines', [])
        if noise_lines:
            from collections import Counter
            kinds = Counter(l.split()[0].upper() for l in noise_lines)
            noise_str = ', '.join(f"{v} {k}" for k, v in sorted(kinds.items()))
            lines.append(f"  Noise:      {noise_str}")
        if 'RNAMP' in p or any(k.upper() in ('TNREDAMP',) for k in p):
            lines.append(f"  Red noise:  Yes")

        # JUMPs
        n_jumps = len(p.get('_jump_lines', []))
        if n_jumps:
            lines.append(f"  JUMPs:      {n_jumps}")

        # Ephem / clock
        lines.append(f"  Ephemeris:  {p.get('EPHEM', 'N/A')}")
        lines.append(f"  Clock:      {p.get('CLOCK', p.get('CLK', 'N/A'))}")
        lines.append(f"  Timescale:  {p.get('_par_timescale', 'TDB')}")

        text = '\n'.join(lines)
        print(text)

    def parameter_table(self, fit_result: Optional[Dict[str, Any]] = None) -> None:
        """Print a table comparing initial and current parameter values.

        Parameters
        ----------
        fit_result : dict, optional
            Fit result dict (with 'final_params' and 'uncertainties').
            If provided, shows fitted values and uncertainties.

        Returns
        -------
        str
            Formatted table string.
        """
        initial = self._initial_params
        current = self.params

        if fit_result is not None:
            fitted = fit_result.get('final_params', {})
            unc = fit_result.get('uncertainties', {})
            param_names = list(fitted.keys())
        else:
            fitted = {}
            unc = {}
            # Show key timing params
            param_names = [k for k in current if not k.startswith('_')
                           and not k.startswith('DMX') and not k.startswith('JUMP')
                           and isinstance(current[k], (int, float))
                           and k not in ('NTOA', 'CHI2', 'START', 'FINISH')]

        lines = []
        hdr = f"{'Parameter':<14s} {'Initial':>22s} {'Current':>22s}"
        if fit_result is not None:
            hdr += f" {'Uncertainty':>14s} {'Delta/sigma':>12s}"
        lines.append(hdr)
        lines.append('-' * len(hdr))

        for name in param_names:
            init_val = initial.get(name)
            curr_val = fitted.get(name, current.get(name))
            row = f"{name:<14s} "
            if isinstance(init_val, str):
                row += f"{init_val:>22s} {str(curr_val):>22s}"
            elif isinstance(init_val, (int, float)):
                row += f"{init_val:>22.15g} {curr_val:>22.15g}"
            else:
                row += f"{'N/A':>22s} {str(curr_val):>22s}"

            if fit_result is not None and name in unc:
                u = unc[name]
                row += f" {u:>14.6e}"
                if isinstance(init_val, (int, float)) and isinstance(curr_val, (int, float)) and u > 0:
                    delta = abs(curr_val - init_val)
                    row += f" {delta/u:>12.2f}"
                else:
                    row += f" {'':>12s}"
            lines.append(row)

        text = '\n'.join(lines)
        print(text)

    def weighted_rms(self, residuals_us: Optional[np.ndarray] = None) -> float:
        """Compute weighted RMS of residuals in microseconds.

        Parameters
        ----------
        residuals_us : ndarray, optional
            Residuals to use. If None, computes current residuals.

        Returns
        -------
        float
            Weighted RMS in microseconds.
        """
        if residuals_us is None:
            result = self.compute_residuals(subtract_tzr=True)
            residuals_us = result['residuals_us']
        errors_us = np.array([t.error_us for t in self.toas_data])
        weights = 1.0 / errors_us**2
        return float(np.sqrt(np.average(residuals_us**2, weights=weights)))

    def save_par(self, path, fit_result=None):
        """Save the current parameters to a par file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        fit_result : dict, optional
            Fit result dict for writing uncertainties and fit flags.
        """
        from jug.io.par_writer import write_par_file
        uncertainties = {}
        fit_params = set()
        if fit_result is not None:
            uncertainties = fit_result.get('uncertainties', {})
            fit_params = set(fit_result.get('final_params', {}).keys())
        write_par_file(self.params, path,
                       uncertainties=uncertainties, fit_params=fit_params)

    def save_tim(self, path, deleted_indices=None):
        """Save the tim file, commenting out deleted TOAs.

        Parameters
        ----------
        path : str or Path
            Output file path.
        deleted_indices : set of int, optional
            Zero-based indices of TOAs to comment out. Default: none.
        """
        from jug.io.tim_writer import write_tim_file
        if deleted_indices is None:
            deleted_indices = set()
        return write_tim_file(self.tim_file, path, deleted_indices)

    def inspect(self) -> str:
        """List available inspection methods and data attributes.

        Call this after loading a pulsar to see what you can do.
        """
        lines = [
            f"=== {self.params.get('PSR', 'Unknown')} - Available Inspections ===",
            "",
            "Session methods:",
            "  session.summary()                    - Overview of pulsar and data",
            "  session.free_params                  - Parameters with fit flag on (from par file)",
            "  session.set_free('F2', 'PX')         - Turn on fit flag for parameters",
            "  session.set_frozen('PX')             - Turn off fit flag for parameters",
            "  session.parameter_table()            - List all timing parameters",
            "  session.parameter_table(fit_result)  - Compare pre/post-fit with uncertainties",
            "  session.weighted_rms()               - Compute current weighted RMS (us)",
            "  session.compute_residuals()           - Compute timing residuals",
            "  session.fit_parameters(fit_params)    - Fit specified parameters",
            "  session.get_initial_params()          - Original par file parameters",
            "  session.save_par(path, fit_result)    - Write par file",
            "  session.save_tim(path, deleted)       - Write tim file",
            "",
            "Data attributes:",
            f"  session.params                       - Current parameters ({len([k for k in self.params if not k.startswith('_')])} keys)",
            f"  session.toas_data                    - TOA objects ({len(self.toas_data)} TOAs)",
            f"  session.par_file                     - {self.par_file}",
            f"  session.tim_file                     - {self.tim_file}",
            "",
            "Fit result keys (after fitting):",
            "  result['residuals_us']               - Post-fit residuals (us)",
            "  result['residuals_prefit_us']         - Pre-fit residuals (us)",
            "  result['final_params']               - Fitted parameter values",
            "  result['uncertainties']              - Parameter uncertainties",
            "  result['noise_realizations']         - Noise realizations (RedNoise, DMNoise, ECORR)",
            "  result['covariance']                 - Parameter covariance matrix",
            "  result['final_chi2']                 - Chi-squared",
            "  result['iterations']                 - Number of iterations",
            "  result['converged']                  - Convergence flag",
            "",
            "Noise estimation (requires numpyro):",
            "  from jug.noise.map_estimator import estimate_noise_parameters",
        ]
        text = '\n'.join(lines)
        print(text)

    def __repr__(self) -> str:
        return (
            f"TimingSession("
            f"par='{self.par_file.name}', "
            f"tim='{self.tim_file.name}', "
            f"ntoas={len(self.toas_data)})"
        )
