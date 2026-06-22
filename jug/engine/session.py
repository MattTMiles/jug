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
        # Snapshot original _high_precision strings (never modified)
        hp = self.params.get('_high_precision')
        self._original_high_precision = dict(hp) if hp else {}
        
        # Cache for expensive computations
        # Key change: cache separately by subtract_tzr mode for correctness
        self._cached_result_by_mode: Dict[bool, Dict[str, Any]] = {}  # {subtract_tzr: result}
        self._cached_delays: Optional[Dict[str, Any]] = None  # For fast postfit
        self._cached_toa_data: Optional[Dict[str, Any]] = None  # For ultra-fast postfit
        # Geometry cache: parameter-independent arrays (TDB, SSB pos, sun pos).
        # Populated on first compute_residuals call; never cleared after fits.
        # Only cleared if the TOA set changes (not yet supported, but noted here).
        self._geometry_cache: Dict[str, Any] = {}
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

        # Shallow copy so no mutation escapes to caller.
        # _high_precision is a nested dict but we only ever write new keys into
        # it (never mutate existing values), so shallow copy is safe here.
        params = dict(params)

        def format_param_value(param_name: str, value: Any) -> str:
            """Format parameter value for par file.

            Uses the ``_high_precision`` string cache when available so that
            parameters like F0, F1, PEPOCH, TASC, etc. are written with their
            full original precision rather than being truncated to 15
            significant digits of float64.
            """
            # Check _high_precision cache first (updated by _update_param)
            hp = params.get('_high_precision', {})
            hp_str = hp.get(param_name)
            if hp_str is not None:
                return f"{param_name:<12} {hp_str}"
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
            # Numeric values - use repr(float()) for max float64 precision
            if isinstance(value, (int, float)) or hasattr(value, '__float__'):
                return f"{param_name:<12} {repr(float(value))}"
            else:
                return f"{param_name:<12} {value}"

        # Keep temporary par files in the original coordinate family.
        from jug.io.astrometry_state import (
            ecliptic_aliases_from_equatorial,
            native_ecliptic_family,
            temp_par_skip_keys,
        )
        ecliptic_temp = bool(params.get('_ecliptic_coords'))
        temp_uses_lambda = native_ecliptic_family(self._initial_params) == 'lambda'
        if ecliptic_temp and temp_uses_lambda:
            params.update(ecliptic_aliases_from_equatorial(params, params))

        # Build temp par file with updated params
        # Convert KIN/KOM from DT92 back to IAU convention for par file
        # (compute_residuals_simple will apply IAU->DT92 again when reading)
        binary = params.get('BINARY', '').upper()
        if binary == 'T2' and ('KIN' in params or 'KOM' in params):
            if 'KIN' in params:
                params['KIN'] = 180.0 - float(params['KIN'])
            if 'KOM' in params:
                params['KOM'] = 90.0 - float(params['KOM'])

        with open(self.par_file, 'r') as f:
            original_lines = f.readlines()

        # If the original par file was TCB and we converted to TDB at init,
        # we must NOT keep any original lines verbatim — they contain TCB
        # values.  Keeping them in a file that will be read as TCB causes
        # double-conversion for any params that DID change (written as TDB)
        # while unchanged params keep TCB values → inconsistent model.
        # Instead, we skip the "keep original line" optimisation so every
        # param is written from self.params (all TDB).
        tcb_converted = params.get('_tcb_converted', False)

        updated_lines = []
        updated_params = set()
        jump_index = 0  # Track JUMP line index for JUMP1, JUMP2, ... mapping

        # Build case-insensitive lookup: map uppercase key → actual key in params
        params_upper = {k.upper(): k for k in params if not k.startswith('_')}

        for line in original_lines:
            line_stripped = line.strip()
            if line_stripped.startswith('#') or not line_stripped:
                updated_lines.append(line)
                continue

            parts = line_stripped.split()
            if parts:
                param_name = parts[0]

                # Force UNITS to TDB when session did TCB→TDB conversion
                if param_name == 'UNITS' and tcb_converted:
                    updated_lines.append(f"UNITS        TDB\n")
                    updated_params.add('UNITS')
                    continue

                # JUMP lines need special handling: par file has "JUMP"
                # but fitted values are stored as JUMP1, JUMP2, ...
                if param_name == 'JUMP':
                    jump_index += 1
                    jump_key = f'JUMP{jump_index}'
                    if jump_key in params:
                        new_val = params[jump_key]
                        # Reconstruct JUMP line with updated offset value
                        if parts[1].upper() == 'MJD' and len(parts) >= 5:
                            # MJD-based: JUMP MJD start end offset [fit] [unc]
                            parts[4] = f'{new_val:.15e}'
                        elif len(parts) >= 4:
                            # Flag-based: JUMP -flag val offset [fit] [unc]
                            parts[3] = f'{new_val:.15e}'
                        updated_lines.append(' '.join(parts) + '\n')
                        updated_params.add(jump_key)
                    else:
                        updated_lines.append(line)
                    continue

                # Case-insensitive lookup: par file may have TNRedAmp, params has TNREDAMP
                actual_key = params_upper.get(param_name.upper())
                if actual_key is not None:
                    new_value = params[actual_key]
                    # If value hasn't changed from initial AND the file was
                    # NOT TCB-converted, keep original line to preserve full
                    # precision (avoids float64 truncation).
                    # For TCB-converted files we must NOT keep original lines
                    # because they contain TCB values while params are TDB.
                    if not tcb_converted:
                        initial_val = self._initial_params.get(actual_key)
                        # The float64 value comparison is not sufficient for
                        # high-precision parameters: a GLS fit can refine F0
                        # (and similar) at the SUB-float64 level, which is
                        # captured only in the ``_high_precision`` string while
                        # the float64 value still compares equal. Keeping the
                        # original par line in that case writes the PREFIT
                        # high-precision value, discarding the refinement and
                        # reintroducing a ~1 ns linear-in-time phase error in
                        # this from-scratch evaluation (the fitter's own
                        # differential residuals and save_par/write_par_file
                        # both use the refined value, so the standalone
                        # post-fit compute_residuals would silently disagree).
                        hp_now = params.get('_high_precision', {}).get(actual_key)
                        hp_orig = self._original_high_precision.get(actual_key)
                        if (initial_val is not None and initial_val == new_value
                                and hp_now == hp_orig):
                            updated_lines.append(line)
                            updated_params.add(actual_key)
                            continue

                    new_line = format_param_value(actual_key, new_value)

                    # Preserve flags if present (for numeric params with fit flags)
                    if len(parts) > 2 and not isinstance(new_value, str):
                        flags = ' '.join(parts[2:])
                        new_line += f" {flags}"

                    updated_lines.append(new_line + '\n')
                    updated_params.add(actual_key)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        # Add any new parameters not in original file (skip internal keys).
        skip_ecliptic_derived = temp_par_skip_keys(params, self._initial_params)

        for param_name, value in params.items():
            if ecliptic_temp and param_name.upper() in skip_ecliptic_derived:
                continue
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
                verbose=False,
                geometry_cache=self._geometry_cache,
                toas=self.toas_data,
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
        
        # If params provided, merge overrides onto self.params and delegate to
        # _compute_residuals_with_params, which is the single canonical
        # implementation for building a temp par file with updated values.
        # It handles _high_precision strings, TCB conversion, ecliptic
        # round-trip, and KIN/KOM convention correctly.
        if params is not None:
            if self.verbose:
                print("  Using temp par file (params override)")
            merged = dict(self.params)
            merged.update(params)
            # Explicit overrides must take precedence over the
            # ``_high_precision`` string cache: _compute_residuals_with_params
            # writes hp strings to the temp par file in preference to dict
            # values, which would silently IGNORE overrides of hp-cached
            # parameters (F0, TASC, T0, PEPOCH, ...).
            hp = dict(merged.get('_high_precision', {}))
            for k, v in params.items():
                if k in hp:
                    if isinstance(v, str):
                        hp[k] = v
                    elif isinstance(v, np.longdouble):
                        hp[k] = str(v)
                    else:
                        hp[k] = repr(float(v))
            merged['_high_precision'] = hp
            return self._compute_residuals_with_params(merged, subtract_tzr)
        
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
                verbose=False,
                geometry_cache=self._geometry_cache,
                toas=self.toas_data,
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
        subtract_noise_sec: Optional[np.ndarray] = None,
        fit_dmx: bool = True,
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
        fit_dmx : bool, default True
            If True (default), DMX_* bins in the PAR are auto-added as fitted
            timing parameters (PINT-like). If False, the fixed DMX delays from
            the PAR are still applied to residuals but DMX_* are NOT fitted.
            Use False for global-DM recovery checks (fitting DM + all DMX is
            degenerate).

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

        # Snapshot current params before fitting so the fit report can show
        # changes relative to what went INTO this fit, not the original par file.
        prefit_params = dict(self.params)
        
        # FAST PATH: Use cached arrays if available
        # The fitter needs geometry arrays (dt_sec, tdb_mjd, freq_bary_mhz, etc.)
        # which are identical regardless of subtract_tzr mode, so accept either.
        cached_result = self._cached_result_by_mode.get(False) or self._cached_result_by_mode.get(True)
        if cached_result is None:
            if verbose:
                print("  Computing residuals for fitting cache...")
            self.compute_residuals(subtract_tzr=False, force_recompute=False)
            cached_result = self._cached_result_by_mode.get(False)
        
        # Check if we have all required cached data
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
                subtract_noise_sec=subtract_noise_sec,
                fit_dmx=fit_dmx,
            )
            
            # Run cached fit
            result = fit_parameters_optimized_cached(
                setup,
                max_iter=max_iter,
                convergence_threshold=convergence_threshold,
                verbose=verbose,
                solver_mode=solver_mode
            )

            # Barycentric arrival times the fitter evaluated the binary model
            # at (prefit-parameter values; enterprise-compatible, see
            # compute_residuals_simple). For BATs at the fitted parameters,
            # call compute_residuals() after this fit.
            for _bat_key in ('bat_mjd_ld', 'bat_mjd', 'bat_sec'):
                if _bat_key in cached_result and _bat_key not in result:
                    result[_bat_key] = cached_result[_bat_key]

            # Expose barycentric observing frequencies (MHz) and per-TOA flags
            # alongside bat_mjd/design_matrix. Sourced from the same residual
            # path (compute_residuals_simple) and the same self.toas_data order
            # as bat_mjd, so they are row-for-row aligned (unmasked, full length).
            if 'freq_bary_mhz' in cached_result and 'freq_bary_mhz' not in result:
                result['freq_bary_mhz'] = cached_result['freq_bary_mhz']
            if 'toa_flags' not in result:
                result['toa_flags'] = toa_flags
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
                verbose=verbose,
                fit_dmx=fit_dmx,
            )
        
        # Update session params with fitted values (CRITICAL for iterative fitting!)
        # Without this, subsequent fits would use old params and diverge.
        if result.get('success', True) and 'final_params' in result:
            from jug.io.par_reader import format_ra, format_dec
            updated_params = result['final_params'].copy()
            updated_params.update(result.get('final_dmx_params', {}))
            # Convert RAJ/DECJ from radians back to string format for consistency
            # Refresh the cached radian astrometry (_raj_rad/_decj_rad) from the
            # fitted value BEFORE formatting RAJ/DECJ to sexagesimal strings.
            # The forward model (simple_calculator: ra_rad = params['_raj_rad'])
            # prefers these cached radians; if left at the par value they make
            # L_hat (hence Roemer) stale while PMRA/PX are fitted -- a position
            # error that the parallax amplifies (J0437-4715: 0.2 mas RAJ shift x
            # 8 us parallax = ~84 ps annual). The ecliptic path below refreshes
            # them via sync; this covers equatorial (RAJ/DECJ) fits.
            if 'RAJ' in updated_params:
                updated_params['_raj_rad'] = float(updated_params['RAJ'])
                updated_params['RAJ'] = format_ra(updated_params['RAJ'])
            if 'DECJ' in updated_params:
                updated_params['_decj_rad'] = float(updated_params['DECJ'])
                updated_params['DECJ'] = format_dec(updated_params['DECJ'])

            if self.params.get('_ecliptic_coords'):
                from jug.io.astrometry_state import (
                    ecliptic_aliases_from_equatorial,
                    sync_ecliptic_public_to_internal,
                )
                ecl_params = dict(self.params)
                ecl_params.update(updated_params)
                sync_ecliptic_public_to_internal(ecl_params, updated_params)
                for key in (
                    '_ecliptic_lon_deg', '_ecliptic_lat_deg',
                    '_ecliptic_pm_lon', '_ecliptic_pm_lat',
                    'RAJ', 'DECJ', '_raj_rad', '_decj_rad',
                    'PMRA', 'PMDEC',
                ):
                    if key in ecl_params:
                        updated_params[key] = ecl_params[key]
                updated_params.update(ecliptic_aliases_from_equatorial(self.params, updated_params))

            self.params.update(updated_params)
            # Strip stale aliases: if canonical key just updated, remove any
            # alias keys that still hold the old par-file value.
            # Example: par file has A1DOT, fitter updates XDOT (canonical) →
            # session.params ends up with both; downstream code reading A1DOT
            # gets the stale value.  Remove aliases for every updated canonical.
            from jug.model.parameter_spec import _ALIAS_MAP
            _reverse_alias = {}
            for _a, _c in _ALIAS_MAP.items():
                _reverse_alias.setdefault(_c, []).append(_a)
            for _key in updated_params:
                for _alias in _reverse_alias.get(_key, []):
                    self.params.pop(_alias, None)

            # Update _high_precision strings for fitted parameters whose
            # values changed.  Always compute from the ORIGINAL longdouble
            # string + cumulative float64 delta from _initial_params.
            # new_ld = original_ld + (current_f64 - initial_f64)
            # This avoids double-counting corrections across successive fits
            # and preserves sub-float64 digits — critical for F0 where a
            # naive float64→string conversion loses ~3 digits and causes
            # multi-cycle phase errors (~ΔF0 × dt_max).
            hp = self.params.get('_high_precision')
            if hp is not None:
                _final_ld = result.get('final_params_ld', {}) or {}
                for p, new_val in updated_params.items():
                    if p.startswith('_') or not isinstance(new_val, (int, float)):
                        continue
                    init_val = self._initial_params.get(p)
                    orig_hp_str = self._original_high_precision.get(p)
                    _ld_val = _final_ld.get(p)
                    if init_val is not None and orig_hp_str is not None and (
                            float(init_val) != float(new_val) or _ld_val is not None):
                        if _ld_val is not None:
                            # The fitter carried this parameter in LONGDOUBLE
                            # (sub-float64-ULP steps; required for F0 in GLS
                            # red-noise fits). Use its value directly — the
                            # float64 delta reconstruction below cannot
                            # represent it and the 1-ULP snap-back would
                            # discard a real refinement. str(longdouble)
                            # round-trips exactly; skip notation matching.
                            new_ld = np.longdouble(_ld_val)
                            if str(np.longdouble(orig_hp_str)) != str(new_ld):
                                hp[p] = str(new_ld)
                            continue
                        # If a high-precision parameter moved by only one float64 ULP,
                        # the float64 result cannot represent the fitted sub-ULP value.
                        # Keep the original high-precision value rather than snapping to
                        # an arbitrary neighboring float, which can produce ns-level phase
                        # errors for spin parameters over long data spans.
                        if abs(float(new_val) - float(init_val)) <= abs(np.spacing(float(init_val))):
                            self.params[p] = init_val
                            updated_params[p] = init_val
                            if p in result.get('final_params', {}):
                                result['final_params'][p] = init_val
                            hp[p] = orig_hp_str
                            continue
                        orig_ld = np.longdouble(orig_hp_str)
                        delta = np.longdouble(float(new_val) - float(init_val))
                        new_ld = orig_ld + delta
                        # Match original notation style
                        if 'e' in orig_hp_str or 'E' in orig_hp_str:
                            n_digits = len(orig_hp_str.split('e')[0].split('E')[0].replace('-', '').replace('.', '')) - 1
                            hp[p] = np.format_float_scientific(
                                new_ld, precision=max(n_digits, 15), trim='0',
                            )
                        else:
                            n_dec = len(orig_hp_str.split('.')[-1]) if '.' in orig_hp_str else 15
                            hp[p] = np.format_float_positional(
                                new_ld, precision=max(n_dec, 15),
                                unique=False, trim='0',
                            )

        # Clear ALL cached geometry after a fit.
        # Delay arrays (dt_sec, roemer_shapiro_sec, prebinary_delay_sec, etc.)
        # depend on parameter values which just changed. Carrying forward stale
        # geometry causes the fitter's differential approach to accumulate
        # errors across fits (dt_sec has old delays baked in, but initial_*_delay
        # is computed with new params → inconsistency → parameter drift).
        # The next fit_parameters or compute_residuals call will recompute
        # everything fresh with the updated params.
        self._cached_result_by_mode.clear()

        # Also invalidate cached TOA data -- the fast evaluator only handles
        # spin/DM changes, so stale dt_sec causes wrong postfit residuals
        # when binary, astrometric, FD, or SW parameters were fitted.
        self._cached_toa_data = None

        # Canonicalize the returned post-fit residuals to the from-scratch
        # evaluator so result['residuals_us'] is bit-identical to a subsequent
        # compute_residuals() call (eliminates the ~0.01 ns RMS / ~0.15 ns p2p
        # float64 evaluation-order gap between the fitter's fast differential
        # evaluator and the from-scratch kernel). The from-scratch path is full
        # longdouble and matches PINT, so this is at least as accurate as the
        # differential residuals it replaces. This is only safe now that the
        # keep-original-par-line optimisation no longer discards the GLS
        # sub-float64 F0 refinement (see _compute_residuals_with_params); before
        # that fix this swap reintroduced a ~1 ns linear-in-time ramp.
        # Skipped for TOA-masked fits: the all-TOA compute_residuals() cannot
        # reproduce a fitted subset.
        if toa_mask is None and result.get('success', True) and \
                'residuals_us' in result:
            try:
                _post = self.compute_residuals(
                    subtract_tzr=True, force_recompute=True)
                _fresh = np.asarray(_post['residuals_us'], dtype=float)
                if _fresh.shape == np.asarray(result['residuals_us']).shape:
                    result['residuals_us'] = _fresh
                    if 'residuals_sec' in _post:
                        result['residuals_sec'] = np.asarray(
                            _post['residuals_sec'], dtype=float)
            except Exception:
                pass  # fall back to the fitter's differential residuals

        # Attach prefit snapshot so fit report can show per-fit changes
        result['prefit_params'] = prefit_params

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
        # (DMX_nnnn are auto-detected from the par file)
        return [p for p in all_flags
                if not re.match(r'^DMX_\d+$', p)]

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
                           'A1DOT', 'XDOT', 'EDOT',
                           'DR', 'DTH', 'A0', 'B0', 'SHAPMAX',
                           'H3', 'H4', 'STIGMA', 'K96']
            binary_keys.extend(sorted(
                (k for k in p if k.startswith('FB') and k[2:].isdigit()),
                key=lambda k: int(k[2:]),
            ))
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
        """Print a table comparing pre-fit and post-fit parameter values.

        Parameters
        ----------
        fit_result : dict, optional
            Fit result dict (with 'final_params', 'uncertainties', and
            'prefit_params'). If provided, shows fitted values and
            uncertainties relative to the pre-fit snapshot.

        Returns
        -------
        str
            Formatted table string.
        """
        # Use prefit snapshot from this fit if available, else fall back to
        # the original par file values.
        if fit_result is not None:
            initial = fit_result.get('prefit_params', self._initial_params)
        else:
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
        hdr = f"{'Parameter':<14s} {'Pre-fit':>22s} {'Post-fit':>22s}"
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
