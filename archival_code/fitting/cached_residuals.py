"""
Cached Residual Calculator for Fast Fitting
============================================

This module provides a caching layer for timing residual computation
that dramatically speeds up iterative fitting by computing static
components only once.

Key optimization: Separate parameter-dependent from parameter-independent
components, and only recompute what changes during fitting.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import jax.numpy as jnp
import jax

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple


class CachedResidualCalculator:
    """
    Fast residual calculator that caches static timing components.
    
    This class computes timing model components once and reuses them
    across fitting iterations, providing ~5x speedup over naive
    recomputation.
    
    Usage
    -----
    >>> calc = CachedResidualCalculator('pulsar.par', 'pulsar.tim')
    >>> residuals = calc.compute_residuals(f0=339.3, f1=-1.6e-15)
    >>> residuals2 = calc.compute_residuals(f0=339.31, f1=-1.61e-15)  # Fast!
    
    Parameters
    ----------
    par_file : str or Path
        Path to par file
    tim_file : str or Path
        Path to tim file
    clock_dir : str or Path, optional
        Directory containing clock files (default: "data/clock")
    subtract_tzr : bool, optional
        Whether to subtract TZR reference (default: True)
    """
    
    def __init__(
        self,
        par_file,
        tim_file,
        clock_dir="data/clock",
        subtract_tzr=True
    ):
        self.par_file = Path(par_file)
        self.tim_file = Path(tim_file)
        self.clock_dir = clock_dir
        self.subtract_tzr = subtract_tzr
        
        # Parse files
        self.params = parse_par_file(self.par_file)
        self.toas_data = parse_tim_file_mjds(self.tim_file)
        
        # Extract TOA data
        self.toas_mjd = np.array([
            toa.mjd_int + toa.mjd_frac for toa in self.toas_data
        ])
        self.errors_us = np.array([toa.error_us for toa in self.toas_data])
        self.errors_sec = self.errors_us * 1e-6
        self.freqs_mhz = np.array([toa.freq_mhz for toa in self.toas_data])
        
        self.n_toas = len(self.toas_mjd)
        
        # Cache for static components
        self._static_cache = None
        self._fit_params = []
        
    def initialize_cache(self, fit_params: List[str]):
        """
        Initialize cache by computing all static components.
        
        This computes delays once and stores them for fast iteration.
        
        Parameters
        ----------
        fit_params : list of str
            List of parameter names being fitted (e.g., ['F0', 'F1'])
        """
        from jug.fitting.fast_residuals import compute_emission_times_once
        
        self._fit_params = fit_params
        
        print("Initializing fast cache...")
        
        # Compute emission times once (expensive!)
        cache_data = compute_emission_times_once(
            self.par_file,
            self.tim_file,
            self.clock_dir
        )
        
        # Store everything
        self._static_cache = {
            'toas_mjd': cache_data['toas_mjd'],
            't_emission_mjd': cache_data['t_emission_mjd'],
            'errors_sec': cache_data['errors_sec'],
            'weights': cache_data['weights'],
            'pepoch': cache_data['pepoch'],
            'f0_ref': cache_data['f0_ref'],
            'f1_ref': cache_data['f1_ref']
        }
        
        print(f"  Cache initialized for {self.n_toas} TOAs")
        print(f"  Fitting parameters: {fit_params}")
        print(f"  READY FOR FAST ITERATION!")
        
    def compute_residuals_fast(
        self,
        param_updates: Dict[str, float]
    ) -> Tuple[np.ndarray, float]:
        """
        ULTRA-FAST residual computation using cached delays.
        
        This only recomputes spin phase - everything else is cached!
        Expected: ~100x faster per iteration than full recomputation.
        
        Parameters
        ----------
        param_updates : dict
            Dictionary of parameter updates, e.g., {'F0': 339.315, 'F1': -1.6e-15}
        
        Returns
        -------
        residuals_sec : np.ndarray
            Timing residuals in seconds
        rms_us : float
            Weighted RMS residual in microseconds
        """
        if self._static_cache is None:
            raise RuntimeError("Must call initialize_cache() first!")
        
        from jug.fitting.fast_residuals import compute_phase_residuals_jax
        import jax.numpy as jnp
        
        # Extract parameters
        f0 = param_updates.get('F0', self.params['F0'])
        f1 = param_updates.get('F1', self.params.get('F1', 0.0))
        
        # Use cached data
        toas_mjd = jnp.array(self._static_cache['toas_mjd'])
        t_emission_mjd = jnp.array(self._static_cache['t_emission_mjd'])
        weights = jnp.array(self._static_cache['weights'])
        pepoch = self._static_cache['pepoch']
        
        # FAST computation (JAX JIT-compiled!)
        residuals_sec, rms_us = compute_phase_residuals_jax(
            toas_mjd,
            t_emission_mjd,
            f0,
            f1,
            pepoch,
            weights
        )
        
        # Convert back to numpy
        residuals_sec = np.array(residuals_sec)
        rms_us = float(rms_us)
        
        return residuals_sec, rms_us


class OptimizedFitter:
    """
    Optimized fitting class using cached residual calculator.
    
    This provides a high-level interface for fast fitting using
    the cached residual calculator.
    
    Example
    -------
    >>> fitter = OptimizedFitter('pulsar.par', 'pulsar.tim')
    >>> results = fitter.fit(['F0', 'F1'], max_iter=25)
    >>> print(f"Final RMS: {results['final_rms']:.6f} μs")
    """
    
    def __init__(
        self,
        par_file,
        tim_file,
        clock_dir="data/clock"
    ):
        self.calc = CachedResidualCalculator(
            par_file,
            tim_file,
            clock_dir=clock_dir,
            subtract_tzr=True
        )
        
    def fit(
        self,
        fit_params: List[str],
        max_iter: int = 25,
        convergence_threshold: float = 1e-14
    ) -> Dict:
        """
        Fit timing parameters using cached residual computation.
        
        Parameters
        ----------
        fit_params : list of str
            Parameters to fit (e.g., ['F0', 'F1'])
        max_iter : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence criterion on parameter changes
        
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'iterations': number of iterations
            - 'converged': whether fitting converged
            - 'final_params': fitted parameter values
            - 'final_rms': final RMS in microseconds
            - 'covariance': parameter covariance matrix
        """
        from jug.fitting.derivatives_spin import compute_spin_derivatives
        from jug.fitting.wls_fitter import wls_solve_svd
        
        # Initialize cache
        self.calc.initialize_cache(fit_params)
        
        # Get starting values
        param_values = {}
        for param in fit_params:
            param_values[param] = self.calc.params[param]
        
        # Fitting loop
        prev_delta_max = None
        
        for iteration in range(max_iter):
            # Compute residuals with current parameters
            residuals_sec, rms_us = self.calc.compute_residuals_fast(param_values)
            
            # Compute derivatives
            derivs = compute_spin_derivatives(
                self.calc.params,
                self.calc.toas_mjd,
                fit_params
            )
            
            # Build design matrix
            M = np.column_stack([derivs[p] for p in fit_params])
            
            # WLS solve
            delta_params, cov, _ = wls_solve_svd(
                residuals_sec,
                self.calc.errors_sec,
                M
            )
            
            # Update parameters
            for i, param in enumerate(fit_params):
                param_values[param] += delta_params[i]
                self.calc.params[param] = param_values[param]
            
            # Check convergence
            max_delta = np.max(np.abs(delta_params))
            
            if iteration > 0 and prev_delta_max is not None:
                if abs(max_delta - prev_delta_max) < 1e-20:
                    converged = True
                    iterations = iteration + 1
                    break
            
            if max_delta < convergence_threshold:
                converged = True
                iterations = iteration + 1
                break
            
            prev_delta_max = max_delta
        else:
            converged = False
            iterations = max_iter
        
        # Compute final residuals
        residuals_sec, rms_us = self.calc.compute_residuals_fast(param_values)
        
        return {
            'iterations': iterations,
            'converged': converged,
            'final_params': param_values,
            'final_rms': rms_us,
            'covariance': cov,
            'final_residuals_sec': residuals_sec
        }
