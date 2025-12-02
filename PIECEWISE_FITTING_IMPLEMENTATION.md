# Piecewise PEPOCH Fitting: Implementation Instructions

**Date:** 2025-12-02  
**Status:** Ready for implementation  
**Priority:** High - addresses precision limitations for long-baseline datasets

---

## Executive Summary

JUG's current float64 fitting becomes numerically unstable for datasets spanning >10 years. This document provides complete instructions for implementing **piecewise PEPOCH fitting** - a coordinate transformation that maintains float64 precision while fitting a single global F0/F1 model.

**Key insight:** This adds ZERO free parameters. Local PEPOCHs are fixed geometric choices, not fitted quantities.

---

## 1. Problem Statement

### The Precision Issue

JUG uses JAX with float64 for fast fitting. For long baselines, this causes numerical problems:

| Quantity | 6-year dataset | 30-year dataset |
|----------|----------------|-----------------|
| Max dt from PEPOCH | ±3 years | ±15 years |
| dt in seconds | ±9.5×10⁷ s | ±4.7×10⁸ s |
| F1 design matrix element | ~10¹⁵ | ~10¹⁶ |
| Condition number | ~10²⁰ | ~10²³ |

When condition number exceeds ~10¹⁶, float64 WLS solve becomes unreliable.

### Measured Impact (J1909-3744, 6 years)

- ΔF1 error vs longdouble: 7.0×10⁻²³ Hz/s
- Residual difference RMS: 0.01 μs
- Growth rate: 3.5×10⁻⁶ μs/day from PEPOCH
- **R² = 0.997**: Error correlates with ΔF1 × dt²

### Extrapolation to 30 Years

- Residual error would reach ~50 ns
- Approaching unacceptable for precision timing/GW detection

---

## 2. Solution: Piecewise PEPOCH Fitting

### Concept

Divide data into temporal segments (e.g., 500-day chunks), each with a local reference epoch, but fit a **single global model** with continuity constraints.

### Mathematical Framework

**Global model:**
```
phase(t) = F0_global × (t - PEPOCH_global) + (F1_global/2) × (t - PEPOCH_global)²
```

**Segmentation:**
```
Segment i covers [t_start_i, t_end_i]
PEPOCH_local_i = (t_start_i + t_end_i) / 2
```

**Continuity constraint:**
```
F0_local_i = F0_global + F1_global × (PEPOCH_local_i - PEPOCH_global)
F1_local_i = F1_global  (same everywhere)
```

**Local phase computation:**
```
dt_local = t - PEPOCH_local_i  (SMALL: ±250 days, not ±15 years!)
phase = F0_local_i × dt_local + (F1_local_i/2) × dt_local²
```

### Why It Works

| Metric | Global method | Piecewise method |
|--------|---------------|------------------|
| dt magnitude | ±4.7×10⁸ s | ±2.2×10⁷ s |
| Design matrix elements | ~10¹⁶ | ~10¹⁴ |
| Condition number | ~10²³ | ~10¹⁸ |
| Float64 sufficient? | NO | YES |

**Critical:** Same model complexity (2 parameters: F0_global, F1_global). The local PEPOCHs are not free parameters.

---

## 3. Implementation Details

### 3.1 New File: `jug/fitting/piecewise_fitter.py`

Create this module with the following functions:

#### 3.1.1 Segmentation

```python
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import List, Dict, Tuple
from pathlib import Path

SECS_PER_DAY = 86400.0


def create_time_segments(
    tdb_mjd: np.ndarray,
    segment_duration_days: float = 500.0
) -> List[Dict]:
    """
    Divide TOAs into temporal segments.
    
    Parameters
    ----------
    tdb_mjd : np.ndarray
        TOA times in MJD (TDB), shape (n_toas,)
    segment_duration_days : float
        Target segment duration (default: 500 days ≈ 1.4 years)
        
    Returns
    -------
    segments : list of dict
        Each segment contains:
        - 'indices': np.ndarray of indices into original array
        - 'local_pepoch_mjd': float, mean MJD of segment
        - 'tmin_mjd': float, segment start
        - 'tmax_mjd': float, segment end
        - 'n_toas': int, number of TOAs in segment
        
    Notes
    -----
    - TOAs are NOT reordered; indices reference original array positions
    - Segments are created based on fixed time boundaries, not data gaps
    - Segments with zero TOAs are excluded
    """
    # Sort indices by time (keep original ordering reference)
    sort_idx = np.argsort(tdb_mjd)
    t_sorted = tdb_mjd[sort_idx]
    
    t_min = t_sorted[0]
    t_max = t_sorted[-1]
    
    # Create segment boundaries
    n_segments = max(1, int(np.ceil((t_max - t_min) / segment_duration_days)))
    
    segments = []
    for i in range(n_segments):
        seg_start = t_min + i * segment_duration_days
        seg_end = t_min + (i + 1) * segment_duration_days
        
        # Find TOAs in this segment (using original indices)
        mask = (tdb_mjd >= seg_start) & (tdb_mjd < seg_end)
        if i == n_segments - 1:  # Include endpoint in last segment
            mask = (tdb_mjd >= seg_start) & (tdb_mjd <= seg_end)
        
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue  # Skip empty segments
            
        seg_times = tdb_mjd[indices]
        
        segments.append({
            'indices': indices,
            'local_pepoch_mjd': float(np.mean(seg_times)),
            'tmin_mjd': float(np.min(seg_times)),
            'tmax_mjd': float(np.max(seg_times)),
            'n_toas': len(indices)
        })
    
    return segments
```

#### 3.1.2 Piecewise Residual Computation

```python
def compute_residuals_piecewise(
    dt_sec_global: np.ndarray,
    pepoch_global_mjd: float,
    segments: List[Dict],
    f0_global: float,
    f1_global: float
) -> np.ndarray:
    """
    Compute phase residuals using local coordinates per segment.
    
    Parameters
    ----------
    dt_sec_global : np.ndarray
        Time from global PEPOCH in seconds, shape (n_toas,)
        This is: (tdb_mjd - pepoch_global_mjd) * SECS_PER_DAY - total_delay_sec
    pepoch_global_mjd : float
        Global reference epoch in MJD
    segments : list of dict
        Output from create_time_segments()
    f0_global : float
        Current global F0 value (Hz)
    f1_global : float
        Current global F1 value (Hz/s)
        
    Returns
    -------
    residuals_sec : np.ndarray
        Phase residuals in seconds, shape (n_toas,)
        
    Notes
    -----
    For each segment:
    1. Compute dt_epoch = (local_pepoch - global_pepoch) in seconds
    2. Compute F0_local = F0_global + F1_global * dt_epoch
    3. Compute dt_local = dt_sec_global - dt_epoch (SMALL!)
    4. Compute phase = F0_local * dt_local + (F1_global/2) * dt_local²
    5. Wrap phase and convert to time
    """
    n_toas = len(dt_sec_global)
    residuals_sec = np.zeros(n_toas)
    
    for seg in segments:
        idx = seg['indices']
        
        # Epoch offset in seconds
        dt_epoch = (seg['local_pepoch_mjd'] - pepoch_global_mjd) * SECS_PER_DAY
        
        # Local F0 from continuity constraint
        f0_local = f0_global + f1_global * dt_epoch
        
        # Local time coordinates (SMALL!)
        dt_local = dt_sec_global[idx] - dt_epoch
        
        # Phase computation (float64 is fine since dt_local is small)
        phase = dt_local * (f0_local + dt_local * (f1_global / 2.0))
        
        # Wrap to nearest integer pulse
        phase_wrapped = phase - np.round(phase)
        
        # Convert to time residuals
        residuals_sec[idx] = phase_wrapped / f0_local
    
    return residuals_sec
```

#### 3.1.3 Piecewise Design Matrix

```python
def compute_design_matrix_piecewise(
    dt_sec_global: np.ndarray,
    pepoch_global_mjd: float,
    segments: List[Dict],
    f0_global: float,
    f1_global: float,
    fit_params: List[str] = ['F0', 'F1']
) -> np.ndarray:
    """
    Compute design matrix using local coordinates per segment.
    
    Parameters
    ----------
    dt_sec_global : np.ndarray
        Time from global PEPOCH in seconds
    pepoch_global_mjd : float
        Global reference epoch in MJD
    segments : list of dict
        Output from create_time_segments()
    f0_global : float
        Current global F0 value
    f1_global : float
        Current global F1 value
    fit_params : list of str
        Parameters to fit (must be ['F0', 'F1'] for now)
        
    Returns
    -------
    M : np.ndarray
        Design matrix, shape (n_toas, n_params)
        Convention: M[i,j] = -d(phase)/d(param_j) / F0 (PINT compatible)
        
    Notes
    -----
    Mathematical derivation:
    
    Phase in segment:
        φ = F0_local × dt_local + (F1_global/2) × dt_local²
        
    Where:
        F0_local = F0_global + F1_global × dt_epoch
        dt_local = dt_global - dt_epoch
        dt_epoch = (PEPOCH_local - PEPOCH_global) in seconds
        
    Derivatives (in phase units):
        ∂φ/∂F0_global = dt_local
        ∂φ/∂F1_global = dt_epoch × dt_local + dt_local²/2
        
    The dt_epoch term in ∂φ/∂F1_global comes from the chain rule through F0_local.
    
    Convert to time units (PINT convention):
        M[:,j] = -∂φ/∂param_j / F0
    """
    if fit_params != ['F0', 'F1']:
        raise NotImplementedError(
            f"Piecewise fitting currently only supports ['F0', 'F1'], got {fit_params}"
        )
    
    n_toas = len(dt_sec_global)
    n_params = len(fit_params)
    M = np.zeros((n_toas, n_params))
    
    for seg in segments:
        idx = seg['indices']
        
        # Epoch offset in seconds
        dt_epoch = (seg['local_pepoch_mjd'] - pepoch_global_mjd) * SECS_PER_DAY
        
        # Local F0 (for normalization)
        f0_local = f0_global + f1_global * dt_epoch
        
        # Local time coordinates
        dt_local = dt_sec_global[idx] - dt_epoch
        
        # Derivatives in phase units
        d_phase_d_f0 = dt_local
        d_phase_d_f1 = dt_epoch * dt_local + (dt_local ** 2) / 2.0
        
        # Convert to time units with PINT sign convention
        # M = -d(phase)/d(param) / F0
        M[idx, 0] = -d_phase_d_f0 / f0_local  # d(residual)/d(F0)
        M[idx, 1] = -d_phase_d_f1 / f0_local  # d(residual)/d(F1)
    
    return M
```

#### 3.1.4 Main Fitting Function

```python
def fit_parameters_piecewise(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str] = ['F0', 'F1'],
    segment_duration_days: float = 500.0,
    max_iter: int = 25,
    convergence_threshold: float = 1e-14,
    clock_dir: str = "data/clock",
    verbose: bool = True
) -> Dict:
    """
    Fit F0/F1 using piecewise PEPOCH method for numerical stability.
    
    This method divides data into temporal segments, each with a local
    reference epoch, while fitting a single global F0/F1 model. This
    improves numerical conditioning for long-baseline datasets.
    
    Parameters
    ----------
    par_file : Path
        Path to .par file
    tim_file : Path
        Path to .tim file
    fit_params : list of str
        Parameters to fit (currently must be ['F0', 'F1'])
    segment_duration_days : float
        Duration of each segment in days (default: 500)
        Smaller = better conditioning, but more segments
    max_iter : int
        Maximum fitting iterations
    convergence_threshold : float
        Convergence threshold on parameter changes
    clock_dir : str
        Path to clock correction files
    verbose : bool
        Print progress
        
    Returns
    -------
    result : dict
        Same format as fit_parameters_optimized():
        - 'final_params': dict with fitted F0, F1
        - 'uncertainties': dict with parameter uncertainties
        - 'final_rms': float, weighted RMS in microseconds
        - 'iterations': int, number of iterations
        - 'converged': bool
        - 'total_time': float, seconds
        - 'covariance': np.ndarray
        - 'segments': list, segment info for diagnostics
        
    Notes
    -----
    The algorithm:
    1. Load data, compute barycentric delays (cached)
    2. Create time segments with local PEPOCHs
    3. Iterate:
       a. Compute residuals using local coordinates
       b. Compute design matrix using local coordinates  
       c. Solve global WLS: [ΔF0, ΔF1] = (MᵀWM)⁻¹ MᵀW residuals
       d. Update: F0 += ΔF0, F1 += ΔF1
    4. Check convergence
    
    The local coordinates keep dt_local small (~10⁷ s instead of ~10⁸ s),
    improving numerical conditioning by ~100x.
    """
    import time as time_module
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.io.par_reader import parse_par_file, get_longdouble
    from jug.fitting.wls_fitter import wls_solve_svd
    
    if fit_params != ['F0', 'F1']:
        raise NotImplementedError(
            f"Piecewise fitting currently only supports ['F0', 'F1'], got {fit_params}"
        )
    
    total_start = time_module.time()
    
    if verbose:
        print("=" * 80)
        print("JUG PIECEWISE FITTER (Improved numerical precision)")
        print("=" * 80)
    
    # Parse parameter file
    params = parse_par_file(par_file)
    pepoch_global_mjd = float(get_longdouble(params, 'PEPOCH'))
    f0_global = float(params['F0'])
    f1_global = float(params['F1'])
    
    # Compute delays once (expensive)
    if verbose:
        print(f"\nComputing barycentric delays...")
    cache_start = time_module.time()
    
    result = compute_residuals_simple(
        par_file, tim_file,
        clock_dir=clock_dir,
        subtract_tzr=False,
        verbose=False
    )
    
    dt_sec_global = result['dt_sec']  # Time from PEPOCH including all delays
    tdb_mjd = result['tdb_mjd']
    errors_us = result['errors_us']
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec ** 2
    
    cache_time = time_module.time() - cache_start
    if verbose:
        print(f"  Cached delays for {len(dt_sec_global)} TOAs in {cache_time:.3f}s")
    
    # Create segments
    if verbose:
        print(f"\nCreating time segments (duration={segment_duration_days} days)...")
    segments = create_time_segments(tdb_mjd, segment_duration_days)
    if verbose:
        print(f"  Created {len(segments)} segments")
        for i, seg in enumerate(segments):
            print(f"    Segment {i}: {seg['n_toas']} TOAs, "
                  f"MJD {seg['tmin_mjd']:.1f}-{seg['tmax_mjd']:.1f}, "
                  f"local PEPOCH={seg['local_pepoch_mjd']:.1f}")
    
    # Store initial values for prefit residuals
    f0_start = f0_global
    f1_start = f1_global
    
    # Fitting loop
    if verbose:
        print(f"\nFitting F0, F1...")
    
    iteration_times = []
    prev_delta_max = None
    prefit_rms = None
    
    for iteration in range(max_iter):
        iter_start = time_module.time()
        
        # Compute residuals piecewise
        residuals_sec = compute_residuals_piecewise(
            dt_sec_global, pepoch_global_mjd, segments,
            f0_global, f1_global
        )
        
        # Subtract weighted mean
        weighted_mean = np.sum(residuals_sec * weights) / np.sum(weights)
        residuals_sec = residuals_sec - weighted_mean
        
        # Compute RMS
        rms_sec = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights))
        rms_us = rms_sec * 1e6
        
        if iteration == 0:
            prefit_rms = rms_us
        
        # Compute design matrix piecewise
        M = compute_design_matrix_piecewise(
            dt_sec_global, pepoch_global_mjd, segments,
            f0_global, f1_global, fit_params
        )
        
        # Subtract weighted mean from each column
        for j in range(M.shape[1]):
            col_mean = np.sum(M[:, j] * weights) / np.sum(weights)
            M[:, j] = M[:, j] - col_mean
        
        # Solve WLS
        delta_params, cov, _ = wls_solve_svd(
            jnp.array(residuals_sec),
            jnp.array(errors_sec),
            jnp.array(M),
            negate_dpars=False  # Already have correct sign convention
        )
        delta_params = np.array(delta_params)
        cov = np.array(cov)
        
        # Update parameters
        f0_global += delta_params[0]
        f1_global += delta_params[1]
        
        iter_time = time_module.time() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence
        max_delta = np.max(np.abs(delta_params))
        
        if verbose and (iteration < 3 or iteration >= max_iter - 1):
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, "
                  f"max|Δ|={max_delta:.2e}, time={iter_time:.3f}s")
        elif verbose and iteration == 3:
            print(f"  ...")
        
        # Convergence criteria
        if prev_delta_max is not None and abs(max_delta - prev_delta_max) < 1e-20:
            if verbose:
                print(f"  Converged (stagnation) at iteration {iteration+1}")
            converged = True
            iterations = iteration + 1
            break
        
        if max_delta < convergence_threshold:
            if verbose:
                print(f"  Converged (threshold) at iteration {iteration+1}")
            converged = True
            iterations = iteration + 1
            break
        
        prev_delta_max = max_delta
    else:
        converged = False
        iterations = max_iter
    
    total_time = time_module.time() - total_start
    
    # Compute final residuals
    residuals_final_sec = compute_residuals_piecewise(
        dt_sec_global, pepoch_global_mjd, segments,
        f0_global, f1_global
    )
    weighted_mean = np.sum(residuals_final_sec * weights) / np.sum(weights)
    residuals_final_sec = residuals_final_sec - weighted_mean
    residuals_final_us = residuals_final_sec * 1e6
    final_rms = np.sqrt(np.sum(residuals_final_sec**2 * weights) / np.sum(weights)) * 1e6
    
    # Compute prefit residuals
    residuals_prefit_sec = compute_residuals_piecewise(
        dt_sec_global, pepoch_global_mjd, segments,
        f0_start, f1_start
    )
    weighted_mean_pre = np.sum(residuals_prefit_sec * weights) / np.sum(weights)
    residuals_prefit_sec = residuals_prefit_sec - weighted_mean_pre
    residuals_prefit_us = residuals_prefit_sec * 1e6
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"\nConvergence:")
        print(f"  Iterations: {iterations}")
        print(f"  Converged: {converged}")
        print(f"\nTiming:")
        print(f"  Delay computation: {cache_time:.3f}s")
        print(f"  Fitting iterations: {sum(iteration_times):.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"\nFinal parameters:")
        print(f"  F0 = {f0_global:.20f} Hz")
        print(f"  F1 = {f1_global:.20e} Hz/s")
        print(f"\nUncertainties:")
        print(f"  σ(F0) = {np.sqrt(cov[0,0]):.3e} Hz")
        print(f"  σ(F1) = {np.sqrt(cov[1,1]):.3e} Hz/s")
        print(f"\nResiduals:")
        print(f"  Prefit RMS: {prefit_rms:.6f} μs")
        print(f"  Final RMS: {final_rms:.6f} μs")
    
    return {
        'final_params': {'F0': f0_global, 'F1': f1_global},
        'uncertainties': {
            'F0': np.sqrt(cov[0, 0]),
            'F1': np.sqrt(cov[1, 1])
        },
        'prefit_rms': prefit_rms,
        'final_rms': final_rms,
        'prefit_residuals_us': residuals_prefit_us,
        'postfit_residuals_us': residuals_final_us,
        'tdb_mjd': tdb_mjd,
        'errors_us': errors_us,
        'iterations': iterations,
        'converged': converged,
        'total_time': total_time,
        'cache_time': cache_time,
        'covariance': cov,
        'segments': segments  # For diagnostics
    }
```

### 3.2 JAX-Optimized Version (Optional Enhancement)

For maximum performance, create a JIT-compiled version:

```python
@jax.jit
def full_iteration_piecewise_jax(
    dt_sec_global: jnp.ndarray,
    segment_dt_epochs: jnp.ndarray,  # [dt_epoch for each segment]
    segment_starts: jnp.ndarray,     # Start indices
    segment_ends: jnp.ndarray,       # End indices  
    f0_global: float,
    f1_global: float,
    errors: jnp.ndarray,
    weights: jnp.ndarray
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    JIT-compiled piecewise iteration.
    
    Note: JAX doesn't like Python loops with dynamic bounds.
    Use jax.lax.fori_loop or pad segments to fixed size.
    """
    # Implementation depends on JAX expertise
    # May need to restructure for JAX compatibility
    pass
```

---

## 4. Testing Requirements

### 4.1 Test File: `jug/tests/test_piecewise_fitter.py`

```python
"""Tests for piecewise PEPOCH fitting."""

import numpy as np
import pytest
from pathlib import Path

from jug.fitting.piecewise_fitter import (
    create_time_segments,
    compute_residuals_piecewise,
    compute_design_matrix_piecewise,
    fit_parameters_piecewise
)


class TestSegmentation:
    """Test time segmentation."""
    
    def test_basic_segmentation(self):
        """Test that segmentation works correctly."""
        tdb_mjd = np.linspace(58000, 60000, 1000)  # ~5.5 years
        segments = create_time_segments(tdb_mjd, segment_duration_days=500)
        
        # Should have ~4 segments
        assert len(segments) >= 3
        assert len(segments) <= 5
        
        # All TOAs should be assigned
        all_indices = np.concatenate([seg['indices'] for seg in segments])
        assert len(np.unique(all_indices)) == len(tdb_mjd)
    
    def test_empty_gaps(self):
        """Test handling of data gaps."""
        tdb_mjd = np.concatenate([
            np.linspace(58000, 58500, 100),
            np.linspace(59500, 60000, 100)  # 1000-day gap
        ])
        segments = create_time_segments(tdb_mjd, segment_duration_days=500)
        
        # Should handle gap gracefully
        assert all(seg['n_toas'] > 0 for seg in segments)


class TestPiecewiseResiduals:
    """Test piecewise residual computation."""
    
    def test_consistency_with_global(self):
        """Piecewise residuals should match global for perfect data."""
        # Generate test data
        n_toas = 500
        pepoch_mjd = 59000.0
        tdb_mjd = np.linspace(58000, 60000, n_toas)
        
        F0 = 339.315687
        F1 = -1.6e-15
        
        # dt_sec from PEPOCH (simplified - no delays)
        dt_sec_global = (tdb_mjd - pepoch_mjd) * 86400.0
        
        # Create segments
        segments = create_time_segments(tdb_mjd, segment_duration_days=500)
        
        # Compute piecewise residuals
        res_piecewise = compute_residuals_piecewise(
            dt_sec_global, pepoch_mjd, segments, F0, F1
        )
        
        # Compute global residuals (standard method)
        phase_global = dt_sec_global * (F0 + dt_sec_global * (F1 / 2.0))
        phase_wrapped = phase_global - np.round(phase_global)
        res_global = phase_wrapped / F0
        
        # Should match to high precision
        diff = res_piecewise - res_global
        assert np.max(np.abs(diff)) < 1e-15, f"Max diff: {np.max(np.abs(diff))}"


class TestPiecewiseFitting:
    """Test full piecewise fitting."""
    
    @pytest.fixture
    def test_files(self):
        """Paths to test data files."""
        return {
            'par': Path("data/pulsars/J1909-3744.par"),
            'tim': Path("data/pulsars/J1909-3744.tim")
        }
    
    def test_fit_converges(self, test_files):
        """Test that piecewise fitting converges."""
        if not test_files['par'].exists():
            pytest.skip("Test data not available")
        
        result = fit_parameters_piecewise(
            test_files['par'],
            test_files['tim'],
            verbose=False
        )
        
        assert result['converged']
        assert result['final_rms'] < 1.0  # Should be < 1 μs
    
    def test_matches_standard_fitter(self, test_files):
        """Piecewise results should match standard fitter."""
        if not test_files['par'].exists():
            pytest.skip("Test data not available")
        
        from jug.fitting.optimized_fitter import fit_parameters_optimized
        
        result_piecewise = fit_parameters_piecewise(
            test_files['par'],
            test_files['tim'],
            verbose=False
        )
        
        result_standard = fit_parameters_optimized(
            test_files['par'],
            test_files['tim'],
            fit_params=['F0', 'F1'],
            verbose=False
        )
        
        # F0 should match to ~1e-12 Hz
        f0_diff = abs(result_piecewise['final_params']['F0'] - 
                      result_standard['final_params']['F0'])
        assert f0_diff < 1e-12, f"F0 diff: {f0_diff}"
        
        # F1 should match to ~1e-22 Hz/s (this is the key test!)
        f1_diff = abs(result_piecewise['final_params']['F1'] - 
                      result_standard['final_params']['F1'])
        assert f1_diff < 1e-20, f"F1 diff: {f1_diff}"


class TestSegmentBoundaries:
    """Test for artifacts at segment boundaries."""
    
    def test_no_discontinuities(self):
        """Residuals should be continuous across segment boundaries."""
        # This test requires real data
        # Implementation: check residuals near segment boundaries
        # Assert no jumps > 0.001 μs
        pass  # TODO: Implement with real data
```

### 4.2 Validation Against Longdouble

Create a validation script `validate_piecewise_precision.py`:

```python
"""
Validate piecewise fitting precision against longdouble reference.

This script:
1. Fits with piecewise method (float64)
2. Fits with longdouble reference  
3. Compares F1 values
4. Plots residual differences
"""

import numpy as np
from pathlib import Path

def fit_longdouble_reference(par_file, tim_file, clock_dir):
    """
    Fit using full longdouble precision (ground truth).
    
    This is SLOW but maximally precise.
    """
    # Implementation: replicate fitting logic with np.longdouble
    # everywhere: dt_sec, phase, derivatives, WLS solve
    pass

def main():
    par_file = Path("data/pulsars/J1909-3744.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    # Fit with piecewise
    from jug.fitting.piecewise_fitter import fit_parameters_piecewise
    result_piecewise = fit_parameters_piecewise(par_file, tim_file)
    
    # Fit with longdouble (if implemented)
    # result_longdouble = fit_longdouble_reference(par_file, tim_file)
    
    # Compare
    # f1_diff = result_piecewise['final_params']['F1'] - result_longdouble['F1']
    # print(f"F1 difference: {f1_diff:.3e} Hz/s")
    
    # For now, compare with standard method
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    result_standard = fit_parameters_optimized(
        par_file, tim_file, fit_params=['F0', 'F1']
    )
    
    print("\n" + "="*60)
    print("COMPARISON: Piecewise vs Standard")
    print("="*60)
    print(f"F0 piecewise: {result_piecewise['final_params']['F0']:.20f}")
    print(f"F0 standard:  {result_standard['final_params']['F0']:.20f}")
    print(f"F0 diff:      {result_piecewise['final_params']['F0'] - result_standard['final_params']['F0']:.3e}")
    print()
    print(f"F1 piecewise: {result_piecewise['final_params']['F1']:.20e}")
    print(f"F1 standard:  {result_standard['final_params']['F1']:.20e}")
    print(f"F1 diff:      {result_piecewise['final_params']['F1'] - result_standard['final_params']['F1']:.3e}")


if __name__ == "__main__":
    main()
```

---

## 5. Integration

### 5.1 Update `__init__.py`

Add to `jug/fitting/__init__.py`:

```python
from .piecewise_fitter import (
    fit_parameters_piecewise,
    create_time_segments,
    compute_residuals_piecewise,
    compute_design_matrix_piecewise
)
```

### 5.2 CLI Integration (Optional)

Add to CLI (if exists):

```python
@click.command()
@click.argument('par_file', type=click.Path(exists=True))
@click.argument('tim_file', type=click.Path(exists=True))
@click.option('--fit', '-f', multiple=True, default=['F0', 'F1'])
@click.option('--piecewise/--no-piecewise', default=False,
              help='Use piecewise PEPOCH fitting')
@click.option('--segment-days', default=500.0,
              help='Segment duration for piecewise fitting')
def fit(par_file, tim_file, fit, piecewise, segment_days):
    """Fit timing model parameters."""
    if piecewise:
        from jug.fitting.piecewise_fitter import fit_parameters_piecewise
        result = fit_parameters_piecewise(
            par_file, tim_file,
            fit_params=list(fit),
            segment_duration_days=segment_days
        )
    else:
        from jug.fitting.optimized_fitter import fit_parameters_optimized
        result = fit_parameters_optimized(
            par_file, tim_file,
            fit_params=list(fit)
        )
```

---

## 6. Success Criteria

**Primary Goal:** Minimize the difference between piecewise (float64) and longdouble methods. Longdouble is the ground truth - it has ~19 decimal digits of precision, which is more than sufficient for any pulsar timing application. The piecewise method is purely a numerical workaround to make float64/JAX behave more like longdouble.

| Criterion | Requirement | How to Verify |
|-----------|-------------|---------------|
| **F1 vs longdouble** | `\|F1_piecewise - F1_longdouble\| < 1e-24 Hz/s` | Compare with longdouble reference fit |
| **Residuals vs longdouble** | RMS(residual difference) < 0.001 μs | Subtract residual arrays, compute RMS |
| Residual RMS | Unchanged from standard method | Compare final RMS values |
| Speed | <10% slower than standard float64 | Time both methods |
| Segment continuity | No boundary jumps >0.001 μs | Check residuals at boundaries |
| Tests pass | All pytest tests pass | `pytest jug/tests/test_piecewise_fitter.py` |

**Key insight:** We're not trying to improve on longdouble - we're trying to match it while staying in float64/JAX for speed.

---

## 7. Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `jug/fitting/piecewise_fitter.py` | CREATE | Main implementation |
| `jug/tests/test_piecewise_fitter.py` | CREATE | Unit tests |
| `jug/fitting/__init__.py` | MODIFY | Add exports |
| `validate_piecewise_precision.py` | CREATE | Validation script (root dir) |

---

## 8. References

- **`PIECEWISE_FITTING_PROPOSAL.md`** - Full mathematical background and justification
- **`PIECEWISE_PRECISION_HANDOFF.md`** - Related hybrid method (different approach)
- **`jug/fitting/optimized_fitter.py`** - Current implementation to build upon
- **`jug/fitting/wls_fitter.py`** - WLS solver to reuse
- **`jug/fitting/derivatives_spin.py`** - Sign conventions reference

---

## 9. Questions for Clarification

Before implementation, clarify:

1. **Segment size:** Is 500 days optimal? Should it be configurable?
2. **JAX version:** Is a pure-numpy version acceptable first, with JAX optimization later?
3. **F2 support:** Should this extend to F2 (cubic term)?
4. **Error handling:** What should happen if a segment has <3 TOAs?

---

**End of Implementation Instructions**
