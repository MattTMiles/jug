"""
Optimized Fitter -- general-purpose timing model parameter fitting.
==================================================================

Fits any combination of timing model parameters (spin, DM, astrometry,
binary, FD, solar wind, JUMPs, DMX) using weighted least squares with
PINT-style damping and full nonlinear validation.

Entry Points
-------------
``fit_parameters_optimized(par_file, tim_file, fit_params, ...)``
    File-based entry point.  Parses par/tim, builds setup, runs iterations.

``fit_parameters_optimized_cached(setup, ...)``
    Cached entry point (used by the GUI / session layer).  Reuses a
    ``GeneralFitSetup`` that was built once from ``_build_general_fit_setup_from_cache``.

Internal Structure
------------------
1. **Setup** -- ``_build_setup_common`` / ``_build_general_fit_setup_from_files`` /
   ``_build_general_fit_setup_from_cache`` compute all expensive delays once and
   bundle them into a ``GeneralFitSetup`` dataclass.

2. **Iteration** -- ``_run_general_fit_iterations`` takes a setup and runs the
   WLS + damping loop: build design matrix, solve, validate step against full
   nonlinear model, repeat until convergence.

3. **Helpers** -- ``_update_param``, ``_reconvert_ecliptic_to_equatorial``,
   ``compute_dm_delay_fast``, ``_solve_augmented_cholesky``, ``_build_jump_masks``,
   ``_compute_full_model_residuals``.

Usage Example
-------------
>>> from jug.fitting.optimized_fitter import fit_parameters_optimized
>>> result = fit_parameters_optimized(
...     par_file="J1909.par", tim_file="J1909.tim",
...     fit_params=['F0', 'F1', 'RAJ', 'DECJ']
... )
>>> print(f"Fitted F0: {result['final_params']['F0']:.15f} Hz")
>>> print(f"Final RMS: {result['final_rms']:.6f} mus")
"""

# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import math
from dataclasses import dataclass

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, validate_par_timescale
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.derivatives_dm import compute_dm_derivatives
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY
from jug.fitting.wls_fitter import wls_solve_svd
from jug.fitting.binary_registry import compute_binary_delay, compute_binary_derivatives
import scipy.linalg as _scipy_linalg

# Import ParameterSpec system for spec-driven routing
from jug.model.parameter_spec import (
    is_spin_param,
    is_dm_param,
    is_jump_param,
    get_spin_params_from_list,
    get_dm_params_from_list,
    get_binary_params_from_list,
    get_astrometry_params_from_list,
    get_fd_params_from_list,
    get_sw_params_from_list,
    canonicalize_param_name,
    validate_fit_param,
)
from jug.utils.constants import HIGH_PRECISION_PARAMS


def _update_param(params: Dict, param: str, value: float) -> None:
    """Update a parameter value, keeping _high_precision cache consistent.

    When the fitter updates a high-precision parameter (F0, F1, PEPOCH, etc.),
    the ``_high_precision`` string cache must also be updated. Otherwise,
    ``get_longdouble()`` returns the stale prefit value from the cache instead
    of the fitted value, causing phase-computation errors of order 100 ns for
    F0 over multi-year data spans.

    For ecliptic parameters (ELONG, ELAT, PMELONG, PMELAT), update the
    internal ``_ecliptic_*`` keys and reconvert to equatorial coordinates
    so that delay models (which use RAJ/DECJ) stay consistent.

    Parameters
    ----------
    params : dict
        Parameter dictionary (modified in-place).
    param : str
        Parameter name.
    value : float
        New parameter value (float64).
    """
    param_upper = param.upper()

    if param_upper == 'ELONG':
        params['ELONG'] = value
        params['_ecliptic_lon_deg'] = value
        _reconvert_ecliptic_to_equatorial(params)
        return
    elif param_upper == 'ELAT':
        params['ELAT'] = value
        params['_ecliptic_lat_deg'] = value
        _reconvert_ecliptic_to_equatorial(params)
        return
    elif param_upper == 'PMELONG':
        params['PMELONG'] = value
        params['_ecliptic_pm_lon'] = value
        _reconvert_ecliptic_to_equatorial(params)
        return
    elif param_upper == 'PMELAT':
        params['PMELAT'] = value
        params['_ecliptic_pm_lat'] = value
        _reconvert_ecliptic_to_equatorial(params)
        return

    params[param_upper] = value
    # Keep _high_precision in sync so get_longdouble() returns the fitted
    # value during iterations (otherwise compute_phase_residuals uses stale
    # F0/F1 from the original par file, breaking the nonlinear model).
    # Session.py reconstructs full longdouble HP after the fit completes.
    hp = params.get('_high_precision')
    if hp is not None and param_upper in hp:
        hp[param_upper] = repr(float(value))


def _reconvert_ecliptic_to_equatorial(params: Dict) -> None:
    """Reconvert ecliptic coords to equatorial after an ecliptic param update."""
    from jug.io.par_reader import (
        OBLIQUITY_ARCSEC, format_ra, format_dec
    )
    import numpy as np_

    ecl_lon_deg = params.get('_ecliptic_lon_deg', 0.0)
    ecl_lat_deg = params.get('_ecliptic_lat_deg', 0.0)
    ecl_frame = str(params.get('_ecliptic_frame', 'IERS2010'))
    obl_rad = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010']) * np_.pi / (180.0 * 3600.0)

    lon_rad = np_.radians(ecl_lon_deg)
    lat_rad = np_.radians(ecl_lat_deg)
    cos_lon, sin_lon = np_.cos(lon_rad), np_.sin(lon_rad)
    cos_lat, sin_lat = np_.cos(lat_rad), np_.sin(lat_rad)
    cos_obl, sin_obl = np_.cos(obl_rad), np_.sin(obl_rad)

    x = cos_lon * cos_lat
    y = sin_lon * cos_lat * cos_obl - sin_lat * sin_obl
    z = sin_lon * cos_lat * sin_obl + sin_lat * cos_obl

    ra_rad = np_.arctan2(y, x) % (2 * np_.pi)
    dec_rad = np_.arctan2(z, np_.sqrt(x**2 + y**2))

    params['RAJ'] = format_ra(ra_rad)
    params['DECJ'] = format_dec(dec_rad)

    # Reconvert proper motions if present
    pm_lon = params.get('_ecliptic_pm_lon', 0.0)
    pm_lat = params.get('_ecliptic_pm_lat', 0.0)
    if pm_lon != 0.0 or pm_lat != 0.0:
        dx = -sin_lon * pm_lon - cos_lon * sin_lat * pm_lat
        dy = cos_lon * pm_lon - sin_lon * sin_lat * pm_lat
        dz = cos_lat * pm_lat

        dx_eq = dx
        dy_eq = dy * cos_obl - dz * sin_obl
        dz_eq = dy * sin_obl + dz * cos_obl

        cos_ra, sin_ra = np_.cos(ra_rad), np_.sin(ra_rad)
        cos_dec, sin_dec = np_.cos(dec_rad), np_.sin(dec_rad)
        params['PMRA'] = -sin_ra * dx_eq + cos_ra * dy_eq
        params['PMDEC'] = -cos_ra * sin_dec * dx_eq - sin_ra * sin_dec * dy_eq + cos_dec * dz_eq


@dataclass
class GeneralFitSetup:
    """
    Setup data for general parameter fitting.
    
    This bundles all arrays needed for the iteration loop,
    separating expensive setup from fast iterations.
    
    Attributes
    ----------
    params : dict
        Initial parameter dictionary
    fit_param_list : list of str
        Parameters to fit
    param_values_start : list of float
        Initial parameter values
    toas_mjd : np.ndarray
        TOA times in MJD
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz
    errors_us : np.ndarray
        TOA uncertainties in microseconds
    errors_sec : np.ndarray
        TOA uncertainties in seconds
    weights : np.ndarray
        TOA weights (1/sigma^2)
    dt_sec_cached : np.ndarray
        Precomputed time differences (float64, for backward compat)
    dt_sec_ld : np.ndarray or None
        Precomputed time differences in longdouble (for phase precision)
    tdb_mjd : np.ndarray
        TDB times in MJD
    initial_dm_delay : np.ndarray or None
        Initial DM delay (for DM fitting)
    dm_params : list of str
        DM parameters being fit (empty if none)
    spin_params : list of str
        Spin parameters being fit
    binary_params : list of str
        Binary parameters being fit (empty if none)
    astrometry_params : list of str
        Astrometry parameters being fit (empty if none)
    roemer_shapiro_sec : np.ndarray or None
        Roemer + Shapiro delays in seconds (legacy, for backward compatibility)
    prebinary_delay_sec : np.ndarray or None
        Pre-binary delays in seconds (roemer_shapiro + DM + SW + tropo, PINT-compatible)
    ssb_obs_pos_ls : np.ndarray or None
        SSB to observatory position in light-seconds (for astrometry fitting)
    """
    params: Dict[str, float]
    fit_param_list: List[str]
    param_values_start: List[float]
    toas_mjd: np.ndarray
    freq_mhz: np.ndarray
    errors_us: np.ndarray
    errors_sec: np.ndarray
    weights: np.ndarray
    dt_sec_cached: np.ndarray
    dt_sec_ld: Optional[np.ndarray]  # longdouble version for phase precision
    tdb_mjd: np.ndarray
    initial_dm_delay: Optional[np.ndarray]
    dm_params: List[str]
    spin_params: List[str]
    binary_params: List[str]
    astrometry_params: List[str]
    fd_params: List[str]  # FD parameters being fit
    sw_params: List[str]  # Solar wind parameters being fit
    roemer_shapiro_sec: Optional[np.ndarray]
    prebinary_delay_sec: Optional[np.ndarray]  # PINT-compatible pre-binary time
    initial_binary_delay: Optional[np.ndarray]  # For binary fitting iteration
    ssb_obs_pos_ls: Optional[np.ndarray]
    obs_sun_pos_ls: Optional[np.ndarray]  # Sun position relative to observer (for Shapiro recomputation)
    obs_planet_pos_ls: Optional[dict]  # Planet positions relative to observer (for planet Shapiro)
    initial_astrometric_delay: Optional[np.ndarray]  # For astrometry fitting iteration
    initial_fd_delay: Optional[np.ndarray]  # For FD fitting iteration
    initial_sw_delay: Optional[np.ndarray]  # For SW fitting iteration
    sw_geometry_pc: Optional[np.ndarray]  # Solar wind geometry factor per TOA
    toa_flags: Optional[List[Dict[str, str]]]  # Per-TOA flags from tim file
    ecorr_whitener: object  # ECORRWhitener or None (for block-diagonal covariance)
    # Red/DM noise Fourier basis and prior (Phase 1 integration)
    red_noise_basis: Optional[np.ndarray]  # (n_toa, 2*n_harmonics) Fourier design matrix
    red_noise_prior: Optional[np.ndarray]  # (2*n_harmonics,) diagonal prior variances
    dm_noise_basis: Optional[np.ndarray]   # (n_toa, 2*n_harmonics) chromatic Fourier design matrix
    dm_noise_prior: Optional[np.ndarray]   # (2*n_harmonics,) diagonal prior variances
    chromatic_noise_basis: Optional[np.ndarray]   # (n_toa, 2*n_harmonics) chromatic Fourier design matrix
    chromatic_noise_prior: Optional[np.ndarray]   # (2*n_harmonics,) diagonal prior variances
    ecorr_basis: Optional[np.ndarray]      # (n_toa, n_epochs) quantization matrix
    ecorr_prior: Optional[np.ndarray]      # (n_epochs,) ECORR^2 prior variances (s^2)
    # Band noise (multiple frequency bands)
    band_noise_bases: Optional[List[np.ndarray]]   # list of (n_toa, 2*n_harmonics) per band
    band_noise_priors: Optional[List[np.ndarray]]  # list of (2*n_harmonics,) per band
    band_noise_labels: Optional[List[str]]         # e.g. ["BandNoise_0_1000", "BandNoise_1000_2000"]
    # Group noise (multiple backend groups)
    group_noise_bases: Optional[List[np.ndarray]]  # list of (n_toa, 2*n_harmonics) per group
    group_noise_priors: Optional[List[np.ndarray]] # list of (2*n_harmonics,) per group
    group_noise_labels: Optional[List[str]]        # e.g. ["GroupNoise_CPSR2_20CM"]
    # DMX design matrix (Phase 2 integration)
    dmx_design_matrix: Optional[np.ndarray]  # (n_toa, n_dmx_ranges) DMX design matrix
    dmx_labels: Optional[List[str]]          # DMX parameter labels
    # DMJUMP design matrix
    dmjump_design_matrix: Optional[np.ndarray]  # (n_toa, n_dmjumps) DMJUMP design matrix
    dmjump_labels: Optional[List[str]]          # DMJUMP parameter labels
    # JUMP masks {JUMP1: bool_mask, JUMP2: bool_mask, ...}
    jump_masks: Optional[Dict[str, np.ndarray]]
    # FDJUMP masks and metadata
    fdjump_masks: Optional[Dict[str, np.ndarray]]
    fdjump_params: List[str]
    initial_fdjump_delay: Optional[np.ndarray]  # For FDJUMP fitting iteration
    # JUMP phase offsets from par file (longdouble, F0 * JUMP_value per TOA)
    jump_phase: Optional[np.ndarray]
    # TZR phase for correct pulse numbering (longdouble scalar)
    tzr_phase: Optional[float]
    # Noise configuration (Phase 3 integration)
    noise_config: object  # NoiseConfig or None


# =============================================================================
# Parameter Routing Helpers (spec-driven replacements for startswith checks)
# =============================================================================

def _get_param_default_value(param: str) -> float:
    """
    Get default value for a parameter not found in .par file.

    This is the spec-driven replacement for:
        if param.startswith('F') and param[1:].isdigit(): default = 0.0
        elif param.startswith('DM') and ...: default = 0.0

    Parameters
    ----------
    param : str
        Parameter name

    Returns
    -------
    float or None
        Default value if known, None if no default available

    Notes
    -----
    Spin and DM parameters default to 0.0 (higher-order terms often missing).
    Returns None for unknown parameters to trigger an error.
    """
    if is_spin_param(param):
        return 0.0
    elif is_dm_param(param):
        return 0.0
    else:
        return None  # No default - will raise error


def _format_param_value_for_print(param: str, value: float, uncertainty: float = None) -> str:
    """
    Format parameter value for printing.

    This is the spec-driven replacement for startswith checks in print statements.

    Parameters
    ----------
    param : str
        Parameter name
    value : float
        Parameter value
    uncertainty : float, optional
        Uncertainty (if available)

    Returns
    -------
    str
        Formatted string
    """
    if is_spin_param(param) and abs(value) < 1e-10:
        if uncertainty is not None:
            return f"  {param} = {value:.20e} +/- {uncertainty:.6e}"
        else:
            if value != 0:
                return f"  {param} = {value:.20e}"
            else:
                return f"  {param} = {value:.15f}"
    elif is_dm_param(param):
        if uncertainty is not None:
            return f"  {param} = {value:.10f} +/- {uncertainty:.6e} pc cm^-^3"
        else:
            return f"  {param} = {value:.10f} pc cm^-^3"
    else:
        if uncertainty is not None:
            return f"  {param} = {value:.15f} +/- {uncertainty:.6e}"
        else:
            return f"  {param} = {value:.15f}"


def compute_dm_delay_fast(tdb_mjd: np.ndarray, freq_mhz: np.ndarray,
                          dm_params: Dict[str, float], dm_epoch: float) -> np.ndarray:
    """Fast computation of DM delay without file I/O.

    Parameters
    ----------
    tdb_mjd : np.ndarray
        TDB times in MJD
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz
    dm_params : dict
        DM parameters {'DM': value, 'DM1': value, ...}
    dm_epoch : float
        DMEPOCH in MJD

    Returns
    -------
    dm_delay_sec : np.ndarray
        DM delay in seconds
    """
    # Build DM polynomial coefficients
    dm_coeffs = []
    dm_factorials = []
    for i in range(10):  # Support up to DM9
        param = f'DM{i}' if i > 0 else 'DM'
        if param in dm_params and dm_params[param] is not None:
            dm_coeffs.append(dm_params[param])
            dm_factorials.append(math.factorial(i))
        elif param == 'DM':
            dm_coeffs.append(0.0)
            dm_factorials.append(1.0)
        else:
            break

    dm_coeffs = np.array(dm_coeffs)
    dm_factorials = np.array(dm_factorials)

    # Compute DM polynomial: DM(t) = sum(DM_i * (t-DMEPOCH)^i / i!)
    # Note: PINT uses years, so convert MJD difference to years
    dt_years = (tdb_mjd - dm_epoch) / 365.25

    dm_eff = np.zeros_like(tdb_mjd)
    for i, (coeff, factorial) in enumerate(zip(dm_coeffs, dm_factorials)):
        dm_eff += coeff * (dt_years ** i) / factorial

    # Compute DM delay: tau_DM = K_DM * DM(t) / freq^2
    dm_delay_sec = K_DM_SEC * dm_eff / (freq_mhz ** 2)

    return dm_delay_sec


def _has_gpu():
    """Check whether JAX has a GPU backend available (cached)."""
    if not hasattr(_has_gpu, '_result'):
        _has_gpu._result = jax.default_backend() != 'cpu'
    return _has_gpu._result


def _solve_augmented_cholesky(M2, r1, prior_inv, col_norms, n_timing_cols, has_offset):
    """Solve the augmented (GLS) WLS system via normal equations + Cholesky.

    Equivalent to the SVD-on-augmented-system approach but ~70x faster for
    large design matrices.  Auto-dispatches: uses JAX on GPU for the
    normal-equation matmul and Cholesky (1.7x faster than CPU scipy for
    59k-TOA problems); uses scipy LAPACK on CPU where it is faster.
    Falls back to scipy SVD if Cholesky fails.

    Parameters
    ----------
    M2 : (n_toa, n_cols) -- column-normalised, weight-scaled design matrix
    r1 : (n_toa,)        -- weight-scaled residuals
    prior_inv : (n_cols,) -- diagonal prior inverse (0 for unregularised cols)
    col_norms : (n_cols,) -- column norms used for preconditioning
    n_timing_cols : int   -- number of timing model columns (incl. offset)
    has_offset : bool     -- whether column 0 is an offset column

    Returns
    -------
    delta_params, cov, delta_params_all, cov_all, noise_coeffs
    """
    n_cols = M2.shape[1]

    if _has_gpu():
        # GPU path: JAX matmul + Cholesky on device
        import jax.scipy.linalg as _jax_linalg

        M2_j = jnp.asarray(M2)
        r1_j = jnp.asarray(r1)

        MtM_j = M2_j.T @ M2_j
        Mtr_j = M2_j.T @ r1_j
        MtM_j = MtM_j.at[jnp.diag_indices(n_cols)].add(jnp.asarray(prior_inv))

        L_j = _jax_linalg.cho_factor(MtM_j, lower=True)
        delta_normalized_j = _jax_linalg.cho_solve(L_j, Mtr_j)
        cov_normalized_j = _jax_linalg.cho_solve(L_j, jnp.eye(n_cols))

        # JAX returns NaN (not an exception) when Cholesky fails
        cholesky_ok = not jnp.any(jnp.isnan(delta_normalized_j))

        if cholesky_ok:
            delta_normalized = np.asarray(delta_normalized_j)
            cov_normalized = np.asarray(cov_normalized_j)
            MtM = np.asarray(MtM_j)
        else:
            MtM = np.asarray(MtM_j)
            cholesky_ok = False  # fall through to SVD below
    else:
        # CPU path: scipy LAPACK (faster than JAX XLA on CPU)
        MtM = np.array(M2.T @ M2)
        Mtr = np.array(M2.T @ r1)
        MtM[np.diag_indices(n_cols)] += prior_inv

        try:
            L = _scipy_linalg.cho_factor(MtM, lower=True, check_finite=False)
            delta_normalized = _scipy_linalg.cho_solve(L, Mtr, check_finite=False)
            cov_normalized = _scipy_linalg.cho_solve(L, np.eye(n_cols), check_finite=False)
            cholesky_ok = True
        except _scipy_linalg.LinAlgError:
            cholesky_ok = False

    if not cholesky_ok:
        # Cholesky failed -- fall back to SVD on CPU (rare path)
        if not isinstance(M2, np.ndarray):
            M2 = np.asarray(M2)
        if not isinstance(r1, np.ndarray):
            r1 = np.asarray(r1)
        has_prior = np.any(prior_inv > 0)
        if has_prior:
            sqrt_prior_inv = np.sqrt(prior_inv)
            M_aug = np.vstack([M2, np.diag(sqrt_prior_inv)])
            r_aug = np.concatenate([r1, np.zeros(n_cols)])
        else:
            M_aug = M2
            r_aug = r1
        U, Sdiag, VT = _scipy_linalg.svd(M_aug, full_matrices=False)
        threshold = 1e-14 * max(M_aug.shape)
        Sdiag_inv = np.where(Sdiag > threshold * Sdiag[0], 1.0 / Sdiag, 0.0)
        delta_normalized = VT.T @ (Sdiag_inv * (U.T @ r_aug))
        cov_normalized = (VT.T * Sdiag_inv**2) @ VT

    if np.any(np.isnan(cov_normalized)):
        cov_normalized = np.asarray(jnp.linalg.pinv(jnp.array(MtM)))

    delta_params_all = delta_normalized / col_norms
    cov_all = (cov_normalized / col_norms).T / col_norms
    t0 = 1 if has_offset else 0
    delta_params = delta_params_all[t0:n_timing_cols]
    cov = cov_all[t0:n_timing_cols, t0:n_timing_cols]
    noise_coeffs = delta_params_all[n_timing_cols:]
    return delta_params, cov, delta_params_all, cov_all, noise_coeffs


def _solve_gls_woodbury(M_timing, F_noise, residuals, sigma, phiinv_noise,
                         has_offset, n_timing_params,
                         precomputed=None, compute_noise_cov=False):
    """Solve GLS timing fit using the Woodbury identity (Tempo2-equivalent).

    Mathematically equivalent to Tempo2's cholesky plugin:
      C = N + F φ F^T
      M^T C^{-1} M δp = M^T C^{-1} y
    but avoids forming the full n_toa × n_toa covariance matrix by using
      C^{-1} = N^{-1} - N^{-1} F (φ^{-1} + F^T N^{-1} F)^{-1} F^T N^{-1}

    After solving for timing parameters, noise coefficients are extracted
    via the Wiener filter: a = Σ^{-1} F^T N^{-1} (y - M δp).

    Parameters
    ----------
    M_timing : (n_toa, n_timing_cols) unweighted timing design matrix
    F_noise : (n_toa, n_noise) unweighted noise basis functions
    residuals : (n_toa,) residuals in seconds
    sigma : (n_toa,) TOA uncertainties in seconds
    phiinv_noise : (n_noise,) inverse prior variance for noise coefficients
    has_offset : bool
    n_timing_params : int
    precomputed : dict or None
        Cached products from _precompute_woodbury_products(). When provided,
        skips recomputing FtNiF, L_sigma, Ni, and GPU-resident arrays.
    compute_noise_cov : bool
        If True, compute noise posterior covariance diagonal (expensive).
        Only needed on the final iteration for error bars.

    Returns
    -------
    delta_params, cov, delta_params_all, cov_all, noise_coeffs
    """
    n_toa, n_tcols = M_timing.shape
    n_noise = F_noise.shape[1]

    if precomputed is not None:
        Ni = precomputed['Ni']
        FtNiF = precomputed['FtNiF']
        L_sigma = precomputed['L_sigma']
        F_j = precomputed.get('F_j')
        Ni_j = precomputed.get('Ni_j')
        use_gpu = F_j is not None
    else:
        Ni = 1.0 / (sigma ** 2)
        use_gpu = _has_gpu()
        F_j = None
        Ni_j = None

    # Compute iteration-dependent products (M and y change each iteration)
    if use_gpu:
        if F_j is None:
            F_j = jnp.asarray(F_noise)
            Ni_j = jnp.asarray(Ni)
        M_j = jnp.asarray(M_timing)
        y_j = jnp.asarray(residuals)
        NiM_j = M_j * Ni_j[:, None]
        Niy_j = y_j * Ni_j
        if precomputed is None:
            NiF_j = F_j * Ni_j[:, None]
            FtNiF = np.asarray(F_j.T @ NiF_j)
        FtNiM = np.asarray(F_j.T @ NiM_j)
        MtNiM = np.asarray(M_j.T @ NiM_j)
        FtNiy = np.asarray(F_j.T @ Niy_j)
        MtNiy = np.asarray(M_j.T @ Niy_j)
    else:
        NiM = M_timing * Ni[:, None]
        Niy = residuals * Ni
        if precomputed is None:
            NiF = F_noise * Ni[:, None]
            FtNiF = F_noise.T @ NiF
        FtNiM = F_noise.T @ NiM
        MtNiM = M_timing.T @ NiM
        FtNiy = F_noise.T @ Niy
        MtNiy = M_timing.T @ Niy

    # Σ = φ^{-1} + F^T N^{-1} F (factorized once, reused from precomputed)
    if precomputed is None:
        Sigma = FtNiF.copy()
        Sigma[np.diag_indices(n_noise)] += phiinv_noise
        try:
            L_sigma = _scipy_linalg.cho_factor(Sigma, lower=True, check_finite=False)
        except _scipy_linalg.LinAlgError:
            Sigma[np.diag_indices(n_noise)] += 1e-10 * np.max(np.diag(Sigma))
            L_sigma = _scipy_linalg.cho_factor(Sigma, lower=True, check_finite=False)

    # Woodbury terms
    B = _scipy_linalg.cho_solve(L_sigma, FtNiM, check_finite=False)
    c = _scipy_linalg.cho_solve(L_sigma, FtNiy, check_finite=False)

    # GLS timing normal equations
    MtCiM = MtNiM - FtNiM.T @ B
    MtCiy = MtNiy - FtNiM.T @ c

    # Symmetric normalization for conditioning
    d = np.sqrt(np.maximum(np.diag(MtCiM), 0.0))
    d = np.where(d == 0, 1.0, d)
    MtCiM_n = MtCiM / d[:, None] / d[None, :]
    MtCiy_n = MtCiy / d

    # Solve normalized timing system
    try:
        L_t = _scipy_linalg.cho_factor(MtCiM_n, lower=True, check_finite=False)
        dp_n = _scipy_linalg.cho_solve(L_t, MtCiy_n, check_finite=False)
        cov_n = _scipy_linalg.cho_solve(L_t, np.eye(n_tcols), check_finite=False)
    except _scipy_linalg.LinAlgError:
        U, S, VT = _scipy_linalg.svd(MtCiM_n, full_matrices=False)
        thresh = 1e-14 * S[0] * n_tcols
        S_inv = np.where(S > thresh, 1.0 / S, 0.0)
        dp_n = VT.T @ (S_inv * (U.T @ MtCiy_n))
        cov_n = (VT.T * S_inv) @ VT

    # Unnormalize
    delta_timing = dp_n / d
    cov_timing = cov_n / d[:, None] / d[None, :]

    # Noise coefficients via Wiener filter
    r_post = residuals - M_timing @ delta_timing
    if use_gpu:
        FtNi_rpost = np.asarray(F_j.T @ jnp.asarray(r_post * Ni))
    else:
        FtNi_rpost = F_noise.T @ (r_post * Ni)
    noise_coeffs = _scipy_linalg.cho_solve(L_sigma, FtNi_rpost, check_finite=False)

    # Build output
    t0 = 1 if has_offset else 0
    delta_params = delta_timing[t0:]
    cov = cov_timing[t0:, t0:]
    delta_params_all = np.concatenate([delta_timing, noise_coeffs])

    n_all = n_tcols + n_noise
    cov_all = np.zeros((n_all, n_all))
    cov_all[:n_tcols, :n_tcols] = cov_timing

    if compute_noise_cov:
        try:
            L_S = _scipy_linalg.cho_factor(MtCiM, lower=True, check_finite=False)
            B_hat = _scipy_linalg.cho_solve(L_S, B.T, check_finite=False)
            noise_cov_correction_diag = np.sum(B * B_hat.T, axis=1)
            SigmaInv_diag = _scipy_linalg.cho_solve(
                L_sigma, np.eye(n_noise), check_finite=False
            ).diagonal()
            noise_cov_diag = SigmaInv_diag + noise_cov_correction_diag
            np.fill_diagonal(cov_all[n_tcols:, n_tcols:], noise_cov_diag)
        except _scipy_linalg.LinAlgError:
            pass

    return delta_params, cov, delta_params_all, cov_all, noise_coeffs


def _compute_FtNiF_sparse(F_noise, Ni, blocks):
    """Compute F^T N^{-1} F exploiting block sparsity structure.

    For ECORR (disjoint 0/1 indicators), the self-block is diagonal.
    For masked blocks (band/group noise), only TOAs within the mask contribute.
    Cross-blocks between non-overlapping masks are zero and skipped.
    """
    n_noise = F_noise.shape[1]
    FtNiF = np.zeros((n_noise, n_noise))

    for i, bi in enumerate(blocks):
        si, ni = bi['offset'], bi['ncols']
        Fi = F_noise[:, si:si+ni]

        # Self-block
        if bi['type'] == 'ecorr':
            # ECORR is 0/1 indicators with disjoint groups → diagonal
            for k, idx in enumerate(bi['group_indices']):
                FtNiF[si+k, si+k] = np.sum(Ni[idx])
        elif bi['type'] == 'masked':
            m = bi['mask']
            Fi_m = Fi[m]
            Ni_m = Ni[m]
            FtNiF[si:si+ni, si:si+ni] = Fi_m.T @ (Fi_m * Ni_m[:, None])
        else:  # dense
            FtNiF[si:si+ni, si:si+ni] = Fi.T @ (Fi * Ni[:, None])

        # Cross-blocks with subsequent blocks
        for j in range(i+1, len(blocks)):
            bj = blocks[j]
            sj, nj = bj['offset'], bj['ncols']
            Fj = F_noise[:, sj:sj+nj]

            # Determine overlapping TOA mask
            if bi['type'] == 'masked' and bj['type'] == 'masked':
                overlap = bi['mask'] & bj['mask']
                if not np.any(overlap):
                    continue
                cross = Fi[overlap].T @ (Fj[overlap] * Ni[overlap, None])
            elif bi['type'] == 'masked':
                m = bi['mask']
                cross = Fi[m].T @ (Fj[m] * Ni[m, None])
            elif bj['type'] == 'masked':
                m = bj['mask']
                cross = Fi[m].T @ (Fj[m] * Ni[m, None])
            elif bi['type'] == 'ecorr':
                # ECORR cross dense/masked: sum within each group
                cross = np.zeros((ni, nj))
                for k, idx in enumerate(bi['group_indices']):
                    cross[k, :] = Ni[idx] @ Fj[idx]
            elif bj['type'] == 'ecorr':
                cross = np.zeros((ni, nj))
                for k, idx in enumerate(bj['group_indices']):
                    cross[:, k] = Fi[idx].T @ Ni[idx]
            else:
                cross = Fi.T @ (Fj * Ni[:, None])

            FtNiF[si:si+ni, sj:sj+nj] = cross
            FtNiF[sj:sj+nj, si:si+ni] = cross.T

    return FtNiF


def _precompute_woodbury_products(F_noise, sigma, phiinv_noise, noise_block_info=None):
    """Precompute constant Woodbury products for reuse across iterations.

    Since F_noise (noise basis) and sigma (TOA errors) don't change between
    iterations, FtNiF and L_sigma can be computed once.

    Parameters
    ----------
    F_noise : (n_toa, n_noise) noise basis matrix
    sigma : (n_toa,) TOA uncertainties
    phiinv_noise : (n_noise,) inverse prior variance
    noise_block_info : list of dict, optional
        Block structure for sparse-aware FtNiF computation. Each dict has:
        - 'offset': column offset in F_noise
        - 'ncols': number of columns
        - 'type': 'dense', 'ecorr', or 'masked'
        - 'mask': (n_toa,) bool array for 'masked' type
        - 'group_indices': list of arrays for 'ecorr' type

    Returns
    -------
    dict with keys: Ni, FtNiF, L_sigma, F_j (GPU), Ni_j (GPU)
    """
    n_noise = F_noise.shape[1]
    Ni = 1.0 / (sigma ** 2)

    if noise_block_info is not None:
        FtNiF = _compute_FtNiF_sparse(F_noise, Ni, noise_block_info)
    elif _has_gpu():
        F_j = jnp.asarray(F_noise)
        Ni_j = jnp.asarray(Ni)
        NiF_j = F_j * Ni_j[:, None]
        FtNiF = np.asarray(F_j.T @ NiF_j)
    else:
        NiF = F_noise * Ni[:, None]
        FtNiF = F_noise.T @ NiF

    # GPU-resident copies for per-iteration products
    if _has_gpu():
        F_j = jnp.asarray(F_noise)
        Ni_j = jnp.asarray(Ni)
    else:
        F_j = None
        Ni_j = None

    Sigma = FtNiF.copy()
    Sigma[np.diag_indices(n_noise)] += phiinv_noise
    try:
        L_sigma = _scipy_linalg.cho_factor(Sigma, lower=True, check_finite=False)
    except _scipy_linalg.LinAlgError:
        Sigma[np.diag_indices(n_noise)] += 1e-10 * np.max(np.diag(Sigma))
        L_sigma = _scipy_linalg.cho_factor(Sigma, lower=True, check_finite=False)

    return {
        'Ni': Ni, 'FtNiF': FtNiF, 'L_sigma': L_sigma,
        'F_j': F_j, 'Ni_j': Ni_j,
    }


def fit_parameters_optimized(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    max_iter: int = 100,
    convergence_threshold: float = 1e-14,
    clock_dir: str | None = None,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Dict:
    """
    Fit timing model parameters to TOA data.

    This is the file-based entry point. For repeated fits with the same
    pulsar (e.g. in the GUI), use :func:`fit_parameters_optimized_cached`
    which reuses precomputed setup arrays.

    Handles any combination of spin, DM, astrometry, binary, FD, solar-wind,
    JUMP, and DMX parameters.

    Parameters
    ----------
    par_file : Path
        Path to .par file
    tim_file : Path
        Path to .tim file
    fit_params : list of str
        Parameters to fit (e.g., ['F0', 'F1', 'RAJ', 'DECJ'])
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Convergence threshold on parameter changes
    clock_dir : str or None
        Path to clock correction files. If None, uses the
        data/clock directory in the JUG package installation.
    verbose : bool
        Print progress to stdout
    device : str, optional
        Device preference: 'cpu', 'gpu', or 'auto'.
        If None, uses global preference (default: 'cpu').

    Returns
    -------
    result : dict
        - 'final_params': Fitted parameter dict
        - 'uncertainties': Parameter uncertainties
        - 'final_rms': Final RMS in microseconds
        - 'iterations': Number of iterations
        - 'converged': Whether fit converged
        - 'total_time': Total fitting time
        - 'covariance': Covariance matrix

    Examples
    --------
    >>> result = fit_parameters_optimized(
    ...     par_file=Path("J1909.par"),
    ...     tim_file=Path("J1909.tim"),
    ...     fit_params=['F0', 'F1']
    ... )
    >>> print(f"F0 = {result['final_params']['F0']:.15f} Hz")
    """
    # Set default clock directory relative to package installation
    if clock_dir is None:
        module_dir = Path(__file__).parent
        clock_dir = str(module_dir.parent.parent / "data" / "clock")

    return _fit_parameters_general(
        par_file, tim_file, fit_params, max_iter, convergence_threshold,
        clock_dir, verbose, device
    )



def _build_jump_masks(
    params: Dict,
    toa_flags: List[Dict[str, str]],
    jump_params: List[str],
) -> Dict[str, np.ndarray]:
    """Build boolean masks for JUMP parameters from _jump_lines.

    Each JUMP line has the form:
        JUMP -<flag> <value> <offset> [fit_flag] [uncertainty]
        JUMP MJD <mjd1> <mjd2> <offset> [fit_flag]

    Returns a dict mapping JUMP parameter names (JUMP1, JUMP2, ...) to boolean
    masks of shape (n_toas,).
    """
    from jug.noise.white import build_backend_mask
    jump_lines = params.get('_jump_lines', [])
    n_toas = len(toa_flags)
    masks: Dict[str, np.ndarray] = {}

    for idx, raw_line in enumerate(jump_lines):
        jump_name = f'JUMP{idx + 1}'
        if jump_name not in jump_params:
            continue

        parts = raw_line.strip().split()
        # parts[0] = 'JUMP'
        if len(parts) < 3:
            continue

        if parts[1].upper() == 'MJD':
            # MJD range: JUMP MJD <start> <end> <value> ...
            if len(parts) < 5:
                continue
            mjd_start = float(parts[2])
            mjd_end = float(parts[3])
            toas_mjd = np.array([f.get('__mjd', 0.0) for f in toa_flags])
            # Need actual MJDs -- fall back to all-ones if not available
            masks[jump_name] = np.ones(n_toas, dtype=bool)
        else:
            # Flag-based: JUMP -<flag_name> <flag_value> <value> ...
            flag_name = parts[1].lstrip('-')
            flag_value = parts[2]
            masks[jump_name] = build_backend_mask(toa_flags, flag_name, flag_value)

    # Ensure all requested JUMP params have a mask (all-ones fallback)
    for jp in jump_params:
        if jp not in masks:
            masks[jp] = np.ones(n_toas, dtype=bool)

    return masks


def _build_setup_common(
    params: Dict[str, Any],
    fit_params: List[str],
    toas_mjd: np.ndarray,
    errors_us: np.ndarray,
    toa_flags: Optional[List[Dict[str, str]]],
    dt_sec_cached: np.ndarray,
    dt_sec_ld: Optional[np.ndarray],
    tdb_mjd: np.ndarray,
    freq_mhz_bary: np.ndarray,
    extras: Dict[str, Any],
    noise_config: object,
    verbose: bool = False,
    subtract_noise_sec: Optional[np.ndarray] = None,
) -> GeneralFitSetup:
    """Shared setup builder for both file-based and cache-based paths.

    Parameters
    ----------
    params : dict
        Par file parameters (RAJ/DECJ already converted to radians).
    fit_params : list of str
        Already canonicalized and validated parameter names.
    toas_mjd, errors_us, toa_flags : arrays
        TOA data (already masked if applicable).
    dt_sec_cached, dt_sec_ld, tdb_mjd, freq_mhz_bary : arrays
        Precomputed timing arrays (already masked).
    extras : dict
        Data-source-specific arrays: ``prebinary_delay_sec``,
        ``roemer_shapiro_sec``, ``ssb_obs_pos_ls``, ``sw_geometry_pc``.
    noise_config : NoiseConfig
    verbose : bool
    subtract_noise_sec : ndarray of float, optional
        Per-TOA noise realization (in seconds) to subtract from dt_sec_cached
        before fitting. This implements the Tempo2-style workflow where noise
        is subtracted from the data and then refit without that noise process.
    """
    # --- White noise scaling (EFAC/EQUAD) and ECORR whitener ---------------
    ecorr_whitener = None
    noise_entries = None
    noise_lines = params.get('_noise_lines')
    if noise_lines and toa_flags is not None:
        from jug.noise.white import parse_noise_lines, apply_white_noise
        
        # No more unsupported noise keywords to warn about
        noise_entries = parse_noise_lines(noise_lines)
        if noise_entries:
            active_entries = [
                e for e in noise_entries
                if (e.kind == 'EFAC' and noise_config.is_enabled("EFAC"))
                or (e.kind == 'EQUAD' and noise_config.is_enabled("EQUAD"))
                or (e.kind == 'ECORR' and noise_config.is_enabled("ECORR"))
            ]
            if active_entries:
                errors_us = apply_white_noise(errors_us, toa_flags, active_entries)
            # Build ECORR whitener if ECORR is enabled
            if noise_config.is_enabled("ECORR"):
                ecorr_entries = [e for e in noise_entries if e.kind == 'ECORR']
                if ecorr_entries:
                    from jug.noise.ecorr import build_ecorr_whitener
                    ecorr_whitener = build_ecorr_whitener(
                        toas_mjd, toa_flags, active_entries
                    )
            if verbose:
                efac_count = sum(1 for e in active_entries if e.kind == 'EFAC')
                equad_count = sum(1 for e in active_entries if e.kind == 'EQUAD')
                ecorr_count = sum(1 for e in active_entries if e.kind == 'ECORR')
                ecorr_msg = ''
                if ecorr_count and ecorr_whitener is not None:
                    n_groups = len(ecorr_whitener.epoch_groups)
                    ecorr_msg = f', {ecorr_count} ECORR ({n_groups} epoch groups)'
                print(f"  Applied white noise: {efac_count} EFAC, {equad_count} EQUAD{ecorr_msg}")

    # --- Red noise and DM noise Fourier bases ------------------------------
    red_noise_basis = None
    red_noise_prior = None
    dm_noise_basis = None
    dm_noise_prior = None
    chromatic_noise_basis = None
    chromatic_noise_prior = None
    from jug.noise.red_noise import parse_red_noise_params, parse_dm_noise_params, parse_chromatic_noise_params
    red_noise_proc = parse_red_noise_params(params)
    dm_noise_proc = parse_dm_noise_params(params)
    chromatic_noise_proc = parse_chromatic_noise_params(params)
    
    # Info message if Tempo2-native red noise format detected
    if "RNAMP" in params and "RNIDX" in params and noise_config.is_enabled("RedNoise"):
        if red_noise_proc is not None and verbose:
            print("[SETUP] Converting RNAMP/RNIDX to TNRedAmp/TNRedGam format")

    if red_noise_proc is not None and noise_config.is_enabled("RedNoise"):
        red_noise_basis, red_noise_prior = red_noise_proc.build_basis_and_prior(toas_mjd)
        print(f"[SETUP] Building RED NOISE basis: {red_noise_basis.shape[1]} columns")
        if verbose:
            print(f"  Red noise: log10_A={red_noise_proc.log10_A:.3f}, "
                  f"gamma={red_noise_proc.log10_A:.3f}, "
                  f"{red_noise_proc.n_harmonics} harmonics -> {red_noise_basis.shape[1]} columns")

    if dm_noise_proc is not None and noise_config.is_enabled("DMNoise"):
        dm_noise_basis, dm_noise_prior = dm_noise_proc.build_basis_and_prior(
            toas_mjd, freq_mhz_bary
        )
        print(f"[SETUP] Building DM NOISE basis: {dm_noise_basis.shape[1]} columns")
        if verbose:
            print(f"  DM noise: log10_A={dm_noise_proc.log10_A:.3f}, "
                  f"gamma={dm_noise_proc.gamma:.3f}, "
                  f"{dm_noise_proc.n_harmonics} harmonics -> {dm_noise_basis.shape[1]} columns")

    if chromatic_noise_proc is not None and noise_config.is_enabled("ChromaticNoise"):        
        chromatic_noise_basis, chromatic_noise_prior = chromatic_noise_proc.build_basis_and_prior(
            toas_mjd, freq_mhz_bary
        )
        print(f"[SETUP] Building CHROMATIC NOISE basis: {chromatic_noise_basis.shape[1]} columns")
        if verbose:
            print(f"  Chromatic noise: log10_A={chromatic_noise_proc.log10_A:.3f}, "
                  f"gamma={chromatic_noise_proc.gamma:.3f}, "
                  f"chrom_idx={chromatic_noise_proc.chrom_idx:.3f}, "
                  f"{chromatic_noise_proc.n_harmonics} harmonics -> {chromatic_noise_basis.shape[1]} columns")

    # --- Band noise and Group noise Fourier bases ----------------------------
    band_noise_bases = None
    band_noise_priors = None
    band_noise_labels = None
    group_noise_bases = None
    group_noise_priors = None
    group_noise_labels = None
    from jug.noise.red_noise import parse_band_noise_params, parse_group_noise_params
    if noise_config.is_enabled("BandNoise"):
        band_procs = parse_band_noise_params(params)
        if band_procs:
            band_noise_bases = []
            band_noise_priors = []
            band_noise_labels = []
            for bp in band_procs:
                F, phi = bp.build_basis_and_prior(toas_mjd, freq_mhz_bary)
                band_noise_bases.append(F)
                band_noise_priors.append(phi)
                label = f"BandNoise_{int(bp.freq_lo)}_{int(bp.freq_hi)}"
                band_noise_labels.append(label)
                n_cols = F.shape[1]
                print(f"[SETUP] Building BAND NOISE basis ({label}): {n_cols} columns")

    if noise_config.is_enabled("GroupNoise"):
        group_procs = parse_group_noise_params(params)
        if group_procs:
            group_flags = None
            if toa_flags is not None:
                group_flags = np.array([f.get('group', '') for f in toa_flags])
            group_noise_bases = []
            group_noise_priors = []
            group_noise_labels = []
            for gp in group_procs:
                F, phi = gp.build_basis_and_prior(toas_mjd, group_flags=group_flags)
                group_noise_bases.append(F)
                group_noise_priors.append(phi)
                label = f"GroupNoise_{gp.group_name}"
                group_noise_labels.append(label)
                n_cols = F.shape[1]
                print(f"[SETUP] Building GROUP NOISE basis ({label}): {n_cols} columns")

    # --- DMX design matrix -------------------------------------------------
    dmx_design_matrix = None
    dmx_labels = None
    from jug.model.dmx import parse_dmx_ranges, build_dmx_design_matrix
    dmx_ranges = parse_dmx_ranges(params)
    if dmx_ranges:
        dmx_design_matrix, dmx_labels = build_dmx_design_matrix(
            toas_mjd, freq_mhz_bary, dmx_ranges
        )
        if verbose:
            print(f"  DMX: {len(dmx_ranges)} ranges -> {dmx_design_matrix.shape[1]} columns")
        
        # Apply DMEFAC scaling to DMX design matrix
        if noise_lines and toa_flags is not None:
            from jug.noise.white import parse_noise_lines, build_backend_mask
            dmefac_lines = [l for l in noise_lines if l.split()[0].upper() == 'DMEFAC']
            if dmefac_lines:
                # Parse DMEFAC entries
                dmefac_entries = parse_noise_lines(dmefac_lines)
                dmefac_entries = [e for e in dmefac_entries if e.kind == 'DMEFAC']
                
                if dmefac_entries:
                    # Build per-TOA DMEFAC array (default 1.0)
                    dmefac_array = np.ones(len(toas_mjd), dtype=np.float64)
                    for entry in dmefac_entries:
                        mask = build_backend_mask(toa_flags, entry.flag_name, entry.flag_value)
                        dmefac_array[mask] = entry.value
                    
                    # Scale DMX design matrix rows by dividing by DMEFAC
                    # This reduces DM precision for backends with DMEFAC > 1
                    dmx_design_matrix = dmx_design_matrix / dmefac_array[:, np.newaxis]
                    
                    if verbose:
                        n_scaled = np.sum(dmefac_array != 1.0)
                        print(f"  DMEFAC: Applied scaling to {n_scaled}/{len(toas_mjd)} TOAs "
                              f"({len(dmefac_entries)} backend groups)")

    # --- DMJUMP design matrix ----------------------------------------------
    dmjump_design_matrix = None
    dmjump_labels = None
    if noise_lines and toa_flags is not None:
        from jug.noise.white import build_backend_mask
        
        dmjump_lines = [l for l in noise_lines if l.split()[0].upper() == 'DMJUMP']
        if dmjump_lines:
            # Parse DMJUMP lines: DMJUMP -fe <flag_value> <initial_value>
            dmjump_specs = []
            for line in dmjump_lines:
                parts = line.split()
                if len(parts) >= 4:
                    flag_name = parts[1].lstrip('-')
                    flag_value = parts[2]
                    try:
                        initial_dm_offset = float(parts[3])
                    except ValueError:
                        continue
                    dmjump_specs.append((flag_name, flag_value, initial_dm_offset))
            
            if dmjump_specs:
                # Build DMJUMP design matrix columns
                n_dmjumps = len(dmjump_specs)
                dmjump_design_matrix = np.zeros((len(toas_mjd), n_dmjumps), dtype=np.float64)
                dmjump_labels = []
                
                # DM delay derivative: K_DM / freq^2 (same as DMX columns)
                dm_deriv = K_DM_SEC / (freq_mhz_bary ** 2)
                
                for i, (flag_name, flag_value, initial_offset) in enumerate(dmjump_specs):
                    mask = build_backend_mask(toa_flags, flag_name, flag_value)
                    dmjump_design_matrix[mask, i] = dm_deriv[mask]
                    dmjump_labels.append(f"DMJUMP_{flag_name}_{flag_value}")
                
                if verbose:
                    print(f"  DMJUMP: {n_dmjumps} DM offsets -> {n_dmjumps} columns")

    # --- ECORR GLS basis (alternative to whitener) -------------------------
    ecorr_basis = None
    ecorr_prior = None
    if noise_config.is_enabled("ECORR") and noise_lines and toa_flags is not None:
        from jug.noise.ecorr import build_ecorr_basis_and_prior
        _entries = noise_entries if noise_entries is not None else parse_noise_lines(noise_lines)
        result_ecorr = build_ecorr_basis_and_prior(toas_mjd, toa_flags, _entries)
        if result_ecorr is not None:
            ecorr_basis, ecorr_prior = result_ecorr
            ecorr_whitener = None  # Avoid double-counting
            if verbose:
                print(f"  ECORR: {ecorr_basis.shape[1]} epoch groups in GLS basis")

    # --- Auto-include fit-flagged JUMPs when correlated noise is active ------
    # Without JUMP columns in the GLS design matrix, the Fourier noise basis
    # absorbs per-backend offsets, inflating the noise realization.
    has_correlated_noise = (red_noise_basis is not None or dm_noise_basis is not None
                            or chromatic_noise_basis is not None or ecorr_basis is not None)
    if has_correlated_noise and toa_flags is not None:
        from jug.model.parameter_spec import is_jump_param
        existing_jumps = {p for p in fit_params if is_jump_param(p)}
        jump_lines = params.get('_jump_lines', [])
        added_jumps = []
        for idx, jl in enumerate(jump_lines):
            jname = f'JUMP{idx + 1}'
            if jname in existing_jumps:
                continue
            jparts = jl.strip().split()
            try:
                fit_idx = 5 if jparts[1].upper() == 'MJD' else 4
                if len(jparts) > fit_idx and int(jparts[fit_idx]) == 1:
                    fit_params = list(fit_params) + [jname]
                    added_jumps.append(jname)
            except (IndexError, ValueError):
                pass
        if added_jumps:
            print(f"[SETUP] Auto-added {len(added_jumps)} fit-flagged JUMPs for GLS noise fit")

        # Also auto-add fit-flagged FDJUMPs
        existing_fdjumps = {p for p in fit_params if p.startswith('FDJUMP')}
        fit_flags = params.get('_fit_flags', {})
        added_fdjumps = []
        for key in sorted(params.keys()):
            if key.startswith('FDJUMP') and not key.startswith('_') and key not in existing_fdjumps:
                if fit_flags.get(key):
                    fit_params = list(fit_params) + [key]
                    added_fdjumps.append(key)
        if added_fdjumps:
            print(f"[SETUP] Auto-added {len(added_fdjumps)} fit-flagged FDJUMPs for GLS noise fit")

    # --- Derived weight arrays ---------------------------------------------
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2

    # --- Extract starting parameter values ---------------------------------
    param_values_start = []
    for idx, param in enumerate(fit_params):
        if param not in params:
            # Try canonical name and aliases (e.g., A1DOT <-> XDOT)
            canonical = canonicalize_param_name(param)
            if canonical != param and canonical in params:
                params[param] = params[canonical]
            else:
                # Try all aliases
                from jug.model.parameter_spec import get_spec
                spec = get_spec(param)
                if spec and spec.aliases:
                    for alias in spec.aliases:
                        if alias in params:
                            params[param] = params[alias]
                            break
            if param not in params:
                default_value = _get_param_default_value(param)
                if default_value is None:
                    raise ValueError(f"Parameter {param} not found in .par file and no default available")
                params[param] = default_value
                if verbose:
                    print(f"Warning: {param} not in .par file, using default value: {default_value}")

        value = params[param]
        if param == 'RAJ' and isinstance(value, str):
            from jug.io.par_reader import parse_ra
            value = parse_ra(value)
        elif param == 'DECJ' and isinstance(value, str):
            from jug.io.par_reader import parse_dec
            value = parse_dec(value)
        elif param == 'ELONG':
            value = float(params.get('_ecliptic_lon_deg', params.get('ELONG', 0.0)))
        elif param == 'ELAT':
            value = float(params.get('_ecliptic_lat_deg', params.get('ELAT', 0.0)))
        elif param == 'PMELONG':
            value = float(params.get('_ecliptic_pm_lon', params.get('PMELONG', 0.0)))
        elif param == 'PMELAT':
            value = float(params.get('_ecliptic_pm_lat', params.get('PMELAT', 0.0)))
        elif param == 'SINI' and isinstance(value, str) and value.upper() == 'KIN':
            kin_deg = float(params.get('KIN', 0.0))
            value = float(jnp.sin(jnp.deg2rad(kin_deg)))
        elif isinstance(value, str):
            value = float(value)
        param_values_start.append(value)

    # --- Classify parameters (spec-driven) ---------------------------------
    spin_params = get_spin_params_from_list(fit_params)
    dm_params = get_dm_params_from_list(fit_params)
    binary_params = get_binary_params_from_list(fit_params)
    astrometry_params = get_astrometry_params_from_list(fit_params)
    fd_params = get_fd_params_from_list(fit_params)
    sw_params = get_sw_params_from_list(fit_params)

    # --- JUMP masks --------------------------------------------------------
    from jug.model.parameter_spec import is_jump_param
    jump_params_list = [p for p in fit_params if is_jump_param(p)]
    jump_masks = _build_jump_masks(params, toa_flags, jump_params_list) if jump_params_list and toa_flags else None

    # Drop JUMPs whose mask is empty (no matching TOAs) -- matches Tempo2 behaviour
    if jump_masks:
        empty_jumps = [jp for jp in jump_params_list if not jump_masks.get(jp, np.zeros(1, dtype=bool)).any()]
        if empty_jumps:
            # Build index set of params to keep
            remove_set = set(empty_jumps)
            keep_indices = [i for i, p in enumerate(fit_params) if p not in remove_set]
            fit_params = [fit_params[i] for i in keep_indices]
            param_values_start = [param_values_start[i] for i in keep_indices]
            for jp in empty_jumps:
                jump_masks.pop(jp, None)
            jump_params_list = [p for p in jump_params_list if p not in remove_set]
            if verbose:
                print(f"  Skipped {len(empty_jumps)} JUMPs with no matching TOAs: {empty_jumps}")

    # --- FDJUMP masks ------------------------------------------------------
    fdjump_params_list = [p for p in fit_params if p.startswith('FDJUMP')]
    fdjump_masks = None
    initial_fdjump_delay = None
    if fdjump_params_list and toa_flags:
        from jug.noise.white import build_backend_mask
        fdjump_masks = {}
        for fp in fdjump_params_list:
            meta = params.get(f'_fdjump_meta_{fp}')
            if meta:
                mask = build_backend_mask(toa_flags, meta['flag_name'], meta['flag_value'])
                if mask.any():
                    fdjump_masks[fp] = mask
                else:
                    if verbose:
                        print(f"  Skipped FDJUMP {fp} with no matching TOAs")
        if fdjump_masks:
            from jug.fitting.derivatives_fdjump import compute_fdjump_delay
            initial_fdjump_delay = compute_fdjump_delay(
                params, freq_mhz_bary, fdjump_params_list, fdjump_masks
            )

    # --- DM delay cache ----------------------------------------------------
    initial_dm_delay = None
    if dm_params:
        dm_epoch = params.get('DMEPOCH', params.get('PEPOCH', 55000.0))
        initial_dm_params = {p: params[p] for p in dm_params if p in params}
        initial_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_mhz_bary, initial_dm_params, dm_epoch)

    # --- Observer position (needed for DDK binary and astrometry) ----------
    ssb_obs_pos_ls = extras.get('ssb_obs_pos_ls')
    # Sun position relative to observer (for Shapiro delay recomputation)
    obs_sun_pos_ls = extras.get('obs_sun_pos_ls')
    # Planet positions relative to observer (for planet Shapiro recomputation)
    obs_planet_pos_ls = extras.get('obs_planet_pos_ls')

    # --- Binary delay setup ------------------------------------------------
    roemer_shapiro_sec = extras.get('roemer_shapiro_sec')
    prebinary_delay_sec = extras.get('prebinary_delay_sec')
    initial_binary_delay = None
    if binary_params:
        if prebinary_delay_sec is None:
            if roemer_shapiro_sec is None:
                raise ValueError(
                    "Binary fitting requires prebinary_delay_sec or roemer_shapiro_sec. "
                    "Please update compute_residuals_simple to return 'prebinary_delay_sec'."
                )
            prebinary_delay_sec = roemer_shapiro_sec  # Fallback
        toas_prebinary = tdb_mjd - prebinary_delay_sec / SECS_PER_DAY
        initial_binary_delay = np.array(compute_binary_delay(
            toas_prebinary, params, obs_pos_ls=ssb_obs_pos_ls))

    # --- Astrometry delay setup --------------------------------------------
    initial_astrometric_delay = None
    if astrometry_params:
        if ssb_obs_pos_ls is None:
            raise ValueError("Astrometry fitting requires ssb_obs_pos_ls in compute_residuals output.")
        from jug.fitting.derivatives_astrometry import compute_astrometric_delay
        initial_astrometric_delay = np.array(compute_astrometric_delay(
            params, tdb_mjd, ssb_obs_pos_ls, obs_sun_pos_ls=obs_sun_pos_ls,
            obs_planet_pos_ls=obs_planet_pos_ls))

    # --- FD delay setup ----------------------------------------------------
    initial_fd_delay = None
    if fd_params:
        from jug.fitting.derivatives_fd import compute_fd_delay
        initial_fd_params = {p: params[p] for p in fd_params if p in params}
        initial_fd_delay = np.asarray(compute_fd_delay(freq_mhz_bary, initial_fd_params), dtype=np.float64)

    # --- Solar wind delay setup --------------------------------------------
    sw_geometry_pc = extras.get('sw_geometry_pc')
    initial_sw_delay = None
    if sw_params:
        if sw_geometry_pc is None:
            raise ValueError("NE_SW fitting requires sw_geometry_pc.")
        ne_sw_val = float(params.get('NE_SW', params.get('NE1AU', 0.0)))
        initial_sw_delay = K_DM_SEC * ne_sw_val * sw_geometry_pc / (np.array(freq_mhz_bary) ** 2)

    # --- Subtract noise realization from dt_sec (Tempo2-style workflow) ------
    jump_phase = extras.get('jump_phase')
    tzr_phase = extras.get('tzr_phase')
    if subtract_noise_sec is not None:
        dt_sec_cached = dt_sec_cached - subtract_noise_sec
        if dt_sec_ld is not None:
            dt_sec_ld = dt_sec_ld - np.asarray(subtract_noise_sec, dtype=np.longdouble)
        if verbose:
            print(f"  Applied noise subtraction to dt_sec: "
                  f"RMS correction = {np.std(subtract_noise_sec)*1e6:.3f} mus")

    # --- Assemble GeneralFitSetup ------------------------------------------
    return GeneralFitSetup(
        params=dict(params),
        fit_param_list=fit_params,
        param_values_start=param_values_start,
        toas_mjd=np.array(toas_mjd),
        freq_mhz=np.array(freq_mhz_bary),
        errors_us=np.array(errors_us),
        errors_sec=np.array(errors_sec),
        weights=np.array(weights),
        dt_sec_cached=np.array(dt_sec_cached),
        dt_sec_ld=np.array(dt_sec_ld, dtype=np.longdouble) if dt_sec_ld is not None else None,
        tdb_mjd=np.array(tdb_mjd),
        initial_dm_delay=initial_dm_delay,
        dm_params=dm_params,
        spin_params=spin_params,
        binary_params=binary_params,
        astrometry_params=astrometry_params,
        fd_params=fd_params,
        sw_params=sw_params,
        roemer_shapiro_sec=roemer_shapiro_sec,
        prebinary_delay_sec=prebinary_delay_sec,
        initial_binary_delay=initial_binary_delay,
        ssb_obs_pos_ls=ssb_obs_pos_ls,
        obs_sun_pos_ls=obs_sun_pos_ls,
        obs_planet_pos_ls=obs_planet_pos_ls,
        initial_astrometric_delay=initial_astrometric_delay,
        initial_fd_delay=initial_fd_delay,
        initial_sw_delay=initial_sw_delay,
        sw_geometry_pc=sw_geometry_pc,
        toa_flags=toa_flags,
        ecorr_whitener=ecorr_whitener,
        red_noise_basis=red_noise_basis,
        red_noise_prior=red_noise_prior,
        dm_noise_basis=dm_noise_basis,
        dm_noise_prior=dm_noise_prior,
        chromatic_noise_basis=chromatic_noise_basis,
        chromatic_noise_prior=chromatic_noise_prior,
        ecorr_basis=ecorr_basis,
        ecorr_prior=ecorr_prior,
        band_noise_bases=band_noise_bases,
        band_noise_priors=band_noise_priors,
        band_noise_labels=band_noise_labels,
        group_noise_bases=group_noise_bases,
        group_noise_priors=group_noise_priors,
        group_noise_labels=group_noise_labels,
        dmx_design_matrix=dmx_design_matrix,
        dmx_labels=dmx_labels,
        dmjump_design_matrix=dmjump_design_matrix,
        dmjump_labels=dmjump_labels,
        jump_masks=jump_masks,
        fdjump_masks=fdjump_masks,
        fdjump_params=fdjump_params_list,
        initial_fdjump_delay=initial_fdjump_delay,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
        noise_config=noise_config,
    )


def _build_general_fit_setup_from_files(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    clock_dir: str,
    verbose: bool,
    noise_config: Optional[object] = None
) -> GeneralFitSetup:
    """Build fitting setup from par/tim files (expensive I/O + compute).

    Parses files, computes residuals, then delegates to the shared
    ``_build_setup_common()`` builder for noise wiring and parameter setup.
    """
    # Canonicalize and validate fit_params
    fit_params = [canonicalize_param_name(p) for p in fit_params]
    for p in fit_params:
        validate_fit_param(p)

    # Parse files
    params = parse_par_file(par_file)
    validate_par_timescale(params, context="create_general_fit_setup")
    toas_data = parse_tim_file_mjds(tim_file)

    # Convert RAJ/DECJ from strings to radians
    from jug.io.par_reader import parse_ra, parse_dec
    if 'RAJ' in params and isinstance(params['RAJ'], str):
        params['RAJ'] = parse_ra(params['RAJ'])
    if 'DECJ' in params and isinstance(params['DECJ'], str):
        params['DECJ'] = parse_dec(params['DECJ'])

    # Tempo2's T2 model uses IAU convention for KIN/KOM.
    # JUG's DDK code (from PINT) uses DT92 convention.
    binary = params.get('BINARY', '').upper()
    if binary == 'T2' and ('KIN' in params or 'KOM' in params):
        if 'KIN' in params:
            params['KIN'] = 180.0 - float(params['KIN'])
        if 'KOM' in params:
            params['KOM'] = 90.0 - float(params['KOM'])

    # Extract TOA arrays
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    toa_flags = [toa.flags for toa in toas_data]

    # Build NoiseConfig
    from jug.engine.noise_mode import NoiseConfig
    if noise_config is None:
        noise_config = NoiseConfig.from_par(params)

    # Compute expensive residuals / delays (the I/O-heavy step)
    if verbose:
        print(f"\nCaching expensive delays...")
    result = compute_residuals_simple(
        par_file, tim_file, clock_dir=clock_dir,
        subtract_tzr=False, verbose=False,
    )

    return _build_setup_common(
        params=params,
        fit_params=fit_params,
        toas_mjd=toas_mjd,
        errors_us=errors_us,
        toa_flags=toa_flags,
        dt_sec_cached=result['dt_sec'],
        dt_sec_ld=result.get('dt_sec_ld'),
        tdb_mjd=result['tdb_mjd'],
        freq_mhz_bary=result['freq_bary_mhz'],
        extras={
            'prebinary_delay_sec': result.get('prebinary_delay_sec'),
            'roemer_shapiro_sec': result.get('roemer_shapiro_sec'),
            'ssb_obs_pos_ls': result.get('ssb_obs_pos_ls'),
            'obs_sun_pos_ls': result.get('obs_sun_pos_ls'),
            'obs_planet_pos_ls': result.get('obs_planet_pos_ls'),
            'sw_geometry_pc': result.get('sw_geometry_pc'),
            'jump_phase': result.get('jump_phase'),
        },
        noise_config=noise_config,
        verbose=verbose,
    )


def _compute_full_model_residuals(
    params: Dict,
    setup: GeneralFitSetup,
) -> tuple:
    """
    Compute TRUE residuals using the full nonlinear model.
    
    This function recomputes all delays (DM, binary, astrometric) from the
    current parameter values, ensuring the residuals reflect the actual model.
    This is analogous to PINT's ModelState.resids which recomputes the full model.
    
    Parameters
    ----------
    params : dict
        Current parameter dictionary with updated values
    setup : GeneralFitSetup
        The fitting setup with cached geometry data
        
    Returns
    -------
    residuals_sec : np.ndarray
        Time residuals in seconds
    chi2 : float
        Chi-squared statistic
    rms_us : float
        RMS of residuals in microseconds
    wrms_us : float
        Weighted RMS in microseconds
    """
    # Get cached arrays -- use longdouble dt_sec for phase precision
    from jug.residuals.simple_calculator import compute_phase_residuals
    dt_sec_base = setup.dt_sec_ld if setup.dt_sec_ld is not None else np.array(setup.dt_sec_cached, dtype=np.longdouble)
    tdb_mjd = setup.tdb_mjd
    freq_mhz = setup.freq_mhz
    weights = setup.weights
    errors_sec = setup.errors_sec

    # Start with longdouble dt_sec (contains initial delays)
    dt_sec_np = dt_sec_base.copy()

    # Apply DM delay correction (float64 corrections promoted to longdouble)
    dm_params = setup.dm_params
    if dm_params:
        dm_epoch = params.get('DMEPOCH', params.get('PEPOCH', 55000.0))
        current_dm_params = {p: params[p] for p in dm_params}
        new_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_mhz, current_dm_params, dm_epoch)
        dm_delay_change = new_dm_delay - setup.initial_dm_delay
        dt_sec_np = dt_sec_np - dm_delay_change

    # Apply binary delay correction (route to correct binary model)
    binary_params = setup.binary_params
    if binary_params and setup.initial_binary_delay is not None:
        toas_prebinary = tdb_mjd - setup.prebinary_delay_sec / SECS_PER_DAY
        new_binary_delay = np.array(compute_binary_delay(
            toas_prebinary, params, obs_pos_ls=setup.ssb_obs_pos_ls))
        binary_delay_change = new_binary_delay - setup.initial_binary_delay
        dt_sec_np = dt_sec_np - binary_delay_change

    # Apply astrometric delay correction
    astrometry_params = setup.astrometry_params
    if astrometry_params and setup.initial_astrometric_delay is not None:
        from jug.fitting.derivatives_astrometry import compute_astrometric_delay
        new_astrometric_delay = np.array(compute_astrometric_delay(
            params, tdb_mjd, setup.ssb_obs_pos_ls,
            obs_sun_pos_ls=setup.obs_sun_pos_ls,
            obs_planet_pos_ls=setup.obs_planet_pos_ls
        ))
        astrometric_delay_change = new_astrometric_delay - setup.initial_astrometric_delay
        dt_sec_np = dt_sec_np - astrometric_delay_change

    # Apply FD delay correction
    fd_params = setup.fd_params
    if fd_params and setup.initial_fd_delay is not None:
        from jug.fitting.derivatives_fd import compute_fd_delay
        current_fd_params = {p: params[p] for p in fd_params if p in params}
        new_fd_delay = np.asarray(compute_fd_delay(freq_mhz, current_fd_params), dtype=np.float64)
        fd_delay_change = new_fd_delay - setup.initial_fd_delay
        dt_sec_np = dt_sec_np - fd_delay_change

    # Apply FDJUMP delay correction
    if setup.fdjump_params and setup.fdjump_masks:
        from jug.fitting.derivatives_fdjump import compute_fdjump_delay
        new_fdjump_delay = compute_fdjump_delay(
            params, freq_mhz, setup.fdjump_params, setup.fdjump_masks
        )
        if setup.initial_fdjump_delay is not None:
            fdjump_delay_change = new_fdjump_delay - setup.initial_fdjump_delay
        else:
            fdjump_delay_change = new_fdjump_delay
        dt_sec_np = dt_sec_np - fdjump_delay_change

    # Apply solar wind delay correction
    sw_params = setup.sw_params
    if sw_params and setup.initial_sw_delay is not None:
        ne_sw_val = float(params.get('NE_SW', params.get('NE1AU', 0.0)))
        new_sw_delay = K_DM_SEC * ne_sw_val * setup.sw_geometry_pc / (freq_mhz ** 2)
        sw_delay_change = new_sw_delay - setup.initial_sw_delay
        dt_sec_np = dt_sec_np - sw_delay_change

    # Update jump_phase for fitted JUMP parameters (JUMPs are phase offsets, not delays)
    current_jump_phase = setup.jump_phase
    if setup.jump_masks:
        jump_params_list = [p for p in (setup.fit_param_list or []) if is_jump_param(p)]
        if jump_params_list:
            from jug.io.par_reader import get_longdouble
            base = setup.jump_phase if setup.jump_phase is not None else np.zeros(len(tdb_mjd), dtype=np.longdouble)
            current_jump_phase = np.array(base, dtype=np.longdouble).copy()
            F0_ld = get_longdouble(params, 'F0')
            for jp in jump_params_list:
                mask = setup.jump_masks.get(jp)
                if mask is not None:
                    initial_val = setup.param_values_start[setup.fit_param_list.index(jp)]
                    current_val = params.get(jp, 0.0)
                    current_jump_phase[mask] += F0_ld * np.longdouble(current_val - initial_val)

    # Phase computation via shared canonical function (longdouble precision)
    residuals_us, residuals_sec, _ = compute_phase_residuals(
        dt_sec_np, params, weights, subtract_mean=True,
        tzr_phase=setup.tzr_phase,
        jump_phase=current_jump_phase
    )

    # Compute statistics
    sum_weights = np.sum(weights)
    # Chi2: use block-diagonal C^{-1} when ECORR whitener is available
    ecorr_w = getattr(setup, 'ecorr_whitener', None)
    if ecorr_w is not None:
        ecorr_w.prepare(errors_sec)
        chi2 = ecorr_w.chi2(residuals_sec)
    else:
        chi2 = np.sum((residuals_sec / errors_sec) ** 2)
    rms_us = np.sqrt(np.sum(residuals_sec**2 * weights) / sum_weights) * 1e6
    wrms_us = np.sqrt(np.sum((residuals_sec * 1e6)**2 * weights) / sum_weights)
    
    return residuals_sec, chi2, rms_us, wrms_us


def _run_general_fit_iterations(
    setup: GeneralFitSetup,
    max_iter: int,
    convergence_threshold: float,
    verbose: bool,
    solver_mode: str = "exact"
) -> Dict:
    """
    Run general fitting iterations using precomputed setup.

    This is the "iterate" phase that reuses cached arrays.
    
    Uses PINT-style damping: each WLS step is validated against the full 
    nonlinear model, and step size is reduced if chi2 worsens.

    Parameters
    ----------
    setup : GeneralFitSetup
        Precomputed setup data
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Convergence threshold
    verbose : bool
        Print progress
    solver_mode : str, default "exact"
        Solver mode: "exact" (SVD, bit-for-bit reproducible) or
        "fast" (QR/lstsq, faster but may differ slightly).

    Returns
    -------
    result : dict
        Fit results
    """
    # Normalize solver_mode
    solver_mode = solver_mode.lower().strip() if solver_mode else "exact"
    if solver_mode not in ("exact", "fast"):
        solver_mode = "exact"
    
    # Unpack setup
    params = setup.params
    fit_params = setup.fit_param_list
    param_values_start = setup.param_values_start
    toas_mjd = setup.toas_mjd
    freq_mhz = setup.freq_mhz
    errors_us = setup.errors_us
    errors_sec = setup.errors_sec
    weights = setup.weights
    dt_sec_cached = setup.dt_sec_ld if setup.dt_sec_ld is not None else np.array(setup.dt_sec_cached, dtype=np.longdouble)
    tdb_mjd = setup.tdb_mjd
    initial_dm_delay = setup.initial_dm_delay
    dm_params = setup.dm_params
    spin_params = setup.spin_params
    binary_params = setup.binary_params
    initial_binary_delay = setup.initial_binary_delay
    prebinary_delay_sec = setup.prebinary_delay_sec
    astrometry_params = setup.astrometry_params
    initial_astrometric_delay = setup.initial_astrometric_delay
    ssb_obs_pos_ls = setup.ssb_obs_pos_ls
    fd_params = setup.fd_params
    initial_fd_delay = setup.initial_fd_delay
    sw_params_iter = setup.sw_params
    initial_sw_delay = setup.initial_sw_delay
    sw_geometry_pc = setup.sw_geometry_pc
    jump_masks = setup.jump_masks
    jump_phase_arr = setup.jump_phase

    # Precompute PINT-compatible pre-binary TOAs for binary fitting (only if needed)
    toas_prebinary_for_binary = None
    if binary_params and prebinary_delay_sec is not None:
        toas_prebinary_for_binary = tdb_mjd - prebinary_delay_sec / SECS_PER_DAY
    
    # Initialize iteration
    param_values_curr = param_values_start.copy()
    iteration = 0
    converged = False

    # Pre-compute sum of weights ONCE outside iteration loop (performance optimization)
    # This is mathematically identical - weights array doesn't change during fitting
    sum_weights = np.sum(weights)

    # Pre-compute param list categorization ONCE (these never change during fitting)
    spin_params_list = get_spin_params_from_list(fit_params)
    dm_params_list = get_dm_params_from_list(fit_params)
    binary_params_list = get_binary_params_from_list(fit_params)
    astrometry_params_list = get_astrometry_params_from_list(fit_params)
    fd_params_list = get_fd_params_from_list(fit_params)
    sw_params_list = get_sw_params_from_list(fit_params)
    jump_params_list = [p for p in fit_params if is_jump_param(p)]

    # Pre-import derivative modules once (avoid repeated import lookups in hot loop)
    from jug.fitting.derivatives_spin import compute_spin_derivatives
    from jug.fitting.derivatives_astrometry import (
        compute_astrometric_delay, compute_astrometry_derivatives
    )
    from jug.fitting.derivatives_fd import compute_fd_delay, compute_fd_derivatives
    from jug.fitting.derivatives_sw import compute_sw_derivatives
    from jug.residuals.simple_calculator import compute_phase_residuals
    from jug.io.par_reader import get_longdouble

    # Convergence criteria
    xtol = 1e-12
    min_iterations = 5
    
    # Compute initial full-model chi2 for comparison
    for i, param in enumerate(fit_params):
        _update_param(params, param, param_values_curr[i])
    _, current_chi2, current_rms_us, _ = _compute_full_model_residuals(params, setup)
    best_chi2 = current_chi2
    best_param_values = param_values_curr.copy()
    best_nonlinear_rms = current_rms_us
    best_noise_coeffs = None  # Fourier/DMX coefficients from augmented solve
    # Save solver data for linear postfit (needed for augmented fits)
    _saved_residuals_sec = None
    _saved_M = None
    _saved_delta_all = None
    _saved_lambda = 1.0
    
    # Track RMS history (using full-model RMS)
    rms_history = [current_rms_us]
    # Accumulated noise signal for TNsubtractPoly (Tempo2 algorithm):
    # Subtract previously estimated noise from residuals before GLS so
    # the noise model only estimates the DELTA noise each iteration.
    _accumulated_noise_sec = np.zeros(len(toas_mjd))
    # Track red and DM noise signals separately for polynomial subtraction
    _accumulated_red_noise_sec = np.zeros(len(toas_mjd))
    _accumulated_dm_noise_sec = np.zeros(len(toas_mjd))
    _accumulated_other_noise_sec = np.zeros(len(toas_mjd))
    _best_ns_rms = None  # best noise-subtracted RMS for GLS damping

    # Precompute noise column counts (constant across iterations)
    n_red_noise_cols = setup.red_noise_basis.shape[1] if getattr(setup, 'red_noise_basis', None) is not None else 0
    n_dm_noise_cols = setup.dm_noise_basis.shape[1] if getattr(setup, 'dm_noise_basis', None) is not None else 0
    n_chromatic_noise_cols = setup.chromatic_noise_basis.shape[1] if getattr(setup, 'chromatic_noise_basis', None) is not None else 0
    n_ecorr_cols = setup.ecorr_basis.shape[1] if getattr(setup, 'ecorr_basis', None) is not None else 0
    n_dmx_cols = setup.dmx_design_matrix.shape[1] if getattr(setup, 'dmx_design_matrix', None) is not None else 0
    n_dmjump_cols = setup.dmjump_design_matrix.shape[1] if getattr(setup, 'dmjump_design_matrix', None) is not None else 0
    n_band_noise_cols_list = []
    n_band_noise_total = 0
    if getattr(setup, 'band_noise_bases', None):
        for F in setup.band_noise_bases:
            n = F.shape[1]
            n_band_noise_cols_list.append(n)
            n_band_noise_total += n
    n_group_noise_cols_list = []
    n_group_noise_total = 0
    if getattr(setup, 'group_noise_bases', None):
        for F in setup.group_noise_bases:
            n = F.shape[1]
            n_group_noise_cols_list.append(n)
            n_group_noise_total += n
    n_augmented = (n_red_noise_cols + n_dm_noise_cols + n_chromatic_noise_cols
                   + n_ecorr_cols + n_dmx_cols + n_dmjump_cols
                   + n_band_noise_total + n_group_noise_total)

    # Precompute GLS Woodbury products once (noise bases and errors are constant)
    _woodbury_precomputed = None
    _gls_phiinv_raw = None
    _gls_F_noise = None
    if n_augmented > 0:
        # Build phiinv
        _gls_phiinv_raw = np.zeros(n_augmented)
        offset_p = 0
        if n_red_noise_cols > 0:
            _gls_phiinv_raw[offset_p:offset_p + n_red_noise_cols] = 1.0 / setup.red_noise_prior
            offset_p += n_red_noise_cols
        if n_dm_noise_cols > 0:
            _gls_phiinv_raw[offset_p:offset_p + n_dm_noise_cols] = 1.0 / setup.dm_noise_prior
            offset_p += n_dm_noise_cols
        if n_chromatic_noise_cols > 0:
            _gls_phiinv_raw[offset_p:offset_p + n_chromatic_noise_cols] = 1.0 / setup.chromatic_noise_prior
            offset_p += n_chromatic_noise_cols
        if n_ecorr_cols > 0:
            _gls_phiinv_raw[offset_p:offset_p + n_ecorr_cols] = 1.0 / setup.ecorr_prior
            offset_p += n_ecorr_cols
        if n_band_noise_total > 0:
            for bi, bp in enumerate(setup.band_noise_priors):
                nc = n_band_noise_cols_list[bi]
                _gls_phiinv_raw[offset_p:offset_p + nc] = 1.0 / bp
                offset_p += nc
        if n_group_noise_total > 0:
            for gi, gp in enumerate(setup.group_noise_priors):
                nc = n_group_noise_cols_list[gi]
                _gls_phiinv_raw[offset_p:offset_p + nc] = 1.0 / gp
                offset_p += nc
        if n_dmx_cols + n_dmjump_cols > 0:
            _gls_phiinv_raw[offset_p:] = 1e-40

        # Build concatenated noise basis and block info for sparse FtNiF
        noise_parts = []
        noise_block_info = []
        col_off = 0
        n_toa = len(toas_mjd)
        for label, basis, btype in [
            ('red', setup.red_noise_basis, 'dense'),
            ('dm', setup.dm_noise_basis, 'dense'),
            ('chrom', setup.chromatic_noise_basis, 'dense'),
        ]:
            if basis is not None and basis.shape[1] > 0:
                nc = basis.shape[1]
                noise_parts.append(basis)
                noise_block_info.append({'offset': col_off, 'ncols': nc, 'type': btype})
                col_off += nc
        if setup.ecorr_basis is not None and n_ecorr_cols > 0:
            E = setup.ecorr_basis
            noise_parts.append(E)
            # Extract group indices for sparse ECORR computation
            group_indices = []
            for k in range(n_ecorr_cols):
                group_indices.append(np.nonzero(E[:, k])[0])
            noise_block_info.append({
                'offset': col_off, 'ncols': n_ecorr_cols,
                'type': 'ecorr', 'group_indices': group_indices
            })
            col_off += n_ecorr_cols
        if n_band_noise_total > 0:
            for F in setup.band_noise_bases:
                nc = F.shape[1]
                mask = np.any(F != 0, axis=1)
                noise_parts.append(F)
                noise_block_info.append({
                    'offset': col_off, 'ncols': nc,
                    'type': 'masked', 'mask': mask
                })
                col_off += nc
        if n_group_noise_total > 0:
            for F in setup.group_noise_bases:
                nc = F.shape[1]
                mask = np.any(F != 0, axis=1)
                noise_parts.append(F)
                noise_block_info.append({
                    'offset': col_off, 'ncols': nc,
                    'type': 'masked', 'mask': mask
                })
                col_off += nc
        if n_dmx_cols > 0:
            noise_parts.append(setup.dmx_design_matrix)
            noise_block_info.append({
                'offset': col_off, 'ncols': n_dmx_cols, 'type': 'dense'
            })
            col_off += n_dmx_cols
        if n_dmjump_cols > 0:
            noise_parts.append(setup.dmjump_design_matrix)
            noise_block_info.append({
                'offset': col_off, 'ncols': n_dmjump_cols, 'type': 'dense'
            })
            col_off += n_dmjump_cols

        _gls_F_noise = np.hstack(noise_parts)
        _woodbury_precomputed = _precompute_woodbury_products(
            _gls_F_noise, errors_sec, _gls_phiinv_raw,
            noise_block_info=noise_block_info
        )

    if verbose:
        print(f"\n{'Iter':<6} {'RMS (mus)':<12} {'DeltaParam':<15} {'lambda_':<8} {'Status':<20}")
        print("-" * 75)
    
    # ITERATION LOOP (Tempo2-style: full step + re-baseline)
    for iteration in range(max_iter):
        # Update params dict with current values
        for i, param in enumerate(fit_params):
            _update_param(params, param, param_values_curr[i])
        
        # Build design matrix using linearized residuals
        # (We compute residuals for the design matrix using cached delays for spin/DM/binary,
        # but we validate the full step against the true model)
        dt_sec_np = dt_sec_cached.copy()

        # If fitting DM parameters, update dt_sec with new DM delay
        if dm_params:
            dm_epoch = params.get('DMEPOCH', params.get('PEPOCH', 55000.0))
            current_dm_params = {p: params[p] for p in dm_params}
            new_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_mhz, current_dm_params, dm_epoch)
            dm_delay_change = new_dm_delay - initial_dm_delay
            dt_sec_np = dt_sec_np - dm_delay_change

        # If fitting binary parameters, update dt_sec with new binary delay
        # Use PINT-compatible pre-binary time (roemer_shapiro + DM + SW + tropo)
        if binary_params and initial_binary_delay is not None:
            new_binary_delay = np.array(compute_binary_delay(
                toas_prebinary_for_binary, params, obs_pos_ls=ssb_obs_pos_ls))
            binary_delay_change = new_binary_delay - initial_binary_delay
            dt_sec_np = dt_sec_np - binary_delay_change

        # If fitting astrometric parameters, update dt_sec with new astrometric delay
        if astrometry_params and initial_astrometric_delay is not None:
            new_astrometric_delay = np.array(compute_astrometric_delay(
                params, tdb_mjd, ssb_obs_pos_ls,
                obs_sun_pos_ls=setup.obs_sun_pos_ls,
                obs_planet_pos_ls=setup.obs_planet_pos_ls
            ))
            astrometric_delay_change = new_astrometric_delay - initial_astrometric_delay
            dt_sec_np = dt_sec_np - astrometric_delay_change

        # If fitting FD parameters, update dt_sec with new FD delay
        if fd_params and initial_fd_delay is not None:
            current_fd_params = {p: params[p] for p in fd_params if p in params}
            new_fd_delay = np.asarray(compute_fd_delay(freq_mhz, current_fd_params), dtype=np.float64)
            fd_delay_change = new_fd_delay - initial_fd_delay
            dt_sec_np = dt_sec_np - fd_delay_change

        # If fitting FDJUMP parameters, update dt_sec with new FDJUMP delay
        fdjump_params_iter = setup.fdjump_params
        fdjump_masks_iter = setup.fdjump_masks
        if fdjump_params_iter and fdjump_masks_iter:
            from jug.fitting.derivatives_fdjump import compute_fdjump_delay
            new_fdjump_delay = compute_fdjump_delay(
                params, freq_mhz, fdjump_params_iter, fdjump_masks_iter
            )
            initial_fdj = setup.initial_fdjump_delay
            if initial_fdj is not None:
                fdjump_delay_change = new_fdjump_delay - initial_fdj
            else:
                fdjump_delay_change = new_fdjump_delay
            dt_sec_np = dt_sec_np - fdjump_delay_change

        # If fitting SW parameters, update dt_sec with new solar wind delay
        if sw_params_iter and initial_sw_delay is not None:
            ne_sw_val = float(params.get('NE_SW', params.get('NE1AU', 0.0)))
            new_sw_delay = K_DM_SEC * ne_sw_val * sw_geometry_pc / (freq_mhz ** 2)
            sw_delay_change = new_sw_delay - initial_sw_delay
            dt_sec_np = dt_sec_np - sw_delay_change

        # Update jump_phase for fitted JUMP parameters (JUMPs are phase offsets, not delays)
        current_jump_phase = jump_phase_arr
        if jump_params_list and jump_masks:
            base = jump_phase_arr if jump_phase_arr is not None else np.zeros(len(toas_mjd), dtype=np.longdouble)
            current_jump_phase = np.array(base, dtype=np.longdouble).copy()
            F0_ld = get_longdouble(params, 'F0')
            for jp in jump_params_list:
                mask = jump_masks.get(jp)
                if mask is not None:
                    initial_val = param_values_start[fit_params.index(jp)]
                    current_val = params.get(jp, 0.0)
                    current_jump_phase[mask] += F0_ld * np.longdouble(current_val - initial_val)

        # Compute phase residuals from updated dt_sec via shared function (longdouble)
        _, residuals, _ = compute_phase_residuals(
            dt_sec_np, params, weights, subtract_mean=True,
            tzr_phase=setup.tzr_phase,
            jump_phase=current_jump_phase
        )

        # TNsubtractPoly: subtract accumulated noise from previous iterations
        # so GLS only estimates the delta-noise this iteration
        if n_augmented > 0 and np.any(_accumulated_noise_sec != 0):
            residuals = residuals - _accumulated_noise_sec

        # Build design matrix - BATCHED derivative computation
        M_columns = []

        # Batch spin parameters (spec-driven routing via component)
        spin_derivs = {}
        if spin_params_list:
            spin_derivs = compute_spin_derivatives(params, toas_mjd, spin_params_list)

        # Batch DM parameters (spec-driven routing via component)
        dm_derivs = {}
        if dm_params_list:
            dm_derivs = compute_dm_derivatives(params, toas_mjd, freq_mhz, dm_params_list)

        # Batch binary parameters (routed via binary_registry)
        # Use PINT-compatible pre-binary time (roemer_shapiro + DM + SW + tropo)
        binary_derivs = {}
        if binary_params_list:
            if setup.prebinary_delay_sec is None:
                raise ValueError(
                    "Binary fitting requires prebinary_delay_sec in setup. "
                    "Ensure compute_residuals_simple returns 'prebinary_delay_sec'."
                )
            toas_prebinary_mjd = toas_mjd - setup.prebinary_delay_sec / SECS_PER_DAY
            # Pass obs_pos_ls for DDK Kopeikin parallax corrections
            binary_derivs = compute_binary_derivatives(
                params, toas_prebinary_mjd, binary_params_list, 
                obs_pos_ls=setup.ssb_obs_pos_ls
            )

        # Batch astrometry parameters (spec-driven routing via component)
        astrometry_derivs = {}
        if astrometry_params_list:
            if setup.ssb_obs_pos_ls is None:
                raise ValueError(
                    "Astrometry fitting requires ssb_obs_pos_ls in setup. "
                    "Ensure compute_residuals_simple returns 'ssb_obs_pos_ls'."
                )
            astrometry_derivs = compute_astrometry_derivatives(
                params, toas_mjd, setup.ssb_obs_pos_ls, astrometry_params_list
            )

        # Batch FD parameters (frequency-dependent delay derivatives)
        fd_derivs = {}
        if fd_params_list:
            fd_derivs = compute_fd_derivatives(params, freq_mhz, fd_params_list)

        # Batch solar wind parameters (NE_SW)
        sw_derivs = {}
        if sw_params_list:
            if setup.sw_geometry_pc is None:
                raise ValueError(
                    "NE_SW fitting requires sw_geometry_pc in setup."
                )
            sw_derivs = compute_sw_derivatives(setup.sw_geometry_pc, freq_mhz, sw_params_list)

        # JUMP parameters: derivative is the mask (1 where JUMP applies, 0 elsewhere)
        # Convention: JUG subtracts JUMP from dt_sec, so M_JUMP = -d(r)/d(JUMP) = +1
        jump_derivs = {}
        if jump_params_list and jump_masks:
            for jp in jump_params_list:
                mask = jump_masks.get(jp)
                if mask is not None:
                    jump_derivs[jp] = mask.astype(np.float64)

        # FDJUMP parameters: frequency-dependent jumps
        fdjump_derivs = {}
        fdjump_params_list = setup.fdjump_params
        fdjump_masks = setup.fdjump_masks
        if fdjump_params_list and fdjump_masks:
            from jug.fitting.derivatives_fdjump import compute_fdjump_derivatives
            fdjump_derivs = compute_fdjump_derivatives(
                params, freq_mhz, fdjump_params_list,
                fdjump_masks=fdjump_masks,
            )

        # Merge all derivative dicts into one lookup table
        all_derivs = {}
        all_derivs.update(spin_derivs)
        all_derivs.update(dm_derivs)
        all_derivs.update(binary_derivs)
        all_derivs.update(astrometry_derivs)
        all_derivs.update(fd_derivs)
        all_derivs.update(sw_derivs)
        all_derivs.update(jump_derivs)
        all_derivs.update(fdjump_derivs)

        # Assemble columns in original fit_params order
        for param in fit_params:
            if param not in all_derivs:
                raise ValueError(
                    f"No derivative computed for parameter '{param}'. "
                    f"Check that it is registered in parameter_spec.py and has a derivative function."
                )
            M_columns.append(all_derivs[param])
        
        # Column counts already precomputed outside loop
        n_timing_params = len(M_columns)
        has_offset = n_augmented > 0
        total_cols = (1 if has_offset else 0) + n_timing_params + n_augmented

        # Build design matrix: timing columns + precomputed noise basis
        n_timing_cols = (1 if has_offset else 0) + n_timing_params
        M = np.empty((len(toas_mjd), total_cols), dtype=np.float64)
        col = 0
        if has_offset:
            M[:, 0] = 1.0
            col = 1
        for i, mc in enumerate(M_columns):
            M[:, col + i] = np.asarray(mc, dtype=np.float64)
        col += n_timing_params
        if n_augmented > 0:
            M[:, col:] = _gls_F_noise

        # Mean subtraction for WLS (non-augmented) path
        if not has_offset:
            col_means = (weights @ M) / sum_weights
            M -= col_means[np.newaxis, :]
        
        # Solve WLS to get step direction
        # When ECORR whitener is present AND no noise augmentation, pre-whiten
        # residuals and M so the standard WLS solver recovers the GLS solution.
        # When augmented (GLS), skip pre-whitening -- the ECORR basis columns
        # in the augmented system already account for ECORR correlations.
        ecorr_w = getattr(setup, 'ecorr_whitener', None)
        if ecorr_w is not None and n_augmented == 0:
            ecorr_w.prepare(errors_sec)
            r_solve = ecorr_w.whiten_residuals(residuals)
            M_solve = ecorr_w.whiten_matrix(M)
            sigma_solve = np.ones_like(errors_sec)  # already whitened
        else:
            r_solve = residuals
            M_solve = M
            sigma_solve = errors_sec

        _iter_noise_coeffs = None  # noise Fourier/DMX coefficients this iteration
        _wls_delta_all = None  # full delta including offset for WLS linearized RMS

        if n_augmented > 0:
            # GLS path: Woodbury solver with precomputed products
            M_t = M_solve[:, :n_timing_cols]
            F_n = M_solve[:, n_timing_cols:]
            is_final = (iteration == max_iter - 1)
            delta_params, cov, delta_params_all, cov_all, _iter_noise_coeffs = \
                _solve_gls_woodbury(M_t, F_n, r_solve, sigma_solve,
                                    _gls_phiinv_raw, has_offset, n_timing_params,
                                    precomputed=_woodbury_precomputed,
                                    compute_noise_cov=is_final)
        elif solver_mode == "fast":
            # WLS FAST solver: QR-based lstsq with proper conditioning
            r1 = r_solve / sigma_solve
            M1 = M_solve / sigma_solve[:, None]
            col_norms = np.asarray(jnp.sqrt(jnp.sum(jnp.array(M1)**2, axis=0)))
            col_norms = np.where(col_norms == 0, 1.0, col_norms)
            M2 = M1 / col_norms[None, :]
            delta_normalized = np.asarray(jnp.linalg.lstsq(jnp.array(M2), jnp.array(r1), rcond=None)[0])
            delta_all = delta_normalized / col_norms
            M2tM2 = M2.T @ M2
            M2tM2_j = jnp.array(M2tM2)
            try:
                cov_normalized = np.asarray(jnp.linalg.inv(M2tM2_j))
            except Exception:
                cov_normalized = np.asarray(jnp.linalg.pinv(M2tM2_j))
            cov_all = (cov_normalized / col_norms).T / col_norms
            # Strip offset column from delta_params and cov
            t0 = 1 if has_offset else 0
            delta_params = delta_all[t0:]
            cov = cov_all[t0:, t0:]
            _wls_delta_all = delta_all
        else:
            # WLS EXACT solver: SVD-based (bit-for-bit reproducible)
            delta_all, cov_all, _ = wls_solve_svd(
                residuals=r_solve,
                sigma=sigma_solve,
                M=M_solve,
                threshold=1e-14,
                negate_dpars=False
            )
            delta_all = np.array(delta_all)
            cov_all = np.array(cov_all)
            # Strip offset column
            t0 = 1 if has_offset else 0
            delta_params = delta_all[t0:]
            cov = cov_all[t0:, t0:]
            _wls_delta_all = delta_all
        
        # Step acceptance strategy depends on fit type:
        # - WLS (n_augmented == 0): Damping loop with raw RMS comparison.
        # - GLS (n_augmented > 0): Damping loop using noise-subtracted RMS.
        #   The raw RMS is dominated by correlated noise and barely changes
        #   between iterations, causing over-damping. By subtracting the
        #   Wiener-filter noise estimate, we expose the true timing improvement.
        lambda_ = 1.0
        step_accepted = False
        chi2_decrease = 0
        
        # NaN/Inf guard
        trial_param_values = [
            param_values_curr[i] + delta_params[i]
            for i in range(len(fit_params))
        ]
        if any(np.isnan(v) or np.isinf(v) for v in trial_param_values):
            if verbose:
                print(f"         (step rejected: NaN/Inf in parameters)")
            param_values_curr = list(best_param_values)
            converged = True
            break

        if n_augmented > 0:
            # GLS path: damping with noise-subtracted RMS comparison
            lambda_ = 1.0
            for damping_iter in range(8):
                trial_param_values = [
                    param_values_curr[i] + lambda_ * delta_params[i]
                    for i in range(len(fit_params))
                ]
                for i, p in enumerate(fit_params):
                    _update_param(params, p, trial_param_values[i])
                nl_resid_sec, trial_nl_chi2, trial_nl_rms, _ = _compute_full_model_residuals(params, setup)

                # Compute noise-subtracted RMS for this trial
                # Re-solve noise coefficients via Wiener filter for trial residuals
                trial_noise_signal = np.zeros_like(nl_resid_sec)
                if _iter_noise_coeffs is not None:
                    trial_noise_signal = _gls_F_noise @ (lambda_ * _iter_noise_coeffs)
                trial_clean = nl_resid_sec - _accumulated_noise_sec - trial_noise_signal
                trial_ns_rms = np.sqrt(np.mean(trial_clean**2)) * 1e6

                # Compare noise-subtracted RMS (initialise reference on first iteration)
                if _best_ns_rms is None or (iteration == 1 and damping_iter == 0):
                    _best_ns_rms = trial_ns_rms + 1.0  # ensure first step accepted

                if trial_ns_rms < _best_ns_rms * (1.0 + 1e-6):
                    step_accepted = True
                    _best_ns_rms = min(_best_ns_rms, trial_ns_rms)
                    param_values_curr = trial_param_values
                    best_param_values = trial_param_values.copy()
                    best_nonlinear_rms = trial_nl_rms
                    current_rms_us = trial_nl_rms
                    current_chi2 = trial_nl_chi2
                    best_chi2 = trial_nl_chi2

                    _saved_residuals_sec = residuals.copy()
                    _saved_M = M.copy()
                    _saved_delta_all = (lambda_ * delta_params_all).copy()
                    _saved_cov_all = cov_all
                    if _iter_noise_coeffs is not None:
                        best_noise_coeffs = _iter_noise_coeffs.copy()
                    _saved_lambda = lambda_
                    break
                lambda_ *= 0.5
        else:
            # WLS path: damping loop — try full step, halve if nonlinear model worsens.
            lambda_ = 1.0
            for damping_iter in range(8):
                trial_param_values = [
                    param_values_curr[i] + lambda_ * delta_params[i]
                    for i in range(len(fit_params))
                ]
                for i, p in enumerate(fit_params):
                    _update_param(params, p, trial_param_values[i])
                _, trial_nl_chi2, trial_nl_rms, _ = _compute_full_model_residuals(params, setup)

                if trial_nl_rms < best_nonlinear_rms * (1.0 + 1e-6):
                    step_accepted = True
                    param_values_curr = trial_param_values
                    best_param_values = trial_param_values.copy()
                    best_nonlinear_rms = min(best_nonlinear_rms, trial_nl_rms)
                    current_rms_us = trial_nl_rms
                    current_chi2 = trial_nl_chi2
                    best_chi2 = trial_nl_chi2

                    _saved_residuals_sec = residuals.copy()
                    _saved_M = M.copy()
                    _saved_delta_all = (lambda_ * _wls_delta_all).copy()
                    _saved_lambda = lambda_
                    break
                lambda_ *= 0.5

        if not step_accepted:
            # Restore best params
            param_values_curr = list(best_param_values)
            for i, p in enumerate(fit_params):
                _update_param(params, p, param_values_curr[i])
            if iteration >= min_iterations:
                if verbose:
                    print(f"         (step rejected: nonlinear RMS {trial_nl_rms:.3f} > best {best_nonlinear_rms:.3f}, converged)")
                converged = True
                break
            else:
                if verbose:
                    print(f"         (step rejected: nonlinear RMS {trial_nl_rms:.3f} > best {best_nonlinear_rms:.3f}, continuing...)")

        # Re-baseline after accepted step: recompute all fitted delays at
        # accepted params and update dt_sec_cached + initial_*_delay so the
        # next iteration computes only SMALL delay changes.
        if step_accepted:
            if dm_params:
                dm_epoch = params.get('DMEPOCH', params.get('PEPOCH', 55000.0))
                accepted_dm = compute_dm_delay_fast(tdb_mjd, freq_mhz,
                    {p: params[p] for p in dm_params}, dm_epoch)
                dt_sec_cached = dt_sec_cached - (accepted_dm - initial_dm_delay)
                initial_dm_delay = accepted_dm
                setup.initial_dm_delay = accepted_dm

            if binary_params and initial_binary_delay is not None:
                accepted_binary = np.array(compute_binary_delay(
                    toas_prebinary_for_binary, params, obs_pos_ls=ssb_obs_pos_ls))
                dt_sec_cached = dt_sec_cached - (accepted_binary - initial_binary_delay)
                initial_binary_delay = accepted_binary
                setup.initial_binary_delay = accepted_binary

            if astrometry_params and initial_astrometric_delay is not None:
                accepted_astro = np.array(compute_astrometric_delay(
                    params, tdb_mjd, ssb_obs_pos_ls,
                    obs_sun_pos_ls=setup.obs_sun_pos_ls,
                    obs_planet_pos_ls=setup.obs_planet_pos_ls))
                dt_sec_cached = dt_sec_cached - (accepted_astro - initial_astrometric_delay)
                initial_astrometric_delay = accepted_astro
                setup.initial_astrometric_delay = accepted_astro

            if fd_params and initial_fd_delay is not None:
                accepted_fd = np.asarray(compute_fd_delay(freq_mhz,
                    {p: params[p] for p in fd_params if p in params}), dtype=np.float64)
                dt_sec_cached = dt_sec_cached - (accepted_fd - initial_fd_delay)
                initial_fd_delay = accepted_fd
                setup.initial_fd_delay = accepted_fd

            if sw_params_iter and initial_sw_delay is not None:
                ne_sw_val = float(params.get('NE_SW', params.get('NE1AU', 0.0)))
                accepted_sw = K_DM_SEC * ne_sw_val * sw_geometry_pc / (freq_mhz ** 2)
                dt_sec_cached = dt_sec_cached - (accepted_sw - initial_sw_delay)
                initial_sw_delay = accepted_sw
                setup.initial_sw_delay = accepted_sw

            # Update setup's dt_sec baseline so _compute_full_model_residuals
            # evaluates from the re-baselined state (not original params).
            if setup.dt_sec_ld is not None:
                setup.dt_sec_ld = dt_sec_cached.copy()
            setup.dt_sec_cached = np.float64(dt_sec_cached).copy()

            # Re-evaluate nonlinear RMS with fresh baselines
            nl_resid_sec, current_chi2, current_rms_us, _ = _compute_full_model_residuals(params, setup)
            best_nonlinear_rms = current_rms_us
            best_chi2 = current_chi2

            # ----- Accumulated noise tracking + TNsubtractPoly -----
            # Track the noise contribution and remove polynomial component.
            # Tempo2 applies TNsubtractPoly once after convergence, not every
            # iteration.  We track noise each iteration but only decompose
            # the polynomial at the end (see post-convergence block below).
            if n_augmented > 0 and _iter_noise_coeffs is not None:
                # Split noise coefficients into red and DM components
                damped_coeffs = lambda_ * _iter_noise_coeffs
                coeff_offset = 0
                if n_red_noise_cols > 0:
                    red_coeffs = damped_coeffs[coeff_offset:coeff_offset + n_red_noise_cols]
                    _accumulated_red_noise_sec += setup.red_noise_basis @ red_coeffs
                    coeff_offset += n_red_noise_cols
                if n_dm_noise_cols > 0:
                    dm_coeffs = damped_coeffs[coeff_offset:coeff_offset + n_dm_noise_cols]
                    _accumulated_dm_noise_sec += setup.dm_noise_basis @ dm_coeffs
                    coeff_offset += n_dm_noise_cols

                # Rebuild combined accumulated noise for subtraction
                _accumulated_noise_sec = (
                    _accumulated_red_noise_sec.copy()
                    + _accumulated_dm_noise_sec.copy()
                    + _accumulated_other_noise_sec.copy()
                )
                # Accumulate remaining noise components (chromatic, ecorr, etc.)
                if coeff_offset < len(damped_coeffs):
                    other_delta = (
                        _gls_F_noise[:, coeff_offset:] @ damped_coeffs[coeff_offset:]
                    )
                    _accumulated_other_noise_sec += other_delta
                    _accumulated_noise_sec += other_delta
        
        # Track RMS history (full-model RMS)
        rms_history.append(current_rms_us)

        # Check convergence based on parameter changes becoming small
        param_norm = np.linalg.norm(param_values_curr)
        delta_norm = np.linalg.norm(delta_params)
        param_converged = delta_norm <= xtol * (param_norm + xtol)
        
        # Also check RMS convergence (if RMS change is tiny, we've converged)
        rms_converged = False
        if len(rms_history) >= 2:
            rms_change = abs(rms_history[-1] - rms_history[-2])
            rms_converged = rms_change < 1e-8  # Less than 0.01 ns change
        
        converged = iteration >= min_iterations and (param_converged or rms_converged)

        if verbose:
            status = ""
            if converged:
                if param_converged:
                    status = "[x] Params converged"
                elif rms_converged:
                    status = "[x] RMS stable"
            max_delta = np.max(np.abs(delta_params))
            print(f"{iteration+1:<6} {current_rms_us:>11.6f}  {max_delta:>13.6e}  1.0      {status:<20}")

        if converged:
            break

    # Use best parameters found
    for i, param in enumerate(fit_params):
        _update_param(params, param, best_param_values[i])
        param_values_curr[i] = best_param_values[i]

    # ----- TNsubtractPoly (post-convergence) -----
    # Tempo2 algorithm: fit timing-model polynomials to the accumulated
    # noise signals and transfer the polynomial component into timing
    # parameters.  This prevents the correlated-noise model from absorbing
    # low-order polynomial timing signal (F0/F1 in red noise, DM/DM1/DM2
    # in DM noise).  Applied once after convergence, matching Tempo2.
    if n_augmented > 0 and (np.any(_accumulated_red_noise_sec != 0)
                            or np.any(_accumulated_dm_noise_sec != 0)):

        # Red noise → F0/F1 + offset
        _tn_spin_fit = [p for p in ['F0', 'F1'] if p in fit_params]
        if n_red_noise_cols > 0 and _tn_spin_fit:
            _tn_spin_derivs = compute_spin_derivatives(
                params, toas_mjd, _tn_spin_fit)
            _tn_cols = [_tn_spin_derivs[p] for p in _tn_spin_fit]
            _tn_cols.append(np.ones(len(toas_mjd)))  # constant offset
            M_poly = np.column_stack(_tn_cols)
            w = 1.0 / errors_sec
            Mw = M_poly * w[:, None]
            yw = _accumulated_red_noise_sec * w
            try:
                dp = np.linalg.lstsq(Mw, yw, rcond=None)[0]
                poly_signal = M_poly @ dp
                _accumulated_red_noise_sec -= poly_signal
                for ci, sp in enumerate(_tn_spin_fit):
                    pi = fit_params.index(sp)
                    param_values_curr[pi] += dp[ci]
                    _update_param(params, sp, param_values_curr[pi])
                    best_param_values[pi] = param_values_curr[pi]
                if verbose:
                    dp_strs = [f"{sp}={dp[ci]:+.6e}" for ci, sp in enumerate(_tn_spin_fit)]
                    print(f"TNsubtractPoly (red): {', '.join(dp_strs)}")
            except Exception:
                pass

        # DM noise → DM/DM1/DM2
        _tn_dm_fit = [p for p in ['DM', 'DM1', 'DM2'] if p in fit_params]
        if n_dm_noise_cols > 0 and _tn_dm_fit:
            _tn_dm_derivs = compute_dm_derivatives(
                params, toas_mjd, freq_mhz, _tn_dm_fit)
            _tn_dm_cols = [_tn_dm_derivs[p] for p in _tn_dm_fit]
            M_dm_poly = np.column_stack(_tn_dm_cols)
            w = 1.0 / errors_sec
            Mw = M_dm_poly * w[:, None]
            yw = _accumulated_dm_noise_sec * w
            try:
                dp = np.linalg.lstsq(Mw, yw, rcond=None)[0]
                poly_signal = M_dm_poly @ dp
                _accumulated_dm_noise_sec -= poly_signal
                for ci, dp_name in enumerate(_tn_dm_fit):
                    pi = fit_params.index(dp_name)
                    param_values_curr[pi] += dp[ci]
                    _update_param(params, dp_name, param_values_curr[pi])
                    best_param_values[pi] = param_values_curr[pi]
                if verbose:
                    dp_strs = [f"{dp_name}={dp[ci]:+.6e}" for ci, dp_name in enumerate(_tn_dm_fit)]
                    print(f"TNsubtractPoly (DM):  {', '.join(dp_strs)}")
            except Exception:
                pass

        # Update accumulated noise after polynomial subtraction
        _accumulated_noise_sec = (
            _accumulated_red_noise_sec + _accumulated_dm_noise_sec
            + _accumulated_other_noise_sec
        )
    
    # Compute final residuals.
    # For augmented fits (DMX/noise basis columns), use LINEAR postfit residuals
    # (r_pre - M @ delta). The nonlinear recompute + separate DMX subtraction
    # introduces an offset mismatch because the solver jointly optimizes timing,
    # DMX, and an offset column, but the split approach doesn't correctly account
    # for the joint offset. The linear postfit is exact for single-iteration WLS
    # and matches PINT's behavior.
    # For WLS fits, the delta-based nonlinear evaluation accumulates cross-term
    # errors over iterations; the linearized postfit from the final iteration is
    # more accurate (matching Tempo2's approach).
    if _saved_residuals_sec is not None:
        if n_augmented > 0:
            # GLS: Subtract only the timing model correction (timing params + offset
            # + DMX + DMJUMP). All noise realizations (Red, DM, Chromatic, ECORR,
            # Band, Group) are left in the residuals for GUI subtract workflow.
            delta_model_only = _saved_delta_all.copy()
            noise_start = n_timing_cols
            noise_end = (n_timing_cols + n_red_noise_cols + n_dm_noise_cols
                         + n_chromatic_noise_cols + n_ecorr_cols)
            delta_model_only[noise_start:noise_end] = 0.0
            # Band and group noise columns sit after DMX/DMJUMP
            bg_start = noise_end + n_dmx_cols + n_dmjump_cols
            bg_end = bg_start + n_band_noise_total + n_group_noise_total
            delta_model_only[bg_start:bg_end] = 0.0
            linear_correction = _saved_M @ delta_model_only
        else:
            # WLS: Subtract full linearized correction
            linear_correction = _saved_M @ _saved_delta_all
        residuals_final_sec = _saved_residuals_sec - linear_correction
        residuals_final_us = residuals_final_sec * 1e6
    else:
        residuals_final_sec, final_chi2, final_rms_us, final_wrms_us = _compute_full_model_residuals(params, setup)
        residuals_final_us = residuals_final_sec * 1e6
    
    # Compute prefit residuals
    for i, param in enumerate(fit_params):
        _update_param(params, param, param_values_start[i])
    
    residuals_prefit_sec, _, prefit_rms_us, _ = _compute_full_model_residuals(params, setup)
    residuals_prefit_us = residuals_prefit_sec * 1e6
    
    # Restore final parameter values
    for i, param in enumerate(fit_params):
        _update_param(params, param, param_values_curr[i])
    
    # Compute uncertainties
    uncertainties = {param: np.sqrt(cov[i, i]) for i, param in enumerate(fit_params)}

    # Use noise coefficients from the joint GLS solve.
    # After damped iterations, the noise coefficients may correspond to a
    # different timing solution. Re-solve for optimal noise at the final
    # nonlinear residuals using the Woodbury identity (noise-only Wiener filter).
    # This is appropriate post-convergence because the timing model is fixed.
    noise_realizations = {}
    if n_augmented > 0:
        # Re-solve noise at the final nonlinear residuals
        nl_resid_sec, _, _, _ = _compute_full_model_residuals(params, setup)

        # Collect noise basis and prior
        noise_bases = []
        noise_labels = []
        noise_ncols = []
        offset = 0
        if n_red_noise_cols > 0 and setup.red_noise_basis is not None:
            noise_bases.append(setup.red_noise_basis)
            noise_labels.append(('RedNoise', n_red_noise_cols))
            noise_ncols.append(n_red_noise_cols)
        if n_dm_noise_cols > 0 and setup.dm_noise_basis is not None:
            noise_bases.append(setup.dm_noise_basis)
            noise_labels.append(('DMNoise', n_dm_noise_cols))
            noise_ncols.append(n_dm_noise_cols)
        if n_chromatic_noise_cols > 0 and setup.chromatic_noise_basis is not None:
            noise_bases.append(setup.chromatic_noise_basis)
            noise_labels.append(('ChromaticNoise', n_chromatic_noise_cols))
            noise_ncols.append(n_chromatic_noise_cols)
        if n_ecorr_cols > 0 and setup.ecorr_basis is not None:
            noise_bases.append(setup.ecorr_basis)
            noise_labels.append(('ECORR', n_ecorr_cols))
            noise_ncols.append(n_ecorr_cols)
        if n_band_noise_total > 0 and setup.band_noise_bases is not None:
            for bi, F in enumerate(setup.band_noise_bases):
                noise_bases.append(F)
                noise_labels.append((setup.band_noise_labels[bi], n_band_noise_cols_list[bi]))
                noise_ncols.append(n_band_noise_cols_list[bi])
        if n_group_noise_total > 0 and setup.group_noise_bases is not None:
            for gi, F in enumerate(setup.group_noise_bases):
                noise_bases.append(F)
                noise_labels.append((setup.group_noise_labels[gi], n_group_noise_cols_list[gi]))
                noise_ncols.append(n_group_noise_cols_list[gi])
        if n_dmx_cols > 0 and setup.dmx_design_matrix is not None:
            noise_bases.append(setup.dmx_design_matrix)
            noise_labels.append(('DMX', n_dmx_cols))
            noise_ncols.append(n_dmx_cols)
        if n_dmjump_cols > 0 and setup.dmjump_design_matrix is not None:
            noise_bases.append(setup.dmjump_design_matrix)
            noise_labels.append(('DMJUMP', n_dmjump_cols))
            noise_ncols.append(n_dmjump_cols)

        if noise_bases:
            F_all = np.hstack(noise_bases)
            # Solve: c = (Phi^{-1} + F^T N^{-1} F)^{-1} F^T N^{-1} r
            Ninv = 1.0 / errors_sec**2
            FtNi = F_all.T * Ninv[np.newaxis, :]
            FtNiF = FtNi @ F_all
            FtNiF[np.diag_indices_from(FtNiF)] += _gls_phiinv_raw
            try:
                L = _scipy_linalg.cho_factor(FtNiF)
                optimal_noise_coeffs = _scipy_linalg.cho_solve(L, FtNi @ nl_resid_sec)
                C_post = _scipy_linalg.cho_solve(L, np.eye(FtNiF.shape[0]))
            except Exception:
                optimal_noise_coeffs = best_noise_coeffs if best_noise_coeffs is not None else np.zeros(n_augmented)
                C_post = None

            # Build noise realizations from re-solved coefficients
            offset = 0
            for label, nc in noise_labels:
                coeffs = optimal_noise_coeffs[offset:offset + nc]
                idx = [i for i, (l, _) in enumerate(noise_labels) if l == label][0]
                F = noise_bases[idx]
                noise_realizations[label] = (F @ coeffs) * 1e6
                if C_post is not None:
                    C_block = C_post[offset:offset + nc, offset:offset + nc]
                    noise_realizations[f'{label}_err'] = np.sqrt(np.sum((F @ C_block) * F, axis=1)) * 1e6

                # DMX/DMJUMP: subtract from residuals (timing model, not noise)
                if label == 'DMX':
                    if _saved_residuals_sec is None:
                        residuals_final_sec = residuals_final_sec - F @ coeffs
                        residuals_final_us = residuals_final_sec * 1e6
                elif label == 'DMJUMP':
                    if _saved_residuals_sec is None:
                        residuals_final_sec = residuals_final_sec - F @ coeffs
                        residuals_final_us = residuals_final_sec * 1e6
                offset += nc

    # Recompute final stats with DMX-absorbed residuals
    # Re-subtract weighted mean (DMX absorption shifts the mean)
    weights = setup.weights
    sum_weights = np.sum(weights)
    wmean = np.sum(weights * residuals_final_sec) / sum_weights
    residuals_final_sec = residuals_final_sec - wmean
    residuals_final_us = residuals_final_sec * 1e6
    ecorr_w = getattr(setup, 'ecorr_whitener', None)
    if ecorr_w is not None:
        ecorr_w.prepare(errors_sec)
        final_chi2 = ecorr_w.chi2(residuals_final_sec)
    else:
        final_chi2 = np.sum((residuals_final_sec / errors_sec) ** 2)
    final_rms_us = np.sqrt(np.sum(residuals_final_sec**2 * weights) / sum_weights) * 1e6
    final_wrms_us = np.sqrt(np.sum((residuals_final_sec * 1e6)**2 * weights) / sum_weights)

    # Always compute full nonlinear RMS for accurate reporting.
    # The linearized residuals are kept for display but final_rms should
    # reflect the actual timing model accuracy.
    nl_residuals_sec, nl_chi2, nl_rms_us, nl_wrms_us = _compute_full_model_residuals(params, setup)
    final_rms_us = nl_rms_us

    # For GLS fits, compute chi2 from noise-subtracted nonlinear residuals.
    # The raw nonlinear chi2 ignores the correlated noise model, giving
    # artificially high chi2/dof. Tempo2 reports chi2 after subtracting
    # the noise realization (red noise + DM noise Fourier components).
    if n_augmented > 0 and noise_realizations:
        nl_residuals_us = nl_residuals_sec * 1e6
        noise_total_us = np.zeros(len(nl_residuals_us))
        for key, vals in noise_realizations.items():
            if not key.endswith('_err') and len(vals) == len(noise_total_us):
                noise_total_us += vals
        whitened_us = nl_residuals_us - noise_total_us
        whitened_sec = whitened_us * 1e-6
        if ecorr_w is not None:
            ecorr_w.prepare(errors_sec)
            final_chi2 = ecorr_w.chi2(whitened_sec)
        else:
            final_chi2 = np.sum((whitened_sec / errors_sec) ** 2)
        # Return raw (non-noise-subtracted) nonlinear residuals.
        # Users can subtract noise_realizations themselves if desired.
        residuals_final_us = nl_residuals_us
    else:
        final_chi2 = nl_chi2

    return {
        'final_params': {param: params[param] for param in fit_params},
        'uncertainties': uncertainties,
        'final_rms': final_rms_us,  # TRUE full-model RMS
        'prefit_rms': prefit_rms_us,
        'converged': converged,
        'iterations': iteration + 1,
        'residuals_us': residuals_final_us,
        'residuals_prefit_us': residuals_prefit_us,
        'errors_us': errors_us,
        'tdb_mjd': tdb_mjd,
        'covariance': cov,
        'final_chi2': final_chi2,
        'noise_realizations': noise_realizations,
        'n_noise_params': n_augmented + (1 if n_augmented > 0 else 0),
    }




def _build_general_fit_setup_from_cache(
    session_cached_data: Dict[str, Any],
    params_dict: Dict[str, float],
    fit_params: List[str],
    toa_mask: Optional[np.ndarray] = None,
    noise_config: Optional[object] = None,
    subtract_noise_sec: Optional[np.ndarray] = None
) -> GeneralFitSetup:
    """Build fitting setup from TimingSession cached data (fast, no I/O).

    Extracts arrays from cache, applies TOA mask, then delegates to the
    shared ``_build_setup_common()`` builder for noise wiring and
    parameter setup.
    
    Parameters
    ----------
    subtract_noise_sec : ndarray of float, optional
        Per-TOA noise realization (in seconds) to subtract from dt_sec_cached.
        If provided, toa_mask is applied before passing to _build_setup_common.
    """
    # Canonicalize and validate fit_params
    fit_params = [canonicalize_param_name(p) for p in fit_params]
    for p in fit_params:
        validate_fit_param(p)

    # Extract cached arrays
    dt_sec_cached = session_cached_data['dt_sec']
    dt_sec_ld = session_cached_data.get('dt_sec_ld')
    tdb_mjd = session_cached_data['tdb_mjd']
    freq_mhz_bary = session_cached_data['freq_bary_mhz']
    toas_mjd = session_cached_data['toas_mjd']
    errors_us = session_cached_data['errors_us']
    toa_flags = session_cached_data.get('toa_flags')

    # Build extras dict (pre-masking)
    extras = {
        'prebinary_delay_sec': session_cached_data.get('prebinary_delay_sec'),
        'roemer_shapiro_sec': session_cached_data.get('roemer_shapiro_sec'),
        'ssb_obs_pos_ls': session_cached_data.get('ssb_obs_pos_ls'),
        'sw_geometry_pc': session_cached_data.get('sw_geometry_pc'),
        'jump_phase': session_cached_data.get('jump_phase'),
        'tzr_phase': session_cached_data.get('tzr_phase'),
    }

    # Apply TOA mask if provided
    if toa_mask is not None:
        dt_sec_cached = dt_sec_cached[toa_mask]
        if dt_sec_ld is not None:
            dt_sec_ld = dt_sec_ld[toa_mask]
        tdb_mjd = tdb_mjd[toa_mask]
        freq_mhz_bary = freq_mhz_bary[toa_mask]
        toas_mjd = toas_mjd[toa_mask]
        errors_us = errors_us[toa_mask]
        if toa_flags is not None:
            toa_flags = [toa_flags[i] for i, m in enumerate(toa_mask) if m]
        if subtract_noise_sec is not None:
            subtract_noise_sec = subtract_noise_sec[toa_mask]
        for key in ('prebinary_delay_sec', 'roemer_shapiro_sec',
                     'ssb_obs_pos_ls', 'sw_geometry_pc', 'jump_phase'):
            if extras[key] is not None:
                extras[key] = extras[key][toa_mask]

    # Handle missing prebinary_delay_sec with informative warning
    if extras.get('prebinary_delay_sec') is None and extras.get('roemer_shapiro_sec') is not None:
        import warnings
        warnings.warn(
            "Cache missing 'prebinary_delay_sec' - falling back to roemer_shapiro_sec only.\n"
            "  FIX: Restart with fresh data or call session.compute_residuals(force_recompute=True).",
            UserWarning, stacklevel=2,
        )

    # Build NoiseConfig
    from jug.engine.noise_mode import NoiseConfig
    if noise_config is None:
        print(f"[FITTER] noise_config was None, auto-detecting from par file")
        noise_config = NoiseConfig.from_par(params_dict)
    else:
        enabled = {k: v for k, v in noise_config.enabled.items() if v}
        print(f"[FITTER] noise_config received, enabled: {list(enabled.keys())}")

    return _build_setup_common(
        params=params_dict,
        fit_params=fit_params,
        toas_mjd=toas_mjd,
        errors_us=errors_us,
        toa_flags=toa_flags,
        dt_sec_cached=dt_sec_cached,
        dt_sec_ld=dt_sec_ld,
        tdb_mjd=tdb_mjd,
        freq_mhz_bary=freq_mhz_bary,
        extras=extras,
        noise_config=noise_config,
        verbose=False,
        subtract_noise_sec=subtract_noise_sec,
    )


def fit_parameters_optimized_cached(
    setup: GeneralFitSetup,
    max_iter: int = 100,
    convergence_threshold: float = 1e-14,
    verbose: bool = False,
    solver_mode: str = "exact"
) -> Dict:
    """
    Fit parameters using precomputed setup (cached path).

    This is the public cached entrypoint that produces IDENTICAL results
    to fit_parameters_optimized() but reuses cached arrays.

    Parameters
    ----------
    setup : GeneralFitSetup
        Precomputed setup from _build_general_fit_setup_from_cache
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Convergence threshold
    verbose : bool
        Print progress
    solver_mode : str, default "exact"
        Solver mode: "exact" (SVD, bit-for-bit reproducible) or
        "fast" (QR/lstsq, faster but may differ slightly).

    Returns
    -------
    result : dict
        Fit results (identical format to fit_parameters_optimized)
    """
    total_start = time.time()

    # Run iterations (single pass with improved convergence criteria)
    result = _run_general_fit_iterations(
        setup, max_iter, convergence_threshold, verbose, solver_mode=solver_mode
    )
    
    total_time = time.time() - total_start
    result['total_time'] = total_time
    result['cache_time'] = 0.0  # Already cached
    result['jit_time'] = 0.0
    
    return result


def _fit_parameters_general(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    max_iter: int,
    convergence_threshold: float,
    clock_dir: str,
    verbose: bool,
    device: Optional[str]
) -> Dict:
    """General parameter fitter -- handles any parameter combination.

    Fits any mix of spin, DM, astrometric, binary, FD, solar-wind,
    JUMP, and DMX parameters.  This is the file-based path; the GUI
    uses fit_parameters_optimized_cached() instead.
    """
    
    total_start = time.time()
    
    # STEP 1: Build setup from files (expensive)
    cache_start = time.time()
    setup = _build_general_fit_setup_from_files(
        par_file, tim_file, fit_params, clock_dir, verbose
    )
    cache_time = time.time() - cache_start
    
    if verbose:
        spin_params = setup.spin_params
        dm_params = setup.dm_params
        binary_params = setup.binary_params
        param_summary = []
        if spin_params:
            param_summary.append(f"{len(spin_params)} spin")
        if dm_params:
            param_summary.append(f"{len(dm_params)} DM")
        if binary_params:
            param_summary.append(f"{len(binary_params)} binary")
        print(f"\nFitting {' + '.join(param_summary)} parameters")
        for param, val in zip(fit_params, setup.param_values_start):
            print(_format_param_value_for_print(param, val))
        print(f"  TOAs: {len(setup.toas_mjd)}")
        print(f"  Cached dt_sec in {cache_time:.3f}s")
    
    # STEP 2: Run iterations (fast, reuses setup)
    result = _run_general_fit_iterations(
        setup, max_iter, convergence_threshold, verbose
    )
    
    total_time = time.time() - total_start
    
    # Add timing info
    result['total_time'] = total_time
    result['cache_time'] = cache_time
    result['jit_time'] = 0.0
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Final RMS: {result['final_rms']:.6f} mus")
        print(f"\nFitted parameters:")
        for param in fit_params:
            val = result['final_params'][param]
            err = result['uncertainties'][param]
            print(_format_param_value_for_print(param, val, err))
        print(f"\nTotal time: {total_time:.3f}s")
        print(f"Cache time: {cache_time:.3f}s")
        print(f"{'='*80}")
    
    return result



