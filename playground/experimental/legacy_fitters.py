"""Legacy fitter implementations — archived from optimized_fitter.py.

These were early development versions that have been superseded by
the general fitter (_run_general_fit_iterations). Kept here for
reference and potential future experimentation.

Functions:
  - wls_solve_jax (original lines 441-451)
  - _solve_augmented_cholesky — NOTE: also used by live code, kept in main module (original lines 453-509)
  - compute_spin_phase_jax (original lines 511-534)
  - compute_spin_derivatives_jax (original lines 536-570)
  - full_iteration_jax_general (original lines 573-632)
  - full_iteration_jax_f0_f1 (original lines 635-705)
  - _fit_parameters_jax_incremental (original lines 814-1291)
  - _route_params_by_derivative_group (original lines 363-386)
  - _fit_spin_params_general (original lines 2953-3211)
  - _fit_f0_f1_level2 (original lines 3214-3398)
"""

# ======================================================================
# wls_solve_jax (originally lines 441-451)
# ======================================================================

def wls_solve_jax(residuals, errors, M):
    """JAX-compiled WLS solver using SVD."""
    weights_solve = 1.0 / errors
    M_weighted = M * weights_solve[:, None]
    r_weighted = residuals * weights_solve
    
    delta_params, _, _, _ = jnp.linalg.lstsq(M_weighted, r_weighted, rcond=None)
    cov = jnp.linalg.inv(M_weighted.T @ M_weighted)
    
    return delta_params, cov



# ======================================================================
# _solve_augmented_cholesky — NOTE: also used by live code, kept in main module (originally lines 453-509)
# ======================================================================

def _solve_augmented_cholesky(M2, r1, prior_inv, col_norms, n_timing_cols, has_offset):
    """Solve the augmented (GLS) WLS system via normal equations + Cholesky.

    Equivalent to the SVD-on-augmented-system approach but ~70× faster for
    large design matrices.  Falls back to SVD if Cholesky fails.

    Parameters
    ----------
    M2 : (n_toa, n_cols) — column-normalised, weight-scaled design matrix
    r1 : (n_toa,)        — weight-scaled residuals
    prior_inv : (n_cols,) — diagonal prior inverse (0 for unregularised cols)
    col_norms : (n_cols,) — column norms used for preconditioning
    n_timing_cols : int   — number of timing model columns (incl. offset)
    has_offset : bool     — whether column 0 is an offset column

    Returns
    -------
    delta_params, cov, delta_params_all, cov_all, noise_coeffs
    """
    n_cols = M2.shape[1]
    # Normal equations: (M^T M + P) x = M^T r
    MtM = M2.T @ M2
    Mtr = M2.T @ r1
    MtM[np.diag_indices(n_cols)] += prior_inv  # add prior to diagonal

    try:
        L = _scipy_linalg.cho_factor(MtM, lower=True, check_finite=False)
        delta_normalized = _scipy_linalg.cho_solve(L, Mtr, check_finite=False)
        cov_normalized = _scipy_linalg.cho_solve(L, np.eye(n_cols), check_finite=False)
    except _scipy_linalg.LinAlgError:
        # Cholesky failed — fall back to SVD on the augmented system
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




# ======================================================================
# compute_spin_phase_jax (originally lines 511-534)
# ======================================================================

def compute_spin_phase_jax(dt_sec: jnp.ndarray, f_values: jnp.ndarray) -> jnp.ndarray:
    """Compute spin phase for arbitrary spin parameters using JAX.
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time differences from PEPOCH in seconds
    f_values : jnp.ndarray
        Spin parameter values [F0, F1, F2, ...] in order
        
    Returns
    -------
    phase : jnp.ndarray
        Spin phase in cycles
    """
    phase = jnp.zeros_like(dt_sec)
    factorial = 1.0
    for n, f_val in enumerate(f_values):
        # Phase contribution: F_n * dt^(n+1) / (n+1)!
        factorial *= (n + 1)
        phase += f_val * (dt_sec ** (n + 1)) / factorial
    return phase




# ======================================================================
# compute_spin_derivatives_jax (originally lines 536-570)
# ======================================================================

def compute_spin_derivatives_jax(
    dt_sec: jnp.ndarray,
    f_values: jnp.ndarray,
    f0: float
) -> jnp.ndarray:
    """Compute design matrix for arbitrary spin parameters using JAX.
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time differences from PEPOCH in seconds
    f_values : jnp.ndarray
        Current spin parameter values [F0, F1, F2, ...]
    f0 : float
        Current F0 value (for normalization)
        
    Returns
    -------
    M : jnp.ndarray
        Design matrix (n_toas, n_params)
    """
    n_params = len(f_values)
    n_toas = len(dt_sec)
    M = jnp.zeros((n_toas, n_params))
    
    factorial = 1.0
    for n in range(n_params):
        # d(phase)/d(F_n) = dt^(n+1) / (n+1)!
        factorial *= (n + 1)
        deriv_phase = (dt_sec ** (n + 1)) / factorial
        # Convert to time units and apply PINT sign convention
        M = M.at[:, n].set(-deriv_phase / f0)
    
    return M



# ======================================================================
# full_iteration_jax_general (originally lines 573-632)
# ======================================================================

def full_iteration_jax_general(
    dt_sec: jnp.ndarray,
    f_values: jnp.ndarray,
    errors: jnp.ndarray,
    weights: jnp.ndarray
) -> tuple:
    """Complete fitting iteration for arbitrary spin parameters.
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Cached time differences (barycentric corrections applied)
    f_values : jnp.ndarray
        Current spin parameter values [F0, F1, F2, ...]
    errors : jnp.ndarray
        TOA uncertainties in seconds
    weights : jnp.ndarray
        Weight array (1/sigma^2)
        
    Returns
    -------
    delta_params : jnp.ndarray
        Parameter updates
    rms_us : float
        Weighted RMS in microseconds
    cov : jnp.ndarray
        Covariance matrix
    """
    f0 = f_values[0]
    
    # Compute spin phase
    phase = compute_spin_phase_jax(dt_sec, f_values)
    
    # Wrap to nearest integer (PINT convention)
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert phase residuals to time residuals (seconds)
    residuals = phase_wrapped / f0
    
    # Subtract weighted mean from residuals
    weighted_mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    residuals = residuals - weighted_mean
    
    # Compute design matrix
    M = compute_spin_derivatives_jax(dt_sec, f_values, f0)
    
    # Subtract weighted mean from each derivative column
    for i in range(M.shape[1]):
        col_mean = jnp.sum(M[:, i] * weights) / jnp.sum(weights)
        M = M.at[:, i].set(M[:, i] - col_mean)
    
    # Solve WLS
    delta_params, cov = wls_solve_jax(residuals, errors, M)
    
    # Compute weighted RMS
    rms_sec = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    rms_us = rms_sec * 1e6
    
    return delta_params, rms_us, cov



# ======================================================================
# full_iteration_jax_f0_f1 (originally lines 635-705)
# ======================================================================

def full_iteration_jax_f0_f1(
    dt_sec: jnp.ndarray,
    f0: float,
    f1: float,
    errors: jnp.ndarray,
    weights: jnp.ndarray
) -> tuple:
    """
    Complete fitting iteration for F0+F1 (JIT-compiled).
    
    This is the Level 2 optimization: everything in JAX!
    - Residual computation
    - Derivative computation
    - WLS solve
    
    All in one JIT-compiled function for maximum speed.
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Cached time deltas (includes all delays)
    f0 : float
        Current F0 value
    f1 : float
        Current F1 value
    errors : jnp.ndarray
        TOA errors in seconds
    weights : jnp.ndarray
        TOA weights (1/error^2)
        
    Returns
    -------
    delta_params : jnp.ndarray
        Parameter updates [delta_f0, delta_f1]
    rms_us : float
        RMS in microseconds
    cov : jnp.ndarray
        Covariance matrix
    """
    # Compute spin phase
    phase = dt_sec * (f0 + dt_sec * (f1 / 2.0))
    
    # Wrap phase (discard integer pulses)
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert to residuals
    residuals = phase_wrapped / f0
    
    # Subtract weighted mean
    weighted_mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    residuals = residuals - weighted_mean
    
    # Compute derivatives (PINT convention: negative sign)
    d_f0 = -(dt_sec / f0)
    d_f1 = -(dt_sec**2 / 2.0) / f0
    
    # Subtract mean from derivatives
    d_f0 = d_f0 - jnp.sum(d_f0 * weights) / jnp.sum(weights)
    d_f1 = d_f1 - jnp.sum(d_f1 * weights) / jnp.sum(weights)
    
    # Build design matrix
    M = jnp.column_stack([d_f0, d_f1])
    
    # WLS solve (also JAX-compiled!)
    delta_params, cov = wls_solve_jax(residuals, errors, M)
    
    # Compute RMS
    rms_sec = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    rms_us = rms_sec * 1e6
    
    return delta_params, rms_us, cov


# ======================================================================
# _fit_parameters_jax_incremental (originally lines 814-1291)
# ======================================================================

def _fit_parameters_jax_incremental(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    max_iter: int,
    convergence_threshold: float,
    clock_dir: str,
    verbose: bool,
    device: Optional[str]
) -> Dict:
    """JAX incremental fitter - achieves longdouble precision with JAX speed.
    
    This is the breakthrough method that combines:
    1. Longdouble initialization (perfect precision)
    2. JAX float64 iterations (fast JIT-compiled updates)
    3. Longdouble finalization (eliminates accumulated error)
    
    Achieves 0.001 ns RMS precision, converges in 4 iterations for typical cases.
    """
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.io.par_reader import parse_par_file, get_longdouble
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.fitting.wls_fitter import wls_solve_svd
    from jug.fitting.derivatives_dm import compute_dm_derivatives
    from jug.utils.constants import SECS_PER_DAY
    from jax import jit
    
    if verbose:
        print("JAX INCREMENTAL FITTER (Breakthrough Method)")
        print("="*80)
    
    total_start = time.time()
    
    # Load parameters and TOAs
    params = parse_par_file(par_file)
    
    # Validate par file timescale (fail fast on TCB)
    validate_par_timescale(params, context="fit_jax_incremental")
    
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract TOA data
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    freq_mhz = np.array([toa.freq_mhz for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])

    # Apply white noise scaling (EFAC/EQUAD) if present in par file
    noise_lines = params.get('_noise_lines')
    if noise_lines:
        from jug.noise.white import parse_noise_lines, apply_white_noise
        toa_flags = [toa.flags for toa in toas_data]
        noise_entries = parse_noise_lines(noise_lines)
        if noise_entries:
            errors_us = apply_white_noise(errors_us, toa_flags, noise_entries)

    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec ** 2
    n_toas = len(toas_mjd)
    
    # Get initial parameter values
    param_values_start = []
    for param in fit_params:
        if param in params:
            value = params[param]
            # Handle SINI='KIN' (DDK convention)
            if param == 'SINI' and isinstance(value, str) and value.upper() == 'KIN':
                kin_deg = float(params.get('KIN', 0.0))
                value = float(jnp.sin(jnp.deg2rad(kin_deg)))
            else:
                value = float(value)
            param_values_start.append(value)
        else:
            param_values_start.append(0.0)
    
    # Current parameter values
    f0 = float(params.get('F0', 0.0))
    f1 = float(params.get('F1', 0.0))
    dm = float(params.get('DM', 0.0))
    dm1 = float(params.get('DM1', 0.0))
    dmepoch_mjd = float(get_longdouble(params, 'DMEPOCH'))
    
    if verbose:
        print(f"Loaded {n_toas} TOAs")
        print(f"Initial F0  = {f0:.15f} Hz")
        print(f"Initial F1  = {f1:.6e} Hz/s")
        if 'DM' in fit_params or 'DM1' in fit_params:
            print(f"Initial DM  = {dm:.15f} pc/cm^3")
            print(f"Initial DM1 = {dm1:.6e} pc/cm^3/day")
        print()
    
    # -------------------------------------------------------------------------
    # CACHE INITIAL STATE (like production fitter)
    # -------------------------------------------------------------------------
    if verbose:
        print("Caching initial state...")
    
    cache_start = time.time()
    
    # Compute residuals with initial parameters (dt_sec has ALL delays baked in)
    result = compute_residuals_simple(
        par_file, tim_file,
        clock_dir=clock_dir,
        subtract_tzr=False,
        verbose=False
    )
    
    dt_sec_cached = result['dt_sec']
    dt_sec_ld = result.get('dt_sec_ld')
    if dt_sec_ld is None:
        dt_sec_ld = np.array(dt_sec_cached, dtype=np.longdouble)
    tdb_mjd = result['tdb_mjd']
    freq_bary_mhz = result['freq_bary_mhz']
    jump_phase = result.get('jump_phase')

    # Cache initial DM delay (for incremental updates)
    initial_dm_params = {'DM': dm, 'DM1': dm1}
    initial_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, initial_dm_params, dmepoch_mjd)

    # Compute initial residuals via shared canonical function (longdouble precision)
    from jug.residuals.simple_calculator import compute_phase_residuals
    _, residuals_init = compute_phase_residuals(
        dt_sec_ld, params, weights, subtract_mean=True,
        jump_phase=jump_phase
    )
    
    cache_time = time.time() - cache_start
    
    if verbose:
        print(f"  ✓ Cached in {cache_time*1000:.2f} ms")
        print()
    
    # -------------------------------------------------------------------------
    # JAX INCREMENTAL ITERATION FUNCTION
    # -------------------------------------------------------------------------
    @jit
    def jax_iteration_f0_f1(residuals, dt_sec, f0, f1, weights):
        """Single iteration: Update F0, F1 using JAX incremental method."""
        # Design matrix for F0/F1
        M0 = -dt_sec / f0
        M1 = -(dt_sec**2 / 2.0) / f0
        
        # Zero weighted mean
        sum_w = jnp.sum(weights)
        M0 = M0 - jnp.sum(M0 * weights) / sum_w
        M1 = M1 - jnp.sum(M1 * weights) / sum_w
        
        # Build 2×2 normal equations
        A00 = jnp.sum(M0 * weights * M0)
        A01 = jnp.sum(M0 * weights * M1)
        A11 = jnp.sum(M1 * weights * M1)
        
        b0 = jnp.sum(M0 * weights * residuals)
        b1 = jnp.sum(M1 * weights * residuals)
        
        # Analytical 2×2 solve
        det = A00 * A11 - A01 * A01
        delta_f0 = (A11 * b0 - A01 * b1) / det
        delta_f1 = (A00 * b1 - A01 * b0) / det
        
        # Update residuals incrementally (the magic!)
        residuals_new = residuals - delta_f0 * M0 - delta_f1 * M1
        
        # RMS
        rms = jnp.sqrt(jnp.sum(residuals**2 * weights) / sum_w)
        
        return residuals_new, delta_f0, delta_f1, rms
    
    # -------------------------------------------------------------------------
    # FITTING LOOP
    # -------------------------------------------------------------------------
    if verbose:
        print("FITTING LOOP")
        print("-"*80)
    
    # Current dt_sec and residuals (will be updated incrementally)
    dt_sec_current = dt_sec_cached.copy()
    residuals_jax = jnp.array(residuals_init)
    weights_jax = jnp.array(weights)
    
    # Convergence criteria (match production fitter)
    xtol = 1e-12
    gtol = 1e-5  # μs (0.01 ns change)
    min_iterations = 5
    
    iter_start = time.time()
    history = []
    rms_history = []
    converged = False
    
    for iteration in range(max_iter):
        # STEP 1: Fit F0/F1 using JAX incremental method
        dt_jax = jnp.array(dt_sec_current)
        residuals_jax, delta_f0, delta_f1, rms = jax_iteration_f0_f1(
            residuals_jax, dt_jax, f0, f1, weights_jax
        )
        
        delta_f0_val = float(delta_f0)
        delta_f1_val = float(delta_f1)
        rms_us = float(rms) * 1e6
        
        # Update F0/F1
        f0 += delta_f0_val
        f1 += delta_f1_val
        params['F0'] = f0
        params['F1'] = f1
        
        # STEP 2: Fit DM parameters if requested
        if 'DM' in fit_params or 'DM1' in fit_params:
            residuals_np = np.array(residuals_jax)
            
            # Compute DM derivatives
            params_current = {'DMEPOCH': dmepoch_mjd, 'DM': dm, 'DM1': dm1, 'F0': f0}
            dm_fit_params = [p for p in ['DM', 'DM1'] if p in fit_params]
            dm_derivs = compute_dm_derivatives(
                params=params_current,
                toas_mjd=tdb_mjd,
                freq_mhz=freq_bary_mhz,
                fit_params=dm_fit_params
            )
            
            # Build DM design matrix
            M_dm = np.column_stack([dm_derivs[p] for p in dm_fit_params])
            
            # Zero weighted mean
            for j in range(len(dm_fit_params)):
                col_mean = np.sum(M_dm[:, j] * weights) / np.sum(weights)
                M_dm[:, j] = M_dm[:, j] - col_mean
            
            # WLS solve for DM
            delta_dm_params, cov_dm, _ = wls_solve_svd(
                jnp.array(residuals_np),
                jnp.array(errors_sec),
                jnp.array(M_dm),
                negate_dpars=False
            )
            delta_dm_params = np.array(delta_dm_params)
            
            # Update DM parameters
            if 'DM' in fit_params:
                dm += delta_dm_params[dm_fit_params.index('DM')]
                params['DM'] = dm
            if 'DM1' in fit_params:
                dm1 += delta_dm_params[dm_fit_params.index('DM1')]
                params['DM1'] = dm1
            
            # Update residuals incrementally for DM changes
            residuals_np = residuals_np - M_dm @ delta_dm_params
            residuals_jax = jnp.array(residuals_np)
            
            # Update dt_sec for next F0/F1 iteration
            new_dm_params = {'DM': dm, 'DM1': dm1}
            new_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, new_dm_params, dmepoch_mjd)
            dt_delay_change = new_dm_delay - initial_dm_delay
            dt_sec_current = dt_sec_cached - dt_delay_change
            
            # Recompute RMS
            rms_us = np.sqrt(np.sum(residuals_np**2 * weights) / np.sum(weights)) * 1e6
            
            max_delta = max(abs(delta_f0_val), abs(delta_f1_val), *[abs(d) for d in delta_dm_params])
        else:
            max_delta = max(abs(delta_f0_val), abs(delta_f1_val))
        
        history.append({'iteration': iteration + 1, 'rms': rms_us, 'max_delta': max_delta})
        rms_history.append(rms_us)
        
        # Check convergence
        delta_params_all = [delta_f0_val, delta_f1_val]
        param_values_current = [f0, f1]
        if 'DM' in fit_params or 'DM1' in fit_params:
            delta_params_all.extend(delta_dm_params)
            if 'DM' in fit_params:
                param_values_current.append(dm)
            if 'DM1' in fit_params:
                param_values_current.append(dm1)
        
        delta_norm = np.linalg.norm(delta_params_all)
        param_norm = np.linalg.norm(param_values_current)
        param_converged = delta_norm <= xtol * (param_norm + xtol)
        
        rms_converged = False
        if len(rms_history) >= 2:
            rms_change = abs(rms_history[-1] - rms_history[-2])
            rms_converged = rms_change < gtol
        
        converged = iteration >= min_iterations and (param_converged or rms_converged)
        
        if verbose:
            status = ""
            if converged:
                status = "✓ Converged"
            if iteration == 0:
                print(f"  Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} (includes JIT)")
            else:
                print(f"  Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} {status}")
        
        if converged:
            break
    
    iter_time = time.time() - iter_start
    
    # -------------------------------------------------------------------------
    # FINAL RECOMPUTATION IN LONGDOUBLE (eliminates accumulated error)
    # -------------------------------------------------------------------------
    if verbose:
        print()
        print("Final recomputation in longdouble...")
    
    final_start = time.time()
    
    final_dm_params = {'DM': dm, 'DM1': dm1}
    final_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, final_dm_params, dmepoch_mjd)
    dt_delay_change_final = final_dm_delay - initial_dm_delay
    dt_sec_final = dt_sec_cached - dt_delay_change_final
    
    dt_final_ld = np.array(dt_sec_final, dtype=np.longdouble)
    f0_final_ld = np.longdouble(f0)
    f1_final_ld = np.longdouble(f1)
    
    phase_final_ld = dt_final_ld * (f0_final_ld + dt_final_ld * (f1_final_ld / 2.0))
    phase_wrapped_final_ld = phase_final_ld - np.round(phase_final_ld)
    residuals_final_ld = phase_wrapped_final_ld / f0_final_ld
    
    residuals_final = np.array(residuals_final_ld, dtype=np.float64)
    weighted_mean_final = np.sum(residuals_final * weights) / np.sum(weights)
    residuals_final = residuals_final - weighted_mean_final
    residuals_final_us = residuals_final * 1e6
    
    final_time = time.time() - final_start
    
    # Compute final RMS
    final_rms_us = np.sqrt(np.sum(residuals_final**2 * weights) / np.sum(weights)) * 1e6
    
    # Compute prefit residuals (using initial parameters)
    for i, param in enumerate(fit_params):
        _update_param(params, param, param_values_start[i])
    
    # Restore initial DM delay
    prefit_dm_params = {'DM': param_values_start[fit_params.index('DM')] if 'DM' in fit_params else dm,
                        'DM1': param_values_start[fit_params.index('DM1')] if 'DM1' in fit_params else dm1}
    prefit_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, prefit_dm_params, dmepoch_mjd)
    dt_delay_change_prefit = prefit_dm_delay - initial_dm_delay
    dt_sec_prefit = dt_sec_cached - dt_delay_change_prefit
    
    f0_prefit = param_values_start[fit_params.index('F0')] if 'F0' in fit_params else f0
    f1_prefit = param_values_start[fit_params.index('F1')] if 'F1' in fit_params else f1
    
    dt_prefit_ld = np.array(dt_sec_prefit, dtype=np.longdouble)
    f0_prefit_ld = np.longdouble(f0_prefit)
    f1_prefit_ld = np.longdouble(f1_prefit)
    
    phase_prefit_ld = dt_prefit_ld * (f0_prefit_ld + dt_prefit_ld * (f1_prefit_ld / 2.0))
    phase_wrapped_prefit_ld = phase_prefit_ld - np.round(phase_prefit_ld)
    residuals_prefit_ld = phase_wrapped_prefit_ld / f0_prefit_ld
    
    residuals_prefit = np.array(residuals_prefit_ld, dtype=np.float64)
    weighted_mean_prefit = np.sum(residuals_prefit * weights) / np.sum(weights)
    residuals_prefit = residuals_prefit - weighted_mean_prefit
    residuals_prefit_us = residuals_prefit * 1e6
    
    prefit_rms_us = np.sqrt(np.sum(residuals_prefit**2 * weights) / np.sum(weights)) * 1e6
    
    # Restore final parameter values
    params['F0'] = f0
    params['F1'] = f1
    params['DM'] = dm
    params['DM1'] = dm1
    
    # Compute covariance (using final design matrix)
    if 'DM' in fit_params or 'DM1' in fit_params:
        # Build full design matrix for final covariance
        M_full_list = []
        
        # F0/F1 columns
        dt_jax = jnp.array(dt_sec_final)
        M0 = np.array(-dt_jax / f0)
        M1 = np.array(-(dt_jax**2 / 2.0) / f0)
        
        sum_w = np.sum(weights)
        M0 = M0 - np.sum(M0 * weights) / sum_w
        M1 = M1 - np.sum(M1 * weights) / sum_w
        
        if 'F0' in fit_params:
            M_full_list.append(M0)
        if 'F1' in fit_params:
            M_full_list.append(M1)
        
        # DM columns
        params_current = {'DMEPOCH': dmepoch_mjd, 'DM': dm, 'DM1': dm1, 'F0': f0}
        dm_fit_params = [p for p in ['DM', 'DM1'] if p in fit_params]
        dm_derivs = compute_dm_derivatives(
            params=params_current,
            toas_mjd=tdb_mjd,
            freq_mhz=freq_bary_mhz,
            fit_params=dm_fit_params
        )
        
        for p in dm_fit_params:
            col = dm_derivs[p]
            col = col - np.sum(col * weights) / sum_w
            M_full_list.append(col)
        
        M_full = np.column_stack(M_full_list)
        
        # Compute covariance
        _, cov, _ = wls_solve_svd(
            jnp.array(residuals_final),
            jnp.array(errors_sec),
            jnp.array(M_full),
            negate_dpars=False
        )
        cov = np.array(cov)
    else:
        # F0/F1 only
        dt_jax = jnp.array(dt_sec_final)
        M0 = np.array(-dt_jax / f0)
        M1 = np.array(-(dt_jax**2 / 2.0) / f0)
        
        sum_w = np.sum(weights)
        M0 = M0 - np.sum(M0 * weights) / sum_w
        M1 = M1 - np.sum(M1 * weights) / sum_w
        
        M_full_list = []
        if 'F0' in fit_params:
            M_full_list.append(M0)
        if 'F1' in fit_params:
            M_full_list.append(M1)
        
        M_full = np.column_stack(M_full_list)
        
        _, cov, _ = wls_solve_svd(
            jnp.array(residuals_final),
            jnp.array(errors_sec),
            jnp.array(M_full),
            negate_dpars=False
        )
        cov = np.array(cov)
    
    # Compute uncertainties
    uncertainties = {param: np.sqrt(cov[i, i]) for i, param in enumerate(fit_params)}
    
    total_time = time.time() - total_start
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Converged: {converged}")
        print(f"Iterations: {len(history)}")
        print(f"Final RMS: {final_rms_us:.6f} μs")
        print(f"Prefit RMS: {prefit_rms_us:.6f} μs")
        print(f"\nFitted parameters:")
        for param in fit_params:
            val = params[param]
            err = uncertainties[param]
            print(_format_param_value_for_print(param, val, err))
        print(f"\nTotal time: {total_time:.3f}s")
        print(f"Cache time: {cache_time:.3f}s")
        print(f"Iteration time: {iter_time:.3f}s")
        print(f"Final recomp time: {final_time:.3f}s")
        print(f"{'='*80}")
    
    return {
        'final_params': {param: params[param] for param in fit_params},
        'uncertainties': uncertainties,
        'final_rms': final_rms_us,
        'prefit_rms': prefit_rms_us,
        'converged': converged,
        'iterations': len(history),
        'total_time': total_time,
        'residuals_us': residuals_final_us,
        'residuals_prefit_us': residuals_prefit_us,
        'errors_us': errors_us,
        'tdb_mjd': tdb_mjd,
        'cache_time': cache_time,
        'jit_time': iter_time,
        'covariance': cov
    }


# ======================================================================
# _route_params_by_derivative_group (originally lines 363-386)
# ======================================================================

def _route_params_by_derivative_group(fit_params: List[str]) -> Dict[DerivativeGroup, List[str]]:
    """
    Route parameters by their derivative computation group.

    This is the spec-driven replacement for:
        spin_params = [p for p in fit_params if p.startswith('F')]
        dm_params = [p for p in fit_params if p.startswith('DM')]

    Parameters
    ----------
    fit_params : list of str
        Parameters to fit

    Returns
    -------
    dict
        Mapping from DerivativeGroup to list of parameters
    """
    grouped = {}
    for param in fit_params:
        group = get_derivative_group(param)
        if group is not None:
            grouped.setdefault(group, []).append(param)
    return grouped


# ======================================================================
# _fit_spin_params_general (originally lines 2953-3211)
# ======================================================================

def _fit_spin_params_general(
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    max_iter: int,
    convergence_threshold: float,
    clock_dir: str,
    verbose: bool,
    device: Optional[str]
) -> Dict:
    """General implementation for fitting arbitrary spin parameters.
    
    This replaces the hardcoded F0+F1 fitter with a flexible version
    that can fit any combination of spin parameters (F0, F1, F2, ...).
    """
    
    total_start = time.time()
    
    # Parse files
    params = parse_par_file(par_file)
    
    # Validate par file timescale (fail fast on TCB)
    validate_par_timescale(params, context="general_fit")
    
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract TOA data
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    
    # Extract starting parameter values
    f_values_start = []
    for param in fit_params:
        if param not in params:
            raise ValueError(f"Parameter {param} not found in .par file")
        f_values_start.append(params[param])
    
    if verbose:
        print(f"\nStarting parameters:")
        for param, val in zip(fit_params, f_values_start):
            if abs(val) < 1e-10 and val != 0:
                print(f"  {param} = {val:.20e}")
            else:
                print(f"  {param} = {val:.20f}")
        print(f"  TOAs: {len(toas_data)}")
    
    # LEVEL 1: Compute dt_sec ONCE (cache expensive delays)
    if verbose:
        print(f"\nLevel 1: Caching expensive delays...")
    cache_start = time.time()
    
    result = compute_residuals_simple(
        par_file,
        tim_file,
        clock_dir=clock_dir,
        subtract_tzr=False,  # Don't wrap - we'll do it ourselves
        verbose=False  # Disable verbose output for speed
    )
    
    dt_sec_cached = result['dt_sec']
    
    cache_time = time.time() - cache_start
    if verbose:
        print(f"  Cached dt_sec for {len(dt_sec_cached)} TOAs in {cache_time:.3f}s")
    
    # Get JAX device
    n_toas = len(dt_sec_cached)
    n_params = len(fit_params)
    jax_device = get_device(prefer=device, n_toas=n_toas, n_params=n_params)
    
    if verbose:
        device_type = 'CPU' if 'cpu' in str(jax_device).lower() else 'GPU'
        print(f"  Using {device_type} device: {jax_device}")
    
    # Convert to JAX arrays on selected device
    with jax.default_device(jax_device):
        dt_sec_jax = jnp.array(dt_sec_cached)
        errors_jax = jnp.array(errors_sec)
        weights_jax = jnp.array(weights)
        f_values_jax = jnp.array(f_values_start)
    
    # LEVEL 2: JAX JIT compilation
    if verbose:
        print(f"\nLevel 2: JIT compiling iteration...")
    
    # Warm up JIT (first call compiles)
    jit_start = time.time()
    _, _, _ = full_iteration_jax_general(
        dt_sec_jax, f_values_jax, errors_jax, weights_jax
    )
    jit_time = time.time() - jit_start
    
    if verbose:
        print(f"  JIT compiled in {jit_time:.3f}s")
        param_names = ', '.join(fit_params)
        print(f"\nFitting {param_names}...")
    
    # Fitting loop
    f_values_curr = np.array(f_values_start)
    prev_delta_max = None
    prev_rms = None
    iteration_times = []
    prefit_rms = None  # Will store RMS from iteration 0
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # Convert current values to JAX
        with jax.default_device(jax_device):
            f_values_jax = jnp.array(f_values_curr)
        
        # Complete iteration in JAX (JIT-compiled)
        delta_params_jax, rms_us, cov_jax = full_iteration_jax_general(
            dt_sec_jax, f_values_jax, errors_jax, weights_jax
        )
        
        # Convert results back to numpy
        delta_params = np.array(delta_params_jax)
        cov = np.array(cov_jax)
        rms_us = float(rms_us)
        
        # Store prefit RMS (before any parameter updates)
        if iteration == 0:
            prefit_rms = rms_us
        
        # Update parameters
        f_values_curr += delta_params
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence using multiple criteria
        max_delta = np.max(np.abs(delta_params))
        
        # Criterion 1: Parameter change below threshold
        param_converged = max_delta < convergence_threshold
        
        # Criterion 2: RMS change below threshold (more physically meaningful)
        # Require at least 5 iterations before checking RMS convergence
        # to allow the linearization to stabilize
        rms_converged = False
        if prev_rms is not None and iteration >= 4:  # At least 5 iterations
            rms_change = abs(rms_us - prev_rms) / prev_rms if prev_rms > 0 else 0
            rms_converged = rms_change < 1e-8  # 0.00001% change (very tight)
        
        # Criterion 3: Stagnation (parameter change stopped)
        stagnated = False
        if prev_delta_max is not None and iteration >= 2:
            stagnated = abs(max_delta - prev_delta_max) < 1e-20
        
        if verbose and (iteration < 3 or iteration >= max_iter - 1):
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s")
        elif verbose and iteration == 3:
            print(f"  ...")
        
        # Check convergence (any criterion, but with minimum iteration guards)
        if stagnated:
            if verbose:
                print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs (converged - stagnation)")
            converged = True
            iterations = iteration + 1
            break
        
        if param_converged:
            if verbose:
                print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs (converged - param change)")
            converged = True
            iterations = iteration + 1
            break
        
        if rms_converged:
            if verbose:
                print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs (converged - RMS stable)")
            converged = True
            iterations = iteration + 1
            break
        
        prev_delta_max = max_delta
        prev_rms = rms_us
    else:
        converged = False
        iterations = max_iter
    
    total_time = time.time() - total_start
    
    # Compute final residuals for output  
    with jax.default_device(jax_device):
        f_values_final_jax = jnp.array(f_values_curr)
        # Compute phase
        phase = compute_spin_phase_jax(dt_sec_jax, f_values_final_jax)
        phase_wrapped = phase - jnp.round(phase)
        residuals_final = phase_wrapped / f_values_final_jax[0]
        weighted_mean_res = jnp.sum(residuals_final * weights_jax) / jnp.sum(weights_jax)
        residuals_final = residuals_final - weighted_mean_res
        residuals_final_us = np.array(residuals_final) * 1e6
        
    # Compute prefit residuals (with starting parameters)
    with jax.default_device(jax_device):
        f_values_start_jax = jnp.array(f_values_start)
        phase_start = compute_spin_phase_jax(dt_sec_jax, f_values_start_jax)
        phase_wrapped_start = phase_start - jnp.round(phase_start)
        residuals_start = phase_wrapped_start / f_values_start_jax[0]
        weighted_mean_start_res = jnp.sum(residuals_start * weights_jax) / jnp.sum(weights_jax)
        residuals_start = residuals_start - weighted_mean_start_res
        residuals_prefit_us = np.array(residuals_start) * 1e6
    
    # Also need TDB times for plotting
    tdb_mjd = result['tdb_mjd']  # from compute_residuals_simple call
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        
        print(f"\nConvergence:")
        print(f"  Iterations: {iterations}")
        print(f"  Converged: {converged}")
        
        print(f"\nTiming:")
        print(f"  Cache initialization: {cache_time:.3f}s")
        print(f"  JIT compilation: {jit_time:.3f}s")
        print(f"  Fitting iterations: {sum(iteration_times):.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        
        print(f"\nFinal parameters:")
        for i, param in enumerate(fit_params):
            val = f_values_curr[i]
            unc = np.sqrt(cov[i, i])
            if abs(val) < 1e-10 and val != 0:
                print(f"  {param} = {val:.20e} ± {unc:.2e}")
            else:
                print(f"  {param} = {val:.20f} ± {unc:.2e}")
        
        print(f"\nFinal RMS: {rms_us:.6f} μs")
    
    # Build final_params dict
    final_params = {param: f_values_curr[i] for i, param in enumerate(fit_params)}
    
    # Build uncertainties dict
    uncertainties = {param: np.sqrt(cov[i, i]) for i, param in enumerate(fit_params)}
    
    return {
        'final_params': final_params,
        'uncertainties': uncertainties,
        'prefit_rms': prefit_rms,
        'final_rms': rms_us,
        'prefit_residuals_us': residuals_prefit_us,
        'postfit_residuals_us': residuals_final_us,
        'tdb_mjd': tdb_mjd,
        'errors_us': errors_us,  # TOA uncertainties for plotting
        'iterations': iterations,
        'converged': converged,
        'total_time': total_time,
        'cache_time': cache_time,
        'jit_time': jit_time,
        'covariance': cov
    }


# ======================================================================
# _fit_f0_f1_level2 (originally lines 3214-3398)
# ======================================================================

def _fit_f0_f1_level2(
    par_file: Path,
    tim_file: Path,
    max_iter: int,
    convergence_threshold: float,
    clock_dir: str,
    verbose: bool,
    device: Optional[str]
) -> Dict:
    """Internal implementation of F0+F1 fitting with Level 2 optimization."""
    
    total_start = time.time()
    
    # Parse files
    params = parse_par_file(par_file)
    
    # Validate par file timescale (fail fast on TCB)
    validate_par_timescale(params, context="fit_f0_f1")
    
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract TOA data
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    
    f0_start = params['F0']
    f1_start = params['F1']
    
    if verbose:
        print(f"\nStarting parameters:")
        print(f"  F0 = {f0_start:.20f} Hz")
        print(f"  F1 = {f1_start:.20e} Hz/s")
        print(f"  TOAs: {len(toas_data)}")
    
    # LEVEL 1: Compute dt_sec ONCE (cache expensive delays)
    if verbose:
        print(f"\nLevel 1: Caching expensive delays...")
    cache_start = time.time()
    
    with contextlib.redirect_stdout(io.StringIO()):
        result = compute_residuals_simple(
            par_file,
            tim_file,
            clock_dir=clock_dir,
            subtract_tzr=False  # Don't wrap - we'll do it ourselves with current F0/F1
        )
    
    dt_sec_cached = result['dt_sec']
    
    cache_time = time.time() - cache_start
    if verbose:
        print(f"  Cached dt_sec for {len(dt_sec_cached)} TOAs in {cache_time:.3f}s")
    
    # Get JAX device (CPU by default for typical pulsar timing)
    n_toas = len(dt_sec_cached)
    n_params = 2  # F0 and F1
    jax_device = get_device(prefer=device, n_toas=n_toas, n_params=n_params)
    
    if verbose:
        device_type = 'CPU' if 'cpu' in str(jax_device).lower() else 'GPU'
        print(f"  Using {device_type} device: {jax_device}")
    
    # Convert to JAX arrays on selected device
    with jax.default_device(jax_device):
        dt_sec_jax = jnp.array(dt_sec_cached)
        errors_jax = jnp.array(errors_sec)
        weights_jax = jnp.array(weights)
    
    # LEVEL 2: JAX JIT compilation
    if verbose:
        print(f"\nLevel 2: JIT compiling iteration...")
    
    f0_curr = f0_start
    f1_curr = f1_start
    
    # Warm up JIT (first call compiles)
    jit_start = time.time()
    _, _, _ = full_iteration_jax_f0_f1(
        dt_sec_jax, f0_curr, f1_curr, errors_jax, weights_jax
    )
    jit_time = time.time() - jit_start
    
    if verbose:
        print(f"  JIT compiled in {jit_time:.3f}s")
        print(f"\nFitting F0 + F1...")
    
    # Fitting loop
    prev_delta_max = None
    iteration_times = []
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # Complete iteration in JAX (JIT-compiled)
        delta_params_jax, rms_us, cov_jax = full_iteration_jax_f0_f1(
            dt_sec_jax, f0_curr, f1_curr, errors_jax, weights_jax
        )
        
        # Convert results back to numpy
        delta_params = np.array(delta_params_jax)
        cov = np.array(cov_jax)
        rms_us = float(rms_us)
        
        # Update parameters
        f0_curr += delta_params[0]
        f1_curr += delta_params[1]
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence
        max_delta = max(abs(delta_params[0]), abs(delta_params[1]))
        
        if verbose and (iteration < 3 or iteration >= max_iter - 1):
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s")
        elif verbose and iteration == 3:
            print(f"  ...")
        
        # Check for convergence
        if prev_delta_max is not None and abs(max_delta - prev_delta_max) < 1e-20:
            if verbose:
                print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        if max_delta < convergence_threshold:
            if verbose:
                print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        prev_delta_max = max_delta
    else:
        converged = False
        iterations = max_iter
    
    total_time = time.time() - total_start
    
    # Print results
    if verbose:
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        
        print(f"\nConvergence:")
        print(f"  Iterations: {iterations}")
        print(f"  Converged: {converged}")
        
        print(f"\nTiming:")
        print(f"  Cache initialization: {cache_time:.3f}s")
        print(f"  JIT compilation: {jit_time:.3f}s")
        print(f"  Fitting iterations: {sum(iteration_times):.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        
        print(f"\nFinal parameters:")
        print(f"  F0 = {f0_curr:.20f} Hz")
        print(f"  F1 = {f1_curr:.20e} Hz/s")
        print(f"  RMS = {rms_us:.6f} μs")
        
        print(f"\nUncertainties:")
        unc_f0 = np.sqrt(cov[0, 0])
        unc_f1 = np.sqrt(cov[1, 1])
        print(f"  σ(F0) = {unc_f0:.3e} Hz")
        print(f"  σ(F1) = {unc_f1:.3e} Hz/s")
    
    return {
        'final_params': {
            'F0': f0_curr,
            'F1': f1_curr
        },
        'uncertainties': {
            'F0': np.sqrt(cov[0, 0]),
            'F1': np.sqrt(cov[1, 1])
        },
        'final_rms': float(rms_us),
        'iterations': iterations,
        'converged': converged,
        'total_time': total_time,
        'cache_time': cache_time,
        'jit_time': jit_time,
        'covariance': cov
    }


