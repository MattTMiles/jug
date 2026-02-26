#!/usr/bin/env python3
"""
Stage-by-Stage Benchmark for JUG
================================

Quantifies cold-start costs and breaks down compute_residuals into stages.

Measures:
- Import times (engine, residuals, fitter modules)
- Session creation (par/tim parse)
- Residual compute stages (clock, TDB, barycentric, JAX kernel)
- Fit stages (setup, iterations, solver)

Runs pipeline twice to separate cold vs warm performance.

Output:
- Sorted timing table
- JSON results file for commit comparison

Usage:
    python -m jug.scripts.benchmark_stages [--par FILE] [--tim FILE] [--output FILE]
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageTimer:
    """Context manager for timing code stages with accumulation support."""
    
    times: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    _stack: List[tuple] = field(default_factory=list)
    
    @contextmanager
    def stage(self, name: str):
        """Time a named stage. Accumulates if called multiple times."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.times[name] = self.times.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
    
    def get_time(self, name: str) -> float:
        """Get total time for a stage."""
        return self.times.get(name, 0.0)
    
    def get_count(self, name: str) -> int:
        """Get call count for a stage."""
        return self.counts.get(name, 0)
    
    def total(self) -> float:
        """Get total time across all stages."""
        return sum(self.times.values())
    
    def report(self, title: str = "Timing Report") -> str:
        """Generate formatted timing report."""
        lines = []
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)
        lines.append(f"{'Stage':<45} {'Time (ms)':>10} {'%':>8} {'Calls':>6}")
        lines.append("-" * 70)
        
        total = self.total()
        # Sort by time descending
        sorted_stages = sorted(self.times.items(), key=lambda x: -x[1])
        
        for name, t in sorted_stages:
            pct = (t / total * 100) if total > 0 else 0
            count = self.counts.get(name, 1)
            lines.append(f"{name:<45} {t*1000:>10.2f} {pct:>7.1f}% {count:>6}")
        
        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<45} {total*1000:>10.2f} {'100.0':>7}%")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "times_ms": {k: v * 1000 for k, v in self.times.items()},
            "counts": dict(self.counts),
            "total_ms": self.total() * 1000
        }


def benchmark_imports(timer: StageTimer) -> None:
    """Benchmark module import times."""
    
    # Core Python imports (baseline)
    with timer.stage("import: numpy"):
        import numpy as np
    
    with timer.stage("import: astropy.time"):
        from astropy.time import Time
    
    with timer.stage("import: astropy.coordinates"):
        from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
    
    with timer.stage("import: jax"):
        from jug.utils.jax_setup import ensure_jax_x64
        ensure_jax_x64()
        import jax
        import jax.numpy as jnp
    
    # JUG modules
    with timer.stage("import: jug.engine.session"):
        from jug.engine.session import TimingSession
    
    with timer.stage("import: jug.residuals.simple_calculator"):
        from jug.residuals.simple_calculator import compute_residuals_simple
    
    with timer.stage("import: jug.fitting.optimized_fitter"):
        from jug.fitting.optimized_fitter import fit_parameters_optimized


def benchmark_session_creation(timer: StageTimer, par_file: Path, tim_file: Path):
    """Benchmark session creation stages."""
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    
    with timer.stage("session: parse_par_file"):
        params = parse_par_file(par_file)
    
    with timer.stage("session: parse_tim_file_mjds"):
        toas_data = parse_tim_file_mjds(tim_file)
    
    return params, toas_data


def benchmark_residuals_detailed(
    timer: StageTimer,
    par_file: Path,
    tim_file: Path,
    params: dict,
    toas_data: list,
    run_label: str = ""
) -> dict:
    """Benchmark compute_residuals with detailed stage breakdown."""
    import numpy as np
    from pathlib import Path
    from astropy.time import Time
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris, EarthLocation
    from astropy import units as u
    import jax.numpy as jnp
    
    from jug.io.par_reader import parse_ra, parse_dec, get_longdouble
    from jug.io.tim_reader import compute_tdb_standalone_vectorized
    from jug.io.clock import parse_clock_file
    from jug.delays.barycentric import (
        compute_ssb_obs_pos_vel,
        compute_pulsar_direction,
        compute_roemer_delay,
        compute_shapiro_delay,
        compute_barycentric_freq
    )
    from jug.delays.combined import compute_total_delay_jax
    from jug.utils.constants import SECS_PER_DAY, T_SUN_SEC, T_PLANET, OBSERVATORIES
    
    prefix = f"residuals{run_label}: "
    
    # Clock parsing
    module_dir = Path(__file__).parent
    clock_dir = module_dir.parent.parent / "data" / "clock"
    
    with timer.stage(prefix + "clock parsing"):
        mk_clock = parse_clock_file(clock_dir / "mk2utc.clk")
        gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")
        bipm_clock = parse_clock_file(clock_dir / "tai2tt_bipm2024.clk")
    
    # Extract TOA arrays
    with timer.stage(prefix + "TOA array extraction"):
        mjd_ints = [toa.mjd_int for toa in toas_data]
        mjd_fracs = [toa.mjd_frac for toa in toas_data]
        freq_mhz = np.array([toa.freq_mhz for toa in toas_data])
        errors_us = np.array([toa.error_us for toa in toas_data])
    
    # Observatory location
    obs_itrf_km = OBSERVATORIES.get("meerkat")
    location = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km
    )
    
    # TDB conversion
    with timer.stage(prefix + "compute_tdb_standalone_vectorized"):
        tdb_mjd = compute_tdb_standalone_vectorized(
            mjd_ints, mjd_fracs,
            mk_clock, gps_clock, bipm_clock,
            location
        )
    
    # Astrometry setup
    with timer.stage(prefix + "astrometry setup"):
        ra_rad = parse_ra(params['RAJ'])
        dec_rad = parse_dec(params['DECJ'])
        pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
        pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
        posepoch = params.get('POSEPOCH', params['PEPOCH'])
        parallax_mas = params.get('PX', 0.0)
    
    # SSB position/velocity (includes Earth ephemeris lookup)
    with timer.stage(prefix + "compute_ssb_obs_pos_vel"):
        ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km)
    
    # Pulsar direction
    with timer.stage(prefix + "compute_pulsar_direction"):
        L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_mjd)
    
    # Roemer delay
    with timer.stage(prefix + "compute_roemer_delay"):
        roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)
    
    # Sun position for Shapiro
    with timer.stage(prefix + "sun ephemeris"):
        times = Time(tdb_mjd, format='mjd', scale='tdb')
        with solar_system_ephemeris.set('de440'):
            sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
    
    # Shapiro delay
    with timer.stage(prefix + "compute_shapiro_delay"):
        obs_sun_pos_km = sun_pos - ssb_obs_pos_km
        sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)
    
    roemer_shapiro = roemer_sec + sun_shapiro_sec
    
    # Barycentric frequency
    with timer.stage(prefix + "compute_barycentric_freq"):
        freq_bary_mhz = compute_barycentric_freq(freq_mhz, ssb_obs_vel_km_s, L_hat)
    
    # Prepare JAX arrays
    with timer.stage(prefix + "JAX array preparation"):
        import math
        tdb_jax = jnp.array(tdb_mjd, dtype=jnp.float64)
        freq_bary_jax = jnp.array(freq_bary_mhz, dtype=jnp.float64)
        obs_sun_jax = jnp.array(obs_sun_pos_km, dtype=jnp.float64)
        L_hat_jax = jnp.array(L_hat, dtype=jnp.float64)
        roemer_shapiro_jax = jnp.array(roemer_shapiro, dtype=jnp.float64)
        
        # DM coefficients
        dm_coeffs = []
        k = 0
        while True:
            key = 'DM' if k == 0 else f'DM{k}'
            if key in params:
                dm_coeffs.append(float(params[key]))
                k += 1
            else:
                break
        dm_coeffs = np.array(dm_coeffs if dm_coeffs else [0.0])
        dm_coeffs_jax = jnp.array(dm_coeffs, dtype=jnp.float64)
        dm_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(dm_coeffs))], dtype=jnp.float64)
        dm_epoch_jax = jnp.array(float(params.get('DMEPOCH', params['PEPOCH'])), dtype=jnp.float64)
        
        # FD parameters
        fd_coeffs = []
        fd_idx = 1
        while f'FD{fd_idx}' in params:
            fd_coeffs.append(float(params[f'FD{fd_idx}']))
            fd_idx += 1
        fd_coeffs_jax = jnp.array(fd_coeffs, dtype=jnp.float64) if fd_coeffs else jnp.array([0.0], dtype=jnp.float64)
        has_fd_jax = jnp.array(len(fd_coeffs) > 0)
        ne_sw_jax = jnp.array(float(params.get('NE_SW', 0.0)))
        
        # Binary params (ELL1)
        has_binary = 'PB' in params
        has_binary_jax = jnp.array(has_binary)
        if has_binary:
            pb_jax = jnp.array(float(params['PB']))
            a1_jax = jnp.array(float(params['A1']))
            tasc_jax = jnp.array(float(params.get('TASC', 0.0)))
            eps1_jax = jnp.array(float(params.get('EPS1', 0.0)))
            eps2_jax = jnp.array(float(params.get('EPS2', 0.0)))
            pbdot_jax = jnp.array(float(params.get('PBDOT', 0.0)))
            xdot_jax = jnp.array(float(params.get('XDOT', 0.0)))
            gamma_jax = jnp.array(float(params.get('GAMMA', 0.0)))
            
            # Shapiro
            if 'M2' in params and 'SINI' in params:
                M2 = float(params['M2'])
                SINI = float(params['SINI'])
                r_shap_jax = jnp.array(T_SUN_SEC * M2)
                s_shap_jax = jnp.array(SINI)
            else:
                r_shap_jax = jnp.array(0.0)
                s_shap_jax = jnp.array(0.0)
        else:
            pb_jax = a1_jax = tasc_jax = eps1_jax = eps2_jax = jnp.array(0.0)
            pbdot_jax = xdot_jax = gamma_jax = r_shap_jax = s_shap_jax = jnp.array(0.0)
    
    # JAX delay kernel
    with timer.stage(prefix + "compute_total_delay_jax"):
        total_delay_jax = compute_total_delay_jax(
            tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
            dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
            ne_sw_jax, fd_coeffs_jax, has_fd_jax,
            roemer_shapiro_jax, has_binary_jax,
            pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax,
            pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax
        ).block_until_ready()
    
    # Phase computation
    with timer.stage(prefix + "phase computation"):
        total_delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble)
        
        F0 = get_longdouble(params, 'F0')
        F1 = get_longdouble(params, 'F1', default=0.0)
        F2 = get_longdouble(params, 'F2', default=0.0)
        PEPOCH = get_longdouble(params, 'PEPOCH')
        
        F1_half = F1 / np.longdouble(2.0)
        F2_sixth = F2 / np.longdouble(6.0)
        PEPOCH_sec = PEPOCH * np.longdouble(SECS_PER_DAY)
        
        tdb_mjd_ld = np.array(tdb_mjd, dtype=np.longdouble)
        tdb_sec = tdb_mjd_ld * np.longdouble(SECS_PER_DAY)
        dt_sec = tdb_sec - PEPOCH_sec - total_delay_sec
        
        phase = dt_sec * (F0 + dt_sec * (F1_half + dt_sec * F2_sixth))
        phase_wrapped = phase - np.round(phase)
    
    # RMS calculation
    with timer.stage(prefix + "RMS calculation"):
        residuals_us = np.asarray(phase_wrapped / F0 * 1e6, dtype=np.float64)
        weights = 1.0 / (errors_us ** 2)
        weighted_mean = np.sum(residuals_us * weights) / np.sum(weights)
        residuals_us = residuals_us - weighted_mean
        rms_us = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    
    return {
        'residuals_us': residuals_us,
        'rms_us': float(rms_us),
        'tdb_mjd': np.array(tdb_mjd, dtype=np.float64),
        'dt_sec': np.array(dt_sec, dtype=np.float64),
        'freq_bary_mhz': np.array(freq_bary_mhz, dtype=np.float64),
        'errors_us': errors_us,
        'n_toas': len(residuals_us)
    }


def benchmark_fit(
    timer: StageTimer,
    par_file: Path,
    tim_file: Path,
    fit_params: List[str],
    run_label: str = ""
) -> dict:
    """Benchmark fitting with detailed stage breakdown."""
    import numpy as np
    from jug.engine.session import TimingSession
    from jug.fitting.optimized_fitter import (
        _build_general_fit_setup_from_cache,
        _run_general_fit_iterations,
        GeneralFitSetup
    )
    from jug.fitting.wls_fitter import wls_solve_svd
    from jug.fitting.derivatives_spin import compute_spin_derivatives
    from jug.fitting.derivatives_dm import compute_dm_derivatives
    
    prefix = f"fit{run_label}: "
    
    # Create session
    with timer.stage(prefix + "TimingSession creation"):
        session = TimingSession(par_file, tim_file, verbose=False)
    
    # Compute residuals (needed for cache)
    with timer.stage(prefix + "initial residuals (cache)"):
        result = session.compute_residuals(subtract_tzr=False)
    
    # Get cached data
    cached_result = session._cached_result_by_mode.get(False)
    
    # Build setup from cache
    with timer.stage(prefix + "_build_general_fit_setup_from_cache"):
        toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
        errors_us = np.array([toa.error_us for toa in session.toas_data])
        
        session_cached_data = {
            'dt_sec': cached_result['dt_sec'],
            'tdb_mjd': cached_result['tdb_mjd'],
            'freq_bary_mhz': cached_result['freq_bary_mhz'],
            'toas_mjd': toas_mjd,
            'errors_us': errors_us
        }
        
        setup = _build_general_fit_setup_from_cache(
            session_cached_data,
            session.params,
            fit_params,
            toa_mask=None
        )
    
    # Run iterations with detailed timing
    params = dict(setup.params)
    param_values_curr = list(setup.param_values_start)
    weights = setup.weights
    errors_sec = setup.errors_sec
    dt_sec_cached = setup.dt_sec_cached
    tdb_mjd = setup.tdb_mjd
    freq_mhz = setup.freq_mhz
    
    sum_weights = np.sum(weights)
    
    iteration_times = []
    for iteration in range(5):  # Cap at 5 iterations for benchmark
        iter_start = time.perf_counter()
        
        # Update params
        for i, param in enumerate(fit_params):
            params[param] = param_values_curr[i]
        
        # Residual evaluation
        with timer.stage(prefix + "residual evaluation"):
            f0 = params['F0']
            f_values = [params.get(f'F{i}', 0.0) for i in range(10)]
            f_values = [v for v in f_values if v != 0.0 or f_values.index(v) == 0]
            
            phase = np.zeros_like(dt_sec_cached)
            factorial = 1.0
            for n, f_val in enumerate(f_values):
                factorial *= (n + 1) if n > 0 else 1.0
                phase += f_val * (dt_sec_cached ** (n + 1)) / factorial
            
            phase_wrapped = phase - np.round(phase)
            residuals = phase_wrapped / f0
            weighted_mean = np.sum(residuals * weights) / sum_weights
            residuals = residuals - weighted_mean
        
        # Derivative computation
        with timer.stage(prefix + "derivative computation"):
            spin_params_list = [p for p in fit_params if p.startswith('F')]
            dm_params_list = [p for p in fit_params if p.startswith('DM')]
            
            M_columns = []
            if spin_params_list:
                spin_derivs = compute_spin_derivatives(params, tdb_mjd, spin_params_list)
                for p in spin_params_list:
                    M_columns.append(spin_derivs[p])
            
            if dm_params_list:
                dm_derivs = compute_dm_derivatives(params, tdb_mjd, freq_mhz, dm_params_list)
                for p in dm_params_list:
                    M_columns.append(dm_derivs[p])
        
        # Design matrix assembly
        with timer.stage(prefix + "design matrix assembly"):
            M = np.column_stack(M_columns)
            for i in range(M.shape[1]):
                col_mean = np.sum(M[:, i] * weights) / sum_weights
                M[:, i] = M[:, i] - col_mean
        
        # Solver
        with timer.stage(prefix + "wls_solve_svd"):
            delta_params, cov, _ = wls_solve_svd(
                residuals=residuals,
                sigma=errors_sec,
                M=M,
                threshold=1e-14,
                negate_dpars=False
            )
            delta_params = np.array(delta_params)
        
        # Update parameters
        param_values_curr = [param_values_curr[i] + delta_params[i] for i in range(len(fit_params))]
        
        iter_time = time.perf_counter() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence
        if np.max(np.abs(delta_params)) < 1e-14:
            break
    
    timer.times[prefix + "total iterations"] = sum(iteration_times)
    timer.counts[prefix + "total iterations"] = len(iteration_times)
    
    # Compute final RMS
    rms_us = np.sqrt(np.sum(residuals**2 * weights) / sum_weights) * 1e6
    
    return {
        'final_params': {param: params[param] for param in fit_params},
        'final_rms': float(rms_us),
        'iterations': len(iteration_times),
        'iteration_times_ms': [t * 1000 for t in iteration_times]
    }


def run_benchmark(par_file: Path, tim_file: Path, output_file: Optional[Path] = None):
    """Run complete benchmark suite."""
    
    results = {
        "par_file": str(par_file),
        "tim_file": str(tim_file),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cold_run": {},
        "warm_run": {}
    }
    
    # =========================================================================
    # COLD RUN (fresh state)
    # =========================================================================
    print("\n" + "=" * 70)
    print("COLD RUN (includes JIT compilation, cache miss)")
    print("=" * 70)
    
    cold_timer = StageTimer()
    
    # Imports
    print("\n1. Benchmarking imports...")
    benchmark_imports(cold_timer)
    
    # Session creation
    print("2. Benchmarking session creation...")
    params, toas_data = benchmark_session_creation(cold_timer, par_file, tim_file)
    
    # Residuals (first call - JIT compilation)
    print("3. Benchmarking residual computation (cold)...")
    res_result = benchmark_residuals_detailed(
        cold_timer, par_file, tim_file, params, toas_data, " (cold)"
    )
    print(f"   RMS: {res_result['rms_us']:.6f} mus, TOAs: {res_result['n_toas']}")
    
    # Fit
    print("4. Benchmarking fit (cold)...")
    fit_result = benchmark_fit(cold_timer, par_file, tim_file, ['F0', 'F1'], " (cold)")
    print(f"   Final RMS: {fit_result['final_rms']:.6f} mus, Iterations: {fit_result['iterations']}")
    
    print("\n" + cold_timer.report("COLD RUN TIMING"))
    results["cold_run"] = cold_timer.to_dict()
    results["cold_run"]["rms_us"] = res_result['rms_us']
    results["cold_run"]["n_toas"] = res_result['n_toas']
    
    # =========================================================================
    # WARM RUN (JIT cached, imports done)
    # =========================================================================
    print("\n" + "=" * 70)
    print("WARM RUN (JIT cached, imports done)")
    print("=" * 70)
    
    warm_timer = StageTimer()
    
    # Skip imports (already done)
    
    # Session creation (re-parse files)
    print("\n1. Benchmarking session creation...")
    params2, toas_data2 = benchmark_session_creation(warm_timer, par_file, tim_file)
    
    # Residuals (second call - JIT warm)
    print("2. Benchmarking residual computation (warm)...")
    res_result2 = benchmark_residuals_detailed(
        warm_timer, par_file, tim_file, params2, toas_data2, " (warm)"
    )
    print(f"   RMS: {res_result2['rms_us']:.6f} mus")
    
    # Fit (warm)
    print("3. Benchmarking fit (warm)...")
    fit_result2 = benchmark_fit(warm_timer, par_file, tim_file, ['F0', 'F1'], " (warm)")
    print(f"   Final RMS: {fit_result2['final_rms']:.6f} mus, Iterations: {fit_result2['iterations']}")
    
    print("\n" + warm_timer.report("WARM RUN TIMING"))
    results["warm_run"] = warm_timer.to_dict()
    results["warm_run"]["rms_us"] = res_result2['rms_us']
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: COLD vs WARM")
    print("=" * 70)
    
    cold_total = cold_timer.total()
    warm_total = warm_timer.total()
    
    print(f"{'Metric':<40} {'Cold':>12} {'Warm':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Total time':<40} {cold_total*1000:>10.1f} ms {warm_total*1000:>10.1f} ms {cold_total/warm_total:>9.2f}x")
    
    # Compare key stages
    key_stages = [
        ("residuals (cold): compute_total_delay_jax", "residuals (warm): compute_total_delay_jax"),
        ("residuals (cold): compute_ssb_obs_pos_vel", "residuals (warm): compute_ssb_obs_pos_vel"),
        ("fit (cold): wls_solve_svd", "fit (warm): wls_solve_svd"),
    ]
    
    for cold_key, warm_key in key_stages:
        cold_t = cold_timer.get_time(cold_key)
        warm_t = warm_timer.get_time(warm_key)
        if cold_t > 0 and warm_t > 0:
            name = cold_key.replace(" (cold)", "")
            speedup = cold_t / warm_t if warm_t > 0 else 0
            print(f"{name:<40} {cold_t*1000:>10.1f} ms {warm_t*1000:>10.1f} ms {speedup:>9.2f}x")
    
    print("=" * 70)
    
    # Identify bottlenecks
    print("\n" + "=" * 70)
    print("TOP BOTTLENECKS (Cold Run)")
    print("=" * 70)
    sorted_cold = sorted(cold_timer.times.items(), key=lambda x: -x[1])[:10]
    for name, t in sorted_cold:
        pct = t / cold_total * 100
        print(f"  {name:<50} {t*1000:>8.1f} ms ({pct:>5.1f}%)")
    
    print("\n" + "=" * 70)
    print("TOP BOTTLENECKS (Warm Run)")
    print("=" * 70)
    sorted_warm = sorted(warm_timer.times.items(), key=lambda x: -x[1])[:10]
    for name, t in sorted_warm:
        pct = t / warm_total * 100
        print(f"  {name:<50} {t*1000:>8.1f} ms ({pct:>5.1f}%)")
    
    # Save JSON results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='JUG Stage-by-Stage Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--par',
        type=Path,
        default=Path("data/pulsars/J1909-3744_tdb_wrong.par"),
        help='Path to .par file'
    )
    parser.add_argument(
        '--tim',
        type=Path,
        default=Path("data/pulsars/J1909-3744.tim"),
        help='Path to .tim file'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path("bench_results.json"),
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not args.par.exists():
        print(f"ERROR: Par file not found: {args.par}")
        sys.exit(1)
    if not args.tim.exists():
        print(f"ERROR: Tim file not found: {args.tim}")
        sys.exit(1)
    
    print("=" * 70)
    print("JUG Stage-by-Stage Benchmark")
    print("=" * 70)
    print(f"Par file: {args.par}")
    print(f"Tim file: {args.tim}")
    print(f"Output:   {args.output}")
    
    run_benchmark(args.par, args.tim, args.output)


if __name__ == '__main__':
    main()
