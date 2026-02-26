#!/usr/bin/env python
"""Benchmark the fitting pipeline -- profile where time is spent.

Usage:
    JAX_PLATFORMS=cpu conda run -n discotech python jug/tests/benchmark_fit.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import numpy as np
import time
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

# -- Data paths -----------------------------------------------------------
PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/"
           "32ch_tdb_ads/J0125-2327_tdb.par")
TIM = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/"
           "32ch_tdb_ads/J0125-2327.tim")

# Fallback to test data if MPTA data not available
if not PAR.exists():
    PAR = Path("data/pulsars/J1909-3744_tdb.par")
    TIM = Path("data/pulsars/J1909-3744.tim")

assert PAR.exists(), f"Par file not found: {PAR}"
assert TIM.exists(), f"Tim file not found: {TIM}"

# -- Timer utility --------------------------------------------------------
timings = defaultdict(list)

@contextmanager
def timed(label, skip=False):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if not skip:
        timings[label].append(dt)


def print_timings(title=""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    # Sort by total time descending
    rows = []
    for label, times in timings.items():
        total = sum(times)
        n = len(times)
        mean = total / n
        rows.append((total, mean, n, label))
    rows.sort(reverse=True)
    total_all = sum(r[0] for r in rows)
    
    print(f"\n{'Label':<45} {'Total':>8} {'Mean':>8} {'N':>4} {'%':>6}")
    print("-" * 75)
    for total, mean, n, label in rows:
        pct = 100 * total / total_all if total_all > 0 else 0
        print(f"{label:<45} {total*1000:>7.1f}ms {mean*1000:>7.2f}ms {n:>4} {pct:>5.1f}%")
    print("-" * 75)
    print(f"{'TOTAL':<45} {total_all*1000:>7.1f}ms")


# -- 1. Setup phase (one-off) --------------------------------------------
print(f"Pulsar: {PAR.stem}")
print(f"Tim:    {TIM}")

from jug.engine.session import TimingSession

with timed("session_init"):
    session = TimingSession(PAR, TIM, verbose=False)

with timed("compute_residuals"):
    result = session.compute_residuals(subtract_tzr=False)

n_toas = len(result['residuals_us'])
print(f"N_TOAs: {n_toas}")

# -- 2. Get the fit parameters the user would be fitting ------------------
# Read directly from par file -- parameters with fit flag "1" after value
fit_params = []
with open(PAR) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 3 and parts[-2] == '1':
            name = parts[0]
            if name not in ('MODE', 'NITS', 'NTOA', 'EPHVER', 'SWM'):
                fit_params.append(name)

# Fallback: if parsing didn't find any, use a standard set
if not fit_params:
    fit_params = ['F0', 'F1', 'DM']

print(f"Fit params ({len(fit_params)}): {fit_params}")

# -- 3. Full fit timing (end-to-end) -------------------------------------
print("\n--- Warmup fit (JIT compilation) ---")
with timed("fit_warmup"):
    session.fit_parameters(fit_params=fit_params, max_iter=1, verbose=False)

# Reset session for clean benchmark
session = TimingSession(PAR, TIM, verbose=False)
session.compute_residuals(subtract_tzr=False)

print("\n--- Benchmark fit (3 iterations) ---")
with timed("fit_3iter"):
    r = session.fit_parameters(fit_params=fit_params, max_iter=3, verbose=False)
print(f"  Final RMS: {r.get('rms_us', r.get('final_rms', '?'))}")

# -- 4. Profile individual components ------------------------------------
# Re-import internals to profile them directly
from jug.fitting.optimized_fitter import (
    _compute_full_model_residuals,
)
from jug.residuals.simple_calculator import compute_phase_residuals
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.derivatives_dm import compute_dm_derivatives
from jug.fitting.derivatives_astrometry import (
    compute_astrometry_derivatives, compute_astrometric_delay
)
from jug.fitting.derivatives_fd import compute_fd_derivatives, compute_fd_delay
from jug.fitting.derivatives_sw import compute_sw_derivatives
from jug.fitting.binary_registry import compute_binary_delay, compute_binary_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.model.parameter_spec import (
    get_spin_params_from_list, get_dm_params_from_list,
    get_binary_params_from_list, get_astrometry_params_from_list,
    get_fd_params_from_list, get_sw_params_from_list,
)
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY
import jax.numpy as jnp

print("\n--- Component profiling (10 reps each) ---")

# Rebuild setup using the cached path (same as what the GUI uses)
from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache

session2 = TimingSession(PAR, TIM, verbose=False)
result2 = session2.compute_residuals(subtract_tzr=False)

# Build cached data dict exactly like session.fit_parameters does
cached_result = session2._cached_result_by_mode.get(False)
toas_mjd_raw = np.array([toa.mjd_int + toa.mjd_frac for toa in session2.toas_data])
errors_us_raw = np.array([toa.error_us for toa in session2.toas_data])
toa_flags_raw = [toa.flags for toa in session2.toas_data]

session_cached_data = {
    'dt_sec': cached_result['dt_sec'],
    'dt_sec_ld': cached_result.get('dt_sec_ld'),
    'tdb_mjd': cached_result['tdb_mjd'],
    'freq_bary_mhz': cached_result['freq_bary_mhz'],
    'toas_mjd': toas_mjd_raw,
    'errors_us': errors_us_raw,
    'toa_flags': toa_flags_raw,
    'roemer_shapiro_sec': cached_result.get('roemer_shapiro_sec'),
    'prebinary_delay_sec': cached_result.get('prebinary_delay_sec'),
    'ssb_obs_pos_ls': cached_result.get('ssb_obs_pos_ls'),
    'sw_geometry_pc': cached_result.get('sw_geometry_pc'),
}

with timed("build_fit_setup"):
    setup = _build_general_fit_setup_from_cache(
        session_cached_data, session2.params, fit_params
    )

# Extract arrays
toas_mjd = setup.toas_mjd
freq_mhz = setup.freq_mhz
dt_sec_ld = setup.dt_sec_ld
weights = setup.weights
errors_sec = setup.errors_sec
ssb_obs_pos_ls = setup.ssb_obs_pos_ls

params_dict = dict(setup.params)

spin_params_list = get_spin_params_from_list(fit_params)
dm_params_list = get_dm_params_from_list(fit_params)
binary_params_list = get_binary_params_from_list(fit_params)
astrometry_params_list = get_astrometry_params_from_list(fit_params)
fd_params_list = get_fd_params_from_list(fit_params)
sw_params_list = get_sw_params_from_list(fit_params)

print(f"  Spin: {spin_params_list}")
print(f"  DM:   {dm_params_list}")
print(f"  Binary: {binary_params_list}")
print(f"  Astro: {astrometry_params_list}")
print(f"  FD:   {fd_params_list}")
print(f"  SW:   {sw_params_list}")

N_REPS = 10
N_WARMUP = 2  # untimed warmup reps for JIT compilation

# Profile each component
for rep in range(-N_WARMUP, N_REPS):
    is_warmup = rep < 0
    # Phase residuals (longdouble path)
    dt_copy = dt_sec_ld.copy()
    with timed("phase_residuals", skip=is_warmup):
        res_us, res_sec = compute_phase_residuals(dt_copy, params_dict, weights)

    # Spin derivatives
    if spin_params_list:
        with timed("deriv_spin", skip=is_warmup):
            compute_spin_derivatives(params_dict, toas_mjd, spin_params_list)

    # DM derivatives
    if dm_params_list:
        with timed("deriv_dm", skip=is_warmup):
            compute_dm_derivatives(params_dict, toas_mjd, freq_mhz, dm_params_list)

    # Binary delay
    if binary_params_list:
        prebinary = setup.prebinary_delay_sec
        toas_pb = toas_mjd - prebinary / SECS_PER_DAY
        with timed("binary_delay", skip=is_warmup):
            compute_binary_delay(toas_pb, params_dict)
        with timed("deriv_binary", skip=is_warmup):
            compute_binary_derivatives(params_dict, toas_pb, binary_params_list,
                                       obs_pos_ls=ssb_obs_pos_ls if ssb_obs_pos_ls is not None else None)

    # Astrometry delay
    if astrometry_params_list:
        with timed("astro_delay", skip=is_warmup):
            compute_astrometric_delay(params_dict, toas_mjd, ssb_obs_pos_ls)
        with timed("deriv_astro", skip=is_warmup):
            compute_astrometry_derivatives(
                params_dict, toas_mjd, ssb_obs_pos_ls, astrometry_params_list
            )

    # FD
    if fd_params_list:
        fd_p = {p: params_dict[p] for p in fd_params_list if p in params_dict}
        with timed("fd_delay", skip=is_warmup):
            compute_fd_delay(freq_mhz, fd_p)
        with timed("deriv_fd", skip=is_warmup):
            compute_fd_derivatives(params_dict, freq_mhz, fd_params_list)

    # SW
    if sw_params_list and setup.sw_geometry_pc is not None:
        with timed("deriv_sw", skip=is_warmup):
            compute_sw_derivatives(setup.sw_geometry_pc, freq_mhz, sw_params_list)

    # Design matrix assembly + column mean subtraction
    M_cols = []
    all_derivs = {}
    all_derivs.update(compute_spin_derivatives(params_dict, toas_mjd, spin_params_list) if spin_params_list else {})
    all_derivs.update(compute_dm_derivatives(params_dict, toas_mjd, freq_mhz, dm_params_list) if dm_params_list else {})
    all_derivs.update(compute_astrometry_derivatives(params_dict, toas_mjd, ssb_obs_pos_ls, astrometry_params_list) if astrometry_params_list else {})
    all_derivs.update(compute_fd_derivatives(params_dict, freq_mhz, fd_params_list) if fd_params_list else {})
    for p in fit_params:
        if p in all_derivs:
            M_cols.append(np.asarray(all_derivs[p]))

    with timed("design_matrix_assemble", skip=is_warmup):
        M = np.column_stack(M_cols) if M_cols else np.zeros((n_toas, 0))

    with timed("col_mean_subtract", skip=is_warmup):
        sum_w = np.sum(weights)
        for i in range(M.shape[1]):
            col_mean = np.sum(M[:, i] * weights) / sum_w
            M[:, i] -= col_mean

    # ECORR whitening
    ecorr_w = getattr(setup, 'ecorr_whitener', None)
    if ecorr_w is not None:
        ecorr_w.prepare(errors_sec)
        with timed("ecorr_whiten_residuals", skip=is_warmup):
            ecorr_w.whiten_residuals(res_sec)
        with timed("ecorr_whiten_matrix", skip=is_warmup):
            ecorr_w.whiten_matrix(M)

    # WLS solve
    r_solve = res_sec if ecorr_w is None else ecorr_w.whiten_residuals(res_sec)
    M_solve = M if ecorr_w is None else ecorr_w.whiten_matrix(M)
    sigma_solve = errors_sec if ecorr_w is None else np.ones_like(errors_sec)

    with timed("wls_solve_svd", skip=is_warmup):
        wls_solve_svd(
            jnp.array(r_solve), jnp.array(sigma_solve),
            jnp.array(M_solve), threshold=1e-14
        )

    # Full model residuals (as called during damping)
    with timed("full_model_residuals", skip=is_warmup):
        _compute_full_model_residuals(params_dict, setup)


print_timings(f"Fitting Profile -- {PAR.stem} ({n_toas} TOAs, {len(fit_params)} params)")
