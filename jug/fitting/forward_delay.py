"""Unified timing delay-change forward model shared by NumPy and JAX paths."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from jug.fitting.binary_delay_plan import resolve_binary_structure
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY

if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import GeneralFitSetup


_EPOCH_KEYS = frozenset({"PEPOCH", "POSEPOCH", "DMEPOCH"})


def _assert_no_epoch_fit_params(fit_params):
    """Epochs are settings, never fit parameters."""
    fit_upper = {str(p).upper() for p in (fit_params or [])}
    bad = _EPOCH_KEYS & fit_upper
    if bad:
        raise ValueError(
            f"Epoch parameters {sorted(bad)} must not appear in fit_params; "
            "they are reference settings, not timing fit parameters."
        )


def pval(params, key, default=0.0):
    """Traceable scalar getter."""
    v = params.get(key, None)
    return default if v is None else v


def _dm_delay(xp, tdb_mjd, freq_mhz, dm_params, dm_epoch):
    """DM delay in seconds."""
    dm_coeffs = []
    dm_factorials = []
    for i in range(10):
        param = f"DM{i}" if i > 0 else "DM"
        if param in dm_params and dm_params[param] is not None:
            dm_coeffs.append(dm_params[param])
            dm_factorials.append(math.factorial(i))
        elif param == "DM":
            dm_coeffs.append(0.0)
            dm_factorials.append(1.0)
        else:
            break
    tdb = xp.asarray(tdb_mjd, dtype=xp.float64)
    dt_years = (tdb - dm_epoch) / 365.25
    dm_eff = xp.zeros_like(tdb)
    for i, (coeff, fac) in enumerate(zip(dm_coeffs, dm_factorials)):
        dm_eff = dm_eff + coeff * (dt_years ** i) / fac
    return K_DM_SEC * dm_eff / (xp.asarray(freq_mhz, dtype=xp.float64) ** 2)


def _fdjump_delay(xp, params, freq_mhz, fdjump_params, fdjump_masks):
    """Traceable FDJUMP delay."""
    freq_ghz = xp.asarray(freq_mhz, dtype=xp.float64) / 1000.0
    delay = xp.zeros_like(freq_ghz)
    for name in fdjump_params:
        meta = params.get(f"_fdjump_meta_{name}")
        if meta is None:
            continue
        fd_idx = meta["fd_index"]
        log_scale = meta.get("log_scale", True)
        value = pval(params, name, 0.0)
        freq_term = (xp.log(freq_ghz) ** fd_idx) if log_scale else (freq_ghz ** fd_idx)
        mask = xp.asarray(fdjump_masks.get(name, np.ones(len(freq_mhz), dtype=bool)))
        delay = delay + xp.where(mask, value * freq_term, 0.0)
    return delay


def compute_total_delay_change(params, setup, *, xp, binary_plan=None):
    """Sum of (new - initial) delay over deterministic delay terms."""
    tdb_mjd = xp.asarray(setup.tdb_mjd, dtype=xp.float64)
    freq_mhz = xp.asarray(setup.freq_mhz, dtype=xp.float64)
    delay_change = xp.zeros_like(tdb_mjd)

    if setup.dm_params and setup.initial_dm_delay is not None:
        dm_epoch = float(params.get("DMEPOCH", params.get("PEPOCH", 55000.0)))
        dm_dict = {}
        for p in setup.dm_params:
            key = "DM" if p in ("DM", "DM0") else p
            dm_dict[key] = pval(params, p, pval(params, key, 0.0))
        new_dm = _dm_delay(xp, tdb_mjd, freq_mhz, dm_dict, dm_epoch)
        delay_change = delay_change + (
            new_dm - xp.asarray(setup.initial_dm_delay, dtype=xp.float64)
        )

    if (
        setup.dmx_design_matrix is not None
        and setup.dmx_labels
        and setup.initial_dmx_delay is not None
    ):
        cur = xp.asarray([pval(params, lbl, 0.0) for lbl in setup.dmx_labels], dtype=xp.float64)
        matrix = xp.asarray(setup.dmx_design_matrix, dtype=xp.float64)
        new_dmx = matrix @ cur
        delay_change = delay_change + (
            new_dmx - xp.asarray(setup.initial_dmx_delay, dtype=xp.float64)
        )

    if setup.binary_params and setup.initial_binary_delay is not None:
        if setup.prebinary_delay_sec is None:
            raise ValueError("Binary delay-change requires prebinary_delay_sec in setup.")
        toas_prebinary = tdb_mjd - (
            xp.asarray(setup.prebinary_delay_sec, dtype=xp.float64) / SECS_PER_DAY
        )
        plan = binary_plan
        if plan is None:
            plan = resolve_binary_structure(
                setup.params, setup.fit_param_list, obs_pos_ls=setup.ssb_obs_pos_ls
            )
        new_binary = xp.asarray(
            plan.evaluate(toas_prebinary, params, setup.ssb_obs_pos_ls, xp),
            dtype=xp.float64,
        )
        delay_change = delay_change + (
            new_binary - xp.asarray(setup.initial_binary_delay, dtype=xp.float64)
        )

    if setup.astrometry_params and setup.initial_astrometric_delay is not None:
        from jug.fitting.derivatives_astrometry import compute_astrometric_delay

        new_astro = xp.asarray(
            compute_astrometric_delay(
                params,
                tdb_mjd,
                xp.asarray(setup.ssb_obs_pos_ls, dtype=xp.float64),
                obs_sun_pos_ls=(
                    None
                    if setup.obs_sun_pos_ls is None
                    else xp.asarray(setup.obs_sun_pos_ls, dtype=xp.float64)
                ),
                obs_planet_pos_ls=setup.obs_planet_pos_ls,
            ),
            dtype=xp.float64,
        )
        delay_change = delay_change + (
            new_astro - xp.asarray(setup.initial_astrometric_delay, dtype=xp.float64)
        )

    if setup.fd_params and setup.initial_fd_delay is not None:
        from jug.fitting.derivatives_fd import compute_fd_delay

        current_fd = {p: pval(params, p, 0.0) for p in setup.fd_params if p in params}
        new_fd = xp.asarray(compute_fd_delay(freq_mhz, current_fd), dtype=xp.float64)
        delay_change = delay_change + (
            new_fd - xp.asarray(setup.initial_fd_delay, dtype=xp.float64)
        )

    if setup.fdjump_params and setup.fdjump_masks:
        new_fdj = _fdjump_delay(xp, params, freq_mhz, setup.fdjump_params, setup.fdjump_masks)
        init_fdj = (
            0.0
            if setup.initial_fdjump_delay is None
            else xp.asarray(setup.initial_fdjump_delay, dtype=xp.float64)
        )
        delay_change = delay_change + (new_fdj - init_fdj)

    if setup.sw_params and setup.initial_sw_delay is not None:
        ne_sw = pval(params, "NE_SW", pval(params, "NE1AU", 0.0))
        sw_geom = xp.asarray(setup.sw_geometry_pc, dtype=xp.float64)
        new_sw = K_DM_SEC * ne_sw * sw_geom / (freq_mhz ** 2)
        delay_change = delay_change + (
            new_sw - xp.asarray(setup.initial_sw_delay, dtype=xp.float64)
        )

    return delay_change
