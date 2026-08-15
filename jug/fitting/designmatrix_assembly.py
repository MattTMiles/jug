"""Unified analytic timing design-matrix assembly for WLS and export APIs.

Tempo2 and pint sessions share the same PINT-style simplified derivative blocks
(geometric astrometry, cold-plasma DM, Taylor spin columns, etc.).  The assembly
is independent of ``tempo2_native``; only native autodiff uses the JAX graph mode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Literal

import numpy as np

from jug.fitting.binary_registry import compute_binary_derivatives
from jug.fitting.derivatives_dm import compute_dm_derivatives
from jug.fitting.derivatives_fdjump import compute_fdjump_derivatives
from jug.fitting.derivatives_sw import compute_sw_derivatives
from jug.model.parameter_spec import (
    get_astrometry_params_from_list,
    get_binary_params_from_list,
    get_dm_params_from_list,
    get_fd_params_from_list,
    get_spin_params_from_list,
    get_sw_params_from_list,
    canonicalize_param_name,
    is_fdjump_param,
    is_jump_param,
)
from jug.utils.constants import SECS_PER_DAY
from jug.utils.units import native_derivative_to_fit_column

if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import FDColumnMode, GeneralFitSetup

OutputUnits = Literal["native", "fit"]


def _is_delay_derivative_fd_mode(fd_column_mode: str) -> bool:
    return fd_column_mode in ("tempo2_delay", "delay_only")


def _instantaneous_spin_frequency_hz(params: Dict[str, Any], tdb_mjd: np.ndarray) -> np.ndarray:
    """Evaluate f(t) = F0 + F1*dt + F2*dt^2/2 + ... in Hz."""
    pepoch = float(params.get("PEPOCH", tdb_mjd[0]))
    dt_sec = (np.asarray(tdb_mjd, dtype=np.float64) - pepoch) * SECS_PER_DAY
    freq = np.zeros_like(dt_sec, dtype=np.float64)
    for order in range(21):
        key = f"F{order}"
        if key not in params:
            if order == 0:
                freq.fill(1.0)
            break
        coeff = float(params[key])
        if order == 0:
            freq += coeff
        else:
            freq += coeff * (dt_sec ** order) / float(math.factorial(order))
    return freq


def compute_fd_derivatives_for_mode(
    *,
    params: Dict[str, Any],
    freq_mhz: np.ndarray,
    fit_params: List[str],
    tdb_mjd: np.ndarray,
    fd_column_mode: str,
) -> Dict[str, np.ndarray]:
    """Compute FD derivative columns with explicit convention dispatch."""
    from jug.fitting.derivatives_fd import compute_fd_derivatives

    base = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in compute_fd_derivatives(params, freq_mhz, fit_params).items()
    }
    if _is_delay_derivative_fd_mode(fd_column_mode):
        return base

    spin_freq = _instantaneous_spin_frequency_hz(params, tdb_mjd)
    f0 = float(params.get("F0", 1.0))
    scale = spin_freq / f0
    return {k: col * scale for k, col in base.items()}


def assemble_analytic_derivative_blocks(
    setup: "GeneralFitSetup",
    fit_params: List[str],
    *,
    params: Dict[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Return native-unit design-matrix columns as ``d(delay)/d(param)``."""
    from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives
    from jug.fitting.derivatives_spin import compute_spin_derivatives

    params = dict(params if params is not None else setup.params)
    tdb_mjd = np.asarray(setup.tdb_mjd, dtype=np.float64)
    freq_mhz = np.asarray(setup.freq_mhz, dtype=np.float64)

    spin_params = get_spin_params_from_list(fit_params)
    dm_params = get_dm_params_from_list(fit_params)
    dmx_params = [p for p in fit_params if setup.dmx_labels and p in setup.dmx_labels]
    binary_params = get_binary_params_from_list(fit_params)
    astrometry_params = get_astrometry_params_from_list(fit_params)
    fd_params = get_fd_params_from_list(fit_params)
    sw_params = get_sw_params_from_list(fit_params)
    jump_params = [p for p in fit_params if is_jump_param(p)]
    fdjump_params = [
        canonicalize_param_name(p) for p in fit_params if is_fdjump_param(p)
    ]

    derivs: Dict[str, np.ndarray] = {}

    if spin_params:
        dt_for_spin = (
            np.asarray(setup.dt_sec_ld, dtype=np.float64)
            if setup.dt_sec_ld is not None
            else np.asarray(setup.dt_sec_cached, dtype=np.float64)
        )
        derivs.update(
            compute_spin_derivatives(
                params,
                tdb_mjd,
                spin_params,
                compatibility=setup.compatibility,
                dt_sec=dt_for_spin,
            )
        )

    if dm_params:
        dm_derivs = compute_dm_derivatives(params, tdb_mjd, freq_mhz, dm_params)
        derivs.update({name: -np.asarray(col) for name, col in dm_derivs.items()})

    if dmx_params and setup.dmx_design_matrix is not None and setup.dmx_labels is not None:
        dmx_index = {label: idx for idx, label in enumerate(setup.dmx_labels)}
        for label in dmx_params:
            idx = dmx_index.get(label)
            if idx is not None:
                derivs[label] = np.asarray(setup.dmx_design_matrix[:, idx], dtype=np.float64)

    if binary_params:
        if setup.prebinary_delay_sec is None:
            raise ValueError(
                "Binary design-matrix columns require prebinary_delay_sec in setup."
            )
        toas_prebinary_mjd = tdb_mjd - setup.prebinary_delay_sec / SECS_PER_DAY
        derivs.update(
            compute_binary_derivatives(
                params,
                toas_prebinary_mjd,
                binary_params,
                obs_pos_ls=setup.ssb_obs_pos_ls,
            )
        )

    if astrometry_params:
        if setup.ssb_obs_pos_ls is None:
            raise ValueError("Astrometry design-matrix columns require ssb_obs_pos_ls.")
        derivs.update(
            compute_astrometry_derivatives(
                params,
                tdb_mjd,
                setup.ssb_obs_pos_ls,
                astrometry_params,
            )
        )

    if fd_params:
        derivs.update(
            compute_fd_derivatives_for_mode(
                params=params,
                freq_mhz=freq_mhz,
                fit_params=fd_params,
                tdb_mjd=tdb_mjd,
                fd_column_mode=setup.fd_column_mode,
            )
        )

    if sw_params:
        if setup.sw_geometry_pc is None:
            raise ValueError("Solar-wind design-matrix columns require sw_geometry_pc.")
        derivs.update(compute_sw_derivatives(setup.sw_geometry_pc, freq_mhz, sw_params))

    if jump_params and setup.jump_masks:
        for name in jump_params:
            mask = setup.jump_masks.get(name)
            if mask is not None:
                derivs[name] = -np.asarray(mask, dtype=np.float64)

    if fdjump_params and setup.fdjump_masks:
        derivs.update(
            compute_fdjump_derivatives(
                params,
                freq_mhz,
                fdjump_params,
                fdjump_masks=setup.fdjump_masks,
            )
        )

    return derivs


def assemble_analytic_designmatrix(
    setup: "GeneralFitSetup",
    fit_params: List[str],
    *,
    params: Dict[str, Any] | None = None,
    output_units: OutputUnits = "fit",
) -> np.ndarray:
    """Assemble analytic design-matrix columns in fit or native param units."""
    tdb_mjd = np.asarray(setup.tdb_mjd, dtype=np.float64)
    derivs = assemble_analytic_derivative_blocks(setup, fit_params, params=params)

    cols = []
    for param in fit_params:
        if param not in derivs:
            raise ValueError(
                f"No design-matrix derivative computed for parameter {param!r}."
            )
        col_native = np.asarray(derivs[param], dtype=np.float64)
        if output_units == "fit":
            col_native = np.asarray(
                native_derivative_to_fit_column(param, col_native),
                dtype=np.float64,
            )
        cols.append(col_native)

    return np.column_stack(cols) if cols else np.empty((len(tdb_mjd), 0), dtype=np.float64)