"""Shared helpers for tempo2-native chain dev_oracle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
from jug.utils.constants import SECS_PER_DAY


def load_wsrt167_fixture():
    from tempo2_fixtures import get_tempo2_fixture

    return get_tempo2_fixture("wsrt167")


def compute_native_terms_for_fixture(fixture: dict) -> Any:
    """Build native JAX terms from a tempo2 fixture par/tim pair."""
    from jug.utils.jax_setup import ensure_jax_x64

    ensure_jax_x64()
    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility="tempo2",
        skip_native_bclt_overlay=True,
    )
    return prepare_tempo2_chain_from_simple_result(
        jug,
        params,
        toas,
    )


def compute_native_terms_model_epoch(fixture: dict) -> Any:
    """Alias for the unified JAX path."""
    return compute_native_terms_for_fixture(fixture)


def native_batcorr_days(native_terms) -> np.ndarray:
    import jax

    bat = jax.device_get(native_terms.bat_corr_day + native_terms.bat_corr_day_residual)
    return np.asarray(bat, dtype=np.float64)


def delta_ns(a, b, *, is_mjd: bool = False) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if is_mjd:
        x = x * SECS_PER_DAY
    return x * 1e9


def rms_ns(a, b, *, is_mjd: bool = False) -> float:
    """RMS difference in nanoseconds."""
    return float(np.sqrt(np.mean(delta_ns(a, b, is_mjd=is_mjd) ** 2)))


def rms_cm(a_ls, b_ls) -> float:
    """RMS vector difference in centimetres (inputs in light-seconds)."""
    from jug.utils.constants import C_KM_S

    diff = np.asarray(a_ls, dtype=np.float64) - np.asarray(b_ls, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))) * C_KM_S * 100)
