"""Shared helpers for tempo2-native chain dev_oracle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2_native.chain_jax import prepare_native_chain_from_simple_result
from jug.utils.constants import SECS_PER_DAY


def load_wsrt167_fixture():
    from tempo2_fixtures import get_tempo2_fixture

    return get_tempo2_fixture("wsrt167")


def compute_native_terms_for_fixture(fixture: dict) -> Any:
    """Build native JAX terms from a tempo2 fixture par/tim pair."""
    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility="tempo2"
    )
    obs_earth = np.zeros((len(toas), 3), dtype=np.float64)
    from jug.utils.constants import OBSERVATORIES

    for i, toa in enumerate(toas):
        loc = OBSERVATORIES.get(toa.observatory.lower())
        if loc is not None:
            obs_earth[i] = loc
    vel = jug.get("earth_ssb_vel_km_s", jug["ssb_obs_vel_km_s"])
    return prepare_native_chain_from_simple_result(
        jug,
        params,
        toas,
        observatory_earth_km=obs_earth,
        earth_ssb_km=jug["ssb_obs_pos_km"],
        earth_ssb_vel_km_s=vel,
    )


def compute_native_terms_model_epoch(fixture: dict) -> Any:
    """Interim IFTE model-epoch batCorr path (~286 ns batCorr on wsrt167)."""
    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility="tempo2"
    )
    obs_earth = np.zeros((len(toas), 3), dtype=np.float64)
    from jug.utils.constants import OBSERVATORIES

    for i, toa in enumerate(toas):
        loc = OBSERVATORIES.get(toa.observatory.lower())
        if loc is not None:
            obs_earth[i] = loc
    vel = jug.get("earth_ssb_vel_km_s", jug["ssb_obs_vel_km_s"])
    return prepare_native_chain_from_simple_result(
        jug,
        params,
        toas,
        observatory_earth_km=obs_earth,
        earth_ssb_km=jug["ssb_obs_pos_km"],
        earth_ssb_vel_km_s=vel,
        use_model_epoch_batcorr=True,
    )


def native_batcorr_days(native_terms) -> np.ndarray:
    import jax

    bat = jax.device_get(native_terms.bat_corr_day + native_terms.bat_corr_day_residual)
    return np.asarray(bat, dtype=np.float64)


def delta_ns(a, b, *, is_mjd: bool = False) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if is_mjd:
        x = x * SECS_PER_DAY
    return x * 1e9
