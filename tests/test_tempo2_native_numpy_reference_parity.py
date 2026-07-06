"""NumPy reference dev tests — delete with chain_numpy.py."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.residuals.tempo2_native.chain_numpy import compute_tempo2_native_terms_numpy
from tempo2_native_test_helpers import load_wsrt167_fixture


@pytest.fixture(autouse=True)
def _enable_numpy_native_chain(monkeypatch):
    monkeypatch.setenv("JUG_DEV_NUMPY_TEMPO2_CHAIN", "1")


def test_numpy_reference_matches_jax_native_terms():
    fixture = load_wsrt167_fixture()
    from jug.io.par_reader import parse_par_file
    from jug.residuals.simple_calculator import compute_residuals_simple
    from tempo2_native_test_helpers import compute_native_terms_for_fixture

    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    td = jug["term_diagnostics"]
    params = parse_par_file(fixture["par_path"])
    from jug.utils.constants import OBSERVATORIES
    from jug.io.tim_reader import parse_tim_file_mjds

    toas = parse_tim_file_mjds(fixture["tim_path"])
    obs_earth = np.zeros((len(toas), 3), dtype=np.float64)
    for i, toa in enumerate(toas):
        loc = OBSERVATORIES.get(toa.observatory.lower())
        if loc is not None:
            obs_earth[i] = loc
    tdis1 = np.asarray(td["dm_delay_sec"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np_terms = compute_tempo2_native_terms_numpy(
            sat_mjd=td["sat_mjd"],
            correction_tt_sec=td["correction_tt_sec"],
            params=params,
            ssb_obs_pos_km=jug["ssb_obs_pos_km"],
            observatory_earth_km=obs_earth,
            earth_ssb_vel_km_s=jug["ssb_obs_vel_km_s"],
            ephem_path=None,
            tdis1_sec=tdis1,
            tdis2_sec=td["sw_delay_sec"],
            tropospheric_sec=td["tropo_delay_sec"],
            dt_emission_sec=jug["dt_sec"],
            use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
            utc_to_tdb_sec=td.get("utc_to_tdb_sec"),
            formbats_tt_sec=td.get("formbats_correction_tt_sec"),
            ssb_obs_ls_fixed=jug["ssb_obs_pos_ls"],
            obs_sun_ls_fixed=jug["obs_sun_pos_ls"],
            obs_planets_ls_fixed=jug.get("obs_planet_pos_ls"),
            freq_mhz_topocentric=np.array([t.freq_mhz for t in toas], dtype=np.float64),
            planet_shapiro_enabled=True,
        )
    jax_native = compute_native_terms_for_fixture(fixture)
    bat_np = np_terms["bat_corr_day"] + np_terms["bat_corr_day_residual"]
    bat_jax = np.asarray(
        jax.device_get(jax_native.bat_corr_day + jax_native.bat_corr_day_residual)
    )
    delta_ns = (bat_jax - bat_np) * 86400.0 * 1e9
    assert np.sqrt(np.mean(delta_ns**2)) < 1.0
