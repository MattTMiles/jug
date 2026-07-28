"""Regression: tempo2_obs_state must survive native overlay and fit-setup cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from tempo2_test_helpers import load_wsrt167_fixture

pytestmark = [pytest.mark.tempo2]

def test_tempo2_obs_state_in_overlay_payload():
    fixture = load_wsrt167_fixture()
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=False,
    )
    td = result["term_diagnostics"]
    assert "tempo2_obs_state" in td
    obs = td["tempo2_obs_state"]
    n = len(result["dt_sec"])
    for key in ("site_vel_km_s", "earth_ssb_km", "observatory_earth_km"):
        assert key in obs
        assert np.asarray(obs[key]).shape[0] == n


def test_cached_fit_setup_populates_native_chain_static():
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    toas_mjd = np.array([t.mjd_int + t.mjd_frac for t in toas])
    session_cached_data = {
        "dt_sec": result["dt_sec"],
        "dt_sec_ld": result.get("dt_sec_ld"),
        "tdb_mjd": result["tdb_mjd"],
        "freq_bary_mhz": result["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": np.array([t.error_us for t in toas]),
        "toa_flags": [t.flags for t in toas],
        "ssb_obs_pos_ls": result.get("ssb_obs_pos_ls"),
        "term_diagnostics": result.get("term_diagnostics"),
        "model_mjd": result.get("model_mjd"),
        "toas": toas,
    }
    setup = _build_general_fit_setup_from_cache(
        session_cached_data,
        params,
        ["F0", "DM"],
        compatibility="tempo2",
    )
    assert setup.native_chain_static is not None
    assert "tempo2_obs_state" in setup.native_chain_static["term_diagnostics"]
