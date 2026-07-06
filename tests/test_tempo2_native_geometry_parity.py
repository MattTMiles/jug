"""DEV ORACLE — Tempo2ObservatoryState vector parity vs pytempo obsn fields."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.delays.tempo2_geometry import tempo2_observatory_chain_vectors, Tempo2ObservatoryState
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.constants import C_KM_S
from tempo2_native_test_helpers import load_wsrt167_fixture

WSRT167_TRACE_INDICES = [0, 42, 85, 166]


def test_wsrt167_observatory_earth_parity():
    fixture = load_wsrt167_fixture()
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    obs = jug["term_diagnostics"]["tempo2_obs_state"]
    jug_obs_ls = np.asarray(obs["observatory_earth_km"], dtype=np.float64)[:, :3] / C_KM_S
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False)
    pt_obs_ls = np.asarray(psr.observatory_earth[:, :3], dtype=np.float64)
    rms_cm = float(np.sqrt(np.mean(np.sum((jug_obs_ls - pt_obs_ls) ** 2, axis=1))) * C_KM_S * 100)
    assert rms_cm < 1.0


def test_wsrt167_rca_parity():
    fixture = load_wsrt167_fixture()
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    obs = jug["term_diagnostics"]["tempo2_obs_state"]
    jug_state = Tempo2ObservatoryState(
        earth_ssb_km=np.asarray(obs["earth_ssb_km"], dtype=np.float64),
        observatory_earth_km=np.asarray(obs["observatory_earth_km"], dtype=np.float64),
        sun_ssb_km=np.asarray(obs["sun_ssb_km"], dtype=np.float64),
        planet_ssb_km={
            k: np.asarray(v, dtype=np.float64) for k, v in obs["planet_ssb_km"].items()
        },
        site_vel_km_s=np.asarray(obs["site_vel_km_s"], dtype=np.float64),
    )
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False)
    pt_rca_ls = psr.earth_ssb[:, :3] + psr.observatory_earth[:, :3]
    _, jug_rca_ls, _, _ = tempo2_observatory_chain_vectors(jug_state)
    rms_km = float(np.sqrt(np.mean(np.sum((jug_rca_ls - pt_rca_ls) ** 2, axis=1))) * C_KM_S)
    assert rms_km < 5.0
