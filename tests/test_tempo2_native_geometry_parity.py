"""DEV ORACLE — Tempo2ObservatoryState vector parity vs pytempo obsn fields."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.delays.tempo2_ephemeris import (
    earth_geocenter_from_ssb_km,
    resolve_tempo2_ephemeris_path,
    tempo2_read_ephemeris_mjd,
    _open_spk,
)
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


def test_wsrt167_rca_internal_consistency():
    """``rca`` must match ``earth_ssb + observatory_earth`` from JUG SPK state."""
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
    _, jug_rca_ls, _, _ = tempo2_observatory_chain_vectors(jug_state)
    direct_rca_ls = (
        jug_state.earth_ssb_km[:, :3] + jug_state.observatory_earth_km[:, :3]
    ) / C_KM_S
    rms_cm = float(np.sqrt(np.mean(np.sum((jug_rca_ls - direct_rca_ls) ** 2, axis=1))) * C_KM_S * 100)
    assert rms_cm < 0.01


def test_wsrt167_earth_ssb_spk_self_consistency():
    """JUG ``earth_ssb`` must match a direct SPK lookup at ``readEphemeris`` epoch."""
    fixture = load_wsrt167_fixture()
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    td = jug["term_diagnostics"]
    obs = td["tempo2_obs_state"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(td.get("formbats_correction_tt_sec", td["correction_tt_sec"]), dtype=np.float64)
    tt_teph = td.get("correction_tt_teph_sec")
    ephem_mjd = tempo2_read_ephemeris_mjd(
        sat,
        tt,
        correction_tt_teph_sec=None if tt_teph is None else np.asarray(tt_teph, dtype=np.float64),
    )
    from jug.io.par_reader import parse_par_file

    params = parse_par_file(fixture["par_path"])
    ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    kernel = _open_spk(ephem_path)
    jug_earth = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, :3]
    spk_earth = np.zeros_like(jug_earth)
    for i, mjd in enumerate(ephem_mjd):
        pos, _vel = earth_geocenter_from_ssb_km(kernel, float(mjd + 2400000.5))
        spk_earth[i] = pos
    rms_cm = float(np.sqrt(np.mean(np.sum((jug_earth - spk_earth) ** 2, axis=1))) * 100)
    assert rms_cm < 1.0


@pytest.mark.xfail(reason="pytempo earth_ssb export is offset from direct jpl_pleph")
def test_wsrt167_earth_ssb_pytempo_export_diagnostic():
    """Document pytempo obsn earth_ssb offset vs JUG SPK path (~232 cm on wsrt167)."""
    fixture = load_wsrt167_fixture()
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    obs = jug["term_diagnostics"]["tempo2_obs_state"]
    jug_earth = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, :3]
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False)
    pt_earth = np.asarray(psr.earth_ssb[:, :3], dtype=np.float64)
    rms_cm = float(np.sqrt(np.mean(np.sum((jug_earth - pt_earth) ** 2, axis=1))) * 100)
    assert rms_cm < 1.0
