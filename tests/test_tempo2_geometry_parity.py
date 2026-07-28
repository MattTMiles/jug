"""DEV ORACLE — Tempo2ObservatoryState vector parity vs pytempo obsn fields."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.delays.tempo2_ephemeris import (
    bootstrap_tempo2_observatory_state,
    earth_geocenter_from_ssb_km,
    resolve_tempo2_ephemeris_path,
    tempo2_geometry_epochs,
    tempo2_read_ephemeris_au_scale,
    tempo2_read_ephemeris_mjd,
    _open_spk,
)
from jug.delays.tempo2_geometry import tempo2_observatory_chain_vectors, Tempo2ObservatoryState
from jug.io.par_reader import parse_par_file
from jug.utils.constants import C_KM_S
from tempo2_test_helpers import rms_cm, rms_ns

WSRT167_TRACE_INDICES = [0, 42, 85, 166]


def test_wsrt167_observatory_earth_parity(wsrt167_fixture, wsrt167_jug):
    obs = wsrt167_jug["term_diagnostics"]["tempo2_obs_state"]
    jug_obs_ls = np.asarray(obs["observatory_earth_km"], dtype=np.float64)[:, :3] / C_KM_S
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(
        parfile=str(wsrt167_fixture["par_path"]), timfile=str(wsrt167_fixture["tim_path"]), dofit=False
    )
    pt_obs_ls = np.asarray(psr.observatory_earth[:, :3], dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.sum((jug_obs_ls - pt_obs_ls) ** 2, axis=1))) * C_KM_S * 100)
    assert rms < 1.0


def test_wsrt167_rca_internal_consistency(wsrt167_jug):
    """``rca`` must match ``earth_ssb + observatory_earth`` from JUG SPK state."""
    obs = wsrt167_jug["term_diagnostics"]["tempo2_obs_state"]
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
    rms = float(np.sqrt(np.mean(np.sum((jug_rca_ls - direct_rca_ls) ** 2, axis=1))) * C_KM_S * 100)
    assert rms < 0.01


def test_wsrt167_earth_ssb_spk_self_consistency(wsrt167_fixture, wsrt167_jug):
    """JUG ``earth_ssb`` must match SPK × ``IFTE_K`` at ``readEphemeris`` epoch."""
    td = wsrt167_jug["term_diagnostics"]
    obs = td["tempo2_obs_state"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(td.get("formbats_correction_tt_sec", td["correction_tt_sec"]), dtype=np.float64)
    tt_teph = td.get("correction_tt_teph_sec")
    ephem_mjd = tempo2_read_ephemeris_mjd(
        sat,
        tt,
        correction_tt_teph_sec=None if tt_teph is None else np.asarray(tt_teph, dtype=np.float64),
    )
    params = parse_par_file(wsrt167_fixture["par_path"])
    ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    kernel = _open_spk(ephem_path)
    au_scale = tempo2_read_ephemeris_au_scale()
    jug_earth = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, :3]
    spk_earth = np.zeros_like(jug_earth)
    for i, mjd in enumerate(ephem_mjd):
        pos, _vel = earth_geocenter_from_ssb_km(kernel, float(mjd + 2400000.5))
        spk_earth[i] = pos * au_scale
    rms = float(np.sqrt(np.mean(np.sum((jug_earth - spk_earth) ** 2, axis=1))) * 100)
    assert rms < 1.0


def test_wsrt167_earth_ssb_pytempo_parity(wsrt167_fixture, wsrt167_jug):
    """``earth_ssb`` position parity vs pytempo obsn (light-seconds)."""
    obs = wsrt167_jug["term_diagnostics"]["tempo2_obs_state"]
    jug_earth_ls = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, :3] / C_KM_S
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(
        parfile=str(wsrt167_fixture["par_path"]), timfile=str(wsrt167_fixture["tim_path"]), dofit=False
    )
    pt_earth_ls = np.asarray(psr.earth_ssb[:, :3], dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.sum((jug_earth_ls - pt_earth_ls) ** 2, axis=1))) * C_KM_S * 100)
    assert rms < 1.0


def test_wsrt167_geometry_epochs_split(wsrt167_jug):
    """``site_mjd`` (TT) and ``ephemeris_mjd`` (TT+Teph) must differ when Teph is set."""
    td = wsrt167_jug["term_diagnostics"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(td.get("formbats_correction_tt_sec", td["correction_tt_sec"]), dtype=np.float64)
    tt_teph = np.asarray(td["correction_tt_teph_sec"], dtype=np.float64)
    site_mjd, ephem_mjd = tempo2_geometry_epochs(sat, tt, tt_teph)
    assert np.allclose(site_mjd + tt_teph / (86400.0), ephem_mjd)
    assert float(np.max(np.abs(ephem_mjd - site_mjd)) * 86400.0) > 0.0


def test_wsrt167_geometry_bootstrap_converges(wsrt167_fixture, wsrt167_jug):
    """Fixed-point Teph ↔ ephemeris bootstrap must converge (not silent iteration cap)."""
    params = parse_par_file(wsrt167_fixture["par_path"])
    td = wsrt167_jug["term_diagnostics"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(td.get("formbats_correction_tt_sec", td["correction_tt_sec"]), dtype=np.float64)
    from jug.utils.constants import OBSERVATORIES

    obs_itrf = OBSERVATORIES["wsrt"]
    ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    boot = bootstrap_tempo2_observatory_state(
        sat, tt, obs_itrf, ephem_path=ephem_path, params=params
    )
    assert boot.iterations >= 1
    assert boot.iterations <= 8


def test_wsrt167_geometry_terms_breakdown(
    wsrt167_fixture, wsrt167_jug, wsrt167_native_terms, wsrt167_pytempo_oracle
):
    """Per-term vector and BCLT decomposition vs pytempo (interpretable gates)."""
    import jax

    from pytempo.sandbox import tempopulsar

    obs = wsrt167_jug["term_diagnostics"]["tempo2_obs_state"]
    psr = tempopulsar(
        parfile=str(wsrt167_fixture["par_path"]), timfile=str(wsrt167_fixture["tim_path"]), dofit=False
    )

    jug_earth_ls = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, :3] / C_KM_S
    jug_obs_ls = np.asarray(obs["observatory_earth_km"], dtype=np.float64)[:, :3] / C_KM_S
    pt_earth_ls = np.asarray(psr.earth_ssb[:, :3], dtype=np.float64)
    pt_obs_ls = np.asarray(psr.observatory_earth[:, :3], dtype=np.float64)
    jug_rca_ls = jug_earth_ls + jug_obs_ls
    pt_rca_ls = pt_earth_ls + pt_obs_ls

    earth_cm = rms_cm(jug_earth_ls, pt_earth_ls)
    obs_cm = rms_cm(jug_obs_ls, pt_obs_ls)
    rca_cm = rms_cm(jug_rca_ls, pt_rca_ls)
    assert earth_cm < 0.2, f"earth_ssb RMS is {earth_cm:.3f} cm (Teph epoch coupling)"
    assert obs_cm < 0.01, f"observatory_earth RMS is {obs_cm:.4f} cm"
    assert rca_cm < 0.2, f"rca RMS is {rca_cm:.3f} cm"

    for name in ("roemer_sec", "dt_ssb_sec"):
        jug_term = np.asarray(jax.device_get(getattr(wsrt167_native_terms, name)), dtype=np.float64)
        pt_term = np.asarray(wsrt167_pytempo_oracle.fields[name], dtype=np.float64)
        term_rms_ns = rms_ns(jug_term, pt_term)
        assert term_rms_ns < 1.0, f"{name} RMS is {term_rms_ns:.3f} ns"
