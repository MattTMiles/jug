"""DEV ORACLE — BCLT term split vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    delta_ns,
    load_wsrt167_fixture,
    rms_ns,
)


def test_native_roemer_wsrt167_vs_pytempo():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    roemer = np.asarray(jax.device_get(native.roemer_sec), dtype=np.float64)
    delta = delta_ns(roemer, oracle.fields["roemer_sec"])
    roemer_rms = rms_ns(roemer, oracle.fields["roemer_sec"])
    # BCLT roemer uses fixed posPulsar + explicit PM terms (tempo2 calculate_bclt.C).
    assert roemer_rms < 1.0, (
        f"roemer_sec RMS is {roemer_rms:.3f} ns "
        "(geometry ~0.1 cm; remaining gap is tt2tb/Teph coupling — Phase 2)"
    )


def test_native_tdis1_wsrt167_vs_pytempo():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    tdis1 = np.asarray(jax.device_get(native.tdis1_sec), dtype=np.float64)
    tdis1_rms = rms_ns(tdis1, oracle.fields["tdis1_sec"])
    assert tdis1_rms < 1.0, f"tdis1_sec RMS is {tdis1_rms:.3f} ns vs pytempo"


def test_native_tdis2_wsrt167_vs_pytempo():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    tdis2 = np.asarray(jax.device_get(native.tdis2_sec), dtype=np.float64)
    tdis2_rms = rms_ns(tdis2, oracle.fields["tdis2_sec"])
    assert tdis2_rms < 1.0, f"tdis2_sec RMS is {tdis2_rms:.3f} ns vs pytempo"


def test_native_freq_ssb_wsrt167_vs_pytempo():
    """Literal ``dm_delays.C`` voverc + ``dilateFreq`` matches pytempo ``freqSSB``."""
    from pytempo.sandbox import tempopulsar

    from jug.delays.tempo2_geometry import (
        Tempo2ObservatoryState,
        build_tempo2_pulsar_vectors,
        psr_pos_at_delt,
    )
    from jug.io.par_reader import parse_par_file
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.utils.constants import C_KM_S, SECS_PER_DAY

    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    psr = tempopulsar(
        parfile=str(fixture["par_path"]),
        timfile=str(fixture["tim_path"]),
        dofit=False,
    )
    diag = psr.toa_diagnostics(removemean=False)
    freq_topo = np.asarray(diag["freq_mhz"], dtype=np.float64)
    freq_oracle = np.asarray(diag["freq_ssb_hz"], dtype=np.float64)
    einstein = np.asarray(diag["einstein_rate"], dtype=np.float64)
    voverc_oracle = 1.0 - (freq_oracle * einstein) / (freq_topo * 1e6)

    obs_state = jug["term_diagnostics"]["tempo2_obs_state"]
    state = Tempo2ObservatoryState(
        earth_ssb_km=np.asarray(obs_state["earth_ssb_km"]),
        observatory_earth_km=np.asarray(obs_state["observatory_earth_km"]),
        sun_ssb_km=np.asarray(obs_state["sun_ssb_km"]),
        planet_ssb_km={k: np.asarray(v) for k, v in obs_state["planet_ssb_km"].items()},
        site_vel_km_s=np.asarray(obs_state["site_vel_km_s"]),
    )
    pos_p, vel_p, _ = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=bool(params.get("_ecliptic_coords", False))
    )
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    sat = np.asarray(diag["sat_mjd"])
    tt = np.asarray(jug["term_diagnostics"]["correction_tt_sec"])
    tt_tb = np.asarray(jug["term_diagnostics"]["correction_tt_tb_sec"])
    dt_ssb = np.asarray(diag["dt_ssb_sec"])
    delt = (sat - posepoch + (tt + tt_tb + dt_ssb) / SECS_PER_DAY) / 36525.0
    vobs = state.earth_ssb_km[:, 3:6] / C_KM_S + state.site_vel_km_s / C_KM_S
    pos_all = np.stack([psr_pos_at_delt(pos_p, vel_p, float(d)) for d in delt])
    voverc_jug = np.sum(pos_all * vobs, axis=1)
    voverc_rms = float(np.sqrt(np.mean((voverc_oracle - voverc_jug) ** 2)))
    assert voverc_rms < 1.0e-10, f"voverc RMS is {voverc_rms:.3e} vs pytempo freqSSB"
