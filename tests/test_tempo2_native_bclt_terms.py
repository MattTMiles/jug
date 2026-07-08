"""DEV ORACLE — BCLT term split vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax

from tempo2_native_test_helpers import delta_ns, rms_ns


def test_native_roemer_wsrt167_vs_pytempo(wsrt167_native_terms, wsrt167_pytempo_oracle):
    roemer = np.asarray(jax.device_get(wsrt167_native_terms.roemer_sec), dtype=np.float64)
    roemer_rms = rms_ns(roemer, wsrt167_pytempo_oracle.fields["roemer_sec"])
    assert roemer_rms < 1.0, f"roemer_sec RMS is {roemer_rms:.3f} ns vs pytempo"


def test_native_tdis1_wsrt167_vs_pytempo(wsrt167_native_terms, wsrt167_pytempo_oracle):
    tdis1 = np.asarray(jax.device_get(wsrt167_native_terms.tdis1_sec), dtype=np.float64)
    tdis1_rms = rms_ns(tdis1, wsrt167_pytempo_oracle.fields["tdis1_sec"])
    assert tdis1_rms < 1.0, f"tdis1_sec RMS is {tdis1_rms:.3f} ns vs pytempo"


def test_native_tdis2_wsrt167_vs_pytempo(wsrt167_native_terms, wsrt167_pytempo_oracle):
    tdis2 = np.asarray(jax.device_get(wsrt167_native_terms.tdis2_sec), dtype=np.float64)
    tdis2_rms = rms_ns(tdis2, wsrt167_pytempo_oracle.fields["tdis2_sec"])
    assert tdis2_rms < 1.0, f"tdis2_sec RMS is {tdis2_rms:.3f} ns vs pytempo"


def test_native_freq_ssb_wsrt167_vs_pytempo(wsrt167_fixture, wsrt167_jug):
    """Literal ``dm_delays.C`` voverc + ``dilateFreq`` matches pytempo ``freqSSB``."""
    from pytempo.sandbox import tempopulsar

    from jug.delays.barycentric import compute_einstein_rate
    from jug.delays.tempo2_geometry import (
        Tempo2ObservatoryState,
        build_tempo2_pulsar_vectors,
        psr_pos_at_delt,
    )
    from jug.io.par_reader import parse_par_file
    from jug.utils.constants import C_KM_S, SECS_PER_DAY
    from jug.utils.timescales import parse_timescale

    params = parse_par_file(wsrt167_fixture["par_path"])
    jug = wsrt167_jug
    psr = tempopulsar(
        parfile=str(wsrt167_fixture["par_path"]),
        timfile=str(wsrt167_fixture["tim_path"]),
        dofit=False,
    )
    diag = psr.toa_diagnostics(removemean=False)
    td = jug["term_diagnostics"]
    obs = td["tempo2_obs_state"]
    state = Tempo2ObservatoryState(
        earth_ssb_km=np.asarray(obs["earth_ssb_km"], dtype=np.float64),
        observatory_earth_km=np.asarray(obs["observatory_earth_km"], dtype=np.float64),
        sun_ssb_km=np.asarray(obs["sun_ssb_km"], dtype=np.float64),
        planet_ssb_km={
            k: np.asarray(v, dtype=np.float64) for k, v in obs["planet_ssb_km"].items()
        },
        site_vel_km_s=np.asarray(obs["site_vel_km_s"], dtype=np.float64),
    )
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(td["correction_tt_sec"], dtype=np.float64)
    tt_tb = np.asarray(td["correction_tt_tb_sec"], dtype=np.float64)
    dt_ssb = np.asarray(psr.dt_ssb, dtype=np.float64)
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    pos, vel, _ = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=bool(params.get("_ecliptic_coords", False))
    )
    earth_vel = state.earth_ssb_km[:, 3:6]
    site_vel = state.site_vel_km_s
    vobs = earth_vel / C_KM_S + site_vel / C_KM_S
    from jug.io.tim_reader import parse_tim_file_mjds

    toas = parse_tim_file_mjds(wsrt167_fixture["tim_path"])
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    mjd_tt = sat + tt / SECS_PER_DAY
    einstein = np.asarray(
        compute_einstein_rate(mjd_tt, units=parse_timescale(params)), dtype=np.float64
    )
    obs_sun_ls = (state.sun_ssb_km[:, :3] - state.earth_ssb_km[:, :3] - state.observatory_earth_km[:, :3]) / C_KM_S
    freq_ssb_pt = np.asarray(diag["freq_ssb_hz"], dtype=np.float64) / 1.0e6
    for i in range(len(sat)):
        delt = (sat[i] - posepoch + (tt[i] + tt_tb[i] + dt_ssb[i]) / SECS_PER_DAY) / 36525.0
        psr_pos = psr_pos_at_delt(pos, vel, delt)
        rsa = -obs_sun_ls[i]
        ctheta = float(np.dot(psr_pos, rsa) / np.linalg.norm(rsa))
        voverc = float(np.dot(psr_pos, vobs[i]))
        freqf = freq_topo[i] * 1.0e6 * (1.0 - voverc)
        freqf /= einstein[i]
        jug_mhz = freqf / 1.0e6
        assert abs(jug_mhz - freq_ssb_pt[i]) < 1e-6, f"TOA {i}: {jug_mhz} vs {freq_ssb_pt[i]}"
