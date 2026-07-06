"""DEV ORACLE — granular formBats component closure using pytempo delay diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.testing.tempo2_formbats_closure import compare_formbats_components
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    delta_ns,
    load_wsrt167_fixture,
    native_batcorr_days,
)

WSRT167_TRACE_INDICES = [0, 42, 85, 166]


def test_pytempo_formbats_self_closure_wsrt167():
    fixture = load_wsrt167_fixture()
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    closure = np.abs(oracle.fields["bat_corr_closure_ns"])
    assert float(np.max(closure)) < 1.0


def test_jug_formbats_replay_with_pytempo_components_wsrt167():
    """JUG formBats algebra closes when all slots come from pytempo."""
    fixture = load_wsrt167_fixture()
    report = compare_formbats_components(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    assert report.jug_replay_all_pytempo_rms_ns < 1.0


def test_wsrt167_component_ranking_documents_tt_blocker():
    """Per-slot swap-one ranking against pytempo oracle."""
    fixture = load_wsrt167_fixture()
    report = compare_formbats_components(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    assert report.swap_one_rms_ns["tt"] < 1.0
    assert report.swap_one_rms_ns["roemer"] < 5.0
    assert report.swap_one_rms_ns["tdis2"] < 1.0
    assert report.component_rms_ns["tt"] < 1.0


def test_wsrt167_per_component_gates():
    fixture = load_wsrt167_fixture()
    report = compare_formbats_components(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    assert report.component_rms_ns["roemer"] < 5.0
    assert report.component_rms_ns["tdis2"] < 1.0
    assert report.component_rms_ns["tdis1"] < 100.0
    assert report.component_rms_ns["tt_tb"] < 100.0
    assert report.component_rms_ns["tropo"] < 100.0
    assert report.component_rms_ns["shap"] < 100.0


def test_native_strict_formbats_batcorr_wsrt167():
    """Strict native formBats path on wsrt167."""
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    delta = delta_ns(native_batcorr_days(native), oracle.fields["bat_corr_days"], is_mjd=True)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0


def test_native_bclt_roemer_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    roemer = np.asarray(jax.device_get(native.roemer_sec), dtype=np.float64)
    delta = delta_ns(roemer, oracle.fields["roemer_sec"])
    assert np.sqrt(np.mean(delta**2)) < 5.0


def test_native_dt_ssb_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    dt_ssb = np.asarray(jax.device_get(native.dt_ssb_sec), dtype=np.float64)
    delta = delta_ns(dt_ssb, oracle.fields["dt_ssb_sec"])
    assert np.sqrt(np.mean(delta**2)) < 5.0


def test_single_toa_formbats_trace_wsrt167():
    fixture = load_wsrt167_fixture()
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    pt = oracle.fields
    for idx in WSRT167_TRACE_INDICES:
        if int(pt["delay_corr"][idx]) != 1:
            continue
        tt = pt["correction_tt_sec"][idx]
        sec = tt + (
            pt["correction_tt_tb_sec"][idx]
            - pt["tropospheric_sec"][idx]
            + pt["roemer_sec"][idx]
            - pt["shapiro_delay_sec"][idx]
            - pt["tdis1_sec"][idx]
            - pt["tdis2_sec"][idx]
        )
        np.testing.assert_allclose(
            pt["bat_corr_days"][idx],
            sec / 86400.0,
            rtol=0,
            atol=1e-15,
        )


def test_model_epoch_batcorr_still_available_for_interim():
    fixture = load_wsrt167_fixture()
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.residuals.tempo2_native.chain_jax import prepare_native_chain_from_simple_result

    par_path = fixture["par_path"]
    tim_path = fixture["tim_path"]
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(par_path, tim_path, verbose=False, compatibility="tempo2")
    obs_earth = np.zeros((len(toas), 3), dtype=np.float64)
    from jug.utils.constants import OBSERVATORIES

    for i, toa in enumerate(toas):
        loc = OBSERVATORIES.get(toa.observatory.lower())
        if loc is not None:
            obs_earth[i] = loc
    vel = jug["ssb_obs_vel_km_s"]
    interim = prepare_native_chain_from_simple_result(
        jug,
        params,
        toas,
        observatory_earth_km=obs_earth,
        earth_ssb_km=jug["ssb_obs_pos_km"],
        earth_ssb_vel_km_s=vel,
        use_model_epoch_batcorr=True,
    )
    oracle = load_pytempo_native_oracle(par_path, tim_path, fixture_id="wsrt167")
    delta = delta_ns(native_batcorr_days(interim), oracle.fields["bat_corr_days"], is_mjd=True)
    assert np.sqrt(np.mean(delta**2)) < 500.0
