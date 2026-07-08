"""DEV ORACLE — granular formBats component closure using pytempo delay diagnostics.

Gate semantics (wsrt167, unified JAX path):

- ``bat_corr_days`` and per-component gates test **delay physics** (~1 ns target).
- ``bbat_mjd`` is tested separately in ``test_tempo2_native_bbat_parity.py``; raw
  ``bbat_mjd`` can fail at ~304 ns while ``bat_corr_days`` passes because tempo2
  assembles ``bat`` with split ``long double`` summation. See ``PARITY_ROADMAP.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from tempo2_native_test_helpers import delta_ns, native_batcorr_days

WSRT167_TRACE_INDICES = [0, 42, 85, 166]


def test_pytempo_formbats_self_closure_wsrt167(wsrt167_pytempo_oracle):
    closure = np.abs(wsrt167_pytempo_oracle.fields["bat_corr_closure_ns"])
    assert float(np.max(closure)) < 1.0


def test_jug_formbats_replay_with_pytempo_components_wsrt167(wsrt167_formbats_report):
    assert wsrt167_formbats_report.jug_replay_all_pytempo_rms_ns < 1.0


def test_wsrt167_component_ranking_documents_tt_blocker(wsrt167_formbats_report):
    report = wsrt167_formbats_report
    assert report.swap_one_rms_ns["tt"] < 1.0
    assert report.swap_one_rms_ns["roemer"] < 1.0
    assert report.swap_one_rms_ns["tdis2"] < 1.0
    assert report.component_rms_ns["tt"] < 1.0


def test_wsrt167_per_component_gates(wsrt167_formbats_report):
    report = wsrt167_formbats_report
    assert report.component_rms_ns["roemer"] < 1.0
    assert report.component_rms_ns["tdis2"] < 1.0
    assert report.component_rms_ns["tdis1"] < 1.0
    assert report.component_rms_ns["tt_tb"] < 1.0
    assert report.component_rms_ns["tropo"] < 1.0
    assert report.component_rms_ns["shap"] < 1.0


def test_native_strict_formbats_batcorr_wsrt167(wsrt167_native_terms, wsrt167_pytempo_oracle):
    """Delay-component gate on wsrt167 (physics), not MJD assembly.

    Compares ``bat_corr_days`` / integrated formBats correction. Target < 1 ns.
    May fail at ~1.1 ns until clock-outlier TOAs are closed. For ``bbat_mjd``
    assembly (~304 ns), see ``test_tempo2_native_bbat_parity.py``.
    """
    delta = delta_ns(
        native_batcorr_days(wsrt167_native_terms),
        wsrt167_pytempo_oracle.fields["bat_corr_days"],
        is_mjd=True,
    )
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0


def test_native_bclt_roemer_interim_wsrt167(wsrt167_native_terms, wsrt167_pytempo_oracle):
    roemer = np.asarray(jax.device_get(wsrt167_native_terms.roemer_sec), dtype=np.float64)
    delta = delta_ns(roemer, wsrt167_pytempo_oracle.fields["roemer_sec"])
    assert np.sqrt(np.mean(delta**2)) < 1.0


def test_native_dt_ssb_interim_wsrt167(wsrt167_native_terms, wsrt167_pytempo_oracle):
    dt_ssb = np.asarray(jax.device_get(wsrt167_native_terms.dt_ssb_sec), dtype=np.float64)
    delta = delta_ns(dt_ssb, wsrt167_pytempo_oracle.fields["dt_ssb_sec"])
    assert np.sqrt(np.mean(delta**2)) < 1.0


def test_single_toa_formbats_trace_wsrt167(wsrt167_pytempo_oracle):
    pt = wsrt167_pytempo_oracle.fields
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
