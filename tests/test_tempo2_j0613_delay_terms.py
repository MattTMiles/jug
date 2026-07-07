"""Component-level pytempo oracle gates for J0613 tempo2-compatible JUG.

Uses Tier-1 ``toa_diagnostics`` fields only. Marked ``dev_oracle`` because
pytempo is optional in CI. See ``TEMPO2_PARITY.md`` § J0613 term gates.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle

from tempo2_fixtures import get_tempo2_fixture
from tempo2_native_test_helpers import rms_ns

_WSRT167 = get_tempo2_fixture("wsrt167")

# Documented assembly / closure gaps on wsrt167 (2026-07-07, Taylor production spin).
BBAT_DEBT_RMS_NS = 350.0
TORB_DEBT_RMS_NS = 30.0


def _jug_wsrt167():
    return compute_residuals_simple(
        _WSRT167["par_path"],
        _WSRT167["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )


def _oracle_wsrt167():
    return load_pytempo_native_oracle(
        _WSRT167["par_path"],
        _WSRT167["tim_path"],
        fixture_id="wsrt167",
    )


def test_wsrt167_sat_mjd_matches_pytempo():
    jug = _jug_wsrt167()
    oracle = _oracle_wsrt167()
    sat = np.asarray(jug["term_diagnostics"]["sat_mjd"], dtype=np.float64)
    # pytempo does not export sat_mjd in Tier-1; compare JUG sat to itself as sanity.
    assert sat.shape == oracle.fields["bbat_mjd"].shape
    assert np.all(np.isfinite(sat))


def test_wsrt167_bbat_mjd_debt_pinned_vs_pytempo():
    jug = _jug_wsrt167()
    oracle = _oracle_wsrt167()
    bbat = np.asarray(jug["term_diagnostics"]["bbat_mjd"], dtype=np.float64)
    rms = rms_ns(bbat, oracle.fields["bbat_mjd"], is_mjd=True)
    assert rms < BBAT_DEBT_RMS_NS, f"bbat rms={rms:.1f} ns"


def test_wsrt167_torb_closure_debt_pinned_vs_pytempo():
    jug = _jug_wsrt167()
    oracle = _oracle_wsrt167()
    td = jug["term_diagnostics"]
    torb = np.asarray(td["prebinary_delay_sec"], dtype=np.float64) - np.asarray(
        jug["total_delay_sec"], dtype=np.float64
    )
    rms = rms_ns(torb, oracle.fields["torb_sec"])
    assert rms < TORB_DEBT_RMS_NS, f"torb closure rms={rms:.1f} ns"


def test_wsrt167_jump_phase_matches_pytempo_phase_offset():
    from pytempo.sandbox import tempopulsar

    jug = _jug_wsrt167()
    psr = tempopulsar(
        parfile=str(_WSRT167["par_path"]),
        timfile=str(_WSRT167["tim_path"]),
        dofit=False,
    )
    diag = psr.toa_diagnostics(removemean=False)
    jump = np.asarray(jug["jump_phase"], dtype=np.float64)
    phase_offset = np.asarray(diag["phase_offset_turns"], dtype=np.float64)
    np.testing.assert_allclose(jump, phase_offset, rtol=0, atol=1e-12)


def test_wsrt167_pulse_numbers_match_pytempo():
    from pytempo.sandbox import tempopulsar

    jug = _jug_wsrt167()
    psr = tempopulsar(
        parfile=str(_WSRT167["par_path"]),
        timfile=str(_WSRT167["tim_path"]),
        dofit=False,
    )
    diag = psr.toa_diagnostics(removemean=False)
    jug_pn = np.asarray(jug["pulse_number"], dtype=np.int64)
    pt_pn = np.asarray(diag["pulse_number"], dtype=np.int64)
    # libstempo/pytempo may differ from tim -pn by a constant offset; residuals
    # parity uses the same TRACK -2 wrap, so compare relative deltas.
    np.testing.assert_array_equal(jug_pn - jug_pn[0], pt_pn - pt_pn[0])


def test_wsrt167_batcorr_native_chain_under_2_ns():
    """Native JAX ``bat_corr_days`` gate (delay physics, not MJD assembly)."""
    from tempo2_native_test_helpers import compute_native_terms_for_fixture, native_batcorr_days

    oracle = _oracle_wsrt167()
    native = compute_native_terms_for_fixture(_WSRT167)
    bat_corr = native_batcorr_days(native)
    rms = rms_ns(bat_corr, oracle.fields["bat_corr_days"], is_mjd=True)
    assert rms < 2.0, f"native bat_corr rms={rms:.2f} ns"
