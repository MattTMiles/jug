"""TRACK −2 ``pnNew`` / ``addPhase`` oracle tests (Phase D, wsrt167).

Step 1: validate tim ``-pn`` convention and ``track_minus2_frac_phase``.
Step 2 (2026-07-06): wiring ``phase5@bbat`` to production **ruled out** — oracle
path ~17.5 ns vs production ~16.4 ns. See ``PARITY_ROADMAP.md`` § Phase D Step 2.
"""

from __future__ import annotations

import numpy as np
import pytest

from jug.residuals.tempo2_spin import (
    _fortran_mod,
    _fortran_nlong,
    compute_tempo2_phase5,
    compute_tempo2_torb_sec,
)
from jug.testing.tempo2_track2_oracle import (
    compute_pn_new_relative,
    load_track2_oracle_context,
    track2_add_phase_turns,
    track2_frac_phase_oracle,
)
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture

pytest.importorskip("libstempo")

_WSRT167 = get_tempo2_fixture("wsrt167")


def _phase5_at_oracle(ctx):
    torb = compute_tempo2_torb_sec(
        ctx.bbat_mjd, ctx.dt_sec, float(ctx.params["PEPOCH"])
    )
    return compute_tempo2_phase5(
        ctx.bbat_mjd, torb, ctx.params, jump_phase=ctx.jump_phase
    )


def test_tim_pn_delta_matches_pn_new_relative_wsrt167():
    """tim ``-pn[i] - -pn[0]`` must equal tempo2 ``pnNew`` (after ``pn0``)."""
    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    pn_new = compute_pn_new_relative(
        _phase5_at_oracle(ctx), ctx.bbat_mjd, ctx.f0
    )
    np.testing.assert_array_equal(pn_new, ctx.pn_tim - ctx.pn_tim[0])


def test_track2_add_phase_is_minus_pn_add_wsrt167():
    """With relative ``-pn``, ``addPhase`` collapses to ``-pnAdd`` (+1 turn here)."""
    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    add_phase = track2_add_phase_turns(ctx)
    np.testing.assert_allclose(
        add_phase, -ctx.pn_add.astype(np.float64), rtol=0, atol=0
    )


def test_track_minus2_frac_phase_no_longdouble_blowup_wsrt167():
    """Fixed ``pnAct`` must not produce O(10¹⁰) turn residuals on wsrt167."""
    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    frac, _, _ = track2_frac_phase_oracle(ctx)
    assert np.all(np.isfinite(frac))
    assert float(np.max(np.abs(frac))) < 10.0


def test_track2_frac_matches_legacy_plus_one_wsrt167():
    """pnNew path with relative ``-pn`` must match legacy ``-pnAdd`` wrap on phase5."""
    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    frac_pn, _, _ = track2_frac_phase_oracle(ctx)

    phase5 = _phase5_at_oracle(ctx)
    phas1 = float(_fortran_mod(phase5[0], 1.0))
    p5a = phase5 - phas1
    nph = _fortran_nlong(p5a).astype(np.float64)
    frac_legacy = p5a - nph - ctx.pn_add.astype(np.float64)

    np.testing.assert_allclose(frac_pn, frac_legacy, rtol=0, atol=1e-12)


@pytest.mark.dev_oracle
@pytest.mark.tempo2
def test_track2_phase5_spin_matches_pytempo_nphase_wsrt167():
    """``compute_tempo2_phase5`` at pytempo ``bbat`` must match tempo2 ``nphase``."""
    pytest.importorskip("pytempo")
    from pytempo.sandbox import tempopulsar

    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    phase5 = _phase5_at_oracle(ctx)
    phas1 = float(_fortran_mod(phase5[0], 1.0))
    nph_jug = _fortran_nlong(phase5 - phas1).astype(np.float64)
    psr = tempopulsar(
        parfile=str(_WSRT167["par_path"]),
        timfile=str(_WSRT167["tim_path"]),
        dofit=False,
    )
    nph_pt = np.asarray(psr.phase_diagnostics()["nphase"], dtype=np.float64)
    np.testing.assert_array_equal(nph_jug, nph_pt)


@pytest.mark.dev_oracle
@pytest.mark.tempo2
def test_track2_pnnew_residual_floor_documented_wsrt167():
    """Oracle ``phase5@bbat`` + fixed pnNew ~17.5 ns — not better than production ~16.4 ns."""
    ctx = load_track2_oracle_context(
        _WSRT167["par_path"], _WSRT167["tim_path"], use_pytempo=True
    )
    frac, _, _ = track2_frac_phase_oracle(ctx)
    res_us = (frac / ctx.f0 - np.mean(frac / ctx.f0)) * 1e6
    ref = tempo2_reference(ctx.par_path, ctx.tim_path)
    delta_ns = (res_us - ref.residuals_us) * 1000.0
    rms = float(np.sqrt(np.mean(np.square(delta_ns))))
    assert rms < 25.0, f"phase5+pnNew rms={rms:.2f} ns (production ~15.5 ns is better)"
    assert rms > 5.0, "unexpected pass before WSRT padd / wrap fixes"
