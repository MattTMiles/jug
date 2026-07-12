"""Strict raw-input parity gate: IPTA DR2 EPTA J0613-0200, untouched par/tim.

Unlike ``epta_j0613_t2_ipta_all`` (which adds ``TRACK -2`` to the par and bakes
``-pn`` pulse numbers into every TOA), this fixture is the *raw* observatory
data: no TRACK, no pulse-number flags. It therefore exercises the
TRACK-absent Taylor-sequential host path end to end, on a par with implicit
TCB units and a T2/ELL1 binary, mixing a ``-padd`` backend (JBO.DFB.1400)
with a padd-free one (NRT.BON.1400).

Regression provenance (2026-07, tempo2-dev branch): the doctored fixture
masked nothing by itself — the real gap was that only doctored/mini inputs
had *fast* gates. Two raw-data-visible bugs shipped:

1. ``tt_binary`` built from ``tdb_mjd`` instead of ``model_mjd``: for
   implicit-TCB pars the binary was evaluated on the TDB axis against
   TCB-axis TASC/PB, shifting orbital phase by 2*pi*(TCB-TDB)/PB
   (~1 ms * cos(orbital phase) residual error, growing ~1.3 ms/day / LB).
2. ``compute_get_correction_tt_sec`` rewritten without the astropy UTC->TT
   hop, silently dropping the ~64-66 s leap-second + TT(TAI) offset from
   every BAT and corrupting the native-chain Roemer delay by ~5 ms.

This test pins the raw path to strict (<5 ns RMS) live-libstempo parity so
neither class of error can land silently again.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import (
    _assert_residual_parity,
    _delta_stats_ns,
)

pytestmark = pytest.mark.tempo2

FIXTURE_ID = "epta_j0613_raw_mix"


@pytest.fixture(scope="module")
def raw_mix_run():
    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    return fixture, jug, ref


def test_epta_j0613_raw_input_strict_parity(raw_mix_run):
    """Raw (no TRACK, no -pn) J0613 subset must hit strict tempo2 parity."""
    fixture, jug, ref = raw_mix_run
    assert jug["n_toas"] == ref.ntoa == 29
    _assert_residual_parity(jug, ref, fixture["id"])


def test_epta_j0613_raw_input_padd_applied(raw_mix_run):
    """-padd flags must land in jump_phase for the JBO backend only."""
    _, jug, _ = raw_mix_run
    jump_phase = np.asarray(jug["jump_phase"], dtype=np.float64)
    sys_flags = [flags.get("sys", "") for flags in jug["toa_flags"]]
    jbo = np.array([s == "JBO.DFB.1400" for s in sys_flags])
    assert jbo.sum() == 15 and (~jbo).sum() == 14
    np.testing.assert_allclose(jump_phase[jbo], 0.401094, atol=1e-9)
    np.testing.assert_allclose(jump_phase[~jbo], 0.0, atol=1e-12)


def test_epta_j0613_raw_input_no_gross_wrap(raw_mix_run):
    """Guard the ms-scale failure mode explicitly (fractional-period garbage)."""
    _, jug, ref = raw_mix_run
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    # One thousandth of a pulse period (P ~ 3.06 ms) in ns:
    assert stats["max_abs"] < 3.0e3, (
        f"raw-input residuals off by {stats['max_abs']:.0f} ns — "
        "phase/delay chain corrupted (see module docstring)"
    )
