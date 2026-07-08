"""Fast inner-loop tempo2 parity gates for IPTA DR2 J0613-0200.

The default ``-m 'not slow'`` path uses mini fixtures (20 TOAs or fewer) plus
the existing 11-TOA addsat gate.  Full-fixture debt pins stay in this module but
are marked ``slow``.  See ``PARITY_ROADMAP.md`` § Fast gates and CI tiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jug.residuals.simple_calculator import compute_residuals_simple

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import (
    FINAL_MAX_DELTA_NS,
    FINAL_P99_DELTA_NS,
    FINAL_RMS_DELTA_NS,
    _delta_stats_ns,
)

pytestmark = pytest.mark.tempo2

WSRT167_DEBT_RMS_NS = 2.5
NO_TRACK_DEBT_RMS_NS = 100.0
ADDSAT_DEBT_MAX_NS = 1000.0  # 1 µs — catches integer-turn regressions


def _stored_residuals_us(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["residuals_us"], dtype=np.float64)


def _fixture_stored_residuals_us(fixture: dict) -> np.ndarray:
    return _stored_residuals_us(
        Path(fixture["tim_path"]).with_suffix(".libstempo_residuals_us.json")
    )


def _delta_ns(jug, ref_residuals_us) -> np.ndarray:
    return (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref_residuals_us, dtype=np.float64)
    ) * 1000.0


def _strip_track_and_pulse_flags(tmp_path: Path, fixture: dict) -> tuple[Path, Path]:
    """Return temp par/tim with TRACK, -pn, -pnadd, and -addsat removed."""
    par_in = Path(fixture["par_path"])
    tim_in = Path(fixture["tim_path"])
    par_out = tmp_path / "no_track.par"
    tim_out = tmp_path / "no_pn.tim"

    par_lines = [
        line
        for line in par_in.read_text().splitlines()
        if not line.strip().startswith("TRACK")
    ]
    par_out.write_text("\n".join(par_lines) + "\n")

    tim_lines: list[str] = []
    for line in tim_in.read_text().splitlines():
        parts = line.split()
        if not parts:
            tim_lines.append(line)
            continue
        cleaned: list[str] = []
        skip_next = False
        for token in parts:
            if skip_next:
                skip_next = False
                continue
            if token in {"-pn", "-pnadd", "-addsat"}:
                skip_next = True
                continue
            cleaned.append(token)
        tim_lines.append(" ".join(cleaned) if parts else line)
    tim_out.write_text("\n".join(tim_lines) + "\n")
    return par_out, tim_out


def test_epta_j0613_nrt1400_mini_no_track_residual_debt(tmp_path):
    """Non-TRACK path must stay near libstempo on the mini NRT excerpt."""
    fixture = get_tempo2_fixture("epta_j0613_nrt1400_mini")
    par, tim = _strip_track_and_pulse_flags(tmp_path, fixture)
    jug = compute_residuals_simple(par, tim, verbose=False, compatibility="tempo2")
    ref_us = _stored_residuals_us(
        Path(fixture["tim_path"]).with_name(
            "epta_j0613_nrt1400_mini_no_track.libstempo_residuals_us.json"
        )
    )
    stats = _delta_stats_ns(jug["residuals_us"], ref_us)
    assert stats["rms"] < NO_TRACK_DEBT_RMS_NS, (
        f"no-track nrt1400 mini rms={stats['rms']:.2f} ns"
    )


@pytest.mark.slow
def test_epta_j0613_addsat_min_no_integer_wrap_and_bulk_context():
    """TRACK -2 ``-addsat`` mini fixture: no integer wrap and bulk stays bounded."""
    fixture = get_tempo2_fixture("epta_j0613_addsat_min")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref_us = _fixture_stored_residuals_us(fixture)
    delta = _delta_ns(jug, ref_us)
    stats = _delta_stats_ns(jug["residuals_us"], ref_us)
    addsat_idx = [i for i, flags in enumerate(jug["toa_flags"]) if "addsat" in flags]
    assert addsat_idx == [3, 6, 9]
    assert np.max(np.abs(delta[addsat_idx])) < ADDSAT_DEBT_MAX_NS
    assert stats["rms"] < 1000.0


def test_wsrt167_mini_track2_strict_residual_target():
    """Mini WSRT TRACK -2 spin gate for the default fast path."""
    fixture = get_tempo2_fixture("wsrt167_mini")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref_us = _fixture_stored_residuals_us(fixture)
    stats = _delta_stats_ns(jug["residuals_us"], ref_us)
    assert stats["rms"] < FINAL_RMS_DELTA_NS
    assert stats["p99_abs"] < FINAL_P99_DELTA_NS
    assert stats["max_abs"] < FINAL_MAX_DELTA_NS


@pytest.mark.slow
def test_wsrt167_track2_full_fixture_debt_pin_and_strict_target(
    wsrt167_jug, wsrt167_libstempo
):
    """Full 167-TOA wsrt167 gate; session fixtures amortize JUG/libstempo setup."""
    jug = wsrt167_jug
    ref = wsrt167_libstempo
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < WSRT167_DEBT_RMS_NS, (
        f"wsrt167 rms={stats['rms']:.2f} ns exceeds {WSRT167_DEBT_RMS_NS} ns debt cap"
    )
    assert stats["rms"] < FINAL_RMS_DELTA_NS
    assert stats["p99_abs"] < FINAL_P99_DELTA_NS
    assert stats["max_abs"] < FINAL_MAX_DELTA_NS
