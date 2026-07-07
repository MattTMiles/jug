"""Fast inner-loop tempo2 parity gates for IPTA DR2 J0613-0200.

These tests avoid the full 1369-TOA EPTA dataset unless explicitly selected
elsewhere. See ``TEMPO2_PARITY.md`` § J0613 fast gates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import (
    FINAL_MAX_DELTA_NS,
    FINAL_P99_DELTA_NS,
    FINAL_RMS_DELTA_NS,
    _delta_stats_ns,
)

pytestmark = pytest.mark.tempo2

WSRT167_DEBT_RMS_NS = 25.0
NO_TRACK_DEBT_RMS_NS = 100.0
ADDSAT_DEBT_MAX_NS = 1000.0  # 1 µs — catches integer-turn regressions


def _delta_ns(jug, ref) -> np.ndarray:
    return (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
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


def test_epta_j0613_nrt1400_no_track_residual_debt(tmp_path):
    """Non-TRACK path must stay near libstempo on the small NRT excerpt."""
    fixture = get_tempo2_fixture("epta_j0613_t2_nrt1400")
    par, tim = _strip_track_and_pulse_flags(tmp_path, fixture)
    jug = compute_residuals_simple(par, tim, verbose=False, compatibility="tempo2")
    ref = tempo2_reference(par, tim)
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < NO_TRACK_DEBT_RMS_NS, (
        f"no-track nrt1400 rms={stats['rms']:.2f} ns"
    )


def test_epta_j0613_addsat_min_no_integer_wrap():
    """TRACK -2 ``-addsat`` TOAs must not integer-wrap (~±1 s)."""
    fixture = get_tempo2_fixture("epta_j0613_addsat_min")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    delta = _delta_ns(jug, ref)
    addsat_idx = [i for i, flags in enumerate(jug["toa_flags"]) if "addsat" in flags]
    assert addsat_idx == [3, 6, 9]
    assert np.max(np.abs(delta[addsat_idx])) < ADDSAT_DEBT_MAX_NS


def test_epta_j0613_addsat_min_bulk_context_near_libstempo():
    """Mini addsat fixture bulk RMS should stay in the sub-µs debt band."""
    fixture = get_tempo2_fixture("epta_j0613_addsat_min")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < 1000.0


def test_wsrt167_track2_bulk_spin_debt_pin():
    """wsrt167 TRACK -2 bulk floor debt pin (production Taylor spin route)."""
    fixture = get_tempo2_fixture("wsrt167")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < WSRT167_DEBT_RMS_NS, (
        f"wsrt167 rms={stats['rms']:.2f} ns exceeds {WSRT167_DEBT_RMS_NS} ns debt cap"
    )
    assert stats["rms"] > FINAL_RMS_DELTA_NS


@pytest.mark.xfail(
    strict=True,
    reason="wsrt167 strict 5 ns gate — bulk spin floor ~15.5 ns remains",
)
def test_wsrt167_track2_strict_residual_target():
    """Strict wsrt167 target once bulk spin floor closes."""
    fixture = get_tempo2_fixture("wsrt167")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < FINAL_RMS_DELTA_NS
    assert stats["p99_abs"] < FINAL_P99_DELTA_NS
    assert stats["max_abs"] < FINAL_MAX_DELTA_NS
