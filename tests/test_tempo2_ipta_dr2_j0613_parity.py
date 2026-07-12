"""Pre-fit residual parity: IPTA DR2 EPTA J0613-0200 vs libstempo/tempo2."""

from __future__ import annotations

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
    _assert_residual_parity,
    _delta_stats_ns,
)

pytestmark = [pytest.mark.tempo2, pytest.mark.slow]

# IPTA DR2 EPTA single-PTA dataset (J0613-0200.par + J0613-0200_all.tim).
FIXTURE_ID = "epta_j0613_t2_ipta_all"

# Measured 2026-07-12 vs live libstempo after the parity closure work
# (spin-axis on tempo2 clock terms, single-rounding jpl_pleph JD, native
# freqSSB + exact einsteinRate, per-hop clock feedback, tempo2 ELL1
# truncation, FD in binary time). Was 1.2171675 / 6.9378 before. Gated as
# upper bounds with headroom for libstempo build jitter — a return to the
# old ~1 ns floor fails loudly.
MEASURED_RMS_NS = 0.022019  # ns, informational
MEASURED_MAX_NS = 0.07772   # ns, informational
PINNED_RMS_BOUND_NS = 0.1
PINNED_MAX_BOUND_NS = 0.3
ADDSAT_MAX_DELTA_US = 1.0


@pytest.mark.tempo2
def test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity():
    """JUG(tempo2) pre-fit residuals must match libstempo on the full EPTA dataset."""
    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
def test_epta_j0613_ipta_dr2_track_minus2_debt_reduced():
    """Guard TRACK -2 fix: RMS must stay well below the old ~47 ms debt."""
    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    assert jug["n_toas"] == ref.ntoa == 1369
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)

    assert stats["rms"] < 1.0e4  # < 10 µs (was ~47 ms)
    assert stats["rms"] < FINAL_RMS_DELTA_NS  # strict gate green

    # Picosecond-tier pin (see MEASURED_* provenance above).
    assert stats["rms"] < PINNED_RMS_BOUND_NS
    assert stats["max_abs"] < PINNED_MAX_BOUND_NS


@pytest.mark.tempo2
def test_epta_j0613_track_minus2_pulse_numbers_match_libstempo():
    """TRACK -2 must report pulse numbers consistent with libstempo -pn offsets."""
    from jug.testing.sandbox_tempo2 import tempopulsar

    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    psr = tempopulsar(
        parfile=str(fixture["par_path"]),
        timfile=str(fixture["tim_path"]),
        dofit=False,
    )
    lib_pn = np.array(psr.pulsenumbers(), dtype=np.int64)
    jug_pn = np.array(jug["pulse_number"], dtype=np.int64)
    np.testing.assert_array_equal(jug_pn, lib_pn)

    addsat_idx = [
        i for i, flags in enumerate(jug["toa_flags"]) if "addsat" in flags
    ]
    assert addsat_idx == [247, 256, 561]

    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    delta_us = np.asarray(jug["residuals_us"], dtype=np.float64) - ref.residuals_us
    for i in addsat_idx:
        assert abs(delta_us[i]) < ADDSAT_MAX_DELTA_US, (
            f"addsat TOA {i} delta {delta_us[i]:.3f} µs"
        )
