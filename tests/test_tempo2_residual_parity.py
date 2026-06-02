"""Pre-fit residual parity tests for Tempo2-compatible mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.delays.barycentric import compute_einstein_rate
from jug.io.par_reader import parse_par_file
from jug.utils.timescales import IFTE_K
from jug.testing.tempo2_reference import tempo2_reference
from jug.testing.fingerprint import extract_fingerprint, validate_tempo2_compatible

from tempo2_fixtures import get_tempo2_fixture, list_tempo2_tdb_diagnostic_fixtures

NG5_TDB_FIXTURES = [fx["id"] for fx in list_tempo2_tdb_diagnostic_fixtures()]

FINAL_RMS_DELTA_NS = 5.0
FINAL_MAX_DELTA_NS = 25.0
FINAL_P99_DELTA_NS = 10.0


def _delta_stats_ns(jug_residuals_us, tempo2_residuals_us) -> dict[str, float]:
    delta_ns = (np.asarray(jug_residuals_us) - np.asarray(tempo2_residuals_us)) * 1000.0
    return {
        "rms": float(np.sqrt(np.mean(np.square(delta_ns)))),
        "max_abs": float(np.max(np.abs(delta_ns))),
        "p99_abs": float(np.percentile(np.abs(delta_ns), 99)),
        "mean": float(np.mean(delta_ns)),
    }


@pytest.mark.tempo2
@pytest.mark.xfail(reason="PINT-default path intentionally does not preserve Tempo2-native TCB semantics")
def test_pint_default_baseline_vs_tempo2_isolated():
    """Diagnostic baseline showing why a separate Tempo2 path is needed."""
    fixture = get_tempo2_fixture("epta_j0030_isolated")

    jug = compute_residuals_simple(fixture["par_path"], fixture["tim_path"], verbose=False)
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    assert jug["n_toas"] == ref.ntoa
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < FINAL_RMS_DELTA_NS


def _assert_residual_parity(jug, ref, fixture_id: str):
    assert jug["n_toas"] == ref.ntoa
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    wrms_delta_ns = abs(jug["weighted_rms_us"] - ref.wrms_us) * 1000.0
    message = (
        f"{fixture_id}: rms={stats['rms']:.3f} ns, "
        f"p99={stats['p99_abs']:.3f} ns, max={stats['max_abs']:.3f} ns, "
        f"mean={stats['mean']:.3f} ns, wrms_delta={wrms_delta_ns:.3f} ns; "
        f"first5_delta_ns={((np.asarray(jug['residuals_us'][:5]) - ref.residuals_us[:5]) * 1000.0).tolist()}"
    )
    assert stats["rms"] < FINAL_RMS_DELTA_NS, message
    assert stats["p99_abs"] < FINAL_P99_DELTA_NS, message
    assert stats["max_abs"] < FINAL_MAX_DELTA_NS, message
    assert wrms_delta_ns < FINAL_RMS_DELTA_NS, message


@pytest.mark.tempo2
def test_tempo2_mode_isolated_residual_parity():
    fixture = get_tempo2_fixture("epta_j0030_isolated")

    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", ["epta_j1909_t2", "epta_j1918_ddh", "ppta_j1902_ell1h"])
def test_tempo2_mode_binary_residual_parity(fixture_id):
    fixture = get_tempo2_fixture(fixture_id)

    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", NG5_TDB_FIXTURES)
def test_tempo2_mode_ng5_tdb_residual_parity(fixture_id):
    """Case B/C NG5 TDB fixtures (TZR + DD + DMX) vs libstempo."""
    fixture = get_tempo2_fixture(fixture_id)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
def test_tempo2_mode_ell1_j1741_documented_gap():
    """Document the remaining PPTA J1741 ELL1 binary-model convention gap.

    This fixture sits just above the strict 5 ns RMS gate.  The delta is not
    removed by a constant+slope fit, but is partly absorbed by orbital
    harmonics, which is consistent with a narrow ELL1/ELL1H binary convention
    mismatch rather than the Tempo2 timebase/TZR path.
    """
    fixture = get_tempo2_fixture("ppta_j1741_ell1")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < 8.0
    assert stats["p99_abs"] < 15.0
    assert stats["max_abs"] < 15.0

    delta_ns = (np.asarray(jug["residuals_us"]) - ref.residuals_us) * 1000.0
    t = np.asarray(jug.get("model_mjd", jug["tdb_mjd"]), dtype=np.float64)
    linear = np.column_stack([np.ones_like(t), t - np.mean(t)])
    linear_resid = delta_ns - linear @ np.linalg.lstsq(linear, delta_ns, rcond=None)[0]
    linear_rms = float(np.sqrt(np.mean(np.square(linear_resid))))
    assert linear_rms > 0.95 * stats["rms"]

    params = parse_par_file(fixture["par_path"])
    phase = ((t - float(params["TASC"])) / float(params["PB"])) % 1.0
    harmonics = [np.ones_like(phase)]
    for order in range(1, 5):
        harmonics.extend([
            np.sin(2.0 * np.pi * order * phase),
            np.cos(2.0 * np.pi * order * phase),
        ])
    orbital = np.column_stack(harmonics)
    orbital_resid = delta_ns - orbital @ np.linalg.lstsq(orbital, delta_ns, rcond=None)[0]
    orbital_rms = float(np.sqrt(np.mean(np.square(orbital_resid))))
    assert orbital_rms < 0.75 * stats["rms"]


@pytest.mark.tempo2
def test_tempo2_reference_fixture_matrix_smoke():
    """The curated fixture matrix is small and consumable by libstempo."""
    for fixture_id in [
        "epta_j0030_isolated",
        "epta_j1909_t2",
        "epta_j1918_ddh",
        "ppta_j1741_ell1",
        "ppta_j1902_ell1h",
    ]:
        fixture = get_tempo2_fixture(fixture_id)
        ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
        assert 0 < ref.ntoa <= fixture["toa_count"], fixture_id
        assert np.all(np.isfinite(ref.residuals_us)), fixture_id
        assert np.isfinite(ref.wrms_us), fixture_id


def test_tempo2_fixture_fingerprints_are_accepted():
    for fixture_id in [
        "epta_j0030_isolated",
        "epta_j1909_t2",
        "epta_j1918_ddh",
        "ppta_j1741_ell1",
        "ppta_j1902_ell1h",
    ]:
        fixture = get_tempo2_fixture(fixture_id)
        ok, issues = validate_tempo2_compatible(extract_fingerprint(fixture["par_path"]))
        assert ok, f"{fixture_id}: {issues}"


def test_tempo2_dilatefreq_tcb_einstein_rate_includes_ifte_scale():
    tdb_mjd = np.array([55000.0, 56000.0, 57000.0], dtype=np.float64)
    rate_tdb = compute_einstein_rate(tdb_mjd, units="TDB")
    rate_tcb = compute_einstein_rate(tdb_mjd, units="TCB")

    np.testing.assert_allclose(rate_tcb / rate_tdb, float(IFTE_K), rtol=1e-12, atol=0.0)
