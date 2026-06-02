"""Phase C: mode-specific TZR / absolute-phase parity tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture, list_tempo2_parity_fixtures

NG5_TDB_FIXTURES = [
    fx["id"] for fx in list_tempo2_parity_fixtures(cases=("B", "C"), require_green=True)
]


def _delta_stats_ns(jug_residuals_us, tempo2_residuals_us) -> dict[str, float]:
    delta_ns = (np.asarray(jug_residuals_us) - np.asarray(tempo2_residuals_us)) * 1000.0
    return {
        "rms": float(np.sqrt(np.mean(np.square(delta_ns)))),
        "max_abs": float(np.max(np.abs(delta_ns))),
    }


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", NG5_TDB_FIXTURES)
def test_tempo2_tdb_residual_parity_with_tzr(fixture_id):
    """Case B/C remain green after Phase C TZR path (raw δ vs libstempo)."""
    fixture = get_tempo2_fixture(fixture_id)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < 5.0, f"{fixture_id}: rms={stats['rms']:.3f} ns"
    assert stats["max_abs"] < 25.0


@pytest.mark.tempo2
def test_tempo2_tcb_tzr_fixture_residual_parity():
    """TCB binary with TZR (epta_j1909_t2) must stay within residual gate."""
    fixture = get_tempo2_fixture("epta_j1909_t2")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < 5.0
    assert stats["max_abs"] < 25.0


def test_tempo2_tzrmjd_auto_treats_tdb_on_ng5():
    """Tempo2 AUTO + UNITS=TDB must match explicit TDB scale (no UTC conversion)."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from jug.residuals.engine_conventions import EngineConventionProfile
    from jug.io.par_reader import parse_par_file
    from jug.residuals.tzr_geometry import resolve_tzrmjd_epochs

    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    params = parse_par_file(fixture["par_path"])
    profile = EngineConventionProfile.from_params(params, "tempo2")
    loc = EarthLocation.from_geocentric(0 * u.km, 0 * u.km, 0 * u.km)
    epochs = resolve_tzrmjd_epochs(
        params=params,
        tzrmjd_scale="AUTO",
        tzr_is_ssb=False,
        tzr_site="gbt",
        tzr_clock=None,
        bipm_clock=None,
        tzr_location=loc,
        model_timescale="TDB",
        engine_profile=profile,
        verbose=False,
    )
    assert epochs.tzrmjd_scale_resolved == "TDB"
    assert float(epochs.delta_tzr_sec) == 0.0

    t2_auto = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        tzrmjd_scale="AUTO",
    )
    t2_tdb = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        tzrmjd_scale="TDB",
    )
    delta = np.asarray(t2_auto["residuals_us"]) - np.asarray(t2_tdb["residuals_us"])
    assert float(np.max(np.abs(delta))) < 1e-6


def test_pint_and_tempo2_residuals_differ_on_ecliptic_case_c():
    """Phase C: tempo2 vs pint differ on Case C (phase mean + geometry)."""
    fixture = get_tempo2_fixture("ng5_j1600_tdb_ecliptic_cross_engine")
    pint = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="pint",
    )
    tempo2 = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    diff = np.asarray(tempo2["residuals_us"]) - np.asarray(pint["residuals_us"])
    # Unweighted vs weighted phase mean dominates (~tens of ns); geometry adds more.
    assert float(np.max(np.abs(diff))) > 0.01
    assert float(np.abs(np.mean(diff))) > 0.01


def test_mismatched_engine_conventions_raises_on_compute():
    """Mixed compatibility and engine_conventions must fail fast."""
    from jug.residuals.engine_conventions import EngineConventionProfile

    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    tempo2_profile = EngineConventionProfile.from_params(
        {"UNITS": "TDB", "EPHEM": "DE405"}, "tempo2"
    )
    with pytest.raises(ValueError, match="does not match compatibility"):
        compute_residuals_simple(
            fixture["par_path"],
            fixture["tim_path"],
            verbose=False,
            compatibility="pint",
            engine_conventions=tempo2_profile,
        )


def test_tempo2_minus_pint_planet_shapiro_on_ng5():
    """Confirm main-TOA paths still diverge on planet Shapiro (Phase B guard)."""
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    pint = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="pint"
    )
    tempo2 = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    planet_diff = np.max(
        np.abs(
            tempo2["term_diagnostics"]["planet_shapiro_sec"]
            - pint["term_diagnostics"]["planet_shapiro_sec"]
        )
    )
    assert planet_diff > 0.0


@pytest.mark.tempo2
def test_subtract_tzr_changes_residuals_when_tzrmjd_present():
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    with_tzr = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        subtract_tzr=True,
    )
    without_tzr = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        subtract_tzr=False,
    )
    delta = np.asarray(with_tzr["residuals_us"]) - np.asarray(without_tzr["residuals_us"])
    assert float(np.std(delta)) > 0.0
