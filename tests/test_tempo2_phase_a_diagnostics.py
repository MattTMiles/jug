"""Phase A tempo2 diagnostic tests (Case B/C TDB fixtures)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = pytest.mark.tempo2

from jug.residuals.compatibility_providers import get_delay_provider
from jug.residuals.diagnostic_conventions import DiagnosticConventions
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.phase_a_comparison import compare_fixture_phase_a, rank_phase_b_ports
from jug.testing.tempo2_diagnostics import tempo2_term_diagnostics

from tempo2_fixtures import get_tempo2_fixture, list_tempo2_parity_fixtures


TDB_DIAGNOSTIC_FIXTURES = [
    fx["id"] for fx in list_tempo2_parity_fixtures(cases=("B", "C"), require_green=False)
]


def test_pint_mode_ignores_implicit_planet_shapiro_on_ng5_fixture():
    """Pint runtime must not inherit tempo2 implicit PLANET_SHAPIRO defaults."""
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="pint",
    )
    assert result["diagnostic_conventions"]["tempo2_tdb_defaults"] == "explicit_par"
    planet = result["term_diagnostics"]["planet_shapiro_sec"]
    assert float(np.max(np.abs(planet))) == 0.0


def test_tempo2_mode_applies_implicit_planet_shapiro_when_omitted():
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    assert result["diagnostic_conventions"]["tempo2_tdb_defaults"] == "implicit_tempo2"
    planet = result["term_diagnostics"]["planet_shapiro_sec"]
    assert float(np.max(np.abs(planet))) > 0.0


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
def test_delay_provider_schema_matches(compatibility):
    provider = get_delay_provider(compatibility)
    assert provider.compatibility == compatibility
    assert provider.provider_name.endswith("_delay_provider")
    assert provider.phase_mean_mode in ("weighted", "unweighted")


def test_term_diagnostics_schema_on_tcb_fixture():
    fixture = get_tempo2_fixture("epta_j0030_isolated")
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    terms = result["term_diagnostics"]
    n = result["n_toas"]
    for key in (
        "roemer_sec",
        "sun_shapiro_sec",
        "planet_shapiro_sec",
        "roemer_shapiro_sec",
        "dm_delay_sec",
        "sw_delay_sec",
        "tropo_delay_sec",
        "prebinary_delay_sec",
        "total_delay_sec",
        "freq_bary_mhz",
        "utc_to_tdb_sec",
    ):
        assert terms[key].shape == (n,)
        assert np.all(np.isfinite(terms[key]))
    assert "metadata" in terms
    assert result["delay_provider"] == "tempo2_delay_provider"


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", TDB_DIAGNOSTIC_FIXTURES)
def test_phase_a_tdb_diagnostic_runs(fixture_id):
    fixture = get_tempo2_fixture(fixture_id)
    report = compare_fixture_phase_a(fixture)
    assert report.fixture_id == fixture_id
    assert "jug_tempo2_minus_oracle" in report.residual_stats
    assert report.residual_stats["jug_tempo2_minus_oracle"].rms_ns > 0.0
    ranking = rank_phase_b_ports(report)
    assert len(ranking) >= 1


def test_tempo2_mode_uses_native_tdb_geometry_backend():
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    result = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    assert result["term_diagnostics"]["metadata"]["geometry_backend"] == "tempo2_tdb_native"
    assert "engine_conventions" in result
    assert result["engine_conventions"]["implicit_tempo2_defaults"] is True


def test_pint_and_tempo2_tdb_geometry_diverge_on_planet_shapiro():
    fixture = get_tempo2_fixture("ng5_j1600_tdb_equatorial")
    pint = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="pint"
    )
    tempo2 = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    assert pint["term_diagnostics"]["metadata"]["geometry_backend"] == "astropy_jpl"
    planet_diff = np.max(
        np.abs(
            tempo2["term_diagnostics"]["planet_shapiro_sec"]
            - pint["term_diagnostics"]["planet_shapiro_sec"]
        )
    )
    assert planet_diff > 0.0


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", TDB_DIAGNOSTIC_FIXTURES)
def test_phase_a_tdb_residual_parity(fixture_id):
    """Case B/C TDB parity gate (Phase B native geometry)."""
    fixture = get_tempo2_fixture(fixture_id)
    report = compare_fixture_phase_a(fixture)
    stats = report.residual_stats["jug_tempo2_minus_oracle"]
    assert stats.rms_ns < 5.0


@pytest.mark.tempo2
def test_raw_metric_rejects_weighted_centering_for_tempo2_acceptance():
    conv = DiagnosticConventions(residual_metric="weighted_centered")
    with pytest.raises(ValueError, match="tempo2 acceptance requires"):
        conv.validate_for_tempo2_acceptance()


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", TDB_DIAGNOSTIC_FIXTURES)
def test_tempo2_oracle_term_properties(fixture_id):
    fixture = get_tempo2_fixture(fixture_id)
    oracle = tempo2_term_diagnostics(fixture["par_path"], fixture["tim_path"])
    assert oracle.ntoa == fixture["toa_count"]
    assert oracle.roemer_sec is not None
    assert oracle.shapiro_sun_sec is not None
    assert oracle.term_status["roemer_sec"] == "ok"
    assert oracle.term_status["shapiro_sun_sec"] == "ok"


@pytest.mark.tempo2
def test_phase_a_report_serializes():
    fixture = get_tempo2_fixture(TDB_DIAGNOSTIC_FIXTURES[0])
    report = compare_fixture_phase_a(fixture)
    payload = json.dumps(report.to_dict())
    assert "ranking" in payload
