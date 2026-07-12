"""Structural and parity tests for libstempo-generated tempo2 fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jug.testing.fingerprint import extract_fingerprint, validate_tempo2_compatible
from tempo2_fixtures import (
    CANONICAL_SIM_FIXTURE_IDS,
    get_tempo2_fixture,
    list_tempo2_sim_fixtures,
)

# Documented relaxed gates for known debt classes (see PARITY_ROADMAP.md).
# sim_dd_tdb (TDB spin-epoch) closed by 8a1a34d; sim_dd_ecliptic_tcb (ecliptic
# frame rotation) closed by the ecliptic obsn[] equ2ecl fix — both now run
# under the strict default gate.
RELAXED_RESIDUAL_GATES = {
    "sim_t2_track2_addsat": {
        "rms_delta_ns": 100.0,
        "max_delta_ns": 200.0,
        "p99_delta_ns": 200.0,
        "reason": "TRACK -2 -addsat scatter on simulated TOAs",
    },
}

REQUIRED_OPTION_TAGS = {
    "BINARY=isolated",
    "BINARY=T2",
    "BINARY=ELL1",
    "BINARY=ELL1H",
    "BINARY=DD",
    "BINARY=DDH",
    "BINARY=BT",
    "BINARY=DDK",
    "UNITS=TCB",
    "UNITS=TDB",
    "COORDS=ecliptic",
    "TRACK=-2",
    "TIM=-pn",
    "TIM=-addsat",
    "TIM=multi-sys",
    "FD",
    "DILATEFREQ=N",
}

REQUIRED_MANIFEST_KEYS = {
    "binary",
    "designmatrix_params",
    "generated",
    "generator",
    "id",
    "option_tags",
    "par",
    "parity_status",
    "provenance",
    "tim",
    "toa_count",
}


def test_simulated_manifest_has_required_coverage():
    fixtures = list_tempo2_sim_fixtures(require_green=False)
    tags = {tag for fixture in fixtures for tag in fixture.get("option_tags", [])}
    assert REQUIRED_OPTION_TAGS <= tags


def test_simulated_fixtures_are_tiny():
    for fixture in list_tempo2_sim_fixtures(require_green=False):
        assert fixture["generated"] is True
        assert fixture["provenance"] == "simulated_libstempo_fakepulsar"
        assert fixture["fixture_source"] == "sim"
        assert 5 <= int(fixture["toa_count"]) <= 12, fixture["id"]
        assert fixture["par_path"].exists(), fixture["id"]
        assert fixture["tim_path"].exists(), fixture["id"]


def test_simulated_manifest_schema_and_canonical_ids():
    fixtures = list_tempo2_sim_fixtures(require_green=False)
    fixture_ids = {fixture["id"] for fixture in fixtures}
    assert CANONICAL_SIM_FIXTURE_IDS <= fixture_ids
    assert len(fixtures) >= len(CANONICAL_SIM_FIXTURE_IDS)
    for fixture in fixtures:
        assert REQUIRED_MANIFEST_KEYS <= set(fixture.keys()), fixture["id"]
        assert fixture["option_tags"], fixture["id"]
        assert fixture["parity_status"] == "green_required", fixture["id"]


def test_simulated_manifest_is_deterministically_formatted():
    manifest_path = list_tempo2_sim_fixtures(require_green=False)[0]["par_path"].parents[1] / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8")
    assert payload.endswith("\n")
    rows = json.loads(payload)
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    for row in rows:
        assert list(row.keys()) == sorted(row.keys())


def test_simulated_fixture_fingerprints_are_accepted():
    for fixture in list_tempo2_sim_fixtures(require_green=False):
        ok, issues = validate_tempo2_compatible(extract_fingerprint(fixture["par_path"]))
        assert ok, f"{fixture['id']}: {issues}"


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture", list_tempo2_sim_fixtures(), ids=lambda fx: fx["id"])
def test_simulated_tempo2_residual_parity(fixture):
    pytest.importorskip("libstempo")
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.testing.tempo2_reference import tempo2_reference
    from tempo2_fixture_assertions import assert_residual_parity

    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    fixture_id = fixture["id"]
    gate = RELAXED_RESIDUAL_GATES.get(fixture_id)
    if gate is not None:
        assert_residual_parity(
            jug,
            ref,
            fixture_id,
            rms_delta_ns=gate["rms_delta_ns"],
            max_delta_ns=gate["max_delta_ns"],
            p99_delta_ns=gate["p99_delta_ns"],
        )
        return
    assert_residual_parity(jug, ref, fixture_id)


@pytest.mark.tempo2
@pytest.mark.parametrize(
    "fixture",
    [fx for fx in list_tempo2_sim_fixtures() if fx.get("designmatrix_params")],
    ids=lambda fx: fx["id"],
)
def test_simulated_tempo2_designmatrix_parity(fixture):
    pytest.importorskip("libstempo")
    from jug.fitting.optimized_fitter import compute_designmatrix
    from jug.testing.tempo2_reference import tempo2_reference
    from jug.utils.units import validate_column_units
    from tempo2_fixture_assertions import assert_column_matches, tempo2_to_pint_vela_scale

    fit_params = list(fixture["designmatrix_params"])
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params,
        compatibility="tempo2",
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert jug.matrix.shape[0] == ref.ntoa
    assert jug.labels == fit_params
    assert jug.unit_convention == "pint-vela"
    assert jug.column_units == validate_column_units(jug.labels)

    ref_label_to_idx = {label: idx for idx, label in enumerate(ref.designmatrix_labels)}
    for jug_idx, param in enumerate(jug.labels):
        assert param in ref_label_to_idx
        ref_col = ref.designmatrix[:, ref_label_to_idx[param]] * tempo2_to_pint_vela_scale(param)
        assert_column_matches(param, jug.matrix[:, jug_idx], ref_col)


@pytest.mark.tempo2
def test_simulated_tempo2_autodiff_designmatrix_astrometry_nonzero():
    from jug.fitting.optimized_fitter import compute_designmatrix

    fixture = get_tempo2_fixture("sim_isolated_tcb")
    fit_params = ["F0", "F1", "RAJ", "DECJ", "DM"]
    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params,
        compatibility="tempo2",
        design_matrix_method="autodiff",
    )

    assert jug.labels == fit_params
    assert np.all(np.isfinite(jug.matrix))
    col_norms = {
        param: float(np.linalg.norm(jug.matrix[:, idx]))
        for idx, param in enumerate(jug.labels)
    }
    assert col_norms["RAJ"] > 0.0
    assert col_norms["DECJ"] > 0.0
