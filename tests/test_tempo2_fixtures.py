"""Tests for tempo2 fixture manifest helpers."""

from __future__ import annotations

from tempo2_fixtures import (
    list_tempo2_parity_fixtures,
    list_tempo2_tdb_diagnostic_fixtures,
)


def test_list_tempo2_parity_fixtures_all_cases_without_status_filter():
    rows = list_tempo2_parity_fixtures(require_green=False)
    ids = {row["id"] for row in rows}
    assert "epta_j0030_isolated" in ids
    assert "ng5_j1600_tdb_equatorial" in ids
    assert "ng5_j1600_tdb_ecliptic_cross_engine" in ids


def test_list_tempo2_parity_fixtures_case_filter_only():
    rows = list_tempo2_parity_fixtures(cases=("B", "C"), require_green=False)
    ids = [row["id"] for row in rows]
    assert ids == [
        "ng5_j1600_tdb_ecliptic_cross_engine",
        "ng5_j1600_tdb_equatorial",
    ]


def test_list_tempo2_parity_fixtures_require_green_strict_on_selected_cases():
    rows = list_tempo2_parity_fixtures(cases=("B", "C"), require_green=True)
    assert [row["id"] for row in rows] == [
        "ng5_j1600_tdb_ecliptic_cross_engine",
        "ng5_j1600_tdb_equatorial",
    ]
    assert all(row.get("parity_status") == "green" for row in rows)


def test_list_tempo2_parity_fixtures_require_green_excludes_missing_status():
    rows = list_tempo2_parity_fixtures(require_green=True)
    ids = {row["id"] for row in rows}
    assert "ng5_j1600_tdb_equatorial" in ids
    assert "epta_j0030_isolated" not in ids


def test_list_tempo2_tdb_diagnostic_fixtures_matches_case_bc_without_status_filter():
    diagnostic_ids = [fx["id"] for fx in list_tempo2_tdb_diagnostic_fixtures()]
    parity_ids = [
        fx["id"]
        for fx in list_tempo2_parity_fixtures(cases=("B", "C"), require_green=False)
    ]
    assert diagnostic_ids == parity_ids
