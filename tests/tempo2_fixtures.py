"""Helpers for curated Tempo2-style real and simulated test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_TEMPO2_REAL = Path(__file__).parent / "data_tempo2"
DATA_TEMPO2_SIM = Path(__file__).parent / "data_tempo2_sim"
MANIFEST_REAL = DATA_TEMPO2_REAL / "manifest.json"
MANIFEST_SIM = DATA_TEMPO2_SIM / "manifest.json"

CANONICAL_SIM_FIXTURE_IDS = frozenset(
    {
        "sim_isolated_tcb",
        "sim_t2_tcb",
        "sim_ell1_tcb",
        "sim_ell1h_tcb",
        "sim_dd_tcb",
        "sim_ddh_tcb",
        "sim_bt_tcb",
        "sim_ddk_tcb",
        "sim_dd_tdb",
        "sim_dd_ecliptic_tcb",
        "sim_t2_track2_pn",
        "sim_t2_track2_addsat",
        "sim_t2_multisys",
        "sim_fd_tcb",
        "sim_dilatefreq_no",
    }
)

_GREEN_STATUSES = frozenset({"green", "green_required"})


def _load_manifest_rows(
    manifest_path: Path,
    data_dir: Path,
    *,
    fixture_source: str,
) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["par_path"] = data_dir / row["par"]
        item["tim_path"] = data_dir / row["tim"]
        item["fixture_source"] = fixture_source
        enriched.append(item)
    return enriched


def load_tempo2_fixture_manifest(
    *,
    include_real: bool = True,
    include_sim: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_real:
        rows.extend(_load_manifest_rows(MANIFEST_REAL, DATA_TEMPO2_REAL, fixture_source="real"))
    if include_sim:
        rows.extend(_load_manifest_rows(MANIFEST_SIM, DATA_TEMPO2_SIM, fixture_source="sim"))
    ids = [row["id"] for row in rows]
    duplicates = sorted({fixture_id for fixture_id in ids if ids.count(fixture_id) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate tempo2 fixture ids across manifests: {duplicates}")
    return rows


def _status_matches(row: dict[str, Any], *, require_green: bool) -> bool:
    if not require_green:
        return True
    parity_status = row.get("parity_status")
    if parity_status is None:
        return False
    return str(parity_status).lower() in _GREEN_STATUSES


def list_tempo2_real_fixtures(*, require_green: bool = True) -> list[dict[str, Any]]:
    selected = [
        row
        for row in load_tempo2_fixture_manifest(include_real=True, include_sim=False)
        if _status_matches(row, require_green=require_green)
    ]
    return sorted(selected, key=lambda row: row["id"])


def list_tempo2_sim_fixtures(*, require_green: bool = True) -> list[dict[str, Any]]:
    selected = [
        row
        for row in load_tempo2_fixture_manifest(include_real=False, include_sim=True)
        if _status_matches(row, require_green=require_green)
    ]
    return sorted(selected, key=lambda row: row["id"])


def list_tempo2_fixtures_by_tag(
    tag: str,
    *,
    include_real: bool = True,
    include_sim: bool = True,
    require_green: bool = True,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in load_tempo2_fixture_manifest(include_real=include_real, include_sim=include_sim)
        if tag in row.get("option_tags", []) and _status_matches(row, require_green=require_green)
    ]
    return sorted(selected, key=lambda row: row["id"])


def get_tempo2_fixture(fixture_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in load_tempo2_fixture_manifest()
        if row["id"] == fixture_id
    ]
    if not matches:
        known = ", ".join(row["id"] for row in load_tempo2_fixture_manifest())
        raise KeyError(f"Unknown Tempo2 fixture: {fixture_id}. Known fixtures: {known}")
    if len(matches) > 1:
        raise KeyError(f"Ambiguous Tempo2 fixture id {fixture_id!r} in multiple manifests")
    return matches[0]


def list_tempo2_parity_fixtures(
    *,
    cases: tuple[str, ...] | None = None,
    require_green: bool = True,
    include_sim: bool = False,
) -> list[dict[str, Any]]:
    """Return Tempo2 parity fixtures, optionally filtered by case and status.

    Case filtering:

    - ``cases=None``: include all manifest rows.
    - ``cases=(...)``: include only rows whose ``fixture_case`` matches.

    Status filtering:

    - ``require_green=True``: keep only rows with ``parity_status == "green"``
      or ``parity_status == "green_required"``. Rows with missing status are
      excluded (strict mode).
    - ``require_green=False``: no status filtering.

    By default only real fixtures are returned to preserve legacy parity tests.
    """
    normalized_cases = {case.upper() for case in cases} if cases is not None else None
    selected: list[dict[str, Any]] = []
    for row in load_tempo2_fixture_manifest(include_real=True, include_sim=include_sim):
        if normalized_cases is not None:
            fixture_case = str(row.get("fixture_case", "")).upper()
            if fixture_case not in normalized_cases:
                continue
        if not _status_matches(row, require_green=require_green):
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: row["id"])


def list_tempo2_tdb_diagnostic_fixtures() -> list[dict[str, Any]]:
    """Return Case B/C TDB fixtures for Phase A/B diagnostics."""
    return list_tempo2_parity_fixtures(cases=("B", "C"), require_green=False)
