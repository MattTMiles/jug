"""Helpers for curated Tempo2-style PPTA/EPTA test fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_TEMPO2 = Path(__file__).parent / "data_tempo2"
MANIFEST = DATA_TEMPO2 / "manifest.json"


def load_tempo2_fixture_manifest() -> list[dict[str, Any]]:
    with MANIFEST.open() as f:
        rows = json.load(f)
    for row in rows:
        row["par_path"] = DATA_TEMPO2 / row["par"]
        row["tim_path"] = DATA_TEMPO2 / row["tim"]
    return rows


def get_tempo2_fixture(fixture_id: str) -> dict[str, Any]:
    for row in load_tempo2_fixture_manifest():
        if row["id"] == fixture_id:
            return row
    raise KeyError(f"Unknown Tempo2 fixture: {fixture_id}")


def list_tempo2_parity_fixtures(
    *,
    cases: tuple[str, ...] | None = None,
    require_green: bool = True,
) -> list[dict[str, Any]]:
    """Return Tempo2 parity fixtures, optionally filtered by case and status.

    Case filtering:

    - ``cases=None``: include all manifest rows.
    - ``cases=(...)``: include only rows whose ``fixture_case`` matches.

    Status filtering:

    - ``require_green=True``: keep only rows with ``parity_status == "green"``.
      Rows with missing status are excluded (strict mode).
    - ``require_green=False``: no status filtering.
    """
    normalized_cases = {c.upper() for c in cases} if cases is not None else None
    selected: list[dict[str, Any]] = []
    for row in load_tempo2_fixture_manifest():
        if normalized_cases is not None:
            fixture_case = str(row.get("fixture_case", "")).upper()
            if fixture_case not in normalized_cases:
                continue
        if require_green:
            parity_status = row.get("parity_status")
            if parity_status is None or str(parity_status).lower() != "green":
                continue
        selected.append(row)
    return sorted(selected, key=lambda row: row["id"])


def list_tempo2_tdb_diagnostic_fixtures() -> list[dict[str, Any]]:
    """Return Case B/C TDB fixtures for Phase A/B diagnostics."""
    return list_tempo2_parity_fixtures(cases=("B", "C"), require_green=False)
