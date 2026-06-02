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


def list_tempo2_tdb_diagnostic_fixtures() -> list[dict[str, Any]]:
    """Return Case B/C TDB fixtures for Phase A/B diagnostics."""
    return [
        row
        for row in load_tempo2_fixture_manifest()
        if row.get("fixture_case") in ("B", "C")
        or row["id"].startswith("ng5_j1600_tdb_")
    ]
