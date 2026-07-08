"""Tier-1 pytempo diagnostic oracle — dev_oracle only, never production imports.

Wraps ``pytempo.sandbox.tempopulsar(...).toa_diagnostics()`` with cheat-sheet
guards from ``PARITY_ROADMAP.md`` §0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIER1_FIELDS = (
    "acceptance_residual_sec",
    "pulse_number",
    "bbat_mjd",
    "bat_corr_days",
    "torb_sec",
    "roemer_sec",
    "sun_shapiro_sec",
    "tdis1_sec",
    "tdis2_sec",
)

DELAY_CHAIN_FIELDS = (
    "correction_tt_sec",
    "correction_tt_tb_sec",
    "tropospheric_sec",
    "shapiro_delay_sec",
    "shapiro_planets_stored_sec",
    "shklovskii_sec",
    "dt_ssb_sec",
    "delay_corr",
    "clock_corr",
    "bat_corr_from_components_days",
    "bat_corr_closure_ns",
    "bat_from_components_mjd",
    "bbat_from_components_mjd",
    "bat_mjd_closure_ns",
    "bbat_mjd_closure_ns",
    "calc_shapiro",
    "planet_shapiro",
    "correct_troposphere",
)

ALL_ORACLE_FIELDS = TIER1_FIELDS + DELAY_CHAIN_FIELDS


@dataclass(frozen=True)
class PytempoNativeOracle:
    """Normalized Tier-1 pytempo per-TOA fields for native-chain gates."""

    fixture_id: str
    fields: dict[str, np.ndarray]
    scalars: dict[str, int | float | bool]
    residual_sec_reliable: bool


def load_pytempo_native_oracle(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    include_delay_chain: bool = True,
) -> PytempoNativeOracle:
    """Load pytempo ``toa_diagnostics(removemean=False)`` with Tier-1 extractors."""
    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(parfile=str(par), timfile=str(tim), dofit=False)
    diag = psr.toa_diagnostics(removemean=False)
    reliable_arr = diag.get("residual_sec_reliable")
    if reliable_arr is None:
        reliable = True
    else:
        reliable = bool(np.all(np.asarray(reliable_arr)))

    field_names = list(ALL_ORACLE_FIELDS if include_delay_chain else TIER1_FIELDS)
    fields = {
        name: np.asarray(diag[name], dtype=np.float64)
        for name in field_names
        if name in diag
    }
    scalars = {
        key: int(diag[key])
        for key in ("calc_shapiro", "planet_shapiro", "correct_troposphere")
        if key in diag
    }
    return PytempoNativeOracle(
        fixture_id=fixture_id,
        fields=fields,
        scalars=scalars,
        residual_sec_reliable=reliable,
    )


def tier1_field(oracle: PytempoNativeOracle, name: str) -> np.ndarray:
    """Return a Tier-1 field or raise with cheat-sheet context."""
    if name not in oracle.fields:
        raise KeyError(f"pytempo Tier-1 field {name!r} missing from oracle")
    return oracle.fields[name]
