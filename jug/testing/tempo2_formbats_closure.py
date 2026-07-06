"""Granular formBats component closure helpers (dev_oracle only).

Uses pytempo ``toa_diagnostics()`` delay-chain fields to rank JUG vs tempo2
term gaps and to validate JUG formBats algebra in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jug.residuals.tempo2_native.probes import (
    formbats_replay_batcorr_sec,
    formbats_replay_batcorr_tempo2_sec,
    rms_ns,
)
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from jug.utils.constants import SECS_PER_DAY


@dataclass
class FormbatsComponentReport:
    """Per-component RMS gaps between JUG exports and pytempo oracle."""

    fixture_id: str
    n_toa: int
    pytempo_batcorr_closure_max_ns: float
    jug_replay_all_jug_rms_ns: float
    jug_replay_all_pytempo_rms_ns: float
    component_rms_ns: dict[str, float]
    swap_one_rms_ns: dict[str, float]
    notes: list[str] = field(default_factory=list)


def _jug_formbats_slots(jug_result: dict) -> dict[str, np.ndarray]:
    td = jug_result["term_diagnostics"]
    return {
        "tt": np.asarray(
            td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
            dtype=np.float64,
        ),
        "tt_tb": np.asarray(td["correction_tt_tb_sec"], dtype=np.float64),
        "tropo": np.asarray(td["tropo_delay_sec"], dtype=np.float64),
        "roemer": -np.asarray(td["roemer_sec"], dtype=np.float64),
        "shap": np.asarray(td["sun_shapiro_sec"], dtype=np.float64)
        + np.asarray(td["planet_shapiro_sec"], dtype=np.float64),
        "tdis1": np.asarray(td["dm_delay_sec"], dtype=np.float64)
        + np.asarray(td.get("dmx_delay_sec", 0.0), dtype=np.float64),
        "tdis2": np.asarray(td["sw_delay_sec"], dtype=np.float64),
    }


def _pytempo_formbats_slots(oracle_fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "tt": np.asarray(oracle_fields["correction_tt_sec"], dtype=np.float64),
        "tt_tb": np.asarray(oracle_fields["correction_tt_tb_sec"], dtype=np.float64),
        "tropo": np.asarray(oracle_fields["tropospheric_sec"], dtype=np.float64),
        "roemer": np.asarray(oracle_fields["roemer_sec"], dtype=np.float64),
        "shap": np.asarray(oracle_fields["shapiro_delay_sec"], dtype=np.float64),
        "tdis1": np.asarray(oracle_fields["tdis1_sec"], dtype=np.float64),
        "tdis2": np.asarray(oracle_fields["tdis2_sec"], dtype=np.float64),
    }


def replay_batcorr_days(slots: dict[str, np.ndarray]) -> np.ndarray:
    sec = formbats_replay_batcorr_tempo2_sec(
        slots["tt"],
        slots["tt_tb"],
        slots["tropo"],
        slots["roemer"],
        slots["shap"],
        slots["tdis1"],
        slots["tdis2"],
    )
    return sec / SECS_PER_DAY


def compare_formbats_components(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    jug_result: dict | None = None,
) -> FormbatsComponentReport:
    """Rank JUG vs pytempo formBats slots and swap-one sensitivities."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    par_path = Path(par)
    tim_path = Path(tim)
    if jug_result is None:
        jug_result = compute_residuals_simple(
            par_path, tim_path, verbose=False, compatibility="tempo2"
        )
    oracle = load_pytempo_native_oracle(
        par_path, tim_path, fixture_id=fixture_id, include_delay_chain=True
    )
    pt = oracle.fields
    jug = _jug_formbats_slots(jug_result)
    base = _pytempo_formbats_slots(pt)

    component_rms_ns = {
        name: rms_ns(jug[name] - base[name])
        for name in base
    }
    pt_target = np.asarray(pt["bat_corr_days"], dtype=np.float64)
    swap_one_rms_ns = {}
    for name in base:
        trial = dict(base)
        trial[name] = jug[name]
        swap_one_rms_ns[name] = rms_ns(replay_batcorr_days(trial) - pt_target, is_mjd=True)

    closure = np.abs(np.asarray(pt.get("bat_corr_closure_ns", 0.0), dtype=np.float64))
    notes = [
        "swap_one: start from pytempo slots, replace one slot with JUG export.",
        "tt uses JUG formbats_correction_tt_sec (astropy TT−sat), not utc_to_tdb proxy.",
    ]
    return FormbatsComponentReport(
        fixture_id=fixture_id or par_path.stem,
        n_toa=int(len(pt_target)),
        pytempo_batcorr_closure_max_ns=float(np.max(closure)) if closure.size else 0.0,
        jug_replay_all_jug_rms_ns=rms_ns(
            replay_batcorr_days(jug) - pt_target, is_mjd=True
        ),
        jug_replay_all_pytempo_rms_ns=rms_ns(
            replay_batcorr_days(base) - pt_target, is_mjd=True
        ),
        component_rms_ns=component_rms_ns,
        swap_one_rms_ns=swap_one_rms_ns,
        notes=notes,
    )
