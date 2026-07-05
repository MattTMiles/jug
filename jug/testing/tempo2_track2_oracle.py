"""TRACK −2 / ``pnNew`` oracle helpers for tempo2 parity debugging.

Uses pytempo per-TOA diagnostics when available (see ``ref-packages/pytempo/README.md``
Tier 1–2).  Not a runtime dependency of JUG production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2_spin import (
    _c_int_truncate,
    _fortran_mod,
    _fortran_nlong,
    compute_tempo2_phase5,
    compute_tempo2_torb_sec,
    track_minus2_frac_phase,
)


@dataclass
class Track2OracleContext:
    """Inputs for TRACK −2 component checks on one par/tim pair."""

    par_path: Path
    tim_path: Path
    params: dict[str, Any]
    toas: list[Any]
    pn_tim: np.ndarray
    pn_add: np.ndarray
    bbat_mjd: np.ndarray
    dt_sec: np.ndarray
    jump_phase: np.ndarray | None
    f0: float
    pytempo_diag: dict[str, np.ndarray] | None = None


def load_track2_oracle_context(
    par_path: str | Path,
    tim_path: str | Path,
    *,
    use_pytempo: bool = True,
) -> Track2OracleContext:
    """Load wsrt167-style TRACK −2 oracle context (JUG + optional pytempo)."""
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    base = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility="tempo2"
    )

    pn_tim = np.array([int(t.flags["pn"]) for t in toas], dtype=np.int64)
    pn_add = np.full(len(toas), -1, dtype=np.int64)
    running = np.int64(-1)
    for i, toa in enumerate(toas):
        pn_add[i] = running
        pnadd_val = toa.flags.get("pnadd")
        if pnadd_val is not None:
            running += np.int64(int(pnadd_val))

    jump = base.get("jump_phase")
    jump_phase = None if jump is None else np.asarray(jump, dtype=np.float64)

    pytempo_diag = None
    if use_pytempo:
        try:
            from pytempo.sandbox import tempopulsar

            psr = tempopulsar(
                parfile=str(par_path), timfile=str(tim_path), dofit=False
            )
            pytempo_diag = psr.toa_diagnostics(removemean=False)
        except Exception:
            pytempo_diag = None

    if pytempo_diag is not None:
        bbat_mjd = np.asarray(pytempo_diag["bbat_mjd"], dtype=np.float64)
    else:
        bbat_mjd = np.asarray(base["term_diagnostics"]["bbat_mjd"], dtype=np.float64)

    return Track2OracleContext(
        par_path=par_path,
        tim_path=tim_path,
        params=params,
        toas=toas,
        pn_tim=pn_tim,
        pn_add=pn_add,
        bbat_mjd=bbat_mjd,
        dt_sec=np.asarray(base["dt_sec"], dtype=np.float64),
        jump_phase=jump_phase,
        f0=float(params["F0"]),
        pytempo_diag=pytempo_diag,
    )


def compute_pn_new_relative(
    phase5: np.ndarray,
    bbat_mjd: np.ndarray,
    f0: float,
) -> np.ndarray:
    """tempo2 ``pnNew`` after ``pn0`` anchoring (formResiduals.C TRACK −2)."""
    p5 = np.asarray(phase5, dtype=np.float64)
    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    phas1 = float(_fortran_mod(p5[0], 1.0))
    p5 = p5 - phas1
    nph = _fortran_nlong(p5)

    nf0 = int(f0)
    c_bbat = _c_int_truncate(bbat)
    c0 = int(c_bbat[0])
    pn0 = -1
    out = np.empty(len(p5), dtype=np.int64)
    for i in range(len(p5)):
        ntpd = int(c_bbat[i]) - c0
        pn_raw = int(nf0 * ntpd * 86400.0 + int(nph[i]))
        if pn0 == -1:
            pn0 = pn_raw
            out[i] = 0
        else:
            out[i] = pn_raw - pn0
    return out


def track2_add_phase_turns(ctx: Track2OracleContext) -> np.ndarray:
    """Per-TOA ``addPhase`` (turns) from JUG ``track_minus2_frac_phase`` components."""
    torb = compute_tempo2_torb_sec(ctx.bbat_mjd, ctx.dt_sec, float(ctx.params["PEPOCH"]))
    phase5 = compute_tempo2_phase5(
        ctx.bbat_mjd, torb, ctx.params, jump_phase=ctx.jump_phase
    )
    pn_new = compute_pn_new_relative(phase5, ctx.bbat_mjd, ctx.f0)
    pn_act = (ctx.pn_tim - ctx.pn_tim[0]) + ctx.pn_add
    return pn_new.astype(np.float64) - pn_act.astype(np.float64)


def track2_frac_phase_oracle(
    ctx: Track2OracleContext,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(frac_turns, add_phase_turns, pn_new_rel)`` at oracle ``bbat``."""
    torb = compute_tempo2_torb_sec(ctx.bbat_mjd, ctx.dt_sec, float(ctx.params["PEPOCH"]))
    phase5 = compute_tempo2_phase5(
        ctx.bbat_mjd, torb, ctx.params, jump_phase=ctx.jump_phase
    )
    frac, _ = track_minus2_frac_phase(
        phase5, ctx.bbat_mjd, ctx.f0, ctx.pn_tim, ctx.pn_add
    )
    pn_new = compute_pn_new_relative(phase5, ctx.bbat_mjd, ctx.f0)
    add_phase = track2_add_phase_turns(ctx)
    return frac, add_phase, pn_new
