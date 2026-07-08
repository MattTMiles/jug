"""Clock-feedback / -addsat dt-chain diagnostics (dev oracle only).

Compares JUG residual scatter vs libstempo to the per-TOA clkcorr.C feedback
delta (``getCorrectionTT(feedback=3) − getCorrectionTT(feedback=1)``), the
spin-argument gap ``dt_jug − deltaT(pytempo bbat, torb)``, and validates
``-addsat`` SAT application against pytempo ``sat_mjd``.  See
``PARITY_ROADMAP.md`` parity-closure plan (2026-07-08).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jug.io.clock import resolve_clock_dir
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import (
    _load_clock_corrections,
    compute_residuals_simple,
)
from jug.residuals.tempo2_clock import compute_get_correction_tt_sec
from jug.testing.tempo2_reference import tempo2_reference
from jug.utils.constants import OBSERVATORIES, SECS_PER_DAY


@dataclass
class BackendDtchainRow:
    """Per-backend residual statistics."""

    sys_name: str
    n_toa: int
    rms_ns: float
    max_ns: float
    rms_after_feedback_sub_ns: float
    max_after_feedback_sub_ns: float


@dataclass
class AddsatDtchainReport:
    """JUG−libstempo residual vs clock-feedback / dt-chain on one fixture."""

    fixture_id: str
    n_toa: int
    residual_rms_ns: float
    residual_max_ns: float
    feedback_delta_rms_ns: float
    feedback_delta_max_ns: float
    corr_delta_vs_feedback: float
    predicted_rms_after_feedback_ns: float
    predicted_max_after_feedback_ns: float
    dt_chain_gap_rms_ns: float
    corr_delta_vs_dt_chain_gap: float
    predicted_rms_after_dt_chain_ns: float
    sat_vs_pytempo_max_ns: float
    addsat_closure_max_sec: float
    addsat_sat_vs_pytempo_max_ns: float = 0.0
    backends: list[BackendDtchainRow] = field(default_factory=list)
    addsat_toa_indices: list[int] = field(default_factory=list)
    addsat_toa_rms_ns: float = 0.0
    addsat_toa_max_ns: float = 0.0
    non_addsat_rms_ns: float = 0.0
    notes: list[str] = field(default_factory=list)


def _parse_tim_sys_names(tim_path: Path) -> list[str]:
    """Return ``-sys`` backend name per TOA line (``?`` if absent)."""
    sys_names: list[str] = []
    for line in tim_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("FORMAT", "MODE", "INCLUDE")):
            continue
        match = re.search(r"-sys\s+(\S+)", line)
        sys_names.append(match.group(1) if match else "?")
    return sys_names


def _compute_feedback_delta_sec(
    toas: list[Any],
    *,
    obs_clocks: dict,
    obs_clock: dict,
    bipm_clock: dict,
    all_obs_codes: list[str],
    time_offsets: np.ndarray,
) -> np.ndarray:
    """Per-TOA clkcorr.C feedback delta in seconds (feedback_iters 3 − 1)."""
    common = dict(
        toas=toas,
        obs_clocks=obs_clocks,
        obs_clock_default=obs_clock,
        bipm_clock=bipm_clock,
        all_obs_codes=all_obs_codes,
        time_offsets=time_offsets,
    )
    tt_fb = compute_get_correction_tt_sec(**common, feedback_iters=3)
    tt_raw = compute_get_correction_tt_sec(**common, feedback_iters=1)
    return np.asarray(tt_fb, dtype=np.float64) - np.asarray(tt_raw, dtype=np.float64)


def _load_clock_context(
    par_path: Path,
    tim_path: Path,
    *,
    compatibility: str = "tempo2",
) -> tuple[list[Any], dict, dict[str, dict], dict, list[str], np.ndarray, str]:
    """Load TOAs and clock tables the same way as ``compute_residuals_simple``."""
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    observatory = toas[0].observatory if toas else "auto"
    if observatory == "auto" or OBSERVATORIES.get(observatory.lower()) is None:
        observatory = toas[0].observatory
    all_obs_codes = sorted(set(t.observatory.lower() for t in toas))
    tzr_site_code = str(params.get("TZRSITE", "")).lower()
    if tzr_site_code and tzr_site_code not in all_obs_codes and tzr_site_code not in (
        "ssb",
        "@",
        "coe",
        "",
    ):
        all_obs_codes = sorted(set(all_obs_codes) | {tzr_site_code})
    mjd_utc = np.array([t.mjd_int + t.mjd_frac for t in toas])
    clock_dir = resolve_clock_dir(None, compatibility=compatibility)
    clk = _load_clock_corrections(
        observatory, all_obs_codes, clock_dir, params, mjd_utc, verbose=False
    )
    time_offsets = np.array([float(t.flags.get("to", 0.0)) for t in toas])
    return (
        toas,
        params,
        clk["obs_clocks"],
        clk["obs_clock"],
        clk["bipm_clock"],
        all_obs_codes,
        time_offsets,
        observatory,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def diagnose_addsat_dtchain(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    compatibility: str = "tempo2",
) -> AddsatDtchainReport:
    """Diagnose residual scatter vs clkcorr feedback delta and ``-addsat`` SAT."""
    par_path = Path(par)
    tim_path = Path(tim)

    params = parse_par_file(par_path)
    pepoch = float(params["PEPOCH"])

    jug = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility=compatibility
    )
    ref = tempo2_reference(par_path, tim_path)

    # pytempo ``bbat``/``torb`` diagnostics after libstempo sandbox (in-process
    # tempo2 state is independent of the sandbox worker).
    from pytempo.sandbox import tempopulsar as pytempo_pulsar

    pt = pytempo_pulsar(parfile=str(par_path), timfile=str(tim_path), dofit=False)
    pt_diag = pt.toa_diagnostics(removemean=False)

    dt_jug = np.asarray(jug["dt_sec"], dtype=np.float64)
    deltaT_pt = (
        (np.asarray(pt_diag["bbat_mjd"], dtype=np.float64) - pepoch) * SECS_PER_DAY
        + np.asarray(pt_diag["torb_sec"], dtype=np.float64)
    )
    dt_chain_gap_ns = (dt_jug - deltaT_pt) * 1e9
    gap_mean = float(np.mean(dt_chain_gap_ns))

    delta_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    delta_ns = delta_ns - np.mean(delta_ns)
    corr_dt = _safe_corr(delta_ns, dt_chain_gap_ns)
    predicted_dt = delta_ns - (dt_chain_gap_ns - gap_mean)

    (
        toas,
        _params2,
        obs_clocks,
        obs_clock,
        bipm_clock,
        all_obs_codes,
        time_offsets,
        _observatory,
    ) = _load_clock_context(par_path, tim_path, compatibility=compatibility)

    feedback_delta_sec = _compute_feedback_delta_sec(
        toas,
        obs_clocks=obs_clocks,
        obs_clock=obs_clock,
        bipm_clock=bipm_clock,
        all_obs_codes=all_obs_codes,
        time_offsets=time_offsets,
    )
    feedback_delta_ns = feedback_delta_sec * 1e9

    fb_mean = float(np.mean(feedback_delta_ns))
    predicted_fb = delta_ns - (feedback_delta_ns - fb_mean)
    corr_fb = _safe_corr(delta_ns, feedback_delta_ns)

    jug_sat = np.array(
        [np.longdouble(t.mjd_int) + np.longdouble(t.mjd_frac) for t in toas],
        dtype=np.float64,
    )
    pt_sat = np.asarray(pt_diag["sat_mjd"], dtype=np.float64)
    sat_vs_pt_max_ns = float(np.max(np.abs((jug_sat - pt_sat) * SECS_PER_DAY)) * 1e9)

    addsat_closure_max_sec = 0.0
    addsat_sat_max_ns = 0.0
    for idx, toa in enumerate(toas):
        if "addsat" not in toa.flags:
            continue
        try:
            expected_sec = float(toa.flags["addsat"])
        except (ValueError, TypeError):
            continue
        addsat_closure_max_sec = max(
            addsat_closure_max_sec, abs(abs(expected_sec) - 1.0)
        )
        jug_sat_i = float(np.longdouble(toa.mjd_int) + np.longdouble(toa.mjd_frac))
        addsat_sat_max_ns = max(
            addsat_sat_max_ns,
            abs(jug_sat_i - pt_sat[idx]) * SECS_PER_DAY * 1e9,
        )

    addsat_idx = [i for i, t in enumerate(toas) if "addsat" in t.flags]
    non_idx = [i for i in range(len(delta_ns)) if i not in addsat_idx]
    addsat_rms = addsat_max = 0.0
    non_addsat_rms = float(np.sqrt(np.mean(delta_ns[non_idx] ** 2))) if non_idx else 0.0
    if addsat_idx:
        sub = delta_ns[addsat_idx]
        addsat_rms = float(np.sqrt(np.mean(sub**2)))
        addsat_max = float(np.max(np.abs(sub)))

    sys_names = _parse_tim_sys_names(tim_path)
    if len(sys_names) != len(delta_ns):
        sys_names = ["?"] * len(delta_ns)

    backends: list[BackendDtchainRow] = []
    for sys_name in sorted(set(sys_names)):
        idx = [i for i, s in enumerate(sys_names) if s == sys_name]
        sub = delta_ns[idx]
        sub_pred = predicted_fb[idx]
        backends.append(
            BackendDtchainRow(
                sys_name=sys_name,
                n_toa=len(idx),
                rms_ns=float(np.sqrt(np.mean(sub**2))),
                max_ns=float(np.max(np.abs(sub))),
                rms_after_feedback_sub_ns=float(np.sqrt(np.mean(sub_pred**2))),
                max_after_feedback_sub_ns=float(np.max(np.abs(sub_pred))),
            )
        )

    notes = [
        "feedback_delta = getCorrectionTT(feedback_iters=3) − getCorrectionTT(1).",
        "dt_chain_gap = dt_jug − ((pytempo bbat − PEPOCH)*86400 + torb).",
        "predicted residual subtracts (term_ns − mean) for each diagnostic term.",
        "Merged IPTA clock chains converge in one iter → feedback_delta often 0.",
        "Naive dt_jug−deltaT(pytempo) correlates weakly with residual on IPTA full "
        "set (~0.07): Taylor spin partially absorbs tempo2 deltaT; residual debt is "
        "multi-term (site/backend scatter), not a single linear dt gap.",
    ]
    return AddsatDtchainReport(
        fixture_id=fixture_id or par_path.stem,
        n_toa=int(jug["n_toas"]),
        residual_rms_ns=float(np.sqrt(np.mean(delta_ns**2))),
        residual_max_ns=float(np.max(np.abs(delta_ns))),
        feedback_delta_rms_ns=float(np.sqrt(np.mean(feedback_delta_ns**2))),
        feedback_delta_max_ns=float(np.max(np.abs(feedback_delta_ns))),
        corr_delta_vs_feedback=corr_fb,
        predicted_rms_after_feedback_ns=float(np.sqrt(np.mean(predicted_fb**2))),
        predicted_max_after_feedback_ns=float(np.max(np.abs(predicted_fb))),
        dt_chain_gap_rms_ns=float(np.sqrt(np.mean(dt_chain_gap_ns**2))),
        corr_delta_vs_dt_chain_gap=corr_dt,
        predicted_rms_after_dt_chain_ns=float(np.sqrt(np.mean(predicted_dt**2))),
        sat_vs_pytempo_max_ns=sat_vs_pt_max_ns,
        addsat_sat_vs_pytempo_max_ns=float(addsat_sat_max_ns),
        addsat_closure_max_sec=float(addsat_closure_max_sec),
        backends=backends,
        addsat_toa_indices=addsat_idx,
        addsat_toa_rms_ns=addsat_rms,
        addsat_toa_max_ns=addsat_max,
        non_addsat_rms_ns=non_addsat_rms,
        notes=notes,
    )


def format_addsat_dtchain_report(report: AddsatDtchainReport) -> str:
    """Human-readable diagnostic report."""
    lines = [
        f"fixture: {report.fixture_id}",
        f"n_toa: {report.n_toa}",
        "",
        "=== residual vs libstempo (mean removed) ===",
        f"RMS: {report.residual_rms_ns:.3f} ns  max: {report.residual_max_ns:.3f} ns",
        f"non-addsat RMS: {report.non_addsat_rms_ns:.3f} ns",
        "",
        "=== clkcorr feedback delta (3 − 1 iter) ===",
        f"RMS: {report.feedback_delta_rms_ns:.3f} ns  max: {report.feedback_delta_max_ns:.3f} ns",
        f"corr(residual, feedback_delta): {report.corr_delta_vs_feedback:.4f}",
        f"predicted RMS after subtract: {report.predicted_rms_after_feedback_ns:.3f} ns",
        "",
        "=== spin dt-chain gap (dt_jug − pytempo deltaT) ===",
        f"gap RMS: {report.dt_chain_gap_rms_ns:.3f} ns",
        f"corr(residual, dt_chain_gap): {report.corr_delta_vs_dt_chain_gap:.4f}",
        f"predicted RMS after subtract: {report.predicted_rms_after_dt_chain_ns:.3f} ns",
        "",
        "=== -addsat SAT oracle ===",
        f"max |jug_sat − pytempo sat_mjd|: {report.sat_vs_pytempo_max_ns:.3f} ns",
        f"addsat TOA indices: {report.addsat_toa_indices}",
        f"addsat TOA residual RMS/max: {report.addsat_toa_rms_ns:.3f} / {report.addsat_toa_max_ns:.3f} ns",
        "",
        "=== per-backend (-sys) ===",
    ]
    for row in sorted(report.backends, key=lambda r: -r.n_toa):
        lines.append(
            f"  {row.sys_name:16} n={row.n_toa:4} "
            f"rms={row.rms_ns:8.3f} max={row.max_ns:8.3f}  "
            f"pred_rms={row.rms_after_feedback_sub_ns:8.3f}"
        )
    if report.notes:
        lines.extend(["", *report.notes])
    return "\n".join(lines)
