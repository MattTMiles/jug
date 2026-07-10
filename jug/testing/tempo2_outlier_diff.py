"""Per-TOA clock and Roemer diff harness for tempo2 parity outliers.

Compares JUG ``term_diagnostics`` against libstempo per-TOA properties for fixtures
that fail the strict ns gate (notably ``epta_j0030_isolated`` and ``wsrt167``).

Test-only: uses libstempo via the sandbox oracle. See ``jug/testing/DEV_ORACLE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference
from jug.utils.constants import SECS_PER_DAY


def _sandbox_array(psr: Any, name: str) -> np.ndarray | None:
    if not hasattr(psr, name):
        return None
    value = getattr(psr, name)
    arr = value() if callable(value) else value
    return np.asarray(arr, dtype=np.float64)


@dataclass
class ToaClockRoemerDiff:
    """Per-TOA JUG vs libstempo clock / Roemer comparison."""

    index: int
    mjd: float
    freq_mhz: float
    residual_diff_ns: float
    roemer_diff_ns: float
    sun_shapiro_diff_ns: float
    sat_diff_ns: float
    bat_diff_ns: float
    jug_correction_tt_us: float
    jug_correction_tt_tb_us: float
    jug_prebinary_us: float
    t2_bat_corr_us: float
    flags: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierClockRoemerReport:
    """Summary of a fixture-level clock / Roemer diff run."""

    fixture_id: str
    n_toa: int
    residual_rms_ns: float
    roemer_rms_ns: float
    sat_rms_ns: float
    bat_rms_ns: float
    outlier_indices: list[int]
    rows: list[ToaClockRoemerDiff]
    notes: list[str] = field(default_factory=list)


def compare_clock_roemer_per_toa(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    outlier_threshold_ns: float = 10.0,
    policy: Any | None = None,
) -> OutlierClockRoemerReport:
    """Compare JUG vs libstempo clock / Roemer terms for every TOA.

    Sign convention: libstempo ``roemer`` matches ``-jug_roemer_sec`` to ~ULP;
    ``roemer_diff_ns = (jug_roemer_sec + t2_roemer_sec) * 1e9``.

    ``bat_diff_ns`` compares JUG formBats ``bat_mjd`` (diagnostic) to libstempo
    ``stoas + batCorrs``. Production spin does **not** use formBats ``bat``.
    """
    from jug.testing.sandbox_tempo2 import Policy, tempopulsar

    par_path = Path(par)
    tim_path = Path(tim)
    jug = compute_residuals_simple(par_path, tim_path, verbose=False, compatibility="tempo2")
    ref = tempo2_reference(par_path, tim_path)
    residual_diff_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64) * 1.0e-6
        - np.asarray(ref.residuals_us, dtype=np.float64) * 1.0e-6
    ) * 1.0e9
    residual_diff_ns = residual_diff_ns - np.mean(residual_diff_ns)

    td = jug["term_diagnostics"]
    jug_roemer = np.asarray(td["roemer_sec"], dtype=np.float64)
    jug_shapiro = np.asarray(td["sun_shapiro_sec"], dtype=np.float64)
    jug_sat = np.asarray(td.get("sat_mjd", jug["tdb_mjd"]), dtype=np.float64)
    jug_bat = np.asarray(td["bat_mjd"], dtype=np.float64) if "bat_mjd" in td else None
    jug_corr_tt = np.asarray(td.get("correction_tt_sec", np.zeros_like(jug_roemer)), dtype=np.float64)
    jug_corr_tt_tb = np.asarray(
        td.get("correction_tt_tb_sec", np.zeros_like(jug_roemer)), dtype=np.float64
    )
    jug_prebinary = np.asarray(td["prebinary_delay_sec"], dtype=np.float64)

    psr = tempopulsar(
        parfile=str(par_path),
        timfile=str(tim_path),
        dofit=False,
        policy=policy or Policy(call_timeout_s=180.0),
    )
    t2_roemer = _sandbox_array(psr, "roemer")
    t2_shapiro = _sandbox_array(psr, "shapiro_sun")
    t2_stoas = _sandbox_array(psr, "stoas")
    t2_bat_corr = _sandbox_array(psr, "batCorrs")
    if t2_roemer is None or t2_stoas is None or t2_bat_corr is None:
        raise RuntimeError("libstempo missing roemer/stoas/batCorrs for clock/Roemer diff")

    roemer_diff_ns = (jug_roemer + t2_roemer) * 1.0e9
    shapiro_diff_ns = np.zeros_like(roemer_diff_ns)
    if t2_shapiro is not None:
        shapiro_diff_ns = (jug_shapiro - t2_shapiro) * 1.0e9

    sat_diff_ns = (jug_sat - t2_stoas) * SECS_PER_DAY * 1.0e9
    t2_bat_mjd = t2_stoas + t2_bat_corr
    bat_diff_ns = np.zeros_like(roemer_diff_ns)
    if jug_bat is not None:
        bat_diff_ns = (jug_bat - t2_bat_mjd) * SECS_PER_DAY * 1.0e9

    freqs = np.asarray(jug.get("freq_bary_mhz", jug["term_diagnostics"].get("freq_bary_mhz")), dtype=np.float64)
    if freqs.size != residual_diff_ns.size:
        freqs = np.full(residual_diff_ns.size, np.nan, dtype=np.float64)

    mjds = np.asarray(jug["model_mjd"], dtype=np.float64)
    outlier_mask = np.abs(residual_diff_ns) >= outlier_threshold_ns
    outlier_indices = np.where(outlier_mask)[0].tolist()

    rows: list[ToaClockRoemerDiff] = []
    flag_list = jug.get("toa_flags") or [{} for _ in range(len(residual_diff_ns))]
    for i in range(len(residual_diff_ns)):
        rows.append(
            ToaClockRoemerDiff(
                index=i,
                mjd=float(mjds[i]),
                freq_mhz=float(freqs[i]) if i < len(freqs) else float("nan"),
                residual_diff_ns=float(residual_diff_ns[i]),
                roemer_diff_ns=float(roemer_diff_ns[i]),
                sun_shapiro_diff_ns=float(shapiro_diff_ns[i]),
                sat_diff_ns=float(sat_diff_ns[i]),
                bat_diff_ns=float(bat_diff_ns[i]),
                jug_correction_tt_us=float(jug_corr_tt[i] * 1.0e6),
                jug_correction_tt_tb_us=float(jug_corr_tt_tb[i] * 1.0e6),
                jug_prebinary_us=float(jug_prebinary[i] * 1.0e6),
                t2_bat_corr_us=float(t2_bat_corr[i] * SECS_PER_DAY * 1.0e6),
                flags=dict(flag_list[i]) if i < len(flag_list) else {},
            )
        )

    notes = [
        "roemer_diff_ns ≈ 0 when JUG and libstempo Roemer agree (sign: jug = -t2 property).",
        "sat_diff_ns compares JUG sat_mjd to libstempo stoas (site arrival after clock).",
        "bat_diff_ns is diagnostic-only; production spin uses geometry model_mjd, not formBats bat.",
    ]
    return OutlierClockRoemerReport(
        fixture_id=fixture_id or par_path.stem,
        n_toa=len(rows),
        residual_rms_ns=float(np.sqrt(np.mean(residual_diff_ns ** 2))),
        roemer_rms_ns=float(np.sqrt(np.mean(roemer_diff_ns ** 2))),
        sat_rms_ns=float(np.sqrt(np.mean(sat_diff_ns ** 2))),
        bat_rms_ns=float(np.sqrt(np.mean(bat_diff_ns ** 2))) if jug_bat is not None else float("nan"),
        outlier_indices=outlier_indices,
        rows=rows,
        notes=notes,
    )


def format_outlier_report(report: OutlierClockRoemerReport, *, show_all: bool = False) -> str:
    """Human-readable report; by default prints outlier TOAs only."""
    lines = [
        f"fixture: {report.fixture_id}",
        f"n_toa: {report.n_toa}",
        f"residual_rms_ns: {report.residual_rms_ns:.2f}",
        f"roemer_rms_ns: {report.roemer_rms_ns:.2f}",
        f"sat_rms_ns: {report.sat_rms_ns:.2f}",
        f"bat_rms_ns: {report.bat_rms_ns:.2f}",
        f"outliers (|residual_diff| >= threshold): {report.outlier_indices}",
        "",
        "idx     MJD   freq  resid  roemer   sat    bat(s) corr_tt  prebin  flags",
    ]
    rows = report.rows if show_all else [r for r in report.rows if r.index in report.outlier_indices]
    if not rows and report.rows:
        rows = report.rows
    for row in rows:
        flag = row.flags.get("sys") or row.flags.get("group") or ""
        lines.append(
            f"{row.index:3d} {row.mjd:10.4f} {row.freq_mhz:6.0f} "
            f"{row.residual_diff_ns:7.1f} {row.roemer_diff_ns:6.2f} "
            f"{row.sat_diff_ns:6.2f} {row.bat_diff_ns / 1.0e9:8.3f}s "
            f"{row.jug_correction_tt_us:7.2f} {row.jug_prebinary_us:8.1f} {flag}"
        )
    if report.notes:
        lines.extend(["", *report.notes])
    return "\n".join(lines)
