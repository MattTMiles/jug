"""Phase A term-by-term comparison runner for tempo2 diagnostics.

Compares JUG pint/tempo2 modes against libstempo oracle terms and ranks
which delay component dominates the raw residual gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jug.residuals.diagnostic_conventions import DiagnosticConventions
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_diagnostics import Tempo2TermDiagnostics, tempo2_term_diagnostics


@dataclass
class TermDeltaStats:
    term: str
    rms_ns: float
    wrms_ns: float
    p99_ns: float
    max_abs_ns: float
    mean_ns: float
    annual_amp_ns: float | None = None


@dataclass
class PhaseAComparisonReport:
    fixture_id: str
    par_path: Path
    tim_path: Path
    conventions: DiagnosticConventions
    residual_stats: dict[str, TermDeltaStats] = field(default_factory=dict)
    term_stats: dict[str, TermDeltaStats] = field(default_factory=dict)
    ranking: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _stats_dict(stats: dict[str, TermDeltaStats]) -> dict[str, Any]:
            return {
                name: {
                    "rms_ns": s.rms_ns,
                    "wrms_ns": s.wrms_ns,
                    "p99_ns": s.p99_ns,
                    "max_abs_ns": s.max_abs_ns,
                    "mean_ns": s.mean_ns,
                    "annual_amp_ns": s.annual_amp_ns,
                }
                for name, s in stats.items()
            }

        return {
            "fixture_id": self.fixture_id,
            "par_path": str(self.par_path),
            "tim_path": str(self.tim_path),
            "conventions": {
                "residual_metric": self.conventions.residual_metric,
                "tempo2_tdb_defaults": self.conventions.tempo2_tdb_defaults,
                "oracle_terms": self.conventions.oracle_terms,
                "term_set": self.conventions.term_set,
            },
            "residual_stats": _stats_dict(self.residual_stats),
            "term_stats": _stats_dict(self.term_stats),
            "ranking": list(self.ranking),
            "notes": list(self.notes),
        }


def delta_stats_ns(
    a: np.ndarray,
    b: np.ndarray,
    *,
    errors_us: np.ndarray | None = None,
    model_mjd: np.ndarray | None = None,
    residual_metric: str = "raw",
) -> TermDeltaStats:
    """Compute raw (or weighted-centered) delta statistics in nanoseconds."""
    delta_ns = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * 1000.0
    if residual_metric == "weighted_centered" and errors_us is not None:
        weights = 1.0 / np.square(np.asarray(errors_us, dtype=np.float64))
        delta_ns = delta_ns - np.average(delta_ns, weights=weights)
    elif residual_metric == "weighted_centered":
        delta_ns = delta_ns - np.mean(delta_ns)

    wrms = float(np.sqrt(np.mean(np.square(delta_ns))))
    if errors_us is not None and residual_metric == "raw":
        weights = 1.0 / np.square(np.asarray(errors_us, dtype=np.float64))
        wrms = float(
            np.sqrt(np.sum(weights * np.square(delta_ns)) / np.sum(weights))
        )

    annual_amp = None
    if model_mjd is not None and delta_ns.size >= 3:
        t = np.asarray(model_mjd, dtype=np.float64)
        t0 = np.mean(t)
        design = np.column_stack(
            [
                np.ones_like(t),
                np.sin(2.0 * np.pi * (t - t0) / 365.25),
                np.cos(2.0 * np.pi * (t - t0) / 365.25),
            ]
        )
        coef, _, _, _ = np.linalg.lstsq(design, delta_ns, rcond=None)
        annual_amp = float(np.sqrt(coef[1] ** 2 + coef[2] ** 2))

    return TermDeltaStats(
        term="",
        rms_ns=float(np.sqrt(np.mean(np.square(delta_ns)))),
        wrms_ns=wrms,
        p99_ns=float(np.percentile(np.abs(delta_ns), 99)),
        max_abs_ns=float(np.max(np.abs(delta_ns))),
        mean_ns=float(np.mean(delta_ns)),
        annual_amp_ns=annual_amp,
    )


def _apply_metric(
    jug_vals: np.ndarray,
    oracle_vals: np.ndarray,
    *,
    errors_us: np.ndarray | None,
    model_mjd: np.ndarray | None,
    conventions: DiagnosticConventions,
    term: str,
) -> TermDeltaStats:
    stats = delta_stats_ns(
        jug_vals,
        oracle_vals,
        errors_us=errors_us,
        model_mjd=model_mjd,
        residual_metric=conventions.residual_metric,
    )
    stats.term = term
    return stats


def compare_fixture_phase_a(
    fixture: dict[str, Any],
    *,
    conventions: DiagnosticConventions | None = None,
) -> PhaseAComparisonReport:
    """Run Phase A comparison for one fixture."""
    conv = conventions or DiagnosticConventions()
    conv.validate_for_tempo2_acceptance()

    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    fixture_id = str(fixture["id"])

    oracle = tempo2_term_diagnostics(par_path, tim_path, conventions=conv)
    jug_t2 = compute_residuals_simple(
        par_path,
        tim_path,
        verbose=False,
        compatibility="tempo2",
        diagnostic_conventions=conv,
    )
    jug_pint = compute_residuals_simple(
        par_path,
        tim_path,
        verbose=False,
        compatibility="pint",
        diagnostic_conventions=conv,
    )

    model_mjd = jug_t2.get("model_mjd", jug_t2["tdb_mjd"])
    report = PhaseAComparisonReport(
        fixture_id=fixture_id,
        par_path=par_path,
        tim_path=tim_path,
        conventions=conv,
    )

    report.residual_stats["jug_tempo2_minus_oracle"] = _apply_metric(
        jug_t2["residuals_us"],
        oracle.residuals_us,
        errors_us=oracle.errors_us,
        model_mjd=model_mjd,
        conventions=conv,
        term="jug_tempo2_minus_oracle",
    )
    report.residual_stats["jug_pint_minus_oracle"] = _apply_metric(
        jug_pint["residuals_us"],
        oracle.residuals_us,
        errors_us=oracle.errors_us,
        model_mjd=model_mjd,
        conventions=conv,
        term="jug_pint_minus_oracle",
    )
    report.residual_stats["jug_tempo2_minus_jug_pint"] = _apply_metric(
        jug_t2["residuals_us"],
        jug_pint["residuals_us"],
        errors_us=jug_t2["errors_us"],
        model_mjd=model_mjd,
        conventions=conv,
        term="jug_tempo2_minus_jug_pint",
    )

    # Mode-internal term deltas (same physics path on TDB should collapse here)
    for term_key in (
        "roemer_sec",
        "sun_shapiro_sec",
        "planet_shapiro_sec",
        "roemer_shapiro_sec",
        "dm_delay_sec",
        "sw_delay_sec",
        "freq_bary_mhz",
    ):
        t2_vals = jug_t2.get("term_diagnostics", {}).get(term_key, jug_t2.get(term_key))
        pint_vals = jug_pint.get("term_diagnostics", {}).get(term_key, jug_pint.get(term_key))
        if t2_vals is None or pint_vals is None:
            continue
        stats = _apply_metric(
            np.asarray(t2_vals, dtype=np.float64),
            np.asarray(pint_vals, dtype=np.float64),
            errors_us=jug_t2["errors_us"],
            model_mjd=model_mjd,
            conventions=conv,
            term=f"jug_tempo2_minus_jug_pint::{term_key}",
        )
        report.term_stats[f"pint_mode_delta::{term_key}"] = stats

    term_pairs: list[tuple[str, str, np.ndarray | None]] = [
        ("roemer_sec", "roemer_sec", oracle.roemer_sec),
        ("sun_shapiro_sec", "shapiro_sun_sec", oracle.shapiro_sun_sec),
        ("freq_bary_mhz", "ssbfreqs_mhz", oracle.ssbfreqs_mhz),
    ]
    if oracle.bbat_mjd is not None:
        term_pairs.append(("bbat_mjd", "bbat_mjd", oracle.bbat_mjd))
    if oracle.pulse_number is not None:
        term_pairs.append(("pulse_number", "pulse_number", oracle.pulse_number))
    if oracle.phase_offset_turns is not None and np.any(oracle.phase_offset_turns):
        term_pairs.append(("jump_phase", "phase_offset_turns", oracle.phase_offset_turns))
    if conv.term_set == "extended":
        term_pairs.append(("tzr_phase", "tzr_phase", None))

    for jug_key, oracle_key, oracle_vals in term_pairs:
        if oracle_vals is None:
            report.notes.append(f"{oracle_key}: unavailable in oracle")
            continue
        jug_vals = jug_t2.get("term_diagnostics", {}).get(jug_key)
        if jug_vals is None:
            jug_vals = jug_t2.get(jug_key)
        if jug_vals is None:
            report.notes.append(f"{jug_key}: unavailable in JUG")
            continue
        if jug_key == "jump_phase":
            jug_vals = np.asarray(jug_vals, dtype=np.float64)
            oracle_vals = np.asarray(oracle_vals, dtype=np.float64)
        if jug_key == "freq_bary_mhz":
            oracle_vals = np.asarray(oracle_vals, dtype=np.float64) / 1.0e6
        stats = _apply_metric(
            np.asarray(jug_vals, dtype=np.float64),
            np.asarray(oracle_vals, dtype=np.float64),
            errors_us=oracle.errors_us,
            model_mjd=model_mjd,
            conventions=conv,
            term=jug_key,
        )
        # tempo2 ``roemer`` includes PM/parallax terms; JUG ``roemer_sec`` is geometric
        # only — skip absolute oracle ranking when magnitudes are incomparable.
        if jug_key == "roemer_sec" and stats.rms_ns > 1.0e6:
            report.notes.append(
                f"{jug_key}: oracle comparison skipped (JUG geometric vs tempo2 full Roemer)"
            )
            continue
        report.term_stats[f"oracle_delta::{jug_key}"] = stats

    roemer_shapiro = jug_t2.get("roemer_shapiro_sec")
    if oracle.roemer_sec is not None and oracle.shapiro_sun_sec is not None and roemer_shapiro is not None:
        oracle_combo = np.asarray(oracle.roemer_sec) + np.asarray(oracle.shapiro_sun_sec)
        stats = _apply_metric(
            np.asarray(roemer_shapiro, dtype=np.float64),
            oracle_combo,
            errors_us=oracle.errors_us,
            model_mjd=model_mjd,
            conventions=conv,
            term="roemer_shapiro_sec",
        )
        if stats.rms_ns <= 1.0e6:
            report.term_stats["oracle_delta::roemer_shapiro_sec"] = stats
        else:
            report.notes.append(
                "roemer_shapiro_sec: oracle combo skipped (JUG geometric vs tempo2 full Roemer)"
            )

    # Rank by oracle residual gap first, then oracle term gaps, then pint/tempo2 term gaps
    residual_rms = report.residual_stats.get("jug_tempo2_minus_oracle")
    if residual_rms is not None:
        report.notes.append(
            f"Residual gap jug(tempo2)-oracle: RMS={residual_rms.rms_ns:.3f} ns, "
            f"annual~={residual_rms.annual_amp_ns}"
        )

    ranked = sorted(
        (
            (name, stats)
            for name, stats in report.term_stats.items()
            if name.startswith("oracle_delta::")
        ),
        key=lambda item: item[1].rms_ns,
        reverse=True,
    )
    if not ranked:
        ranked = sorted(
            (
                (name, stats)
                for name, stats in report.term_stats.items()
                if name.startswith("pint_mode_delta::")
            ),
            key=lambda item: item[1].rms_ns,
            reverse=True,
        )
    report.ranking = [name.split("::", 1)[-1] for name, _ in ranked]
    return report


def rank_phase_b_ports(report: PhaseAComparisonReport) -> list[str]:
    """Map Phase A term ranking to suggested Phase B port order."""
    priority = [
        "roemer_shapiro_sec",
        "roemer_sec",
        "sun_shapiro_sec",
        "freq_bary_mhz",
        "dm_delay_sec",
        "sw_delay_sec",
        "tzr_phase",
        "tropo_delay_sec",
    ]
    ranked_names = [name.split("::", 1)[-1] for name in report.ranking]
    ranked = [t for t in priority if t in ranked_names]
    for term in ranked_names:
        if term not in ranked:
            ranked.append(term)
    return ranked
