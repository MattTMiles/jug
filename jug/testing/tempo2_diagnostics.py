"""Test-only tempo2 term diagnostics for Phase A comparisons.

Uses libstempo properties only. Does not participate in the runtime
``compatibility="tempo2"`` code path.

See ``jug/testing/DEV_ORACLE.md`` — delete oracle backends when JUG is standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jug.residuals.diagnostic_conventions import DiagnosticConventions


@dataclass
class Tempo2TermDiagnostics:
    """Per-TOA tempo2 delay/residual terms normalized for comparison."""

    residuals_us: np.ndarray
    errors_us: np.ndarray
    roemer_sec: np.ndarray | None = None
    shapiro_sun_sec: np.ndarray | None = None
    ssbfreqs_mhz: np.ndarray | None = None
    bat_corrs_sec: np.ndarray | None = None
    stoas_mjd: np.ndarray | None = None
    toas_mjd: np.ndarray | None = None
    bbat_mjd: np.ndarray | None = None
    phase_turns: np.ndarray | None = None
    nphase: np.ndarray | None = None
    phase_offset_turns: np.ndarray | None = None
    pulse_number: np.ndarray | None = None
    term_status: dict[str, str] = field(default_factory=dict)
    conventions: DiagnosticConventions | None = None

    @property
    def ntoa(self) -> int:
        return int(self.residuals_us.size)


def _safe_array(psr: Any, name: str) -> tuple[np.ndarray | None, str]:
    if not hasattr(psr, name):
        return None, "missing_property"
    try:
        value = getattr(psr, name)
        arr = value() if callable(value) else value
        out = np.asarray(arr, dtype=np.float64)
        if out.size == 0:
            return None, "empty"
        return out, "ok"
    except Exception as exc:
        return None, f"error:{type(exc).__name__}"


def tempo2_term_diagnostics(
    par: str | Path,
    tim: str | Path,
    *,
    conventions: DiagnosticConventions | None = None,
    policy: Any | None = None,
) -> Tempo2TermDiagnostics:
    """Collect per-term diagnostics for Phase A oracle comparisons via libstempo."""
    from jug.testing.sandbox_tempo2 import Policy, tempopulsar

    conv = conventions or DiagnosticConventions()

    psr = tempopulsar(
        parfile=str(par),
        timfile=str(tim),
        dofit=False,
        policy=policy or Policy(call_timeout_s=180.0),
    )

    residuals, st_res = _safe_array(psr, "residuals")
    if residuals is not None:
        residuals_us = residuals * 1.0e6
        st_res = "ok_seconds_to_us"
    else:
        residuals_us = np.array([], dtype=np.float64)

    errors, st_err = _safe_array(psr, "toaerrs")
    if errors is None:
        errors_us = np.ones_like(residuals_us)
        st_err = "fallback_unity"
    else:
        errors_us = errors

    roemer, st_roemer = _safe_array(psr, "roemer")
    shapiro, st_shapiro = _safe_array(psr, "shapiro_sun")
    ssbfreqs, st_ssb = _safe_array(psr, "ssbfreqs")
    bat_corrs, st_bat = _safe_array(psr, "batCorrs")
    stoas, st_stoas = _safe_array(psr, "stoas")
    toas, st_toas = _safe_array(psr, "toas")

    term_status = {
        "residuals_us": st_res,
        "errors_us": st_err,
        "roemer_sec": st_roemer,
        "shapiro_sun_sec": st_shapiro,
        "ssbfreqs_mhz": st_ssb,
        "bat_corrs_sec": st_bat,
        "stoas_mjd": st_stoas,
        "toas_mjd": st_toas,
        "bbat_mjd": "libstempo_toas_only",
        "phase_turns": "unavailable",
        "nphase": "unavailable",
        "phase_offset_turns": "unavailable",
        "pulse_number": "unavailable",
    }

    if conv.oracle_terms == "tempo2_general2_plugin":
        term_status["oracle_note"] = (
            "general2_plugin not wired in Phase A; libstempo properties only"
        )

    bbat_mjd = None
    pulse_number = None
    try:
        from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle

        pt = load_pytempo_native_oracle(par, tim)
        if "bbat_mjd" in pt.fields:
            bbat_mjd = pt.fields["bbat_mjd"]
            term_status["bbat_mjd"] = "pytempo_tier1"
        if "pulse_number" in pt.fields:
            pulse_number = pt.fields["pulse_number"]
            term_status["pulse_number"] = "pytempo_tier1"
    except Exception:
        pass

    return Tempo2TermDiagnostics(
        residuals_us=residuals_us,
        errors_us=errors_us,
        roemer_sec=roemer,
        shapiro_sun_sec=shapiro,
        ssbfreqs_mhz=ssbfreqs,
        bat_corrs_sec=bat_corrs,
        stoas_mjd=stoas,
        toas_mjd=toas,
        bbat_mjd=bbat_mjd,
        pulse_number=pulse_number,
        term_status=term_status,
        conventions=conv,
    )
