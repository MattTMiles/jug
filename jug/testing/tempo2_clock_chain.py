"""formBats / getCorrectionTT clock-chain diagnostics (dev oracle only).

Compares JUG ``correction_tt`` / ``correction_tt_tb`` against libstempo
``batCorrs`` decomposed via tempo2 ``formBats.C`` signs and the production
``model_mjd`` epoch chain. See Steps 10–12 in ``TEMPO2_PARITY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.constants import SECS_PER_DAY
from jug.utils.timescales import IFTE_KM1, IFTE_MJD0, IFTE_TEPH0_SEC


def _sandbox_array(psr: Any, name: str) -> np.ndarray:
    if not hasattr(psr, name):
        raise RuntimeError(f"libstempo missing property {name!r}")
    value = getattr(psr, name)
    arr = value() if callable(value) else value
    return np.asarray(arr, dtype=np.float64)


@dataclass
class ClockChainFormbatsReport:
    """JUG vs libstempo formBats clock-chain decomposition on one fixture."""

    fixture_id: str
    n_toa: int
    production_rms_ns: float
    sat_rms_ns: float
    bat_corr_lib_vs_pt_rms_ns: float
    tt_mean_sec: float
    tt_tb_mean_sec: float
    tt_tb_implied_mean_sec: float
    tt_tb_gap_mean_sec: float
    utc_to_tdb_mean_sec: float
    formbats_canonical_mean_sec: float
    bat_corr_mean_sec: float
    formbats_offset_mean_sec: float
    jug_bat_vs_lib_rms_sec: float
    notes: list[str] = field(default_factory=list)


def compare_formbats_clock_chain(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    policy: Any | None = None,
) -> ClockChainFormbatsReport:
    """Decompose tempo2 ``formBats.C`` clock chain for JUG vs libstempo/pytempo."""
    from jug.testing.sandbox_tempo2 import Policy, tempopulsar
    from jug.testing.tempo2_reference import tempo2_reference
    from pytempo.sandbox import tempopulsar as pytempo_pulsar

    par_path = Path(par)
    tim_path = Path(tim)
    jug = compute_residuals_simple(par_path, tim_path, verbose=False, compatibility="tempo2")
    ref = tempo2_reference(par_path, tim_path)
    resid_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    resid_ns = resid_ns - np.mean(resid_ns)
    production_rms_ns = float(np.sqrt(np.mean(resid_ns**2)))

    td = jug["term_diagnostics"]
    tt = np.asarray(td["correction_tt_sec"], dtype=np.float64)
    tt_tb = np.asarray(td["correction_tt_tb_sec"], dtype=np.float64)
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    jug_bat = np.asarray(td["bat_mjd"], dtype=np.float64)
    tropo = np.asarray(td["tropo_delay_sec"], dtype=np.float64)
    dm = np.asarray(td["dm_delay_sec"], dtype=np.float64)
    sw = np.asarray(td["sw_delay_sec"], dtype=np.float64)
    utc_to_tdb = np.asarray(td["utc_to_tdb_sec"], dtype=np.float64)

    psr = tempopulsar(
        parfile=str(par_path),
        timfile=str(tim_path),
        dofit=False,
        policy=policy or Policy(call_timeout_s=180.0),
    )
    stoas = _sandbox_array(psr, "stoas")
    bat_corr_days = _sandbox_array(psr, "batCorrs")
    bat_corr_sec = bat_corr_days * SECS_PER_DAY

    pt = pytempo_pulsar(parfile=str(par_path), timfile=str(tim_path), dofit=False)
    pt_diag = pt.toa_diagnostics(removemean=False)
    pt_bat_corr_sec = np.asarray(pt_diag["bat_corr_days"], dtype=np.float64) * SECS_PER_DAY
    pt_roemer = np.asarray(pt_diag["roemer_sec"], dtype=np.float64)
    pt_shap = np.asarray(pt_diag["sun_shapiro_sec"], dtype=np.float64)

    delay_part = -tropo + pt_roemer - pt_shap - (dm + sw)
    formbats_canonical = tt + tt_tb + delay_part
    tt_tb_implied = bat_corr_sec - tt - delay_part

    sat_rms_ns = float(np.sqrt(np.mean(((sat - stoas) * SECS_PER_DAY) ** 2)) * 1e9)
    bat_corr_lib_vs_pt_rms_ns = float(
        np.sqrt(np.mean((bat_corr_sec - pt_bat_corr_sec) ** 2)) * 1e9
    )
    jug_bat_vs_lib_rms_sec = float(
        np.sqrt(np.mean(((jug_bat - (stoas + bat_corr_days)) * SECS_PER_DAY) ** 2))
    )

    tt_tb_gap = tt_tb_implied - tt_tb
    formbats_offset = bat_corr_sec - formbats_canonical

    notes = [
        "formBats canonical uses JUG tt/tt_tb + pytempo roemer/shap + JUG dm/sw/tropo.",
        "tt_tb_implied inverts libstempo batCorrs with canonical delay signs.",
        "Production spin uses IFTE(tdb_ld) model_mjd, not formBats bat/bbat.",
    ]
    return ClockChainFormbatsReport(
        fixture_id=fixture_id or par_path.stem,
        n_toa=int(tt.size),
        production_rms_ns=production_rms_ns,
        sat_rms_ns=sat_rms_ns,
        bat_corr_lib_vs_pt_rms_ns=bat_corr_lib_vs_pt_rms_ns,
        tt_mean_sec=float(np.mean(tt)),
        tt_tb_mean_sec=float(np.mean(tt_tb)),
        tt_tb_implied_mean_sec=float(np.mean(tt_tb_implied)),
        tt_tb_gap_mean_sec=float(np.mean(tt_tb_gap)),
        utc_to_tdb_mean_sec=float(np.mean(utc_to_tdb)),
        formbats_canonical_mean_sec=float(np.mean(formbats_canonical)),
        bat_corr_mean_sec=float(np.mean(bat_corr_sec)),
        formbats_offset_mean_sec=float(np.mean(formbats_offset)),
        jug_bat_vs_lib_rms_sec=jug_bat_vs_lib_rms_sec,
        notes=notes,
    )


@dataclass
class BatcorrEpochChainReport:
    """Step 12: libstempo ``batCorrs`` vs production ``model_mjd`` epoch chain."""

    fixture_id: str
    n_toa: int
    production_rms_ns: float
    batcorr_model_identity_rms_ns: float
    batcorr_utc_model_tdb_rms_ns: float
    model_tdb_vs_tt_tb_rms_ns: float
    model_tdb_vs_ifte_linear_rms_ns: float
    formbats_dm_sw_rms_ns: float
    formbats_tdis_implied_rms_ns: float
    utc_to_tdb_mean_sec: float
    model_tdb_mean_sec: float
    tt_tb_mean_sec: float
    notes: list[str] = field(default_factory=list)


def compare_batcorr_epoch_chain(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
    policy: Any | None = None,
) -> BatcorrEpochChainReport:
    """Step 12: relate libstempo ``batCorrs`` to JAX ``model_mjd`` / ``tdb_mjd`` chain."""
    from jug.testing.sandbox_tempo2 import Policy, tempopulsar
    from jug.testing.tempo2_reference import tempo2_reference
    from pytempo.sandbox import tempopulsar as pytempo_pulsar

    par_path = Path(par)
    tim_path = Path(tim)
    jug = compute_residuals_simple(par_path, tim_path, verbose=False, compatibility="tempo2")
    ref = tempo2_reference(par_path, tim_path)
    resid_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    resid_ns = resid_ns - np.mean(resid_ns)
    production_rms_ns = float(np.sqrt(np.mean(resid_ns**2)))

    td = jug["term_diagnostics"]
    tt = np.asarray(td["correction_tt_sec"], dtype=np.float64)
    tt_tb = np.asarray(td["correction_tt_tb_sec"], dtype=np.float64)
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    prebin = np.asarray(td["prebinary_delay_sec"], dtype=np.float64)
    tropo = np.asarray(td["tropo_delay_sec"], dtype=np.float64)
    dm = np.asarray(td["dm_delay_sec"], dtype=np.float64)
    sw = np.asarray(td["sw_delay_sec"], dtype=np.float64)
    utc_to_tdb = np.asarray(td["utc_to_tdb_sec"], dtype=np.float64)
    model = np.asarray(jug["model_mjd"], dtype=np.float64)
    tdb = np.asarray(jug["tdb_mjd"], dtype=np.float64)

    psr = tempopulsar(
        parfile=str(par_path),
        timfile=str(tim_path),
        dofit=False,
        policy=policy or Policy(call_timeout_s=180.0),
    )
    bat_corr_sec = _sandbox_array(psr, "batCorrs") * SECS_PER_DAY

    pt = pytempo_pulsar(parfile=str(par_path), timfile=str(tim_path), dofit=False)
    pt_diag = pt.toa_diagnostics(removemean=False)
    pt_roemer = np.asarray(pt_diag["roemer_sec"], dtype=np.float64)
    pt_shap = np.asarray(pt_diag["sun_shapiro_sec"], dtype=np.float64)

    model_tdb_sec = (model - tdb) * SECS_PER_DAY
    ifte_linear_sec = (
        float(IFTE_KM1) * (tdb - float(IFTE_MJD0)) * SECS_PER_DAY + float(IFTE_TEPH0_SEC)
    )

    batcorr_from_model = (model - sat) * SECS_PER_DAY - prebin
    batcorr_from_chain = utc_to_tdb + model_tdb_sec - prebin

    delay_nd = -tropo + pt_roemer - pt_shap
    formbats_dm_sw = tt + tt_tb + delay_nd - (dm + sw)
    tdis_implied = tt + tt_tb + delay_nd - bat_corr_sec

    def _rms_ns(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)) * 1e9)

    notes = [
        "libstempo batCorrs = (model_mjd - sat)*86400 - prebinary_delay_sec (Step 12).",
        "Step 11 TT_TB gap used dm+sw as tdis — confounded; tt2tdb correction_tt_tb is ~14 s.",
        "Naive formBats with dm+sw misses utc_to_tdb (~65 s); use production epoch chain.",
    ]
    return BatcorrEpochChainReport(
        fixture_id=fixture_id or par_path.stem,
        n_toa=int(tt.size),
        production_rms_ns=production_rms_ns,
        batcorr_model_identity_rms_ns=_rms_ns(bat_corr_sec - batcorr_from_model),
        batcorr_utc_model_tdb_rms_ns=_rms_ns(bat_corr_sec - batcorr_from_chain),
        model_tdb_vs_tt_tb_rms_ns=_rms_ns(model_tdb_sec - tt_tb),
        model_tdb_vs_ifte_linear_rms_ns=_rms_ns(model_tdb_sec - ifte_linear_sec),
        formbats_dm_sw_rms_ns=_rms_ns(formbats_dm_sw - bat_corr_sec),
        formbats_tdis_implied_rms_ns=_rms_ns(tt + tt_tb + delay_nd - tdis_implied - bat_corr_sec),
        utc_to_tdb_mean_sec=float(np.mean(utc_to_tdb)),
        model_tdb_mean_sec=float(np.mean(model_tdb_sec)),
        tt_tb_mean_sec=float(np.mean(tt_tb)),
        notes=notes,
    )


def format_batcorr_epoch_chain_report(report: BatcorrEpochChainReport) -> str:
    """Human-readable Step 12 batCorrs / model_mjd chain report."""
    lines = [
        f"fixture: {report.fixture_id}",
        f"n_toa: {report.n_toa}",
        f"production JUG−libstempo RMS: {report.production_rms_ns:.2f} ns",
        "",
        "=== libstempo batCorrs identities (RMS ns) ===",
        f"(model_mjd - sat)*86400 - prebinary:  {report.batcorr_model_identity_rms_ns:.3f}",
        f"utc_to_tdb + (model-tdb) - prebinary: {report.batcorr_utc_model_tdb_rms_ns:.3f}",
        "",
        "=== TDB → TCB model epoch (RMS ns) ===",
        f"(model_mjd - tdb_mjd) vs tt2tdb tt_tb:     {report.model_tdb_vs_tt_tb_rms_ns:.3f}",
        f"(model_mjd - tdb_mjd) vs IFTE linear+Teph0: {report.model_tdb_vs_ifte_linear_rms_ns:.3f}",
        f"means: utc_to_tdb={report.utc_to_tdb_mean_sec:+.3f}s  "
        f"model-tdb={report.model_tdb_mean_sec:+.3f}s  tt_tb={report.tt_tb_mean_sec:+.3f}s",
        "",
        "=== formBats.C naive split (RMS ns) ===",
        f"tt+tt_tb+delays - (dm+sw):               {report.formbats_dm_sw_rms_ns:.3f}",
        f"tt+tt_tb+delays - tdis_implied:          {report.formbats_tdis_implied_rms_ns:.3f}",
        "",
        "=== interpretation ===",
        "batCorrs follows production IFTE/JAX model_mjd, not isolated tt2tdb+dm+sw.",
        "Fix diagnostic formBats from (model_mjd - sat) - prebinary/86400.",
    ]
    if report.notes:
        lines.extend(["", *report.notes])
    return "\n".join(lines)


def format_clock_chain_report(report: ClockChainFormbatsReport) -> str:
    """Human-readable Step 11 clock-chain report."""
    lines = [
        f"fixture: {report.fixture_id}",
        f"n_toa: {report.n_toa}",
        f"production JUG−libstempo RMS: {report.production_rms_ns:.2f} ns",
        "",
        "=== closed links ===",
        f"sat vs libstempo stoas RMS: {report.sat_rms_ns:.3f} ns",
        f"libstempo batCorrs vs pytempo bat_corr RMS: {report.bat_corr_lib_vs_pt_rms_ns:.3f} ns",
        "",
        "=== clock means (sec) ===",
        f"JUG getCorrectionTT (tt):     {report.tt_mean_sec:+.6f}",
        f"JUG correctionTT_TB:          {report.tt_tb_mean_sec:+.6f}",
        f"implied TT+TT_TB from batCorrs: {report.tt_tb_implied_mean_sec:+.6f}",
        f"TT_TB gap (implied − JUG):      {report.tt_tb_gap_mean_sec:+.6f}",
        f"JUG utc_to_tdb:                 {report.utc_to_tdb_mean_sec:+.6f}",
        "",
        "=== formBats batCorr ===",
        f"canonical mean:               {report.formbats_canonical_mean_sec:+.6f}",
        f"libstempo batCorrs mean:        {report.bat_corr_mean_sec:+.6f}",
        f"offset (batCorrs − canonical):  {report.formbats_offset_mean_sec:+.6f}",
        f"JUG formBats bat vs lib bat RMS: {report.jug_bat_vs_lib_rms_sec:.3f} s",
        "",
        "=== interpretation (Step 11; see Step 12 correction) ===",
        "TT_TB implied gap used dm+sw as tdis — confounded by ~65 s utc_to_tdb slot.",
        "Step 12: batCorrs = (model_mjd-sat)*86400-prebinary closes at ~286 ns.",
    ]
    if report.notes:
        lines.extend(["", *report.notes])
    return "\n".join(lines)
