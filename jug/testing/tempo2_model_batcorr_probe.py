"""Step 13 model-epoch batCorr/bbat diagnostic (in-repo replacement for /tmp probe)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jug.io.par_reader import parse_par_file
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2_spin import compute_tempo2_bbat_mjd
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from jug.testing.tempo2_reference import tempo2_reference
from jug.utils.constants import SECS_PER_DAY


@dataclass
class ModelBatcorrReport:
    fixture_id: str
    model_batcorr_vs_lib_rms_ns: float
    model_bat_vs_lib_rms_ns: float
    bundled_bat_vs_lib_rms_sec: float
    model_bbat_vs_oracle_rms_ns: float
    shklovskii_max_us: float
    dt_ld_vs_bbat_identity_rms_ns: float
    spin_model_bbat_rms_ns: float
    spin_production_rms_ns: float
    oracle_bbat_vs_pt_rms_ns: float
    spin_pt_bbat_rms_ns: float


def compare_model_batcorr_diagnostic(
    par: str | Path,
    tim: str | Path,
    *,
    fixture_id: str = "",
) -> ModelBatcorrReport:
    par_path = Path(par)
    tim_path = Path(tim)
    jug = compute_residuals_simple(par_path, tim_path, verbose=False, compatibility="tempo2")
    ref = tempo2_reference(par_path, tim_path, include_batcorr=True)
    td = jug["term_diagnostics"]
    params = parse_par_file(par_path)
    pepoch = float(params["PEPOCH"])

    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    model = np.asarray(jug["model_mjd"], dtype=np.float64)
    prebin = np.asarray(td["prebinary_delay_sec"], dtype=np.float64)
    model_batcorr = (model - sat) - prebin / SECS_PER_DAY
    model_bat = sat + model_batcorr
    model_bbat = compute_tempo2_bbat_mjd(model, prebin)
    oracle_bbat_no_shk = model - prebin / SECS_PER_DAY

    lib_batcorr = np.asarray(ref.bat_corr_days, dtype=np.float64)
    bundled_bat = np.asarray(td.get("bat_mjd", sat), dtype=np.float64)
    shk = np.asarray(td.get("shklovskii_sec", 0.0), dtype=np.float64)

    resid_prod = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    resid_prod = resid_prod - np.mean(resid_prod)

    oracle_bbat_vs_pt = float("nan")
    spin_pt_bbat = float("nan")
    dt_ld_vs_bbat = float("nan")
    try:
        pt = load_pytempo_native_oracle(par_path, tim_path, fixture_id=fixture_id)
        pt_bbat = pt.fields["bbat_mjd"]
        pt_torb = pt.fields["torb_sec"]
        oracle_bbat_vs_pt = float(
            np.sqrt(np.mean((oracle_bbat_no_shk - pt_bbat) ** 2)) * SECS_PER_DAY * 1e9
        )
        dt_ld = np.asarray(jug["dt_sec_ld"], dtype=np.float64)
        dt_from_bbat = (model_bbat - pepoch) * SECS_PER_DAY + pt_torb
        dt_ld_vs_bbat = float(np.sqrt(np.mean((dt_ld - dt_from_bbat) ** 2)) * 1e9)
    except Exception:
        pass

    return ModelBatcorrReport(
        fixture_id=fixture_id or par_path.stem,
        model_batcorr_vs_lib_rms_ns=float(
            np.sqrt(np.mean((model_batcorr - lib_batcorr) ** 2)) * SECS_PER_DAY * 1e9
        ),
        model_bat_vs_lib_rms_ns=float(
            np.sqrt(np.mean(((model_bat - (sat + lib_batcorr)) * SECS_PER_DAY) ** 2))
            * 1e9
        ),
        bundled_bat_vs_lib_rms_sec=float(
            np.sqrt(np.mean(((bundled_bat - (sat + lib_batcorr)) * SECS_PER_DAY) ** 2))
        ),
        model_bbat_vs_oracle_rms_ns=float(
            np.sqrt(np.mean((model_bbat - oracle_bbat_no_shk) ** 2)) * SECS_PER_DAY * 1e9
        ),
        shklovskii_max_us=float(np.max(np.abs(shk)) * 1e6),
        dt_ld_vs_bbat_identity_rms_ns=dt_ld_vs_bbat,
        spin_model_bbat_rms_ns=float("nan"),
        spin_production_rms_ns=float(np.sqrt(np.mean(resid_prod**2))),
        oracle_bbat_vs_pt_rms_ns=oracle_bbat_vs_pt,
        spin_pt_bbat_rms_ns=spin_pt_bbat,
    )
