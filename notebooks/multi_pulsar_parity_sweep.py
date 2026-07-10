#!/usr/bin/env python
"""Sweep NG15 pulsars for JUG/PINT residual and delay-component parity.

Writes a CSV summary. Intended as broad triage; keep the main notebook for
single-pulsar detail.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/mattm/soft/JUG")
DEFAULT_PARTIM = ROOT / "data/pulsars/NG_data/NG_15yr_partim"
DEFAULT_JUG_CLOCK = ROOT / "data/clock"
DEFAULT_PINT_CLOCK = Path("/tmp/jug_pint_clock_override")


def _prepare_path():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ["JAX_PLATFORMS"] = "cpu"


def _first_float_token(line: str):
    parts = line.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _read_two_column_clock(path: Path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            pass
    return rows


def _write_pint_tempo_clock_from_two_column(src: Path, dst: Path):
    lines = [
        "# Synthetic PINT TEMPO clock generated from JUG ao2gps.clk\n",
        "   MJD       AO-REF      GPS-REF NS      DATE    COMMENTS\n",
        "=========    ========    ======== ==    ========  ========\n",
    ]
    for mjd, offset_sec in _read_two_column_clock(src):
        lines.append(f"{mjd:9.2f}{0.0:12.3f}{offset_sec * 1e6:12.3f} 3 f  synthetic from ao2gps.clk\n")
    dst.write_text("".join(lines))


def prepare_pint_clock_override(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
            continue
        if not src.is_file():
            continue
        lines = src.read_text(errors="ignore").splitlines(keepends=True)
        numeric, nonnumeric = [], []
        for i, line in enumerate(lines):
            mjd = _first_float_token(line)
            if mjd is None:
                nonnumeric.append(line)
            else:
                numeric.append((mjd, i, line))
        if any(numeric[i + 1][0] < numeric[i][0] for i in range(len(numeric) - 1)):
            numeric.sort(key=lambda x: (x[0], x[1]))
            dst.write_text("".join(nonnumeric + [line for _, _, line in numeric]))
        else:
            dst.symlink_to(src)
    ao2gps = src_dir / "ao2gps.clk"
    if ao2gps.exists():
        time_ao = dst_dir / "time_ao.dat"
        if time_ao.exists() or time_ao.is_symlink():
            time_ao.unlink()
        _write_pint_tempo_clock_from_two_column(ao2gps, time_ao)
    os.environ["PINT_CLOCK_OVERRIDE"] = str(dst_dir)


def _par_uses_dilatefreq(par_path: Path):
    for line in par_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "DILATEFREQ":
            return parts[1].upper() in {"Y", "T", "1", "TRUE"}
    return False


def _enable_pint_dilatefreq(model, par_path: Path):
    if not _par_uses_dilatefreq(par_path):
        return model
    from jug.delays.barycentric import compute_einstein_rate

    base = model.barycentric_radio_freq

    def barycentric_radio_freq_dilated(toas):
        freq = base(toas)
        rate = compute_einstein_rate(np.asarray(toas.table["tdbld"], dtype=float), units="TDB")
        return freq / rate

    model.barycentric_radio_freq = barycentric_radio_freq_dilated
    return model


def _pcomp(model, toas, component, method):
    comp = model.components.get(component)
    if comp is None:
        return np.zeros(toas.ntoas, dtype=float)
    try:
        return getattr(comp, method)(toas).to("s").value
    except Exception:
        return np.zeros(toas.ntoas, dtype=float)


def _component_deltas_us(jug_session, pint_model, pint_toas):
    jc = jug_session.compute_residuals(subtract_tzr=True, force_recompute=True)
    pint_geo_shap = (
        _pcomp(pint_model, pint_toas, "AstrometryEquatorial", "solar_system_geometric_delay")
        + _pcomp(pint_model, pint_toas, "AstrometryEcliptic", "solar_system_geometric_delay")
        + _pcomp(pint_model, pint_toas, "SolarSystemShapiro", "solar_system_shapiro_delay")
    )
    pint_dm = _pcomp(pint_model, pint_toas, "DispersionDM", "constant_dispersion_delay")
    pint_dmx = _pcomp(pint_model, pint_toas, "DispersionDMX", "DMX_dispersion_delay")
    pint_sw = _pcomp(pint_model, pint_toas, "SolarWindDispersion", "solar_wind_delay")
    pint_tropo = _pcomp(pint_model, pint_toas, "TroposphereDelay", "troposphere_delay")
    pint_total = pint_model.delay(pint_toas).to("s").value
    pint_prebinary = pint_geo_shap + pint_dm + pint_dmx + pint_sw + pint_tropo

    jug_total = np.asarray(jc["total_delay_sec"], dtype=float)
    jug_prebinary = np.asarray(jc["prebinary_delay_sec"], dtype=float)
    jug_dm = np.asarray(jc["dm_delay_sec"], dtype=float)
    jug_dmx = np.asarray(jc.get("dmx_delay_sec", np.zeros_like(jug_dm)), dtype=float)
    jug_sw = np.asarray(jc.get("sw_delay_sec", np.zeros_like(jug_dm)), dtype=float)
    jug_tropo = np.asarray(jc.get("tropo_delay_sec", np.zeros_like(jug_dm)), dtype=float)
    jug_geo_shap = np.asarray(jc["roemer_shapiro_sec"], dtype=float)

    return {
        "total_delay": (jug_total - pint_total) * 1e6,
        "geo+shapiro": (jug_geo_shap - pint_geo_shap) * 1e6,
        "DM": (jug_dm - pint_dm) * 1e6,
        "DMX": (jug_dmx - pint_dmx) * 1e6,
        "solar_wind": (jug_sw - pint_sw) * 1e6,
        "tropo": (jug_tropo - pint_tropo) * 1e6,
        "prebinary": (jug_prebinary - pint_prebinary) * 1e6,
        "post_prebinary": ((jug_total - jug_prebinary) - (pint_total - pint_prebinary)) * 1e6,
    }


def _rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def _find_tim_for_par(par_path: Path):
    stem = par_path.name[:-4]
    candidates = [
        par_path.with_suffix(".tim"),
        par_path.parent / f"{stem}.tim",
        par_path.parent / f"{stem.replace('_PINT_', '_')}.tim",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    hits = sorted(par_path.parent.glob(stem.split(".nb")[0] + "*.tim"))
    return hits[0] if hits else None


def run_one(par_path: Path, tim_path: Path, maxiter: int, clock_dir: Path):
    import pint.fitter
    import pint.logging
    import pint.models
    import pint.residuals
    import pint.toa
    from jug.engine.session import TimingSession

    pint.logging.setup(level="ERROR")

    pint_model = pint.models.get_model(str(par_path), allow_T2=True, allow_tcb=True)
    _enable_pint_dilatefreq(pint_model, par_path)
    pint_toas = pint.toa.get_TOAs(str(tim_path), model=pint_model)
    pint_model.find_empty_masks(pint_toas, freeze=True)
    pint_prefit_us = pint.residuals.Residuals(pint_toas, pint_model).time_resids.to("s").value * 1e6
    fitter = pint.fitter.GLSFitter(pint_toas, pint_model)
    fitter.fit_toas(maxiter=maxiter)
    pint_post_us = fitter.resids.calc_time_resids().to("s").value * 1e6

    prefit_session = TimingSession(str(par_path), str(tim_path), clock_dir=str(clock_dir))
    session = TimingSession(str(par_path), str(tim_path), clock_dir=str(clock_dir))
    jug_result = session.fit_parameters(max_iter=maxiter)
    jug_prefit_us = np.asarray(jug_result["residuals_prefit_us"], dtype=float)
    jug_post_us = np.asarray(jug_result["residuals_us"], dtype=float)

    diff_prefit = jug_prefit_us - pint_prefit_us
    diff_post = jug_post_us - pint_post_us
    deltas = _component_deltas_us(session, fitter.model, pint_toas)
    top_component, top_std = max(((k, np.std(v)) for k, v in deltas.items()), key=lambda kv: kv[1])

    return {
        "pulsar": par_path.name.split("_PINT_")[0],
        "par": str(par_path),
        "tim": str(tim_path),
        "n_toa": len(jug_post_us),
        "prefit_diff_rms_ps": _rms(diff_prefit) * 1e6,
        "postfit_diff_rms_ps": _rms(diff_post) * 1e6,
        "postfit_diff_max_ps": float(np.max(np.abs(diff_post)) * 1e6),
        "top_component": top_component,
        "top_component_std_ps": float(top_std * 1e6),
        **{f"{k}_std_ps": float(np.std(v) * 1e6) for k, v in deltas.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partim", type=Path, default=DEFAULT_PARTIM)
    parser.add_argument("--clock-dir", type=Path, default=DEFAULT_JUG_CLOCK)
    parser.add_argument("--pint-clock-dir", type=Path, default=DEFAULT_PINT_CLOCK)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", type=str, default=None, help="Substring filter for par filename")
    parser.add_argument("--exclude", action="append", default=[], help="Substring filter to exclude; may be repeated")
    parser.add_argument("--out", type=Path, default=Path("notebooks/multi_pulsar_parity_sweep.csv"))
    args = parser.parse_args()

    _prepare_path()
    prepare_pint_clock_override(args.clock_dir, args.pint_clock_dir)

    par_files = sorted(args.partim.glob("*.par"))
    if args.only:
        par_files = [p for p in par_files if args.only in p.name]
    for pattern in args.exclude:
        par_files = [p for p in par_files if pattern not in p.name]
    if args.limit is not None:
        par_files = par_files[: args.limit]

    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, par_path in enumerate(par_files, 1):
        tim_path = _find_tim_for_par(par_path)
        if tim_path is None:
            print(f"[{i}/{len(par_files)}] {par_path.name}: no tim")
            continue
        try:
            print(f"[{i}/{len(par_files)}] {par_path.name}")
            rows.append(run_one(par_path, tim_path, args.maxiter, args.clock_dir))
        except Exception as exc:
            rows.append({"pulsar": par_path.name, "par": str(par_path), "tim": str(tim_path), "error": repr(exc)})
            print(f"  ERROR {type(exc).__name__}: {exc}")
        pd.DataFrame(rows).to_csv(args.out, index=False)
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    if not df.empty and "postfit_diff_rms_ps" in df:
        print(df.sort_values("postfit_diff_rms_ps", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

