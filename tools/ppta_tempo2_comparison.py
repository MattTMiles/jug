#!/usr/bin/env python3
"""
PPTA DR4 JUG vs Tempo2 parity comparison.

Usage:
    JAX_PLATFORMS=cpu python3 tools/ppta_tempo2_comparison.py [--postfit] [--out results.csv]

Compares prefit (and optionally postfit) weighted RMS and parameter values
between JUG and Tempo2 for the 37 PPTA DR4 pulsars.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

PPTA_DIR = Path(__file__).parent.parent / "data" / "pulsars" / "PPTA_data" / \
    "ppta_dr4-data_dev-data-partim-MTM" / "data" / "partim" / "MTM"

# Parameters to compare postfit values for
COMPARE_PARAMS = [
    "F0", "F1", "F2", "DM", "DM1", "DM2", "PX",
    "PB", "A1", "ECC", "OM", "E", "EPS1", "EPS2",
    "SINI", "M2", "PBDOT", "OMDOT", "GAMMA", "H3", "STIG", "H4",
    "T0", "TASC", "XDOT", "A1DOT",
    "ELONG", "ELAT", "PMELONG", "PMELAT", "RAJ", "DECJ", "PMRA", "PMDEC",
    "FD1", "FD2", "FD3", "FD4",
    "FB0", "FB1", "FB2", "FB3", "FB4", "FB5",
]

# Keywords that are never fit parameters
_SKIP_KEYWORDS = {
    "PEPOCH", "POSEPOCH", "DMEPOCH", "BINARY", "UNITS", "EPHEM", "CLK",
    "START", "FINISH", "TZRMJD", "TZRSITE", "TZRFRQ", "MODE", "NTOA",
    "CHI2", "CHI2R", "TRES", "INFO", "TIMEEPH", "T2CMETHOD", "CORRECT_TROPOSPHERE",
    "PLANET_SHAPIRO", "DILATEFREQ", "NE_SW_IFUNC_NHARMS", "CLOCK",
    "DMDATA", "SWM", "NITS", "TRACK", "NE_SW", "EPHVER",
}


def extract_free_params(par_path):
    """Extract parameters with fit flag=1 from a par file."""
    free = []
    jump_idx = 0
    with open(par_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("C "):
                continue
            parts = line.split()
            key = parts[0].upper()

            if key == "JUMP":
                jump_idx += 1
                for i in range(3, len(parts)):
                    try:
                        float(parts[i])
                        if i + 1 < len(parts):
                            try:
                                flag = int(parts[i + 1])
                                if flag == 1:
                                    free.append(f"JUMP{jump_idx}")
                                break
                            except ValueError:
                                continue
                        break
                    except ValueError:
                        continue
                continue

            if key in _SKIP_KEYWORDS:
                continue
            if key.startswith("DMX"):
                continue

            if len(parts) >= 3:
                try:
                    flag = int(parts[2])
                    if flag == 1:
                        from jug.model.parameter_spec import canonicalize_param_name
                        free.append(canonicalize_param_name(key))
                except ValueError:
                    pass
    return free


# ── Tempo2 ────────────────────────────────────────────────────────────────────

def _run_tempo2(par, tim, args, timeout=120):
    """Run tempo2 and return stdout, suppressing warnings."""
    cmd = ["tempo2"] + args + ["-f", str(par), str(tim)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr


def _parse_general2_residuals(stdout):
    """Parse tempo2 general2 output: lines of '{residual} {error}'."""
    data = []
    for line in stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                res = float(parts[0])
                err = float(parts[1])
                data.append((res, err))
            except ValueError:
                continue
    return np.array(data) if data else np.empty((0, 2))


def _compute_wrms(res_sec, err_us):
    """Compute weighted RMS from residuals (seconds) and errors (microseconds)."""
    res_us = res_sec * 1e6
    w = 1.0 / err_us**2
    wmean = np.sum(w * res_us) / np.sum(w)
    wrms = np.sqrt(np.sum(w * (res_us - wmean)**2) / np.sum(w))
    return wrms


def _parse_tempo2_par(par_path):
    """Parse a tempo2 par file into a dict of parameter values."""
    params = {}
    with open(par_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("C "):
                continue
            parts = line.split()
            key = parts[0].upper()
            if key in ("JUMP", "MODE"):
                continue
            if len(parts) >= 2:
                try:
                    # Handle Fortran D notation
                    val_str = parts[1].replace('D', 'E').replace('d', 'e')
                    params[key] = float(val_str)
                except ValueError:
                    params[key] = parts[1]
    return params


def tempo2_prefit(par, tim):
    """Get tempo2 prefit residuals."""
    t0 = time.time()
    stdout, _ = _run_tempo2(par, tim,
        ["-output", "general2", "-s", "{pre} {err}\n", "-nofit"])
    data = _parse_general2_residuals(stdout)
    if len(data) == 0:
        raise RuntimeError("No residuals from tempo2")
    wrms = _compute_wrms(data[:, 0], data[:, 1])
    elapsed = time.time() - t0
    return len(data), wrms, elapsed


def tempo2_postfit(par, tim):
    """Get tempo2 postfit residuals and parameters."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy par and tim to temp dir so new.par doesn't pollute data dir
        tmp_par = Path(tmpdir) / par.name
        tmp_tim = Path(tmpdir) / tim.name
        shutil.copy2(par, tmp_par)
        shutil.copy2(tim, tmp_tim)

        # Run tempo2 with fit + output residuals
        stdout, _ = _run_tempo2(tmp_par, tmp_tim,
            ["-output", "general2", "-s", "{post} {err}\n", "-newpar"])
        data = _parse_general2_residuals(stdout)
        if len(data) == 0:
            raise RuntimeError("No postfit residuals from tempo2")
        wrms = _compute_wrms(data[:, 0], data[:, 1])

        # Parse postfit parameters from new.par
        new_par = Path(tmpdir) / "new.par"
        params = {}
        if new_par.exists():
            t2_params = _parse_tempo2_par(new_par)
            # Tempo2 outputs in the original timescale (TCB or TDB)
            # Convert TCB params to TDB for comparison with JUG
            units = str(t2_params.get("UNITS", "TDB")).upper()
            if units == "TCB":
                from jug.utils.timescales import (
                    scale_parameter_tcb_to_tdb, convert_tcb_epoch_to_tdb,
                    SCALED_PARAMETERS, EPOCH_PARAMETERS
                )
                scale_map = {name: dim for name, dim in SCALED_PARAMETERS}
                for p in COMPARE_PARAMS:
                    if p in t2_params and isinstance(t2_params[p], float):
                        val = t2_params[p]
                        if p in EPOCH_PARAMETERS:
                            val = float(convert_tcb_epoch_to_tdb(
                                np.longdouble(val)))
                        elif p in scale_map and scale_map[p] != 0:
                            val = scale_parameter_tcb_to_tdb(val, scale_map[p])
                        params[p] = val
            else:
                for p in COMPARE_PARAMS:
                    if p in t2_params and isinstance(t2_params[p], float):
                        params[p] = t2_params[p]

    elapsed = time.time() - t0
    return wrms, params, elapsed


# ── JUG ───────────────────────────────────────────────────────────────────────

def jug_prefit(par, tim):
    from jug.engine import open_session
    t0 = time.time()
    s = open_session(str(par), str(tim), verbose=False)
    r = s.compute_residuals(subtract_tzr=True)
    res = r["residuals_us"]
    errs = r["errors_us"]
    w = 1.0 / errs**2
    wmean = np.sum(w * res) / np.sum(w)
    wsdev = np.sqrt(np.sum(w * (res - wmean)**2) / np.sum(w))
    elapsed = time.time() - t0
    return r["n_toas"], wsdev, elapsed


def jug_postfit(par, tim):
    from jug.engine import open_session
    from jug.engine.noise_mode import NoiseConfig
    t0 = time.time()
    s = open_session(str(par), str(tim), verbose=False)
    free_params = extract_free_params(par)
    nc = NoiseConfig(enabled={
        "EFAC": False, "EQUAD": False,
        "ECORR": False, "RedNoise": False, "DMNoise": False, "DMX": True,
    })
    result = s.fit_parameters(fit_params=free_params, max_iter=1,
                              noise_config=nc, verbose=False)
    elapsed = time.time() - t0
    res = np.array(result["residuals_us"])
    errs = np.array(result["errors_us"])
    w = 1.0 / errs**2
    wmean = np.sum(w * res) / np.sum(w)
    wrms = np.sqrt(np.sum(w * (res - wmean)**2) / np.sum(w))
    params = {k: float(v) for k, v in result.get("final_params", {}).items()
              if k in COMPARE_PARAMS}
    return wrms, params, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def compare_one(par, tim, do_postfit):
    name = par.stem
    row = {
        "name": name, "par": par.name, "status": "OK",
        "n_toas": None,
        "t2_prefit_us": None, "jug_prefit_us": None, "prefit_ratio": None,
        "t2_postfit_us": None, "jug_postfit_us": None, "postfit_ratio": None,
        "param_diffs": "",
        "t2_prefit_time_s": None, "jug_prefit_time_s": None,
        "t2_postfit_time_s": None, "jug_postfit_time_s": None,
    }

    # Prefit - Tempo2
    try:
        n, t2_pre, t2_pre_t = tempo2_prefit(par, tim)
        row["n_toas"] = n
        row["t2_prefit_us"] = round(t2_pre, 4)
        row["t2_prefit_time_s"] = round(t2_pre_t, 1)
    except Exception:
        row["status"] = f"T2_PREFIT_ERR: {traceback.format_exc().splitlines()[-1]}"
        return row

    # Prefit - JUG
    try:
        n_jug, jug_pre, jug_pre_t = jug_prefit(par, tim)
        row["jug_prefit_us"] = round(jug_pre, 4)
        row["jug_prefit_time_s"] = round(jug_pre_t, 1)
    except Exception:
        row["status"] = f"JUG_PREFIT_ERR: {traceback.format_exc().splitlines()[-1]}"
        return row

    if row["t2_prefit_us"] and row["jug_prefit_us"]:
        ratio = row["jug_prefit_us"] / row["t2_prefit_us"]
        row["prefit_ratio"] = round(ratio, 4)

    if not do_postfit:
        return row

    # Postfit - Tempo2
    try:
        t2_post, t2_params, t2_post_t = tempo2_postfit(par, tim)
        row["t2_postfit_us"] = round(t2_post, 4)
        row["t2_postfit_time_s"] = round(t2_post_t, 1)
    except Exception:
        row["status"] = f"T2_POSTFIT_ERR: {traceback.format_exc().splitlines()[-1]}"
        return row

    # Postfit - JUG
    try:
        jug_post, jug_params, jug_post_t = jug_postfit(par, tim)
        row["jug_postfit_us"] = round(jug_post, 4)
        row["jug_postfit_time_s"] = round(jug_post_t, 1)
    except Exception:
        row["status"] = f"JUG_POSTFIT_ERR: {traceback.format_exc().splitlines()[-1]}"
        return row

    if row["t2_postfit_us"] and row["jug_postfit_us"]:
        ratio = row["jug_postfit_us"] / row["t2_postfit_us"]
        row["postfit_ratio"] = round(ratio, 4)

    # Compare parameters
    diffs = []
    for p in COMPARE_PARAMS:
        if p in jug_params and p in t2_params:
            jv = jug_params[p]
            tv = t2_params[p]
            if tv != 0:
                rel = (jv - tv) / abs(tv)
                if abs(rel) > 0.005:  # >0.5%
                    diffs.append(f"{p}:{rel:+.3e}({abs(rel)*100:.2f}%)")
    row["param_diffs"] = " ".join(diffs)

    return row


def main():
    parser = argparse.ArgumentParser(description="PPTA JUG vs Tempo2 parity")
    parser.add_argument("--postfit", action="store_true",
                        help="Also compare postfit (runs fits)")
    parser.add_argument("--out", default="ppta_tempo2_parity.csv",
                        help="Output CSV path")
    parser.add_argument("--pulsar", default=None,
                        help="Run only this pulsar (e.g. J1909-3744)")
    args = parser.parse_args()

    par_files = sorted(PPTA_DIR.glob("*.par"))
    if args.pulsar:
        par_files = [p for p in par_files if args.pulsar in p.stem]

    if not par_files:
        print(f"No par files found in {PPTA_DIR}")
        sys.exit(1)

    print(f"Running JUG vs Tempo2 comparison for {len(par_files)} pulsars")
    if args.postfit:
        print("  (including postfit)")
    print()

    rows = []
    for par in par_files:
        tim = par.with_suffix(".tim")
        if not tim.exists():
            print(f"  SKIP {par.stem}: no .tim file")
            continue

        t0 = time.time()
        try:
            row = compare_one(par, tim, args.postfit)
        except Exception:
            row = {"name": par.stem, "par": par.name,
                   "status": f"ERROR: {traceback.format_exc().splitlines()[-1]}"}
        elapsed = time.time() - t0

        # Print summary
        status_parts = []
        if row.get("prefit_ratio"):
            r = row["prefit_ratio"]
            tag = "GOOD" if 0.95 <= r <= 1.05 else "MOD" if 0.8 <= r <= 1.2 else "BAD"
            status_parts.append(f"pre: T2={row['t2_prefit_us']:.4f} JUG={row['jug_prefit_us']:.4f} "
                              f"ratio={r:.3f} [{tag}]")
        if row.get("postfit_ratio"):
            r = row["postfit_ratio"]
            tag = "GOOD" if 0.95 <= r <= 1.05 else "MOD" if 0.8 <= r <= 1.2 else "BAD"
            status_parts.append(f"post: T2={row['t2_postfit_us']:.4f} JUG={row['jug_postfit_us']:.4f} "
                              f"ratio={r:.3f} [{tag}]")
        if row.get("param_diffs"):
            status_parts.append(f"  params: {row['param_diffs'][:80]}")

        summary = " | ".join(status_parts) if status_parts else row.get("status", "?")
        print(f"  {par.stem:30s} {summary}  ({elapsed:.1f}s)")

        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = [
            "name", "par", "status", "n_toas",
            "t2_prefit_us", "jug_prefit_us", "prefit_ratio",
            "t2_postfit_us", "jug_postfit_us", "postfit_ratio",
            "param_diffs",
            "t2_prefit_time_s", "jug_prefit_time_s",
            "t2_postfit_time_s", "jug_postfit_time_s",
        ]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to: {args.out}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for phase, prefix in [("Prefit", "prefit"), ("Postfit", "postfit")]:
        ratios = [r[f"{prefix}_ratio"] for r in rows
                  if r.get(f"{prefix}_ratio") is not None]
        if not ratios:
            continue
        good = sum(1 for r in ratios if 0.95 <= r <= 1.05)
        mod = sum(1 for r in ratios if (0.8 <= r < 0.95) or (1.05 < r <= 1.2))
        bad = sum(1 for r in ratios if r < 0.8 or r > 1.2)
        print(f"\n{phase}: {len(ratios)}/{len(rows)} computed")
        print(f"  GOOD (<5%): {good}  MODERATE (5-20%): {mod}  BAD (>20%): {bad}")
        print(f"  Median ratio: {np.median(ratios):.4f}  "
              f"Max: {max(ratios):.4f}  Min: {min(ratios):.4f}")
        if bad + mod > 0:
            print(f"\n  PROBLEM PULSARS ({prefix} ratio outside [0.95, 1.05]):")
            for r in rows:
                rv = r.get(f"{prefix}_ratio")
                if rv is not None and (rv < 0.95 or rv > 1.05):
                    extra = f"  {r.get('param_diffs', '')[:60]}" if r.get('param_diffs') else ""
                    print(f"    {r['name']:30s} ratio={rv:.3f}{extra}")

    # Speed
    for tool in ["t2", "jug"]:
        for phase in ["prefit", "postfit"]:
            times = [r[f"{tool}_{phase}_time_s"] for r in rows
                     if r.get(f"{tool}_{phase}_time_s") is not None]
            if times:
                label = f"{'Tempo2' if tool == 't2' else 'JUG'} {phase}"
                print(f"\n  {label}: median={np.median(times):.1f}s  "
                      f"mean={np.mean(times):.1f}s  max={max(times):.1f}s")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
