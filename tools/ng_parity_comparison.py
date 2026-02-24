#!/usr/bin/env python3
"""
NANOGrav 15yr PINT vs JUG parity comparison.

Usage:
    JAX_PLATFORMS=cpu python3 tools/ng_parity_comparison.py [--postfit] [--out results.csv]

Outputs a CSV with prefit (and optionally postfit) weighted RMS for each
pulsar, plus postfit parameter differences.
"""
import argparse
import csv
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

NG_DIR = Path(__file__).parent.parent / "data" / "pulsars" / "NG_data" / "NG_15yr_partim"

# Parameters to compare postfit values for
COMPARE_PARAMS = ["F0", "F1", "F2", "DM", "PX", "PB", "A1", "ECC", "OM", "E", "EPS1", "EPS2",
                  "SINI", "M2", "PBDOT", "OMDOT", "GAMMA", "H3", "STIG", "H4", "KIN", "KOM",
                  "T0", "TASC", "XDOT", "A1DOT", "NE_SW",
                  "ELONG", "ELAT", "PMELONG", "PMELAT", "RAJ", "DECJ", "PMRA", "PMDEC",
                  "FD1", "FD2", "FD3", "FD4",
                  "FB0", "FB1", "FB2", "FB3", "FB4", "FB5", "FB6", "FB7", "FB8", "FB9"]

# Keywords that are never fit parameters (skip when parsing fit flags)
_SKIP_KEYWORDS = {
    "PEPOCH", "POSEPOCH", "DMEPOCH", "BINARY", "UNITS", "EPHEM", "CLK",
    "START", "FINISH", "TZRMJD", "TZRSITE", "TZRFRQ", "MODE", "NTOA",
    "CHI2", "CHI2R", "TRES", "INFO", "TIMEEPH", "T2CMETHOD", "CORRECT_TROPOSPHERE",
    "PLANET_SHAPIRO", "DILATEFREQ", "NE_SW_IFUNC_NHARMS", "CLOCK",
    "DMDATA", "SWM", "NITS", "TRACK",
}


def extract_free_params(par_path):
    """Extract parameters with fit flag=1 from a par file.

    Returns a list of parameter names as JUG expects them (e.g., JUMP1).
    DMX_* parameters are excluded because JUG auto-includes them from the par file.
    """
    free = []
    jump_idx = 0
    with open(par_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("C "):
                continue
            parts = line.split()
            key = parts[0].upper()

            # JUMP lines: "JUMP -flag value value [1 [uncertainty]]"
            if key == "JUMP":
                # Find the fit flag — it's after the jump value
                # Format varies: JUMP -fe L-wide -2.5351e-05 1 3.2e-09
                #   parts: [JUMP, -fe, L-wide, -2.5351e-05, 1, 3.2e-09]
                jump_idx += 1
                # Fit flag is typically at index 4 or later
                for i in range(3, len(parts)):
                    try:
                        val = float(parts[i])
                        # The next part after the numeric jump value could be the flag
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

            # Skip DMX_* — JUG auto-includes these from the par file
            if key.startswith("DMX"):
                continue

            # Standard format: PARAM value [fit_flag] [uncertainty]
            if len(parts) >= 3:
                try:
                    flag = int(parts[2])
                    if flag == 1:
                        # Canonicalize aliases (e.g., A1DOT → XDOT)
                        from jug.model.parameter_spec import canonicalize_param_name
                        free.append(canonicalize_param_name(key))
                except ValueError:
                    pass
    return free


# ── PINT ──────────────────────────────────────────────────────────────────────

def pint_prefit(par, tim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pint.models import get_model
        from pint.toa import get_TOAs
        from pint.residuals import Residuals
        import astropy.units as u
        import numpy as np
        t0 = time.time()
        m = get_model(str(par))
        t = get_TOAs(str(tim), model=m)
        r = Residuals(t, m)
        # Use raw TOA errors (not noise-model-scaled) to match JUG's metric
        raw_errs = t.get_errors().to(u.s).value
        w = 1.0 / raw_errs**2
        res = r.time_resids.to(u.s).value
        wmean = np.sum(w * res) / np.sum(w)
        wsdev = np.sqrt(np.sum(w * (res - wmean)**2) / np.sum(w))
        elapsed = time.time() - t0
        return len(t), wsdev * 1e6, elapsed  # return in microseconds


def pint_postfit(par, tim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pint.models import get_model
        from pint.toa import get_TOAs
        from pint.residuals import Residuals
        from pint.fitter import WLSFitter
        import astropy.units as u
        import numpy as np
        t0 = time.time()
        m = get_model(str(par))
        t = get_TOAs(str(tim), model=m)
        # Neutralise EFAC/EQUAD so PINT uses raw TOA errors,
        # matching JUG's NoiseConfig(EFAC=False, EQUAD=False, ECORR=False).
        for comp in m.components.values():
            for pname in comp.params:
                pu = pname.upper()
                if 'EFAC' in pu:
                    getattr(comp, pname).value = 1.0
                elif 'EQUAD' in pu:
                    getattr(comp, pname).value = 0.0
        f = WLSFitter(t, m)
        f.fit_toas()  # maxiter=1 by default
        r = Residuals(t, f.model)
        # Use raw TOA errors (not noise-scaled) for consistent wrms metric
        raw_errs = t.get_errors().to(u.s).value
        w = 1.0 / raw_errs**2
        res = r.time_resids.to(u.s).value
        wmean = np.sum(w * res) / np.sum(w)
        wrms = np.sqrt(np.sum(w * (res - wmean)**2) / np.sum(w)) * 1e6
        elapsed = time.time() - t0
        # Collect fitted param values
        params = {}
        for p in COMPARE_PARAMS:
            try:
                comp = getattr(f.model, p, None)
                if comp is not None and comp.value is not None:
                    params[p] = float(comp.value)
            except Exception:
                pass
        return wrms, params, elapsed


# ── JUG ───────────────────────────────────────────────────────────────────────

def jug_prefit(par, tim):
    import numpy as np
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
    import numpy as np
    t0 = time.time()
    s = open_session(str(par), str(tim), verbose=False)
    free_params = extract_free_params(par)
    # Disable EFAC/EQUAD to avoid fitter divergence in augmented solve;
    # DMX is always auto-included as augmented columns regardless of config.
    nc = NoiseConfig(enabled={
        "EFAC": False, "EQUAD": False,
        "ECORR": False, "RedNoise": False, "DMNoise": False, "DMX": True,
    })
    result = s.fit_parameters(fit_params=free_params, max_iter=1, noise_config=nc)
    elapsed = time.time() - t0
    # Compute wrms using raw errors (matches pint_postfit metric)
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
    name = par.stem.replace("_PINT", "").split("_")[0]
    row = {"name": name, "par": par.name, "status": "OK",
           "n_toas": None,
           "pint_prefit_us": None, "jug_prefit_us": None, "prefit_ratio": None,
           "pint_postfit_us": None, "jug_postfit_us": None, "postfit_ratio": None,
           "param_diffs": "",
           "pint_prefit_time_s": None, "jug_prefit_time_s": None,
           "pint_postfit_time_s": None, "jug_postfit_time_s": None}

    # Prefit
    try:
        n, pint_pre, pint_pre_t = pint_prefit(par, tim)
        row["n_toas"] = n
        row["pint_prefit_us"] = pint_pre
        row["pint_prefit_time_s"] = pint_pre_t
    except Exception:
        row["status"] = "PINT_PREFIT_ERROR"
        row["param_diffs"] = traceback.format_exc().splitlines()[-1]
        return row

    try:
        _, jug_pre, jug_pre_t = jug_prefit(par, tim)
        row["jug_prefit_us"] = jug_pre
        row["jug_prefit_time_s"] = jug_pre_t
        row["prefit_ratio"] = jug_pre / pint_pre if pint_pre else None
    except Exception:
        row["status"] = "JUG_PREFIT_ERROR"
        row["param_diffs"] = traceback.format_exc().splitlines()[-1]
        return row

    if not do_postfit:
        return row

    # Postfit
    try:
        pint_post, pint_params, pint_post_t = pint_postfit(par, tim)
        row["pint_postfit_us"] = pint_post
        row["pint_postfit_time_s"] = pint_post_t
    except Exception:
        row["status"] = "PINT_POSTFIT_ERROR"
        row["param_diffs"] = traceback.format_exc().splitlines()[-1]
        return row

    try:
        jug_post, jug_params, jug_post_t = jug_postfit(par, tim)
        row["jug_postfit_us"] = jug_post
        row["jug_postfit_time_s"] = jug_post_t
        row["postfit_ratio"] = jug_post / pint_post if pint_post else None
        # Parameter differences
        diffs = []
        for p in COMPARE_PARAMS:
            if p in pint_params and p in jug_params:
                diff = jug_params[p] - pint_params[p]
                rel = abs(diff / pint_params[p]) if pint_params[p] else 0
                if rel > 1e-6:
                    diffs.append(f"{p}:{diff:+.3e}({rel*100:.2f}%)")
        row["param_diffs"] = " ".join(diffs)
    except Exception:
        row["status"] = "JUG_POSTFIT_ERROR"
        row["param_diffs"] = traceback.format_exc().splitlines()[-1]

    return row


def grade(ratio):
    if ratio is None:
        return "N/A"
    if ratio < 0.95 or ratio > 1.05:
        return "BAD" if (ratio < 0.80 or ratio > 1.20) else "MODERATE"
    return "GOOD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postfit", action="store_true", help="Also run postfit comparison")
    parser.add_argument("--out", default="ng_parity_results.csv")
    parser.add_argument("--pulsar", help="Run only this pulsar name (substring match)")
    args = parser.parse_args()

    pairs = sorted([(p, p.with_suffix(".tim"))
                    for p in NG_DIR.glob("*.par")
                    if p.with_suffix(".tim").exists()])

    if args.pulsar:
        pairs = [(p, t) for p, t in pairs if args.pulsar in p.stem]

    print(f"Found {len(pairs)} par/tim pairs in {NG_DIR}")
    print(f"Mode: {'prefit + postfit' if args.postfit else 'prefit only'}")
    print()

    rows = []
    for i, (par, tim) in enumerate(pairs, 1):
        name = par.stem
        print(f"[{i:02d}/{len(pairs)}] {name} ... ", end="", flush=True)
        t0 = time.time()
        row = compare_one(par, tim, args.postfit)
        elapsed = time.time() - t0
        prefit_grade = grade(row.get("prefit_ratio"))
        postfit_grade = grade(row.get("postfit_ratio")) if args.postfit else ""
        status_str = row["status"]
        if row["pint_prefit_us"] and row["jug_prefit_us"]:
            status_str = (f"pre: PINT={row['pint_prefit_us']:.4f} JUG={row['jug_prefit_us']:.4f} "
                          f"ratio={row['prefit_ratio']:.3f} [{prefit_grade}]")
            if args.postfit and row["pint_postfit_us"] and row["jug_postfit_us"]:
                status_str += (f" | post: PINT={row['pint_postfit_us']:.4f} "
                               f"JUG={row['jug_postfit_us']:.4f} ratio={row['postfit_ratio']:.3f} "
                               f"[{postfit_grade}]")
        print(f"{status_str}  ({elapsed:.1f}s)")
        rows.append(row)

    # Write CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    pre_ratios = [r["prefit_ratio"] for r in rows if r["prefit_ratio"] is not None]
    print(f"Prefit: {len(pre_ratios)}/{len(rows)} pulsars computed")
    if pre_ratios:
        good = sum(1 for r in pre_ratios if 0.95 <= r <= 1.05)
        mod = sum(1 for r in pre_ratios if (0.80 <= r < 0.95) or (1.05 < r <= 1.20))
        bad = sum(1 for r in pre_ratios if r < 0.80 or r > 1.20)
        print(f"  GOOD (<5%): {good}  MODERATE (5-20%): {mod}  BAD (>20%): {bad}")
        print(f"  Median ratio: {np.median(pre_ratios):.4f}  "
              f"Max ratio: {max(pre_ratios):.4f}  Min ratio: {min(pre_ratios):.4f}")
        print()
        print("PROBLEM PULSARS (prefit ratio outside [0.95, 1.05]):")
        for r in rows:
            if r["prefit_ratio"] is not None and not (0.95 <= r["prefit_ratio"] <= 1.05):
                print(f"  {r['par']:55s}  ratio={r['prefit_ratio']:.3f}")
        errors = [r for r in rows if "ERROR" in r["status"]]
        if errors:
            print()
            print("ERRORS:")
            for r in errors:
                print(f"  {r['par']:55s}  {r['status']}: {r['param_diffs'][:80]}")

    if args.postfit:
        post_ratios = [r["postfit_ratio"] for r in rows if r["postfit_ratio"] is not None]
        print()
        print(f"Postfit: {len(post_ratios)}/{len(rows)} pulsars computed")
        if post_ratios:
            good = sum(1 for r in post_ratios if 0.95 <= r <= 1.05)
            mod = sum(1 for r in post_ratios if (0.80 <= r < 0.95) or (1.05 < r <= 1.20))
            bad = sum(1 for r in post_ratios if r < 0.80 or r > 1.20)
            print(f"  GOOD (<5%): {good}  MODERATE (5-20%): {mod}  BAD (>20%): {bad}")
            print(f"  Median ratio: {np.median(post_ratios):.4f}  "
                  f"Max ratio: {max(post_ratios):.4f}  Min ratio: {min(post_ratios):.4f}")
            print()
            print("PROBLEM PULSARS (postfit ratio outside [0.95, 1.05]):")
            for r in rows:
                if r["postfit_ratio"] is not None and not (0.95 <= r["postfit_ratio"] <= 1.05):
                    print(f"  {r['par']:55s}  ratio={r['postfit_ratio']:.3f}  {r['param_diffs'][:60]}")

    # Speed summary
    print()
    print("SPEED:")
    jug_pre_times = [r["jug_prefit_time_s"] for r in rows if r["jug_prefit_time_s"] is not None]
    pint_pre_times = [r["pint_prefit_time_s"] for r in rows if r["pint_prefit_time_s"] is not None]
    if jug_pre_times:
        print(f"  JUG prefit:  median={np.median(jug_pre_times):.1f}s  "
              f"mean={np.mean(jug_pre_times):.1f}s  max={max(jug_pre_times):.1f}s")
    if pint_pre_times:
        print(f"  PINT prefit: median={np.median(pint_pre_times):.1f}s  "
              f"mean={np.mean(pint_pre_times):.1f}s  max={max(pint_pre_times):.1f}s")
    if args.postfit:
        jug_post_times = [r["jug_postfit_time_s"] for r in rows if r["jug_postfit_time_s"] is not None]
        pint_post_times = [r["pint_postfit_time_s"] for r in rows if r["pint_postfit_time_s"] is not None]
        if jug_post_times:
            print(f"  JUG postfit:  median={np.median(jug_post_times):.1f}s  "
                  f"mean={np.mean(jug_post_times):.1f}s  max={max(jug_post_times):.1f}s")
        if pint_post_times:
            print(f"  PINT postfit: median={np.median(pint_post_times):.1f}s  "
                  f"mean={np.mean(pint_post_times):.1f}s  max={max(pint_post_times):.1f}s")

    print()
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
