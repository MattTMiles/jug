#!/usr/bin/env python3
"""General-purpose JUG vs PINT parameter comparison script.

Usage
-----
    python compare_jug_pint.py <par> <tim> [options]

Options
-------
    --noise             Use noise model (GLS for PINT, augmented SVD for JUG)
    --noise-free        Force WLS even if noise params present (default when no noise params)
    --fit-params P1 P2  Override auto-detected fit params
    --clock-dir DIR     Clock file directory (default: JUG built-in)
    --no-plot           Print table only (no matplotlib window)
    --save FIG.png      Save figure instead of showing
    --pint-maxiter N    PINT max iterations (default: 5 for GLS, 20 for WLS)

Output
------
    - Parameter table (stdout): value, uncertainty, |Δ|/σ for each fitted param
    - Plot: one point-with-error-bar per code per param, colored by |Δ|/σ_PINT
    - Noise panel: JUG noise realizations per component (if noise-aware)
    - Whitened residual comparison: JUG vs PINT post-fit residuals vs time

Three-way logic (if TEMPO2 available)
--------------------------------------
    JUG == PINT: correct
    JUG == PINT, T2 differs: accept JUG/PINT
    PINT == T2, JUG differs: JUG wrong
    all differ: show normalized distances, flag if JUG not closest to PINT

Ridge degeneracy
----------------
    Degenerate params (T0, OM, PB, OMDOT, KOM, M2, PMRA, PMDEC, PX) are shown
    but labeled [DEG] — σ-parity not enforced, distance metric still computed.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Parameter metadata
# ---------------------------------------------------------------------------

# PINT uses different internal names for some par-file parameters
PINT_ALIASES = {
    "XDOT": "A1DOT",
    "OMDOT": "OMDOT",   # same
}

# Params that are exactly degenerate for near-circular DDK orbits
DEGENERATE_PARAMS = {"T0", "OM", "PB", "OMDOT", "KOM", "M2", "PMRA", "PMDEC", "PX"}

# Params where unit mismatch makes direct numeric comparison unreliable
# (RAJ in hours vs angle; DECJ in deg vs angle)
ANGLE_PARAMS = {"RAJ", "DECJ"}

# Param categories for grouped plotting
PARAM_CATEGORIES = {
    "Spin":       ["F0", "F1", "F2", "F3"],
    "Astrometry": ["RAJ", "DECJ", "PMRA", "PMDEC", "PX", "PX_A1"],
    "DM":         ["DM", "DM1", "DM2", "DM3"],
    "Binary":     ["PB", "PBDOT", "A1", "XDOT", "ECC", "T0", "OM", "OMDOT",
                   "M2", "SINI", "KIN", "KOM", "TASC", "EPS1", "EPS2",
                   "EPS1DOT", "EPS2DOT", "H3", "STIG"],
    "FD":         [f"FD{i}" for i in range(1, 12)],
    "JUMP":       [],   # filled dynamically
}


# ---------------------------------------------------------------------------
# Par-file helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TCB → TDB conversion
# ---------------------------------------------------------------------------

def detect_timescale(par_file: Path) -> str:
    """Return 'TCB', 'TDB', or 'TT' from par file UNITS line."""
    with open(par_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0].upper() == "UNITS" and len(parts) >= 2:
                return parts[1].upper()
    return "TDB"  # default


def save_as_tdb_par(par_file: Path) -> Path:
    """Convert a TCB par file to TDB, saving as <name>_tdb.par.

    If the file is already TDB, returns the original path unchanged.

    The converted file is saved next to the original:
        J0437-4715.par  →  J0437-4715_tdb.par

    Uses JUG's existing Irwin & Fukushima (1999) conversion pipeline.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from jug.io.par_reader import parse_par_file
    from jug.io.par_writer import write_par_file
    from jug.utils.timescales import convert_par_params_to_tdb, parse_timescale

    params = parse_par_file(par_file)
    timescale = parse_timescale(params)

    if timescale != "TCB":
        return par_file  # already TDB/TT — no conversion needed

    # Build output path: insert _tdb before .par
    stem = par_file.stem
    if not stem.endswith("_tdb"):
        out_path = par_file.parent / f"{stem}_tdb.par"
    else:
        out_path = par_file  # already named *_tdb.par

    # Convert in place using JUG's pipeline
    params, log = convert_par_params_to_tdb(params, verbose=False)
    fit_flags = set(params.get("_fit_flags", {}).keys())
    write_par_file(params, out_path, fit_params=fit_flags)

    print(f"[tcb→tdb] Converted {par_file.name} → {out_path.name}")
    for msg in log:
        print(f"  {msg}")

    return out_path


def ensure_tdb_par(par_file: Path, auto_convert: bool = True) -> Path:
    """Return a TDB par file, converting from TCB if needed.

    Parameters
    ----------
    par_file : Path
        Input par file (TCB or TDB).
    auto_convert : bool
        If True (default), auto-convert TCB → TDB and return converted path.
        If False, return original path and let callers handle the timescale.

    Returns
    -------
    Path
        Path to a TDB par file (may be the original if already TDB).
    """
    ts = detect_timescale(par_file)
    if ts == "TDB":
        return par_file
    if ts == "TCB":
        if auto_convert:
            return save_as_tdb_par(par_file)
        else:
            print(f"[warning] Par file is TCB but auto_convert=False: {par_file.name}")
            return par_file
    print(f"[warning] Unexpected UNITS={ts} in {par_file.name}")
    return par_file


# ---------------------------------------------------------------------------
# Pre-fit residuals (physics model check before any fitting)
# ---------------------------------------------------------------------------

def compute_prefit_residuals_jug(par_file, tim_file, clock_dir=None):
    """Compute JUG pre-fit residuals (no fitting, just evaluate the model)."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    result = compute_residuals_simple(
        str(par_file), str(tim_file),
        clock_dir=clock_dir, verbose=False,
    )
    return {
        "residuals_us": np.array(result["residuals_us"]),
        "errors_us":    np.array(result.get("errors_us", result.get("toa_errors_us", []))),
        "tdb_mjd":      np.array(result.get("tdb_mjd", [])),
        "wrms_us":      result["weighted_rms_us"],
        "n_toas":       result["n_toas"],
    }


def _patch_par_for_pint(par_file):
    """Return StringIO of par file with BINARY T2 → DD/ELL1 for PINT compatibility.

    PINT does not support the Tempo2 T2 binary model.
    Substitution: T2 → ELL1 if EPS1/EPS2 present, else DD.
    """
    import io
    with open(par_file) as f:
        lines = f.readlines()

    has_eps = any(
        l.strip().split()[0].upper() in ("EPS1", "EPS2")
        for l in lines if l.strip() and not l.startswith("#")
    )

    patched = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0].upper() == "BINARY" and len(parts) >= 2:
            if parts[1].upper() == "T2":
                replacement = "ELL1" if has_eps else "DD"
                print(f"[pint] BINARY T2 → {replacement} (PINT does not support T2)")
                line = f"BINARY {replacement}\n"
        patched.append(line)

    return io.StringIO("".join(patched))


def compute_prefit_residuals_pint(par_file, tim_file, clock_dir=None):
    """Compute PINT pre-fit residuals (no fitting)."""
    import pint.models
    import pint.toa
    import pint.residuals

    if clock_dir is not None:
        import os
        os.environ["PINT_CLOCK_OVERRIDE"] = str(clock_dir)

    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model = pint.models.get_model(_patch_par_for_pint(par_file))
    bipm = (model.CLOCK.value.replace("TT(", "").replace(")", "")
            if hasattr(model, "CLOCK") else "BIPM2023")
    toas = pint.toa.get_TOAs(str(tim_file), planets=True,
                              ephem=model.EPHEM.value if hasattr(model, "EPHEM") else "DE436",
                              bipm_version=bipm, model=model)
    res  = pint.residuals.Residuals(toas, model)
    res_us  = res.time_resids.to("us").value
    errs_us = toas.get_errors().to("us").value
    weights = 1.0 / errs_us ** 2
    wrms    = float(np.sqrt(np.sum(weights * res_us ** 2) / np.sum(weights)))

    try:
        tdb_mjd = toas.get_mjds().value
    except Exception:
        try:
            tdb_mjd = np.array([t.tdb.mjd for t in toas])
        except Exception:
            tdb_mjd = np.zeros(len(res_us))

    return {
        "residuals_us": res_us,
        "errors_us":    errs_us,
        "tdb_mjd":      tdb_mjd,
        "wrms_us":      wrms,
        "n_toas":       toas.ntoas,
    }


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation_matrix(cov: np.ndarray, param_names: list[str]):
    """Convert covariance to correlation matrix. Returns (corr, names)."""
    diag = np.sqrt(np.diag(cov))
    # avoid division by zero
    outer = np.outer(diag, diag)
    outer[outer == 0] = 1.0
    corr = cov / outer
    return corr, param_names


def get_fit_params_from_par(par_file: Path) -> list[str]:
    """Extract parameters with fit flag = 1 from par file."""
    sys.path.insert(0, str(Path(__file__).parent))
    from jug.io.par_reader import parse_par_file
    params = parse_par_file(par_file)
    fit_flags = params.get("_fit_flags", {})
    return list(fit_flags.keys())


def has_noise_model(par_file: Path) -> bool:
    """Return True if par file contains any noise model parameters."""
    noise_keywords = {
        "T2EFAC", "T2EQUAD", "EFAC", "EQUAD", "ECORR", "TNECORR",
        "TNEF", "TNEQ", "TNREDAMP", "TNDMAMP", "TNCHROMAMP",
        "TNDMAMP", "TNREDGAM", "TNDMGAM",
    }
    with open(par_file) as f:
        for line in f:
            word = line.strip().split()[0].upper() if line.strip() else ""
            if word in noise_keywords:
                return True
    return False


# ---------------------------------------------------------------------------
# JUG fit
# ---------------------------------------------------------------------------

_NOISE_BASIS_PARAMS = {
    # Red / DM noise
    "RNAMP", "RNIDX", "TNREDAMP", "TNREDGAM", "TNDMAMP", "TNDMGAM",
    # White noise (EFAC/EQUAD/ECORR in all naming conventions).
    # Must strip ECORR so GLS mode is not triggered; otherwise ECORR
    # (532 epoch groups) absorbs secular binary-param signal (K96 PM),
    # producing wrong KOM/KIN even though WRMS looks better.
    "EFAC", "EQUAD", "ECORR",
    "T2EFAC", "T2EQUAD", "T2ECORR",
    "TNEF", "TNEQ", "TNECORR",
}

def _strip_noise_params(par_file: Path) -> Path:
    """Write a temp par file with all noise params removed for pure WLS."""
    import tempfile
    lines = Path(par_file).read_text().splitlines(keepends=True)
    kept = [l for l in lines
            if not l.split() or l.split()[0].upper() not in _NOISE_BASIS_PARAMS]
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)
    tmp.writelines(kept)
    tmp.close()
    return Path(tmp.name)


def run_jug_fit(par_file, tim_file, fit_params, clock_dir=None, verbose=False,
                noise_free=False):
    """Run JUG fit. Returns dict with keys: params, uncertainties, wrms_us,
    residuals_us, errors_us, tdb_mjd, noise_realizations, chi2."""
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    import logging as _l
    _tmp_par = None
    if noise_free:
        _tmp_par = _strip_noise_params(par_file)
        par_file = _tmp_par
    if not verbose:
        _l.disable(_l.CRITICAL)
    result = fit_parameters_optimized(
        par_file, tim_file, fit_params,
        clock_dir=clock_dir,
        verbose=verbose,
    )
    if _tmp_par is not None:
        _tmp_par.unlink(missing_ok=True)
    if not verbose:
        _l.disable(_l.NOTSET)
    return {
        "params":            result["final_params"],
        "uncertainties":     result.get("uncertainties", {}),
        "wrms_us":           result["final_rms"],
        "residuals_us":      np.array(result.get("residuals_us", [])),
        "errors_us":         np.array(result.get("errors_us", [])),
        "tdb_mjd":           np.array(result.get("tdb_mjd", [])),
        "noise_realizations": result.get("noise_realizations", {}),
        "chi2":              result.get("final_chi2", float("nan")),
        "converged":         result.get("converged", None),
        "iterations":        result.get("iterations", None),
    }


# ---------------------------------------------------------------------------
# PINT fit
# ---------------------------------------------------------------------------

def run_pint_fit(par_file, tim_file, fit_params, gls=False,
                 clock_dir=None, maxiter=None, verbose=False):
    """Run PINT fit. Returns dict with same keys as run_jug_fit.

    PINT does not expose per-process noise realizations; noise_realizations={}.
    """
    import pint.models
    import pint.toa
    import pint.fitter

    if not verbose:
        logging.getLogger("pint").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    if clock_dir is not None:
        import os
        os.environ["PINT_CLOCK_OVERRIDE"] = str(clock_dir)

    model = pint.models.get_model(_patch_par_for_pint(par_file))
    bipm = (model.CLOCK.value.replace("TT(", "").replace(")", "")
            if hasattr(model, "CLOCK") else "BIPM2023")
    ephem = model.EPHEM.value if hasattr(model, "EPHEM") else "DE436"
    toas = pint.toa.get_TOAs(str(tim_file), planets=True,
                              ephem=ephem, bipm_version=bipm, model=model)

    # Set exactly the params we want to fit
    for p in model.free_params[:]:
        getattr(model, p).frozen = True
    for p in fit_params:
        pint_name = PINT_ALIASES.get(p, p)
        if hasattr(model, pint_name):
            getattr(model, pint_name).frozen = False
        elif hasattr(model, p):
            getattr(model, p).frozen = False

    # Re-freeze JUMPs with no matching TOAs (overrides above; must come last)
    if hasattr(model, "find_empty_masks"):
        model.find_empty_masks(toas, freeze=True)

    fitter_cls = pint.fitter.GLSFitter if gls else pint.fitter.WLSFitter
    _maxiter = maxiter if maxiter is not None else (5 if gls else 20)
    fitter = fitter_cls(toas, model)
    fitter.fit_toas(maxiter=_maxiter)

    res_us  = fitter.resids.time_resids.to("us").value
    errs_us = toas.get_errors().to("us").value
    weights = 1.0 / errs_us ** 2
    wrms    = float(np.sqrt(np.sum(weights * res_us ** 2) / np.sum(weights)))

    # Extract tdb_mjd from TOAs
    try:
        tdb_mjd = toas.get_mjds().value
    except Exception:
        try:
            tdb_mjd = np.array([t.tdb.mjd for t in toas])
        except Exception:
            tdb_mjd = np.zeros(len(res_us))

    params  = {}
    uncerts = {}
    fitparams = fitter.get_fitparams()

    for p in fit_params:
        pint_name = PINT_ALIASES.get(p, p)
        # Try both the aliased name and the original
        for try_name in [pint_name, p]:
            if try_name in fitparams:
                obj = fitparams[try_name]
                try:
                    params[p]  = float(obj.quantity.value)
                    uncerts[p] = float(obj.uncertainty_value or 0.0)
                except Exception:
                    pass
                break

    return {
        "params":             params,
        "uncertainties":      uncerts,
        "wrms_us":            wrms,
        "residuals_us":       res_us,
        "errors_us":          errs_us,
        "tdb_mjd":            tdb_mjd,
        "noise_realizations": {},   # PINT doesn't expose per-process realizations
        "chi2":               float("nan"),
        "converged":          None,
        "iterations":         None,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _normalized_distance(jug_val, pint_val, pint_unc):
    """Return |JUG - PINT| / σ_PINT, or inf if σ=0."""
    if pint_unc == 0.0:
        return float("inf")
    return abs(float(jug_val) - float(pint_val)) / float(pint_unc)


def _print_ridge_checks(jug, pint_r):
    """Check ridge consistency for degenerate DDK binary parameters.

    For near-circular orbits:
      ΔT0 ≈ (PB/2π) * ΔOM         (T0 ↔ OM ridge)
      ΔPB ↔ ΔOMDOT (correlated but not a simple ratio)
    """
    jp = jug["params"]
    pp = pint_r["params"]

    checks = []

    if "T0" in jp and "T0" in pp and "OM" in jp and "OM" in pp and "PB" in jp and "PB" in pp:
        T0_dif  = jp["T0"]   - pp["T0"]    # days
        OM_dif  = jp["OM"]   - pp["OM"]    # radians
        PB      = 0.5 * (jp["PB"] + pp["PB"])  # days
        predicted_T0_dif = (PB / (2 * np.pi)) * OM_dif
        ridge_err = T0_dif - predicted_T0_dif
        # Scale: σ of T0 from PINT (if available)
        t0_unc = pint_r["uncertainties"].get("T0", 0.0)
        if t0_unc > 0:
            ridge_nsigma = abs(ridge_err) / t0_unc
            checks.append(
                f"  T0↔OM ridge:  ΔT0={T0_dif*86400:.3f} s, predicted={(predicted_T0_dif*86400):.3f} s, "
                f"residual={ridge_err*86400:.3f} s  ({ridge_nsigma:.1f}σ) "
                + ("OK" if ridge_nsigma < 5 else "RIDGE MISMATCH — possible derivative bug")
            )
        else:
            checks.append(
                f"  T0↔OM ridge:  ΔT0={T0_dif*86400:.3f} s, predicted={(predicted_T0_dif*86400):.3f} s"
            )

    if checks:
        print("Ridge consistency checks:")
        for c in checks:
            print(c)
        print()


def print_comparison_table(jug, pint_r, fit_params):
    """Print parameter comparison table to stdout."""
    print()
    print("=" * 100)
    print(f"{'Param':<12} {'JUG value':>22} {'JUG σ':>14} {'PINT value':>22} {'PINT σ':>14} {'|Δ|/σ':>8}  {'Notes'}")
    print("=" * 100)

    for p in fit_params:
        jug_val  = jug["params"].get(p)
        jug_unc  = jug["uncertainties"].get(p, 0.0)
        pint_val = pint_r["params"].get(p)
        pint_unc = pint_r["uncertainties"].get(p, 0.0)

        notes = []
        if p in DEGENERATE_PARAMS:
            notes.append("DEG")
        if p in ANGLE_PARAMS:
            notes.append("ANGLE")

        if p in ANGLE_PARAMS:
            # Units differ between codes (JUG: radians, PINT: hours/degrees)
            jv_str = f"{jug_val:.10g}" if jug_val is not None else "—"
            pv_str = f"{pint_val:.10g}" if pint_val is not None else "—"
            print(f"  {p:<10} {jv_str:>22} {'':>14} {pv_str:>22} {'':>14} {'—':>8}  ANGLE (units differ)")
            continue

        if jug_val is None or pint_val is None:
            print(f"  {p:<10} {'—':>22} {'—':>14} {'—':>22} {'—':>14} {'—':>8}  missing in one code")
            continue

        dist = _normalized_distance(jug_val, pint_val, pint_unc)
        dist_str = f"{dist:.2f}" if np.isfinite(dist) else "—"

        # Flag divergence
        if np.isfinite(dist):
            if dist < 1.0:
                pass
            elif dist < 3.0:
                notes.append("!1-3σ")
            elif dist < 10.0:
                notes.append("!!>3σ")
            else:
                notes.append("!!!>10σ")

        notes_str = ", ".join(notes) if notes else ""
        print(f"  {p:<10} {jug_val:>22.10g} {jug_unc:>14.4g} "
              f"{pint_val:>22.10g} {pint_unc:>14.4g} {dist_str:>8}  {notes_str}")

    print("=" * 100)
    print(f"  {'WRMS (µs)':<10} {jug['wrms_us']:>22.6f} {'':>14} {pint_r['wrms_us']:>22.6f}")
    print()

    # Ridge consistency checks for near-circular DDK orbits
    _print_ridge_checks(jug, pint_r)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _categorize_param(p):
    """Return category string for param p."""
    for cat, members in PARAM_CATEGORIES.items():
        if p in members:
            return cat
    if p.startswith("JUMP"):
        return "JUMP"
    if p.startswith("FD"):
        return "FD"
    return "Other"


def _sigma_color(dist):
    """Return matplotlib color for a normalized distance."""
    if not np.isfinite(dist):
        return "gray"
    if dist < 1.0:
        return "#2ecc71"   # green
    elif dist < 3.0:
        return "#f39c12"   # orange
    elif dist < 10.0:
        return "#e74c3c"   # red
    else:
        return "#8e44ad"   # purple — severely drifted


def plot_parameter_comparison(jug, pint_r, fit_params, save_path=None):
    """Plot parameter comparison: one panel per category, color by |Δ|/σ."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Collect plottable params (numeric, both codes have value)
    plottable = []
    for p in fit_params:
        if p in ANGLE_PARAMS:
            continue
        jv = jug["params"].get(p)
        pv = pint_r["params"].get(p)
        ju = jug["uncertainties"].get(p, 0.0)
        pu = pint_r["uncertainties"].get(p, 0.0)
        if jv is None or pv is None:
            continue
        plottable.append((p, float(jv), float(ju), float(pv), float(pu),
                          _categorize_param(p)))

    if not plottable:
        print("[plot] No plottable params found.")
        return

    # Group by category
    cats_present = []
    cat_params = {}
    for entry in plottable:
        cat = entry[5]
        if cat not in cat_params:
            cat_params[cat] = []
            cats_present.append(cat)
        cat_params[cat].append(entry)

    n_cats = len(cats_present)
    fig = plt.figure(figsize=(max(14, 3 * n_cats), 6 * n_cats))
    gs = gridspec.GridSpec(n_cats, 1, hspace=0.5)

    for idx, cat in enumerate(cats_present):
        entries = cat_params[cat]
        ax = fig.add_subplot(gs[idx])

        n = len(entries)
        x = np.arange(n)
        labels = [e[0] for e in entries]

        for xi, (p, jv, ju, pv, pu, _) in enumerate(entries):
            dist = _normalized_distance(jv, pv, pu)
            col = _sigma_color(dist)

            # Normalize: show (JUG - PINT) / σ_JUG in units of JUG σ,
            # OR show absolute values if σ differs by >10x (can't share y axis).
            # Strategy: show relative to PINT value, in units of PINT σ (or JUG σ if PINT σ=0)
            ref_unc = pu if pu > 0 else ju
            if ref_unc == 0:
                continue

            jug_rel = (jv - pv) / ref_unc
            jug_err = ju / ref_unc if ref_unc > 0 else 0.0
            pint_rel = 0.0
            pint_err = pu / ref_unc if ref_unc > 0 else 0.0

            ax.errorbar(xi - 0.15, jug_rel,  yerr=jug_err,  fmt="o",
                        color=col, capsize=4, ms=6, label="JUG"  if xi == 0 else "")
            ax.errorbar(xi + 0.15, pint_rel, yerr=pint_err, fmt="s",
                        color="steelblue", capsize=4, ms=6, label="PINT" if xi == 0 else "",
                        alpha=0.7)

            # Annotate |Δ|/σ above the JUG point
            if np.isfinite(dist):
                ax.annotate(f"{dist:.1f}σ", (xi - 0.15, jug_rel + jug_err + 0.1),
                            fontsize=7, ha="center", color=col)

            if p in DEGENERATE_PARAMS:
                ax.axvline(xi, color="gray", alpha=0.2, lw=8, zorder=0)

        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axhline(+1, color="gray", lw=0.5, ls=":")
        ax.axhline(-1, color="gray", lw=0.5, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("(JUG − PINT) / σ")
        ax.set_title(f"{cat} parameters", fontsize=10)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Add WRMS info to title
    wrms_diff_ns = abs(jug["wrms_us"] - pint_r["wrms_us"]) * 1000
    fig.suptitle(
        f"JUG vs PINT parameter comparison\n"
        f"JUG WRMS={jug['wrms_us']:.4f} µs  |  PINT WRMS={pint_r['wrms_us']:.4f} µs  |  Δ={wrms_diff_ns:.2f} ns\n"
        f"Gray bands = degenerate params  |  Color = |Δ|/σ: green<1σ, orange<3σ, red<10σ, purple>10σ",
        fontsize=10,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_noise_realizations(jug, pint_r, save_path=None):
    """Plot JUG noise realizations and compare whitened residuals."""
    noise = jug.get("noise_realizations", {})
    # Filter to main components (skip _err keys)
    components = {k: v for k, v in noise.items() if not k.endswith("_err")}
    if not components:
        print("[noise plot] No noise realizations in JUG result.")
        return

    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jug_mjd  = jug["tdb_mjd"]
    pint_mjd = pint_r["tdb_mjd"]

    n_comp = len(components)
    fig, axes = plt.subplots(n_comp + 2, 1,
                             figsize=(14, 3 * (n_comp + 2)),
                             sharex=False)

    # Panel 0: JUG post-fit residuals
    ax = axes[0]
    if len(jug_mjd) > 0:
        ax.errorbar(jug_mjd, jug["residuals_us"] * 1e3,
                    yerr=jug["errors_us"] * 1e3,
                    fmt=",", alpha=0.5, label="JUG residuals")
        ax.errorbar(pint_mjd, pint_r["residuals_us"] * 1e3,
                    yerr=pint_r["errors_us"] * 1e3,
                    fmt=",", alpha=0.5, label="PINT residuals")
    ax.set_ylabel("Residual (ns)")
    ax.set_title("Post-fit residuals")
    ax.legend(fontsize=8)

    # Panel 1: residual difference JUG - PINT (requires same TOAs in same order)
    ax = axes[1]
    if len(jug_mjd) == len(pint_mjd) and np.allclose(jug_mjd, pint_mjd, atol=1e-9):
        diff_ns = (jug["residuals_us"] - pint_r["residuals_us"]) * 1e3
        ax.scatter(jug_mjd, diff_ns, s=2, alpha=0.5, color="purple")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("JUG − PINT (ns)")
        ax.set_title(f"Residual difference  RMS={np.std(diff_ns):.2f} ns")
    else:
        ax.text(0.5, 0.5, "TOA order mismatch — cannot diff residuals",
                ha="center", va="center", transform=ax.transAxes)

    # Panels 2+: JUG noise realizations per component
    for i, (label, vals) in enumerate(components.items()):
        ax = axes[i + 2]
        err_key = label + "_err"
        errs = noise.get(err_key)
        if len(jug_mjd) == len(vals):
            if errs is not None:
                ax.errorbar(jug_mjd, vals, yerr=errs, fmt=",", alpha=0.6)
            else:
                ax.scatter(jug_mjd, vals, s=2, alpha=0.6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("µs")
        ax.set_title(f"JUG noise: {label}")

    fig.tight_layout()
    sp = save_path.replace(".png", "_noise.png") if save_path else None
    if sp:
        fig.savefig(sp, dpi=150, bbox_inches="tight")
        print(f"[noise plot] Saved to {sp}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare JUG and PINT timing solutions for a pulsar."
    )
    parser.add_argument("par",  type=Path, help="Par file path")
    parser.add_argument("tim",  type=Path, help="Tim file path")
    parser.add_argument("--noise", action="store_true",
                        help="Use noise model (GLS). Auto-detected if not set.")
    parser.add_argument("--noise-free", action="store_true",
                        help="Force WLS (ignore noise params in par)")
    parser.add_argument("--fit-params", nargs="+", metavar="P",
                        help="Override fit params (default: read from par file)")
    parser.add_argument("--clock-dir", type=Path, default=None,
                        help="Clock file directory")
    parser.add_argument("--no-plot", action="store_true",
                        help="Print table only, no matplotlib")
    parser.add_argument("--save", metavar="FILE",
                        help="Save figure to file (PNG)")
    parser.add_argument("--pint-maxiter", type=int, default=None,
                        help="PINT fitter max iterations")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose output from fitters")
    args = parser.parse_args()

    par = args.par
    tim = args.tim

    if not par.exists():
        sys.exit(f"ERROR: par file not found: {par}")
    if not tim.exists():
        sys.exit(f"ERROR: tim file not found: {tim}")

    # Auto-detect timescale and convert TCB → TDB if needed
    ts = detect_timescale(par)
    if ts == "TCB":
        print(f"[timescale] Detected TCB par file — converting to TDB...")
        par = save_as_tdb_par(par)
        print(f"[timescale] Using converted par: {par}")
    elif ts == "TDB":
        print(f"[timescale] Par file is TDB — no conversion needed")
    else:
        print(f"[timescale] UNITS={ts} — proceeding without conversion")
    print()

    # Auto-detect fit params
    if args.fit_params:
        fit_params = args.fit_params
        print(f"[config] Fit params (user-specified): {fit_params}")
    else:
        fit_params = get_fit_params_from_par(par)
        print(f"[config] Fit params (from par file, n={len(fit_params)}): {fit_params}")

    # Auto-detect noise
    if args.noise_free:
        use_noise = False
    elif args.noise:
        use_noise = True
    else:
        use_noise = has_noise_model(par)
    print(f"[config] Noise model: {'GLS' if use_noise else 'WLS (noise-free)'}")
    print(f"[config] Par:  {par}")
    print(f"[config] Tim:  {tim}")
    if args.clock_dir:
        print(f"[config] Clock dir: {args.clock_dir}")
    print()

    # --- JUG fit ---
    print("Running JUG fit...")
    try:
        jug = run_jug_fit(par, tim, fit_params,
                          clock_dir=str(args.clock_dir) if args.clock_dir else None,
                          verbose=args.verbose,
                          noise_free=not use_noise)
        print(f"  JUG  WRMS = {jug['wrms_us']:.6f} µs  "
              f"(converged={jug['converged']}, iter={jug['iterations']})")
    except Exception as e:
        sys.exit(f"JUG fit failed: {e}")

    # --- PINT fit ---
    print("Running PINT fit...")
    try:
        pint_r = run_pint_fit(par, tim, fit_params,
                               gls=use_noise,
                               clock_dir=args.clock_dir,
                               maxiter=args.pint_maxiter,
                               verbose=args.verbose)
        print(f"  PINT WRMS = {pint_r['wrms_us']:.6f} µs")
    except Exception as e:
        sys.exit(f"PINT fit failed: {e}")

    print()

    # --- Table ---
    print_comparison_table(jug, pint_r, fit_params)

    # Summary stats
    dists = []
    for p in fit_params:
        if p in ANGLE_PARAMS:
            continue
        jv = jug["params"].get(p)
        pv = pint_r["params"].get(p)
        pu = pint_r["uncertainties"].get(p, 0.0)
        if jv is not None and pv is not None and pu > 0:
            dists.append((p, _normalized_distance(jv, pv, pu)))

    if dists:
        ok    = sum(1 for _, d in dists if d < 1.0)
        warn  = sum(1 for _, d in dists if 1.0 <= d < 3.0)
        bad   = sum(1 for _, d in dists if 3.0 <= d < 10.0)
        drift = sum(1 for _, d in dists if d >= 10.0)
        print(f"Summary (|Δ|/σ_PINT):  <1σ: {ok}  1-3σ: {warn}  3-10σ: {bad}  >10σ: {drift}")
        if bad + drift > 0:
            worst = sorted(dists, key=lambda x: -x[1])[:5]
            print(f"Worst: " + "  ".join(f"{p}={d:.1f}σ" for p, d in worst))
        print()

    # --- Plots ---
    if not args.no_plot:
        print("Generating parameter plot...")
        plot_parameter_comparison(jug, pint_r, fit_params, save_path=args.save)

        if use_noise and jug.get("noise_realizations"):
            print("Generating noise realization plot...")
            plot_noise_realizations(jug, pint_r, save_path=args.save)


if __name__ == "__main__":
    main()
