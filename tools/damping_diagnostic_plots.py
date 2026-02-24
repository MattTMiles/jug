#!/usr/bin/env python3
"""
Diagnostic plots comparing JUG vs PINT postfit residuals for pulsars where
JUG's damping produces different WRMS than PINT's full-step WLS.

Generates side-by-side residual plots, WRMS annotations, and parameter step
comparison for J1640+2224 and J1022+1001.

Usage:
    JAX_PLATFORMS=cpu python3 tools/damping_diagnostic_plots.py
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

NG_DIR = Path(__file__).parent.parent / "data" / "pulsars" / "NG_data" / "NG_15yr_partim"

PULSARS = {
    "J1640+2224": "J1640+2224_PINT_20220305.nb",
    "J1022+1001": "J1022+1001_PINT_20220304.nb",
}

COMPARE_PARAMS = [
    "F0", "F1", "F2", "DM", "PX", "PB", "A1", "ECC", "OM", "E",
    "EPS1", "EPS2", "SINI", "M2", "PBDOT", "OMDOT", "GAMMA",
    "H3", "STIG", "H4", "KIN", "KOM", "T0", "TASC", "XDOT", "A1DOT",
    "NE_SW", "ELONG", "ELAT", "PMELONG", "PMELAT", "RAJ", "DECJ",
    "PMRA", "PMDEC", "FD1", "FD2", "FD3", "FD4",
    "FB0", "FB1", "FB2", "FB3", "FB4",
]


def run_pint(par, tim):
    """Run PINT WLS fit and return residuals + params."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pint.models import get_model
        from pint.toa import get_TOAs
        from pint.residuals import Residuals
        from pint.fitter import WLSFitter
        import astropy.units as u

        m = get_model(str(par))
        t = get_TOAs(str(tim), model=m)

        # Neutralise EFAC/EQUAD
        for comp in m.components.values():
            for pname in comp.params:
                pu = pname.upper()
                if "EFAC" in pu:
                    getattr(comp, pname).value = 1.0
                elif "EQUAD" in pu:
                    getattr(comp, pname).value = 0.0

        # Prefit residuals
        r_pre = Residuals(t, m)
        prefit_res = r_pre.time_resids.to(u.s).value * 1e6  # μs
        prefit_mjd = t.get_mjds().value

        # Prefit params
        prefit_params = {}
        for p in COMPARE_PARAMS:
            try:
                comp = getattr(m, p, None)
                if comp is not None and comp.value is not None:
                    prefit_params[p] = float(comp.value)
            except Exception:
                pass

        # Fit
        f = WLSFitter(t, m)
        f.fit_toas()

        # Postfit residuals
        r_post = Residuals(t, f.model)
        postfit_res = r_post.time_resids.to(u.s).value * 1e6
        raw_errs = t.get_errors().to(u.s).value * 1e6  # μs
        mjds = prefit_mjd

        # Postfit params
        postfit_params = {}
        for p in COMPARE_PARAMS:
            try:
                comp = getattr(f.model, p, None)
                if comp is not None and comp.value is not None:
                    postfit_params[p] = float(comp.value)
            except Exception:
                pass

    return {
        "prefit_res": prefit_res,
        "postfit_res": postfit_res,
        "errs": raw_errs,
        "mjds": mjds,
        "prefit_params": prefit_params,
        "postfit_params": postfit_params,
    }


def run_jug(par, tim):
    """Run JUG fit and return residuals + params."""
    from jug.engine import open_session
    from jug.engine.noise_mode import NoiseConfig

    # Import extract_free_params from the comparison script
    sys.path.insert(0, str(Path(__file__).parent))
    from ng_parity_comparison import extract_free_params

    s = open_session(str(par), str(tim), verbose=False)

    # Prefit
    r_pre = s.compute_residuals(subtract_tzr=True)
    prefit_res = np.array(r_pre["residuals_us"])
    prefit_mjd = np.array(r_pre["tdb_mjd"])

    # Prefit params
    prefit_params = {}
    for p in COMPARE_PARAMS:
        if p in s.params:
            try:
                prefit_params[p] = float(s.params[p])
            except (ValueError, TypeError):
                pass

    # Fit
    free_params = extract_free_params(par)
    nc = NoiseConfig(enabled={
        "EFAC": False, "EQUAD": False,
        "ECORR": False, "RedNoise": False, "DMNoise": False, "DMX": True,
    })
    result = s.fit_parameters(fit_params=free_params, max_iter=1, noise_config=nc)

    postfit_res = np.array(result["residuals_us"])
    errs = np.array(result["errors_us"])
    mjds = np.array(result["tdb_mjd"])

    postfit_params = {}
    for k, v in result.get("final_params", {}).items():
        if k in COMPARE_PARAMS:
            postfit_params[k] = float(v)

    return {
        "prefit_res": prefit_res,
        "postfit_res": postfit_res,
        "errs": errs,
        "mjds": mjds,
        "prefit_params": prefit_params,
        "postfit_params": postfit_params,
    }


def compute_wrms(res, errs):
    """Weighted RMS in same units as res."""
    w = 1.0 / errs**2
    wmean = np.sum(w * res) / np.sum(w)
    return np.sqrt(np.sum(w * (res - wmean) ** 2) / np.sum(w))


def compute_chi2(res, errs):
    """Chi-squared."""
    return np.sum((res / errs) ** 2)


def make_plots(name, pint_data, jug_data, outdir):
    """Create diagnostic plot for one pulsar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # --- Row 1: Postfit residuals vs MJD ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    pint_wrms = compute_wrms(pint_data["postfit_res"], pint_data["errs"])
    jug_wrms = compute_wrms(jug_data["postfit_res"], jug_data["errs"])
    pint_chi2 = compute_chi2(pint_data["postfit_res"], pint_data["errs"])
    jug_chi2 = compute_chi2(jug_data["postfit_res"], jug_data["errs"])
    n_toas = len(pint_data["postfit_res"])
    pint_rchi2 = pint_chi2 / n_toas
    jug_rchi2 = jug_chi2 / n_toas

    ax1.errorbar(
        pint_data["mjds"], pint_data["postfit_res"], yerr=pint_data["errs"],
        fmt=".", ms=2, alpha=0.4, color="C0", elinewidth=0.5,
    )
    ax1.axhline(0, color="k", ls="--", lw=0.5)
    ax1.set_title(f"PINT Postfit — {name}")
    ax1.set_xlabel("MJD")
    ax1.set_ylabel("Residual (μs)")
    ax1.text(
        0.02, 0.98,
        f"WRMS = {pint_wrms:.3f} μs\n"
        f"χ² = {pint_chi2:.1f}\n"
        f"χ²/N = {pint_rchi2:.3f}\n"
        f"N = {n_toas}",
        transform=ax1.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax2.errorbar(
        jug_data["mjds"], jug_data["postfit_res"], yerr=jug_data["errs"],
        fmt=".", ms=2, alpha=0.4, color="C1", elinewidth=0.5,
    )
    ax2.axhline(0, color="k", ls="--", lw=0.5)
    ax2.set_title(f"JUG Postfit — {name}")
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("Residual (μs)")
    ax2.text(
        0.02, 0.98,
        f"WRMS = {jug_wrms:.3f} μs\n"
        f"χ² = {jug_chi2:.1f}\n"
        f"χ²/N = {jug_rchi2:.3f}\n"
        f"N = {n_toas}\n"
        f"Ratio JUG/PINT = {jug_wrms / pint_wrms:.3f}",
        transform=ax2.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # Use same y-limits for both panels
    ymax = max(
        np.max(np.abs(pint_data["postfit_res"])),
        np.max(np.abs(jug_data["postfit_res"])),
    )
    for ax in [ax1, ax2]:
        ax.set_ylim(-1.3 * ymax, 1.3 * ymax)

    # --- Row 2: Residual histograms + parameter step comparison ---
    ax3 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-1.3 * ymax, 1.3 * ymax, 60)
    ax3.hist(
        pint_data["postfit_res"], bins=bins, alpha=0.6, label=f"PINT (WRMS={pint_wrms:.3f})",
        color="C0", density=True,
    )
    ax3.hist(
        jug_data["postfit_res"], bins=bins, alpha=0.6, label=f"JUG (WRMS={jug_wrms:.3f})",
        color="C1", density=True,
    )
    ax3.set_xlabel("Residual (μs)")
    ax3.set_ylabel("Density")
    ax3.set_title("Postfit Residual Distribution")
    ax3.legend(fontsize=9)

    # Parameter step comparison
    ax4 = fig.add_subplot(gs[1, 1])
    common_params = sorted(
        set(pint_data["prefit_params"]) & set(pint_data["postfit_params"])
        & set(jug_data["prefit_params"]) & set(jug_data["postfit_params"])
    )
    if common_params:
        pint_steps = []
        jug_steps = []
        labels = []
        for p in common_params:
            p_pre = pint_data["prefit_params"][p]
            if abs(p_pre) < 1e-30:
                continue
            pint_step = (pint_data["postfit_params"][p] - p_pre) / abs(p_pre)
            jug_step = (jug_data["postfit_params"][p] - jug_data["prefit_params"][p]) / abs(p_pre)
            if abs(pint_step) < 1e-10 and abs(jug_step) < 1e-10:
                continue
            pint_steps.append(pint_step)
            jug_steps.append(jug_step)
            labels.append(p)

        if labels:
            x = np.arange(len(labels))
            w = 0.35
            ax4.barh(x - w / 2, pint_steps, w, label="PINT step", color="C0", alpha=0.7)
            ax4.barh(x + w / 2, jug_steps, w, label="JUG step", color="C1", alpha=0.7)
            ax4.set_yticks(x)
            ax4.set_yticklabels(labels, fontsize=8)
            ax4.set_xlabel("Fractional param change (Δp/|p_prefit|)")
            ax4.set_title("Parameter Step Sizes")
            ax4.legend(fontsize=9)
            ax4.axvline(0, color="k", ls="--", lw=0.5)

    # --- Row 3: Prefit vs postfit WRMS comparison + summary text ---
    ax5 = fig.add_subplot(gs[2, 0])
    prefit_pint_wrms = compute_wrms(pint_data["prefit_res"], pint_data["errs"])
    prefit_jug_wrms = compute_wrms(jug_data["prefit_res"], jug_data["errs"])

    categories = ["Prefit\nPINT", "Prefit\nJUG", "Postfit\nPINT", "Postfit\nJUG"]
    values = [prefit_pint_wrms, prefit_jug_wrms, pint_wrms, jug_wrms]
    colors = ["C0", "C1", "C0", "C1"]
    bars = ax5.bar(categories, values, color=colors, alpha=0.7,
                   edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax5.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax5.set_ylabel("WRMS (μs)")
    ax5.set_title("WRMS Comparison: Pre- vs Post-fit")

    # Summary text
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    better = "JUG" if jug_wrms < pint_wrms else "PINT"
    worse = "PINT" if better == "JUG" else "JUG"
    improvement = abs(pint_wrms - jug_wrms) / max(pint_wrms, jug_wrms) * 100

    summary = (
        f"Pulsar: {name}\n"
        f"N TOAs: {n_toas}\n"
        f"─────────────────────────────\n"
        f"Prefit  WRMS: PINT={prefit_pint_wrms:.4f} μs, JUG={prefit_jug_wrms:.4f} μs\n"
        f"Postfit WRMS: PINT={pint_wrms:.4f} μs, JUG={jug_wrms:.4f} μs\n"
        f"─────────────────────────────\n"
        f"Postfit ratio (JUG/PINT): {jug_wrms / pint_wrms:.4f}\n"
        f"{better} achieves {improvement:.1f}% lower WRMS\n"
        f"─────────────────────────────\n"
        f"χ²/N: PINT={pint_rchi2:.4f}, JUG={jug_rchi2:.4f}\n"
    )
    if jug_wrms < pint_wrms:
        summary += (
            f"\n✓ JUG's damped step prevents WLS overshoot\n"
            f"  PINT takes full undamped step → WRMS degrades\n"
            f"  JUG correctly identifies smaller step is better"
        )
    ax6.text(
        0.05, 0.95, summary, transform=ax6.transAxes, va="top",
        fontsize=10, fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        f"JUG vs PINT Damping Diagnostic — {name}",
        fontsize=14, fontweight="bold",
    )
    outpath = outdir / f"damping_diagnostic_{name}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


def main():
    outdir = Path(__file__).parent.parent / "docs" / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, stem in PULSARS.items():
        par = NG_DIR / f"{stem}.par"
        tim = NG_DIR / f"{stem}.tim"
        if not par.exists() or not tim.exists():
            print(f"SKIP {name}: files not found ({par})")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {name}...")
        print(f"{'='*60}")

        print("  Running PINT...")
        pint_data = run_pint(par, tim)
        print(f"  PINT postfit WRMS: {compute_wrms(pint_data['postfit_res'], pint_data['errs']):.4f} μs")

        print("  Running JUG...")
        jug_data = run_jug(par, tim)
        print(f"  JUG  postfit WRMS: {compute_wrms(jug_data['postfit_res'], jug_data['errs']):.4f} μs")

        make_plots(name, pint_data, jug_data, outdir)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
