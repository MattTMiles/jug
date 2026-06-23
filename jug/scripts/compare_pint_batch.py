"""Batch JUG-vs-PINT comparison: one comprehensive diagnostic PDF per pulsar.

Point at a directory of par/tim files; for each pair this fits both JUG and
PINT and writes <JNAME>_jug_vs_pint.pdf reproducing the diagnostic content of
notebooks/jug_vs_pint_noise_comparison.ipynb:

  p1  summary & audit: WRMS/chi2, prefit/postfit diff, TDB parity, weights,
      phase-wrap check, binary verdict, signals, worst parameter offsets
  p2  prefit & postfit residuals + JUG-PINT diffs vs MJD
  p3  postfit diff distribution: histogram, QQ, Lorenz CDF, diff vs error
  p4  diff drivers: vs TOA error, vs frequency, vs MJD colored by freq,
      barycentric frequency parity
  p5  delay-component deltas (tdb, geo+shap, DM, DMX, SW, tropo, FD, signal,
      prebinary, post_prebinary)
  p6  residual-diff attribution by (observatory, backend) group
  p7  long-timescale drift + Lomb-Scargle periodogram of the diff
  p8  stability: Allan-like deviation, lag ACF, residual periodogram overlay
  p9  parameter comparison: (JUG-PINT)/sigma and sigma ratios
  p10 correlated-noise component RMS + canonical marginalized GLS objective
  p11 correlated-noise realization overlays and differences
  p12 red-noise basis & fitter-prior comparison (when RN/TNRed present)
  p13 covariance comparison: sigma ratios + largest correlation differences
  p14 DMX gauge/non-gauge delay-impact comparison (when DMX present)
  p15 binary diagnostic (2d4 fixed algebra: own-fit / expected / corrected)
  p16 speed benchmark (end-to-end and warm fit-only, with baseline regression
      tracking in <out>/.jug_speed_baseline.json) [--speed-repeats 0 to skip]
  p17 Tempo2 cross-check: postfit overlay + noise components (needs tempo2)
  p18 worst-TOA explainer with per-component breakdown

Deterministic signals PINT cannot model (jug.signals registry: chromatic
events, CW, burst memory) are detected from the par file, evaluated with
JUG's waveform at the par values (BARYCENTRIC frequencies, matching JUG
core), and injected into PINT as a FROZEN per-TOA delay component before the
fit. Residual-level subtraction would be too late (the fit would already
have absorbed the signal into DM/DMX/spin); adjust_TOAs is the fallback.

Usage:
    python -m jug.scripts.compare_pint_batch DATA_DIR \
        [--out /home/mattm/projects/jug_test_files/large_tests] \
        [--maxiter 5] [--clock-dir JUG/data/clock] [--signal-mode component]
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

JUG_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# par/tim pairing
# ---------------------------------------------------------------------------

def find_pairs(data_dir: Path):
    pairs = []
    tims = sorted(data_dir.glob("*.tim"))
    for par in sorted(data_dir.glob("*.par")):
        stem = par.name[: -len(".par")]
        cand = data_dir / (stem + ".tim")
        tim = cand if cand.exists() else None
        if tim is None:
            jname = re.split(r"[._]", stem)[0]
            matches = [t for t in tims if t.name.startswith(jname)]
            if len(matches) == 1:
                tim = matches[0]
        if tim is None:
            print(f"  SKIP {par.name}: no matching tim file")
            continue
        pairs.append((re.split(r"[._]", stem)[0], par, tim))
    return pairs


# ---------------------------------------------------------------------------
# PINT parity environment (kept in sync with vetted notebook)
# ---------------------------------------------------------------------------

def _first_float_token(line):
    parts = line.split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _read_two_column_clock(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
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


def _write_pint_tempo_clock_from_two_column(src, dst):
    rows = _read_two_column_clock(src)
    lines = [
        "# Synthetic PINT TEMPO clock generated from JUG clock data\n",
        "# Keeps PINT observatory clocks on the same chain as JUG.\n",
        "   MJD       OBS-REF     GPS-REF NS      DATE    COMMENTS\n",
        "=========    ========    ======== ==    ========  ========\n",
    ]
    for mjd, offset_sec in rows:
        lines.append(
            f"{mjd:9.2f}{0.0:12.3f}{offset_sec * 1e6:12.3f} 3 f  "
            f"synthetic from {Path(src).name}\n"
        )
    Path(dst).write_text("".join(lines))


def prepare_pint_environment(clock_dir, pint_clock_dir):
    """Apply the same PINT clock conventions used by the vetted notebook."""
    src_dir = Path(clock_dir).resolve()
    dst_dir = Path(pint_clock_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    changed = []

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
        inverted = any(
            numeric[i + 1][0] < numeric[i][0]
            for i in range(len(numeric) - 1)
        )
        if inverted:
            numeric.sort(key=lambda item: (item[0], item[1]))
            dst.write_text("".join(nonnumeric + [line for _, _, line in numeric]))
            changed.append(src.name)
        else:
            dst.symlink_to(src)

    for clock_name, tempo_name in (
        ("ao2gps.clk", "time_ao.dat"),
        ("gbt2gps.clk", "time_gbt.dat"),
    ):
        clock_src = src_dir / clock_name
        if not clock_src.exists():
            continue
        tempo_dst = dst_dir / tempo_name
        if tempo_dst.exists() or tempo_dst.is_symlink():
            tempo_dst.unlink()
        _write_pint_tempo_clock_from_two_column(clock_src, tempo_dst)
        changed.append(f"{tempo_name}<-{clock_name}")

    os.environ["PINT_CLOCK_OVERRIDE"] = str(dst_dir)

    # PINT caches clock objects globally. Clear them so every batch run sees
    # the override even when another PINT workload ran earlier in the process.
    import pint.observatory as pint_observatory

    pint_observatory._gps_clock = None
    pint_observatory._bipm_clock_versions.clear()
    for name in pint_observatory.Observatory.names():
        obs = pint_observatory.get_observatory(name)
        if hasattr(obs, "_clock"):
            obs._clock = None

    # Match Tempo2/JUG's MeerKAT chain: mk2utc.clk already reaches UTC, so
    # applying gps2utc again is a double correction.
    try:
        meerkat = pint_observatory.get_observatory("meerkat")
        if getattr(meerkat, "clock_files", None) and "mk2utc.clk" not in meerkat.clock_files:
            meerkat.clock_files = ["mk2utc.clk"]
        if getattr(meerkat, "apply_gps2utc", False):
            meerkat.apply_gps2utc = False
        meerkat._clock = None
    except Exception:
        pass
    return changed


def _par_uses_dilatefreq(par_path):
    for line in Path(par_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "DILATEFREQ":
            return parts[1].upper() in {"Y", "T", "1", "TRUE"}
    return False


def prepare_pint_model(model, par_path):
    """Apply notebook-compatible model conventions before loading TOAs."""
    if not _par_uses_dilatefreq(par_path):
        return model
    from jug.delays.barycentric import compute_einstein_rate

    base = model.barycentric_radio_freq

    def barycentric_radio_freq_dilated(toas):
        freq = base(toas)
        rate = compute_einstein_rate(
            np.asarray(toas.table["tdbld"], dtype=float), units="TDB"
        )
        return freq / rate

    model.barycentric_radio_freq = barycentric_radio_freq_dilated
    return model


def prepare_pint_toas(model, toas):
    """Repair PINT mjd_float precision and freeze empty masks like notebook."""
    import astropy.units as u

    corrected = np.asarray([float(t.value) for t in toas.table["mjd"]], dtype=float)
    toas.table["mjd_float"] = corrected * u.d
    model.find_empty_masks(toas, freeze=True)
    return toas


# ---------------------------------------------------------------------------
# Deterministic signals PINT cannot model
# ---------------------------------------------------------------------------

def detect_jug_signals(par_path: Path):
    from jug.io.par_reader import parse_par_file
    from jug.signals.base import SIGNAL_REGISTRY, detect_signals

    params = parse_par_file(str(par_path))
    signals = detect_signals(params)
    # Strip prefix per signal = longest common prefix of its par keys. The old
    # `key.split("_")[0] + "_"` produced e.g. "CHROMBUMPAMP_" for the
    # underscore-less ChromBump/ChromAnnual keys, which matched NOTHING -- so
    # those dummy params leaked into PINT, which instantiates ChromBump/
    # ChromAnnual as NoiseComponents lacking get_noise_weights and crashes the
    # GLS fit. commonprefix gives "CHROMBUMP"/"CHROMANNUAL" and strips them all.
    prefixes = set()
    for cls in SIGNAL_REGISTRY.values():
        keys = [k.upper() for k in cls.required_par_keys()]
        if not keys:
            continue
        # Underscore-style keys (CW_LOG10_H, BWM_T0, CHROMEV_AMP): the signal
        # owns the whole first segment, so strip "<seg>_". Underscore-less keys
        # (CHROMBUMPAMP, CHROMANNUALPHASE): use the longest common prefix
        # (CHROMBUMP / CHROMANNUAL). The old split("_")[0]+"_" wrongly made
        # "CHROMBUMPAMP_" for the latter, matching nothing -> dummy params
        # leaked into PINT -> get_noise_weights crash.
        if "_" in keys[0]:
            prefixes.add(keys[0].split("_")[0] + "_")
        else:
            prefixes.add(os.path.commonprefix(keys))
    return signals, prefixes


def write_pint_safe_par(par_path: Path, strip_prefixes, workdir: Path) -> Path:
    par_path, workdir = Path(par_path), Path(workdir)
    out = workdir / (par_path.stem + ".pint.par")
    kept = []
    for line in par_path.read_text().splitlines():
        token = line.split()[0].upper() if line.split() else ""
        if any(token.startswith(p) for p in strip_prefixes):
            continue
        kept.append(line)
    out.write_text("\n".join(kept) + "\n")
    return out


def load_pint_model_retry(par_path: Path, workdir: Path, max_strips: int = 25):
    import pint.models

    path = par_path
    stripped = []
    for _ in range(max_strips):
        try:
            return (pint.models.get_model(str(path), allow_T2=True,
                                          allow_tcb=True), stripped, path)
        except Exception as e:
            m = (re.search(r"Unrecognized parfile line ['\"]?(\w+)", str(e))
                 or re.search(r"parameter ['\"]?(\w+)", str(e)))
            if m is None:
                raise
            bad = m.group(1)
            stripped.append(bad)
            lines = [l for l in path.read_text().splitlines()
                     if not (l.split() and l.split()[0].upper() == bad.upper())]
            path = workdir / (par_path.stem + ".strip.par")
            path.write_text("\n".join(lines) + "\n")
    raise RuntimeError(f"could not load {par_path} into PINT after {max_strips} strips")


def inject_frozen_delay_component(model, toas, delay_sec):
    """JUG-evaluated signal waveform -> frozen PINT delay component."""
    import astropy.units as u
    from pint.models.timing_model import DelayComponent

    idx = np.asarray(toas.table["index"], dtype=int)
    by_index = np.zeros(int(idx.max()) + 1, dtype=float)
    by_index[idx] = np.asarray(delay_sec, dtype=float)

    class JUGFrozenSignalDelay(DelayComponent):
        register = False
        category = "frequency_dependent"  # post-binary, like FD

        def __init__(self):
            super().__init__()
            self.delay_funcs_component += [self.jug_frozen_signal_delay]

        def jug_frozen_signal_delay(self, toas_arg, acc_delay=None):
            i = np.asarray(toas_arg.table["index"], dtype=int)
            out = np.zeros(len(i), dtype=float)
            valid = (i >= 0) & (i < len(by_index))
            out[valid] = by_index[i[valid]]
            return out * u.s

    try:
        model.add_component(JUGFrozenSignalDelay(), validate=False)
        model.validate()
        model.delay(toas)
        return True
    except Exception as e:
        print(f"    frozen-component injection failed ({e}); "
              f"falling back to adjust_TOAs")
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wrms(x, w):
    x = x - np.average(x, weights=w)
    return float(np.sqrt(np.sum(w * x ** 2) / np.sum(w)))


def _center(x):
    x = np.asarray(x, dtype=float)
    return x - np.mean(x)


# ---------------------------------------------------------------------------
# Convergence measure
# ---------------------------------------------------------------------------
# "How converged is the postfit solution?" = the size of the NEXT fit step in
# units of each parameter's formal uncertainty: max_i |dp_i| / sigma_i, from
# ONE extra iteration seeded AT the plotted solution. Scale-free, model-
# agnostic, identical definition for both codes. A fit sitting at the true
# minimum does not move (<< 1); a fit that quit early on a flat objective /
# degenerate valley still moves. Lower = more converged.

def _pint_next_step_sigma(toas, model):
    """Max next-step / sigma for PINT, seeded at the converged model."""
    import copy
    import pint.fitter
    try:
        f = pint.fitter.GLSFitter(toas, copy.deepcopy(model))
        before = {p: getattr(f.model, p).value for p in f.model.free_params}
        f.fit_toas(maxiter=1)
        smax, worst = 0.0, None
        for p in f.model.free_params:
            par = getattr(f.model, p)
            unc = getattr(par, "uncertainty_value", None)
            bv = before.get(p)
            if unc and np.isfinite(unc) and unc > 0 and bv is not None:
                s = abs(par.value - bv) / unc
                if s > smax:
                    smax, worst = s, p
        return smax, worst
    except Exception as exc:  # pragma: no cover - diagnostic only
        return np.nan, f"err: {exc}"


def _jug_next_step_sigma(session, fit_result):
    """Max next-step / sigma for JUG, continuing from the converged session.

    session.fit_parameters() writes the fitted values back into the session,
    so a 1-iteration continuation measures movement from the postfit solution.
    Mutates the session by one sub-sigma step (negligible for the plots, which
    use already-captured residual arrays)."""
    try:
        before = dict(fit_result.get("final_params", {}))
        unc0 = dict(fit_result.get("uncertainties", {}) or {})
        r2 = session.fit_parameters(max_iter=1)
        after = r2.get("final_params", {})
        unc = r2.get("uncertainties", {}) or unc0
        smax, worst = 0.0, None
        for p, bv in before.items():
            if p not in after:
                continue
            try:
                bvf, avf = float(bv), float(after[p])
                uf = float(unc.get(p, unc0.get(p)))
            except (TypeError, ValueError):
                continue  # string-valued params (RAJ/DECJ) — skip
            if np.isfinite(uf) and uf > 0:
                s = abs(avf - bvf) / uf
                if s > smax:
                    smax, worst = s, p
        return smax, worst
    except Exception as exc:  # pragma: no cover - diagnostic only
        return np.nan, f"err: {exc}"


def _remove_spin_gauge(diff_us, mjd, errors_us):
    """Project constant/F0/F1 gauge terms out of a residual difference."""
    y = np.asarray(diff_us, dtype=float)
    t = np.asarray(mjd, dtype=float)
    err = np.asarray(errors_us, dtype=float)
    x = t - np.mean(t)
    scale = np.max(np.abs(x))
    if not np.isfinite(scale) or scale == 0.0:
        return _center(y)
    x /= scale
    A = np.column_stack((np.ones_like(x), x, x * x))
    sw = np.where(np.isfinite(err) & (err > 0.0), 1.0 / err, 0.0)
    coeff, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return y - A @ coeff


def _pint_comp_sec(model, toas, component, func):
    comp = model.components.get(component)
    if comp is None:
        return np.zeros(toas.ntoas)
    try:
        return getattr(comp, func)(toas).to("s").value
    except Exception:
        return np.zeros(toas.ntoas)


def _jug_full_gls_chi2(session, residuals_us, errors_us, comps=None):
    """Canonical r^T C^-1 r using JUG's correlated-noise basis and prior.

    comps (optional): the session's residual-component dict (D["comps"]).
    When given, the chromatic/DM/band/SW bases use BARYCENTRIC freq (matching
    the fitter, optimized_fitter.py), and the GW + SW components are included.
    Without it (legacy) the DM/chrom bases fall back to topocentric freq and
    GW/SW are omitted -- which UNDERCOUNTS JUG's noise (the fitter models them).
    """
    from jug.noise.ecorr import build_ecorr_basis_and_prior
    from jug.noise.red_noise import (
        parse_band_noise_params,
        parse_chromatic_noise_params,
        parse_dm_noise_params,
        parse_group_noise_params,
        parse_gw_noise_params,
        parse_red_noise_params,
        parse_sw_noise_params,
    )
    from jug.noise.white import parse_noise_lines

    mjd = np.array([t.mjd_int + t.mjd_frac for t in session.toas_data], dtype=float)
    freq_topo = np.array([t.freq_mhz for t in session.toas_data], dtype=float)
    # The fitter builds the dispersive bases with BARYCENTRIC freq; use it when
    # the comps dict is available so this diagnostic matches the actual fit.
    freq = (np.asarray(comps.get("freq_bary_mhz"), dtype=float)
            if comps is not None and comps.get("freq_bary_mhz") is not None
            else freq_topo)
    sw_geom = comps.get("sw_geometry_pc") if comps is not None else None
    flags = [t.flags for t in session.toas_data]
    bases, priors = [], []

    proc = parse_red_noise_params(session.params)
    if proc is not None:
        F, phi = proc.build_basis_and_prior(mjd)
        bases.append(F); priors.append(phi)
    proc = parse_dm_noise_params(session.params)
    if proc is not None:
        F, phi = proc.build_basis_and_prior(mjd, freq)
        bases.append(F); priors.append(phi)
    proc = parse_chromatic_noise_params(session.params)
    if proc is not None:
        F, phi = proc.build_basis_and_prior(mjd, freq)
        bases.append(F); priors.append(phi)
    for proc in parse_band_noise_params(session.params):
        F, phi = proc.build_basis_and_prior(mjd, freq)
        bases.append(F); priors.append(phi)
    # GW background (achromatic red process) -- omitted before; the fitter
    # models it (-> GWNoise realisation), so its absence here inflated chi2.
    proc = parse_gw_noise_params(session.params)
    if proc is not None:
        F, phi = proc.build_basis_and_prior(mjd)
        bases.append(F); priors.append(phi)
    # Stochastic solar-wind noise (chromatic, SW geometry) -- needs per-TOA
    # sw_geometry_pc from comps; skipped if unavailable.
    proc = parse_sw_noise_params(session.params)
    if proc is not None and sw_geom is not None:
        F, phi = proc.build_basis_and_prior(mjd, freq, np.asarray(sw_geom, dtype=float))
        bases.append(F); priors.append(phi)
    group_flags = np.array([f.get("group", "") for f in flags])
    for proc in parse_group_noise_params(session.params):
        F, phi = proc.build_basis_and_prior(mjd, group_flags=group_flags)
        bases.append(F); priors.append(phi)
    noise_lines = session.params.get("_noise_lines", [])
    if noise_lines:
        ecorr = build_ecorr_basis_and_prior(
            mjd, flags, parse_noise_lines(noise_lines)
        )
        if ecorr is not None:
            F, phi = ecorr
            bases.append(F); priors.append(phi)

    # Profile constant phase offset exactly as PINT's GLS objective does.
    bases.append(np.ones((len(mjd), 1), dtype=float))
    priors.append(np.array([1e40], dtype=float))
    F = np.hstack(bases)
    phi = np.concatenate(priors)
    r = np.asarray(residuals_us, dtype=float) * 1e-6
    sigma = np.asarray(errors_us, dtype=float) * 1e-6
    Ni = 1.0 / sigma**2
    FtNi = F.T * Ni
    rhs = FtNi @ r
    A = FtNi @ F
    A[np.diag_indices_from(A)] += 1.0 / np.maximum(phi, 1e-300)
    return float(r @ (r * Ni) - rhs @ np.linalg.solve(A, rhs))


def _noise_component_rows(jug_nr, pint_nr):
    aliases = (
        ("Red noise", "RedNoise", "pl_red_noise"),
        ("ECORR", "ECORR", "ecorr_noise"),
        ("DM noise", "DMNoise", "pl_DM_noise"),
        ("Chromatic noise", "ChromaticNoise", "pl_chrom_noise"),
        ("GW noise", "GWNoise", "pl_gw_noise"),
        ("SW noise", "SWNoise", "pl_SW_noise"),
    )
    rows = []
    for label, jkey, pkey in aliases:
        if jkey not in jug_nr or pkey not in pint_nr:
            continue
        j = np.asarray(jug_nr[jkey], dtype=float)
        p = np.asarray(pint_nr[pkey], dtype=float)
        if j.shape == p.shape:
            rows.append((label, jkey, pkey, j, p))
    return rows


def _safe_page(pdf, title, fn, *args, **kwargs):
    """Render one page; on failure emit a page with the traceback."""
    try:
        fn(*args, **kwargs)
    except Exception:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, f"PAGE FAILED: {title}", fontsize=12,
                 va="top", color="red", family="monospace")
        fig.text(0.05, 0.90, traceback.format_exc(), fontsize=6,
                 va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# data gathering
# ---------------------------------------------------------------------------

def gather(jname, par, tim, maxiter, clock_dir, pint_clock_dir, signal_mode,
           convergence=False, pint_maxiter=5):
    """Run both fits and collect everything the pages need into a dict."""
    import astropy.units as u
    import pint.fitter
    import pint.logging
    import pint.toa
    from pint.residuals import Residuals

    from jug.engine.session import TimingSession
    from jug.fitting.binary_registry import compute_binary_delay as jug_binary_delay
    from jug.io.par_reader import get_longdouble
    from jug.model.parameter_spec import canonicalize_param_name

    pint.logging.setup(level="ERROR")
    workdir = Path(tempfile.mkdtemp(prefix=f"jugcmp_{jname}_"))
    D = {"jname": jname, "par": str(par), "tim": str(tim), "notes": []}

    # --- signals ------------------------------------------------------------
    signals, prefixes = detect_jug_signals(par)
    pint_par = write_pint_safe_par(par, prefixes, workdir) if signals else par
    if signals:
        D["notes"].append("JUG signals: " + "; ".join(s.summary() for s in signals))

    # --- PINT ----------------------------------------------------------------
    clock_changes = prepare_pint_environment(clock_dir, pint_clock_dir)
    if clock_changes:
        D["notes"].append("PINT clock parity override: " + ", ".join(clock_changes))
    pint_model, stripped, pint_par_used = load_pint_model_retry(pint_par, workdir)
    pint_model = prepare_pint_model(pint_model, pint_par_used)
    if stripped:
        D["notes"].append(f"par lines stripped for PINT: {stripped}")
    D["pint_par_used"] = str(pint_par_used)
    # Glitch parameters: JUG models glitches in the FORWARD pass (phase is
    # applied) but has no fit DERIVATIVES for them yet -- full glitch-fit support
    # awaits an autodiff path. So FREEZE every glitch param in BOTH codes: the
    # glitch is still applied deterministically at its par value, it just isn't
    # fitted, which keeps the JUG-vs-PINT parameter comparison fair instead of
    # crashing JUG's fitter (e.g. J0900-3144 GLF0_1). The PINT freeze happens
    # here (before PINT's fit below); the JUG freeze happens after the session
    # is built (free_params reads _fit_flags).
    import re as _re_gl
    _GLITCH_RE = _re_gl.compile(r'^GL(EP|PH|F0|F1|F2|F0D|TD)_\d+$')
    _frozen_glitch = []
    for _pn in list(pint_model.params):
        if _GLITCH_RE.match(_pn):
            _po = getattr(pint_model, _pn, None)
            if _po is not None and not getattr(_po, "frozen", True):
                _po.frozen = True
                _frozen_glitch.append(_pn)
    D["_frozen_glitch"] = _frozen_glitch
    pint_toas = pint.toa.get_TOAs(str(tim), model=pint_model)
    prepare_pint_toas(pint_model, pint_toas)

    wave = None
    if signals:
        toas_mjd = np.asarray(pint_toas.table["tdbld"], dtype=float)
        # BARYCENTRIC freqs: must match JUG core (freq_bary_mhz).
        freqs = pint_model.barycentric_radio_freq(pint_toas).to("MHz").value
        wave = np.zeros(pint_toas.ntoas)
        for s in signals:
            wave = wave + np.asarray(s.compute_waveform(toas_mjd, freqs))
        injected = False
        if signal_mode == "component":
            injected = inject_frozen_delay_component(pint_model, pint_toas, wave)
            if injected:
                D["notes"].append("signal injected into PINT as frozen delay component")
        if not injected and signal_mode != "off":
            from astropy.time import TimeDelta
            pint_toas.adjust_TOAs(TimeDelta(-wave * u.s))
            D["notes"].append("signal subtracted from PINT TOAs (adjust_TOAs fallback)")
    D["wave_us"] = wave * 1e6 if wave is not None else None

    pint_prefit_us = Residuals(pint_toas, pint_model,
                               subtract_mean=True).time_resids.to("us").value
    # PINT fit. DEFAULT = plain GLSFitter. It reaches the same minimum as JUG on
    # well-conditioned pulsars (validated: J0437 binary params match JUG exactly)
    # and populates resids.noise_resids NATIVELY/correctly -- so the noise
    # comparison is right. DownhillGLSFitter was tried as default but regressed
    # the common case: its required_chi2_decrease=1e-2 early-stop UNDER-converges
    # high-precision MSPs (J0437 T0/OM off ~42 sigma) and zeroes the noise
    # realization (PINT FIXME), forcing a noise workaround. So Downhill is OPT-IN
    # (env JUG_PINT_FITTER=downhill) for the rare degenerate binary (e.g. J1022)
    # where plain GLS under-converges at a low maxiter.
    #
    # ROBUSTNESS: PINT's DD/DDK binary model hard-raises if a step drives ECC out
    # of [0,1) (flaky on near-circular orbits, e.g. J1933-6211). Retry plain GLS
    # with progressively fewer iterations (fresh fitter each, since a raised
    # fit_toas leaves it dirty); fall back to the unfitted model so the pulsar
    # still produces a page.
    import copy as _copy_fit
    from pint.fitter import MaxiterReached, StepProblem
    _use_downhill = os.environ.get("JUG_PINT_FITTER", "gls").lower() == "downhill"
    fitter = None
    mf = None
    _pint_conv = None
    _pint_fitter_name = "GLSFitter"
    if _use_downhill:
        _pint_fitter_name = "DownhillGLSFitter"
        _PINT_CAP = max(pint_maxiter, 30)  # Downhill early-stops; high cap is cheap
        try:
            _f = pint.fitter.DownhillGLSFitter(pint_toas, _copy_fit.deepcopy(pint_model))
            try:
                _f.fit_toas(maxiter=_PINT_CAP)
                _pint_conv = True
            except MaxiterReached:
                _pint_conv = False
                D["notes"].append(
                    f"PINT DownhillGLS did NOT converge in {_PINT_CAP} iters; best-so-far")
            except StepProblem:
                _pint_conv = False
                D["notes"].append(
                    "PINT DownhillGLS hit StepProblem (param at boundary); best-so-far")
            fitter, mf = _f, _f.model
        except Exception as _de:
            D["notes"].append(
                f"PINT DownhillGLS failed hard ({type(_de).__name__}: {_de}); "
                f"falling back to plain GLSFitter")
            _pint_fitter_name = "GLSFitter"
    if fitter is None:  # plain GLS (default, or Downhill hard-failure fallback)
        for _mi in range(max(pint_maxiter, 1), 0, -1):
            try:
                _f = pint.fitter.GLSFitter(pint_toas, _copy_fit.deepcopy(pint_model))
                _f.fit_toas(maxiter=_mi)
                fitter, mf = _f, _f.model
                if _pint_conv is None:
                    _pint_conv = True  # ran to maxiter (plain GLS has no early-stop)
                if _mi < pint_maxiter:
                    D["notes"].append(
                        f"PINT plain GLS reduced to maxiter={_mi} (ValueError at "
                        f"higher -- e.g. ECC<0 on near-circular binary; flaky, not a JUG bug)")
                break
            except ValueError:
                continue
        if mf is None:
            fitter = pint.fitter.GLSFitter(pint_toas, _copy_fit.deepcopy(pint_model))
            mf = pint_model
            _pint_conv = False
            D["notes"].append(
                "PINT fit FAILED at every maxiter (binary param out of range); "
                "using UNFITTED PINT model -- comparison DEGRADED. JUG unaffected.")
    D["pint_converged"] = _pint_conv
    D["pint_fitter_name"] = _pint_fitter_name
    # NOISE REALIZATION FIX: DownhillGLSFitter does NOT populate the correlated-
    # noise realization (PINT flags it: "FIXME: set up noise residuals"); its
    # resids.noise_resids come out ZERO even when converged (at convergence
    # current_state.xhat -- a STEP -- is ~0 in the noise block). That zeroed every
    # PINT noise overlay. We must NOT re-extract by taking another fit step: on a
    # degenerate binary a single plain-GLS Newton step from Downhill's converged
    # point JUMPS to a different (worse) basin. Instead compute the GP conditional
    # mean at FIXED timing params (Downhill's solution): solve the noise-ONLY GLS
    # system  (F^T N^-1 F + diag(1/phi)) a = F^T N^-1 r , no timing columns, so the
    # timing solution is untouched. r = postfit resids at mf, F = noise basis, phi
    # = noise prior variances. Then noise_resids[cat] = get_noise_basis @ a[cat].
    # NB: Downhill sets noise_ampls to a dict of ZERO arrays (truthy), so do NOT
    # guard on "not noise_ampls" -- always overwrite with the fixed-timing solve.
    if isinstance(fitter, pint.fitter.DownhillGLSFitter):
        try:
            import astropy.units as _u_n
            _r_s = np.asarray(
                Residuals(pint_toas, mf).time_resids.to(_u_n.s).value, dtype=np.float64)
            _F = np.asarray(mf.noise_model_designmatrix(pint_toas), dtype=np.float64)
            _phi = np.asarray(mf.noise_model_basis_weight(pint_toas), dtype=np.float64)
            _Nv = np.asarray(
                mf.scaled_toa_uncertainty(pint_toas).to_value(_u_n.s) ** 2, dtype=np.float64)
            _cinv = 1.0 / _Nv
            _A = _F.T @ (_cinv[:, None] * _F) + np.diag(1.0 / _phi)
            _b = _F.T @ (_cinv * _r_s)
            _a = np.linalg.solve(_A, _b)  # GP amplitudes in seconds (timing fixed)
            _na = {}
            for _cat, (_off, _sz) in mf.noise_model_dimensions(pint_toas).items():
                _na[_cat] = _a[_off:_off + _sz] * _u_n.s
            fitter.resids.noise_ampls = _na
        except Exception as _ne:
            D["notes"].append(
                f"PINT fixed-timing noise extraction failed "
                f"({type(_ne).__name__}: {_ne}); PINT noise overlay may be empty")
    pint_resid = Residuals(pint_toas, mf, subtract_mean=True)
    D["pint_post_us"] = pint_resid.time_resids.to("us").value
    D["pint_pre_us"] = pint_prefit_us
    try:
        D["pint_chi2"] = float(fitter.resids.chi2)
        D["pint_dof"] = int(fitter.resids.dof)
    except Exception:
        D["pint_chi2"], D["pint_dof"] = np.nan, 0
    try:
        D["pint_err_us"] = mf.scaled_toa_uncertainty(pint_toas).to("us").value
    except Exception:
        D["pint_err_us"] = pint_toas.get_errors().to("us").value

    # --- JUG -----------------------------------------------------------------
    session = TimingSession(str(par), str(tim), clock_dir=clock_dir)
    # PINT-parity gauge: PINT's GLSFitter does NOT implement TNsubtractPoly, so
    # it leaves the low-frequency DM/spin power split between the timing
    # polynomial (DM/DM1/DM2, F0/F1) and the noise Fourier coeffs by the prior
    # (the enterprise gauge). Many MPTA pars set `TNsubtractPoly 1`, which makes
    # JUG transfer the noise realization's polynomial into those timing params
    # (the Tempo2/TempoNest gauge) AFTER the GLS solve -- a different, equally
    # valid gauge that shifts DM1/DM2 (and F0/F1) by O(sigma) vs PINT and
    # produces a large apparent postfit JUG-PINT difference (e.g. J0437-4715
    # DM1 +4.2e-5 vs PINT -4.39e-5, 446 ns) even though the marginalized fit is
    # identical. Disable it here so this comparison is apples-to-apples against
    # PINT's gauge. (Production JUG fits still honor the par directive.)
    # Set JUG_KEEP_TNPOLY=1 to HONOR the par's TNsubtractPoly (Tempo2 gauge) and
    # reproduce a notebook-style comparison; default zeroes it for PINT parity.
    if os.environ.get('JUG_KEEP_TNPOLY') != '1':
        for _k in ('TNSUBTRACTPOLY', 'TNsubtractPoly', 'TNSubtractPoly'):
            if _k in session.params:
                session.params[_k] = 0
    jug_pre = session.compute_residuals(subtract_tzr=True)
    D["jug_pre_us"] = np.asarray(jug_pre["residuals_us"], dtype=float)
    # Freeze glitch params in JUG too (mirror of the PINT freeze above):
    # free_params reads _fit_flags, so dropping those flags excludes the glitch
    # params from the fit while the forward model still applies the glitch.
    _ff = session.params.get("_fit_flags")
    if _ff:
        for _gp in [k for k in list(_ff) if _GLITCH_RE.match(k)]:
            _ff.pop(_gp, None)
            if _gp not in D["_frozen_glitch"]:
                D["_frozen_glitch"].append(_gp)
    if D["_frozen_glitch"]:
        D["notes"].append(
            "glitch params frozen in BOTH codes (no JUG glitch-fit derivatives "
            "yet; applied deterministically at par values): "
            + ", ".join(sorted(D["_frozen_glitch"])))
    # fit_dmx defaults to flag-driven (fit each DMX bin iff flagged free in the
    # PAR), which already mirrors PINT, so no special handling is needed here.
    fit_result = session.fit_parameters(max_iter=maxiter)
    comps = session.compute_residuals(subtract_tzr=True, force_recompute=True)
    D["comps"] = comps
    D["jug_post_fresh_us"] = np.asarray(comps["residuals_us"], dtype=float)
    D["err_us"] = np.asarray(comps["errors_us"], dtype=float)
    D["mjd"] = np.asarray(comps["tdb_mjd"], dtype=float)
    D["freq_topo"] = np.array([t.freq_mhz for t in session.toas_data], dtype=float)
    D["w"] = 1.0 / D["err_us"] ** 2
    D["session"] = session
    D["fit_result"] = fit_result if isinstance(fit_result, dict) else {}
    returned = np.asarray(D["fit_result"].get("residuals_us", []), dtype=float)
    fresh = D["jug_post_fresh_us"]
    D["jug_post_us"] = returned if returned.shape == fresh.shape else fresh
    D["returned_fresh_ns"] = (
        float(np.std(_center(returned - fresh))) * 1e3
        if returned.shape == fresh.shape else np.nan
    )
    jug_nr = {
        k: np.asarray(v, dtype=float)
        for k, v in D["fit_result"].get("noise_realizations", {}).items()
        if not k.endswith("_err")
    }
    pint_nr = {}
    try:
        pint_nr = {
            k: np.asarray(v.to("us").value, dtype=float)
            for k, v in fitter.resids.noise_resids.items()
        }
    except Exception:
        pass
    D["noise_rows"] = _noise_component_rows(jug_nr, pint_nr)
    # Flag correlated-noise components PINT modelled that JUG did not reproduce
    # (e.g. GW noise TNGW*, stochastic solar-wind PLSWNoise SW*, ChromBump).
    # JUG silently ignores unrecognised noise keywords, so without this note the
    # large JUG-vs-PINT differences on such pulsars look like bugs rather than a
    # missing-model mismatch.
    _pint_noise_label = {
        "pl_red_noise": "RedNoise", "pl_DM_noise": "DMNoise",
        "pl_chrom_noise": "ChromaticNoise", "ecorr_noise": "ECORR",
        "pl_gw_noise": "GWNoise (TNGW*)", "pl_SW_noise": "SWNoise (PLSWNoise)",
    }
    _jug_present = set(jug_nr)
    _missing = [
        _pint_noise_label.get(pk, pk)
        for pk in pint_nr
        if _pint_noise_label.get(pk, pk).split()[0] not in _jug_present
        and pk != "ecorr_noise"  # ECORR handled separately/whitener
    ]
    if _missing:
        D["notes"].append(
            "JUG did NOT model noise PINT has: " + ", ".join(sorted(_missing))
            + " -- JUG-vs-PINT differences expected (incomplete noise model)."
        )
    D["jug_noise_total_us"] = sum(
        (v for k, v in jug_nr.items() if k not in {"DMX", "DMJUMP"}),
        np.zeros(len(D["jug_post_us"])),
    )
    D["pint_noise_total_us"] = sum(
        pint_nr.values(), np.zeros(len(D["pint_post_us"]))
    )
    try:
        D["jug_gls_chi2"] = _jug_full_gls_chi2(
            session, D["jug_post_us"], D["fit_result"].get("errors_us", D["err_us"]),
            comps=D.get("comps"),
        )
    except Exception as exc:
        D["jug_gls_chi2"] = np.nan
        D["notes"].append(f"JUG full-GLS chi2 failed: {exc}")
    try:
        D["pint_gls_chi2"] = float(fitter.resids.calc_chi2())
    except Exception:
        D["pint_gls_chi2"] = np.nan
    D["F0"] = float(session.params.get("F0", 1.0))
    D["fitter"] = fitter
    D["mf"] = mf
    D["pint_toas"] = pint_toas

    if len(D["jug_post_us"]) != pint_toas.ntoas:
        raise RuntimeError(f"TOA count mismatch JUG={len(D['jug_post_us'])} "
                           f"PINT={pint_toas.ntoas}")

    D["diff_us"] = _center(D["jug_post_us"] - D["pint_post_us"])
    D["diff_spin_clean_us"] = _remove_spin_gauge(
        D["diff_us"], D["mjd"], D["pint_err_us"]
    )
    D["diff_pre_us"] = _center(D["jug_pre_us"] - D["pint_pre_us"])

    # --- convergence: next-step / sigma for each code (lower = more converged) --
    # OPT-IN (--convergence): each measure does a near-full extra fit (PINT
    # GLSFitter + JUG continuation), ~doubling per-pulsar cost, so it is off by
    # default. WRMS + GLS chi2 (cheap) are always shown.
    if convergence:
        D["pint_step_sigma"], D["pint_step_worst"] = _pint_next_step_sigma(pint_toas, mf)
        # JUG continuation mutates the session by a sub-sigma step; run last,
        # after all residual arrays above are captured.
        D["jug_step_sigma"], D["jug_step_worst"] = _jug_next_step_sigma(
            session, D["fit_result"]
        )
    else:
        D["pint_step_sigma"], D["pint_step_worst"] = np.nan, None
        D["jug_step_sigma"], D["jug_step_worst"] = np.nan, None

    # --- TDB / bary-freq parity ------------------------------------------------
    ptdb = np.asarray(pint_toas.table["tdbld"], dtype=np.longdouble)
    jtdb = np.asarray(comps.get("tdb_mjd_ld", comps["tdb_mjd"]), dtype=np.longdouble)
    D["tdb_diff_ns"] = np.asarray((jtdb - ptdb) * np.longdouble(86400.0),
                                  dtype=float) * 1e9
    pfb = mf.barycentric_radio_freq(pint_toas).to("MHz").value
    jfb = np.asarray(comps["freq_bary_mhz"], dtype=float)
    D["freq_rel_ppb"] = (jfb - pfb) / pfb * 1e9

    # --- component deltas --------------------------------------------------------
    geo = (_pint_comp_sec(mf, pint_toas, "AstrometryEquatorial", "solar_system_geometric_delay")
           + _pint_comp_sec(mf, pint_toas, "AstrometryEcliptic", "solar_system_geometric_delay")
           + _pint_comp_sec(mf, pint_toas, "SolarSystemShapiro", "solar_system_shapiro_delay"))
    pdm = _pint_comp_sec(mf, pint_toas, "DispersionDM", "constant_dispersion_delay")
    pdmx = _pint_comp_sec(mf, pint_toas, "DispersionDMX", "DMX_dispersion_delay")
    psw = _pint_comp_sec(mf, pint_toas, "SolarWindDispersion", "solar_wind_delay")
    ptr = _pint_comp_sec(mf, pint_toas, "TroposphereDelay", "troposphere_delay")
    pfd = _pint_comp_sec(mf, pint_toas, "FD", "FD_delay")
    ptot = mf.delay(pint_toas).to("s").value
    ppre = geo + pdm + pdmx + psw + ptr

    z = np.zeros(len(D["mjd"]))
    jtot = np.asarray(comps["total_delay_sec"], dtype=float)
    jpre = np.asarray(comps["prebinary_delay_sec"], dtype=float)
    logf = np.log(jfb / 1000.0)
    jfd = np.zeros(len(jtot))
    i = 1
    while f"FD{i}" in session.params:
        jfd += float(session.params[f"FD{i}"]) * logf ** i
        i += 1
    jsig = np.asarray(comps.get("signal_delay_sec", z), dtype=float)
    psig = wave if wave is not None else z
    D["deltas_ns"] = {
        "tdb": D["tdb_diff_ns"],
        "geo+shap": _center(np.asarray(comps["roemer_shapiro_sec"], float) - geo) * 1e9,
        "DM": _center(np.asarray(comps["dm_delay_sec"], float) - pdm) * 1e9,
        "DMX": _center(np.asarray(comps.get("dmx_delay_sec", z), float) - pdmx) * 1e9,
        "solar_wind": _center(np.asarray(comps.get("sw_delay_sec", z), float) - psw) * 1e9,
        "tropo": _center(np.asarray(comps.get("tropo_delay_sec", z), float) - ptr) * 1e9,
        "FD": _center(jfd - pfd) * 1e9,
        "signal": _center(jsig - psig) * 1e9,
        "prebinary": _center(jpre - ppre) * 1e9,
        "post_prebinary": _center((jtot - jpre) - (ptot - ppre)) * 1e9,
    }

    # --- matched-parameter component deltas -------------------------------------
    # The deltas above use each code at its OWN fitted params, so they mix two
    # effects: (a) genuine forward-MODEL differences and (b) the two fitters
    # converging to slightly different parameters. To isolate (a), re-evaluate
    # PINT's delay components at JUG's FITTED params (a copy of the model, no
    # refit) and difference against the SAME JUG components. If these are ~0
    # while deltas_ns is larger, the postfit per-component differences are
    # fit-driven (parameter split), not forward-model bugs.
    try:
        import copy as _copy
        _jug_final = D["fit_result"].get("final_params", {})
        mfm = _copy.deepcopy(mf)
        for _name in list(mfm.free_params):
            # PINT free-param names differ from JUG's canonical final_params keys
            # for aliased params (PINT STIGMA -> JUG STIG, PINT A1DOT -> JUG XDOT).
            # Without mapping, those are silently skipped and PINT keeps its OWN
            # fitted value -> a spurious matched-param delta (huge for edge-on DDH,
            # where an unsynced STIGMA is a ~us Shapiro difference).
            _jkey = _name if _name in _jug_final else canonicalize_param_name(_name)
            if _jkey not in _jug_final:
                continue
            try:
                _po = getattr(mfm, _name)
                if _name == "RAJ":
                    _po.quantity = (float(_jug_final[_jkey]) * u.rad).to(u.hourangle)
                elif _name == "DECJ":
                    _po.quantity = (float(_jug_final[_jkey]) * u.rad).to(u.deg)
                else:
                    _po.value = float(_jug_final[_jkey])
            except Exception:
                pass
        # PINT's OWN marginalized GLS chi2 evaluated at JUG's fitted params (mfm),
        # vs at PINT's own params (D["pint_gls_chi2"]). If chi2@JUG < chi2@PINT, JUG
        # found a better fit by PINT's own objective (a deeper local min of the
        # non-convex binary GLS landscape; cf J1017-7156 edge-on DDH). Same mfm copy,
        # no refit, so it's a pure point-evaluation of PINT's likelihood at JUG's params.
        try:
            D["pint_chi2_at_jug"] = float(Residuals(pint_toas, mfm).calc_chi2())
        except Exception as _exc:
            D["pint_chi2_at_jug"] = np.nan
            D["notes"].append(f"PINT chi2 @ JUG params failed: {_exc}")
        geo_m = (_pint_comp_sec(mfm, pint_toas, "AstrometryEquatorial", "solar_system_geometric_delay")
                 + _pint_comp_sec(mfm, pint_toas, "AstrometryEcliptic", "solar_system_geometric_delay")
                 + _pint_comp_sec(mfm, pint_toas, "SolarSystemShapiro", "solar_system_shapiro_delay"))
        pdm_m = _pint_comp_sec(mfm, pint_toas, "DispersionDM", "constant_dispersion_delay")
        pdmx_m = _pint_comp_sec(mfm, pint_toas, "DispersionDMX", "DMX_dispersion_delay")
        psw_m = _pint_comp_sec(mfm, pint_toas, "SolarWindDispersion", "solar_wind_delay")
        ptr_m = _pint_comp_sec(mfm, pint_toas, "TroposphereDelay", "troposphere_delay")
        pfd_m = _pint_comp_sec(mfm, pint_toas, "FD", "FD_delay")
        ptot_m = mfm.delay(pint_toas).to("s").value
        ppre_m = geo_m + pdm_m + pdmx_m + psw_m + ptr_m
        D["deltas_ns_matched"] = {
            "geo+shap": _center(np.asarray(comps["roemer_shapiro_sec"], float) - geo_m) * 1e9,
            "DM": _center(np.asarray(comps["dm_delay_sec"], float) - pdm_m) * 1e9,
            "DMX": _center(np.asarray(comps.get("dmx_delay_sec", z), float) - pdmx_m) * 1e9,
            "solar_wind": _center(np.asarray(comps.get("sw_delay_sec", z), float) - psw_m) * 1e9,
            "tropo": _center(np.asarray(comps.get("tropo_delay_sec", z), float) - ptr_m) * 1e9,
            "FD": _center(jfd - pfd_m) * 1e9,
            "prebinary": _center(jpre - ppre_m) * 1e9,
            "post_prebinary": _center((jtot - jpre) - (ptot_m - ppre_m)) * 1e9,
        }
    except Exception as exc:
        D["deltas_ns_matched"] = None
        D["notes"].append(f"matched-param component deltas failed: {exc}")

    # --- parameter comparison ------------------------------------------------------
    rows = []
    final = D["fit_result"].get("final_params", {})
    unc = D["fit_result"].get("uncertainties", {})
    for name in sorted(final):
        pint_name = "A1DOT" if name == "XDOT" else name
        po = getattr(mf, pint_name, None)
        if po is None or getattr(po, "value", None) is None:
            continue
        try:
            jv = float(final[name])
            if name in ("RAJ", "DECJ"):
                # JUG stores RAJ/DECJ in RADIANS; PINT's AngleParameter.value
                # is hourangle (RAJ) / degrees (DECJ). Comparing .value directly
                # subtracts mismatched units -> spurious millions-of-sigma
                # offsets. Convert PINT to radians to match JUG.
                pv = float(po.quantity.to(u.rad).value)
                ps = (float(po.uncertainty.to(u.rad).value)
                      if po.uncertainty is not None else np.nan)
            else:
                pv = float(po.value)
                ps = (float(po.uncertainty.value)
                      if po.uncertainty is not None else np.nan)
            js = float(unc.get(name, np.nan))
        except Exception:
            continue
        rows.append((name, jv, pv, js, ps))
    D["param_rows"] = rows

    # --- binary (2d4 fixed algebra) ----------------------------------------------
    D["binary"] = {}
    binary_model = str(session.params.get("BINARY", "")).strip().upper()
    pint_bin = next(((n, c) for n, c in mf.components.items()
                     if n.startswith("Binary")), None)
    if binary_model and pint_bin is not None:
        prebin = np.asarray(comps["prebinary_delay_sec"])
        prebin_mjd = jtdb - np.longdouble(prebin) / np.longdouble(86400.0)
        obs_pos = np.asarray(comps.get("ssb_obs_pos_ls", np.zeros((len(prebin), 3))))
        jug_bin = np.asarray(jug_binary_delay(prebin_mjd, session.params,
                                              obs_pos_ls=obs_pos), dtype=float)
        acc_auth = mf.delay(pint_toas, cutoff_component=pint_bin[0],
                            include_last=False)
        acc_jt = np.asarray((ptdb - prebin_mjd) * np.longdouble(86400.0),
                            dtype=float) * u.s
        bcomp = pint_bin[1]
        pint_own_jt = bcomp.binarymodel_delay(pint_toas, acc_delay=acc_jt).to("s").value
        # acc gap: stale-prebinary cross-check (notebook WARN)
        acc_gap_ns = float(np.max(np.abs(acc_auth.to("s").value - ppre))) * 1e9

        BIN_SYNC = ["A1", "PB", "T0", "ECC", "OM", "OMDOT", "PBDOT", "A1DOT",
                    "XDOT", "EDOT", "GAMMA", "M2", "SINI", "KIN", "KOM",
                    "H3", "H4", "STIGMA", "TASC", "EPS1", "EPS2",
                    "EPS1DOT", "EPS2DOT"]
        i = 0
        while f"FB{i}" in session.params or hasattr(mf, f"FB{i}"):
            BIN_SYNC.append(f"FB{i}")
            i += 1
        saved = {}
        try:
            for bp_name in BIN_SYNC:
                canon = canonicalize_param_name(bp_name)
                key = canon if canon in session.params else (
                    bp_name if bp_name in session.params else None)
                if key is None or not hasattr(mf, bp_name):
                    continue
                po = getattr(mf, bp_name)
                if getattr(po, "value", None) is None:
                    continue
                saved[bp_name] = po.value
                try:
                    po.value = get_longdouble(session.params, key)
                except Exception:
                    saved.pop(bp_name)
            pint_jugp_jt = bcomp.binarymodel_delay(
                pint_toas, acc_delay=acc_jt).to("s").value
        finally:
            for bp_name, val in saved.items():
                try:
                    getattr(mf, bp_name).value = val
                except Exception:
                    pass

        # JUG at PINT params; get_longdouble prefers '_high_precision' strings,
        # so override BOTH the dict value and the side-store.
        pp = dict(session.params)
        hp = dict(pp.get("_high_precision", {}))
        for bp_name, val in saved.items():
            canon = canonicalize_param_name(bp_name)
            key = canon if canon in pp else bp_name
            ld = np.longdouble(str(val))
            pp[key] = ld
            hp.pop(bp_name, None)
            hp[key] = str(ld)
        pp["_high_precision"] = hp
        jug_at_pint = np.asarray(jug_binary_delay(prebin_mjd, pp,
                                                  obs_pos_ls=obs_pos), dtype=float)

        blue = _center(jug_bin - pint_own_jt)
        red = _center(jug_at_pint - pint_own_jt)
        pb_val = float(session.params.get("PB", 0.0))
        if pb_val == 0.0 and "FB0" in session.params:
            pb_val = 1.0 / (float(session.params["FB0"]) * 86400.0)
        epoch = float(session.params.get("T0", session.params.get("TASC", 0.0)))
        D["binary"] = dict(
            model=binary_model, blue=blue, red=red, green=_center(blue - red),
            registry=_center(jug_bin - pint_jugp_jt),
            production=_center((jtot - jpre) - (ptot - ppre) - (jfd - pfd)),
            phase=((D["mjd"] - epoch) / pb_val) % 1.0 if pb_val else None,
            pb=pb_val, epoch=epoch, synced=sorted(saved), acc_gap_ns=acc_gap_ns,
        )

    # --- groups ----------------------------------------------------------------
    groups = defaultdict(list)
    for i, t in enumerate(session.toas_data):
        groups[(t.observatory.lower(), t.flags.get("be", ""))].append(i)
    D["groups"] = {k: np.asarray(v, dtype=int) for k, v in groups.items()}

    # --- audit summary -----------------------------------------------------------
    D["audit"] = {
        "EPHEM": str(session.params.get("EPHEM", "?")),
        "CLK": str(session.params.get("CLK", session.params.get("CLOCK", "?"))),
        "UNITS": str(session.params.get("UNITS", "?")),
        "PINT_EPHEM": str(getattr(mf, "EPHEM", None) and mf.EPHEM.value),
        "PINT_CLK": str(getattr(mf, "CLOCK", None) and mf.CLOCK.value),
        "nfit_jug": len(final), "nfit_pint": len(mf.free_params),
    }
    if "diff_vs_signal" not in D and wave is not None and np.std(wave) > 0:
        D["diff_vs_signal"] = float(np.corrcoef(D["diff_us"],
                                                _center(wave * 1e6))[0, 1])
    return D


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def page_summary(pdf, D):
    w = D["w"]
    # chi2 for BOTH codes with PINT's scaled errors, weighted-mean subtracted:
    # JUG's errors_us has different EFAC/EQUAD application, which would
    # otherwise masquerade as a chi2 difference.
    pe = D["pint_err_us"]
    wp = 1.0 / pe ** 2

    def _chi2(r):
        r = r - np.average(r, weights=wp)
        return float(np.sum((r / pe) ** 2))

    jug_chi2 = _chi2(D["jug_post_us"])
    pint_chi2_common = _chi2(D["pint_post_us"])
    dof = len(D["jug_post_us"]) - D["audit"]["nfit_jug"] - 1
    frac_wrap = np.max(np.abs(D["diff_us"])) * 1e-6 * D["F0"]
    # Compare the errors the FITTERS actually used: JUG's fit result carries
    # white-noise-scaled errors; comps['errors_us'] is the raw tim errors.
    jug_fit_err = np.asarray(D["fit_result"].get("errors_us", D["err_us"]),
                             dtype=float)
    err_ratio = jug_fit_err / D["pint_err_us"]
    L = []
    L.append(f"{D['jname']}   nTOA={len(D['mjd'])}   span={D['mjd'].min():.0f}-{D['mjd'].max():.0f}")
    L.append(f"par: {Path(D['par']).name}    tim: {Path(D['tim']).name}")
    L.append("")
    L.append(f"EPHEM JUG={D['audit']['EPHEM']} PINT={D['audit']['PINT_EPHEM']}    "
             f"CLK JUG={D['audit']['CLK']} PINT={D['audit']['PINT_CLK']}    "
             f"UNITS={D['audit']['UNITS']}")
    L.append(f"free params: JUG={D['audit']['nfit_jug']}  PINT={D['audit']['nfit_pint']}")
    L.append("")
    L.append(f"WRMS  JUG ={_wrms(D['jug_post_us'], w):.6f} us    "
             f"PINT={_wrms(D['pint_post_us'], w):.6f} us")
    L.append(f"chi2/dof (PINT scaled errors, both): "
             f"JUG={jug_chi2/max(dof,1):.6f}   PINT={pint_chi2_common/max(dof,1):.6f}   "
             f"(PINT own: {D['pint_chi2']/max(D['pint_dof'],1):.6f})")
    sj, sp = D.get("jug_step_sigma", np.nan), D.get("pint_step_sigma", np.nan)
    if np.isfinite(sj) and np.isfinite(sp):
        L.append(f"CONVERGENCE max|dp/sigma| (next step, lower=more converged): "
                 f"JUG={sj:.2e} (worst {D.get('jug_step_worst')})   "
                 f"PINT={sp:.2e} (worst {D.get('pint_step_worst')})")
    else:
        L.append("CONVERGENCE max|dp/sigma|: not computed "
                 "(pass --convergence; adds an extra fit per code per pulsar)")
    L.append(f"GLS chi2 (own noise, lower=better fit): "
             f"JUG={D.get('jug_gls_chi2', np.nan):.4f}   "
             f"PINT={D.get('pint_gls_chi2', np.nan):.4f}")
    # PINT's OWN marginalized GLS chi2 (calc_chi2) at PINT's params vs at JUG's params.
    # Same noise model + objective for both points => a fair head-to-head: which
    # parameter vector fits better by PINT's own likelihood. Lower = better.
    _cp = D.get("pint_gls_chi2", np.nan)        # PINT @ PINT params
    _cj = D.get("pint_chi2_at_jug", np.nan)     # PINT @ JUG params
    _cdelta = _cj - _cp                         # <0 => JUG params fit better
    L.append(f"PINT marginalized chi2 @PINT={_cp:.4f}  @JUG={_cj:.4f}  "
             f"(JUG-PINT={_cdelta:+.4f}; <0 => JUG fits better by PINT's own objective)")
    _pconv = D.get("pint_converged", None)
    _pconv_txt = ("converged" if _pconv is True
                  else "NOT CONVERGED (best-so-far)" if _pconv is False
                  else "unknown")
    L.append(f"PINT fitter: {D.get('pint_fitter_name','GLSFitter')}  |  convergence: {_pconv_txt}")
    L.append("")
    L.append(f"PREfit  JUG-PINT: std={np.std(D['diff_pre_us'])*1e3:.4f} ns  "
             f"max={np.max(np.abs(D['diff_pre_us']))*1e3:.4f} ns")
    L.append(f"POSTfit JUG-PINT: std={np.std(D['diff_us'])*1e3:.4f} ns  "
             f"max={np.max(np.abs(D['diff_us']))*1e3:.4f} ns")
    L.append(f"  after constant/F0/F1 gauge projection: "
             f"std={np.std(D['diff_spin_clean_us'])*1e3:.4f} ns  "
             f"max={np.max(np.abs(D['diff_spin_clean_us']))*1e3:.4f} ns")
    L.append(f"JUG returned-vs-fresh residual consistency: "
             f"std={D['returned_fresh_ns']:.4f} ns")
    L.append(f"TDB parity: std={np.std(_center(D['tdb_diff_ns'])):.4f} ns  "
             f"max|.|={np.max(np.abs(_center(D['tdb_diff_ns']))):.4f} ns")
    L.append(f"bary-freq parity: max|rel|={np.max(np.abs(D['freq_rel_ppb'])):.4f} ppb")
    L.append(f"max|diff| as phase fraction = {frac_wrap:.2e}  "
             f"({'OK' if frac_wrap < 0.01 else 'CHECK WRAPS'})")
    L.append(f"fitter TOA error ratio JUG/PINT (white-noise scaled): "
             f"median={np.median(err_ratio):.6f}  "
             f"range=[{err_ratio.min():.6f}, {err_ratio.max():.6f}]")
    if D["binary"]:
        b = D["binary"]
        L.append("")
        L.append(f"BINARY {b['model']}  (std, ps):")
        L.append(f"  own-fit delta (C0)       = {np.std(b['blue'])*1e12:9.3f}")
        L.append(f"  expected fit difference  = {np.std(b['green'])*1e12:9.3f}")
        L.append(f"  corrected model/path (C3)= {np.std(b['red'])*1e12:9.3f}   <- genuine JUG-vs-PINT binary error")
        L.append(f"  registry matched check   = {np.std(b['registry'])*1e12:9.3f}")
        L.append(f"  production subtraction   = {np.std(b['production'])*1e12:9.3f}   (own-fit signal + ~14 ps float64 floor)")
        L.append(f"  acc(prebinary) gap       = {b['acc_gap_ns']:9.3f} ns "
                 f"{'(OK)' if b['acc_gap_ns'] < 1 else '(STALE prebinary?)'}")
        L.append(f"  synced params: {b['synced']}")
        verdict = ("BENIGN fit gauge" if np.std(b['red']) * 1e9 < 1.0
                   else "INVESTIGATE: matched-param error > 1 ns")
        L.append(f"  => {verdict}")
    if "diff_vs_signal" in D:
        L.append(f"corr(JUG-PINT diff, signal waveform) = {D['diff_vs_signal']:+.3f}")
    for n in D["notes"]:
        L.append(f"NOTE: {n}")
    L.append("")
    L.append("worst parameter offsets (JUG-PINT)/sigma_PINT:")
    rows = sorted(D["param_rows"],
                  key=lambda r: -abs((r[1] - r[2]) / r[4]) if r[4] and np.isfinite(r[4]) and r[4] > 0 else 0)
    for name, jv, pv, js, ps in rows[:12]:
        nsig = (jv - pv) / ps if ps and np.isfinite(ps) and ps > 0 else np.nan
        sr = js / ps if ps and np.isfinite(js) and ps > 0 else np.nan
        L.append(f"  {name:12s} d/sig={nsig:+10.4f}   sig_J/sig_P={sr:8.4f}   "
                 f"JUG={jv:.12g}  PINT={pv:.12g}")

    fig = plt.figure(figsize=(11, 8.5))
    # Colored head-to-head banner: PINT's own marginalized chi2 at JUG's vs PINT's params.
    # GREEN = JUG fits better, RED = PINT fits better, BLUE = effectively tied.
    _SAME_THRESH = 1.0   # chi2 units; |delta| below this = statistically the same fit
    if not (np.isfinite(_cj) and np.isfinite(_cp)):
        _bcol, _btxt = "0.4", "PINT chi2 @JUG vs @PINT: N/A (calc_chi2 unavailable)"
    elif _cdelta < -_SAME_THRESH:
        _bcol = "#1a7f37"  # green
        _btxt = f"JUG FITS BETTER by PINT's own objective:  chi2 @JUG={_cj:.3f} < @PINT={_cp:.3f}  (delta={_cdelta:+.3f})"
    elif _cdelta > _SAME_THRESH:
        _bcol = "#cf222e"  # red
        _btxt = f"PINT FITS BETTER:  chi2 @PINT={_cp:.3f} < @JUG={_cj:.3f}  (delta={_cdelta:+.3f})"
    else:
        _bcol = "#0969da"  # blue
        _btxt = f"JUG ≈ PINT (tied within {_SAME_THRESH:g}):  chi2 @JUG={_cj:.3f}  @PINT={_cp:.3f}  (delta={_cdelta:+.3f})"
    # If PINT did not converge, the @PINT chi2 is not a real minimum, so the
    # head-to-head verdict is unreliable -- flag it loudly.
    if D.get("pint_converged") is False:
        _btxt += "   [PINT NOT CONVERGED -- verdict unreliable]"
        _bcol = "#bf8700"  # amber override
    fig.text(0.04, 0.985, _btxt, fontsize=11, fontweight="bold", color=_bcol,
             va="top", family="monospace")
    fig.text(0.04, 0.955, "\n".join(L), fontsize=8, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_residuals(pdf, D):
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax = axes[0]
    ax.scatter(D["mjd"], D["jug_pre_us"], s=2, alpha=0.4, label="JUG prefit")
    ax.scatter(D["mjd"], D["pint_pre_us"], s=2, alpha=0.4, label="PINT prefit")
    ax.set_ylabel("prefit (us)")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_title(f"{D['jname']}: residuals JUG vs PINT")
    ax = axes[1]
    ax.scatter(D["mjd"], D["diff_pre_us"] * 1e3, s=2, alpha=0.5, c="C2")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("prefit diff (ns)")
    ax = axes[2]
    w = D["w"]
    wrms_j = _wrms(D["jug_post_us"], w)
    wrms_p = _wrms(D["pint_post_us"], w)
    ax.errorbar(D["mjd"], D["jug_post_us"], yerr=D["err_us"], fmt=".", ms=2,
                alpha=0.35, lw=0.4, label=f"JUG postfit  (WRMS={wrms_j*1e3:.1f} ns)")
    ax.scatter(D["mjd"], D["pint_post_us"], s=2, alpha=0.4, c="C1",
               label=f"PINT postfit  (WRMS={wrms_p*1e3:.1f} ns)")
    ax.set_ylabel("postfit (us)")
    ax.legend(fontsize=8, markerscale=3)
    ax = axes[3]
    ax.scatter(D["mjd"], D["diff_us"] * 1e3, s=2, alpha=0.5, c="C2")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("postfit diff (ns)")
    ax.set_xlabel("TDB MJD")
    plt.tight_layout(rect=(0, 0, 1, 0.93))

    # --- big bold convergence verdict --------------------------------------
    sj = D.get("jug_step_sigma", np.nan)
    sp = D.get("pint_step_sigma", np.nan)
    cj = D.get("jug_gls_chi2", np.nan)
    cp = D.get("pint_gls_chi2", np.nan)
    # Floor-aware verdict: a next-step < 1e-3 sigma is converged for all
    # practical purposes (solution within 0.001 of the error bar). Only call a
    # "winner" when one code is meaningfully unconverged (>1e-3 sigma) AND the
    # gap is >3x — otherwise the difference is measurement noise (the JUG
    # continuation re-enters its WLS phase, so its step is not byte-identical in
    # definition to PINT's GLS step).
    FLOOR = 1e-3
    if not (np.isfinite(sj) and np.isfinite(sp)):
        who, col = "n/a", "0.2"
    elif max(sj, sp) < FLOOR:
        who, col = "BOTH CONVERGED", "0.15"
    elif sj < sp and sp / max(sj, 1e-30) > 3:
        who, col = "JUG more converged", "C0"
    elif sp < sj and sj / max(sp, 1e-30) > 3:
        who, col = "PINT more converged", "C1"
    else:
        who, col = "COMPARABLE", "0.15"
    fig.text(0.5, 0.987,
             f"{D['jname']}: postfit residuals JUG vs PINT",
             ha="center", va="top", fontsize=11, fontweight="bold")
    if np.isfinite(sj) and np.isfinite(sp):
        fig.text(0.5, 0.96,
                 f"CONVERGENCE (next-step max|Δp/σ|):  "
                 f"JUG={sj:.1e}  PINT={sp:.1e}  →  {who}",
                 ha="center", va="top", fontsize=12, fontweight="bold", color=col)
        conv_note = "  [<1e-3 σ = converged]"
    else:
        fig.text(0.5, 0.96,
                 "CONVERGENCE: pass --convergence to compute next-step max|Δp/σ|",
                 ha="center", va="top", fontsize=10, color="0.4")
        conv_note = ""
    fig.text(0.5, 0.935,
             f"WRMS JUG={wrms_j*1e3:.2f} ns  PINT={wrms_p*1e3:.2f} ns     "
             f"GLS χ² JUG={cj:.3f} PINT={cp:.3f}  (lower=better){conv_note}",
             ha="center", va="top", fontsize=9)
    axes[0].set_title("")
    pdf.savefig(fig)
    plt.close(fig)


def page_distribution(pdf, D):
    from scipy import stats
    d_ns = D["diff_us"] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.hist(d_ns, bins=100, color="C2", alpha=0.8)
    ax.set_xlabel("postfit JUG-PINT (ns)")
    ax.set_ylabel("count")
    ax.set_title(f"std={np.std(d_ns):.4f} ns")
    ax = axes[0, 1]
    stats.probplot(d_ns / np.std(d_ns), dist="norm", plot=ax)
    ax.set_title("QQ vs normal (diff/std)")
    ax = axes[1, 0]
    s = np.sort(np.abs(d_ns))[::-1]
    cum = np.cumsum(s) / np.sum(s)
    ax.plot(np.arange(1, len(s) + 1) / len(s) * 100, cum * 100)
    ax.set_xlabel("% worst TOAs")
    ax.set_ylabel("% of total |diff|")
    ax.set_title("Lorenz: is the diff carried by a few TOAs?")
    ax.grid(alpha=0.3)
    ax = axes[1, 1]
    ax.scatter(D["err_us"], np.abs(d_ns), s=3, alpha=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TOA error (us)")
    ax.set_ylabel("|diff| (ns)")
    ax.set_title("diff vs TOA uncertainty")
    fig.suptitle(f"{D['jname']}: postfit diff distribution", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_drivers(pdf, D):
    d_ns = D["diff_us"] * 1e3
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = axes[0, 0]
    ax.scatter(D["freq_topo"], d_ns, s=3, alpha=0.4)
    ax.set_xlabel("topocentric freq (MHz)")
    ax.set_ylabel("diff (ns)")
    ax.set_title("diff vs observing frequency")
    ax = axes[0, 1]
    sc = ax.scatter(D["mjd"], d_ns, s=3, alpha=0.5, c=D["freq_topo"], cmap="viridis")
    plt.colorbar(sc, ax=ax, label="freq (MHz)")
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("diff (ns)")
    ax.set_title("diff vs MJD, colored by frequency")
    ax = axes[1, 0]
    ax.scatter(D["mjd"], _center(D["tdb_diff_ns"]), s=3, alpha=0.4, c="C4")
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("TDB diff (ns, centered)")
    ax.set_title("TDB parity (clock chain + time transforms)")
    ax = axes[1, 1]
    ax.scatter(D["mjd"], D["freq_rel_ppb"], s=3, alpha=0.4, c="C5")
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("bary freq rel diff (ppb)")
    ax.set_title("barycentric frequency parity (Doppler chain)")
    fig.suptitle(f"{D['jname']}: residual-diff drivers", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_components(pdf, D):
    keys = list(D["deltas_ns"])
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.6 * len(keys)),
                             sharex=True, squeeze=False)
    for i, k in enumerate(keys):
        d = _center(D["deltas_ns"][k])
        ax = axes[i, 0]
        ax.scatter(D["mjd"], d, s=2, alpha=0.5)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_ylabel(f"{k}\n(ns)", fontsize=8)
        ax.text(0.99, 0.9, f"std={np.std(d):.4f}  max={np.max(np.abs(d)):.4f} ns",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[-1, 0].set_xlabel("TDB MJD")
    fig.suptitle(f"{D['jname']}: delay-component deltas (JUG-PINT, centered)", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_components_matched(pdf, D):
    """Delay-component deltas with PINT evaluated at JUG's FITTED params.

    Isolates forward-MODEL differences from fit-convergence differences: if a
    component is ~0 here but nonzero on the regular components page, that
    component's postfit difference is fit-driven (the codes settling on
    slightly different parameters), not a model bug."""
    dm = D.get("deltas_ns_matched")
    if not dm:
        return
    keys = list(dm)
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.6 * len(keys)),
                             sharex=True, squeeze=False)
    for i, k in enumerate(keys):
        d = _center(dm[k])
        own = _center(D["deltas_ns"][k]) if k in D.get("deltas_ns", {}) else None
        ax = axes[i, 0]
        ax.scatter(D["mjd"], d, s=2, alpha=0.6, c="C3")
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_ylabel(f"{k}\n(ns)", fontsize=8)
        own_txt = (f"  | own-fit std={np.std(own):.4f}" if own is not None else "")
        ax.text(0.99, 0.9,
                f"matched std={np.std(d):.4f} max={np.max(np.abs(d)):.4f} ns{own_txt}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    axes[-1, 0].set_xlabel("TDB MJD")
    fig.suptitle(f"{D['jname']}: delay-component deltas at MATCHED params "
                 f"(PINT @ JUG fitted params) — forward-model floor", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    plt.close(fig)


def page_groups(pdf, D):
    d_ns = D["diff_us"] * 1e3
    items = sorted(D["groups"].items(), key=lambda kv: -len(kv[1]))[:8]
    fig, axes = plt.subplots(len(items), 1, figsize=(11, 2.0 * len(items)),
                             sharex=True, squeeze=False)
    for row, (g, idx) in enumerate(items):
        ax = axes[row, 0]
        r = d_ns[idx]
        # top correlated component
        best, best_r = None, 0.0
        for k, v in D["deltas_ns"].items():
            arr = _center(v)[idx]
            if np.std(arr) == 0 or np.std(r) == 0:
                continue
            c = np.corrcoef(r, arr)[0, 1]
            if np.isfinite(c) and abs(c) > abs(best_r):
                best, best_r = k, c
        ax.scatter(D["mjd"][idx], r, s=3, alpha=0.5, label=f"diff std={np.std(r):.3f} ns")
        if best is not None:
            arr = _center(D["deltas_ns"][best])[idx]
            scale = np.std(r) / np.std(arr) if np.std(arr) > 0 else 1.0
            ax.plot(np.sort(D["mjd"][idx]),
                    (arr * scale)[np.argsort(D["mjd"][idx])], c="C3", lw=0.8,
                    alpha=0.7, label=f"top: {best} (r={best_r:+.2f}, scaled)")
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_ylabel(f"{'/'.join(map(str, g))}\n(ns)", fontsize=8)
        ax.legend(fontsize=7, markerscale=3, loc="upper right")
    axes[-1, 0].set_xlabel("TDB MJD")
    fig.suptitle(f"{D['jname']}: diff by (observatory, backend) with top component", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_longtime(pdf, D):
    from astropy.timeseries import LombScargle
    d_ns = D["diff_us"] * 1e3
    mjd = D["mjd"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    ax = axes[0]
    bins = np.arange(mjd.min(), mjd.max() + 30, 30)
    idx = np.digitize(mjd, bins) - 1
    bm = np.array([np.mean(d_ns[idx == k]) if np.any(idx == k) else np.nan
                   for k in range(len(bins) - 1)])
    ax.scatter(mjd, d_ns, s=2, alpha=0.2)
    ax.plot(0.5 * (bins[:-1] + bins[1:]), bm, "o-", c="C3", ms=3, lw=1,
            label="30-day binned mean")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("diff (ns)")
    ax.legend(fontsize=8)
    ax.set_title("long-timescale drift of JUG-PINT diff")

    ax = axes[1]
    span = mjd.max() - mjd.min()
    f, p = LombScargle(mjd, d_ns).autopower(
        minimum_frequency=0.5 / span, maximum_frequency=0.2, samples_per_peak=8)
    ax.plot(1.0 / f, p, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("period (days)")
    ax.set_ylabel("LS power")
    for per, lab in [(365.25, "1 yr"), (182.6, "1/2 yr")]:
        ax.axvline(per, color="C3", ls=":", lw=0.8)
        ax.text(per, ax.get_ylim()[1] * 0.9, lab, fontsize=7, color="C3")
    if D["binary"] and D["binary"]["pb"]:
        ax.axvline(D["binary"]["pb"], color="C2", ls=":", lw=0.8)
        ax.text(D["binary"]["pb"], ax.get_ylim()[1] * 0.8, "PB", fontsize=7, color="C2")
    ax.set_title("Lomb-Scargle periodogram of the diff")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_stability(pdf, D):
    from astropy.timeseries import LombScargle
    d_ns = D["diff_us"] * 1e3
    mjd = D["mjd"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # Allan-like deviation of the diff
    ax = axes[0, 0]
    taus, adev = [], []
    span = mjd.max() - mjd.min()
    tau = 10.0
    while tau < span / 3:
        bins = np.arange(mjd.min(), mjd.max() + tau, tau)
        idx = np.digitize(mjd, bins) - 1
        bm = np.array([np.mean(d_ns[idx == k]) for k in range(len(bins) - 1)
                       if np.any(idx == k)])
        if len(bm) > 2:
            taus.append(tau)
            adev.append(np.sqrt(0.5 * np.mean(np.diff(bm) ** 2)))
        tau *= 2
    ax.loglog(taus, adev, "o-")
    ax.set_xlabel("tau (days)")
    ax.set_ylabel("Allan-like dev of diff (ns)")
    ax.set_title("structure function (white diff: slope -1/2)")
    ax.grid(alpha=0.3, which="both")

    # lag ACF of normalized diff (TOA ordering)
    ax = axes[0, 1]
    x = d_ns / np.std(d_ns)
    n = len(x)
    maxlag = min(200, n // 4)
    acf = [1.0] + [float(np.mean(x[:-k] * x[k:])) for k in range(1, maxlag)]
    ax.plot(acf, lw=0.8)
    ax.axhline(0, color="k", lw=0.4)
    for s in (1, -1):
        ax.axhline(s * 2 / np.sqrt(n), color="C3", ls=":", lw=0.8)
    ax.set_xlabel("TOA lag")
    ax.set_ylabel("ACF")
    ax.set_title("diff autocorrelation (index lag)")

    # residual periodograms overlay
    ax = axes[1, 0]
    for arr, lab, c in [(D["jug_post_us"], "JUG", "C0"),
                        (D["pint_post_us"], "PINT", "C1")]:
        f, p = LombScargle(mjd, _center(arr)).autopower(
            minimum_frequency=0.5 / span, maximum_frequency=0.2,
            samples_per_peak=5)
        ax.plot(1.0 / f, p, lw=0.6, alpha=0.8, c=c, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("period (days)")
    ax.set_ylabel("LS power")
    ax.legend(fontsize=8)
    ax.set_title("postfit residual periodograms (should overlay)")

    # cumulative |diff| vs MJD
    ax = axes[1, 1]
    o = np.argsort(mjd)
    ax.plot(mjd[o], np.cumsum(np.abs(d_ns[o])) / np.sum(np.abs(d_ns)), lw=1)
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("cumulative fraction of |diff|")
    ax.set_title("where in time the diff accumulates")
    ax.grid(alpha=0.3)
    fig.suptitle(f"{D['jname']}: stability / spectral checks", y=1.0)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_params(pdf, D):
    rows = [r for r in D["param_rows"]
            if r[4] and np.isfinite(r[4]) and r[4] > 0]
    if not rows:
        return
    names = [r[0] for r in rows]
    nsig = [(r[1] - r[2]) / r[4] for r in rows]
    sratio = [r[3] / r[4] if np.isfinite(r[3]) else np.nan for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x, nsig, color="C0")
    ax.axhline(0, color="k", lw=0.5)
    for thr in (-1, 1):
        ax.axhline(thr, color="C3", ls=":", lw=0.8)
    ax.set_ylabel("(JUG - PINT) / sigma_PINT")
    ax.set_title(f"{D['jname']}: fitted parameter comparison")
    ax = axes[1]
    ax.bar(x, sratio, color="C2")
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_ylabel("sigma_JUG / sigma_PINT")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_noise_gls(pdf, D):
    rows = D.get("noise_rows", [])
    if not rows and not (np.isfinite(D.get("jug_gls_chi2", np.nan)) or
                         np.isfinite(D.get("pint_gls_chi2", np.nan))):
        return
    labels = [r[0] for r in rows] + ["noise-subtracted residual"]
    jug_vals = [np.std(r[3]) for r in rows]
    pint_vals = [np.std(r[4]) for r in rows]
    jug_vals.append(np.std(D["jug_post_us"] - D["jug_noise_total_us"]))
    pint_vals.append(np.std(D["pint_post_us"] - D["pint_noise_total_us"]))
    arr = np.column_stack([jug_vals, pint_vals])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["JUG", "PINT"])
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(2):
            ax.text(j, i, f"{arr[i, j]:.4f}", ha="center", va="center", fontsize=8,
                    color="white" if arr[i, j] < np.nanmax(arr) * 0.6 else "black")
    ax.set_title("Component RMS (us)")
    fig.colorbar(im, ax=ax, label="RMS (us)")

    dof = max(1, len(D["mjd"]) - D["audit"]["nfit_jug"])
    vals = [D.get("jug_gls_chi2", np.nan) / dof,
            D.get("pint_gls_chi2", np.nan) / max(1, D.get("pint_dof", dof))]
    ax = axes[1]
    ax.bar([0, 1], vals, color=["C0", "C1"])
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["JUG", "PINT"])
    ax.set_ylabel("canonical full-GLS chi2 / dof")
    ax.set_title(f"Marginalized GLS objective\nJUG={vals[0]:.5f}  PINT={vals[1]:.5f}")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_noise_components(pdf, D):
    rows = D.get("noise_rows", [])
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, max(4, 3.2 * len(rows))),
                             squeeze=False)
    for i, (label, _, _, j, p) in enumerate(rows):
        d = _center(j - p)
        ax = axes[i, 0]
        ax.scatter(D["mjd"], j, s=2, alpha=0.35, label="JUG")
        ax.scatter(D["mjd"], p, s=2, alpha=0.35, label="PINT")
        ax.set_ylabel("us"); ax.set_title(f"{label} realization")
        if i == 0: ax.legend(fontsize=8, markerscale=3)
        ax = axes[i, 1]
        ax.scatter(D["mjd"], d * 1e3, s=2, alpha=0.45, c="C2")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_ylabel("JUG-PINT (ns)")
        ax.set_title(f"std={np.std(d)*1e3:.4f} ns, max={np.max(np.abs(d))*1e3:.4f} ns")
    axes[-1, 0].set_xlabel("TDB MJD"); axes[-1, 1].set_xlabel("TDB MJD")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_dmx(pdf, D):
    mf_rows = [r for r in D["param_rows"] if re.match(r"^DMX_\d+$", r[0])]
    if not mf_rows:
        return
    sess = D["session"]
    utc_mjd = np.array([t.mjd_int + t.mjd_frac for t in sess.toas_data], dtype=float)
    topo = D["freq_topo"]
    epochs, jv, pv, js, ps, fmin, ntoa = [], [], [], [], [], [], []
    for name, j, p, ju, pu in mf_rows:
        n = name.split("_")[1]
        r1 = float(sess.params.get(f"DMXR1_{n}", np.nan))
        r2 = float(sess.params.get(f"DMXR2_{n}", np.nan))
        mask = (utc_mjd >= r1) & (utc_mjd < r2)
        epochs.append(0.5 * (r1 + r2)); jv.append(j); pv.append(p)
        js.append(ju); ps.append(pu); ntoa.append(int(mask.sum()))
        fmin.append(float(np.min(topo[mask])) if np.any(mask) else np.nan)
    epochs, jv, pv, js, ps = map(np.asarray, (epochs, jv, pv, js, ps))
    fmin, ntoa = np.asarray(fmin), np.asarray(ntoa)
    delta = jv - pv
    active = (ntoa > 0) & np.isfinite(fmin)
    mean_delta = float(np.mean(delta[active])) if np.any(active) else 0.0
    K_DM_SEC = 4.148808e3
    raw_ps = K_DM_SEC * delta / fmin**2 * 1e12
    nongauge_ps = K_DM_SEC * (delta - mean_delta) / fmin**2 * 1e12

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    axes[0].errorbar(epochs, jv, yerr=js, fmt="o", ms=3, alpha=0.7, label="JUG")
    axes[0].errorbar(epochs, pv, yerr=ps, fmt="s", ms=3, alpha=0.7, label="PINT")
    axes[0].set_ylabel("DMX (pc/cm3)"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"{D['jname']}: DMX bins ({len(mf_rows)}), mean gauge shift={mean_delta:+.3e}")
    axes[1].scatter(epochs, raw_ps, s=12, alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].set_ylabel("raw delay at min freq (ps)")
    axes[2].scatter(epochs, nongauge_ps, s=12, alpha=0.8, c="C2")
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel("non-gauge delay (ps)")
    axes[2].set_title(f"deviation from mean gauge shift; max={np.nanmax(np.abs(nongauge_ps)):.3f} ps")
    axes[3].bar(epochs, ntoa, width=20, alpha=0.65)
    axes[3].set_ylabel("TOAs/bin"); axes[3].set_xlabel("UTC MJD")
    plt.tight_layout(); pdf.savefig(fig); plt.close(fig)


def page_binary(pdf, D):
    b = D["binary"]
    mjd = D["mjd"]
    blue_us, green_us, red_us = b["blue"] * 1e6, b["green"] * 1e6, b["red"] * 1e6
    prod_us = b["production"] * 1e6
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(mjd, prod_us, s=1, alpha=0.15, c="0.45",
               label=f"production subtraction  std={np.std(prod_us)*1e6:.2f} ps")
    ax.scatter(mjd, green_us, s=4, alpha=0.4, c="C2",
               label=f"expected fit difference  std={np.std(green_us)*1e6:.3f} ps")
    ax.scatter(mjd, blue_us, s=2, alpha=0.6, c="C0", zorder=3,
               label=f"own-fit binary delta  std={np.std(blue_us)*1e6:.3f} ps")
    ax.scatter(mjd, red_us, s=3, alpha=0.5, c="C3", zorder=4,
               label=f"corrected (model/path)  std={np.std(red_us)*1e6:.3f} ps")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("TDB MJD")
    ax.set_ylabel("us")
    ax.set_title(f"{D['jname']} binary delta vs MJD ({b['model']})")
    ax.legend(fontsize=8, markerscale=4)

    if b["phase"] is not None:
        o = np.argsort(b["phase"])
        ax = axes[0, 1]
        ax.scatter(np.asarray(b["phase"])[o], green_us[o], s=4, alpha=0.4,
                   c="C2", label="expected fit difference")
        ax.scatter(b["phase"], blue_us, s=2, alpha=0.6, c="C0", zorder=3,
                   label="own-fit binary delta")
        ax.scatter(b["phase"], red_us, s=3, alpha=0.5, c="C3", zorder=4,
                   label="corrected (model/path)")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("orbital phase")
        ax.set_ylabel("us")
        ax.set_title("binary delta vs orbital phase (production excluded: "
                     "float64 floor)")
        ax.legend(fontsize=8, markerscale=4)

    ax = axes[1, 0]
    ax.hist(blue_us, bins=80, color="C0", alpha=0.5,
            label=f"own-fit  std={np.std(blue_us)*1e3:.4g} ns")
    ax.hist(red_us, bins=80, color="C3", alpha=0.75,
            label=f"corrected  std={np.std(red_us)*1e3:.4g} ns")
    ax.axvline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("us")
    ax.set_ylabel("count")
    ax.set_title("distribution: raw vs gauge-corrected")
    ax.legend(fontsize=8)

    if b["phase"] is not None:
        ax = axes[1, 1]
        bins = np.linspace(0, 1, 41)
        idx = np.digitize(b["phase"], bins) - 1
        cen = 0.5 * (bins[:-1] + bins[1:])

        def binned(x):
            return np.array([np.mean(x[idx == k]) if np.any(idx == k) else np.nan
                             for k in range(len(bins) - 1)])

        ax.plot(cen, binned(green_us), "s--", lw=1.2, ms=4, c="C2", alpha=0.8,
                label="expected binned mean")
        ax.plot(cen, binned(red_us), "o-", lw=1.2, ms=4, c="C3",
                label="corrected binned mean")
        ax.plot(cen, binned(blue_us), "o-", lw=2.0, ms=5, c="C0", zorder=5,
                label="raw binned mean")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("orbital phase")
        ax.set_ylabel("mean delta (us)")
        ax.set_title(f"phase-folded mean (epoch={b['epoch']:.3f}, PB={b['pb']:.6f})")
        ax.legend(fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def run_speed(D, maxiter, clock_dir, pint_clock_dir, repeats, out_dir,
              pint_maxiter=5):
    """Time end-to-end and warm fit-only for both codes (notebook 2-speed).

    PINT is timed at pint_maxiter (its plain GLSFitter has no early-stop, so
    timing it at JUG's high maxiter would just measure wasted oscillation
    iterations, not convergence speed)."""
    import copy
    import json as _json
    import time

    import pint.fitter
    import pint.models
    import pint.toa

    from jug.engine.session import TimingSession

    par, tim = D["pint_par_used"], D["tim"]
    jug_par = D["par"]

    def build_pint():
        prepare_pint_environment(clock_dir, pint_clock_dir)
        m = pint.models.get_model(par, allow_T2=True, allow_tcb=True)
        m = prepare_pint_model(m, par)
        t = pint.toa.get_TOAs(tim, model=m)
        prepare_pint_toas(m, t)
        return m, t

    def fit_pint(m, t):
        f = pint.fitter.GLSFitter(t, m)
        f.fit_toas(maxiter=pint_maxiter)

    def build_jug():
        return TimingSession(jug_par, tim, clock_dir=clock_dir)

    # Fair "both converged" timing: time JUG to the SAME convergence level PINT
    # reaches (chi2 stable), not JUG's strict 1e-5 criterion. Beyond chi2
    # convergence JUG only crawls the degenerate eccentric-binary direction
    # (e.g. M2) sub-ns; counting those ~25 extra cheap iters against it
    # understates its speed. JUG_GLS_DTOL=1e-2 stops at chi2 convergence with
    # an RMS identical to strict (verified <0.04 ns difference on J1946). PINT
    # is at pint_maxiter, already chi2-converged. Restored in finally.
    SPEED_JUG_DTOL = "1e-2"
    _prev_dtol = os.environ.get("JUG_GLS_DTOL")
    os.environ["JUG_GLS_DTOL"] = SPEED_JUG_DTOL

    def fit_jug(s):
        s.fit_parameters(max_iter=maxiter)

    try:
        # Explicit JAX warm-up: compile all delay/derivative kernels once,
        # untimed, so no fit below is inflated by one-time JIT compilation
        # (which a changed @jax.jit kernel re-triggers on first use).
        build_jug().fit_parameters(max_iter=2)
        build_pint()  # warms PINT clock/observatory caches too

        # caches are already warm; time end-to-end fresh each repeat
        e2e_p, e2e_j, warm_p, warm_j = [], [], [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            m, t = build_pint(); fit_pint(m, t)
            e2e_p.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            s = build_jug(); fit_jug(s)
            e2e_j.append(time.perf_counter() - t0)
        m_tpl, t_warm = build_pint()
        for _ in range(repeats):
            mc = copy.deepcopy(m_tpl)
            t0 = time.perf_counter(); fit_pint(mc, t_warm)
            warm_p.append(time.perf_counter() - t0)
            s = build_jug()  # untimed build (fresh session, warm disk caches)
            t0 = time.perf_counter(); fit_jug(s)
            warm_j.append(time.perf_counter() - t0)
    finally:
        if _prev_dtol is None:
            os.environ.pop("JUG_GLS_DTOL", None)
        else:
            os.environ["JUG_GLS_DTOL"] = _prev_dtol

    med = lambda x: float(np.median(x))
    sp = dict(e2e_p=e2e_p, e2e_j=e2e_j, warm_p=warm_p, warm_j=warm_j,
              pe=med(e2e_p), je=med(e2e_j), pw=med(warm_p), jw=med(warm_j))

    # regression baseline per pulsar, kept beside the PDFs
    bl_path = Path(out_dir) / ".jug_speed_baseline.json"
    key = f"{D['jname']}_{len(D['mjd'])}toa_iter{maxiter}"
    base = {}
    if bl_path.exists():
        try:
            base = _json.loads(bl_path.read_text())
        except Exception:
            base = {}
    prev = base.get(key, {}).get("jug_warm_ms")
    sp["baseline_prev_ms"] = prev
    sp["baseline_change"] = ((sp["jw"] * 1e3 - prev) / prev
                             if prev else None)
    base[key] = {"jug_warm_ms": sp["jw"] * 1e3, "pint_warm_ms": sp["pw"] * 1e3,
                 "jug_e2e_ms": sp["je"] * 1e3, "pint_e2e_ms": sp["pe"] * 1e3}
    try:
        bl_path.write_text(_json.dumps(base, indent=2))
    except Exception:
        pass
    return sp


def page_speed(pdf, D, sp, maxiter, pint_maxiter=5):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (lbl, pv, jv) in zip(axes, [
            ("END-TO-END (load+fit)", sp["e2e_p"], sp["e2e_j"]),
            ("WARM fit-only (solve)", sp["warm_p"], sp["warm_j"])]):
        vals = [np.median(pv) * 1e3, np.median(jv) * 1e3]
        ax.bar(["PINT", "JUG"], vals, color=["#c0392b", "#27ae60"])
        ax.set_ylabel("fit time (ms)")
        ax.set_title(f"{lbl}\n{np.median(pv)/np.median(jv):.1f}x faster")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
    msg = (f"{D['jname']}: speed benchmark ({len(D['mjd'])} TOAs, "
           f"both timed to chi2-convergence; PINT {pint_maxiter} iters, "
           f"JUG early-stops [cap {maxiter}], N={len(sp['e2e_p'])})")
    if sp["baseline_change"] is not None:
        chg = sp["baseline_change"]
        flag = ("[!! REGRESSION]" if chg > 0.20
                else "[. slower]" if chg > 0.02 else "[OK]")
        msg += (f"\nwarm baseline {sp['baseline_prev_ms']:.1f} ms -> "
                f"{sp['jw']*1e3:.1f} ms ({chg*100:+.1f}%) {flag}")
    else:
        msg += "\nbaseline: none previous; current run recorded"
    if not (sp["je"] < sp["pe"] and sp["jw"] < sp["pw"]):
        msg += "\nWARNING: JUG NOT faster on both metrics"
    fig.suptitle(msg, y=1.02, fontsize=9)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_rn_basis(pdf, D):
    """Red-noise Fourier basis + actual fitter prior comparison (2d-rn-basis)."""
    from jug.noise.red_noise import parse_red_noise_params

    sess = D["session"]
    jug_rn = parse_red_noise_params(sess.params)
    pint_rn = next((c for n, c in D["mf"].components.items()
                    if n == "PLRedNoise" or hasattr(c, "pl_rn_basis_weight_pair")),
                   None)
    if jug_rn is None and pint_rn is None:
        return
    L = [f"{D['jname']}: red-noise basis & fitter-prior comparison", ""]
    if jug_rn is None or pint_rn is None:
        L.append(f"ASYMMETRY: JUG RN={'present' if jug_rn else 'ABSENT'}  "
                 f"PINT PLRedNoise={'present' if pint_rn else 'ABSENT'}")
        L.append("One fitter marginalizes red noise and the other does not -- "
                 "residual diffs will contain the RN realization.")
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, "\n".join(L), fontsize=9, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        return

    mjd_arr = np.asarray([t.mjd_int + t.mjd_frac for t in sess.toas_data], float)
    jug_F, jug_phi = jug_rn.build_basis_and_prior(mjd_arr)
    jug_phi_h = jug_phi[::2]
    span_s = (mjd_arr.max() - mjd_arr.min()) * 86400.0
    jug_freqs = np.arange(1, jug_rn.n_harmonics + 1, dtype=float) / span_s

    pint_F, pint_w = pint_rn.pl_rn_basis_weight_pair(D["pint_toas"])
    pint_w_h = pint_w[::2]
    _, pint_freqs = pint_rn.get_time_frequencies(D["pint_toas"])
    p_amp, p_gam, p_nlin, p_nlog, _ = pint_rn.get_plc_vals()

    L.append(f"JUG   A={10.0**jug_rn.log10_A:.6e}  gamma={jug_rn.gamma:.6f}  "
             f"n_harm={jug_rn.n_harmonics}")
    L.append(f"PINT  A={p_amp:.6e}  gamma={p_gam:.6f}  n_lin={p_nlin}  n_log={p_nlog}")
    L.append(f"basis shapes: JUG {jug_F.shape}  PINT {pint_F.shape}")
    n = min(len(jug_phi_h), len(pint_w_h), 5)
    with np.errstate(divide="ignore", invalid="ignore"):
        L.append(f"freq ratio (first {n}): "
                 f"{np.array2string(jug_freqs[:n] / np.asarray(pint_freqs)[:n], precision=8)}")
        L.append(f"prior ratio (first {n}): "
                 f"{np.array2string(np.asarray(jug_phi_h)[:n] / np.asarray(pint_w_h)[:n], precision=8)}")
    if jug_F.shape == tuple(np.shape(pint_F)):
        cr = (np.linalg.norm(jug_F, axis=0)
              / np.maximum(np.linalg.norm(np.asarray(pint_F), axis=0), 1e-30))
        L.append(f"basis column-norm ratio: min={cr.min():.6f} max={cr.max():.6f} "
                 f"median={np.median(cr):.6f}")
    L.append("")
    L.append("ratios ~1.0 everywhere => fitter RN priors agree; any residual-diff")
    L.append("structure then comes from fitted params / solver, not the RN model.")

    fig = plt.figure(figsize=(11, 9))
    fig.text(0.05, 0.97, "\n".join(L), fontsize=8, va="top", family="monospace")
    ax1 = fig.add_axes([0.08, 0.08, 0.40, 0.48])
    ax1.loglog(jug_freqs, jug_phi_h, "C0o-", ms=3, label="JUG phi (per harm)")
    ax1.loglog(np.asarray(pint_freqs), pint_w_h, "C3s-", ms=3,
               label="PINT weight (per harm)")
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("prior variance (s^2)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    ax2 = fig.add_axes([0.57, 0.08, 0.38, 0.48])
    if len(jug_phi_h) == len(pint_w_h):
        ratio = np.asarray(jug_phi_h) / np.asarray(pint_w_h)
        ax2.semilogx(jug_freqs, ratio, "C2o-", ms=3)
        ax2.axhline(1.0, color="k", lw=0.6, ls="--")
        ax2.set_ylabel("JUG phi / PINT weight")
        ax2.set_title(f"median ratio = {np.median(ratio):.6f}", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "harmonic count mismatch", ha="center")
    ax2.set_xlabel("frequency (Hz)")
    ax2.grid(alpha=0.3, which="both")
    pdf.savefig(fig)
    plt.close(fig)


def page_covariance(pdf, D):
    """Parameter covariance comparison (2l): sigma ratios + correlations."""
    jug_cov = D["fit_result"].get("covariance")
    jug_names = list(D["fit_result"].get("final_params", {}).keys())
    for p in D["fit_result"].get("final_dmx_params", {}):
        if p not in jug_names:
            jug_names.append(p)
    pint_cov, pint_names = None, list(getattr(D["mf"], "free_params", []))
    for attr in ("parameter_covariance_matrix", "parameter_covariance"):
        obj = getattr(D["fitter"], attr, None)
        if obj is None:
            continue
        try:
            pint_cov = np.asarray(obj.matrix, dtype=float)
            labels = obj.labels
            # PINT CovarianceMatrix.labels is a list per AXIS; each axis is a
            # list of (name, (index, size, unit)) tuples. Take axis 0.
            if labels and isinstance(labels[0], list):
                labels = labels[0]
            pint_names = [str(l[0]) if isinstance(l, (tuple, list)) else str(l)
                          for l in labels]
            break
        except Exception:
            try:
                pint_cov = np.asarray(obj, dtype=float)
                break
            except Exception:
                pass
    if jug_cov is None or pint_cov is None:
        return
    jug_cov = np.asarray(jug_cov, dtype=float)
    # JUG's augmented/GLS covariance carries the OFFSET column first; align
    # names when the matrix is one larger than the named parameter list.
    if jug_cov.shape[0] == len(jug_names) + 1:
        jug_names = ["OFFSET"] + jug_names
    # PINT calls it 'Offset'; normalize for matching.
    pint_names = ["OFFSET" if n.lower() == "offset" else n for n in pint_names]

    common = [p for p in jug_names
              if p in pint_names and jug_names.index(p) < jug_cov.shape[0]
              and pint_names.index(p) < pint_cov.shape[0]]
    if not common:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.9, f"{D['jname']}: no overlapping covariance names\n"
                 f"JUG head: {jug_names[:10]}\nPINT head: {pint_names[:10]}",
                 fontsize=8, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        return

    def sig(cov, i):
        v = cov[i, i]
        return np.sqrt(v) if v >= 0 else np.nan

    rows = []
    for p in common:
        ji, pi = jug_names.index(p), pint_names.index(p)
        ju, pu = sig(jug_cov, ji), sig(pint_cov, pi)
        rows.append((p, ju, pu, ju / pu if pu and pu > 0 else np.nan))
    non_dmx = [r for r in rows if not r[0].startswith("DMX_")]
    dmx = [r for r in rows if r[0].startswith("DMX_")]

    def corr(cov, i, j):
        den = np.sqrt(cov[i, i] * cov[j, j])
        return np.nan if den <= 0 else float(cov[i, j] / den)

    pairs = []
    nd_names = [r[0] for r in non_dmx]
    for ai, a in enumerate(nd_names):
        for b in nd_names[ai + 1:]:
            cj = corr(jug_cov, jug_names.index(a), jug_names.index(b))
            cp = corr(pint_cov, pint_names.index(a), pint_names.index(b))
            if np.isfinite(cj) and np.isfinite(cp):
                pairs.append((f"{a}:{b}", cj, cp, abs(cj - cp)))
    pairs.sort(key=lambda r: -r[3])

    L = [f"{D['jname']}: covariance comparison  "
         f"({len(common)} common params, {len(dmx)} DMX)", ""]
    L.append("worst sigma ratios (non-DMX):")
    for p, ju, pu, r in sorted(non_dmx,
                               key=lambda r: -abs(np.log(r[3])) if r[3] and np.isfinite(r[3]) and r[3] > 0 else 0)[:15]:
        L.append(f"  {p:14s} sig_JUG={ju:.6e}  sig_PINT={pu:.6e}  ratio={r:.6f}")
    L.append("")
    L.append("largest off-diagonal correlation differences (non-DMX):")
    for name, cj, cp, d in pairs[:15]:
        L.append(f"  {name:28s} JUG={cj:+.6f}  PINT={cp:+.6f}  |d|={d:.2e}")
    if dmx:
        dr = np.array([r[3] for r in dmx if np.isfinite(r[3])])
        if len(dr):
            L.append("")
            L.append(f"DMX sigma ratio: median={np.median(dr):.6f}  "
                     f"5%={np.percentile(dr,5):.6f}  95%={np.percentile(dr,95):.6f}")

    fig = plt.figure(figsize=(11, 9))
    fig.text(0.05, 0.97, "\n".join(L), fontsize=7, va="top", family="monospace")
    ax = fig.add_axes([0.08, 0.07, 0.87, 0.36])
    names = [r[0] for r in non_dmx]
    ratios = [r[3] for r in non_dmx]
    ax.bar(np.arange(len(names)), ratios, color="C2")
    ax.axhline(1, color="k", lw=0.5, ls="--")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel("sigma_JUG / sigma_PINT")
    pdf.savefig(fig)
    plt.close(fig)


def run_tempo2(D, t2_bin, t2_data_dir):
    """Run Tempo2 general2 on the PINT-safe par/tim (notebook 2b)."""
    import subprocess

    if not os.path.exists(t2_bin):
        return None
    env = os.environ.copy()
    if t2_data_dir:
        env["TEMPO2"] = t2_data_dir
    try:
        result = subprocess.run(
            [t2_bin, "-f", D["pint_par_used"], D["tim"],
             "-output", "general2", "-s",
             "{sat} {freq} {post} {err} {tnrn} {tnrnerr} {tndm} {tndmerr} "
             "{tnchrom} {tnchromerr} {posttn}\n"],
            capture_output=True, text=True, env=env, timeout=600)
    except Exception as e:
        return {"error": str(e)}
    data = []
    for line in result.stdout.strip().split("\n"):
        if line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 11:
                try:
                    data.append([float(x) for x in parts[:11]])
                except ValueError:
                    pass
    if not data:
        tail = (result.stdout.strip().split("\n")[-6:]
                + result.stderr.strip().split("\n")[-4:])
        return {"error": "no parseable general2 output",
                "stderr": "\n".join(tail)}
    a = np.array(data)
    return {"mjd": a[:, 0], "freq": a[:, 1], "post_s": a[:, 2], "err": a[:, 3],
            "rn_s": a[:, 4], "dm_s": a[:, 6], "chrom_s": a[:, 8],
            "posttn_s": a[:, 10]}


def page_tempo2(pdf, D, t2):
    if t2 is None:
        return
    fig = plt.figure(figsize=(11, 9))
    if "error" in t2:
        fig.text(0.05, 0.9, f"{D['jname']}: Tempo2 cross-check FAILED\n"
                 f"{t2.get('error')}\n{t2.get('stderr','')}",
                 fontsize=8, va="top", family="monospace", color="red")
        pdf.savefig(fig)
        plt.close(fig)
        return
    plt.close(fig)

    n_match = len(t2["mjd"]) == len(D["mjd"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax = axes[0]
    ax.scatter(D["mjd"], D["jug_post_us"], s=2, alpha=0.35, label="JUG postfit")
    ax.scatter(D["mjd"], D["pint_post_us"], s=2, alpha=0.35, label="PINT postfit")
    ax.scatter(t2["mjd"], _center(t2["post_s"]) * 1e6, s=2, alpha=0.35,
               label="Tempo2 postfit")
    ax.set_ylabel("residual (us)")
    ax.legend(fontsize=8, markerscale=3)
    w = D["w"]
    title = (f"{D['jname']}: Tempo2 cross-check   "
             f"WRMS JUG={_wrms(D['jug_post_us'], w):.4f}  "
             f"PINT={_wrms(D['pint_post_us'], w):.4f}")
    if n_match:
        t2w = 1.0 / np.asarray(t2["err"]) ** 2
        title += f"  T2={_wrms(_center(t2['post_s'])*1e6, t2w):.4f} us"
    ax.set_title(title, fontsize=9)

    ax = axes[1]
    if n_match:
        d = _center(D["jug_post_us"] - _center(t2["post_s"]) * 1e6)
        ax.scatter(D["mjd"], d * 1e3, s=2, alpha=0.5, c="C3",
                   label=f"JUG - T2  std={np.std(d)*1e3:.3f} ns")
        ax.legend(fontsize=8, markerscale=3)
    else:
        ax.text(0.5, 0.5, f"TOA count mismatch: T2={len(t2['mjd'])} "
                f"JUG={len(D['mjd'])} (cuts differ) -- diff skipped",
                transform=ax.transAxes, ha="center")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("JUG - T2 (ns)")

    ax = axes[2]
    for k, lab in [("rn_s", "TN red"), ("dm_s", "TN DM"), ("chrom_s", "TN chrom")]:
        if np.std(t2[k]) > 0:
            ax.plot(t2["mjd"], t2[k] * 1e6, ".", ms=2, alpha=0.5,
                    label=f"{lab}  std={np.std(t2[k])*1e6:.3f} us")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("T2 noise realizations (us)")
    ax.set_xlabel("MJD")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, markerscale=3)
    else:
        ax.text(0.5, 0.5, "no TN noise processes in par",
                transform=ax.transAxes, ha="center", fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_worst(pdf, D):
    d_ns = D["diff_us"] * 1e3
    order = np.argsort(-np.abs(d_ns))[:12]
    comp_keys = [k for k in D["deltas_ns"] if k != "tdb"]
    L = [f"{D['jname']}: 12 worst |postfit diff| TOAs and per-component deltas (ns)",
         ""]
    hdr = f"{'MJD':>11s} {'freq':>7s} {'err(us)':>8s} {'diff':>9s} | " + \
          " ".join(f"{k[:9]:>9s}" for k in comp_keys)
    L.append(hdr)
    L.append("-" * len(hdr))
    for i in order:
        row = f"{D['mjd'][i]:11.4f} {D['freq_topo'][i]:7.1f} " \
              f"{D['err_us'][i]:8.3f} {d_ns[i]:+9.3f} | "
        row += " ".join(f"{_center(D['deltas_ns'][k])[i]:+9.3f}" for k in comp_keys)
        L.append(row)
    L.append("")
    L.append("Look for: one component matching the diff (real driver); none "
             "matching (fit redistribution / gauge); tdb driving everything "
             "(clock chain).")
    fig = plt.figure(figsize=(13, 8.5))
    fig.text(0.03, 0.95, "\n".join(L), fontsize=7, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# per-pulsar driver
# ---------------------------------------------------------------------------

def compare_pulsar(jname, par, tim, out_dir: Path, maxiter, clock_dir,
                   pint_clock_dir, signal_mode, speed_repeats=1,
                   t2_bin=None, t2_data=None, convergence=False, pint_maxiter=5):
    D = gather(jname, par, tim, maxiter, clock_dir, pint_clock_dir, signal_mode,
               convergence=convergence, pint_maxiter=pint_maxiter)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = None
    if speed_repeats > 0:
        try:
            sp = run_speed(D, maxiter, clock_dir, pint_clock_dir, speed_repeats,
                           out_dir, pint_maxiter=pint_maxiter)
        except Exception as e:
            print(f"    speed benchmark failed: {e}")
    t2 = run_tempo2(D, t2_bin, t2_data) if t2_bin else None

    pdf_path = out_dir / f"{jname}_jug_vs_pint.pdf"
    with PdfPages(pdf_path) as pdf:
        _safe_page(pdf, "summary", page_summary, pdf, D)
        _safe_page(pdf, "residuals", page_residuals, pdf, D)
        _safe_page(pdf, "distribution", page_distribution, pdf, D)
        _safe_page(pdf, "drivers", page_drivers, pdf, D)
        _safe_page(pdf, "components", page_components, pdf, D)
        _safe_page(pdf, "components_matched", page_components_matched, pdf, D)
        _safe_page(pdf, "groups", page_groups, pdf, D)
        _safe_page(pdf, "longtime", page_longtime, pdf, D)
        _safe_page(pdf, "stability", page_stability, pdf, D)
        _safe_page(pdf, "params", page_params, pdf, D)
        _safe_page(pdf, "noise_gls", page_noise_gls, pdf, D)
        _safe_page(pdf, "noise_components", page_noise_components, pdf, D)
        _safe_page(pdf, "rn_basis", page_rn_basis, pdf, D)
        _safe_page(pdf, "covariance", page_covariance, pdf, D)
        _safe_page(pdf, "dmx", page_dmx, pdf, D)
        if D["binary"]:
            _safe_page(pdf, "binary", page_binary, pdf, D)
        if sp is not None:
            _safe_page(pdf, "speed", page_speed, pdf, D, sp, maxiter,
                       pint_maxiter)
        if t2 is not None:
            _safe_page(pdf, "tempo2", page_tempo2, pdf, D, t2)
        _safe_page(pdf, "worst", page_worst, pdf, D)

    summary = {
        "jname": jname, "ntoa": len(D["mjd"]),
        "wrms_jug_us": _wrms(D["jug_post_us"], D["w"]),
        "wrms_pint_us": _wrms(D["pint_post_us"], D["w"]),
        "jug_step_sigma": D.get("jug_step_sigma", np.nan),
        "pint_step_sigma": D.get("pint_step_sigma", np.nan),
        "diff_std_ns": float(np.std(D["diff_us"])) * 1e3,
        "diff_max_ns": float(np.max(np.abs(D["diff_us"]))) * 1e3,
        "diff_spin_clean_std_ns": float(np.std(D["diff_spin_clean_us"])) * 1e3,
        "diff_spin_clean_max_ns": float(np.max(np.abs(D["diff_spin_clean_us"]))) * 1e3,
        "binary_red_ps": (float(np.std(D["binary"]["red"])) * 1e12
                          if D["binary"] else np.nan),
        "notes": D["notes"],
    }
    return summary, pdf_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/mattm/projects/jug_test_files/large_tests"))
    ap.add_argument("--maxiter", type=int, default=100,
                    help="JUG iteration CAP. JUG early-stops at convergence, so "
                         "a high cap is cheap; old default 5 under-converged "
                         "slow eccentric binaries (JUG needs ~30).")
    ap.add_argument("--pint-maxiter", type=int, default=5,
                    help="PINT iteration count (default 5). PINT's plain "
                         "GLSFitter has NO early-stop -- it runs exactly this "
                         "many full GLS solves (~5s each on a large set). It is "
                         "chi2-converged in ~1-3, so keep this low; decoupled "
                         "from --maxiter because raising it only wastes time "
                         "oscillating degenerate params.")
    ap.add_argument("--convergence", action="store_true",
                    help="compute the next-step max|dp/sigma| convergence "
                         "measure for both codes (does an extra fit per code "
                         "per pulsar -- ~2x slower). Off by default; WRMS and "
                         "GLS chi2 are always shown.")
    ap.add_argument("--clock-dir", default=str(JUG_ROOT / "data" / "clock"))
    ap.add_argument("--pint-clock-dir", default="/tmp/jug_pint_clock_override",
                    help="PINT clock override populated from --clock-dir")
    ap.add_argument("--signal-mode", choices=["component", "adjust-toas", "off"],
                    default="component")
    ap.add_argument("--only", default=None,
                    help="comma-separated JNAMEs to restrict to")
    ap.add_argument("--speed-repeats", type=int, default=1,
                    help="timed repeats for the speed page (0 = skip; "
                         "each repeat refits both codes twice)")
    ap.add_argument("--tempo2-bin",
                    default="/home/mattm/miniforge3/envs/discotech/bin/tempo2",
                    help="tempo2 binary for the cross-check page "
                         "('' to disable)")
    ap.add_argument("--tempo2-data",
                    default="/home/mattm/miniforge3/pkgs/tempo2-2025.02.1-hddb8a8a_0/share/tempo2",
                    help="TEMPO2 runtime data dir")
    args = ap.parse_args(argv)

    pairs = find_pairs(args.data_dir)
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        pairs = [p for p in pairs if p[0] in keep]
    if not pairs:
        print(f"no par/tim pairs found in {args.data_dir}")
        return 1
    print(f"{len(pairs)} pulsar(s) found; output -> {args.out}")

    results, failures = [], []
    for jname, par, tim in pairs:
        print(f"\n=== {jname} ===\n  par={par.name}  tim={tim.name}")
        try:
            summary, pdf_path = compare_pulsar(
                jname, par, tim, args.out, args.maxiter,
                args.clock_dir, args.pint_clock_dir, args.signal_mode,
                speed_repeats=args.speed_repeats,
                t2_bin=args.tempo2_bin or None,
                t2_data=args.tempo2_data or None,
                convergence=args.convergence,
                pint_maxiter=args.pint_maxiter)
            results.append(summary)
            for n in summary["notes"]:
                print(f"  NOTE: {n}")
            print(f"  WRMS JUG={summary['wrms_jug_us']:.6f} us "
                  f"PINT={summary['wrms_pint_us']:.6f} us | "
                  f"diff std={summary['diff_std_ns']:.4f} ns | "
                  f"binary red={summary['binary_red_ps']:.3f} ps "
                  f"-> {pdf_path.name}")
        except Exception as e:
            failures.append((jname, str(e)))
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print("\n========== SUMMARY ==========")
    print(f"{'pulsar':14s} {'nTOA':>6s} {'diff std (ns)':>14s} "
          f"{'binary red (ps)':>16s} {'WRMS J (us)':>12s} {'WRMS P (us)':>12s}")
    for r in sorted(results, key=lambda r: -r["diff_std_ns"]):
        print(f"{r['jname']:14s} {r['ntoa']:6d} {r['diff_std_ns']:14.4f} "
              f"{r['binary_red_ps']:16.3f} {r['wrms_jug_us']:12.6f} "
              f"{r['wrms_pint_us']:12.6f}")
    for jname, err in failures:
        print(f"{jname:14s} FAILED: {err}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
