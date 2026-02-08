#!/usr/bin/env python
"""Tempo2 ↔ JUG per-TOA parity harness.

Runs both Tempo2 and JUG on the same par+tim, joins residuals TOA-by-TOA,
reports statistics, and saves a golden artifact for regression testing.

Usage:
    python tools/parity_harness.py                         # J0125-2327 default
    python tools/parity_harness.py --par X.par --tim X.tim # custom pulsar

Requirements:
    - tempo2 must be on PATH (conda environment 'discotech')
    - JUG must be importable

Outputs:
    - Console report with N matched, max|Δ|, RMS(Δ), percentiles, worst TOAs
    - Golden artifact saved to tests/data_golden/<PSRJ>_parity.npz
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def run_tempo2(par_path: Path, tim_path: Path, verbose: bool = False) -> dict:
    """Run Tempo2 and extract per-TOA residuals via general2 plugin.

    Parameters
    ----------
    par_path, tim_path : Path
        Paths to par and tim files.
    verbose : bool
        Print Tempo2 stderr/warnings.

    Returns
    -------
    dict with keys:
        'bat'  : np.ndarray  — barycentric arrival time (MJD, float64)
        'freq' : np.ndarray  — observing frequency (MHz)
        'post' : np.ndarray  — post-fit residual (seconds)
        'err'  : np.ndarray  — TOA uncertainty (μs)
        'n_toa': int
    """
    fmt = "{bat} {freq} {post} {err}\\n"
    cmd = [
        "tempo2",
        "-f", str(par_path), str(tim_path),
        "-output", "general2",
        "-s", fmt,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
        cwd=str(par_path.parent),
    )

    if result.returncode != 0 and verbose:
        print(f"Tempo2 stderr:\n{result.stderr[:2000]}", file=sys.stderr)

    # Parse output — data lines start with a digit (MJD)
    bats, freqs, posts, errs = [], [], [], []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        bats.append(float(parts[0]))
        freqs.append(float(parts[1]))
        posts.append(float(parts[2]))
        errs.append(float(parts[3]))

    # If stdout was empty, try stderr (some Tempo2 builds write data to stderr)
    if not bats:
        for line in result.stderr.splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            bats.append(float(parts[0]))
            freqs.append(float(parts[1]))
            posts.append(float(parts[2]))
            errs.append(float(parts[3]))

    if not bats:
        raise RuntimeError(
            f"No residual data from Tempo2.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    return {
        'bat': np.array(bats, dtype=np.float64),
        'freq': np.array(freqs, dtype=np.float64),
        'post': np.array(posts, dtype=np.float64),   # seconds
        'err': np.array(errs, dtype=np.float64),      # μs
        'n_toa': len(bats),
    }


def run_jug(par_path: Path, tim_path: Path) -> dict:
    """Run JUG evaluate-only and extract per-TOA residuals.

    Returns
    -------
    dict with keys:
        'bat'  : np.ndarray  — barycentric arrival time (MJD, float64)
        'freq' : np.ndarray  — barycentric frequency (MHz)
        'post' : np.ndarray  — post-fit residual (seconds)
        'err'  : np.ndarray  — TOA uncertainty (μs)
        'wrms' : float       — weighted RMS (μs)
        'n_toa': int
    """
    from jug.residuals.simple_calculator import compute_residuals_simple

    r = compute_residuals_simple(
        par_path, tim_path, subtract_tzr=True, verbose=False
    )

    return {
        'bat': r['tdb_mjd'],
        'freq': r['freq_bary_mhz'],
        'post': r['residuals_us'] * 1e-6,  # μs → seconds
        'err': r['errors_us'],
        'wrms': r['rms_us'],
        'n_toa': r['n_toas'],
    }


def match_toas(
    t2: dict, jug: dict,
    bat_tol_days: float = 0.01,  # ~14 min — sanity check only
) -> dict:
    """Join Tempo2 and JUG TOAs by index order.

    Both codes process the .tim file in order, so index-based matching
    is reliable. BAT values may differ by a few ms due to independent
    clock correction chains (Tempo2 vs JUG/Astropy), so we use a loose
    sanity check rather than tight BAT matching.

    Parameters
    ----------
    t2, jug : dict
        Output from run_tempo2 / run_jug.
    bat_tol_days : float
        Maximum allowed BAT difference for sanity check (days).
        Default 0.01 days (~14 min) catches gross ordering errors.

    Returns
    -------
    dict with keys:
        'n_matched' : int
        'bat_t2'    : np.ndarray
        'bat_jug'   : np.ndarray
        'delta_res_ns': np.ndarray  — residual difference in nanoseconds
        'delta_bat_days': np.ndarray  — BAT difference in days
        'freq'      : np.ndarray
        'err'       : np.ndarray
    """
    n_t2 = t2['n_toa']
    n_jug = jug['n_toa']

    if n_t2 != n_jug:
        raise ValueError(
            f"TOA count mismatch: Tempo2={n_t2}, JUG={n_jug}. "
            "Cannot proceed with index-based matching."
        )

    # Sanity check: BATs should be within tolerance (catches ordering errors)
    delta_bat = np.abs(t2['bat'] - jug['bat'])
    max_bat_diff = np.max(delta_bat)
    if max_bat_diff > bat_tol_days:
        worst_idx = np.argmax(delta_bat)
        raise ValueError(
            f"BAT sanity check failed: max ΔBAT = {max_bat_diff:.6e} days "
            f"at index {worst_idx} (tolerance = {bat_tol_days} days). "
            "TOA ordering may differ."
        )

    # Residual difference in nanoseconds
    delta_res_ns = (jug['post'] - t2['post']) * 1e9

    return {
        'n_matched': n_t2,
        'bat_t2': t2['bat'],
        'bat_jug': jug['bat'],
        'delta_res_ns': delta_res_ns,
        'delta_bat_days': t2['bat'] - jug['bat'],
        'freq': t2['freq'],
        'err': t2['err'],
    }


def report(match: dict, t2: dict, jug: dict, psrj: str) -> dict:
    """Print statistics and return summary dict."""
    d = match['delta_res_ns']
    abs_d = np.abs(d)

    # WRMS of Tempo2 residuals
    w = 1.0 / match['err']**2
    t2_wrms = np.sqrt(np.sum(w * (t2['post'] * 1e6)**2) / np.sum(w))

    stats = {
        'psrj': psrj,
        'n_matched': match['n_matched'],
        'max_abs_delta_ns': float(np.max(abs_d)),
        'rms_delta_ns': float(np.sqrt(np.mean(d**2))),
        'mean_delta_ns': float(np.mean(d)),
        'p50_ns': float(np.percentile(abs_d, 50)),
        'p90_ns': float(np.percentile(abs_d, 90)),
        'p99_ns': float(np.percentile(abs_d, 99)),
        'wrms_jug_us': float(jug['wrms']),
        'wrms_t2_us': float(t2_wrms),
        'wrms_diff_ns': float((jug['wrms'] - t2_wrms) * 1e3),
    }

    print(f"\n{'='*72}")
    print(f"{'Tempo2 ↔ JUG Parity Report: ' + psrj:^72s}")
    print(f"{'='*72}")
    print(f"  TOAs matched:     {stats['n_matched']}")
    print(f"  max|Δ|:           {stats['max_abs_delta_ns']:.3f} ns")
    print(f"  RMS(Δ):           {stats['rms_delta_ns']:.3f} ns")
    print(f"  mean(Δ):          {stats['mean_delta_ns']:.3f} ns")
    print(f"  P50|Δ|:           {stats['p50_ns']:.3f} ns")
    print(f"  P90|Δ|:           {stats['p90_ns']:.3f} ns")
    print(f"  P99|Δ|:           {stats['p99_ns']:.3f} ns")
    print(f"  WRMS JUG:         {stats['wrms_jug_us']:.6f} μs")
    print(f"  WRMS T2:          {stats['wrms_t2_us']:.6f} μs")
    print(f"  |ΔWRMS|:          {abs(stats['wrms_diff_ns']):.3f} ns")

    # Worst 20 TOAs
    order = np.argsort(abs_d)[::-1]
    print(f"\n  Worst 20 TOAs:")
    print(f"  {'#':>4s} {'Index':>6s} {'BAT (MJD)':>20s} {'Freq (MHz)':>12s} {'Δ (ns)':>12s} {'err (μs)':>10s}")
    for rank, idx in enumerate(order[:20]):
        print(
            f"  {rank+1:>4d} {idx:>6d} {match['bat_t2'][idx]:>20.10f} "
            f"{match['freq'][idx]:>12.3f} {d[idx]:>+12.3f} {match['err'][idx]:>10.4f}"
        )

    return stats


def save_golden(
    match: dict, stats: dict, t2: dict, jug: dict,
    output_dir: Path, psrj: str,
):
    """Save golden artifact for regression testing.

    Saves:
      - <psrj>_parity.npz  — compact binary with per-TOA data
      - <psrj>_parity.json — human-readable summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # NPZ: per-TOA arrays
    npz_path = output_dir / f"{psrj}_parity.npz"
    np.savez_compressed(
        npz_path,
        bat_t2=match['bat_t2'],
        bat_jug=match['bat_jug'],
        freq=match['freq'],
        err=match['err'],
        delta_res_ns=match['delta_res_ns'],
        jug_res_sec=jug['post'],
        t2_res_sec=t2['post'],
    )

    # JSON: summary
    json_path = output_dir / f"{psrj}_parity.json"
    summary = {
        '_comment': f'Golden parity data for {psrj}',
        '_generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        '_update_instructions': (
            'Regenerate with: python tools/parity_harness.py '
            f'--par <par_file> --tim <tim_file>'
        ),
        **stats,
        'thresholds': {
            'max_abs_delta_ns': 50.0,   # per-TOA: clock chain differences ~25 ns
            'wrms_diff_ns': 1.0,        # WRMS: must agree within 1 ns
        },
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {npz_path}")
    print(f"  Saved: {json_path}")
    return npz_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description="Tempo2 ↔ JUG per-TOA parity check"
    )
    parser.add_argument(
        "--par", type=Path,
        default=Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_test.par"),
        help="Post-fit par file",
    )
    parser.add_argument(
        "--tim", type=Path,
        default=Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim"),
        help="TIM file",
    )
    parser.add_argument(
        "--psrj", type=str, default=None,
        help="Pulsar name (auto-detected from par file if omitted)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "data_golden",
        help="Directory for golden artifacts",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving golden artifacts",
    )
    parser.add_argument(
        "--assert-thresholds", action="store_true",
        help="Exit non-zero if thresholds are exceeded",
    )
    args = parser.parse_args()

    # Auto-detect PSRJ
    psrj = args.psrj
    if psrj is None:
        with open(args.par) as f:
            for line in f:
                if line.startswith('PSRJ'):
                    psrj = line.split()[1]
                    break
        if psrj is None:
            psrj = args.par.stem

    # Config fingerprint check
    from jug.testing.fingerprint import (
        extract_fingerprint, validate_jug_compatible, fingerprint_report
    )
    fp = extract_fingerprint(args.par)
    ok, issues = validate_jug_compatible(fp)
    if not ok:
        print("FATAL: Par file is not JUG-compatible:")
        for issue in issues:
            print(f"  ✗ {issue}")
        sys.exit(1)
    print(fingerprint_report(args.par))

    # Run both codes
    print(f"\nRunning Tempo2...")
    t0 = time.time()
    t2 = run_tempo2(args.par, args.tim, verbose=True)
    print(f"  {t2['n_toa']} TOAs in {time.time()-t0:.1f}s")

    print(f"Running JUG...")
    t0 = time.time()
    jug_r = run_jug(args.par, args.tim)
    print(f"  {jug_r['n_toa']} TOAs in {time.time()-t0:.1f}s")

    # Match and report
    match = match_toas(t2, jug_r)
    stats = report(match, t2, jug_r, psrj)

    # Save golden artifacts
    if not args.no_save:
        save_golden(match, stats, t2, jug_r, args.output_dir, psrj)

    # Assert thresholds
    # Per-TOA: clock chain differences (Tempo2 vs JUG/Astropy) cause ~16 ns
    # systematic offset; 50 ns threshold catches regressions without being
    # unrealistically tight. WRMS must agree within 1 ns.
    if args.assert_thresholds:
        fail = False
        if stats['max_abs_delta_ns'] >= 50.0:
            print(f"\n✗ FAIL: max|Δ| = {stats['max_abs_delta_ns']:.3f} ns >= 50 ns")
            fail = True
        if abs(stats['wrms_diff_ns']) >= 1.0:
            print(f"✗ FAIL: |ΔWRMS| = {abs(stats['wrms_diff_ns']):.3f} ns >= 1.0 ns")
            fail = True
        if fail:
            sys.exit(1)
        else:
            print(f"\n✓ All thresholds passed")


if __name__ == "__main__":
    main()
