#!/usr/bin/env python
"""Phase 1 diagnostic: residual-level diff between JUG fitter and evaluate-only.

Compares TOA-by-TOA residuals from:
  - JUG fitter (fit from prefit par, report internal residuals)
  - JUG evaluate-only (compute_residuals_simple with T2 post-fit par)

Reports per-TOA diff, identifies outlier TOAs, and prints summary statistics.
Also checks for pulse-number mismatches (the most dangerous failure mode).

Usage:
    python tools/diag_j0125_rdiff.py
"""

import numpy as np
from pathlib import Path

PAR_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
T2_PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_test.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

FIT_PARAMS = [
    'RAJ', 'DECJ', 'F0', 'F1', 'DM', 'DM1', 'DM2',
    'PMRA', 'PMDEC', 'PX', 'PB', 'A1', 'PBDOT', 'XDOT',
    'TASC', 'EPS1', 'EPS2', 'FD1', 'FD2', 'H3', 'NE_SW',
]


def main():
    from jug.engine.session import TimingSession
    from jug.residuals.simple_calculator import compute_residuals_simple

    # === Path A: JUG fitter ===
    print("=== Path A: JUG fitter (from prefit par) ===")
    sess = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    sess.compute_residuals(subtract_tzr=False, force_recompute=True)
    fit = sess.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )
    resid_fit = fit['residuals_us']
    wrms_fit = fit['final_rms']
    print(f"  WRMS: {wrms_fit:.6f} μs")
    print(f"  N_TOA: {len(resid_fit)}")

    # === Path B: evaluate-only with T2 par ===
    print("\n=== Path B: evaluate-only (T2 post-fit par) ===")
    r_eval = compute_residuals_simple(T2_PAR, TIM_FILE, subtract_tzr=True, verbose=False)
    resid_eval = r_eval['residuals_us']
    wrms_eval = r_eval['rms_us']
    print(f"  WRMS: {wrms_eval:.6f} μs")
    print(f"  N_TOA: {len(resid_eval)}")

    # === Comparison ===
    assert len(resid_fit) == len(resid_eval), "TOA count mismatch!"
    delta = resid_fit - resid_eval  # μs
    delta_ns = delta * 1e3           # ns

    print(f"\n=== TOA-by-TOA comparison (fit - eval) ===")
    print(f"  Max |Δr|:     {np.max(np.abs(delta_ns)):.3f} ns")
    print(f"  Mean |Δr|:    {np.mean(np.abs(delta_ns)):.3f} ns")
    print(f"  Median |Δr|:  {np.median(np.abs(delta_ns)):.3f} ns")
    print(f"  Std Δr:       {np.std(delta_ns):.3f} ns")

    # WRMS comparison
    print(f"\n=== WRMS comparison ===")
    print(f"  Fitter WRMS:      {wrms_fit:.6f} μs")
    print(f"  Eval-only WRMS:   {wrms_eval:.6f} μs")
    print(f"  WRMS difference:  {(wrms_fit - wrms_eval)*1e3:.3f} ns")

    # Tempo2 reference
    print(f"  Tempo2 TRES:      0.698 μs")
    print(f"  Fitter vs T2:     {(wrms_fit - 0.698)*1e3:.3f} ns")

    # Outliers
    threshold_ns = 10.0
    outlier_mask = np.abs(delta_ns) > threshold_ns
    n_outliers = np.sum(outlier_mask)
    print(f"\n=== Outliers (|Δr| > {threshold_ns} ns) ===")
    print(f"  Count: {n_outliers} / {len(delta_ns)}")
    if n_outliers > 0:
        idx = np.where(outlier_mask)[0]
        print(f"  Indices: {idx[:20]}")
        for i in idx[:10]:
            print(f"    TOA {i}: fit={resid_fit[i]:.6f} μs, eval={resid_eval[i]:.6f} μs, Δ={delta_ns[i]:.3f} ns")

    # Pulse-number check
    print(f"\n=== Pulse-number safety check ===")
    F0_approx = 272.08  # Hz
    pulse_period_us = 1e6 / F0_approx  # ~3675 μs
    catastrophic = np.abs(delta) > 0.5 * pulse_period_us
    n_catastrophic = np.sum(catastrophic)
    print(f"  Catastrophic outliers (|Δr| > {0.5*pulse_period_us:.0f} μs): {n_catastrophic}")
    if n_catastrophic > 0:
        print(f"  *** PULSE NUMBER MISMATCH DETECTED ***")
    else:
        print(f"  ✓ No pulse-number mismatches")

    # _high_precision cache check
    print(f"\n=== _high_precision cache diagnostic ===")
    from jug.io.par_reader import parse_par_file, get_longdouble
    prefit = parse_par_file(PAR_FILE)
    t2 = parse_par_file(T2_PAR)
    for key in ['F0', 'F1', 'PEPOCH']:
        ld_pre = get_longdouble(prefit, key)
        ld_t2 = get_longdouble(t2, key)
        hp_str = prefit.get('_high_precision', {}).get(key, 'N/A')
        flt = prefit.get(key, 'N/A')
        print(f"  {key}:")
        print(f"    prefit float64:  {flt!r}")
        print(f"    prefit _hp str:  {hp_str!r}")
        print(f"    prefit longdbl:  {ld_pre}")
        print(f"    T2 longdbl:      {ld_t2}")
        print(f"    Diff:            {float(ld_t2 - ld_pre):.15e}")


if __name__ == "__main__":
    main()
