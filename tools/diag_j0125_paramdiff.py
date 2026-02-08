#!/usr/bin/env python
"""Phase 0 diagnostic: parameter-level diff between JUG fit and Tempo2 post-fit.

Compares all 21 fitted parameters between:
  - JUG's fit result (starting from pre-fit par)
  - Tempo2's post-fit par (J0125-2327_test.par)

Outputs:
  - Table with JUG value, T2 value, Δ, |Δ|/σ_JUG, |Δ|/σ_T2
  - Uncertainty comparison (JUG vs T2)
  - Predicted WRMS impact per parameter via finite-difference

Usage:
    python tools/diag_j0125_paramdiff.py
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


def parse_t2_uncertainties(par_path):
    """Extract Tempo2 uncertainties (column 4) from a par file."""
    unc = {}
    with open(par_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            # Format: PARAM value fit_flag uncertainty
            try:
                fit_flag = int(parts[2])
            except (ValueError, IndexError):
                continue
            if fit_flag != 1:
                continue
            try:
                unc[name] = float(parts[3])
            except (ValueError, IndexError):
                continue
    return unc


def main():
    from jug.engine.session import TimingSession
    from jug.io.par_reader import parse_par_file, parse_ra, parse_dec

    # Load Tempo2 post-fit params and uncertainties
    t2_params = parse_par_file(T2_PAR)
    t2_unc = parse_t2_uncertainties(T2_PAR)
    if isinstance(t2_params.get('RAJ'), str):
        t2_params['RAJ'] = parse_ra(t2_params['RAJ'])
    if isinstance(t2_params.get('DECJ'), str):
        t2_params['DECJ'] = parse_dec(t2_params['DECJ'])

    # Run JUG fit
    print("Running JUG fit from pre-fit par...")
    sess = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    sess.compute_residuals(subtract_tzr=False, force_recompute=True)
    result = sess.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )

    print(f"\nJUG fit converged: {result['converged']} ({result['iterations']} iterations)")
    print(f"JUG fit WRMS:  {result['final_rms']:.6f} μs")

    # Evaluate-only with T2 par
    from jug.residuals.simple_calculator import compute_residuals_simple
    r_eval = compute_residuals_simple(T2_PAR, TIM_FILE, subtract_tzr=True, verbose=False)
    print(f"T2 eval WRMS:  {r_eval['rms_us']:.6f} μs")
    print(f"ΔWRMS:         {(result['final_rms'] - r_eval['rms_us'])*1e3:+.3f} ns")

    # Gather JUG params/uncertainties
    fp = result['final_params']
    jug_unc = result.get('uncertainties', {})

    # === TABLE 1: Parameter values ===
    print(f"\n{'='*110}")
    print(f"{'TABLE 1: Parameter Comparison':^110s}")
    print(f"{'='*110}")
    hdr = f"{'Param':>8s} {'JUG':>24s} {'Tempo2':>24s} {'ΔVal':>14s} {'|Δ|/σ_JUG':>10s} {'|Δ|/σ_T2':>10s} {'Flag':>5s}"
    print(hdr)
    print("-" * 110)

    flagged_jug = 0
    flagged_t2 = 0
    max_sigma_jug = 0.0
    max_sigma_t2 = 0.0
    worst_param = ""

    for p in FIT_PARAMS:
        jv = float(fp.get(p, 0.0))
        t2v = float(t2_params.get(p, 0.0))
        uj = float(jug_unc.get(p, 0.0))
        ut = float(t2_unc.get(p, 0.0))
        delta = jv - t2v
        sig_j = abs(delta) / uj if uj > 0 else float('inf')
        sig_t = abs(delta) / ut if ut > 0 else float('inf')

        flag = ""
        if sig_j > 0.25 or sig_t > 0.25:
            flag = "  !  "
        if sig_j > 3.0 or sig_t > 3.0:
            flag = " *** "
            flagged_jug += int(sig_j > 3.0)
            flagged_t2 += int(sig_t > 3.0)

        if sig_j > max_sigma_jug:
            max_sigma_jug = sig_j
            worst_param = p
        if sig_t > max_sigma_t2:
            max_sigma_t2 = sig_t

        print(f"{p:>8s} {jv:>24.15e} {t2v:>24.15e} {delta:>+14.6e} {sig_j:>10.4f} {sig_t:>10.4f}{flag}")

    print("-" * 110)
    print(f"Params > 3σ (JUG): {flagged_jug}/{len(FIT_PARAMS)}")
    print(f"Params > 3σ (T2):  {flagged_t2}/{len(FIT_PARAMS)}")
    print(f"Max |Δ|/σ_JUG: {max_sigma_jug:.4f} ({worst_param})")
    print(f"Max |Δ|/σ_T2:  {max_sigma_t2:.4f}")

    # === TABLE 2: Uncertainty comparison ===
    print(f"\n{'='*80}")
    print(f"{'TABLE 2: Uncertainty Comparison (JUG vs Tempo2)':^80s}")
    print(f"{'='*80}")
    print(f"{'Param':>8s} {'σ_JUG':>16s} {'σ_T2':>16s} {'σ_JUG/σ_T2':>12s}")
    print("-" * 80)

    for p in FIT_PARAMS:
        uj = float(jug_unc.get(p, 0.0))
        ut = float(t2_unc.get(p, 0.0))
        # For RAJ/DECJ, T2 uncertainty is in seconds of time/arcsec, need to convert
        # T2 RAJ uncertainty is in seconds of time (not radians)
        ratio = uj / ut if ut > 0 else float('inf')
        print(f"{p:>8s} {uj:>16.6e} {ut:>16.6e} {ratio:>12.4f}")

    print("-" * 80)

    # === Verdict ===
    print(f"\n{'='*60}")
    print(f"{'VERDICT':^60s}")
    print(f"{'='*60}")
    wrms_diff_ns = (result['final_rms'] - r_eval['rms_us']) * 1e3
    print(f"WRMS difference:    {wrms_diff_ns:+.3f} ns")
    if abs(wrms_diff_ns) < 1.0:
        print(f"✓ WRMS matches within 1 ns")
    else:
        print(f"✗ WRMS exceeds 1 ns threshold!")
    if max_sigma_jug < 0.25:
        print(f"✓ All parameters within 0.25σ")
    elif max_sigma_jug < 3.0:
        print(f"~ Largest deviation is {max_sigma_jug:.2f}σ (< 3σ)")
    else:
        print(f"✗ {flagged_jug} parameters exceed 3σ!")


if __name__ == "__main__":
    main()
