#!/usr/bin/env python
"""Task 5: Final reproducibility script — JUG vs Tempo2 parity for J0125-2327.

Three-mode WRMS comparison:
  Mode A: Evaluate-only with T2 post-fit par (no fitting)
  Mode B: Fit from pre-fit par → compare postfit WRMS
  Mode C: Fit from post-fit par → confirm convergence at same minimum

All three modes should produce WRMS within <1 ns of Tempo2's TRES = 0.698 μs.

Usage:
    python tools/diag_j0125_final.py
"""

import time
import numpy as np
from pathlib import Path

PREFIT_PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
POSTFIT_PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_test.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

T2_TRES_POSTFIT = 0.698   # μs, from T2 post-fit par
T2_TRES_PREFIT = 0.705    # μs, from T2 pre-fit par
T2_CHI2R = 1.3555
T2_NTOA = 5083
T2_NDOF = 5061

FIT_PARAMS = [
    'RAJ', 'DECJ', 'F0', 'F1', 'DM', 'DM1', 'DM2',
    'PMRA', 'PMDEC', 'PX', 'PB', 'A1', 'PBDOT', 'XDOT',
    'TASC', 'EPS1', 'EPS2', 'FD1', 'FD2', 'H3', 'NE_SW',
]


def main():
    from jug.engine.session import TimingSession
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.io.par_reader import parse_par_file, parse_ra, parse_dec

    print("=" * 72)
    print(f"{'JUG vs Tempo2 Reproducibility Report: J0125-2327':^72s}")
    print("=" * 72)
    print(f"  Tempo2 TRES (post-fit): {T2_TRES_POSTFIT:.3f} μs")
    print(f"  Tempo2 TRES (pre-fit):  {T2_TRES_PREFIT:.3f} μs")
    print(f"  Tempo2 χ²_r:            {T2_CHI2R}")
    print(f"  Tempo2 NTOA:            {T2_NTOA}")
    print(f"  Fitted parameters:      {len(FIT_PARAMS)}")
    print()

    results = {}

    # ================================================================
    # MODE A: Evaluate-only with T2 post-fit par
    # ================================================================
    print("-" * 72)
    print("MODE A: Evaluate-only with Tempo2 post-fit par")
    print("-" * 72)
    t0 = time.time()
    r_eval = compute_residuals_simple(POSTFIT_PAR, TIM_FILE, subtract_tzr=True, verbose=False)
    dt = time.time() - t0
    wrms_A = r_eval['rms_us']
    n_toa_A = len(r_eval['residuals_us'])
    print(f"  N_toa:    {n_toa_A}")
    print(f"  WRMS:     {wrms_A:.6f} μs")
    print(f"  T2 TRES:  {T2_TRES_POSTFIT:.6f} μs")
    delta_A = (wrms_A - T2_TRES_POSTFIT) * 1e3
    print(f"  Δ:        {delta_A:+.3f} ns")
    print(f"  Time:     {dt:.2f} s")
    results['A'] = {'wrms': wrms_A, 'delta_ns': delta_A}

    # ================================================================
    # MODE B: Fit from pre-fit par
    # ================================================================
    print()
    print("-" * 72)
    print("MODE B: Fit from Tempo2 pre-fit par")
    print("-" * 72)
    t0 = time.time()
    sess_B = TimingSession(PREFIT_PAR, TIM_FILE, verbose=False)
    sess_B.compute_residuals(subtract_tzr=False, force_recompute=True)
    result_B = sess_B.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )
    dt = time.time() - t0
    wrms_B = result_B['final_rms']
    print(f"  Converged: {result_B['converged']} ({result_B['iterations']} iterations)")
    print(f"  Prefit WRMS:  {result_B.get('prefit_rms', 'N/A')} μs")
    print(f"  Postfit WRMS: {wrms_B:.6f} μs")
    print(f"  T2 TRES:      {T2_TRES_POSTFIT:.6f} μs")
    delta_B = (wrms_B - T2_TRES_POSTFIT) * 1e3
    print(f"  Δ:            {delta_B:+.3f} ns")
    print(f"  Time:         {dt:.2f} s")
    results['B'] = {'wrms': wrms_B, 'delta_ns': delta_B, 'converged': result_B['converged']}

    # ================================================================
    # MODE C: Fit from post-fit par (should converge in ≤2 iterations)
    # ================================================================
    print()
    print("-" * 72)
    print("MODE C: Fit from Tempo2 post-fit par (re-fit)")
    print("-" * 72)
    t0 = time.time()
    sess_C = TimingSession(POSTFIT_PAR, TIM_FILE, verbose=False)
    sess_C.compute_residuals(subtract_tzr=False, force_recompute=True)
    result_C = sess_C.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )
    dt = time.time() - t0
    wrms_C = result_C['final_rms']
    print(f"  Converged: {result_C['converged']} ({result_C['iterations']} iterations)")
    print(f"  Postfit WRMS: {wrms_C:.6f} μs")
    print(f"  T2 TRES:      {T2_TRES_POSTFIT:.6f} μs")
    delta_C = (wrms_C - T2_TRES_POSTFIT) * 1e3
    print(f"  Δ:            {delta_C:+.3f} ns")
    print(f"  Time:         {dt:.2f} s")
    results['C'] = {'wrms': wrms_C, 'delta_ns': delta_C, 'converged': result_C['converged']}

    # ================================================================
    # Cross-mode consistency
    # ================================================================
    print()
    print("-" * 72)
    print("CROSS-MODE CONSISTENCY")
    print("-" * 72)
    print(f"  |B - A| (fit vs eval):     {abs(wrms_B - wrms_A)*1e3:.3f} ns")
    print(f"  |C - A| (refit vs eval):   {abs(wrms_C - wrms_A)*1e3:.3f} ns")
    print(f"  |B - C| (fit vs refit):    {abs(wrms_B - wrms_C)*1e3:.3f} ns")

    # ================================================================
    # Parameter consistency: Mode B vs Mode C
    # ================================================================
    print()
    print("-" * 72)
    print("PARAMETER CONSISTENCY: Mode B vs Mode C")
    print("-" * 72)
    fp_B = result_B['final_params']
    fp_C = result_C['final_params']
    unc_B = result_B.get('uncertainties', {})

    print(f"  {'Param':>8s} {'|B-C|':>14s} {'|B-C|/σ_B':>12s}")
    print(f"  {'-'*40}")
    max_sig = 0.0
    for p in FIT_PARAMS:
        vB = float(fp_B.get(p, 0))
        vC = float(fp_C.get(p, 0))
        uB = float(unc_B.get(p, 0))
        delta = abs(vB - vC)
        sig = delta / uB if uB > 0 else float('inf')
        max_sig = max(max_sig, sig)
        print(f"  {p:>8s} {delta:>14.6e} {sig:>12.6f}")
    print(f"  {'-'*40}")
    print(f"  Max |B-C|/σ: {max_sig:.6f}")

    # ================================================================
    # VERDICT
    # ================================================================
    print()
    print("=" * 72)
    print(f"{'FINAL VERDICT':^72s}")
    print("=" * 72)

    all_pass = True
    for mode, data in sorted(results.items()):
        d = abs(data['delta_ns'])
        status = "✓ PASS" if d < 1.0 else "✗ FAIL"
        if d >= 1.0:
            all_pass = False
        print(f"  Mode {mode}: WRMS = {data['wrms']:.6f} μs  |Δ| = {d:.3f} ns  {status}")

    print()
    if all_pass:
        print("  ════════════════════════════════════════════════════════")
        print("  ✓  ALL MODES MATCH TEMPO2 WITHIN 1 ns")
        print("  ════════════════════════════════════════════════════════")
    else:
        print("  ════════════════════════════════════════════════════════")
        print("  ✗  SOME MODES EXCEED 1 ns THRESHOLD")
        print("  ════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
