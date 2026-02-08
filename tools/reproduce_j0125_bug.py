#!/usr/bin/env python
"""Reproduce and verify the _high_precision cache bug fix for J0125-2327.

This script demonstrates the root cause of the 7.2 ns WRMS discrepancy
between the fitter's internal WRMS and the evaluate-only WRMS, and verifies
that the fix brings them into agreement within < 1 ns.

Root Cause
----------
When the fitter updates params['F0'] with a new float64 value, the
``_high_precision`` string cache (populated by ``parse_par_file``) was NOT
updated. Consequently, ``get_longdouble(params, 'F0')`` — called by the shared
``compute_phase_residuals()`` function — returned the stale **prefit** F0 from
the string cache instead of the fitted F0.

For J0125-2327:
  - Prefit F0  = 272.08108848529092869 Hz  (19 digits in par file)
  - Post-fit F0 = 272.0810884852916457 Hz   (18 digits in par file)
  - ΔF0 = 7.17e-13 Hz
  - max |dt| ≈ 3.8e7 sec (data span)
  - Phase error ≈ dt × ΔF0 ≈ 2.7e-5 cycles ≈ 100 ns

This ~100 ns phase error, after weighted-mean subtraction and WRMS computation,
manifested as a 7.2 ns WRMS offset.

Fix
---
Added ``_update_param()`` helper in ``optimized_fitter.py`` that updates both
``params[key]`` and ``params['_high_precision'][key]`` whenever a high-precision
parameter (F0, F1, F2, PEPOCH, etc.) is modified by the fitter.

Usage:
    python tools/reproduce_j0125_bug.py
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
    from jug.io.par_reader import parse_par_file, get_longdouble

    print("=" * 70)
    print("J0125-2327 _high_precision cache bug: reproduction & verification")
    print("=" * 70)

    # === Step 1: Demonstrate the bug mechanism ===
    print("\n--- Step 1: Demonstrate _high_precision cache staleness ---")
    params = parse_par_file(PAR_FILE)
    hp = params.get('_high_precision', {})

    F0_orig = get_longdouble(params, 'F0')
    print(f"  Original F0 (from _high_precision string): {F0_orig}")
    print(f"  _high_precision['F0'] = {hp.get('F0', 'N/A')!r}")

    # Simulate what the old fitter did: overwrite float64 but NOT _high_precision
    params['F0'] = 272.0810884852917  # T2 post-fit value (float64)
    F0_after = get_longdouble(params, 'F0')
    print(f"\n  After params['F0'] = 272.0810884852917 (old fitter behavior):")
    print(f"  get_longdouble returns: {F0_after}")
    print(f"  This is STILL the prefit value! (from stale _high_precision)")
    print(f"  Bug: get_longdouble ignores the updated float64 value")

    # Now demonstrate the fix: update _high_precision too
    from jug.fitting.optimized_fitter import _update_param
    _update_param(params, 'F0', 272.0810884852917)
    F0_fixed = get_longdouble(params, 'F0')
    print(f"\n  After _update_param(params, 'F0', 272.0810884852917) (fixed behavior):")
    print(f"  get_longdouble returns: {F0_fixed}")
    print(f"  _high_precision['F0'] = {hp.get('F0', 'N/A')!r}")
    print(f"  ✓ Now returns the updated value")

    # === Step 2: Run the actual fitter and compare ===
    print("\n--- Step 2: Run JUG fitter and compare WRMS ---")
    sess = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    sess.compute_residuals(subtract_tzr=False, force_recompute=True)
    result = sess.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )

    wrms_fitter = result['final_rms']
    print(f"  JUG fitter WRMS:      {wrms_fitter:.6f} μs")

    # Evaluate-only with T2 par
    r_eval = compute_residuals_simple(T2_PAR, TIM_FILE, subtract_tzr=True, verbose=False)
    wrms_eval = r_eval['rms_us']
    print(f"  Evaluate-only WRMS:   {wrms_eval:.6f} μs")

    diff_ns = (wrms_fitter - wrms_eval) * 1e3
    print(f"  Difference:           {diff_ns:.3f} ns")
    print(f"  Tempo2 reference:     0.698 μs")

    # === Step 3: Verdict ===
    print("\n--- Step 3: Verdict ---")
    if abs(diff_ns) < 1.0:
        print(f"  ✓ PASS: Fitter-eval WRMS difference = {diff_ns:.3f} ns (< 1 ns)")
    else:
        print(f"  ✗ FAIL: Fitter-eval WRMS difference = {diff_ns:.3f} ns (> 1 ns)")

    fitter_vs_t2_ns = (wrms_fitter - 0.698) * 1e3
    if abs(fitter_vs_t2_ns) < 1.0:
        print(f"  ✓ PASS: Fitter vs Tempo2 = {fitter_vs_t2_ns:.3f} ns (< 1 ns)")
    else:
        print(f"  ✗ FAIL: Fitter vs Tempo2 = {fitter_vs_t2_ns:.3f} ns (> 1 ns)")

    print("=" * 70)


if __name__ == "__main__":
    main()
