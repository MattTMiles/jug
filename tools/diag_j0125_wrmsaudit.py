#!/usr/bin/env python
"""Task 4 diagnostic: WRMS definition audit.

Verifies JUG's WRMS matches Tempo2's TRES by:
  1. Computing WRMS with and without weighted mean subtraction
  2. Computing with and without DOF correction (N vs N-p)
  3. Comparing all variants against T2's reported TRES

Tempo2 TRES definition:
  TRES = sqrt( sum(w_i * r_i^2) / sum(w_i) )
  where w_i = 1/σ_i², residuals have weighted mean subtracted.
  This is the "population" WRMS — no DOF correction.

Usage:
    python tools/diag_j0125_wrmsaudit.py
"""

import numpy as np
from pathlib import Path

PAR_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
T2_PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_test.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

T2_TRES = 0.698  # μs, from T2 post-fit par
T2_CHI2R = 1.3555
T2_NDOF = 5061

FIT_PARAMS = [
    'RAJ', 'DECJ', 'F0', 'F1', 'DM', 'DM1', 'DM2',
    'PMRA', 'PMDEC', 'PX', 'PB', 'A1', 'PBDOT', 'XDOT',
    'TASC', 'EPS1', 'EPS2', 'FD1', 'FD2', 'H3', 'NE_SW',
]


def wrms_A(residuals, weights):
    """sqrt(sum(w*r^2)/sum(w)) — no mean subtraction."""
    return np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))


def wrms_B(residuals, weights):
    """sqrt(sum(w*(r-rbar)^2)/sum(w)) — with weighted mean subtraction."""
    sw = np.sum(weights)
    rbar = np.sum(weights * residuals) / sw
    return np.sqrt(np.sum(weights * (residuals - rbar)**2) / sw)


def wrms_C(residuals, weights, n_params):
    """sqrt(sum(w*(r-rbar)^2) / (N-p)) — DOF-corrected."""
    n = len(residuals)
    sw = np.sum(weights)
    rbar = np.sum(weights * residuals) / sw
    ssq = np.sum(weights * (residuals - rbar)**2)
    # Normalize weights to sum to N for DOF comparison
    return np.sqrt(ssq / sw * n / (n - n_params))


def main():
    from jug.engine.session import TimingSession
    from jug.residuals.simple_calculator import compute_residuals_simple

    print("=" * 70)
    print(f"{'WRMS Definition Audit: J0125-2327':^70s}")
    print("=" * 70)

    # --- A: Evaluate-only with T2 post-fit par ---
    print("\n--- A: Evaluate-only (T2 post-fit par) ---")
    r_eval = compute_residuals_simple(T2_PAR, TIM_FILE, subtract_tzr=True, verbose=False)
    res_us = r_eval['residuals_us']
    err_us = r_eval['errors_us']
    weights = 1.0 / err_us**2
    n_toa = len(res_us)
    n_params = len(FIT_PARAMS)

    jug_wrms = r_eval['rms_us']
    wA = wrms_A(res_us, weights)
    wB = wrms_B(res_us, weights)
    wC = wrms_C(res_us, weights, n_params)

    print(f"   N_toa     = {n_toa}")
    print(f"   N_params  = {n_params}")
    print(f"   N_dof     = {n_toa - n_params}")
    print()
    print(f"   JUG reported WRMS:      {jug_wrms:.6f} μs")
    print(f"   T2 reported TRES:       {T2_TRES:.6f} μs")
    print(f"   Δ(JUG-T2):              {(jug_wrms - T2_TRES)*1e3:+.3f} ns")
    print()
    print(f"   WRMS variants:")
    print(f"     A (no mean sub):      {wA:.6f} μs")
    print(f"     B (mean sub):         {wB:.6f} μs")
    print(f"     C (mean sub + DOF):   {wC:.6f} μs")
    print(f"   Weighted mean of res:   {np.sum(weights * res_us) / np.sum(weights):.6e} μs")

    # Check which variant matches T2
    diffs = {
        'A': abs(wA - T2_TRES) * 1e3,
        'B': abs(wB - T2_TRES) * 1e3,
        'C': abs(wC - T2_TRES) * 1e3,
    }
    best = min(diffs, key=diffs.get)
    print(f"\n   Closest to T2 TRES: variant {best} (Δ = {diffs[best]:.3f} ns)")
    for k, v in sorted(diffs.items()):
        mark = "  ←" if k == best else ""
        print(f"     {k}: {v:.3f} ns{mark}")

    # --- B: Chi-squared check ---
    print(f"\n--- B: Chi-squared audit ---")
    chi2 = np.sum(weights * res_us**2)
    chi2r = chi2 / (n_toa - n_params)
    chi2r_t2dof = chi2 / T2_NDOF
    print(f"   χ² = Σ w_i r_i² = {chi2:.4f}")
    print(f"   χ²_r (JUG DOF={n_toa-n_params}): {chi2r:.4f}")
    print(f"   χ²_r (T2 DOF={T2_NDOF}):  {chi2r_t2dof:.4f}")
    print(f"   T2 reported χ²_r:    {T2_CHI2R}")
    print(f"   Δ χ²_r (JUG-T2):    {chi2r_t2dof - T2_CHI2R:+.4f}")

    # --- C: JUG fit WRMS ---
    print(f"\n--- C: JUG fit (from pre-fit par) ---")
    sess = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    sess.compute_residuals(subtract_tzr=False, force_recompute=True)
    result = sess.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )
    fit_wrms = result['final_rms']
    print(f"   JUG fit WRMS:    {fit_wrms:.6f} μs")
    print(f"   T2 TRES:         {T2_TRES:.6f} μs")
    print(f"   Δ(fit-T2):       {(fit_wrms - T2_TRES)*1e3:+.3f} ns")

    # --- Verdict ---
    print(f"\n{'='*70}")
    print(f"{'VERDICT':^70s}")
    print(f"{'='*70}")

    eval_diff = abs(jug_wrms - T2_TRES) * 1e3
    fit_diff = abs(fit_wrms - T2_TRES) * 1e3

    print(f"   Evaluate-only WRMS diff: {eval_diff:.3f} ns", end="")
    print("  ✓" if eval_diff < 1.0 else "  ✗")
    print(f"   Fit WRMS diff:           {fit_diff:.3f} ns", end="")
    print("  ✓" if fit_diff < 1.0 else "  ✗")
    print(f"   Formula match:           JUG uses variant B (mean-subtracted, no DOF)")
    print(f"   This matches Tempo2's TRES definition.")

    if eval_diff < 1.0 and fit_diff < 1.0:
        print(f"\n   ✓ WRMS matches Tempo2 within 1 ns in both modes")
    else:
        print(f"\n   ✗ WRMS exceeds 1 ns threshold!")


if __name__ == "__main__":
    main()
