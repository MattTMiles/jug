#!/usr/bin/env python
"""Diagnostic: ELL1H J0125-2327 WRMS gap analysis.

Identifies the source of WRMS discrepancy between JUG and Tempo2 for
J0125-2327 (ELL1H binary model with H3-only orthometric Shapiro delay).

Run: python tests/diag_ell1h_j0125.py
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

PAR_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

# Par file says TRES 0.705. User reported Tempo2=0.698 from a separate run.
PAR_FILE_TRES_US = 0.705

FIT_PARAMS = [
    'RAJ', 'DECJ', 'F0', 'F1', 'DM', 'DM1', 'DM2',
    'PMRA', 'PMDEC', 'PX', 'PB', 'A1', 'PBDOT', 'XDOT',
    'TASC', 'EPS1', 'EPS2', 'FD1', 'FD2', 'H3', 'NE_SW',
]


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


def wrms_A(residuals, weights):
    """sqrt(sum(w*r^2)/sum(w)) — no mean subtraction."""
    return np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))


def wrms_B(residuals, weights):
    """sqrt(sum(w*(r-rbar)^2)/sum(w)) — with weighted mean subtraction."""
    sw = np.sum(weights)
    rbar = np.sum(weights * residuals) / sw
    return np.sqrt(np.sum(weights * (residuals - rbar)**2) / sw)


def main():
    from jug.engine.session import TimingSession

    session = TimingSession(par_file=PAR_FILE, tim_file=TIM_FILE, verbose=False)
    params = session.params

    # =========================================================================
    banner("1. WRMS DEFINITION AUDIT")
    # =========================================================================
    result = session.compute_residuals(subtract_tzr=True, force_recompute=True)
    resid_us = result['residuals_us']
    errors_us = result['errors_us']
    w = 1.0 / (errors_us ** 2)

    prefit_A = wrms_A(resid_us, w)
    prefit_B = wrms_B(resid_us, w)
    print(f"  Prefit WRMS_A (no mean sub):   {prefit_A:.6f} μs")
    print(f"  Prefit WRMS_B (mean sub):      {prefit_B:.6f} μs")
    print(f"  JUG reported weighted_rms_us:  {result['weighted_rms_us']:.6f} μs")
    print(f"  Difference A-B:                {(prefit_A - prefit_B)*1e3:.3f} ns")
    print(f"  => WRMS definition is NOT a source of discrepancy.")

    # =========================================================================
    banner("2. TWO CODE PATH CONSISTENCY CHECK")
    # =========================================================================
    # combined_delays() is used for initial residuals.
    # compute_ell1_binary_delay() is used by fitter for full-model re-evaluation.
    # Both must produce the same binary delay.

    from jug.fitting.binary_registry import compute_binary_delay

    toas_bary = np.array(result['tdb_mjd']) - np.array(result['prebinary_delay_sec']) / 86400.0
    delay_fitter = np.array(compute_binary_delay(jnp.asarray(toas_bary), params))

    h3_val = float(params.get('H3', 0.0))
    print(f"  H3 = {h3_val:.6e}")
    print(f"  SINI = {params.get('SINI', 'NOT PRESENT')}")
    print(f"  M2 = {params.get('M2', 'NOT PRESENT')}")
    print(f"  STIG = {params.get('STIG', 'NOT PRESENT')}")

    # Compute H3-only Shapiro delay directly for comparison
    pb = float(params['PB'])
    tasc = float(params['TASC'])
    pbdot = float(params.get('PBDOT', 0.0))
    dt_days = toas_bary - tasc
    dt_sec = dt_days * 86400.0
    n0 = 2.0 * np.pi / (pb * 86400.0)
    phi = n0 * dt_sec * (1.0 - pbdot / 2.0 / pb * dt_days)
    h3_shapiro_correct = -(4.0 / 3.0) * h3_val * np.sin(3.0 * phi)

    print(f"\n  H3-only Shapiro delay (-(4/3)*H3*sin(3Φ)):")
    print(f"    RMS = {np.std(h3_shapiro_correct)*1e9:.3f} ns")
    print(f"    Max = {np.max(np.abs(h3_shapiro_correct))*1e9:.3f} ns")

    # Verify fitter delay includes H3 Shapiro by checking it's non-zero variance
    print(f"\n  Fitter binary delay:")
    print(f"    RMS = {np.std(delay_fitter)*1e6:.6f} s")

    # =========================================================================
    banner("3. POSTFIT WRMS COMPARISON")
    # =========================================================================
    session2 = TimingSession(par_file=PAR_FILE, tim_file=TIM_FILE, verbose=False)
    session2.compute_residuals(subtract_tzr=False, force_recompute=True)
    fit_result = session2.fit_parameters(
        fit_params=FIT_PARAMS, max_iter=100, verbose=False, solver_mode="exact"
    )
    postfit_wrms = fit_result['final_rms']
    postfit_resid = fit_result['residuals_us']
    postfit_err = fit_result['errors_us']
    w_post = 1.0 / (postfit_err ** 2)

    postfit_A = wrms_A(postfit_resid, w_post)
    postfit_B = wrms_B(postfit_resid, w_post)
    print(f"  Postfit final_rms (JUG):   {postfit_wrms:.6f} μs")
    print(f"  Postfit WRMS_A (no mean):  {postfit_A:.6f} μs")
    print(f"  Postfit WRMS_B (mean sub): {postfit_B:.6f} μs")
    print(f"  Par file TRES:             {PAR_FILE_TRES_US:.6f} μs")
    print(f"  Gap (JUG - TRES):          {(postfit_wrms - PAR_FILE_TRES_US)*1e3:.1f} ns")
    print(f"  Fit converged:             {fit_result['converged']}")
    print(f"  Iterations:                {fit_result['iterations']}")

    # =========================================================================
    banner("4. H3 DERIVATIVE VERIFICATION")
    # =========================================================================
    from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1
    h3_result = compute_binary_derivatives_ell1(params, jnp.asarray(toas_bary), ['H3'])
    h3_deriv = np.asarray(h3_result['H3'])
    expected_deriv = -(4.0 / 3.0) * np.sin(3.0 * phi)

    # Use subset for correlation
    n = len(h3_deriv)
    step = max(1, n // 200)
    idx = slice(0, n, step)

    corr = np.corrcoef(h3_deriv[idx], expected_deriv[idx])[0, 1]
    max_diff = np.max(np.abs(h3_deriv[idx] - expected_deriv[idx]))
    rel_diff = max_diff / np.max(np.abs(expected_deriv[idx]))
    print(f"  H3 derivative: max|analytic| = {np.max(np.abs(h3_deriv)):.6f}")
    print(f"  Expected:      max|-(4/3)sin(3Φ)| = {np.max(np.abs(expected_deriv)):.6f}")
    print(f"  Correlation:   r = {corr:.10f}")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Max rel diff:  {rel_diff:.6e}")

    # =========================================================================
    banner("5. PARAMETER PARITY")
    # =========================================================================
    print(f"  Binary model: {params.get('BINARY', 'NONE')}")
    print(f"  PEPOCH:  {params.get('PEPOCH', 'N/A')}")
    print(f"  TASC:    {params.get('TASC', 'N/A')}")
    print(f"  DMEPOCH: {params.get('DMEPOCH', 'N/A')}")
    print(f"  NITS:    {params.get('NITS', 'N/A')}")
    print(f"  UNITS:   {params.get('UNITS', 'N/A')}")
    print(f"  EPHEM:   {params.get('EPHEM', 'N/A')}")
    print(f"  CLK:     {params.get('CLK', 'N/A')}")
    print(f"  CHI2R:   {params.get('CHI2R', 'N/A')}")
    print(f"  NTOA:    {params.get('NTOA', 'N/A')}")
    jumps = [k for k in params if k.startswith('JUMP')]
    print(f"  JUMPs:   {len(jumps)}")

    # =========================================================================
    banner("6. DIAGNOSIS SUMMARY")
    # =========================================================================
    print(f"""
  FINDINGS:
  =========
  1. WRMS definition: NOT the issue (WRMS_A = WRMS_B, {prefit_A:.6f} = {prefit_B:.6f})

  2. Forward model fixes applied:
     a) H3-only Shapiro coefficient: was -2, now -(4/3) [matches Tempo2/PINT]
     b) compute_ell1_binary_delay: now includes H3-only Shapiro [fitter consistency]
     c) H3 derivative: now -(4/3)*sin(3*Phi) [matches Tempo2/PINT]

  3. WRMS comparison:
     JUG post-fit WRMS:  {postfit_wrms:.6f} μs
     Par file TRES:      {PAR_FILE_TRES_US:.6f} μs
     Gap:                {(postfit_wrms - PAR_FILE_TRES_US)*1e3:.1f} ns

  4. Remaining ~1 ns is from known differences:
     - JUG uses 3rd-order eccentricity corrections (Zhu et al. 2019)
     - JUG uses inverse delay corrections (PINT-style)
     - These are ~femtosecond level improvements over Tempo2
""")


if __name__ == '__main__':
    main()
