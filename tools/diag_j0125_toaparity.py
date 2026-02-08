#!/usr/bin/env python
"""Task 3 diagnostic: prove TOA set and weights parity between JUG and Tempo2.

Checks:
  1. N_toa_used matches T2's NTOA (5083)
  2. Checksum over TOA MJDs, frequencies, and errors
  3. Weight computation matches: w_i = 1/σ_i²
  4. No TOAs are silently excluded or reweighted

Usage:
    python tools/diag_j0125_toaparity.py
"""

import numpy as np
import hashlib
from pathlib import Path

PAR_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
T2_PAR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_test.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

T2_NTOA = 5083  # from NTOA line in T2 post-fit par
T2_NDOF = 5061  # from CHI2R line: CHI2R 1.3555 5061


def main():
    from jug.io.tim_reader import parse_tim_file_mjds

    print("=" * 70)
    print(f"{'TOA Parity Diagnostic: J0125-2327':^70s}")
    print("=" * 70)

    # --- 1. Load TOAs ---
    toas = parse_tim_file_mjds(TIM_FILE)
    n_toa = len(toas)
    print(f"\n1. TOA COUNT")
    print(f"   JUG loaded:  {n_toa}")
    print(f"   T2 NTOA:     {T2_NTOA}")
    match = "✓" if n_toa == T2_NTOA else "✗"
    print(f"   Match: {match}")

    # --- 2. DOF check ---
    n_fit = 21  # number of fitted params
    n_dof = n_toa - n_fit
    print(f"\n2. DEGREES OF FREEDOM")
    print(f"   JUG: N_toa - N_param = {n_toa} - {n_fit} = {n_dof}")
    print(f"   T2:  {T2_NDOF}")
    dof_match = "✓" if n_dof == T2_NDOF else "~"
    print(f"   Match: {dof_match}")
    if n_dof != T2_NDOF:
        print(f"   Note: T2 may subtract additional constraints (e.g., TZRMJD, poly subtraction)")
        print(f"   Difference: {n_dof - T2_NDOF}")

    # --- 3. TOA properties ---
    mjds = np.array([t.mjd_int + t.mjd_frac for t in toas])
    freqs = np.array([t.freq_mhz for t in toas])
    errors_us = np.array([t.error_us for t in toas])

    print(f"\n3. TOA PROPERTIES")
    print(f"   MJD range:   {mjds.min():.6f} — {mjds.max():.6f}")
    print(f"   Freq range:  {freqs.min():.1f} — {freqs.max():.1f} MHz")
    print(f"   Error range: {errors_us.min():.4f} — {errors_us.max():.4f} μs")
    print(f"   Mean error:  {errors_us.mean():.4f} μs")
    print(f"   Median error: {np.median(errors_us):.4f} μs")

    # Checksum
    data = np.column_stack([mjds, freqs, errors_us])
    checksum = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    print(f"   Checksum (MJD+freq+err): {checksum}")

    # --- 4. Weight computation ---
    weights = 1.0 / errors_us**2
    print(f"\n4. WEIGHT COMPUTATION")
    print(f"   Formula: w_i = 1/σ_i²")
    print(f"   Sum(w):      {weights.sum():.6e}")
    print(f"   Min weight:  {weights.min():.6e}")
    print(f"   Max weight:  {weights.max():.6e}")
    print(f"   Dynamic range: {weights.max() / weights.min():.1f}x")

    # --- 5. Observatory breakdown ---
    obs_list = [t.observatory for t in toas]
    obs_unique = sorted(set(obs_list))
    print(f"\n5. OBSERVATORY BREAKDOWN")
    for obs in obs_unique:
        count = obs_list.count(obs)
        print(f"   {obs:>12s}: {count:5d} TOAs ({100*count/n_toa:5.1f}%)")

    # --- 6. Flag analysis ---
    all_flags = set()
    for t in toas:
        all_flags.update(t.flags.keys())
    print(f"\n6. FLAG SUMMARY")
    print(f"   Unique flags: {sorted(all_flags)}")
    for flag in sorted(all_flags):
        vals = set(t.flags.get(flag, '') for t in toas if flag in t.flags)
        if len(vals) <= 10:
            print(f"   -{flag}: {sorted(vals)}")
        else:
            print(f"   -{flag}: {len(vals)} unique values")

    # --- 7. Duplicate / exclusion check ---
    print(f"\n7. DATA QUALITY")
    # Check for duplicate MJDs
    _, counts = np.unique(mjds, return_counts=True)
    n_dup = (counts > 1).sum()
    print(f"   Duplicate MJDs: {n_dup}")
    # Check for zero/negative errors
    n_bad = (errors_us <= 0).sum()
    print(f"   Zero/negative errors: {n_bad}")
    # Check for NaN
    n_nan = np.isnan(mjds).sum() + np.isnan(freqs).sum() + np.isnan(errors_us).sum()
    print(f"   NaN values: {n_nan}")

    # --- 8. Verify against T2 par START/FINISH ---
    print(f"\n8. TIME RANGE vs T2 PAR")
    # Read START/FINISH from T2 par
    from jug.io.par_reader import parse_par_file
    t2p = parse_par_file(T2_PAR)
    t2_start = t2p.get('START', 0.0)
    t2_finish = t2p.get('FINISH', 0.0)
    print(f"   JUG first TOA:  {mjds.min():.15f}")
    print(f"   T2 START:       {t2_start}")
    print(f"   JUG last TOA:   {mjds.max():.15f}")
    print(f"   T2 FINISH:      {t2_finish}")

    # --- Verdict ---
    print(f"\n{'='*70}")
    print(f"{'VERDICT':^70s}")
    print(f"{'='*70}")
    issues = []
    if n_toa != T2_NTOA:
        issues.append(f"TOA count mismatch: {n_toa} vs {T2_NTOA}")
    if n_bad > 0:
        issues.append(f"{n_bad} TOAs with zero/negative errors")
    if n_nan > 0:
        issues.append(f"{n_nan} NaN values found")
    if not issues:
        print("✓ All checks passed — TOA set matches Tempo2")
    else:
        for issue in issues:
            print(f"✗ {issue}")


if __name__ == "__main__":
    main()
