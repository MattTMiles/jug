#!/usr/bin/env python
"""Cross-tool parity comparison: JUG vs PINT (and optionally Tempo2).

This script compares residual computation and fitting results between
JUG, PINT, and optionally Tempo2 for validation purposes.

Usage:
    python jug/scripts/compare_with_pint_tempo2.py data/pulsars/J1909-3744_tdb.par data/pulsars/J1909-3744.tim
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add JUG to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_jug_residuals(par_file, tim_file, verbose=False):
    """Compute residuals using JUG."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    result = compute_residuals_simple(par_file, tim_file, verbose=verbose)
    return {
        'residuals_us': result['residuals_us'],
        'weighted_rms_us': result['weighted_rms_us'],
        'unweighted_rms_us': result['unweighted_rms_us'],
        'n_toas': result['n_toas'],
    }


def compute_pint_residuals(par_file, tim_file, verbose=False):
    """Compute residuals using PINT."""
    try:
        import pint.models as pm
        import pint.toa as pt
        import pint.fitter as pf
        from astropy import units as u
    except ImportError:
        print("PINT not available - skipping PINT comparison")
        return None
    
    # Suppress PINT warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    m = pm.get_model(par_file)
    t = pt.get_TOAs(tim_file, model=m)
    
    # Compute residuals
    fitter = pf.WLSFitter(t, m)
    
    residuals_us = fitter.resids.time_resids.to(u.us).value
    
    # Weighted RMS
    sigma_us = fitter.resids.get_data_error().to(u.us).value
    weights = 1.0 / (sigma_us ** 2)
    weighted_rms = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    
    # Unweighted RMS
    unweighted_rms = np.std(residuals_us)
    
    return {
        'residuals_us': residuals_us,
        'weighted_rms_us': weighted_rms,
        'unweighted_rms_us': unweighted_rms,
        'n_toas': len(residuals_us),
    }


def compute_tempo2_residuals(par_file, tim_file, verbose=False):
    """Compute residuals using Tempo2 (via libstempo)."""
    try:
        import libstempo as lt
    except ImportError:
        print("libstempo (Tempo2) not available - skipping Tempo2 comparison")
        return None
    
    # Load pulsar with tempo2
    psr = lt.tempopulsar(parfile=str(par_file), timfile=str(tim_file))
    
    residuals_us = psr.residuals() * 1e6  # Convert to microseconds
    
    # Get errors
    sigma_us = psr.toaerrs  # Already in microseconds
    weights = 1.0 / (sigma_us ** 2)
    
    # Weighted RMS
    weighted_rms = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    
    # Unweighted RMS
    unweighted_rms = np.std(residuals_us)
    
    return {
        'residuals_us': residuals_us,
        'weighted_rms_us': weighted_rms,
        'unweighted_rms_us': unweighted_rms,
        'n_toas': len(residuals_us),
    }


def compare_results(jug_result, pint_result, tempo2_result):
    """Compare results between tools."""
    print("\n" + "="*70)
    print("Residual Comparison Summary")
    print("="*70)
    
    headers = ["Metric", "JUG"]
    row_data = [
        ("N_TOAs", jug_result['n_toas']),
        ("Weighted RMS (μs)", f"{jug_result['weighted_rms_us']:.6f}"),
        ("Unweighted RMS (μs)", f"{jug_result['unweighted_rms_us']:.6f}"),
    ]
    
    if pint_result:
        headers.append("PINT")
        row_data[0] = row_data[0] + (pint_result['n_toas'],)
        row_data[1] = row_data[1] + (f"{pint_result['weighted_rms_us']:.6f}",)
        row_data[2] = row_data[2] + (f"{pint_result['unweighted_rms_us']:.6f}",)
    
    if tempo2_result:
        headers.append("Tempo2")
        row_data[0] = row_data[0] + (tempo2_result['n_toas'],)
        row_data[1] = row_data[1] + (f"{tempo2_result['weighted_rms_us']:.6f}",)
        row_data[2] = row_data[2] + (f"{tempo2_result['unweighted_rms_us']:.6f}",)
    
    # Print table
    col_widths = [25] + [15] * (len(headers) - 1)
    header_str = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * sum(col_widths))
    
    for row in row_data:
        row_str = "".join(f"{str(v):<{w}}" for v, w in zip(row, col_widths))
        print(row_str)
    
    # Detailed comparison
    if pint_result:
        print("\n" + "-"*40)
        print("JUG vs PINT Residual Comparison:")
        diff = jug_result['residuals_us'] - pint_result['residuals_us']
        print(f"  Max abs difference: {np.abs(diff).max():.6f} μs")
        print(f"  Mean difference:    {np.mean(diff):.6f} μs")
        print(f"  Std difference:     {np.std(diff):.6f} μs")
        
        # Check if within tolerance
        if np.abs(diff).max() < 0.1:  # 100 ns tolerance
            print("  ✓ Results match within 100 ns")
        else:
            print("  ⚠ Results differ by more than 100 ns")
    
    if tempo2_result:
        print("\n" + "-"*40)
        print("JUG vs Tempo2 Residual Comparison:")
        diff = jug_result['residuals_us'] - tempo2_result['residuals_us']
        print(f"  Max abs difference: {np.abs(diff).max():.6f} μs")
        print(f"  Mean difference:    {np.mean(diff):.6f} μs")
        print(f"  Std difference:     {np.std(diff):.6f} μs")


def main():
    parser = argparse.ArgumentParser(
        description='Compare JUG residuals with PINT and Tempo2'
    )
    parser.add_argument('par_file', help='Path to .par file')
    parser.add_argument('tim_file', help='Path to .tim file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--skip-tempo2', action='store_true',
                        help='Skip Tempo2 comparison')
    parser.add_argument('--skip-pint', action='store_true',
                        help='Skip PINT comparison')
    
    args = parser.parse_args()
    
    par_file = args.par_file
    tim_file = args.tim_file
    
    print("="*70)
    print("Cross-Tool Parity Comparison")
    print("="*70)
    print(f"Par file: {par_file}")
    print(f"Tim file: {tim_file}")
    
    # Compute JUG residuals
    print("\n" + "-"*40)
    print("Computing JUG residuals...")
    jug_result = compute_jug_residuals(par_file, tim_file, verbose=args.verbose)
    print(f"  JUG: {jug_result['n_toas']} TOAs, WRMS = {jug_result['weighted_rms_us']:.6f} μs")
    
    # Compute PINT residuals
    pint_result = None
    if not args.skip_pint:
        print("\n" + "-"*40)
        print("Computing PINT residuals...")
        pint_result = compute_pint_residuals(par_file, tim_file, verbose=args.verbose)
        if pint_result:
            print(f"  PINT: {pint_result['n_toas']} TOAs, WRMS = {pint_result['weighted_rms_us']:.6f} μs")
    
    # Compute Tempo2 residuals
    tempo2_result = None
    if not args.skip_tempo2:
        print("\n" + "-"*40)
        print("Computing Tempo2 residuals...")
        tempo2_result = compute_tempo2_residuals(par_file, tim_file, verbose=args.verbose)
        if tempo2_result:
            print(f"  Tempo2: {tempo2_result['n_toas']} TOAs, WRMS = {tempo2_result['weighted_rms_us']:.6f} μs")
    
    # Compare
    compare_results(jug_result, pint_result, tempo2_result)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
