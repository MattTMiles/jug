#!/usr/bin/env python
"""
Generate Golden Reference Outputs
=================================

This script generates golden reference outputs for regression testing.
Run this ONCE to establish the baseline, then commit the outputs.

The golden outputs are:
- j1909_prefit_residuals.npy - Prefit residuals (microseconds)
- j1909_postfit_residuals.npy - Postfit residuals after F0+F1+DM fit
- j1909_covariance.npy - Parameter covariance matrix
- j1909_scalars.json - Scalar values (WRMS, chi2, dof, fitted params)

Usage:
    python -m jug.tests.golden.generate_golden

IMPORTANT: Only run this when you INTEND to update the goldens.
Any change to golden outputs requires justification and review.
"""

import json
import os
import sys
from pathlib import Path

# Force determinism BEFORE any other imports
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jug.engine.session import TimingSession


# Paths
GOLDEN_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "pulsars"
PAR_FILE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"

# Fit parameters for golden test
FIT_PARAMS = ['F0', 'F1', 'DM']


def generate_goldens():
    """Generate all golden reference outputs."""
    print("="*80)
    print("JUG Golden Reference Generator")
    print("="*80)
    print()
    print(f"PAR file: {PAR_FILE}")
    print(f"TIM file: {TIM_FILE}")
    print(f"Fit params: {FIT_PARAMS}")
    print(f"Output dir: {GOLDEN_DIR}")
    print()

    # Verify data files exist
    if not PAR_FILE.exists():
        raise FileNotFoundError(f"PAR file not found: {PAR_FILE}")
    if not TIM_FILE.exists():
        raise FileNotFoundError(f"TIM file not found: {TIM_FILE}")

    # Create session
    print("Creating TimingSession...")
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

    # Get initial parameters for comparison
    initial_params = dict(session.params)

    # Compute prefit residuals
    print("Computing prefit residuals...")
    prefit_result = session.compute_residuals(subtract_tzr=True)
    prefit_residuals = prefit_result['residuals_us']
    prefit_rms = prefit_result['rms_us']

    print(f"  N TOAs: {len(prefit_residuals)}")
    print(f"  Prefit RMS: {prefit_rms:.6f} mus")

    # Run fit
    print(f"Fitting parameters: {FIT_PARAMS}...")
    fit_result = session.fit_parameters(
        fit_params=FIT_PARAMS,
        max_iter=25,
        convergence_threshold=1e-14,
        solver_mode="exact",
        verbose=False
    )

    postfit_residuals = fit_result['residuals_us']
    covariance = fit_result['covariance']
    postfit_rms = fit_result['final_rms']

    print(f"  Postfit RMS: {postfit_rms:.6f} mus")
    print(f"  Iterations: {fit_result['iterations']}")
    print(f"  Converged: {fit_result['converged']}")

    # Compute chi2 and dof
    errors_us = prefit_result['errors_us']
    chi2 = np.sum((postfit_residuals / errors_us) ** 2)
    n_toas = len(postfit_residuals)
    n_params = len(FIT_PARAMS)
    dof = n_toas - n_params
    reduced_chi2 = chi2 / dof

    print(f"  Chi2: {chi2:.6f}")
    print(f"  DOF: {dof}")
    print(f"  Reduced Chi2: {reduced_chi2:.6f}")

    # Prepare scalars dict
    scalars = {
        'n_toas': n_toas,
        'n_params': n_params,
        'dof': dof,
        'chi2': float(chi2),
        'reduced_chi2': float(reduced_chi2),
        'prefit_rms_us': float(prefit_rms),
        'postfit_rms_us': float(postfit_rms),
        'iterations': int(fit_result['iterations']),
        'converged': bool(fit_result['converged']),
        'fit_params': FIT_PARAMS,
        'final_params': {},
        'uncertainties': {},
        'initial_params': {},
        'param_changes': {},
    }

    # Store parameter values with full precision
    for param in FIT_PARAMS:
        initial_val = initial_params.get(param, 0.0)
        final_val = fit_result['final_params'][param]
        uncertainty = fit_result['uncertainties'][param]
        change = final_val - initial_val

        # Store as hex strings for exact reproduction
        scalars['initial_params'][param] = {
            'value': float(initial_val),
            'hex': initial_val.hex() if isinstance(initial_val, float) else float(initial_val).hex()
        }
        scalars['final_params'][param] = {
            'value': float(final_val),
            'hex': final_val.hex() if isinstance(final_val, float) else float(final_val).hex()
        }
        scalars['uncertainties'][param] = {
            'value': float(uncertainty),
            'hex': uncertainty.hex() if isinstance(uncertainty, float) else float(uncertainty).hex()
        }
        scalars['param_changes'][param] = {
            'value': float(change),
            'hex': change.hex() if isinstance(change, float) else float(change).hex()
        }

        print(f"  {param}: {initial_val} -> {final_val} +/- {uncertainty}")

    # Save outputs
    print()
    print("Saving golden outputs...")

    # Prefit residuals
    prefit_path = GOLDEN_DIR / "j1909_prefit_residuals.npy"
    np.save(prefit_path, prefit_residuals)
    print(f"  Saved: {prefit_path.name} ({prefit_residuals.shape}, {prefit_residuals.dtype})")

    # Postfit residuals
    postfit_path = GOLDEN_DIR / "j1909_postfit_residuals.npy"
    np.save(postfit_path, postfit_residuals)
    print(f"  Saved: {postfit_path.name} ({postfit_residuals.shape}, {postfit_residuals.dtype})")

    # Covariance matrix
    cov_path = GOLDEN_DIR / "j1909_covariance.npy"
    np.save(cov_path, covariance)
    print(f"  Saved: {cov_path.name} ({covariance.shape}, {covariance.dtype})")

    # Scalars
    scalars_path = GOLDEN_DIR / "j1909_scalars.json"
    with open(scalars_path, 'w') as f:
        json.dump(scalars, f, indent=2)
    print(f"  Saved: {scalars_path.name}")

    print()
    print("="*80)
    print("Golden generation complete!")
    print()
    print("IMPORTANT: Review the outputs and commit them to version control.")
    print("Any future changes to these outputs require justification.")
    print("="*80)


def verify_goldens():
    """Verify that golden files exist and are valid."""
    print("Verifying golden files...")

    files = [
        ("j1909_prefit_residuals.npy", np.ndarray),
        ("j1909_postfit_residuals.npy", np.ndarray),
        ("j1909_covariance.npy", np.ndarray),
        ("j1909_scalars.json", dict),
    ]

    all_valid = True
    for filename, expected_type in files:
        path = GOLDEN_DIR / filename
        if not path.exists():
            print(f"  MISSING: {filename}")
            all_valid = False
            continue

        if filename.endswith('.npy'):
            data = np.load(path)
            print(f"  OK: {filename} ({data.shape}, {data.dtype})")
        else:
            with open(path) as f:
                data = json.load(f)
            print(f"  OK: {filename} ({len(data)} keys)")

    return all_valid


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate or verify golden reference outputs")
    parser.add_argument('--verify', action='store_true', help="Only verify existing goldens")
    args = parser.parse_args()

    if args.verify:
        if verify_goldens():
            print("\nAll golden files present and valid.")
        else:
            print("\nSome golden files missing. Run without --verify to generate.")
            sys.exit(1)
    else:
        generate_goldens()
