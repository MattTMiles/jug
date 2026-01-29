"""
Golden Regression Test Infrastructure
=====================================

This package provides golden test infrastructure for JUG, storing reference
outputs that must remain bit-for-bit identical across code changes.

Golden outputs stored here:
- j1909_prefit_residuals.npy - Prefit residuals for J1909-3744
- j1909_postfit_residuals.npy - Postfit residuals after F0+F1+DM fit
- j1909_covariance.npy - Parameter covariance matrix
- j1909_scalars.json - Scalar values (WRMS, chi2, dof, fitted params)

Usage:
    # Generate goldens (run once to establish baseline)
    python -m jug.tests.golden.generate_golden

    # Run golden regression tests
    pytest jug/tests/test_golden_regression.py -v

CRITICAL: All assertions use np.array_equal (NO tolerances).
Any bit-level difference will fail the test.
"""

from pathlib import Path

GOLDEN_DIR = Path(__file__).parent
