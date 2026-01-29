"""
Golden Regression Tests
=======================

Tests that verify JUG produces bit-for-bit identical outputs to stored
golden reference files. ANY difference (even 1 ULP) fails the test.

CRITICAL:
- Uses np.array_equal for arrays (NO tolerances)
- Uses exact equality (==) for scalars
- Uses hex comparison for float precision verification

Run with:
    pytest jug/tests/test_golden_regression.py -v

If tests fail after intentional changes:
    python -m jug.tests.golden.generate_golden
    # Review changes carefully before committing
"""

import json
import os
import sys
from pathlib import Path

# Force determinism BEFORE any other imports
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'

import numpy as np

# Optional pytest import for fixture support
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a dummy pytest module for running without pytest
    class _DummyPytest:
        @staticmethod
        def skip(msg):
            print(f"SKIP: {msg}")
            raise RuntimeError(f"SKIP: {msg}")

        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    pytest = _DummyPytest()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.engine.session import TimingSession


# Paths
GOLDEN_DIR = Path(__file__).parent / "golden"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pulsars"
PAR_FILE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"


def _skip_if_no_data():
    """Return True if test data is missing."""
    return not (PAR_FILE.exists() and TIM_FILE.exists())


def _skip_if_no_goldens():
    """Return True if golden files are missing."""
    required = [
        "j1909_prefit_residuals.npy",
        "j1909_postfit_residuals.npy",
        "j1909_covariance.npy",
        "j1909_scalars.json",
    ]
    return not all((GOLDEN_DIR / f).exists() for f in required)


class TestGoldenRegression:
    """Bit-for-bit regression tests against stored golden outputs."""

    def setup_method(self):
        """Load golden data if available (pytest-compatible setup)."""
        self._do_setup()

    def setup(self):
        """Load golden data if available (manual test runner setup)."""
        self._do_setup()

    def _do_setup(self):
        """Common setup logic."""
        if _skip_if_no_goldens():
            if HAS_PYTEST:
                pytest.skip("Golden files not found. Run: python -m jug.tests.golden.generate_golden")
            else:
                raise RuntimeError("Golden files not found. Run: python -m jug.tests.golden.generate_golden")
        if _skip_if_no_data():
            if HAS_PYTEST:
                pytest.skip("Test data not found")
            else:
                raise RuntimeError("Test data not found")

        # Load golden outputs
        self.golden_prefit = np.load(GOLDEN_DIR / "j1909_prefit_residuals.npy")
        self.golden_postfit = np.load(GOLDEN_DIR / "j1909_postfit_residuals.npy")
        self.golden_covariance = np.load(GOLDEN_DIR / "j1909_covariance.npy")
        with open(GOLDEN_DIR / "j1909_scalars.json") as f:
            self.golden_scalars = json.load(f)

        # Create session
        self.session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

    def test_prefit_residuals_exact(self):
        """Test: Prefit residuals must be bit-for-bit identical to golden."""
        result = self.session.compute_residuals(subtract_tzr=True)
        residuals = result['residuals_us']

        # Shape check
        assert residuals.shape == self.golden_prefit.shape, \
            f"Shape mismatch: got {residuals.shape}, expected {self.golden_prefit.shape}"

        # Bit-for-bit comparison - NO tolerance
        assert np.array_equal(residuals, self.golden_prefit), \
            "Prefit residuals differ from golden (bit-level difference detected)"

    def test_prefit_rms_exact(self):
        """Test: Prefit RMS must be exactly equal to golden."""
        result = self.session.compute_residuals(subtract_tzr=True)
        rms = result['rms_us']

        golden_rms = self.golden_scalars['prefit_rms_us']

        # Exact equality
        assert rms == golden_rms, \
            f"Prefit RMS differs: got {rms}, expected {golden_rms}"

    def test_postfit_residuals_exact(self):
        """Test: Postfit residuals must be bit-for-bit identical to golden."""
        # Ensure cache is populated
        _ = self.session.compute_residuals(subtract_tzr=False)

        # Run fit with same parameters as golden
        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        residuals = result['residuals_us']

        # Shape check
        assert residuals.shape == self.golden_postfit.shape, \
            f"Shape mismatch: got {residuals.shape}, expected {self.golden_postfit.shape}"

        # Bit-for-bit comparison - NO tolerance
        assert np.array_equal(residuals, self.golden_postfit), \
            "Postfit residuals differ from golden (bit-level difference detected)"

    def test_postfit_rms_exact(self):
        """Test: Postfit RMS must be exactly equal to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        rms = result['final_rms']
        golden_rms = self.golden_scalars['postfit_rms_us']

        # Exact equality
        assert rms == golden_rms, \
            f"Postfit RMS differs: got {rms}, expected {golden_rms}"

    def test_covariance_matrix_exact(self):
        """Test: Covariance matrix must be bit-for-bit identical to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        covariance = result['covariance']

        # Shape check
        assert covariance.shape == self.golden_covariance.shape, \
            f"Shape mismatch: got {covariance.shape}, expected {self.golden_covariance.shape}"

        # Bit-for-bit comparison - NO tolerance
        assert np.array_equal(covariance, self.golden_covariance), \
            "Covariance matrix differs from golden (bit-level difference detected)"

    def test_fitted_parameters_exact(self):
        """Test: Fitted parameter values must be exactly equal to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        for param in fit_params:
            final_val = result['final_params'][param]
            golden_val = self.golden_scalars['final_params'][param]['value']
            golden_hex = self.golden_scalars['final_params'][param]['hex']

            # First check: exact value equality
            if final_val != golden_val:
                # Second check: hex representation for exact bit pattern
                final_hex = float(final_val).hex()
                assert final_hex == golden_hex, \
                    f"Parameter {param} differs:\n" \
                    f"  Got: {final_val} ({final_hex})\n" \
                    f"  Expected: {golden_val} ({golden_hex})"

    def test_uncertainties_exact(self):
        """Test: Parameter uncertainties must be exactly equal to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        for param in fit_params:
            uncertainty = result['uncertainties'][param]
            golden_unc = self.golden_scalars['uncertainties'][param]['value']
            golden_hex = self.golden_scalars['uncertainties'][param]['hex']

            # First check: exact value equality
            if uncertainty != golden_unc:
                # Second check: hex representation
                unc_hex = float(uncertainty).hex()
                assert unc_hex == golden_hex, \
                    f"Uncertainty for {param} differs:\n" \
                    f"  Got: {uncertainty} ({unc_hex})\n" \
                    f"  Expected: {golden_unc} ({golden_hex})"

    def test_chi2_exact(self):
        """Test: Chi-squared must be exactly equal to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        # Get errors for chi2 computation
        prefit_result = self.session.compute_residuals(subtract_tzr=True)
        errors_us = prefit_result['errors_us']

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        chi2 = np.sum((result['residuals_us'] / errors_us) ** 2)
        golden_chi2 = self.golden_scalars['chi2']

        assert chi2 == golden_chi2, \
            f"Chi2 differs: got {chi2}, expected {golden_chi2}"

    def test_iterations_exact(self):
        """Test: Number of iterations must be exactly equal to golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        iterations = result['iterations']
        golden_iterations = self.golden_scalars['iterations']

        assert iterations == golden_iterations, \
            f"Iterations differ: got {iterations}, expected {golden_iterations}"

    def test_convergence_status(self):
        """Test: Convergence status must match golden."""
        _ = self.session.compute_residuals(subtract_tzr=False)

        fit_params = self.golden_scalars['fit_params']
        result = self.session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        converged = bool(result['converged'])
        golden_converged = self.golden_scalars['converged']

        assert converged == golden_converged, \
            f"Convergence status differs: got {converged}, expected {golden_converged}"


class TestDeterminism:
    """Tests that verify computation is deterministic across runs."""

    def setup_method(self):
        """Check prerequisites (pytest-compatible setup)."""
        self._do_setup()

    def setup(self):
        """Check prerequisites (manual test runner setup)."""
        self._do_setup()

    def _do_setup(self):
        """Common setup logic."""
        if _skip_if_no_data():
            if HAS_PYTEST:
                pytest.skip("Test data not found")
            else:
                raise RuntimeError("Test data not found")

    def test_multiple_runs_identical(self):
        """Test: Multiple runs produce bit-for-bit identical results."""
        results = []

        for i in range(3):
            session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
            _ = session.compute_residuals(subtract_tzr=False)
            result = session.fit_parameters(
                fit_params=['F0', 'F1', 'DM'],
                max_iter=25,
                convergence_threshold=1e-14,
                solver_mode="exact",
                verbose=False
            )
            results.append(result)

        # Compare all runs to first run
        ref = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert np.array_equal(result['residuals_us'], ref['residuals_us']), \
                f"Run {i} residuals differ from run 1"
            assert result['final_rms'] == ref['final_rms'], \
                f"Run {i} RMS differs from run 1"
            assert np.array_equal(result['covariance'], ref['covariance']), \
                f"Run {i} covariance differs from run 1"

    def test_session_independence(self):
        """Test: Separate sessions produce identical results."""
        # Session 1: Compute twice
        session1 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        result1a = session1.compute_residuals(subtract_tzr=True)
        result1b = session1.compute_residuals(subtract_tzr=True)

        # Session 2: Fresh session
        session2 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        result2 = session2.compute_residuals(subtract_tzr=True)

        # All should be identical
        assert np.array_equal(result1a['residuals_us'], result1b['residuals_us'])
        assert np.array_equal(result1a['residuals_us'], result2['residuals_us'])


class TestEnvironmentDeterminism:
    """Tests that verify environment settings enforce determinism."""

    def test_cpu_platform_set(self):
        """Test: JAX_PLATFORM_NAME is set to cpu."""
        assert os.environ.get('JAX_PLATFORM_NAME') == 'cpu', \
            "JAX_PLATFORM_NAME must be 'cpu' for deterministic tests"

    def test_fast_math_disabled(self):
        """Test: XLA fast math is disabled."""
        xla_flags = os.environ.get('XLA_FLAGS', '')
        assert '--xla_cpu_enable_fast_math=false' in xla_flags, \
            "XLA_FLAGS must contain '--xla_cpu_enable_fast_math=false'"


def run_golden_tests():
    """Run all golden regression tests."""
    print("="*80)
    print("JUG Golden Regression Test Suite")
    print("="*80)

    if _skip_if_no_goldens():
        print("\nGolden files not found!")
        print("Generate them with: python -m jug.tests.golden.generate_golden")
        return False

    if _skip_if_no_data():
        print("\nTest data not found!")
        return False

    # Run with pytest
    import subprocess
    result = subprocess.run(
        ['pytest', __file__, '-v', '--tb=short'],
        cwd=Path(__file__).parent.parent.parent.parent
    )
    return result.returncode == 0


if __name__ == '__main__':
    success = run_golden_tests()
    sys.exit(0 if success else 1)
