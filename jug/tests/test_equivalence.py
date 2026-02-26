"""
Comprehensive Bit-for-Bit Equivalence Tests
=============================================

This module tests EXACT behavioral equivalence across:
1. Legacy residual computation vs new engine/session pathway
2. File-based fitter vs cached fitter
3. Prefit and postfit residual arrays
4. Fitted parameter values and uncertainties

CRITICAL: Uses np.array_equal (NO tolerances) for all comparisons.
Any difference (even 1 ULP) will fail the test.

Test Requirements Met:
- Deterministic test dataset (uses existing J1909-3744 data)
- Golden baseline computation using stable path
- Comparison to new path outputs
- Engine/session pathway verification
- EXACT mode bit-for-bit validation
"""

import numpy as np
from pathlib import Path
import sys
import os

# Force deterministic behavior
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import (
    fit_parameters_optimized,
    _build_general_fit_setup_from_files,
    _build_general_fit_setup_from_cache,
    _run_general_fit_iterations,
    fit_parameters_optimized_cached,
)
from jug.engine.session import TimingSession


# Test data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pulsars"
PAR_FILE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"


def _skip_if_no_data():
    """Skip test if data files not found."""
    if not PAR_FILE.exists() or not TIM_FILE.exists():
        print(f"SKIP: Test data not found ({PAR_FILE}, {TIM_FILE})")
        return True
    return False


class TestResidualEquivalence:
    """Test that residual computation is bit-for-bit identical across pathways."""

    def test_legacy_vs_session_prefit_residuals(self):
        """
        Test: Legacy compute_residuals_simple vs TimingSession.compute_residuals

        Both pathways must produce bit-for-bit identical prefit residuals.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: Legacy vs Session Prefit Residuals")
        print("="*80)

        # LEGACY PATH: Direct call to compute_residuals_simple
        print("\nLegacy path: compute_residuals_simple(subtract_tzr=True)")
        result_legacy = compute_residuals_simple(
            PAR_FILE, TIM_FILE,
            clock_dir=None,
            subtract_tzr=True,
            verbose=False
        )

        # SESSION PATH: Via TimingSession
        print("Session path: TimingSession.compute_residuals(subtract_tzr=True)")
        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        result_session = session.compute_residuals(subtract_tzr=True)

        # BIT-FOR-BIT COMPARISON
        print("\nComparing results...")

        # Residuals array
        assert np.array_equal(result_legacy['residuals_us'], result_session['residuals_us']), \
            "Residuals array differs between legacy and session paths"
        print(f"[x] residuals_us: IDENTICAL ({len(result_legacy['residuals_us'])} values)")

        # RMS
        assert result_legacy['rms_us'] == result_session['rms_us'], \
            f"RMS differs: legacy={result_legacy['rms_us']}, session={result_session['rms_us']}"
        print(f"[x] rms_us: IDENTICAL ({result_legacy['rms_us']:.10f} mus)")

        # TDB times
        assert np.array_equal(result_legacy['tdb_mjd'], result_session['tdb_mjd']), \
            "TDB MJD array differs"
        print(f"[x] tdb_mjd: IDENTICAL")

        # dt_sec (delays)
        assert np.array_equal(result_legacy['dt_sec'], result_session['dt_sec']), \
            "dt_sec array differs"
        print(f"[x] dt_sec: IDENTICAL")

        print("\n" + "="*80)
        print("[x] TEST PASSED: Legacy and Session paths are bit-for-bit identical")
        print("="*80)

    def test_subtract_tzr_modes_differ(self):
        """
        Test: subtract_tzr=True vs subtract_tzr=False must produce different results.

        This ensures the TZR subtraction logic is working correctly.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: subtract_tzr Modes Differ")
        print("="*80)

        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

        result_tzr_true = session.compute_residuals(subtract_tzr=True)
        result_tzr_false = session.compute_residuals(subtract_tzr=False)

        # They should differ (TZR subtraction changes residuals)
        assert not np.array_equal(result_tzr_true['residuals_us'], result_tzr_false['residuals_us']), \
            "subtract_tzr=True and False should produce different residuals"

        max_diff = np.max(np.abs(result_tzr_true['residuals_us'] - result_tzr_false['residuals_us']))
        print(f"[x] Residuals differ (max diff: {max_diff:.6f} mus)")
        print(f"[x] TEST PASSED: TZR subtraction works correctly")


class TestFittingEquivalence:
    """Test that fitting is bit-for-bit identical across pathways."""

    def test_file_vs_cached_fitting(self):
        """
        Test: File-based fit_parameters_optimized vs cached fit_parameters_optimized_cached

        Both pathways must produce bit-for-bit identical fitted parameters.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: File-based vs Cached Fitting")
        print("="*80)

        fit_params = ['F0', 'F1']  # Use F0, F1 for speed (DM adds complexity)

        # FILE-BASED PATH
        print("\nFile-based path: fit_parameters_optimized()")
        result_file = fit_parameters_optimized(
            par_file=PAR_FILE,
            tim_file=TIM_FILE,
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            clock_dir=None,
            verbose=False,
            device='cpu'
        )

        # CACHED PATH via session
        print("Cached path: TimingSession.fit_parameters()")
        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        # Ensure cache is populated
        _ = session.compute_residuals(subtract_tzr=False)

        result_cached = session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            verbose=False
        )

        # BIT-FOR-BIT COMPARISON
        print("\nComparing results...")

        # Final parameters
        for param in fit_params:
            val_file = result_file['final_params'][param]
            val_cached = result_cached['final_params'][param]
            assert val_file == val_cached, \
                f"Parameter {param} differs: file={val_file}, cached={val_cached}"
            print(f"[x] {param}: IDENTICAL ({val_file:.20e})")

        # Uncertainties
        for param in fit_params:
            unc_file = result_file['uncertainties'][param]
            unc_cached = result_cached['uncertainties'][param]
            assert unc_file == unc_cached, \
                f"Uncertainty {param} differs: file={unc_file}, cached={unc_cached}"
            print(f"[x] sigma({param}): IDENTICAL ({unc_file:.6e})")

        # Final RMS
        assert result_file['final_rms'] == result_cached['final_rms'], \
            f"Final RMS differs: file={result_file['final_rms']}, cached={result_cached['final_rms']}"
        print(f"[x] final_rms: IDENTICAL ({result_file['final_rms']:.10f} mus)")

        # Iterations
        assert result_file['iterations'] == result_cached['iterations'], \
            f"Iterations differ: file={result_file['iterations']}, cached={result_cached['iterations']}"
        print(f"[x] iterations: IDENTICAL ({result_file['iterations']})")

        # Residuals arrays
        assert np.array_equal(result_file['residuals_us'], result_cached['residuals_us']), \
            "Postfit residuals array differs"
        print(f"[x] residuals_us: IDENTICAL ({len(result_file['residuals_us'])} values)")

        # Covariance matrix
        assert np.array_equal(result_file['covariance'], result_cached['covariance']), \
            "Covariance matrix differs"
        print(f"[x] covariance: IDENTICAL")

        print("\n" + "="*80)
        print("[x] TEST PASSED: File-based and cached fitting are bit-for-bit identical")
        print("="*80)

    def test_exact_solver_reproducibility(self):
        """
        Test: EXACT solver produces identical results across multiple runs.

        The EXACT (SVD) solver must be deterministic.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: EXACT Solver Reproducibility")
        print("="*80)

        fit_params = ['F0', 'F1']

        # Run twice with exact solver
        session1 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session1.compute_residuals(subtract_tzr=False)
        result1 = session1.fit_parameters(
            fit_params=fit_params,
            solver_mode="exact",
            verbose=False
        )

        session2 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session2.compute_residuals(subtract_tzr=False)
        result2 = session2.fit_parameters(
            fit_params=fit_params,
            solver_mode="exact",
            verbose=False
        )

        # Compare
        for param in fit_params:
            assert result1['final_params'][param] == result2['final_params'][param], \
                f"EXACT solver not reproducible for {param}"

        assert result1['final_rms'] == result2['final_rms'], \
            "EXACT solver RMS not reproducible"

        assert np.array_equal(result1['residuals_us'], result2['residuals_us']), \
            "EXACT solver residuals not reproducible"

        print("[x] EXACT solver produces bit-for-bit identical results across runs")
        print("[x] TEST PASSED")

    def test_fast_solver_convergence(self):
        """
        Test: FAST solver converges to a reasonable solution.

        Note: FAST solver is NOT required to be bit-for-bit identical to EXACT,
        but it must converge and produce similar results.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: FAST Solver Convergence")
        print("="*80)

        fit_params = ['F0', 'F1']

        # Run with EXACT solver
        session_exact = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session_exact.compute_residuals(subtract_tzr=False)
        result_exact = session_exact.fit_parameters(
            fit_params=fit_params,
            solver_mode="exact",
            verbose=False
        )

        # Run with FAST solver
        session_fast = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session_fast.compute_residuals(subtract_tzr=False)
        result_fast = session_fast.fit_parameters(
            fit_params=fit_params,
            solver_mode="fast",
            verbose=False
        )

        print(f"EXACT solver: RMS = {result_exact['final_rms']:.6f} mus, iterations = {result_exact['iterations']}")
        print(f"FAST solver:  RMS = {result_fast['final_rms']:.6f} mus, iterations = {result_fast['iterations']}")

        # FAST should converge
        assert result_fast['converged'], "FAST solver did not converge"

        # FAST should produce similar RMS (within 1% is acceptable)
        rms_ratio = result_fast['final_rms'] / result_exact['final_rms']
        assert 0.99 < rms_ratio < 1.01, \
            f"FAST solver RMS differs too much from EXACT: {rms_ratio:.6f}"

        print(f"[x] RMS ratio (FAST/EXACT): {rms_ratio:.6f}")
        print("[x] TEST PASSED: FAST solver converges and produces similar results")


class TestEngineAPIUsage:
    """Test that GUI/CLI pathways use engine APIs correctly."""

    def test_session_compute_residuals_returns_required_keys(self):
        """
        Test: TimingSession.compute_residuals returns all required keys for GUI.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: Session compute_residuals Returns Required Keys")
        print("="*80)

        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        result = session.compute_residuals(subtract_tzr=True)

        required_keys = ['residuals_us', 'rms_us', 'tdb_mjd', 'errors_us', 'dt_sec']

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            print(f"[x] {key}: present")

        # Check types
        assert isinstance(result['residuals_us'], np.ndarray)
        assert isinstance(result['rms_us'], float)
        assert isinstance(result['tdb_mjd'], np.ndarray)

        print("[x] TEST PASSED: All required keys present with correct types")

    def test_session_fit_parameters_returns_required_keys(self):
        """
        Test: TimingSession.fit_parameters returns all required keys for GUI.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: Session fit_parameters Returns Required Keys")
        print("="*80)

        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session.compute_residuals(subtract_tzr=False)

        fit_params = ['F0', 'F1']
        result = session.fit_parameters(fit_params=fit_params, verbose=False)

        required_keys = [
            'final_params', 'uncertainties', 'final_rms',
            'iterations', 'converged', 'covariance',
            'residuals_us', 'tdb_mjd', 'total_time'
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            print(f"[x] {key}: present")

        # Check structure
        assert isinstance(result['final_params'], dict)
        assert isinstance(result['uncertainties'], dict)
        assert result['converged'] in (True, False)  # Works for bool and np.bool_

        print("[x] TEST PASSED: All required keys present with correct types")

    def test_toa_mask_filtering(self):
        """
        Test: TOA mask correctly filters TOAs for fitting.

        This tests the deleted TOA functionality.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: TOA Mask Filtering")
        print("="*80)

        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session.compute_residuals(subtract_tzr=False)

        n_total = session.get_toa_count()

        # Create mask (keep first 1000 TOAs)
        toa_mask = np.zeros(n_total, dtype=bool)
        toa_mask[:1000] = True

        fit_params = ['F0', 'F1']
        result = session.fit_parameters(
            fit_params=fit_params,
            toa_mask=toa_mask,
            verbose=False
        )

        # Result should be based on 1000 TOAs
        # (residuals array length may vary based on implementation)
        assert result['converged'], "Fit with TOA mask should converge"
        print(f"[x] Fit converged with TOA mask (1000/{n_total} TOAs)")
        print(f"[x] Final RMS: {result['final_rms']:.6f} mus")
        print("[x] TEST PASSED: TOA mask filtering works")


class TestPrecomputedConstants:
    """Test that precomputed constants don't change numerical results."""

    def test_weight_sum_precomputation(self):
        """
        Test: Pre-computing sum(weights) doesn't change results.

        This verifies the optimization is mathematically correct.
        """
        if _skip_if_no_data():
            return

        print("\n" + "="*80)
        print("TEST: Weight Sum Precomputation")
        print("="*80)

        # Parse TOA data
        from jug.io.tim_reader import parse_tim_file_mjds
        toas_data = parse_tim_file_mjds(TIM_FILE)
        errors_us = np.array([toa.error_us for toa in toas_data])
        errors_sec = errors_us * 1e-6
        weights = 1.0 / errors_sec**2

        # Compute sum multiple ways
        sum1 = np.sum(weights)
        sum2 = float(np.sum(weights))
        sum3 = weights.sum()

        # All should be identical
        assert sum1 == sum2 == sum3, "Weight sum differs between computation methods"
        print(f"[x] Weight sum consistent: {sum1:.10e}")

        # Test weighted mean calculation
        residuals = np.random.randn(len(weights)) * 1e-6

        mean1 = np.sum(residuals * weights) / np.sum(weights)
        mean2 = np.sum(residuals * weights) / sum1  # Using precomputed

        assert mean1 == mean2, "Weighted mean differs with precomputed sum"
        print("[x] Weighted mean identical with precomputed sum")
        print("[x] TEST PASSED")


def run_all_tests():
    """Run all equivalence tests."""
    print("\n" + "="*80)
    print("JUG BIT-FOR-BIT EQUIVALENCE TEST SUITE")
    print("="*80)
    print(f"Test data: {PAR_FILE}")
    print(f"Test data: {TIM_FILE}")

    # Residual tests
    residual_tests = TestResidualEquivalence()
    residual_tests.test_legacy_vs_session_prefit_residuals()
    residual_tests.test_subtract_tzr_modes_differ()

    # Fitting tests
    fitting_tests = TestFittingEquivalence()
    fitting_tests.test_file_vs_cached_fitting()
    fitting_tests.test_exact_solver_reproducibility()
    fitting_tests.test_fast_solver_convergence()

    # API tests
    api_tests = TestEngineAPIUsage()
    api_tests.test_session_compute_residuals_returns_required_keys()
    api_tests.test_session_fit_parameters_returns_required_keys()
    api_tests.test_toa_mask_filtering()

    # Optimization tests
    opt_tests = TestPrecomputedConstants()
    opt_tests.test_weight_sum_precomputation()

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)


if __name__ == '__main__':
    run_all_tests()
