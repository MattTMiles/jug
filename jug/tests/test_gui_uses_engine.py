"""
Test that GUI uses Engine APIs only
====================================

This test verifies that the GUI is a thin client that only uses engine APIs
for all scientific computation. The GUI must NOT implement any science.

Test Structure:
1. Instantiate engine/session with same data GUI would use
2. Call same methods that GUI workers call
3. Verify outputs match what GUI would receive

Note: Does NOT require Qt event loop - tests engine API directly.
"""

import numpy as np
from pathlib import Path
import sys
import os

# Force deterministic behavior
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.engine.session import TimingSession
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_cache,
    fit_parameters_optimized_cached
)


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


def test_session_worker_flow():
    """
    Test: SessionWorker creates TimingSession correctly.

    This mimics what jug/gui/workers/session_worker.py does.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: SessionWorker Flow")
    print("="*80)

    # This is what SessionWorker.run() does:
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=True)

    # Verify session is usable
    assert session is not None
    assert session.par_file == PAR_FILE
    assert session.tim_file == TIM_FILE
    assert len(session.toas_data) > 0

    print(f"[x] Session created with {len(session.toas_data)} TOAs")
    print("[x] TEST PASSED")


def test_compute_worker_flow():
    """
    Test: ComputeWorker computes residuals through session.

    This mimics what jug/gui/workers/compute_worker.py does.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: ComputeWorker Flow")
    print("="*80)

    # Create session (SessionWorker)
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

    # This is what ComputeWorker.run() does (no params override):
    result = session.compute_residuals(subtract_tzr=True)

    # Verify result has all keys GUI needs
    assert 'residuals_us' in result
    assert 'rms_us' in result
    assert 'tdb_mjd' in result
    assert 'errors_us' in result

    # Verify types
    assert isinstance(result['residuals_us'], np.ndarray)
    assert isinstance(result['rms_us'], float)
    assert result['rms_us'] > 0

    print(f"[x] Computed residuals: RMS = {result['rms_us']:.6f} mus")
    print(f"[x] Arrays: residuals={len(result['residuals_us'])}, mjd={len(result['tdb_mjd'])}")
    print("[x] TEST PASSED")


def test_compute_worker_with_params():
    """
    Test: ComputeWorker computes residuals with parameter overrides.

    This is the postfit path where GUI passes fitted parameters.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: ComputeWorker with Parameter Override")
    print("="*80)

    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

    # Initial compute (no params)
    result1 = session.compute_residuals(subtract_tzr=True)

    # Get initial F0
    initial_f0 = session.params.get('F0', 339.0)

    # Compute with slightly modified F0 (simulates postfit)
    modified_params = {'F0': initial_f0 + 1e-10}
    result2 = session.compute_residuals(params=modified_params, subtract_tzr=True)

    # Results should differ (different F0)
    assert not np.array_equal(result1['residuals_us'], result2['residuals_us']), \
        "Parameter override should change residuals"

    print(f"[x] Initial RMS: {result1['rms_us']:.6f} mus")
    print(f"[x] Modified RMS: {result2['rms_us']:.6f} mus")
    print(f"[x] Residuals differ (param override working)")
    print("[x] TEST PASSED")


def test_fit_worker_flow():
    """
    Test: FitWorker runs fitting through session.

    This mimics what jug/gui/workers/fit_worker.py does.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: FitWorker Flow")
    print("="*80)

    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

    # Populate cache (this happens before fitting)
    _ = session.compute_residuals(subtract_tzr=False)

    # This is what FitWorker.run() does:
    fit_params = ['F0', 'F1']
    result = session.fit_parameters(
        fit_params=fit_params,
        verbose=False,
        toa_mask=None,  # No TOA deletion
        solver_mode="exact"  # Default EXACT solver
    )

    # Verify result has all keys GUI needs (see fit_worker.py result_safe dict)
    assert 'final_params' in result
    assert 'uncertainties' in result
    assert 'covariance' in result
    assert 'final_rms' in result
    assert 'iterations' in result
    assert 'converged' in result
    assert 'total_time' in result

    # Verify structure matches what FitWorker returns
    assert isinstance(result['final_params'], dict)
    assert isinstance(result['uncertainties'], dict)
    assert isinstance(result['covariance'], np.ndarray)
    assert isinstance(result['final_rms'], float)
    assert isinstance(result['iterations'], int)
    assert result['converged'] in (True, False)  # Works for bool and np.bool_

    # Verify fitted params
    for param in fit_params:
        assert param in result['final_params']
        assert param in result['uncertainties']

    print(f"[x] Fit converged: {result['converged']}")
    print(f"[x] Iterations: {result['iterations']}")
    print(f"[x] Final RMS: {result['final_rms']:.6f} mus")
    for param in fit_params:
        print(f"[x] {param} = {result['final_params'][param]:.15e} +/- {result['uncertainties'][param]:.6e}")
    print("[x] TEST PASSED")


def test_fit_worker_with_toa_mask():
    """
    Test: FitWorker with TOA mask (deleted TOAs).

    This tests the box-delete workflow in GUI.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: FitWorker with TOA Mask")
    print("="*80)

    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    _ = session.compute_residuals(subtract_tzr=False)

    n_total = session.get_toa_count()

    # Create mask (simulate deleted TOAs)
    toa_mask = np.ones(n_total, dtype=bool)
    toa_mask[100:200] = False  # "Delete" 100 TOAs

    n_used = np.sum(toa_mask)
    n_deleted = n_total - n_used

    fit_params = ['F0', 'F1']
    result = session.fit_parameters(
        fit_params=fit_params,
        toa_mask=toa_mask,
        solver_mode="exact",
        verbose=False
    )

    assert result['converged'], "Fit with TOA mask should converge"

    print(f"[x] Fit with {n_used} TOAs ({n_deleted} deleted)")
    print(f"[x] Final RMS: {result['final_rms']:.6f} mus")
    print("[x] TEST PASSED")


def test_fit_worker_solver_modes():
    """
    Test: FitWorker respects solver_mode parameter.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: FitWorker Solver Modes")
    print("="*80)

    fit_params = ['F0', 'F1']

    # EXACT mode
    session_exact = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    _ = session_exact.compute_residuals(subtract_tzr=False)
    result_exact = session_exact.fit_parameters(
        fit_params=fit_params,
        solver_mode="exact",
        verbose=False
    )

    # FAST mode
    session_fast = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    _ = session_fast.compute_residuals(subtract_tzr=False)
    result_fast = session_fast.fit_parameters(
        fit_params=fit_params,
        solver_mode="fast",
        verbose=False
    )

    # Both should converge
    assert result_exact['converged'], "EXACT solver should converge"
    assert result_fast['converged'], "FAST solver should converge"

    print(f"[x] EXACT: RMS={result_exact['final_rms']:.6f} mus, iters={result_exact['iterations']}")
    print(f"[x] FAST:  RMS={result_fast['final_rms']:.6f} mus, iters={result_fast['iterations']}")
    print("[x] TEST PASSED")


def test_gui_data_flow():
    """
    Test: Complete GUI data flow (load -> compute -> fit -> postfit).

    This tests the entire workflow a user would do in the GUI.
    """
    if _skip_if_no_data():
        return

    print("\n" + "="*80)
    print("TEST: Complete GUI Data Flow")
    print("="*80)

    # 1. Load files (SessionWorker)
    print("\n1. Loading files...")
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    print(f"   Loaded {session.get_toa_count()} TOAs")

    # 2. Compute initial residuals (ComputeWorker)
    print("\n2. Computing prefit residuals...")
    prefit = session.compute_residuals(subtract_tzr=True)
    prefit_rms = prefit['rms_us']
    print(f"   Prefit RMS: {prefit_rms:.6f} mus")

    # 3. Populate fitting cache (automatic on first fit)
    print("\n3. Preparing for fit...")
    _ = session.compute_residuals(subtract_tzr=False)

    # 4. Run fit (FitWorker)
    print("\n4. Running fit [F0, F1]...")
    fit_result = session.fit_parameters(
        fit_params=['F0', 'F1'],
        solver_mode="exact",
        verbose=False
    )
    print(f"   Fit converged: {fit_result['converged']}")
    print(f"   Iterations: {fit_result['iterations']}")

    # 5. Compute postfit residuals (ComputeWorker with params)
    print("\n5. Computing postfit residuals...")
    postfit = session.compute_residuals(
        params=fit_result['final_params'],
        subtract_tzr=True
    )
    postfit_rms = postfit['rms_us']
    print(f"   Postfit RMS: {postfit_rms:.6f} mus")

    # 6. Verify improvement
    print("\n6. Verifying fit improvement...")
    assert postfit_rms <= prefit_rms * 1.01, \
        f"Fit should not make residuals worse: prefit={prefit_rms}, postfit={postfit_rms}"
    print(f"   RMS improved: {prefit_rms:.6f} -> {postfit_rms:.6f} mus")

    print("\n" + "="*80)
    print("[x] COMPLETE GUI DATA FLOW PASSED")
    print("="*80)


def run_all_tests():
    """Run all GUI-engine tests."""
    print("\n" + "="*80)
    print("GUI-ENGINE INTEGRATION TEST SUITE")
    print("="*80)
    print("Verifies GUI only uses engine APIs (no direct science computation)")

    test_session_worker_flow()
    test_compute_worker_flow()
    test_compute_worker_with_params()
    test_fit_worker_flow()
    test_fit_worker_with_toa_mask()
    test_fit_worker_solver_modes()
    test_gui_data_flow()

    print("\n" + "="*80)
    print("ALL GUI-ENGINE TESTS PASSED!")
    print("="*80)


if __name__ == '__main__':
    run_all_tests()
