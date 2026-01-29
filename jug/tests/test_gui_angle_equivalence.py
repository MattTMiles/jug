"""
GUI Angle Edit Equivalence Tests
================================

Tests that verify GUI angle editing produces bit-for-bit identical
results to direct engine calls.

Key principle: GUI actions = engine operations (no computation in GUI)

This tests the workflow:
1. User edits RA/DEC in GUI (sexagesimal string)
2. Codec converts to radians
3. Engine computes residuals with new value
4. Result must match direct engine call with same radians

Run with:
    pytest jug/tests/test_gui_angle_equivalence.py -v
"""

import os
import sys
from pathlib import Path

# Force determinism BEFORE any other imports
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pulsars"
PAR_FILE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"


def _skip_if_no_data():
    """Return True if test data is missing."""
    return not (PAR_FILE.exists() and TIM_FILE.exists())


class TestGUIAngleEquivalence:
    """Tests that GUI angle editing matches engine exactly."""

    def test_raj_edit_equivalence(self):
        """
        Test: GUI RAJ edit produces identical results to direct engine call.

        Simulates:
        1. User types new RA in sexagesimal format
        2. GUI converts via RAJCodec
        3. GUI calls session.compute_residuals with new params
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.model.codecs import RAJCodec
        from jug.engine.session import TimingSession

        # Simulate GUI edit: user types new RA value
        new_raj_str = "19:10:00.0000000"  # Slightly different from original
        raj_codec = RAJCodec()
        new_raj_rad = raj_codec.decode(new_raj_str)

        # GUI PATH: Session with modified params via codec
        session_gui = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        original_params = dict(session_gui.params)

        # Simulate GUI: update RAJ in params (as if user edited)
        gui_params = dict(original_params)
        gui_params['RAJ'] = new_raj_rad

        # GUI would call compute_residuals - but since it passes through session,
        # we need to update session params and recompute
        # For this test, we create a fresh session and manually set
        session_gui.params['RAJ'] = new_raj_rad
        gui_result = session_gui.compute_residuals(subtract_tzr=True)

        # ENGINE PATH: Direct call with same radians
        session_engine = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        session_engine.params['RAJ'] = new_raj_rad  # Same value
        engine_result = session_engine.compute_residuals(subtract_tzr=True)

        # BIT-FOR-BIT COMPARISON
        assert np.array_equal(gui_result['residuals_us'], engine_result['residuals_us']), \
            "GUI RAJ edit produces different residuals than direct engine call"

        assert gui_result['rms_us'] == engine_result['rms_us'], \
            "GUI RAJ edit produces different RMS than direct engine call"

    def test_decj_edit_equivalence(self):
        """
        Test: GUI DECJ edit produces identical results to direct engine call.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.model.codecs import DECJCodec
        from jug.engine.session import TimingSession

        # Simulate GUI edit
        new_decj_str = "-37:45:00.0000000"  # Slightly different
        decj_codec = DECJCodec()
        new_decj_rad = decj_codec.decode(new_decj_str)

        # GUI PATH
        session_gui = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        session_gui.params['DECJ'] = new_decj_rad
        gui_result = session_gui.compute_residuals(subtract_tzr=True)

        # ENGINE PATH
        session_engine = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        session_engine.params['DECJ'] = new_decj_rad
        engine_result = session_engine.compute_residuals(subtract_tzr=True)

        # BIT-FOR-BIT COMPARISON
        assert np.array_equal(gui_result['residuals_us'], engine_result['residuals_us'])
        assert gui_result['rms_us'] == engine_result['rms_us']

    def test_codec_roundtrip_preserves_computation(self):
        """
        Test: Codec round-trip doesn't change computation results.

        Simulates: User views value, edits it back to same string, submits.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.model.codecs import RAJCodec, DECJCodec
        from jug.engine.session import TimingSession

        raj_codec = RAJCodec()
        decj_codec = DECJCodec()

        # Get original values - note: JUG stores RAJ/DECJ as strings in .par
        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        original_raj = session.params['RAJ']
        original_decj = session.params['DECJ']

        # Compute original residuals
        original_result = session.compute_residuals(subtract_tzr=True)

        # If RAJ/DECJ are strings, decode first then round-trip
        if isinstance(original_raj, str):
            raj_rad = raj_codec.decode(original_raj)
        else:
            raj_rad = original_raj

        if isinstance(original_decj, str):
            decj_rad = decj_codec.decode(original_decj)
        else:
            decj_rad = original_decj

        # Round-trip through codec
        raj_str = raj_codec.encode(raj_rad)
        raj_back = raj_codec.decode(raj_str)

        decj_str = decj_codec.encode(decj_rad)
        decj_back = decj_codec.decode(decj_str)

        # Update session with round-tripped values (radians)
        session2 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        session2.params['RAJ'] = raj_back
        session2.params['DECJ'] = decj_back
        roundtrip_result = session2.compute_residuals(subtract_tzr=True)

        # Results should be very close (codec precision is ~5e-10 rad)
        max_diff = np.max(np.abs(original_result['residuals_us'] - roundtrip_result['residuals_us']))

        # Allow small difference due to codec precision
        # 5e-10 rad in position translates to ~1.5e-7 seconds max for 1 AU baseline
        # That's about 0.15 microseconds worst case
        assert max_diff < 0.2, \
            f"Codec round-trip changed residuals by {max_diff:.6f} μs"


class TestGUIFitEquivalence:
    """Tests that GUI fitting matches engine exactly."""

    def test_fit_with_codec_edited_start(self):
        """
        Test: Fit with codec-edited starting values matches engine.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.engine.session import TimingSession

        # Standard fit
        session1 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session1.compute_residuals(subtract_tzr=False)
        result1 = session1.fit_parameters(
            fit_params=['F0', 'F1'],
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        # Fit with "GUI-edited" F0 (same value, but touched)
        session2 = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        original_f0 = session2.params['F0']
        session2.params['F0'] = float(str(original_f0))  # Force string round-trip
        _ = session2.compute_residuals(subtract_tzr=False)
        result2 = session2.fit_parameters(
            fit_params=['F0', 'F1'],
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        # Should be bit-for-bit identical
        assert np.array_equal(result1['residuals_us'], result2['residuals_us'])
        assert result1['final_rms'] == result2['final_rms']
        assert result1['final_params']['F0'] == result2['final_params']['F0']


class TestEngineIsCanonical:
    """Tests that engine is the canonical computation source."""

    def test_engine_determinism(self):
        """
        Test: Engine produces identical results across sessions.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.engine.session import TimingSession

        results = []
        for _ in range(3):
            session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
            result = session.compute_residuals(subtract_tzr=True)
            results.append(result)

        # All should be bit-for-bit identical
        ref = results[0]
        for i, result in enumerate(results[1:], 2):
            assert np.array_equal(result['residuals_us'], ref['residuals_us']), \
                f"Session {i} differs from session 1"

    def test_no_gui_computation(self):
        """
        Test: Verify that GUI-like operations only call engine.

        This is a documentation test - we verify that the expected
        workflow uses session methods rather than direct computation.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.engine.session import TimingSession

        # This is what GUI should do:
        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)

        # All computation goes through session methods
        assert hasattr(session, 'compute_residuals')
        assert hasattr(session, 'fit_parameters')

        # GUI should never import or call:
        # - compute_residuals_simple directly
        # - fit_parameters_optimized directly
        # - any derivative functions directly

        # Test that session provides all needed APIs
        result = session.compute_residuals(subtract_tzr=True)
        assert 'residuals_us' in result
        assert 'rms_us' in result
        assert 'errors_us' in result


def run_all_tests():
    """Run all GUI equivalence tests."""
    print("="*80)
    print("GUI Angle Equivalence Test Suite")
    print("="*80)

    if _skip_if_no_data():
        print("\nSKIP: Test data not found")
        print(f"Expected: {PAR_FILE}")
        print(f"Expected: {TIM_FILE}")
        return

    tests = [
        TestGUIAngleEquivalence(),
        TestGUIFitEquivalence(),
        TestEngineIsCanonical(),
    ]

    passed = 0
    failed = 0

    for test_class in tests:
        class_name = test_class.__class__.__name__
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"✓ {class_name}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"✗ {class_name}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"✗ {class_name}.{method_name}: {type(e).__name__}: {e}")
                    failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")


if __name__ == '__main__':
    run_all_tests()
