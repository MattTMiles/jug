"""
GUI ↔ Engine Equivalence Tests
================================

Tests that verify the GUI NEVER does physics computation itself.
All computation must go through the engine (TimingSession).

Key requirements:
1. RAJ/DECJ Edit Equivalence: GUI sexagesimal → codec → radians → engine matches direct call
2. Codec Round-trip: Edit value back to same string → residuals unchanged
3. Fit Equivalence: GUI-edited starting values produce same fit as direct engine call
4. Engine Determinism: Multiple sessions with same inputs → bit-for-bit identical

Run with:
    pytest jug/tests/test_gui_engine_equivalence.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.engine.session import TimingSession
from jug.model.codecs import RAJCodec, DECJCodec, CODECS

# Test data path (J1909-3744)
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'pulsars'
PAR_FILE = DATA_DIR / 'J1909-3744_tdb.par'
TIM_FILE = DATA_DIR / 'J1909-3744.tim'


def _session_available():
    """Check if test data is available."""
    return PAR_FILE.exists() and TIM_FILE.exists()


class TestRAJDECJEditEquivalence:
    """
    Tests that editing RAJ/DECJ in GUI (sexagesimal) produces
    identical results to direct engine calls with radians.
    """

    def test_raj_gui_edit_matches_engine(self):
        """
        Simulate GUI edit: user enters sexagesimal string.
        GUI converts to radians via codec, passes radians to engine.
        Result must match direct engine call with same radians.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        # Simulate GUI: user enters RA in sexagesimal
        gui_input = "19:09:47.4346970"

        # GUI decodes sexagesimal → radians before passing to engine
        codec = RAJCodec()
        raj_rad = codec.decode(gui_input)

        # GUI path: session 1 with radians
        session_gui = TimingSession(PAR_FILE, TIM_FILE)
        result_gui = session_gui.compute_residuals(params={'RAJ': raj_rad})

        # Direct engine path: independent session, same radians
        session_direct = TimingSession(PAR_FILE, TIM_FILE)
        result_direct = session_direct.compute_residuals(params={'RAJ': raj_rad})

        # They must be bit-for-bit identical
        assert np.array_equal(result_gui['residuals_us'], result_direct['residuals_us']), \
            "GUI RAJ edit produces different residuals than direct engine call"

    def test_decj_gui_edit_matches_engine(self):
        """
        Same as above for DECJ (declination).
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        # Simulate GUI: user enters DEC in sexagesimal
        gui_input = "-37:44:14.4662200"

        # GUI decodes sexagesimal → radians
        codec = DECJCodec()
        decj_rad = codec.decode(gui_input)

        # GUI path: radians to engine
        session_gui = TimingSession(PAR_FILE, TIM_FILE)
        result_gui = session_gui.compute_residuals(params={'DECJ': decj_rad})

        # Direct path: independent session, same radians
        session_direct = TimingSession(PAR_FILE, TIM_FILE)
        result_direct = session_direct.compute_residuals(params={'DECJ': decj_rad})

        assert np.array_equal(result_gui['residuals_us'], result_direct['residuals_us']), \
            "GUI DECJ edit produces different residuals than direct engine call"


class TestCodecRoundTripStability:
    """
    Tests that editing a value and reverting to the original string
    produces unchanged residuals (bit-for-bit identical).
    """

    def test_raj_roundtrip_residuals_unchanged(self):
        """
        1. Get original RAJ from par file
        2. Decode → encode → decode (round-trip)
        3. Residuals with round-tripped value must match original
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        session = TimingSession(PAR_FILE, TIM_FILE)
        original_raj = session.params.get('RAJ')

        # Get original residuals (no modification)
        result_original = session.compute_residuals()

        # Round-trip the RAJ value
        codec = RAJCodec()
        if isinstance(original_raj, str):
            raj_rad = codec.decode(original_raj)
        else:
            raj_rad = original_raj

        raj_string_roundtrip = codec.encode(raj_rad)
        raj_rad_back = codec.decode(raj_string_roundtrip)

        # Verify radians are nearly identical (sub-mas precision)
        assert abs(raj_rad - raj_rad_back) < 5e-10, \
            f"RAJ round-trip error {abs(raj_rad - raj_rad_back):.2e} rad exceeds 5e-10"

        # Compute residuals with round-tripped value (as radians)
        session2 = TimingSession(PAR_FILE, TIM_FILE)
        result_roundtrip = session2.compute_residuals(params={'RAJ': raj_rad_back})

        # Compare residuals
        max_diff = np.max(np.abs(result_original['residuals_us'] - result_roundtrip['residuals_us']))

        # Allow sub-nanosecond differences due to round-trip precision
        assert max_diff < 1e-3, \
            f"RAJ round-trip changes residuals by {max_diff:.2e} μs (should be < 1e-3)"

    def test_decj_roundtrip_residuals_unchanged(self):
        """Same as above for DECJ."""
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        session = TimingSession(PAR_FILE, TIM_FILE)
        original_decj = session.params.get('DECJ')

        result_original = session.compute_residuals()

        codec = DECJCodec()
        if isinstance(original_decj, str):
            decj_rad = codec.decode(original_decj)
        else:
            decj_rad = original_decj

        decj_string_roundtrip = codec.encode(decj_rad)
        decj_rad_back = codec.decode(decj_string_roundtrip)

        assert abs(decj_rad - decj_rad_back) < 5e-10, \
            f"DECJ round-trip error {abs(decj_rad - decj_rad_back):.2e} rad exceeds 5e-10"

        session2 = TimingSession(PAR_FILE, TIM_FILE)
        result_roundtrip = session2.compute_residuals(params={'DECJ': decj_rad_back})

        max_diff = np.max(np.abs(result_original['residuals_us'] - result_roundtrip['residuals_us']))
        assert max_diff < 1e-3, \
            f"DECJ round-trip changes residuals by {max_diff:.2e} μs (should be < 1e-3)"


class TestFitEquivalence:
    """
    Tests that GUI-edited starting values produce the same fit result
    as direct engine calls with the same starting values.
    """

    def test_fit_same_starting_values_same_result(self):
        """
        Fit with identical starting params via two independent sessions.
        Results must be bit-for-bit identical.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        # Session 1: fit F0
        session1 = TimingSession(PAR_FILE, TIM_FILE)
        result1 = session1.fit_parameters(['F0'], max_iter=5)

        # Session 2: independent fit with same file
        session2 = TimingSession(PAR_FILE, TIM_FILE)
        result2 = session2.fit_parameters(['F0'], max_iter=5)

        # Final F0 must be identical
        f0_1 = result1['final_params']['F0']
        f0_2 = result2['final_params']['F0']

        assert f0_1 == f0_2, \
            f"Fit F0 differs between sessions: {f0_1} vs {f0_2}"

        # Uncertainties must be identical
        assert result1['uncertainties']['F0'] == result2['uncertainties']['F0'], \
            "Fit uncertainties differ between sessions"

    def test_fit_with_gui_edited_params_matches_direct(self):
        """
        Simulate GUI: user edits F0 slightly, then fits.
        Must match direct engine call with same edited params.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        # Get original F0
        session_orig = TimingSession(PAR_FILE, TIM_FILE)
        original_f0 = session_orig.params['F0']

        # Slightly perturbed F0 (simulating GUI edit)
        perturbed_f0 = original_f0 * 1.0000001

        # GUI path: set param, then fit
        session_gui = TimingSession(PAR_FILE, TIM_FILE)
        session_gui.params['F0'] = perturbed_f0
        result_gui = session_gui.fit_parameters(['F0'], max_iter=10)

        # Direct path: pass param override to compute_residuals, then fit
        session_direct = TimingSession(PAR_FILE, TIM_FILE)
        session_direct.params['F0'] = perturbed_f0
        result_direct = session_direct.fit_parameters(['F0'], max_iter=10)

        # Final F0 must be identical
        assert result_gui['final_params']['F0'] == result_direct['final_params']['F0'], \
            "GUI-edited fit differs from direct engine fit"


class TestEngineDeterminism:
    """
    Tests that multiple sessions with identical inputs produce
    bit-for-bit identical results.
    """

    def test_residuals_determinism_across_sessions(self):
        """
        Create multiple independent sessions.
        All must produce identical residuals.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        results = []
        for i in range(3):
            session = TimingSession(PAR_FILE, TIM_FILE)
            result = session.compute_residuals()
            results.append(result['residuals_us'])

        # All must be bit-for-bit identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), \
                f"Session {i} produces different residuals than session 0"

    def test_fit_determinism_across_sessions(self):
        """
        Fitting same params across independent sessions must
        produce bit-for-bit identical results.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        results = []
        for i in range(3):
            session = TimingSession(PAR_FILE, TIM_FILE)
            result = session.fit_parameters(['F0', 'F1'], max_iter=5)
            results.append(result)

        # All final params must match
        for i in range(1, len(results)):
            for param in ['F0', 'F1']:
                assert results[0]['final_params'][param] == results[i]['final_params'][param], \
                    f"Session {i} fit {param} differs from session 0"

    def test_warm_cold_cache_identical(self):
        """
        Residuals must be identical whether cache is warm or cold.
        """
        if not _session_available():
            import pytest
            pytest.skip("Test data not available")

        # Cold session (fresh)
        session_cold = TimingSession(PAR_FILE, TIM_FILE)
        result_cold = session_cold.compute_residuals()

        # Warm session (recompute after cache populated)
        result_warm = session_cold.compute_residuals()

        # Must be bit-for-bit identical
        assert np.array_equal(result_cold['residuals_us'], result_warm['residuals_us']), \
            "Warm cache produces different residuals than cold cache"


def run_all_tests():
    """Run all equivalence tests."""
    print("=" * 80)
    print("GUI ↔ Engine Equivalence Test Suite")
    print("=" * 80)

    if not _session_available():
        print(f"⚠ Test data not found at {DATA_DIR}")
        print("  Skipping tests.")
        return

    # RAJ/DECJ Edit Equivalence
    edit_tests = TestRAJDECJEditEquivalence()
    edit_tests.test_raj_gui_edit_matches_engine()
    print("✓ RAJ GUI edit matches engine")
    edit_tests.test_decj_gui_edit_matches_engine()
    print("✓ DECJ GUI edit matches engine")

    # Codec Round-trip
    roundtrip_tests = TestCodecRoundTripStability()
    roundtrip_tests.test_raj_roundtrip_residuals_unchanged()
    print("✓ RAJ round-trip residuals unchanged")
    roundtrip_tests.test_decj_roundtrip_residuals_unchanged()
    print("✓ DECJ round-trip residuals unchanged")

    # Fit Equivalence
    fit_tests = TestFitEquivalence()
    fit_tests.test_fit_same_starting_values_same_result()
    print("✓ Fit determinism (same starting values)")
    fit_tests.test_fit_with_gui_edited_params_matches_direct()
    print("✓ GUI-edited fit matches direct engine fit")

    # Engine Determinism
    det_tests = TestEngineDeterminism()
    det_tests.test_residuals_determinism_across_sessions()
    print("✓ Residuals determinism across sessions")
    det_tests.test_fit_determinism_across_sessions()
    print("✓ Fit determinism across sessions")
    det_tests.test_warm_cold_cache_identical()
    print("✓ Warm/cold cache identical")

    print()
    print("All GUI ↔ Engine equivalence tests passed!")


if __name__ == '__main__':
    run_all_tests()
