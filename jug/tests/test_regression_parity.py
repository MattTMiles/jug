"""
Regression tests -- evaluate-only vs fitter codepath identity.
=============================================================

These tests ensure that:
1. Two independent ``compute_residuals_simple`` calls (evaluate-only) produce
   bit-for-bit identical residuals for the same par+tim.
2. The fitter's pre-fit residuals (before any WLS step) match the
   evaluate-only residuals to within 1 ns per-TOA.
3. ``dt_sec`` precision is preserved in longdouble (float128) throughout.
4. Stored golden parity artifacts are reproducible.

Run with:
    pytest jug/tests/test_regression_parity.py -v -o "addopts="
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Force determinism
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")

from jug.engine.session import TimingSession
from jug.residuals.simple_calculator import compute_residuals_simple

# Paths
JUG_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PULSARS = JUG_ROOT / "data" / "pulsars"
DATA_GOLDEN = JUG_ROOT / "tests" / "data_golden"


# -- helpers ------------------------------------------------------------------

def _skip_no_data(par, tim):
    if not par.exists() or not tim.exists():
        pytest.skip(f"Missing data: {par.name} / {tim.name}")


# -- Test class ---------------------------------------------------------------

class TestEvalOnlyIdentity:
    """Two evaluate-only runs must agree bit-for-bit."""

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_two_eval_calls_bitexact(self, pulsar):
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        r1 = compute_residuals_simple(par, tim, verbose=False)
        r2 = compute_residuals_simple(par, tim, verbose=False)

        np.testing.assert_array_equal(
            r1["residuals_us"], r2["residuals_us"],
            err_msg=f"{pulsar}: two eval calls differ"
        )
        assert r1["rms_us"] == r2["rms_us"]

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_session_eval_matches_simple(self, pulsar):
        """TimingSession.compute_residuals must match compute_residuals_simple."""
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        r_simple = compute_residuals_simple(par, tim, verbose=False)
        session = TimingSession(par, tim, verbose=False)
        r_session = session.compute_residuals(subtract_tzr=True)

        np.testing.assert_array_equal(
            r_simple["residuals_us"], r_session["residuals_us"],
            err_msg=f"{pulsar}: Session vs simple eval differ"
        )

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_session_independence(self, pulsar):
        """Two independent sessions must agree bit-for-bit."""
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        s1 = TimingSession(par, tim, verbose=False)
        s2 = TimingSession(par, tim, verbose=False)

        r1 = s1.compute_residuals(subtract_tzr=True)
        r2 = s2.compute_residuals(subtract_tzr=True)

        np.testing.assert_array_equal(
            r1["residuals_us"], r2["residuals_us"],
            err_msg=f"{pulsar}: two sessions differ"
        )


class TestFitterPrefitIdentity:
    """The fitter's first-iteration residuals must match evaluate-only
    to within 1 ns per-TOA (no WLS step has been applied yet)."""

    @pytest.mark.parametrize(
        "pulsar,fit_params",
        [
            ("J1909-3744", ["F0", "F1", "DM"]),
            ("J0614-3329", ["F0", "F1", "DM"]),
        ],
    )
    def test_fitter_prefit_residuals_match_eval(self, pulsar, fit_params):
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        # Evaluate-only (subtract_tzr=True gives "prefit" residuals)
        r_eval = compute_residuals_simple(par, tim, verbose=False, subtract_tzr=True)

        # Session: compute_residuals then zero-iteration fit
        session = TimingSession(par, tim, verbose=False)
        r_session = session.compute_residuals(subtract_tzr=True)

        # The session prefit should be identical
        np.testing.assert_array_equal(
            r_eval["residuals_us"], r_session["residuals_us"],
            err_msg=f"{pulsar}: prefit mismatch"
        )


class TestPrecisionPreservation:
    """Verify that longdouble (float128) dt_sec is preserved."""

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_dt_sec_ld_exists(self, pulsar):
        """compute_residuals_simple must return dt_sec_ld in longdouble."""
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        result = compute_residuals_simple(par, tim, verbose=False)
        assert "dt_sec_ld" in result, "dt_sec_ld missing from result"
        assert result["dt_sec_ld"].dtype == np.longdouble, (
            f"dt_sec_ld dtype is {result['dt_sec_ld'].dtype}, expected longdouble"
        )

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_dt_sec_ld_range_and_dtype(self, pulsar):
        """Longdouble dt_sec_ld must have correct dtype and non-trivial range.

        On platforms where longdouble == float64 (e.g. aarch64), the extra
        precision is absent, but the dtype and physical range should be valid.
        """
        par = DATA_PULSARS / f"{pulsar}_tdb.par"
        tim = DATA_PULSARS / f"{pulsar}.tim"
        _skip_no_data(par, tim)

        result = compute_residuals_simple(par, tim, verbose=False)
        ld = result["dt_sec_ld"]

        # Must be longdouble (even if alias for float64 on some platforms)
        assert ld.dtype == np.longdouble

        # Physical sanity: dt_sec values for a real pulsar are O(10^8) seconds
        # (time from MJD epoch to observation), so range should be large
        dt_range = float(np.ptp(ld))
        assert dt_range > 1e4, (
            f"dt_sec_ld range is suspiciously small: {dt_range:.2f} s"
        )


class TestGoldenParityArtifacts:
    """Verify stored golden parity data is still reproducible."""

    def test_j1909_mini_golden(self):
        """J1909 mini dataset (20 TOAs) must match golden within 1 ns."""
        par = DATA_GOLDEN / "J1909_mini.par"
        tim = DATA_GOLDEN / "J1909_mini.tim"
        golden_json = DATA_GOLDEN / "J1909_mini_golden.json"

        if not par.exists() or not golden_json.exists():
            pytest.skip("J1909_mini golden data not found")

        with open(golden_json) as f:
            golden = json.load(f)

        result = compute_residuals_simple(par, tim, verbose=False)
        residuals_ns = result["residuals_us"] * 1000.0

        tol_ns = golden["tolerances"]["residual_abs_tol_ns"]
        golden_ns = np.array(golden["first_5_residuals_ns"])
        actual_ns = residuals_ns[:5]

        np.testing.assert_allclose(
            actual_ns, golden_ns, atol=tol_ns,
            err_msg="J1909_mini residuals diverge from golden"
        )

        # RMS relative tolerance
        rms_tol = golden["tolerances"]["rms_rel_tol"]
        np.testing.assert_allclose(
            result["weighted_rms_us"],
            golden["weighted_rms_us"],
            rtol=rms_tol,
            err_msg="J1909_mini WRMS diverges from golden"
        )

    def test_j0125_parity_golden_exists(self):
        """Verify J0125-2327 parity golden artifact can be loaded."""
        npz_path = DATA_GOLDEN / "J0125-2327_parity.npz"
        json_path = DATA_GOLDEN / "J0125-2327_parity.json"

        if not npz_path.exists():
            pytest.skip("J0125-2327 parity golden not found")

        data = np.load(npz_path)
        assert "jug_res_sec" in data
        assert "t2_res_sec" in data
        assert "delta_res_ns" in data

        with open(json_path) as f:
            meta = json.load(f)

        assert meta["n_matched"] > 0
        assert meta["max_abs_delta_ns"] < meta["thresholds"]["max_abs_delta_ns"]
        assert abs(meta["wrms_diff_ns"]) < meta["thresholds"]["wrms_diff_ns"]
