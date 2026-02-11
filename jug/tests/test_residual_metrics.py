"""Tests for Phase 3.2–3.3: residual representations and fit summary metrics."""

import numpy as np
import pytest

from jug.engine.stats import (
    compute_normalized_residuals,
    compute_whitened_residuals,
    build_residual_representations,
    fit_summary,
)


# ---------------------------------------------------------------------------
# Phase 3.2 — Residual representations
# ---------------------------------------------------------------------------

class TestNormalizedResiduals:

    def test_basic(self):
        r = np.array([1.0, 2.0, 3.0])
        e = np.array([1.0, 1.0, 1.0])
        norm = compute_normalized_residuals(r, e)
        np.testing.assert_allclose(norm, [1.0, 2.0, 3.0])

    def test_scaling(self):
        r = np.array([4.0, 6.0])
        e = np.array([2.0, 3.0])
        norm = compute_normalized_residuals(r, e)
        np.testing.assert_allclose(norm, [2.0, 2.0])

    def test_zero_error_gives_zero(self):
        """Zero errors should not produce NaN/Inf — produce 0 instead."""
        r = np.array([1.0, 2.0])
        e = np.array([0.0, 1.0])
        norm = compute_normalized_residuals(r, e)
        assert np.isfinite(norm).all()
        assert norm[0] == 0.0  # 1.0 / inf = 0.0
        assert norm[1] == 2.0

    def test_negative_error_gives_zero(self):
        r = np.array([1.0])
        e = np.array([-1.0])
        norm = compute_normalized_residuals(r, e)
        assert norm[0] == 0.0


class TestWhitenedResiduals:

    def test_no_noise_cov(self):
        """Without noise_cov_diag, should be same as normalized."""
        r = np.array([4.0, 6.0])
        e = np.array([2.0, 3.0])
        w = compute_whitened_residuals(r, e)
        norm = compute_normalized_residuals(r, e)
        np.testing.assert_allclose(w, norm)

    def test_with_noise_cov(self):
        """σ_eff = sqrt(σ² + C_diag), so whitened = r / σ_eff."""
        r = np.array([6.0])
        e = np.array([3.0])  # σ² = 9
        cov = np.array([16.0])  # total_var = 25, σ_eff = 5
        w = compute_whitened_residuals(r, e, cov)
        np.testing.assert_allclose(w, [6.0 / 5.0])

    def test_noise_cov_reduces_amplitude(self):
        """Adding noise covariance should reduce whitened residual magnitude."""
        r = np.array([10.0])
        e = np.array([1.0])
        norm = compute_whitened_residuals(r, e)
        whitened = compute_whitened_residuals(r, e, noise_cov_diag=np.array([8.0]))
        assert abs(whitened[0]) < abs(norm[0])


class TestBuildResidualRepresentations:

    def test_keys_present(self):
        pre = np.array([1.0, 2.0])
        post = np.array([0.5, 1.0])
        err = np.array([0.1, 0.2])
        reps = build_residual_representations(pre, post, err)
        assert "prefit_us" in reps
        assert "postfit_us" in reps
        assert "normalized" in reps
        assert "whitened" in reps

    def test_values_correct(self):
        pre = np.array([10.0])
        post = np.array([2.0])
        err = np.array([1.0])
        reps = build_residual_representations(pre, post, err)
        np.testing.assert_allclose(reps["prefit_us"], [10.0])
        np.testing.assert_allclose(reps["postfit_us"], [2.0])
        np.testing.assert_allclose(reps["normalized"], [2.0])

    def test_with_noise(self):
        pre = np.array([10.0])
        post = np.array([6.0])
        err = np.array([3.0])   # σ² = 9
        cov = np.array([16.0])  # total = 25 → σ_eff = 5
        reps = build_residual_representations(pre, post, err, cov)
        np.testing.assert_allclose(reps["whitened"], [6.0 / 5.0])


# ---------------------------------------------------------------------------
# Phase 3.3 — Fit summary
# ---------------------------------------------------------------------------

class TestFitSummary:

    @pytest.fixture
    def basic_fit(self):
        """A simple 100-TOA fit scenario."""
        np.random.seed(42)
        n = 100
        prefit = np.random.normal(0, 5, n)
        postfit = np.random.normal(0, 1, n)
        errors = np.ones(n) * 1.0
        return {
            "prefit_us": prefit,
            "postfit_us": postfit,
            "errors_us": errors,
            "fitted_params": {"F0": 100.0, "F1": -1e-15},
            "initial_params": {"F0": 99.9, "F1": -1.1e-15},
            "iterations": 5,
            "converged": True,
        }

    def test_all_keys_present(self, basic_fit):
        s = fit_summary(**basic_fit)
        assert "prefit_stats" in s
        assert "postfit_stats" in s
        assert "chi2" in s
        assert "param_deltas" in s
        assert "iterations" in s
        assert "converged" in s
        assert "warnings" in s
        assert "residuals" in s

    def test_param_deltas(self, basic_fit):
        s = fit_summary(**basic_fit)
        np.testing.assert_allclose(s["param_deltas"]["F0"], 0.1, atol=1e-12)

    def test_convergence_warning(self, basic_fit):
        basic_fit["converged"] = False
        s = fit_summary(**basic_fit)
        assert any("converge" in w.lower() for w in s["warnings"])

    def test_no_warnings_good_fit(self):
        """A perfectly normal fit should produce no warnings."""
        n = 50
        postfit = np.random.normal(0, 1.0, n)
        errors = np.ones(n) * 1.0
        prefit = postfit * 5  # prefit bigger than postfit
        s = fit_summary(
            prefit, postfit, errors,
            fitted_params={"F0": 1.0},
            initial_params={"F0": 0.9},
            iterations=3,
            converged=True,
        )
        # Should have no warnings (chi2_reduced ~1 for unit normal residuals)
        diverge_warns = [w for w in s["warnings"] if "diverged" in w.lower()]
        converge_warns = [w for w in s["warnings"] if "converge" in w.lower()]
        assert len(diverge_warns) == 0
        assert len(converge_warns) == 0

    def test_diverged_warning(self):
        """Post-fit wRMS > pre-fit wRMS should warn of divergence."""
        n = 20
        prefit = np.ones(n) * 0.1
        postfit = np.ones(n) * 10.0  # much worse
        errors = np.ones(n)
        s = fit_summary(
            prefit, postfit, errors,
            fitted_params={"F0": 1.0},
            initial_params={"F0": 1.0},
            converged=True,
        )
        assert any("diverged" in w.lower() for w in s["warnings"])

    def test_residual_representations_included(self, basic_fit):
        s = fit_summary(**basic_fit)
        assert "normalized" in s["residuals"]
        assert "whitened" in s["residuals"]
        assert len(s["residuals"]["postfit_us"]) == 100
