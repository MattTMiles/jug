"""Tests for ECORR epoch-correlated noise: grouping, whitening, chi2.

Tests cover:
- TOA epoch grouping by time proximity and backend flag
- ECORRWhitener construction from noise entries
- Block-Cholesky whitening of residuals and design matrices
- Chi2 computation with off-diagonal correlations
- Integration: ECORR entries parsed from par -> whitener built -> correct chi2
- Edge cases: no ECORR, single-TOA epochs, multiple backends
"""

import numpy as np
import pytest

from jug.noise.white import WhiteNoiseEntry, parse_noise_lines
from jug.noise.ecorr import (
    EpochGroup,
    ECORRWhitener,
    _group_toas_into_epochs,
    build_ecorr_whitener,
)


# ===================================================================
# _group_toas_into_epochs
# ===================================================================

class TestGroupTOAsIntoEpochs:
    """Test TOA epoch grouping by time proximity."""

    def test_single_epoch(self):
        """All TOAs within dt_days -> one epoch."""
        toas_mjd = np.array([50000.0, 50000.1, 50000.2])
        mask = np.ones(3, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        assert len(epochs) == 1
        assert set(epochs[0]) == {0, 1, 2}

    def test_two_epochs(self):
        """Gap > dt_days splits into two epochs."""
        toas_mjd = np.array([50000.0, 50000.1, 50001.0, 50001.1])
        mask = np.ones(4, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        assert len(epochs) == 2
        assert set(epochs[0]) == {0, 1}
        assert set(epochs[1]) == {2, 3}

    def test_singletons_excluded(self):
        """Epochs with only 1 TOA are not returned."""
        toas_mjd = np.array([50000.0, 50001.0, 50002.0, 50002.1])
        mask = np.ones(4, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        # TOAs 0 and 1 are singletons, TOAs 2+3 form one epoch
        assert len(epochs) == 1
        assert set(epochs[0]) == {2, 3}

    def test_mask_selects_subset(self):
        """Only masked TOAs participate in grouping."""
        toas_mjd = np.array([50000.0, 50000.1, 50000.2, 50001.0])
        mask = np.array([True, False, True, True])
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        # TOAs 0 and 2 form one epoch (both within 0.5 days); TOA 3 is singleton
        assert len(epochs) == 1
        assert set(epochs[0]) == {0, 2}

    def test_empty_mask(self):
        """No TOAs selected -> no epochs."""
        toas_mjd = np.array([50000.0, 50000.1])
        mask = np.zeros(2, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        assert epochs == []

    def test_one_toa(self):
        """Single TOA -> no epoch (need >= 2)."""
        toas_mjd = np.array([50000.0])
        mask = np.ones(1, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        assert epochs == []

    def test_unsorted_mjd(self):
        """TOAs need not be pre-sorted."""
        toas_mjd = np.array([50001.0, 50000.0, 50001.1, 50000.1])
        mask = np.ones(4, dtype=bool)
        epochs = _group_toas_into_epochs(toas_mjd, mask, dt_days=0.5)
        assert len(epochs) == 2
        # Epoch 1: TOAs at 50000.0, 50000.1 -> original indices 1, 3
        # Epoch 2: TOAs at 50001.0, 50001.1 -> original indices 0, 2
        epoch_sets = [set(ep) for ep in epochs]
        assert {1, 3} in epoch_sets
        assert {0, 2} in epoch_sets


# ===================================================================
# build_ecorr_whitener
# ===================================================================

class TestBuildEcorrWhitener:
    """Test ECORRWhitener construction."""

    def test_no_ecorr_returns_none(self):
        """No ECORR entries -> returns None."""
        toas_mjd = np.array([50000.0, 50000.1])
        toa_flags = [{"f": "A"}, {"f": "A"}]
        entries = [WhiteNoiseEntry("EFAC", "f", "A", 1.0)]
        result = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert result is None

    def test_basic_construction(self):
        """ECORR entry with grouped TOAs -> whitener built."""
        toas_mjd = np.array([50000.0, 50000.1, 50001.0, 50001.1])
        toa_flags = [{"f": "A"}] * 4
        entries = [WhiteNoiseEntry("ECORR", "f", "A", 1.0)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 2
        assert w.n_toas == 4

    def test_singletons_tracked(self):
        """TOAs not in any multi-TOA epoch are tracked as singletons."""
        toas_mjd = np.array([50000.0, 50000.1, 50005.0])
        toa_flags = [{"f": "A"}, {"f": "A"}, {"f": "A"}]
        entries = [WhiteNoiseEntry("ECORR", "f", "A", 1.0)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 1
        assert 2 in w.singleton_indices  # TOA 2 is a singleton

    def test_multiple_backends(self):
        """Different backends get separate epoch groups."""
        toas_mjd = np.array([50000.0, 50000.1, 50000.0, 50000.1])
        toa_flags = [
            {"f": "A"}, {"f": "A"},
            {"f": "B"}, {"f": "B"},
        ]
        entries = [
            WhiteNoiseEntry("ECORR", "f", "A", 1.0),
            WhiteNoiseEntry("ECORR", "f", "B", 2.0),
        ]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 2
        # Check ECORR values
        ecorr_values = {g.ecorr_us for g in w.epoch_groups}
        assert ecorr_values == {1.0, 2.0}

    def test_non_matching_flag_ignored(self):
        """TOAs that don't match the ECORR flag are not grouped."""
        toas_mjd = np.array([50000.0, 50000.1, 50000.0])
        toa_flags = [{"f": "A"}, {"f": "A"}, {"f": "B"}]
        entries = [WhiteNoiseEntry("ECORR", "f", "A", 1.0)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 1
        assert set(w.epoch_groups[0].indices) == {0, 1}
        assert 2 in w.singleton_indices


# ===================================================================
# ECORRWhitener operations
# ===================================================================

class TestECORRWhitener:
    """Test whitening and chi2 computations."""

    @pytest.fixture
    def simple_whitener(self):
        """4 TOAs: 2 in one epoch (ECORR=1mus), 2 singletons."""
        toas_mjd = np.array([50000.0, 50000.1, 50005.0, 50006.0])
        toa_flags = [{"f": "A"}] * 4
        entries = [WhiteNoiseEntry("ECORR", "f", "A", 1.0)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        return w

    def test_prepare(self, simple_whitener):
        """prepare() should succeed without errors."""
        sigma_sec = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        simple_whitener.prepare(sigma_sec)

    def test_whiten_residuals_shape(self, simple_whitener):
        """Whitened residuals have same shape as input."""
        sigma_sec = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        simple_whitener.prepare(sigma_sec)
        r = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        r_white = simple_whitener.whiten_residuals(r)
        assert r_white.shape == r.shape

    def test_whiten_matrix_shape(self, simple_whitener):
        """Whitened design matrix has same shape as input."""
        sigma_sec = np.array([1e-6, 1e-6, 1e-6, 1e-6])
        simple_whitener.prepare(sigma_sec)
        M = np.random.randn(4, 3)
        M_white = simple_whitener.whiten_matrix(M)
        assert M_white.shape == M.shape

    def test_singleton_whiten_is_diagonal(self, simple_whitener):
        """For singleton TOAs, whitening = 1/sigma scaling."""
        sigma_sec = np.array([1e-6, 1e-6, 2e-6, 3e-6])
        simple_whitener.prepare(sigma_sec)
        r = np.array([0.0, 0.0, 6e-6, 9e-6])
        r_white = simple_whitener.whiten_residuals(r)
        # TOAs 2 and 3 are singletons -> r_white = r / sigma
        np.testing.assert_almost_equal(r_white[2], 6e-6 / 2e-6)
        np.testing.assert_almost_equal(r_white[3], 9e-6 / 3e-6)

    def test_ecorr_block_covariance(self):
        """Verify block covariance C = diag(sigma^2) + J*11^T is used correctly.

        For a 2-TOA epoch with sigma1=sigma2=sigma, ECORR=j:
        C = [[sigma^2+j^2, j^2], [j^2, sigma^2+j^2]]
        C^{-1} can be computed analytically.
        """
        # 2 TOAs in one epoch, equal sigma
        toas_mjd = np.array([50000.0, 50000.1])
        toa_flags = [{"f": "A"}, {"f": "A"}]
        sigma_us = 1.0  # mus
        ecorr_us = 0.5  # mus
        entries = [WhiteNoiseEntry("ECORR", "f", "A", ecorr_us)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None

        sigma_sec = np.array([sigma_us * 1e-6, sigma_us * 1e-6])
        w.prepare(sigma_sec)

        # Residuals
        r = np.array([1e-6, 2e-6])

        # Analytic C^{-1} for 2x2 block:
        # C = [[sigma^2+j^2, j^2], [j^2, sigma^2+j^2]]  (in seconds^2)
        s2 = (sigma_us * 1e-6) ** 2
        j2 = (ecorr_us * 1e-6) ** 2
        det = (s2 + j2) ** 2 - j2 ** 2
        C_inv = np.array([
            [s2 + j2, -j2],
            [-j2, s2 + j2]
        ]) / det

        # Expected chi2 = r^T C^{-1} r
        expected_chi2 = r @ C_inv @ r
        actual_chi2 = w.chi2(r)
        np.testing.assert_almost_equal(actual_chi2, expected_chi2, decimal=10)

    def test_chi2_no_ecorr_matches_diagonal(self):
        """With ECORR=0, chi2 should match diagonal Sigma(r/sigma)^2."""
        toas_mjd = np.array([50000.0, 50000.1, 50001.0])
        toa_flags = [{"f": "A"}] * 3
        # ECORR=0 means no correlation -- but we can test with very small ECORR
        # Actually test with no ECORR at all -> all singletons
        entries = [WhiteNoiseEntry("ECORR", "f", "B", 1.0)]  # doesn't match
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        # No match -> all singletons
        assert w is not None
        assert len(w.epoch_groups) == 0

        sigma_sec = np.array([1e-6, 2e-6, 3e-6])
        w.prepare(sigma_sec)
        r = np.array([1e-6, 2e-6, 3e-6])
        chi2 = w.chi2(r)
        expected = np.sum((r / sigma_sec) ** 2)
        np.testing.assert_almost_equal(chi2, expected)

    def test_chi2_larger_block(self):
        """3-TOA epoch with known covariance."""
        toas_mjd = np.array([50000.0, 50000.1, 50000.2])
        toa_flags = [{"f": "A"}] * 3
        ecorr_us = 0.8
        entries = [WhiteNoiseEntry("ECORR", "f", "A", ecorr_us)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 1

        sigma_sec = np.array([1e-6, 1.5e-6, 2e-6])
        w.prepare(sigma_sec)
        r = np.array([1e-6, -0.5e-6, 0.3e-6])

        # Build the full covariance matrix analytically
        j2 = (ecorr_us * 1e-6) ** 2
        C = np.diag(sigma_sec ** 2) + j2 * np.ones((3, 3))
        C_inv = np.linalg.inv(C)
        expected_chi2 = r @ C_inv @ r

        actual_chi2 = w.chi2(r)
        np.testing.assert_almost_equal(actual_chi2, expected_chi2, decimal=10)

    def test_whiten_produces_unit_covariance(self):
        """After whitening, the effective covariance should be identity.

        If we whiten M, then M_white^T M_white should equal M^T C^{-1} M.
        """
        toas_mjd = np.array([50000.0, 50000.1])
        toa_flags = [{"f": "A"}] * 2
        ecorr_us = 0.5
        entries = [WhiteNoiseEntry("ECORR", "f", "A", ecorr_us)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)

        sigma_sec = np.array([1e-6, 1.5e-6])
        w.prepare(sigma_sec)

        M = np.array([[1.0, 0.5], [0.3, 0.8]])
        M_white = w.whiten_matrix(M)

        # Expected: M^T C^{-1} M
        j2 = (ecorr_us * 1e-6) ** 2
        C = np.diag(sigma_sec ** 2) + j2 * np.ones((2, 2))
        C_inv = np.linalg.inv(C)
        expected = M.T @ C_inv @ M
        actual = M_white.T @ M_white

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_mixed_epochs_and_singletons(self):
        """Mix of grouped and singleton TOAs."""
        toas_mjd = np.array([50000.0, 50000.1, 50005.0, 50010.0, 50010.1])
        toa_flags = [{"f": "A"}] * 5
        ecorr_us = 1.0
        entries = [WhiteNoiseEntry("ECORR", "f", "A", ecorr_us)]
        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 2  # epochs at 50000 and 50010
        assert 2 in w.singleton_indices  # TOA at 50005 is singleton

        sigma_sec = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        w.prepare(sigma_sec)
        r = np.ones(5) * 1e-6
        chi2 = w.chi2(r)

        # Build full covariance for verification
        j2 = (ecorr_us * 1e-6) ** 2
        C = np.diag(sigma_sec ** 2)
        # Epoch 1: TOAs 0, 1
        for i in [0, 1]:
            for j in [0, 1]:
                C[i, j] += j2
        # Epoch 2: TOAs 3, 4
        for i in [3, 4]:
            for j in [3, 4]:
                C[i, j] += j2
        # TOA 2 is singleton -- no off-diagonal
        C_inv = np.linalg.inv(C)
        expected_chi2 = r @ C_inv @ r
        np.testing.assert_almost_equal(chi2, expected_chi2, decimal=10)


# ===================================================================
# Integration with parse_noise_lines
# ===================================================================

class TestECORRParsingIntegration:
    """Test that parsed ECORR entries flow correctly into whitener."""

    def test_ecorr_from_par_lines(self):
        """ECORR lines parsed and used to build whitener."""
        lines = [
            "T2EFAC -f PUPPI 1.1",
            "T2EQUAD -f PUPPI 0.3",
            "ECORR -f PUPPI 2.5",
        ]
        entries = parse_noise_lines(lines)
        assert len(entries) == 3

        toas_mjd = np.array([50000.0, 50000.1, 50000.2, 50005.0])
        toa_flags = [{"f": "PUPPI"}] * 4

        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 1
        assert w.epoch_groups[0].ecorr_us == 2.5
        assert set(w.epoch_groups[0].indices) == {0, 1, 2}

    def test_ecorr_multiple_backends_from_par(self):
        """Multiple ECORR lines for different backends."""
        lines = [
            "ECORR -f L-wide_PUPPI 1.5",
            "ECORR -f 430_ASP 3.0",
        ]
        entries = parse_noise_lines(lines)

        toas_mjd = np.array([50000.0, 50000.1, 50000.0, 50000.1])
        toa_flags = [
            {"f": "L-wide_PUPPI"}, {"f": "L-wide_PUPPI"},
            {"f": "430_ASP"}, {"f": "430_ASP"},
        ]

        w = build_ecorr_whitener(toas_mjd, toa_flags, entries)
        assert w is not None
        assert len(w.epoch_groups) == 2

        for g in w.epoch_groups:
            if g.flag_value == "L-wide_PUPPI":
                assert g.ecorr_us == 1.5
                assert set(g.indices) == {0, 1}
            elif g.flag_value == "430_ASP":
                assert g.ecorr_us == 3.0
                assert set(g.indices) == {2, 3}

    def test_no_ecorr_in_entries(self):
        """Only EFAC/EQUAD -> build_ecorr_whitener returns None."""
        lines = [
            "T2EFAC -f PUPPI 1.0",
            "T2EQUAD -f PUPPI 0.5",
        ]
        entries = parse_noise_lines(lines)
        toas_mjd = np.array([50000.0, 50000.1])
        toa_flags = [{"f": "PUPPI"}] * 2
        assert build_ecorr_whitener(toas_mjd, toa_flags, entries) is None


# ===================================================================
# EpochGroup dataclass
# ===================================================================

class TestEpochGroup:
    """Test EpochGroup dataclass."""

    def test_frozen(self):
        """EpochGroup should be immutable."""
        g = EpochGroup(indices=(0, 1), ecorr_us=1.0, flag_value="A")
        with pytest.raises(AttributeError):
            g.ecorr_us = 2.0  # type: ignore

    def test_fields(self):
        g = EpochGroup(indices=(0, 1, 2), ecorr_us=2.5, flag_value="PUPPI")
        assert g.indices == (0, 1, 2)
        assert g.ecorr_us == 2.5
        assert g.flag_value == "PUPPI"
