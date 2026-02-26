"""Validate JUG ECORR block-Cholesky whitening against PINT.

Uses PINT's ``ecorr_fit_test`` dataset (4005 TOAs, 1 ECORR + 1 EFAC,
single backend, 235 epochs) to verify that JUG's ECORRWhitener produces
identical chi2 and whitened-residual norms.

The test loads the data entirely through PINT, extracts the epoch
structure (U matrix) and EFAC-scaled uncertainties, then rebuilds the
same problem using JUG's ECORRWhitener and compares results.

This avoids needing JUG to parse PINT-format tim files -- we only
compare the ECORR *math*, not the full pipeline.
"""

import numpy as np
import pytest

pint = pytest.importorskip("pint")
pint_models = pytest.importorskip("pint.models")
pint_toa = pytest.importorskip("pint.toa")
pint_residuals = pytest.importorskip("pint.residuals")

from jug.noise.ecorr import ECORRWhitener, EpochGroup

# ---------------------------------------------------------------------------
# Paths to PINT test data
# ---------------------------------------------------------------------------
PINT_DATADIR = "/home/mattm/soft/PINT/tests/datafile"
ECORR_PAR = f"{PINT_DATADIR}/ecorr_fit_test.par"
ECORR_TIM = f"{PINT_DATADIR}/ecorr_fit_test.tim"


def _load_pint_ecorr_problem():
    """Load the PINT ecorr_fit_test dataset and extract all quantities.

    Returns
    -------
    r_sec : ndarray, shape (ntoa,)
        Prefit residuals in seconds.
    scaled_sigma : ndarray, shape (ntoa,)
        EFAC-scaled TOA uncertainties in seconds.
    epoch_groups : list of EpochGroup
        Epoch groups derived from PINT's quantization matrix U.
    singletons : ndarray
        Indices of TOAs not in any multi-TOA epoch.
    ntoa : int
        Number of TOAs.
    """
    m = pint_models.get_model(ECORR_PAR)
    t = pint_toa.get_TOAs(ECORR_TIM)

    # EFAC-scaled errors
    scaled_sigma = m.scaled_toa_uncertainty(t).to("s").value

    # ECORR basis matrix and weights
    ecorr_comp = m.components["EcorrNoise"]
    U = ecorr_comp.get_noise_basis(t)
    weights = ecorr_comp.get_noise_weights(t)
    ecorr_us = np.sqrt(weights[0]) * 1e6  # single ECORR value

    # Prefit residuals
    rs = pint_residuals.Residuals(t, m)
    r_sec = rs.time_resids.to("s").value

    # Build JUG-style epoch groups from PINT's U matrix
    epoch_groups = []
    all_grouped = set()
    for k in range(U.shape[1]):
        indices = tuple(np.where(U[:, k] > 0)[0])
        epoch_groups.append(
            EpochGroup(indices=indices, ecorr_us=ecorr_us, flag_value="arecibo")
        )
        all_grouped.update(indices)

    singletons = np.array(
        sorted(set(range(t.ntoas)) - all_grouped), dtype=np.intp
    )

    return r_sec, scaled_sigma, epoch_groups, singletons, t.ntoas


# Cache the loaded data across tests in this module
@pytest.fixture(scope="module")
def pint_ecorr_data():
    return _load_pint_ecorr_problem()


# ===================================================================
# Tests
# ===================================================================


class TestECORRvsPINT:
    """Validate JUG ECORR whitening against PINT on 4005-TOA dataset."""

    def test_dataset_sanity(self, pint_ecorr_data):
        """Verify PINT dataset loaded correctly."""
        r_sec, sigma, groups, singletons, ntoa = pint_ecorr_data
        assert ntoa == 4005
        assert len(groups) == 235
        assert len(singletons) == 2  # 2 singleton TOAs

    def test_chi2_matches_pint_full_matrix(self, pint_ecorr_data):
        """JUG block-Cholesky chi2 must match PINT full-matrix chi2.

        Computes chi2 = r^T C^{-1} r two ways:
        1. PINT: full 4005*4005 Cholesky factorization
        2. JUG: block-by-block Cholesky via ECORRWhitener
        """
        r_sec, sigma, groups, singletons, ntoa = pint_ecorr_data

        # --- PINT reference: full matrix ---
        from scipy.linalg import solve_triangular as st

        ecorr_sec = groups[0].ecorr_us * 1e-6
        J = ecorr_sec**2
        # Build full covariance C = diag(sigma^2) + U diag(J) U^T
        C = np.diag(sigma**2)
        for g in groups:
            idx = list(g.indices)
            for i in idx:
                for j in idx:
                    C[i, j] += J
        L_full = np.linalg.cholesky(C)
        r_white_full = st(L_full, r_sec, lower=True)
        chi2_pint = float(np.dot(r_white_full, r_white_full))

        # --- JUG: block Cholesky ---
        w = ECORRWhitener(
            epoch_groups=groups,
            singleton_indices=singletons,
            n_toas=ntoa,
        )
        w.prepare(sigma)
        chi2_jug = w.chi2(r_sec)

        # Must match to machine precision
        np.testing.assert_allclose(chi2_jug, chi2_pint, rtol=1e-12,
                                   err_msg="JUG ECORR chi2 disagrees with PINT")

    def test_whitened_residual_norm_matches(self, pint_ecorr_data):
        """||L^{-1} r||_JUG == ||L^{-1} r||_PINT."""
        r_sec, sigma, groups, singletons, ntoa = pint_ecorr_data

        # PINT reference
        from scipy.linalg import solve_triangular as st

        ecorr_sec = groups[0].ecorr_us * 1e-6
        J = ecorr_sec**2
        C = np.diag(sigma**2)
        for g in groups:
            idx = list(g.indices)
            for i in idx:
                for j in idx:
                    C[i, j] += J
        L_full = np.linalg.cholesky(C)
        r_white_pint = st(L_full, r_sec, lower=True)

        # JUG
        w = ECORRWhitener(
            epoch_groups=groups,
            singleton_indices=singletons,
            n_toas=ntoa,
        )
        w.prepare(sigma)
        r_white_jug = w.whiten_residuals(r_sec)

        # Norms must match (chi2 = norm^2)
        np.testing.assert_allclose(
            np.linalg.norm(r_white_jug),
            np.linalg.norm(r_white_pint),
            rtol=1e-12,
            err_msg="Whitened residual norms disagree",
        )

    def test_whitened_design_matrix_property(self, pint_ecorr_data):
        """M_w^T M_w == M^T C^{-1} M for a random design matrix.

        This validates that the block-Cholesky whitening correctly
        transforms the design matrix for GLS -> OLS equivalence.
        """
        r_sec, sigma, groups, singletons, ntoa = pint_ecorr_data

        # Small random design matrix (3 columns)
        rng = np.random.default_rng(42)
        M = rng.standard_normal((ntoa, 3))

        # Build C^{-1} via full Cholesky (reference)
        ecorr_sec = groups[0].ecorr_us * 1e-6
        J = ecorr_sec**2
        C = np.diag(sigma**2)
        for g in groups:
            idx = list(g.indices)
            for i in idx:
                for j in idx:
                    C[i, j] += J
        C_inv = np.linalg.inv(C)
        expected = M.T @ C_inv @ M

        # JUG whitening
        w = ECORRWhitener(
            epoch_groups=groups,
            singleton_indices=singletons,
            n_toas=ntoa,
        )
        w.prepare(sigma)
        M_white = w.whiten_matrix(M)
        actual = M_white.T @ M_white

        np.testing.assert_allclose(actual, expected, rtol=1e-8,
                                   err_msg="Whitened design matrix property failed")

    def test_epoch_group_structure(self, pint_ecorr_data):
        """Verify epoch structure: sizes, total coverage."""
        _, _, groups, singletons, ntoa = pint_ecorr_data

        # All TOA indices should be accounted for
        all_indices = set()
        for g in groups:
            all_indices.update(g.indices)
        all_indices.update(singletons.tolist())
        assert len(all_indices) == ntoa
        assert all_indices == set(range(ntoa))

        # Every epoch has >= 2 TOAs
        for g in groups:
            assert len(g.indices) >= 2

        # Single ECORR value across all epochs
        ecorr_values = {g.ecorr_us for g in groups}
        assert len(ecorr_values) == 1
