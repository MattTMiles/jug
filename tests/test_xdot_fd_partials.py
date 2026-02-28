"""Finite-difference validation tests for XDOT, FD, and H3/STIG partial derivatives.

Each test computes the analytic partial and a central-difference numerical
derivative from the forward model, then checks that they are well-correlated
(r > 0.95).
"""

import numpy as np
import pytest
import jax.numpy as jnp


# =============================================================================
# XDOT partial derivative tests
# =============================================================================

class TestXDOTPartialDerivative:
    """XDOT = dA1/dt. Chain rule: d(delay)/d(XDOT) = d(delay)/d(A1) * dt_sec."""

    @pytest.fixture
    def dd_params(self):
        return {
            'BINARY': 'DD',
            'PB': 5.7410459,
            'A1': 3.3667144,
            'T0': 55000.0,
            'ECC': 0.01,
            'OM': 45.0,
            'XDOT': 1e-14,
            'PBDOT': 0.0,
            'OMDOT': 0.0,
            'GAMMA': 0.0,
            'SINI': 0.8,
            'M2': 0.3,
            'EDOT': 0.0,
        }

    def test_xdot_derivative_nonzero(self, dd_params):
        """XDOT derivative should be finite and have orbital modulation."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        toas = jnp.linspace(55000.0, 57000.0, 100)
        result = compute_binary_derivatives_dd(dd_params, toas, ['XDOT'])
        deriv = np.asarray(result['XDOT'])
        assert np.all(np.isfinite(deriv)), "XDOT derivative has non-finite values"
        assert np.std(deriv) > 0, "XDOT derivative is constant"

    def test_xdot_finite_difference(self, dd_params):
        """XDOT analytic partial should match central-difference numerical derivative."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)
        result = compute_binary_derivatives_dd(dd_params, jnp.asarray(toas), ['XDOT'])
        analytic = np.asarray(result['XDOT'])

        h = 1e-18  # XDOT is ~1e-14, use tiny step
        params_plus = dd_params.copy()
        params_minus = dd_params.copy()
        params_plus['XDOT'] = dd_params['XDOT'] + h
        params_minus['XDOT'] = dd_params['XDOT'] - h

        delay_plus = np.asarray(compute_dd_binary_delay(toas, params_plus))
        delay_minus = np.asarray(compute_dd_binary_delay(toas, params_minus))
        numeric = (delay_plus - delay_minus) / (2 * h)

        assert np.all(np.isfinite(analytic))
        assert np.all(np.isfinite(numeric))
        scale = np.max(np.abs(numeric))
        if scale > 1e-20:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, f"XDOT analytic/numeric poorly correlated: r={corr:.4f}"

    def test_xdot_grows_with_time_baseline(self, dd_params):
        """XDOT derivative should grow with time baseline (proportional to dt_sec)."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd

        toas_short = jnp.linspace(55000.0, 55100.0, 50)  # 100 days
        toas_long = jnp.linspace(55000.0, 57000.0, 50)    # 2000 days

        d_short = np.asarray(compute_binary_derivatives_dd(dd_params, toas_short, ['XDOT'])['XDOT'])
        d_long = np.asarray(compute_binary_derivatives_dd(dd_params, toas_long, ['XDOT'])['XDOT'])

        # RMS should be much larger for longer baseline
        assert np.std(d_long) > 5 * np.std(d_short), (
            f"XDOT derivative does not grow with baseline: "
            f"short={np.std(d_short):.3e}, long={np.std(d_long):.3e}"
        )


# =============================================================================
# FD partial derivative tests
# =============================================================================

class TestFDPartialDerivatives:
    """FD parameters: delay = FD1*log(f/1GHz) + FD2*log(f/1GHz)^2 + ..."""

    @pytest.fixture
    def freq_mhz(self):
        """Multi-frequency TOA set spanning typical L-band range."""
        np.random.seed(42)
        return np.random.uniform(800, 2000, 50)

    @pytest.fixture
    def fd_params(self):
        return {'FD1': 1e-5, 'FD2': -2e-6, 'FD3': 5e-7}

    def test_fd1_derivative_matches_log_freq(self, freq_mhz, fd_params):
        """d(delay)/d(FD1) should equal log(freq/1GHz)."""
        from jug.fitting.derivatives_fd import compute_fd_derivatives
        result = compute_fd_derivatives(fd_params, freq_mhz, ['FD1'])
        analytic = result['FD1']
        expected = np.log(freq_mhz / 1000.0)
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_fd2_derivative_matches_log_freq_squared(self, freq_mhz, fd_params):
        """d(delay)/d(FD2) should equal log(freq/1GHz)^2."""
        from jug.fitting.derivatives_fd import compute_fd_derivatives
        result = compute_fd_derivatives(fd_params, freq_mhz, ['FD2'])
        analytic = result['FD2']
        expected = np.log(freq_mhz / 1000.0) ** 2
        np.testing.assert_allclose(analytic, expected, rtol=1e-12)

    def test_fd_finite_difference_fd1(self, freq_mhz, fd_params):
        """FD1 analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_fd import compute_fd_derivatives, compute_fd_delay

        analytic = compute_fd_derivatives(fd_params, freq_mhz, ['FD1'])['FD1']

        h = 1e-10
        params_plus = fd_params.copy()
        params_minus = fd_params.copy()
        params_plus['FD1'] = fd_params['FD1'] + h
        params_minus['FD1'] = fd_params['FD1'] - h

        delay_plus = compute_fd_delay(freq_mhz, params_plus)
        delay_minus = compute_fd_delay(freq_mhz, params_minus)
        numeric = (delay_plus - delay_minus) / (2 * h)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-6)

    def test_fd_finite_difference_fd2(self, freq_mhz, fd_params):
        """FD2 analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_fd import compute_fd_derivatives, compute_fd_delay

        analytic = compute_fd_derivatives(fd_params, freq_mhz, ['FD2'])['FD2']

        h = 1e-10
        params_plus = fd_params.copy()
        params_minus = fd_params.copy()
        params_plus['FD2'] = fd_params['FD2'] + h
        params_minus['FD2'] = fd_params['FD2'] - h

        delay_plus = compute_fd_delay(freq_mhz, params_plus)
        delay_minus = compute_fd_delay(freq_mhz, params_minus)
        numeric = (delay_plus - delay_minus) / (2 * h)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-6)

    def test_fd_finite_difference_fd3(self, freq_mhz, fd_params):
        """FD3 analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_fd import compute_fd_derivatives, compute_fd_delay

        analytic = compute_fd_derivatives(fd_params, freq_mhz, ['FD3'])['FD3']

        h = 1e-10
        params_plus = fd_params.copy()
        params_minus = fd_params.copy()
        params_plus['FD3'] = fd_params['FD3'] + h
        params_minus['FD3'] = fd_params['FD3'] - h

        delay_plus = compute_fd_delay(freq_mhz, params_plus)
        delay_minus = compute_fd_delay(freq_mhz, params_minus)
        numeric = (delay_plus - delay_minus) / (2 * h)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-6)


# =============================================================================
# H3/STIG finite-difference validation
# =============================================================================

class TestH3STIGPartialDerivatives:
    """H3/STIG orthometric Shapiro delay derivatives (DDH model)."""

    @pytest.fixture
    def ddh_params(self):
        """DDH parameters with H3/STIG parameterization."""
        T_SUN = 4.925490947e-6
        m2 = 0.3
        stig = 0.5  # sini = 2*0.5/(1+0.25) = 0.8
        h3 = m2 * stig**3 * T_SUN  # ≈ 1.847e-7
        return {
            'BINARY': 'DD',
            'PB': 5.7410459,
            'A1': 3.3667144,
            'T0': 55000.0,
            'ECC': 0.01,
            'OM': 45.0,
            'H3': h3,
            'STIG': stig,
            'PBDOT': 0.0,
            'OMDOT': 0.0,
            'GAMMA': 0.0,
            'SINI': 0.0,
            'M2': 0.0,
        }

    def test_h3_stig_finite_difference(self, ddh_params):
        """H3 (STIG parameterization) analytic partial should match finite difference."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)
        result = compute_binary_derivatives_dd(ddh_params, jnp.asarray(toas), ['H3'])
        analytic = np.asarray(result['H3'])

        h3_val = ddh_params['H3']
        h = h3_val * 1e-5
        params_plus = ddh_params.copy()
        params_minus = ddh_params.copy()
        params_plus['H3'] = h3_val + h
        params_minus['H3'] = h3_val - h

        delay_plus = np.asarray(compute_dd_binary_delay(toas, params_plus))
        delay_minus = np.asarray(compute_dd_binary_delay(toas, params_minus))
        numeric = (delay_plus - delay_minus) / (2 * h)

        assert np.all(np.isfinite(analytic))
        assert np.all(np.isfinite(numeric))
        scale = np.max(np.abs(numeric))
        if scale > 1e-20:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, f"H3(STIG) analytic/numeric poorly correlated: r={corr:.4f}"

    def test_stig_finite_difference(self, ddh_params):
        """STIG analytic partial should match finite difference."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)
        result = compute_binary_derivatives_dd(ddh_params, jnp.asarray(toas), ['STIG'])
        analytic = np.asarray(result['STIG'])

        stig_val = ddh_params['STIG']
        h = stig_val * 1e-5
        params_plus = ddh_params.copy()
        params_minus = ddh_params.copy()
        params_plus['STIG'] = stig_val + h
        params_minus['STIG'] = stig_val - h

        delay_plus = np.asarray(compute_dd_binary_delay(toas, params_plus))
        delay_minus = np.asarray(compute_dd_binary_delay(toas, params_minus))
        numeric = (delay_plus - delay_minus) / (2 * h)

        assert np.all(np.isfinite(analytic))
        assert np.all(np.isfinite(numeric))
        scale = np.max(np.abs(numeric))
        if scale > 1e-20:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, f"STIG analytic/numeric poorly correlated: r={corr:.4f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
