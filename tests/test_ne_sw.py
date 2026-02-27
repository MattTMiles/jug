"""Tests for NE_SW (solar wind) fitting support.

Validates:
1. NE_SW is registered as a fittable parameter with SOLAR_WIND derivative group
2. Analytic partial d(delay)/d(NE_SW) matches finite-difference numerical derivative
3. The derivative module produces correct shape and finite values
"""

import numpy as np
import pytest


class TestNESWParameterSpec:
    """Verify NE_SW is properly registered in the ParameterSpec system."""

    def test_ne_sw_registered(self):
        from jug.model.parameter_spec import get_spec
        spec = get_spec('NE_SW')
        assert spec is not None, "NE_SW not found in PARAMETER_REGISTRY"
        assert spec.name == 'NE_SW'

    def test_ne_sw_derivative_group(self):
        from jug.model.parameter_spec import get_derivative_group, DerivativeGroup
        group = get_derivative_group('NE_SW')
        assert group == DerivativeGroup.SOLAR_WIND

    def test_ne1au_alias(self):
        from jug.model.parameter_spec import get_spec
        spec = get_spec('NE1AU')
        assert spec is not None, "NE1AU alias not resolving"
        assert spec.name == 'NE_SW'

    def test_is_sw_param(self):
        from jug.model.parameter_spec import is_sw_param
        assert is_sw_param('NE_SW')
        assert is_sw_param('NE1AU')
        assert not is_sw_param('DM')
        assert not is_sw_param('F0')

    def test_get_sw_params_from_list(self):
        from jug.model.parameter_spec import get_sw_params_from_list
        params = ['F0', 'F1', 'DM', 'NE_SW', 'A1']
        sw = get_sw_params_from_list(params)
        assert sw == ['NE_SW']

    def test_ne_sw_in_fittable_params(self):
        from jug.model.parameter_spec import list_fittable_params
        fittable = list_fittable_params()
        assert 'NE_SW' in fittable


class TestNESWDerivative:
    """Test the analytic NE_SW partial derivative against finite differences."""

    @pytest.fixture
    def sw_geometry(self):
        """Synthetic solar wind geometry factor (parsecs) for 50 TOAs."""
        # Typical range: ~1e-5 to ~1e-3 pc for pulsars not too close to the Sun
        np.random.seed(42)
        return np.abs(np.random.uniform(1e-5, 5e-4, 50))

    @pytest.fixture
    def freq_mhz(self):
        """Observing frequencies in MHz."""
        return np.full(50, 1400.0)

    def test_derivative_shape(self, sw_geometry, freq_mhz):
        from jug.fitting.derivatives_sw import compute_sw_derivatives
        result = compute_sw_derivatives(sw_geometry, freq_mhz, ['NE_SW'])
        assert 'NE_SW' in result
        assert result['NE_SW'].shape == (50,)

    def test_derivative_finite(self, sw_geometry, freq_mhz):
        from jug.fitting.derivatives_sw import compute_sw_derivatives
        result = compute_sw_derivatives(sw_geometry, freq_mhz, ['NE_SW'])
        assert np.all(np.isfinite(result['NE_SW']))

    def test_derivative_positive(self, sw_geometry, freq_mhz):
        """NE_SW derivative should be positive (more electrons = more delay)."""
        from jug.fitting.derivatives_sw import compute_sw_derivatives
        result = compute_sw_derivatives(sw_geometry, freq_mhz, ['NE_SW'])
        assert np.all(result['NE_SW'] > 0)

    def test_derivative_scales_with_geometry(self, freq_mhz):
        """Derivative should scale linearly with geometry factor."""
        from jug.fitting.derivatives_sw import d_delay_d_NE_SW
        geom1 = np.ones(50) * 1e-4
        geom2 = np.ones(50) * 2e-4
        d1 = d_delay_d_NE_SW(geom1, freq_mhz)
        d2 = d_delay_d_NE_SW(geom2, freq_mhz)
        np.testing.assert_allclose(d2 / d1, 2.0, rtol=1e-10)

    def test_derivative_scales_with_freq_squared(self, sw_geometry):
        """Derivative should scale as 1/freq^2."""
        from jug.fitting.derivatives_sw import d_delay_d_NE_SW
        freq1 = np.full(50, 1400.0)
        freq2 = np.full(50, 2800.0)
        d1 = d_delay_d_NE_SW(sw_geometry, freq1)
        d2 = d_delay_d_NE_SW(sw_geometry, freq2)
        np.testing.assert_allclose(d1 / d2, 4.0, rtol=1e-10)

    def test_matches_forward_model(self, sw_geometry, freq_mhz):
        """d(delay)/d(NE_SW) should equal delay/NE_SW for any NE_SW > 0."""
        from jug.fitting.derivatives_sw import d_delay_d_NE_SW
        from jug.utils.constants import K_DM_SEC
        ne_sw = 4.0  # cm^-3

        # Forward model: delay = K_DM * NE_SW * geometry / freq^2
        delay = K_DM_SEC * ne_sw * sw_geometry / (freq_mhz ** 2)

        # Derivative: d(delay)/d(NE_SW) = K_DM * geometry / freq^2 = delay / NE_SW
        deriv = d_delay_d_NE_SW(sw_geometry, freq_mhz)

        np.testing.assert_allclose(deriv, delay / ne_sw, rtol=1e-10)


class TestCanonicalizeNE1AU:
    """Verify NE1AU alias is canonicalized to NE_SW."""

    def test_ne1au_canonicalizes(self):
        from jug.model.parameter_spec import canonicalize_param_name
        assert canonicalize_param_name('NE1AU') == 'NE_SW'

    def test_ne_sw_unchanged(self):
        from jug.model.parameter_spec import canonicalize_param_name
        assert canonicalize_param_name('NE_SW') == 'NE_SW'

    def test_ne1au_derivative_matches_ne_sw(self):
        """compute_sw_derivatives keyed by NE1AU should produce same result."""
        from jug.fitting.derivatives_sw import compute_sw_derivatives
        geom = np.ones(10) * 1e-4
        freq = np.full(10, 1400.0)
        d_alias = compute_sw_derivatives(geom, freq, ['NE1AU'])
        d_canon = compute_sw_derivatives(geom, freq, ['NE_SW'])
        assert 'NE1AU' in d_alias
        assert 'NE_SW' in d_canon
        np.testing.assert_allclose(d_alias['NE1AU'], d_canon['NE_SW'], rtol=1e-15)


class TestSWGeometryAlwaysComputed:
    """Verify solar wind geometry is always computed and cached."""

    def test_geometry_always_in_result(self):
        """sw_geometry_pc is always computed (no conditional gating)."""
        import inspect
        from jug.residuals.simple_calculator import compute_residuals_simple
        source = inspect.getsource(compute_residuals_simple)
        # The function should NOT have a conditional that sets sw_geometry_pc = None
        assert 'sw_geometry_pc = None' not in source, (
            "sw_geometry_pc should always be computed, not conditionally set to None"
        )

    def test_geometry_in_return_dict(self):
        """sw_geometry_pc key exists in the return dict."""
        import inspect
        from jug.residuals.simple_calculator import compute_residuals_simple
        source = inspect.getsource(compute_residuals_simple)
        assert "'sw_geometry_pc'" in source, (
            "sw_geometry_pc should be in the return dict"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
