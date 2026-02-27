"""Tests for H3/H4 edge cases, STIG/H4 priority warnings, and validate_fit_param.

Validates:
1. H3/H4 with H4=0 triggers warning and produces no NaN/Inf
2. Both STIG and H4 nonzero triggers priority warning, result matches STIG-only
3. validate_fit_param accepts registered params and rejects unregistered ones
"""

import numpy as np
import jax.numpy as jnp
import pytest
import warnings


class TestH3H4EdgeCases:
    """H4=0 with H3!=0 should warn and produce finite zero derivatives."""

    @pytest.fixture
    def toas(self):
        return jnp.linspace(58000.0, 58100.0, 50)

    @pytest.fixture
    def base_params(self):
        return {
            'A1': 1.0,
            'PB': 1.5,
            'T0': 58050.0,
            'ECC': 0.001,
            'OM': 90.0,
            'H3': 1e-6,
            'H4': 0.0,
            'STIG': 0.0,
        }

    def test_h3_nonzero_h4_zero_warns_dd(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        with pytest.warns(UserWarning, match="H4=0.*ill-conditioned"):
            derivs = compute_binary_derivatives_dd(base_params, toas, ['H3'])
        assert 'H3' in derivs
        assert np.all(np.isfinite(derivs['H3']))
        # Should be zero since H4=0 means no valid parameterization
        np.testing.assert_array_equal(np.array(derivs['H3']), 0.0)

    def test_h3_nonzero_h4_zero_warns_ddk(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_binary_derivatives_ddk
        base_params['KIN'] = 60.0
        base_params['KOM'] = 0.0
        base_params['RAJ'] = 1.0
        base_params['DECJ'] = -0.5
        with pytest.warns(UserWarning, match="H4=0.*ill-conditioned"):
            derivs = compute_binary_derivatives_ddk(base_params, toas, ['H3'])
        assert 'H3' in derivs
        assert np.all(np.isfinite(derivs['H3']))

    def test_h3_zero_h4_zero_no_warning(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        base_params['H3'] = 0.0
        base_params['H4'] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            derivs = compute_binary_derivatives_dd(base_params, toas, ['H3'])
        np.testing.assert_array_equal(np.array(derivs['H3']), 0.0)

    def test_h3_nonzero_h4_zero_forward_model_warns(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_dd_binary_delay
        base_params['SINI'] = 0.0
        base_params['M2'] = 0.0
        with pytest.warns(UserWarning, match="H4=0.*ill-conditioned"):
            delay = compute_dd_binary_delay(toas, base_params)
        assert np.all(np.isfinite(delay))


class TestH3H4Priority:
    """When both STIG and H4 are nonzero, STIG takes priority with a warning."""

    @pytest.fixture
    def toas(self):
        return jnp.linspace(58000.0, 58100.0, 50)

    @pytest.fixture
    def base_params(self):
        return {
            'A1': 1.0,
            'PB': 1.5,
            'T0': 58050.0,
            'ECC': 0.001,
            'OM': 90.0,
            'GAMMA': 0.0,
            'PBDOT': 0.0,
            'OMDOT': 0.0,
            'H3': 1e-6,
            'STIG': 0.5,
            'H4': 1e-7,
        }

    def test_stig_h4_both_warns_dd(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        with pytest.warns(UserWarning, match="Both STIG and H4.*H4 ignored"):
            derivs = compute_binary_derivatives_dd(base_params, toas, ['H3'])

    def test_stig_h4_result_matches_stig_only(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        # With both STIG and H4
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            derivs_both = compute_binary_derivatives_dd(base_params, toas, ['H3'])

        # With STIG only (H4=0)
        params_stig = dict(base_params)
        params_stig['H4'] = 0.0
        derivs_stig = compute_binary_derivatives_dd(params_stig, toas, ['H3'])

        np.testing.assert_allclose(
            np.array(derivs_both['H3']),
            np.array(derivs_stig['H3']),
            rtol=1e-12
        )

    def test_stig_h4_forward_model_warns(self, toas, base_params):
        from jug.fitting.derivatives_dd import compute_dd_binary_delay
        base_params['SINI'] = 0.0
        base_params['M2'] = 0.0
        with pytest.warns(UserWarning, match="Both STIG and H4.*H4 ignored"):
            delay = compute_dd_binary_delay(toas, base_params)
        assert np.all(np.isfinite(delay))


class TestValidateFitParam:
    """Test validate_fit_param accepts valid params and rejects invalid ones."""

    def test_spin_params_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('F0') is True
        assert validate_fit_param('F1') is True
        assert validate_fit_param('F2') is True

    def test_ne_sw_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('NE_SW') is True

    def test_ne1au_alias_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('NE1AU') is True

    def test_binary_params_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('PB') is True
        assert validate_fit_param('A1') is True
        assert validate_fit_param('H3') is True
        assert validate_fit_param('KIN') is True

    def test_fd_in_range_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('FD1') is True
        assert validate_fit_param('FD9') is True

    def test_fd_out_of_range_raises(self):
        from jug.model.parameter_spec import validate_fit_param
        with pytest.raises(ValueError, match="not yet implemented"):
            validate_fit_param('FD10')
        with pytest.raises(ValueError, match="not yet implemented"):
            validate_fit_param('FD15')

    def test_fb_in_range_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('FB0') is True
        assert validate_fit_param('FB20') is True

    def test_fb_out_of_range_raises(self):
        from jug.model.parameter_spec import validate_fit_param
        with pytest.raises(ValueError, match="not yet implemented"):
            validate_fit_param('FB21')

    def test_jump_valid(self):
        from jug.model.parameter_spec import validate_fit_param
        assert validate_fit_param('JUMP1') is True
        assert validate_fit_param('JUMP5') is True
        assert validate_fit_param('JUMP_MJD_58000') is True

    def test_dmx_unknown_raises(self):
        from jug.model.parameter_spec import validate_fit_param
        with pytest.raises(ValueError, match="not registered"):
            validate_fit_param('DMX_0001')

    def test_unknown_param_raises(self):
        from jug.model.parameter_spec import validate_fit_param
        with pytest.raises(ValueError, match="not registered"):
            validate_fit_param('TOTALLY_BOGUS')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
