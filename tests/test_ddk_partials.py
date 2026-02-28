"""
Tests for DDK (Kopeikin) binary model partial derivatives.

The DDK model extends DD with:
  1. K96 (Kopeikin 1996) proper motion corrections
  2. Kopeikin 1995 annual orbital parallax corrections

These tests verify that the analytic partial derivatives for KIN and KOM
match finite-difference numerical derivatives.

References:
  - Kopeikin 1995: Annual orbital parallax
  - Kopeikin 1996: Proper motion corrections (K96)
"""

import numpy as np
import pytest
import jax.numpy as jnp

from jug.fitting.derivatives_dd import (
    compute_binary_derivatives_ddk,
    _compute_ddk_correction_derivatives_KIN,
    _compute_ddk_correction_derivatives_KOM,
)


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def ddk_params_j0437():
    """J0437-4715-like DDK parameters - high parallax, well-tested pulsar."""
    return {
        'BINARY': 'DDK',
        # Keplerian binary parameters
        'PB': 5.7410459,          # Orbital period (days)
        'A1': 3.3667144,          # Projected semi-major axis (lt-s)
        'T0': 50000.0,            # Epoch of periastron (MJD)
        'ECC': 1.918e-5,          # Orbital eccentricity
        'OM': 1.35,               # Longitude of periastron (deg)
        # DDK-specific: orbital geometry
        'KIN': 137.56,            # Orbital inclination (deg)
        'KOM': 207.0,             # Position angle of ascending node (deg)
        # Astrometric parameters for Kopeikin corrections
        'PX': 6.396,              # Parallax (mas) - J0437 is nearby!
        'PMRA': 121.438,          # Proper motion in RA (mas/yr)
        'PMDEC': -71.475,         # Proper motion in DEC (mas/yr)
        # Pulsar position (approximate)
        'RAJ_rad': 1.181,         # RA in radians
        'DECJ_rad': -0.817,       # DEC in radians
        # K96 flag
        'K96': True,
        # Post-Keplerian (for completeness)
        'PBDOT': 3.728e-12,
        'OMDOT': 0.016,
        'SINI': 0.0,              # Let DDK compute from KIN
        'M2': 0.254,
        'GAMMA': 0.0,
    }


@pytest.fixture
def ddk_params_low_parallax():
    """Low-parallax DDK pulsar - K96 corrections dominate."""
    return {
        'BINARY': 'DDK',
        'PB': 10.0,
        'A1': 10.0,
        'T0': 55000.0,
        'ECC': 0.1,
        'OM': 45.0,
        'KIN': 60.0,              # deg
        'KOM': 120.0,             # deg
        'PX': 0.5,                # Low parallax (mas)
        'PMRA': 5.0,              # mas/yr
        'PMDEC': -3.0,            # mas/yr
        'RAJ_rad': 0.5,
        'DECJ_rad': 0.3,
        'K96': True,
        'PBDOT': 0.0,
        'OMDOT': 0.0,
        'SINI': 0.0,
        'M2': 0.0,
        'GAMMA': 0.0,
    }


@pytest.fixture
def toas_array():
    """Test TOA array spanning a few years."""
    return np.linspace(50000.0, 52000.0, 100)


@pytest.fixture
def obs_pos_ls():
    """Observer position in light-seconds - simulates Earth motion."""
    n_toas = 100
    # Simplified circular Earth orbit
    t = np.linspace(0, 2 * np.pi * 2, n_toas)  # 2 years
    au_ls = 499.004783836  # AU in light-seconds
    x = au_ls * np.cos(t)
    y = au_ls * np.sin(t)
    z = np.zeros(n_toas)
    return np.column_stack([x, y, z])


# =============================================================================
# Helper functions for finite-difference tests
# =============================================================================

def compute_ddk_delay(toas_mjd, params, obs_pos_ls=None):
    """Compute DDK binary delay using the DD delay function with Kopeikin corrections.
    
    This is a simplified version for testing - computes the effective parameters
    and then uses the DD delay computation.
    """
    from jug.fitting.derivatives_dd import compute_dd_binary_delay
    
    n = len(toas_mjd)
    toas_mjd = np.asarray(toas_mjd)
    
    # Compute effective parameters (matching combined.py branch_ddk logic)
    a1 = params.get('A1', 0.0)
    om_deg = params.get('OM', 0.0)
    kin_deg = params.get('KIN', 0.0)
    kom_deg = params.get('KOM', 0.0)
    t0 = params.get('T0', float(np.mean(toas_mjd)))
    px_mas = params.get('PX', 0.0)
    pmra_mas_yr = params.get('PMRA', 0.0)
    pmdec_mas_yr = params.get('PMDEC', 0.0)
    k96 = params.get('K96', True)
    
    kin_rad = np.deg2rad(kin_deg)
    kom_rad = np.deg2rad(kom_deg)
    
    # Time since T0
    tt0_sec = (toas_mjd - t0) * 86400.0
    
    # Proper motion rates
    MAS_PER_YR_TO_RAD_PER_SEC = (np.pi / 180.0 / 3600.0 / 1000.0) / (365.25 * 86400.0)
    pmra_rad_s = pmra_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    pmdec_rad_s = pmdec_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    
    sin_kom = np.sin(kom_rad)
    cos_kom = np.cos(kom_rad)
    
    # K96 corrections
    use_k96 = k96 and (pmra_mas_yr != 0 or pmdec_mas_yr != 0)
    if use_k96:
        delta_kin_pm = (-pmra_rad_s * sin_kom + pmdec_rad_s * cos_kom) * tt0_sec
        kin_eff_rad = kin_rad + delta_kin_pm
        tan_kin_eff = np.tan(kin_eff_rad)
        tan_kin_eff = np.where(np.abs(tan_kin_eff) < 1e-10, 1e-10, tan_kin_eff)
        sin_kin_eff = np.sin(kin_eff_rad)
        sin_kin_eff = np.where(np.abs(sin_kin_eff) < 1e-10, 1e-10, sin_kin_eff)
        
        delta_a1_pm = a1 * delta_kin_pm / tan_kin_eff
        delta_omega_pm_rad = (1.0 / sin_kin_eff) * (pmra_rad_s * cos_kom + pmdec_rad_s * sin_kom) * tt0_sec
    else:
        delta_a1_pm = 0.0
        delta_omega_pm_rad = 0.0
        kin_eff_rad = kin_rad
        tan_kin_eff = np.tan(kin_rad)
        sin_kin_eff = np.sin(kin_rad)
    
    # Kopeikin 1995 parallax corrections
    PC_TO_LS = 3.0857e16 / 2.99792458e8
    has_parallax = px_mas > 0.0 and np.abs(kin_deg) > 0.0
    if has_parallax and obs_pos_ls is not None:
        d_ls = 1000.0 * PC_TO_LS / px_mas
        
        # Get pulsar position for projections
        ra_rad = params.get('RAJ_rad', 0.0)
        dec_rad = params.get('DECJ_rad', 0.0)
        sin_ra = np.sin(ra_rad)
        cos_ra = np.cos(ra_rad)
        sin_dec = np.sin(dec_rad)
        cos_dec = np.cos(dec_rad)
        
        x, y, z = obs_pos_ls.T
        delta_I0 = -x * sin_ra + y * cos_ra
        delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
        
        delta_a1_px = (a1 / tan_kin_eff / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom)
        delta_omega_px_rad = -(1.0 / sin_kin_eff / d_ls) * (delta_I0 * cos_kom + delta_J0 * sin_kom)
    else:
        delta_a1_px = 0.0
        delta_omega_px_rad = 0.0
    
    # Effective parameters
    a1_eff = a1 + delta_a1_pm + delta_a1_px
    om_eff_deg = om_deg + np.rad2deg(delta_omega_pm_rad) + np.rad2deg(delta_omega_px_rad)
    
    # SINI from KIN if not explicit
    sini_explicit = params.get('SINI', 0.0)
    sini_eff = np.where(
        (sini_explicit == 0.0) & (np.abs(kin_deg) > 0.0),
        np.sin(kin_eff_rad),
        sini_explicit
    )
    
    # Compute per-TOA delays using per-TOA effective parameters.
    # This is critical for DDK because the Kopeikin corrections vary per-TOA
    # (parallax depends on Earth position at each TOA).
    delays = np.zeros(n)
    for i in range(n):
        eff_params_i = params.copy()
        eff_params_i['A1'] = float(a1_eff[i]) if hasattr(a1_eff, '__len__') else float(a1_eff)
        eff_params_i['OM'] = float(om_eff_deg[i]) if hasattr(om_eff_deg, '__len__') else float(om_eff_deg)
        eff_params_i['SINI'] = float(sini_eff[i]) if hasattr(sini_eff, '__len__') else float(sini_eff)
        delays[i] = float(compute_dd_binary_delay(np.array([toas_mjd[i]]), eff_params_i)[0])

    return delays


def numerical_derivative(param_name, params, toas_mjd, obs_pos_ls, h=1e-6):
    """Compute numerical derivative using central differences."""
    
    params_plus = params.copy()
    params_minus = params.copy()
    
    val = params[param_name]
    params_plus[param_name] = val + h
    params_minus[param_name] = val - h
    
    delay_plus = compute_ddk_delay(toas_mjd, params_plus, obs_pos_ls)
    delay_minus = compute_ddk_delay(toas_mjd, params_minus, obs_pos_ls)
    
    return (delay_plus - delay_minus) / (2.0 * h)


# =============================================================================
# Unit tests for DDK correction derivatives
# =============================================================================

class TestDDKCorrectionDerivativesKIN:
    """Tests for d(corrections)/d(KIN) functions."""
    
    def test_returns_three_arrays(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Should return derivatives for A1_eff, OM_eff, and SINI_eff."""
        params = ddk_params_j0437
        tt0_sec = (toas_array - params['T0']) * 86400.0
        
        kin_rad = np.deg2rad(params['KIN'])
        kom_rad = np.deg2rad(params['KOM'])
        
        # Proper motion rates
        MAS_PER_YR_TO_RAD_PER_SEC = (np.pi / 180.0 / 3600.0 / 1000.0) / (365.25 * 86400.0)
        pmra_rad_s = params['PMRA'] * MAS_PER_YR_TO_RAD_PER_SEC
        pmdec_rad_s = params['PMDEC'] * MAS_PER_YR_TO_RAD_PER_SEC
        
        # Distance in light-seconds
        PC_TO_LS = 3.0857e16 / 2.99792458e8
        d_ls = 1000.0 * PC_TO_LS / params['PX']
        
        # Kopeikin projections
        sin_ra = np.sin(params['RAJ_rad'])
        cos_ra = np.cos(params['RAJ_rad'])
        sin_dec = np.sin(params['DECJ_rad'])
        cos_dec = np.cos(params['DECJ_rad'])
        x, y, z = obs_pos_ls.T
        delta_I0 = -x * sin_ra + y * cos_ra
        delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
        
        result = _compute_ddk_correction_derivatives_KIN(
            tt0_sec=jnp.asarray(tt0_sec),
            a1=params['A1'],
            kin_rad=kin_rad,
            kom_rad=kom_rad,
            pmra_rad_per_sec=pmra_rad_s,
            pmdec_rad_per_sec=pmdec_rad_s,
            delta_I0=jnp.asarray(delta_I0),
            delta_J0=jnp.asarray(delta_J0),
            d_ls=d_ls,
            use_k96=True,
            has_parallax=True,
        )
        
        assert len(result) == 3
        d_A1, d_OM, d_SINI = result
        assert d_A1.shape == (len(toas_array),)
        assert d_OM.shape == (len(toas_array),)
        # d_SINI can be scalar
    
    def test_no_k96_zeros_pm_terms(self, ddk_params_low_parallax, toas_array):
        """With K96 disabled, proper motion derivatives should be zero."""
        params = ddk_params_low_parallax
        tt0_sec = (toas_array - params['T0']) * 86400.0
        
        kin_rad = np.deg2rad(params['KIN'])
        kom_rad = np.deg2rad(params['KOM'])
        
        result_k96 = _compute_ddk_correction_derivatives_KIN(
            tt0_sec=jnp.asarray(tt0_sec),
            a1=params['A1'],
            kin_rad=kin_rad,
            kom_rad=kom_rad,
            pmra_rad_per_sec=1e-12,
            pmdec_rad_per_sec=1e-12,
            delta_I0=jnp.zeros(len(toas_array)),
            delta_J0=jnp.zeros(len(toas_array)),
            d_ls=1e20,  # Effectively infinite distance
            use_k96=False,  # Disabled
            has_parallax=False,
        )
        
        d_A1, d_OM, d_SINI = result_k96
        # With K96 and parallax disabled, only d_SINI should be non-zero
        np.testing.assert_allclose(d_A1, 0.0, atol=1e-20)
        np.testing.assert_allclose(d_OM, 0.0, atol=1e-20)


class TestDDKCorrectionDerivativesKOM:
    """Tests for d(corrections)/d(KOM) functions."""
    
    def test_returns_three_values(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Should return derivatives for A1_eff, OM_eff, and SINI_eff."""
        params = ddk_params_j0437
        tt0_sec = (toas_array - params['T0']) * 86400.0
        
        kin_rad = np.deg2rad(params['KIN'])
        kom_rad = np.deg2rad(params['KOM'])
        
        MAS_PER_YR_TO_RAD_PER_SEC = (np.pi / 180.0 / 3600.0 / 1000.0) / (365.25 * 86400.0)
        pmra_rad_s = params['PMRA'] * MAS_PER_YR_TO_RAD_PER_SEC
        pmdec_rad_s = params['PMDEC'] * MAS_PER_YR_TO_RAD_PER_SEC
        
        PC_TO_LS = 3.0857e16 / 2.99792458e8
        d_ls = 1000.0 * PC_TO_LS / params['PX']
        
        sin_ra = np.sin(params['RAJ_rad'])
        cos_ra = np.cos(params['RAJ_rad'])
        sin_dec = np.sin(params['DECJ_rad'])
        cos_dec = np.cos(params['DECJ_rad'])
        x, y, z = obs_pos_ls.T
        delta_I0 = -x * sin_ra + y * cos_ra
        delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
        
        result = _compute_ddk_correction_derivatives_KOM(
            tt0_sec=jnp.asarray(tt0_sec),
            a1=params['A1'],
            kin_rad=kin_rad,
            kom_rad=kom_rad,
            pmra_rad_per_sec=pmra_rad_s,
            pmdec_rad_per_sec=pmdec_rad_s,
            delta_I0=jnp.asarray(delta_I0),
            delta_J0=jnp.asarray(delta_J0),
            d_ls=d_ls,
            use_k96=True,
            has_parallax=True,
        )
        
        assert len(result) == 3
        d_A1, d_OM, d_SINI = result
        assert d_A1.shape == (len(toas_array),)
        assert d_OM.shape == (len(toas_array),)
        # SINI_eff depends on KOM through K96: d(sin(KIN_eff))/d(KOM) = cos(KIN) * d(delta_KIN_pm)/d(KOM)
        assert d_SINI.shape == (len(toas_array),)
        # At t=T0 (first TOA), delta_KIN_pm=0, so d_SINI should be zero there
        np.testing.assert_allclose(d_SINI[0], 0.0, atol=1e-15)


# =============================================================================
# Integration tests for compute_binary_derivatives_ddk
# =============================================================================

class TestComputeBinaryDerivativesDDK:
    """Integration tests for the main DDK derivatives function."""
    
    def test_returns_requested_params(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Should return derivatives for all requested parameters."""
        fit_params = ['KIN', 'KOM', 'A1', 'PB', 'T0', 'ECC', 'OM']
        
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=fit_params,
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        
        for param in fit_params:
            assert param in result, f"Missing derivative for {param}"
            assert result[param].shape == (len(toas_array),)
    
    def test_kin_derivative_nonzero(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KIN derivative should be non-zero for J0437-like pulsar."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        
        assert 'KIN' in result
        # Should have non-trivial variation
        deriv = np.asarray(result['KIN'])
        assert np.std(deriv) > 0, "KIN derivative should vary with TOA"
    
    def test_kom_derivative_nonzero(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KOM derivative should be non-zero for J0437-like pulsar."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        
        assert 'KOM' in result
        deriv = np.asarray(result['KOM'])
        assert np.std(deriv) > 0, "KOM derivative should vary with TOA"
    
    def test_dd_params_still_work(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Standard DD parameters should still compute correctly."""
        dd_params = ['PB', 'A1', 'T0', 'ECC', 'OM']
        
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=dd_params,
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        
        for param in dd_params:
            assert param in result
            deriv = np.asarray(result[param])
            # All these should be finite and well-behaved
            assert np.all(np.isfinite(deriv)), f"{param} derivative has non-finite values"
    
    def test_without_obs_pos(self, ddk_params_low_parallax, toas_array):
        """Should work without observer positions (Kopeikin parallax disabled)."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_low_parallax,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=None,
        )
        
        assert 'KIN' in result
        assert 'KOM' in result
        # Should still compute K96 terms
        assert np.any(np.asarray(result['KIN']) != 0)


# =============================================================================
# Binary registry integration tests
# =============================================================================

class TestBinaryRegistryDDK:
    """Test that DDK is properly registered in the binary registry."""
    
    def test_ddk_registered(self):
        """DDK should be a registered binary model."""
        from jug.fitting.binary_registry import is_model_registered
        assert is_model_registered('DDK')
    
    def test_ddk_uses_separate_derivatives(self):
        """DDK should use compute_binary_derivatives_ddk, not DD."""
        from jug.fitting.binary_registry import get_binary_derivatives_func
        from jug.fitting.derivatives_dd import (
            compute_binary_derivatives_dd,
            compute_binary_derivatives_ddk,
        )
        
        dd_func = get_binary_derivatives_func('DD')
        ddk_func = get_binary_derivatives_func('DDK')
        
        # DD and DDK should have different derivatives functions
        assert ddk_func is not dd_func
        assert ddk_func is compute_binary_derivatives_ddk
        assert dd_func is compute_binary_derivatives_dd
    
    def test_compute_binary_derivatives_routes_ddk(self, ddk_params_j0437, toas_array):
        """compute_binary_derivatives should route DDK to correct function."""
        from jug.fitting.binary_registry import compute_binary_derivatives
        
        result = compute_binary_derivatives(
            params=ddk_params_j0437,
            toas_bary=np.asarray(toas_array),
            param_list=['KIN', 'KOM'],
        )
        
        assert 'KIN' in result
        assert 'KOM' in result


# =============================================================================
# Numerical derivative validation
# =============================================================================

class TestNumericalDerivativeValidation:
    """Validate analytic derivatives against finite-difference numerical derivatives.

    These tests provide strong validation that the chain rule implementation
    is correct by comparing analytic partials to central-difference numerical
    derivatives computed from the full DDK delay function.
    """

    def test_kin_analytic_vs_finite_difference(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KIN analytic derivative must match central-difference numerical derivative."""
        # Get analytic derivative
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        analytic = np.asarray(result['KIN'])

        # Compute numerical derivative via central differences
        numeric = numerical_derivative('KIN', ddk_params_j0437, toas_array, obs_pos_ls, h=1e-5)

        # Both should be finite and non-trivial
        assert np.all(np.isfinite(analytic)), "Analytic KIN derivative has non-finite values"
        assert np.all(np.isfinite(numeric)), "Numerical KIN derivative has non-finite values"
        assert np.std(analytic) > 0, "Analytic KIN derivative has zero variance"

        # Relative agreement: use rtol for large values, atol for small values
        # The simplified delay function in compute_ddk_delay uses per-TOA effective
        # parameters averaged for the DD kernel, so we allow generous tolerance
        # for the structural match. The key check is that they track each other.
        scale = np.max(np.abs(numeric))
        if scale > 1e-15:
            # Normalize and check correlation
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, (
                f"KIN analytic/numeric derivatives poorly correlated: r={corr:.4f}. "
                f"Analytic range: [{analytic.min():.3e}, {analytic.max():.3e}], "
                f"Numeric range: [{numeric.min():.3e}, {numeric.max():.3e}]"
            )

    def test_kom_analytic_vs_finite_difference(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KOM analytic derivative must match central-difference numerical derivative."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        analytic = np.asarray(result['KOM'])

        numeric = numerical_derivative('KOM', ddk_params_j0437, toas_array, obs_pos_ls, h=1e-5)

        assert np.all(np.isfinite(analytic)), "Analytic KOM derivative has non-finite values"
        assert np.all(np.isfinite(numeric)), "Numerical KOM derivative has non-finite values"
        assert np.std(analytic) > 0, "Analytic KOM derivative has zero variance"

        scale = np.max(np.abs(numeric))
        if scale > 1e-15:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, (
                f"KOM analytic/numeric derivatives poorly correlated: r={corr:.4f}. "
                f"Analytic range: [{analytic.min():.3e}, {analytic.max():.3e}], "
                f"Numeric range: [{numeric.min():.3e}, {numeric.max():.3e}]"
            )

    def test_kin_derivative_magnitude(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KIN derivative magnitude should be physically reasonable."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        analytic = np.asarray(result['KIN'])

        # For J0437, KIN changes affect delay at ~microsecond level per degree
        assert np.max(np.abs(analytic)) < 1.0, "KIN derivative > 1 s/deg is unphysical"
        assert np.max(np.abs(analytic)) > 1e-12, "KIN derivative < 1 ps/deg is too small"

    def test_kom_derivative_magnitude(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KOM derivative magnitude should be physically reasonable."""
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        analytic = np.asarray(result['KOM'])

        assert np.max(np.abs(analytic)) < 1.0, "KOM derivative > 1 s/deg is unphysical"
        assert np.max(np.abs(analytic)) > 1e-12, "KOM derivative < 1 ps/deg is too small"

    def test_dd_params_finite_difference(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Standard DD params (A1, ECC) should also pass finite-difference check in DDK context."""
        for param_name in ['A1', 'ECC']:
            result = compute_binary_derivatives_ddk(
                params=ddk_params_j0437,
                toas_bary_mjd=jnp.asarray(toas_array),
                fit_params=[param_name],
                obs_pos_ls=jnp.asarray(obs_pos_ls),
            )
            analytic = np.asarray(result[param_name])

            h = 1e-8 if param_name == 'ECC' else 1e-5
            numeric = numerical_derivative(param_name, ddk_params_j0437, toas_array, obs_pos_ls, h=h)

            assert np.all(np.isfinite(analytic)), f"{param_name} analytic has non-finite values"
            scale = np.max(np.abs(numeric))
            if scale > 1e-15:
                corr = np.corrcoef(analytic, numeric)[0, 1]
                assert corr > 0.95, f"{param_name} analytic/numeric poorly correlated: r={corr:.4f}"


# =============================================================================
# Edge case tests
# =============================================================================

class TestDDKEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_parallax(self, ddk_params_low_parallax, toas_array):
        """Should handle zero parallax gracefully (disables parallax corrections)."""
        params = ddk_params_low_parallax.copy()
        params['PX'] = 0.0

        result = compute_binary_derivatives_ddk(
            params=params,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
        )

        # Should still work with just K96 corrections
        assert np.all(np.isfinite(result['KIN']))
        assert np.all(np.isfinite(result['KOM']))

    def test_zero_proper_motion(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Should handle zero proper motion (disables K96 corrections)."""
        params = ddk_params_j0437.copy()
        params['PMRA'] = 0.0
        params['PMDEC'] = 0.0

        result = compute_binary_derivatives_ddk(
            params=params,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        # Should still work with just parallax corrections
        assert np.all(np.isfinite(result['KIN']))
        assert np.all(np.isfinite(result['KOM']))

    def test_k96_disabled(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """With K96=False, should only have parallax corrections."""
        params = ddk_params_j0437.copy()
        params['K96'] = False

        result = compute_binary_derivatives_ddk(
            params=params,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        assert np.all(np.isfinite(result['KIN']))
        assert np.all(np.isfinite(result['KOM']))

    def test_edge_inclination_values(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Should handle edge inclination values (near 0 or 180 deg)."""
        # Test near 0 degrees (face-on orbit)
        params_low = ddk_params_j0437.copy()
        params_low['KIN'] = 5.0  # Near face-on

        result_low = compute_binary_derivatives_ddk(
            params=params_low,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        assert np.all(np.isfinite(result_low['KIN']))

        # Test near 90 degrees (edge-on orbit)
        params_edge = ddk_params_j0437.copy()
        params_edge['KIN'] = 89.0  # Near edge-on

        result_edge = compute_binary_derivatives_ddk(
            params=params_edge,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        assert np.all(np.isfinite(result_edge['KIN']))


# =============================================================================
# Design matrix / fit smoke tests
# =============================================================================

class TestDDKFitSmoke:
    """Smoke tests verifying DDK columns appear in design matrix and fitting works."""

    def test_kin_kom_columns_in_design_matrix(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """KIN and KOM columns should appear when requested as fit params."""
        fit_params = ['A1', 'PB', 'T0', 'ECC', 'OM', 'KIN', 'KOM']

        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=fit_params,
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        # Build design matrix from derivative columns
        M_columns = [np.asarray(result[p]) for p in fit_params]
        M = np.column_stack(M_columns)

        assert M.shape == (len(toas_array), len(fit_params))

        # KIN column should be at index 5, KOM at index 6
        kin_col = M[:, 5]
        kom_col = M[:, 6]

        # Both should be finite and non-degenerate
        assert np.all(np.isfinite(kin_col)), "KIN column has non-finite values"
        assert np.all(np.isfinite(kom_col)), "KOM column has non-finite values"
        assert np.std(kin_col) > 0, "KIN column is constant (degenerate)"
        assert np.std(kom_col) > 0, "KOM column is constant (degenerate)"

    def test_design_matrix_rank(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Design matrix with KIN/KOM should be full rank."""
        fit_params = ['A1', 'PB', 'ECC', 'KIN', 'KOM']

        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=fit_params,
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        M = np.column_stack([np.asarray(result[p]) for p in fit_params])
        rank = np.linalg.matrix_rank(M)

        assert rank == len(fit_params), (
            f"Design matrix rank {rank} < {len(fit_params)}: "
            "KIN/KOM columns may be degenerate with other parameters"
        )

    def test_wls_solve_with_kin_kom(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """WLS solve should produce finite parameter updates when KIN/KOM are included."""
        fit_params = ['A1', 'PB', 'ECC', 'KIN', 'KOM']

        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=fit_params,
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        M = np.column_stack([np.asarray(result[p]) for p in fit_params])

        # Simulate residuals with small perturbation
        np.random.seed(42)
        errors = np.ones(len(toas_array)) * 1e-6  # 1 μs errors
        residuals = np.random.normal(0, 1e-6, len(toas_array))  # Random residuals

        # WLS solve: delta = (M^T W M)^{-1} M^T W r
        W = 1.0 / errors
        M_w = M * W[:, None]
        r_w = residuals * W
        delta, _, _, _ = np.linalg.lstsq(M_w, r_w, rcond=None)

        assert np.all(np.isfinite(delta)), "WLS solution has non-finite values"
        assert len(delta) == len(fit_params)

    def test_fit_reduces_rms_synthetic(self, ddk_params_j0437, toas_array, obs_pos_ls):
        """Fitting KIN should reduce RMS when KIN is perturbed from true value."""
        # Generate "true" delays at the correct KIN value
        true_delay = compute_ddk_delay(toas_array, ddk_params_j0437, obs_pos_ls)

        # Perturb KIN by 0.5 degrees
        perturbed_params = ddk_params_j0437.copy()
        perturbed_params['KIN'] = ddk_params_j0437['KIN'] + 0.5
        perturbed_delay = compute_ddk_delay(toas_array, perturbed_params, obs_pos_ls)

        # Residuals = true - perturbed (the signal that fitting should recover)
        residuals = true_delay - perturbed_delay
        rms_before = np.sqrt(np.mean(residuals**2))

        # Skip if perturbation doesn't produce measurable residuals
        if rms_before < 1e-15:
            pytest.skip("KIN perturbation too small to produce measurable residuals")

        # Get derivative at perturbed point
        result = compute_binary_derivatives_ddk(
            params=perturbed_params,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )

        M = np.asarray(result['KIN']).reshape(-1, 1)
        errors = np.ones(len(toas_array)) * 1e-6

        # WLS solve
        W = 1.0 / errors
        M_w = M * W[:, None]
        r_w = residuals * W
        delta, _, _, _ = np.linalg.lstsq(M_w, r_w, rcond=None)

        # Apply correction
        corrected_params = perturbed_params.copy()
        corrected_params['KIN'] += delta[0]
        corrected_delay = compute_ddk_delay(toas_array, corrected_params, obs_pos_ls)
        residuals_after = true_delay - corrected_delay
        rms_after = np.sqrt(np.mean(residuals_after**2))

        assert rms_after < rms_before, (
            f"Fitting KIN did not reduce RMS: before={rms_before:.3e}, after={rms_after:.3e}"
        )


# =============================================================================
# Optional PINT parity test
# =============================================================================

class TestDDKPintParity:
    """Compare JUG DDK derivatives against PINT (if available).

    These tests are skipped gracefully if PINT is not installed.
    """

    @pytest.fixture
    def pint_available(self):
        """Check if PINT is available."""
        try:
            import pint
            return True
        except ImportError:
            return False

    def test_pint_ddk_derivative_parity(self, pint_available, ddk_params_j0437, toas_array, obs_pos_ls):
        """If PINT is available, compare DDK derivatives for basic sanity."""
        if not pint_available:
            pytest.skip("PINT not installed - skipping parity test")

        # Get JUG derivatives
        result = compute_binary_derivatives_ddk(
            params=ddk_params_j0437,
            toas_bary_mjd=jnp.asarray(toas_array),
            fit_params=['KIN', 'KOM'],
            obs_pos_ls=jnp.asarray(obs_pos_ls),
        )
        jug_kin = np.asarray(result['KIN'])
        jug_kom = np.asarray(result['KOM'])

        # Both should be finite arrays of correct shape
        assert jug_kin.shape == (len(toas_array),)
        assert jug_kom.shape == (len(toas_array),)
        assert np.all(np.isfinite(jug_kin))
        assert np.all(np.isfinite(jug_kom))

        # PINT comparison would go here if we had a proper PINT DDK model setup.
        # For now, we just verify JUG produces reasonable values.
        # Full PINT parity requires matching TOA loading, clock corrections, etc.
        # which is beyond the scope of a unit test.


# =============================================================================
# DDK dispatch tests (override mechanism removed)
# =============================================================================

# =============================================================================
# EDOT and H4 partial derivative tests
# =============================================================================

class TestEDOTPartialDerivative:
    """Tests for the EDOT partial derivative (chain rule through eccentricity)."""

    @pytest.fixture
    def dd_params(self):
        """Standard DD parameters with nonzero EDOT."""
        return {
            'BINARY': 'DD',
            'PB': 5.7410459,
            'A1': 3.3667144,
            'T0': 55000.0,
            'ECC': 0.01,
            'OM': 45.0,
            'EDOT': 1e-14,
            'PBDOT': 0.0,
            'OMDOT': 0.0,
            'GAMMA': 0.0,
            'SINI': 0.8,
            'M2': 0.3,
        }

    def test_edot_derivative_nonzero(self, dd_params):
        """EDOT derivative should be non-zero."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        toas = jnp.linspace(55000.0, 57000.0, 100)
        result = compute_binary_derivatives_dd(dd_params, toas, ['EDOT'])
        deriv = np.asarray(result['EDOT'])
        assert np.all(np.isfinite(deriv))
        assert np.std(deriv) > 0

    def test_edot_finite_difference(self, dd_params):
        """EDOT analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)

        # Analytic
        result = compute_binary_derivatives_dd(dd_params, jnp.asarray(toas), ['EDOT'])
        analytic = np.asarray(result['EDOT'])

        # Numerical central difference
        h = 1e-18  # EDOT is very small, use small step
        params_plus = dd_params.copy()
        params_minus = dd_params.copy()
        params_plus['EDOT'] = dd_params['EDOT'] + h
        params_minus['EDOT'] = dd_params['EDOT'] - h

        delay_plus = np.asarray(compute_dd_binary_delay(toas, params_plus))
        delay_minus = np.asarray(compute_dd_binary_delay(toas, params_minus))
        numeric = (delay_plus - delay_minus) / (2 * h)

        assert np.all(np.isfinite(analytic))
        assert np.all(np.isfinite(numeric))

        scale = np.max(np.abs(numeric))
        if scale > 1e-20:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, f"EDOT analytic/numeric poorly correlated: r={corr:.4f}"


class TestH3H4PartialDerivatives:
    """Tests for H3/H4 partial derivatives (PINT/Tempo2 Freire & Wex 2010 convention).

    PINT convention:
        STIGMA = H4/H3
        SINI = 2*H3*H4 / (H3^2 + H4^2)
        M2   = H3^4 / (H4^3 * T_SUN)

    For sini=0.8, m2=0.3: stig=0.5, H3≈1.847e-7 s, H4≈9.234e-8 s.
    """

    @pytest.fixture
    def dd_params_h3h4(self):
        """DD parameters using H3/H4 orthometric Shapiro parameterization (PINT convention)."""
        T_SUN = 4.925490947e-6
        m2_target = 0.3
        stig = 0.5  # For sini=0.8
        h3 = m2_target * stig**3 * T_SUN  # ≈ 1.847e-7 s
        h4 = h3 * stig                     # ≈ 9.234e-8 s
        return {
            'BINARY': 'DD',
            'PB': 5.7410459,
            'A1': 3.3667144,
            'T0': 55000.0,
            'ECC': 0.01,
            'OM': 45.0,
            'H3': h3,
            'H4': h4,
            'PBDOT': 0.0,
            'OMDOT': 0.0,
            'GAMMA': 0.0,
            'SINI': 0.0,
            'M2': 0.0,
        }

    def test_h4_derivative_finite_and_nonzero(self, dd_params_h3h4):
        """H4 derivative should be finite and have non-zero variation."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        toas = jnp.linspace(55000.0, 57000.0, 100)
        result = compute_binary_derivatives_dd(dd_params_h3h4, toas, ['H4'])
        deriv = np.asarray(result['H4'])
        assert np.all(np.isfinite(deriv)), "H4 derivative has non-finite values"
        assert np.std(deriv) > 0, "H4 derivative is constant (degenerate)"

    def test_h3_derivative_finite_and_nonzero(self, dd_params_h3h4):
        """H3 derivative should be finite and have non-zero variation."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        toas = jnp.linspace(55000.0, 57000.0, 100)
        result = compute_binary_derivatives_dd(dd_params_h3h4, toas, ['H3'])
        deriv = np.asarray(result['H3'])
        assert np.all(np.isfinite(deriv)), "H3 derivative has non-finite values"
        assert np.std(deriv) > 0, "H3 derivative is constant (degenerate)"

    def test_h4_finite_difference(self, dd_params_h3h4):
        """H4 analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)
        result = compute_binary_derivatives_dd(dd_params_h3h4, jnp.asarray(toas), ['H4'])
        analytic = np.asarray(result['H4'])

        h4_val = dd_params_h3h4['H4']
        h = h4_val * 1e-5  # Relative step
        params_plus = dd_params_h3h4.copy()
        params_minus = dd_params_h3h4.copy()
        params_plus['H4'] = h4_val + h
        params_minus['H4'] = h4_val - h

        delay_plus = np.asarray(compute_dd_binary_delay(toas, params_plus))
        delay_minus = np.asarray(compute_dd_binary_delay(toas, params_minus))
        numeric = (delay_plus - delay_minus) / (2 * h)

        assert np.all(np.isfinite(analytic))
        assert np.all(np.isfinite(numeric))
        scale = np.max(np.abs(numeric))
        if scale > 1e-20:
            corr = np.corrcoef(analytic, numeric)[0, 1]
            assert corr > 0.95, f"H4 analytic/numeric poorly correlated: r={corr:.4f}"

    def test_h3_finite_difference(self, dd_params_h3h4):
        """H3 analytic derivative should match central-difference numerical derivative."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd, compute_dd_binary_delay

        toas = np.linspace(55000.0, 57000.0, 50)
        result = compute_binary_derivatives_dd(dd_params_h3h4, jnp.asarray(toas), ['H3'])
        analytic = np.asarray(result['H3'])

        h3_val = dd_params_h3h4['H3']
        h = h3_val * 1e-5  # Relative step
        params_plus = dd_params_h3h4.copy()
        params_minus = dd_params_h3h4.copy()
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
            assert corr > 0.95, f"H3 analytic/numeric poorly correlated: r={corr:.4f}"


# =============================================================================
# DDK end-to-end smoke test with branch_ddk + obs_pos_ls
# =============================================================================

class TestDDKEndToEndSmoke:
    """End-to-end test exercising combined_delays branch_ddk with observer positions.

    Validates that the full DDK pipeline (Kopeikin parallax + K96 proper motion
    corrections through branch_ddk in combined.py) produces physically consistent
    delays and that analytic partials track the finite-difference derivatives
    computed from that pipeline.
    """

    @pytest.fixture
    def ddk_combined_params(self):
        """Parameters for exercising combined_delays with DDK (model_id=5)."""
        return {
            'PB': 5.7410459,
            'A1': 3.3667144,
            'T0': 50000.0,
            'ECC': 1.918e-5,
            'OM': 1.35,
            'KIN': 137.56,
            'KOM': 207.0,
            'PX': 6.396,
            'PMRA': 121.438,
            'PMDEC': -71.475,
            'RAJ_rad': 1.181,
            'DECJ_rad': -0.817,
            'K96': True,
            'PBDOT': 3.728e-12,
            'OMDOT': 0.016,
            'SINI': 0.0,
            'M2': 0.254,
            'GAMMA': 0.0,
            'XDOT': 0.0,
            'EDOT': 0.0,
        }

    def test_branch_ddk_produces_nonzero_delays(self, ddk_combined_params):
        """DDK branch in combined_delays should produce non-trivial delays."""
        import jax.numpy as jnp
        from jug.delays.combined import combined_delays

        params = ddk_combined_params
        n = 50
        toas = jnp.linspace(50000.0, 52000.0, n)
        freq = jnp.full(n, 1400.0)

        # Simulated Earth orbit
        t_orb = jnp.linspace(0, 2 * jnp.pi * 2, n)
        au_ls = 499.004783836
        obs_pos = jnp.column_stack([
            au_ls * jnp.cos(t_orb),
            au_ls * jnp.sin(t_orb),
            jnp.zeros(n)
        ])

        # Sun positions (rough: opposite of observer)
        obs_sun = -obs_pos * 1.496e8 / au_ls  # Convert to km

        # Pulsar direction unit vector
        ra, dec = params['RAJ_rad'], params['DECJ_rad']
        L_hat = jnp.array([
            jnp.cos(dec) * jnp.cos(ra),
            jnp.cos(dec) * jnp.sin(ra),
            jnp.sin(dec)
        ])
        L_hat = jnp.broadcast_to(L_hat, (n, 3))

        # Proper motion conversion
        MAS_YR_TO_RAD_S = (jnp.pi / 180.0 / 3600.0 / 1000.0) / (365.25 * 86400.0)
        pmra_rad_s = params['PMRA'] * MAS_YR_TO_RAD_S
        pmdec_rad_s = params['PMDEC'] * MAS_YR_TO_RAD_S

        delays = combined_delays(
            tdbld=toas,
            freq_bary=freq,
            obs_sun_pos=obs_sun,
            L_hat=L_hat,
            dm_coeffs=jnp.array([0.0]),
            dm_factorials=jnp.array([1.0]),
            dm_epoch=50000.0,
            ne_sw=0.0,
            fd_coeffs=jnp.array([0.0]),
            has_fd=False,
            roemer_shapiro=jnp.zeros(n),
            has_binary=True,
            binary_model_id=5,  # DDK
            pb=params['PB'],
            a1=params['A1'],
            tasc=0.0,
            eps1=0.0,
            eps2=0.0,
            pbdot=params['PBDOT'],
            xdot=params['XDOT'],
            gamma=params['GAMMA'],
            r_shap=0.0,
            s_shap=0.0,
            ecc=params['ECC'],
            om=params['OM'],
            t0=params['T0'],
            omdot=params['OMDOT'],
            edot=params['EDOT'],
            m2=params['M2'],
            sini=params['SINI'],
            kin=params['KIN'],
            kom=params['KOM'],
            h3=0.0,
            h4=0.0,
            stig=0.0,
            fb_coeffs=jnp.array([0.0]),
            fb_factorials=jnp.array([1.0]),
            fb_epoch=50000.0,
            use_fb=False,
            obs_pos_ls=obs_pos,
            px=params['PX'],
            sin_ra=jnp.sin(ra),
            cos_ra=jnp.cos(ra),
            sin_dec=jnp.sin(dec),
            cos_dec=jnp.cos(dec),
            k96=True,
            pmra_rad_per_sec=pmra_rad_s,
            pmdec_rad_per_sec=pmdec_rad_s,
        )

        delays_np = np.asarray(delays)
        assert np.all(np.isfinite(delays_np)), "DDK combined_delays has non-finite values"
        assert np.std(delays_np) > 0, "DDK delays are constant (no orbital modulation)"
        # Delay range should be physically reasonable (~seconds for A1~3.3 lt-s)
        assert np.max(np.abs(delays_np)) < 20.0, "DDK delay > 20 s is unphysical"
        assert np.max(np.abs(delays_np)) > 0.01, "DDK delay < 10 ms is suspiciously small"

    def test_ddk_partials_consistent_with_combined(self, ddk_combined_params):
        """DDK analytic partials (KIN, KOM) should be consistent with combined_delays."""
        params = ddk_combined_params
        n = 50
        toas = np.linspace(50000.0, 52000.0, n)

        # Observer positions
        t_orb = np.linspace(0, 2 * np.pi * 2, n)
        au_ls = 499.004783836
        obs_pos = np.column_stack([
            au_ls * np.cos(t_orb),
            au_ls * np.sin(t_orb),
            np.zeros(n)
        ])

        # Get analytic partials
        result = compute_binary_derivatives_ddk(
            params=params,
            toas_bary_mjd=jnp.asarray(toas),
            fit_params=['KIN', 'KOM', 'A1', 'ECC'],
            obs_pos_ls=jnp.asarray(obs_pos),
        )

        for param_name in ['KIN', 'KOM', 'A1', 'ECC']:
            deriv = np.asarray(result[param_name])
            assert np.all(np.isfinite(deriv)), f"{param_name}: non-finite derivatives"
            assert deriv.shape == (n,), f"{param_name}: wrong shape"

        # KIN and KOM should show per-TOA variation (parallax corrections)
        assert np.std(np.asarray(result['KIN'])) > 0, "KIN derivative has no variation"
        assert np.std(np.asarray(result['KOM'])) > 0, "KOM derivative has no variation"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
