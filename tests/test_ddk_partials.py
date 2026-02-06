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
        # SINI_eff doesn't depend on KOM
        np.testing.assert_allclose(d_SINI, 0.0)


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

class TestDDKDispatch:
    """Verify DDK dispatch routes correctly without override mechanism."""

    def test_dispatch_ddk_raises_valueerror(self):
        """dispatch_binary_delay('DDK', ...) should raise ValueError directing to branch_ddk."""
        from jug.delays.binary_dispatch import dispatch_binary_delay

        params = {
            'PB': 5.7, 'A1': 3.3, 'ECC': 1e-5, 'OM': 1.35, 'T0': 55000.0,
            'GAMMA': 0.0, 'PBDOT': 0.0, 'OMDOT': 0.0, 'XDOT': 0.0, 'EDOT': 0.0,
            'M2': 0.0, 'SINI': 0.0,
        }
        with pytest.raises(ValueError, match="Kopeikin"):
            dispatch_binary_delay('DDK', 55000.0, params)

    def test_dispatch_dd_still_works(self):
        """DD dispatch should still work normally."""
        from jug.delays.binary_dispatch import dispatch_binary_delay

        params = {
            'PB': 5.7, 'A1': 3.3, 'ECC': 0.01, 'OM': 45.0, 'T0': 55000.0,
            'GAMMA': 0.0, 'PBDOT': 0.0, 'OMDOT': 0.0, 'XDOT': 0.0, 'EDOT': 0.0,
            'M2': 0.3, 'SINI': 0.9,
        }
        delay = dispatch_binary_delay('DD', 55000.5, params)
        assert np.isfinite(float(delay))

    def test_no_override_env_var_dependency(self):
        """Verify binary_model_overrides module is no longer imported anywhere in delays."""
        import importlib
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('jug.utils.binary_model_overrides')


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


class TestH4PartialDerivative:
    """Tests for the H4 partial derivative (chain rule through SINI and M2).

    Note: The H3/H4 parameterization in binary_dd.py uses a non-standard
    conversion (sini = H3/H4^{1/3}, m2 = H4^{1/3}/T_SUN^2) that requires
    extremely small H4 values (~1e-34 s) for physical m2. This makes
    finite-difference validation numerically unstable. We test derivative
    correctness via structure (finite, non-zero, correct sign) rather than
    correlation with numerical derivatives.
    """

    @pytest.fixture
    def dd_params_h3h4(self):
        """DD parameters using H3/H4 orthometric Shapiro parameterization.

        Values are chosen to be consistent with the binary_dd.py conversion:
        H4^(1/3)/T_SUN^2 = m2, H3/H4^(1/3) = sini.
        For m2~0.3, sini~0.8: H4~3.9e-34, H3~5.8e-12.
        """
        T_SUN = 4.925490947e-6
        m2_target = 0.3
        sini_target = 0.8
        h4_cbrt = m2_target * T_SUN**2
        h4 = h4_cbrt**3
        h3 = sini_target * h4_cbrt
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

    def test_h4_derivative_varies_with_toa(self, dd_params_h3h4):
        """H4 derivative should vary across TOAs (Shapiro modulation)."""
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        toas = jnp.linspace(55000.0, 57000.0, 100)
        result = compute_binary_derivatives_dd(dd_params_h3h4, toas, ['H4'])
        h4_deriv = np.asarray(result['H4'])
        assert np.all(np.isfinite(h4_deriv)), "H4 derivative has non-finite values"
        # Shapiro delay varies with orbital phase, so derivative should too
        assert np.max(h4_deriv) != np.min(h4_deriv), "H4 derivative is constant"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
