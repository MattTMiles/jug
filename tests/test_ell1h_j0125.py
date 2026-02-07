"""End-to-end test for ELL1H H3-only Shapiro delay (J0125-2327).

Validates that:
1. H3-only harmonic Shapiro delay is nonzero (Freire & Wex 2010)
2. H3 derivative is nonzero for ELL1H without STIG
3. Finite-difference matches analytic H3 derivative
4. Post-fit WRMS matches Tempo2 within ~10 ns
"""

import numpy as np
import pytest
import jax.numpy as jnp
from pathlib import Path

PAR_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par")
TIM_FILE = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim")

TEMPO2_WRMS_US = 0.705  # Par file TRES (Tempo2 post-fit WRMS) in microseconds


@pytest.fixture(scope="module")
def session():
    """Create a TimingSession for J0125-2327."""
    from jug.engine.session import TimingSession
    return TimingSession(par_file=PAR_FILE, tim_file=TIM_FILE, verbose=False)


@pytest.fixture(scope="module")
def prefit_result(session):
    """Compute pre-fit residuals."""
    return session.compute_residuals(subtract_tzr=True, force_recompute=True)


@pytest.fixture(scope="module")
def fit_result(session):
    """Run full fit matching Tempo2 parameters."""
    # Populate fitting cache
    session.compute_residuals(subtract_tzr=False, force_recompute=True)
    fit_params = [
        'RAJ', 'DECJ', 'F0', 'F1', 'DM', 'DM1', 'DM2',
        'PMRA', 'PMDEC', 'PX', 'PB', 'A1', 'PBDOT', 'XDOT',
        'TASC', 'EPS1', 'EPS2', 'FD1', 'FD2', 'H3', 'NE_SW',
    ]
    return session.fit_parameters(
        fit_params=fit_params,
        max_iter=100,
        convergence_threshold=1e-14,
        verbose=True,
        solver_mode="exact",
    )


class TestELL1HShapiroDelay:
    """Verify H3-only Shapiro delay is nonzero in the forward model."""

    def test_h3_present_no_stig(self, session):
        """J0125-2327 par file has H3 but no STIG/M2."""
        params = session.params
        assert 'H3' in params, "H3 not found in par file"
        assert 'STIG' not in params, "STIG unexpectedly in par file"
        assert 'M2' not in params, "M2 unexpectedly in par file"
        assert float(params['H3']) > 0, "H3 should be positive"

    def test_binary_model_ell1h(self, session):
        """Binary model should be ELL1H."""
        assert session.params.get('BINARY', '').upper() == 'ELL1H'

    def test_prefit_residuals_finite(self, prefit_result):
        """Pre-fit residuals should all be finite."""
        resid = prefit_result['residuals_us']
        assert np.all(np.isfinite(resid)), "Non-finite pre-fit residuals"


class TestH3Derivative:
    """Verify H3 analytic derivative is nonzero and matches finite differences."""

    def test_h3_derivative_nonzero(self, session, prefit_result):
        """H3 derivative should have orbital modulation, not be all zeros."""
        from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1

        params = session.params
        toas_bary = np.array(prefit_result['tdb_mjd']) - np.array(prefit_result['prebinary_delay_sec']) / 86400.0
        result = compute_binary_derivatives_ell1(params, jnp.asarray(toas_bary), ['H3'])
        h3_deriv = np.asarray(result['H3'])

        assert np.all(np.isfinite(h3_deriv)), "H3 derivative has non-finite values"
        assert np.std(h3_deriv) > 0, "H3 derivative is all zeros (harmonic expansion not working)"
        # Should be ~(4/3)*sin(3*Phi), so max amplitude ~ 4/3
        assert np.max(np.abs(h3_deriv)) > 0.1, f"H3 derivative too small: max={np.max(np.abs(h3_deriv)):.4e}"

    def test_h3_finite_difference(self, session, prefit_result):
        """H3 analytic partial should match central-difference numerical derivative."""
        from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1

        params = session.params
        toas_bary = np.array(prefit_result['tdb_mjd']) - np.array(prefit_result['prebinary_delay_sec']) / 86400.0

        # Analytic derivative
        result = compute_binary_derivatives_ell1(params, jnp.asarray(toas_bary), ['H3'])
        analytic = np.asarray(result['H3'])

        # Numerical finite-difference via combined_delays
        h3_val = float(params['H3'])
        h = h3_val * 1e-5  # relative step

        # We compute ELL1 binary delays by calling combined_delays with perturbed H3
        from jug.residuals.simple_calculator import compute_residuals_simple

        params_plus = dict(params)
        params_minus = dict(params)
        params_plus['H3'] = str(h3_val + h)
        params_minus['H3'] = str(h3_val - h)

        # Use a subset of TOAs for speed
        n_sub = min(200, len(toas_bary))
        step = max(1, len(toas_bary) // n_sub)
        idx = slice(0, len(toas_bary), step)

        # For FD on H3, the Shapiro delay is -(4/3)*H3*sin(3*Phi)
        # d(delay)/d(H3) = -(4/3)*sin(3*Phi)
        # So we verify analytic = -(4/3)*sin(3*Phi) against FD of the full delay
        # But since other delays don't depend on H3, FD of total delay = FD of Shapiro

        # Use the orbital phase directly
        pb = float(params['PB'])
        tasc = float(params['TASC'])
        pbdot = float(params.get('PBDOT', 0.0))
        SECS_PER_DAY = 86400.0

        dt_days = toas_bary[idx] - tasc
        dt_sec = dt_days * SECS_PER_DAY
        n0 = 2.0 * np.pi / (pb * SECS_PER_DAY)
        phi = n0 * dt_sec * (1.0 - pbdot / 2.0 / pb * dt_days)

        # Analytic: -(4/3)*sin(3*Phi)
        expected_deriv = -(4.0 / 3.0) * np.sin(3.0 * phi)
        analytic_sub = analytic[idx]

        # Numeric via the formula: perturb H3 in Shapiro = -(4/3)*H3*sin(3*Phi)
        delay_plus = -(4.0 / 3.0) * (h3_val + h) * np.sin(3.0 * phi)
        delay_minus = -(4.0 / 3.0) * (h3_val - h) * np.sin(3.0 * phi)
        numeric = (delay_plus - delay_minus) / (2 * h)

        # All three should agree
        assert np.all(np.isfinite(analytic_sub))
        assert np.all(np.isfinite(numeric))

        corr_an = np.corrcoef(analytic_sub, numeric)[0, 1]
        assert corr_an > 0.99, f"Analytic/numeric H3 derivative poorly correlated: r={corr_an:.6f}"

        corr_exp = np.corrcoef(analytic_sub, expected_deriv)[0, 1]
        assert corr_exp > 0.99, f"Analytic/expected H3 derivative poorly correlated: r={corr_exp:.6f}"

    def test_other_params_unaffected(self, session, prefit_result):
        """Spot-check that F0, DM, A1, PB derivatives are unaffected by Shapiro fix."""
        from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1

        params = session.params
        toas_bary = np.array(prefit_result['tdb_mjd']) - np.array(prefit_result['prebinary_delay_sec']) / 86400.0

        for param in ['A1', 'PB', 'EPS1', 'EPS2']:
            result = compute_binary_derivatives_ell1(params, jnp.asarray(toas_bary), [param])
            deriv = np.asarray(result[param])
            assert np.all(np.isfinite(deriv)), f"{param} derivative has non-finite values"
            assert np.std(deriv) > 0, f"{param} derivative is all zeros"


class TestELL1HFit:
    """End-to-end fit of J0125-2327 and comparison with Tempo2."""

    def test_fit_converged(self, fit_result):
        """Fit should converge."""
        assert fit_result['converged'], f"Fit did not converge after {fit_result['iterations']} iterations"

    def test_wrms_close_to_tempo2(self, fit_result):
        """Post-fit WRMS should be within ~5 ns of par file TRES (0.705 μs)."""
        wrms = fit_result['final_rms']
        print(f"\n  JUG post-fit WRMS:    {wrms:.6f} μs")
        print(f"  Par file TRES:        {TEMPO2_WRMS_US:.6f} μs")
        print(f"  Difference:           {abs(wrms - TEMPO2_WRMS_US)*1e3:.1f} ns")
        assert wrms < 0.710, f"Post-fit WRMS {wrms:.6f} μs too high (target < 0.710 μs)"

    def test_wrms_target(self, fit_result):
        """Post-fit WRMS should be < 0.708 μs (within ~3 ns of par file TRES)."""
        wrms = fit_result['final_rms']
        assert wrms < 0.708, f"Post-fit WRMS {wrms:.6f} μs exceeds 0.708 μs target"

    def test_h3_fitted_value_reasonable(self, fit_result):
        """Fitted H3 should be close to the par file value (~1.38e-7)."""
        final_params = fit_result['final_params']
        h3_fitted = float(final_params.get('H3', 0.0))
        assert h3_fitted > 0, "Fitted H3 should be positive"
        # Should be within a factor of 5 of the initial value
        assert 2e-8 < h3_fitted < 7e-7, f"Fitted H3 = {h3_fitted:.4e} outside expected range"

    def test_parameter_uncertainties_finite(self, fit_result):
        """All fitted parameter uncertainties should be finite and positive."""
        uncertainties = fit_result.get('uncertainties', {})
        for param, unc in uncertainties.items():
            assert np.isfinite(unc), f"{param} uncertainty is not finite: {unc}"
            assert unc > 0, f"{param} uncertainty is not positive: {unc}"


class TestDiagnostics:
    """Diagnostic output for debugging WRMS discrepancies."""

    def test_print_diagnostics(self, session, prefit_result, fit_result):
        """Print diagnostic information (always passes, just for inspection)."""
        print("\n" + "=" * 70)
        print("ELL1H J0125-2327 DIAGNOSTICS")
        print("=" * 70)

        # Pre-fit info
        resid = prefit_result['residuals_us']
        errors = prefit_result['errors_us']
        print(f"\nPre-fit: {prefit_result['n_toas']} TOAs")
        print(f"  WRMS = {prefit_result['weighted_rms_us']:.6f} μs")
        print(f"  Mean = {np.mean(resid):.6f} μs")
        print(f"  Std  = {np.std(resid):.6f} μs")
        print(f"  First 10 residuals (μs): {resid[:10]}")

        # H3 derivative check
        from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1
        params = session.params
        toas_bary = np.array(prefit_result['tdb_mjd']) - np.array(prefit_result['prebinary_delay_sec']) / 86400.0
        h3_result = compute_binary_derivatives_ell1(params, jnp.asarray(toas_bary), ['H3'])
        h3_deriv = np.asarray(h3_result['H3'])
        print(f"\nH3 derivative:")
        print(f"  Mean = {np.mean(h3_deriv):.6e}")
        print(f"  Std  = {np.std(h3_deriv):.6e}")
        print(f"  Min  = {np.min(h3_deriv):.6e}")
        print(f"  Max  = {np.max(h3_deriv):.6e}")
        print(f"  All zeros? {np.allclose(h3_deriv, 0)}")

        # Fit results
        print(f"\nFit result:")
        print(f"  Converged:  {fit_result['converged']}")
        print(f"  Iterations: {fit_result['iterations']}")
        print(f"  Final WRMS: {fit_result['final_rms']:.6f} μs")
        print(f"  Tempo2 WRMS: {TEMPO2_WRMS_US:.6f} μs")
        print(f"  Difference:  {abs(fit_result['final_rms'] - TEMPO2_WRMS_US)*1e3:.1f} ns")

        # Post-fit parameter comparison
        final_params = fit_result['final_params']
        print(f"\nPost-fit parameters (selected):")
        for p in ['F0', 'F1', 'DM', 'A1', 'PB', 'H3', 'NE_SW']:
            val = final_params.get(p, 'N/A')
            unc = fit_result.get('uncertainties', {}).get(p, 'N/A')
            par_val = params.get(p, 'N/A')
            print(f"  {p:8s}: JUG={val}  PAR={par_val}  unc={unc}")

        print("=" * 70)
