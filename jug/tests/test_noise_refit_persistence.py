"""
Regression tests for noise refit param persistence.

Verifies that fitted parameters persist across noise configuration changes.
The starting parameters for a fit should be the current session params,
not the original par file values.
"""

import pytest
import numpy as np
from pathlib import Path
from jug.engine.session import TimingSession
from jug.engine.noise_mode import NoiseConfig


@pytest.fixture
def j1909_session():
    """Create a timing session for J1909-3744."""
    par_file = Path("data/pulsars/MPTA_data/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/MPTA_data/J1909-3744.tim")

    if not par_file.exists() or not tim_file.exists():
        pytest.skip("J1909 data files not available")

    return TimingSession(par_file, tim_file, verbose=False)


class TestNoiseRefitPersistence:
    """Test that params persist across noise config changes."""

    def test_params_persist_after_noise_enable(self, j1909_session):
        """Fitted params should persist when noise is enabled for refit."""
        session = j1909_session

        # Store original params
        original_F0 = session.params['F0']
        original_F1 = session.params['F1']
        original_DM = session.params['DM']

        # Fit 1: No noise
        noise_off = NoiseConfig()
        noise_off.disable_all()

        result1 = session.fit_parameters(
            fit_params=['F0', 'F1', 'DM'],
            max_iter=10,
            noise_config=noise_off
        )

        fitted_F0_1 = result1['final_params']['F0']
        fitted_F1_1 = result1['final_params']['F1']
        fitted_DM_1 = result1['final_params']['DM']

        # Verify session params updated
        assert abs(session.params['F0'] - fitted_F0_1) < 1e-15
        assert abs(session.params['F1'] - fitted_F1_1) < 1e-25
        assert abs(session.params['DM'] - fitted_DM_1) < 1e-10

        # Fit 2: Enable noise and refit
        noise_on = NoiseConfig.from_par(session.params)
        noise_on.enable('EFAC')
        noise_on.enable('EQUAD')

        result2 = session.fit_parameters(
            fit_params=['F0', 'F1', 'DM'],
            max_iter=10,
            noise_config=noise_on
        )

        fitted_F0_2 = result2['final_params']['F0']
        fitted_F1_2 = result2['final_params']['F1']
        fitted_DM_2 = result2['final_params']['DM']

        # Key assertion: Fit 2 should have started from fitted_*_1, not original_*
        # If params were reset, fitted_*_2 would be close to original_*
        # If params persisted, fitted_*_2 would be close to fitted_*_1

        diff_F0_from_original = abs(fitted_F0_2 - original_F0)
        diff_F0_from_fit1 = abs(fitted_F0_2 - fitted_F0_1)

        # Fit 2 result should be closer to fit 1 than to original
        # (unless noise causes a large change, but for J1909 it shouldn't)
        assert diff_F0_from_fit1 < diff_F0_from_original * 0.5, \
            f"Fit 2 appears to have started from ORIGINAL params instead of fitted params from fit 1"

        # Session params should be updated to fit 2 results
        assert abs(session.params['F0'] - fitted_F0_2) < 1e-15

    def test_params_persist_after_noise_disable(self, j1909_session):
        """Fitted params should persist when noise is disabled for refit."""
        session = j1909_session

        original_F0 = session.params['F0']

        # Fit 1: With noise
        noise_on = NoiseConfig.from_par(session.params)
        noise_on.enable('EFAC')
        noise_on.enable('EQUAD')

        result1 = session.fit_parameters(
            fit_params=['F0', 'F1', 'DM'],
            max_iter=10,
            noise_config=noise_on
        )

        fitted_F0_1 = result1['final_params']['F0']

        # Fit 2: Disable noise and refit
        noise_off = NoiseConfig()
        noise_off.disable_all()

        result2 = session.fit_parameters(
            fit_params=['F0', 'F1', 'DM'],
            max_iter=10,
            noise_config=noise_off
        )

        fitted_F0_2 = result2['final_params']['F0']

        # Key assertion: Fit 2 should have started from fitted_F0_1
        diff_from_original = abs(fitted_F0_2 - original_F0)
        diff_from_fit1 = abs(fitted_F0_2 - fitted_F0_1)

        assert diff_from_fit1 < diff_from_original * 0.5, \
            f"Fit 2 reverted to original params instead of continuing from fit 1"

    def test_multiple_noise_toggles(self, j1909_session):
        """Params should persist through multiple noise config changes."""
        session = j1909_session

        original_F0 = session.params['F0']
        fit_params = ['F0', 'F1', 'DM']
        F0_history = [original_F0]

        # Scenario: no noise → EFAC → EFAC+RedNoise → RedNoise → no noise
        configs = [
            ('no_noise', NoiseConfig()),
            ('EFAC', NoiseConfig.from_par(session.params)),
            ('EFAC+RedNoise', NoiseConfig.from_par(session.params)),
            ('RedNoise', NoiseConfig.from_par(session.params)),
            ('no_noise_2', NoiseConfig()),
        ]

        configs[0][1].disable_all()
        configs[1][1].enable('EFAC')
        configs[2][1].enable('EFAC')
        configs[2][1].enable('RedNoise')
        configs[3][1].enable('RedNoise')
        configs[4][1].disable_all()

        for name, noise_config in configs:
            result = session.fit_parameters(
                fit_params=fit_params,
                max_iter=10,
                noise_config=noise_config
            )

            fitted_F0 = result['final_params']['F0']
            F0_history.append(fitted_F0)

            # Verify session params updated
            assert abs(session.params['F0'] - fitted_F0) < 1e-15, \
                f"Session params not updated after fit with {name}"

        # Final F0 should be close to previous fit, not original
        final_F0 = F0_history[-1]
        prev_F0 = F0_history[-2]

        diff_from_prev = abs(final_F0 - prev_F0)
        diff_from_original = abs(final_F0 - original_F0)

        # Some configs may cause larger changes, but generally should be
        # continuing from previous fit
        assert diff_from_prev < diff_from_original, \
            f"Final fit reverted to original instead of continuing from previous"

    def test_session_params_updated_after_fit(self, j1909_session):
        """Session params should be updated with fitted values."""
        session = j1909_session

        original_F0 = session.params['F0']

        # Fit
        result = session.fit_parameters(
            fit_params=['F0', 'F1'],
            max_iter=5,
            noise_config=NoiseConfig()
        )

        fitted_F0 = result['final_params']['F0']
        fitted_F1 = result['final_params']['F1']

        # Session params should match fitted params
        assert abs(session.params['F0'] - fitted_F0) < 1e-15
        assert abs(session.params['F1'] - fitted_F1) < 1e-25

        # And should have changed from original (even if tiny)
        assert abs(fitted_F0 - original_F0) > 1e-14, \
            "F0 should have changed from original (otherwise test is too weak)"


class TestNoiseConfigIndependence:
    """Test that noise config changes don't affect param state."""

    def test_noise_toggle_doesnt_reset_params(self, j1909_session):
        """Simply changing noise config should not reset params."""
        session = j1909_session

        # Fit once
        result = session.fit_parameters(
            fit_params=['F0', 'F1'],
            max_iter=5,
            noise_config=NoiseConfig()
        )

        fitted_F0 = result['final_params']['F0']
        fitted_F1 = result['final_params']['F1']

        # Create a different noise config (without fitting)
        noise_on = NoiseConfig.from_par(session.params)
        noise_on.enable('EFAC')

        # Session params should still be the fitted values
        assert abs(session.params['F0'] - fitted_F0) < 1e-15
        assert abs(session.params['F1'] - fitted_F1) < 1e-25


class TestCacheInvalidation:
    """Test that residual cache is managed correctly."""

    def test_cache_cleared_after_fit(self, j1909_session):
        """Residual cache should be cleared after fit (params changed)."""
        session = j1909_session

        # Compute residuals to populate cache
        session.compute_residuals(subtract_tzr=False)
        assert len(session._cached_result_by_mode) > 0

        # Fit
        session.fit_parameters(
            fit_params=['F0'],
            max_iter=5,
            noise_config=NoiseConfig()
        )

        # Cache should be cleared
        assert len(session._cached_result_by_mode) == 0, \
            "Residual cache should be cleared after fit"

    def test_cache_preserved_on_noise_change_only(self, j1909_session):
        """Residual cache should NOT be cleared if only noise config changes."""
        session = j1909_session

        # Compute residuals to populate cache
        session.compute_residuals(subtract_tzr=False)
        cache_size_before = len(session._cached_result_by_mode)
        assert cache_size_before > 0

        # Create a different noise config (without fitting or computing)
        noise_on = NoiseConfig.from_par(session.params)
        noise_on.enable('EFAC')

        # Cache should still be present (no fit happened)
        cache_size_after = len(session._cached_result_by_mode)
        assert cache_size_after == cache_size_before, \
            "Residual cache should not be cleared by noise config object creation"
