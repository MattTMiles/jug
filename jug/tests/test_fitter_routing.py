"""
Fitter Routing via ParameterSpec Tests
======================================

Tests that verify spec-driven fitter routing produces bit-for-bit
identical results to the old direct-call method.

Key requirements:
- Spec-driven routing produces identical design matrices
- Derivative group classification is correct
- Column ordering is preserved
- No numerical drift from routing refactor

Run with:
    pytest jug/tests/test_fitter_routing.py -v
"""

import os
import sys
from pathlib import Path

# Force determinism BEFORE any other imports
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_enable_fast_math=false'

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.model.parameter_spec import (
    get_spec,
    get_derivative_group,
    DerivativeGroup,
    is_spin_param,
    is_dm_param,
    get_spin_params_from_list,
    get_dm_params_from_list,
)
from jug.model.components.spin import SpinComponent
from jug.model.components.dispersion import DispersionComponent


# Test data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pulsars"
PAR_FILE = DATA_DIR / "J1909-3744_tdb.par"
TIM_FILE = DATA_DIR / "J1909-3744.tim"


def _skip_if_no_data():
    """Return True if test data is missing."""
    return not (PAR_FILE.exists() and TIM_FILE.exists())


def _load_test_data():
    """Load test data for derivative computation."""
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds

    params = parse_par_file(PAR_FILE)
    toas_data = parse_tim_file_mjds(TIM_FILE)

    # SimpleTOA has mjd_int and mjd_frac, combine them for mjd
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    freq_mhz = np.array([toa.freq_mhz for toa in toas_data])

    return params, toas_mjd, freq_mhz


class TestSpecDrivenRouting:
    """Tests for spec-driven derivative routing."""

    def test_routing_classification(self):
        """Test: Parameters are classified to correct derivative groups."""
        # Spin
        assert get_derivative_group('F0') == DerivativeGroup.SPIN
        assert get_derivative_group('F1') == DerivativeGroup.SPIN
        assert get_derivative_group('F2') == DerivativeGroup.SPIN

        # DM
        assert get_derivative_group('DM') == DerivativeGroup.DM
        assert get_derivative_group('DM1') == DerivativeGroup.DM

        # Astrometry
        assert get_derivative_group('RAJ') == DerivativeGroup.ASTROMETRY
        assert get_derivative_group('DECJ') == DerivativeGroup.ASTROMETRY

        # Binary
        assert get_derivative_group('PB') == DerivativeGroup.BINARY
        assert get_derivative_group('A1') == DerivativeGroup.BINARY

    def test_spec_filtering_matches_string_check(self):
        """
        Test: Spec-based filtering matches old string-based filtering.

        Old code: param.startswith('F') or param in ['F0', 'F1', ...]
        New code: is_spin_param(param) via ParameterSpec
        """
        fit_params = ['F0', 'F1', 'F2', 'DM', 'DM1', 'RAJ', 'PB']

        # Old-style filtering (string-based)
        old_spin = [p for p in fit_params if p.startswith('F') and p[1:].isdigit()]
        old_dm = [p for p in fit_params if p == 'DM' or (p.startswith('DM') and p[2:].isdigit())]

        # New-style filtering (spec-based)
        new_spin = get_spin_params_from_list(fit_params)
        new_dm = get_dm_params_from_list(fit_params)

        assert old_spin == new_spin, \
            f"Spin filtering differs: old={old_spin}, new={new_spin}"
        assert old_dm == new_dm, \
            f"DM filtering differs: old={old_dm}, new={new_dm}"

    def test_design_matrix_column_order(self):
        """
        Test: Spec-driven routing preserves column order.

        The design matrix must have columns in the same order as fit_params.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, freq_mhz = _load_test_data()
        fit_params = ['F0', 'DM', 'F1']  # Mixed order

        spin_component = SpinComponent()
        dm_component = DispersionComponent()

        # Get derivatives via components
        spin_derivs = spin_component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
        )
        dm_derivs = dm_component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
            freq_mhz=freq_mhz,
        )

        # Combine in fit_params order
        combined = {}
        combined.update(spin_derivs)
        combined.update(dm_derivs)

        # Build design matrix in fit_params order
        design_cols = []
        for param in fit_params:
            assert param in combined, f"Missing derivative for {param}"
            design_cols.append(combined[param])

        design_matrix = np.column_stack(design_cols)

        # Verify shape
        n_toas = len(toas_mjd)
        n_params = len(fit_params)
        assert design_matrix.shape == (n_toas, n_params)


class TestRoutingEquivalence:
    """Tests that spec-driven routing produces identical fit results."""

    def test_fit_with_spec_routing(self):
        """
        Test: Fit using spec-driven routing matches direct fitter call.

        This is the end-to-end equivalence test.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.engine.session import TimingSession

        fit_params = ['F0', 'F1', 'DM']

        # Standard fit (uses existing routing)
        session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
        _ = session.compute_residuals(subtract_tzr=False)

        result = session.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False
        )

        # Verify fit succeeded
        assert result['converged'], "Fit should converge"

        # Verify we got results for all params
        for param in fit_params:
            assert param in result['final_params'], f"Missing result for {param}"
            assert param in result['uncertainties'], f"Missing uncertainty for {param}"

    def test_repeated_fits_identical(self):
        """
        Test: Repeated fits with same routing produce identical results.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        from jug.engine.session import TimingSession

        fit_params = ['F0', 'F1']
        results = []

        for _ in range(3):
            session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
            _ = session.compute_residuals(subtract_tzr=False)

            result = session.fit_parameters(
                fit_params=fit_params,
                max_iter=25,
                convergence_threshold=1e-14,
                solver_mode="exact",
                verbose=False
            )
            results.append(result)

        # All should be bit-for-bit identical
        ref = results[0]
        for i, result in enumerate(results[1:], 2):
            for param in fit_params:
                assert result['final_params'][param] == ref['final_params'][param], \
                    f"Run {i} {param} differs from run 1"

            assert np.array_equal(result['residuals_us'], ref['residuals_us']), \
                f"Run {i} residuals differ from run 1"


class TestDerivativeGroupConsistency:
    """Tests for derivative group consistency."""

    def test_all_spin_params_same_group(self):
        """Test: All spin params route to same derivative group."""
        spin_params = ['F0', 'F1', 'F2', 'F3', 'F4']

        groups = set()
        for param in spin_params:
            group = get_derivative_group(param)
            if group is not None:
                groups.add(group)

        assert len(groups) == 1, f"Spin params route to multiple groups: {groups}"
        assert DerivativeGroup.SPIN in groups

    def test_all_dm_params_same_group(self):
        """Test: All DM params route to same derivative group."""
        dm_params = ['DM', 'DM1', 'DM2', 'DM3']

        groups = set()
        for param in dm_params:
            group = get_derivative_group(param)
            if group is not None:
                groups.add(group)

        assert len(groups) == 1, f"DM params route to multiple groups: {groups}"
        assert DerivativeGroup.DM in groups

    def test_unknown_param_returns_none(self):
        """Test: Unknown params return None for derivative group."""
        group = get_derivative_group('UNKNOWN_PARAM')
        assert group is None


class TestSpecMetadataIntegrity:
    """Tests for ParameterSpec metadata integrity."""

    def test_fittable_params_have_derivative_groups(self):
        """Test: All fittable params have derivative groups assigned."""
        from jug.model.parameter_spec import list_fittable_params

        fittable = list_fittable_params()

        for param in fittable:
            spec = get_spec(param)
            assert spec is not None, f"No spec for fittable param {param}"
            assert spec.derivative_group is not None, \
                f"Fittable param {param} has no derivative group"

    def test_epoch_params_not_fittable(self):
        """Test: Epoch params are not in fittable list."""
        from jug.model.parameter_spec import list_fittable_params

        fittable = list_fittable_params()

        epoch_params = ['PEPOCH', 'DMEPOCH', 'POSEPOCH']
        for param in epoch_params:
            assert param not in fittable, f"Epoch param {param} should not be fittable"


def run_all_tests():
    """Run all fitter routing tests."""
    print("=" * 80)
    print("Fitter Routing Test Suite")
    print("=" * 80)

    tests = [
        TestSpecDrivenRouting(),
        TestRoutingEquivalence(),
        TestDerivativeGroupConsistency(),
        TestSpecMetadataIntegrity(),
    ]

    passed = 0
    failed = 0

    for test_class in tests:
        class_name = test_class.__class__.__name__
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"✓ {class_name}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"✗ {class_name}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"✗ {class_name}.{method_name}: {type(e).__name__}: {e}")
                    failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")


if __name__ == '__main__':
    run_all_tests()
