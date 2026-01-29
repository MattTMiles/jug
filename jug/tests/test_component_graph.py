"""
Component Graph Tests
=====================

Tests that verify component graph wrappers produce bit-for-bit
identical results to direct derivative function calls.

Key requirements:
- SpinComponent.compute_derivatives == compute_spin_derivatives
- DispersionComponent.compute_derivatives == compute_dm_derivatives
- Parameter filtering works correctly
- Interface is consistent

Run with:
    pytest jug/tests/test_component_graph.py -v
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

from jug.model.components.base import TimingComponent
from jug.model.components.spin import SpinComponent
from jug.model.components.dispersion import DispersionComponent
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.derivatives_dm import compute_dm_derivatives


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


class TestComponentProtocol:
    """Tests for component protocol compliance."""

    def test_spin_component_implements_protocol(self):
        """Test: SpinComponent implements TimingComponent protocol."""
        component = SpinComponent()
        assert isinstance(component, TimingComponent)

    def test_dispersion_component_implements_protocol(self):
        """Test: DispersionComponent implements TimingComponent protocol."""
        component = DispersionComponent()
        assert isinstance(component, TimingComponent)

    def test_spin_provides_params(self):
        """Test: SpinComponent.provides_params returns spin parameters."""
        component = SpinComponent()
        params = component.provides_params()

        assert 'F0' in params
        assert 'F1' in params
        assert 'F2' in params
        assert 'DM' not in params

    def test_dispersion_provides_params(self):
        """Test: DispersionComponent.provides_params returns DM parameters."""
        component = DispersionComponent()
        params = component.provides_params()

        assert 'DM' in params
        assert 'DM1' in params
        assert 'DM2' in params
        assert 'F0' not in params


class TestSpinComponentEquivalence:
    """Tests that SpinComponent produces bit-for-bit identical results."""

    def test_spin_derivatives_equivalence(self):
        """
        Test: SpinComponent.compute_derivatives == compute_spin_derivatives

        This is the critical bit-for-bit equivalence test.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, _ = _load_test_data()
        fit_params = ['F0', 'F1']

        # Direct call
        direct_result = compute_spin_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
        )

        # Via component
        component = SpinComponent()
        component_result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
        )

        # BIT-FOR-BIT comparison
        for param in fit_params:
            assert param in direct_result, f"Direct result missing {param}"
            assert param in component_result, f"Component result missing {param}"

            assert np.array_equal(direct_result[param], component_result[param]), \
                f"Spin derivative for {param} differs between direct and component calls"

    def test_spin_filter_removes_non_spin(self):
        """Test: SpinComponent filters out non-spin parameters."""
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, _ = _load_test_data()

        # Mix of spin and non-spin
        fit_params = ['F0', 'F1', 'DM', 'RAJ']

        component = SpinComponent()
        result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
        )

        # Should only have spin params
        assert 'F0' in result
        assert 'F1' in result
        assert 'DM' not in result
        assert 'RAJ' not in result

    def test_spin_empty_fit_params(self):
        """Test: SpinComponent with no spin params returns empty dict."""
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, _ = _load_test_data()

        component = SpinComponent()
        result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=['DM', 'RAJ'],  # No spin params
        )

        assert result == {}


class TestDispersionComponentEquivalence:
    """Tests that DispersionComponent produces bit-for-bit identical results."""

    def test_dm_derivatives_equivalence(self):
        """
        Test: DispersionComponent.compute_derivatives == compute_dm_derivatives

        This is the critical bit-for-bit equivalence test.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, freq_mhz = _load_test_data()
        fit_params = ['DM']

        # Direct call
        direct_result = compute_dm_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            freq_mhz=freq_mhz,
            fit_params=fit_params,
        )

        # Via component
        component = DispersionComponent()
        component_result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
            freq_mhz=freq_mhz,
        )

        # BIT-FOR-BIT comparison
        for param in fit_params:
            assert param in direct_result, f"Direct result missing {param}"
            assert param in component_result, f"Component result missing {param}"

            assert np.array_equal(direct_result[param], component_result[param]), \
                f"DM derivative for {param} differs between direct and component calls"

    def test_dm_filter_removes_non_dm(self):
        """Test: DispersionComponent filters out non-DM parameters."""
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, freq_mhz = _load_test_data()

        # Mix of DM and non-DM
        fit_params = ['DM', 'DM1', 'F0', 'RAJ']

        component = DispersionComponent()
        result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,
            freq_mhz=freq_mhz,
        )

        # Should only have DM params
        assert 'DM' in result
        # DM1 may or may not be present depending on implementation
        assert 'F0' not in result
        assert 'RAJ' not in result

    def test_dm_empty_fit_params(self):
        """Test: DispersionComponent with no DM params returns empty dict."""
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, freq_mhz = _load_test_data()

        component = DispersionComponent()
        result = component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=['F0', 'RAJ'],  # No DM params
            freq_mhz=freq_mhz,
        )

        assert result == {}

    def test_dm_requires_freq(self):
        """Test: DispersionComponent raises error if freq_mhz missing."""
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, _ = _load_test_data()

        component = DispersionComponent()

        try:
            component.compute_derivatives(
                params=params,
                toas_mjd=toas_mjd,
                fit_params=['DM'],
                # freq_mhz not provided
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert 'freq_mhz' in str(e).lower()


class TestCombinedComponentRouting:
    """Tests for routing through multiple components."""

    def test_combined_derivatives_equivalence(self):
        """
        Test: Combined component routing matches direct calls.

        Simulates how the fitter would route through components.
        """
        if _skip_if_no_data():
            print("SKIP: Test data not found")
            return

        params, toas_mjd, freq_mhz = _load_test_data()
        fit_params = ['F0', 'F1', 'DM']

        # Direct calls
        spin_direct = compute_spin_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=['F0', 'F1'],
        )
        dm_direct = compute_dm_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            freq_mhz=freq_mhz,
            fit_params=['DM'],
        )

        # Via components
        spin_component = SpinComponent()
        dm_component = DispersionComponent()

        spin_routed = spin_component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,  # Pass full list
        )
        dm_routed = dm_component.compute_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=fit_params,  # Pass full list
            freq_mhz=freq_mhz,
        )

        # Combine results (as fitter would)
        combined = {}
        combined.update(spin_routed)
        combined.update(dm_routed)

        # BIT-FOR-BIT comparison
        for param in ['F0', 'F1']:
            assert np.array_equal(spin_direct[param], combined[param]), \
                f"Combined routing differs for {param}"

        assert np.array_equal(dm_direct['DM'], combined['DM']), \
            "Combined routing differs for DM"


def run_all_tests():
    """Run all component graph tests."""
    print("=" * 80)
    print("Component Graph Test Suite")
    print("=" * 80)

    if _skip_if_no_data():
        print("\nSKIP: Test data not found")
        print(f"Expected: {PAR_FILE}")
        print(f"Expected: {TIM_FILE}")
        return

    tests = [
        TestComponentProtocol(),
        TestSpinComponentEquivalence(),
        TestDispersionComponentEquivalence(),
        TestCombinedComponentRouting(),
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
