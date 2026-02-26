"""
ParameterSpec System Tests
==========================

Tests for the ParameterSpec registry and helper functions.

Run with:
    pytest jug/tests/test_parameter_spec.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.model.parameter_spec import (
    ParameterSpec,
    DerivativeGroup,
    PARAMETER_REGISTRY,
    get_spec,
    canonicalize_param_name,
    list_params_by_group,
    list_params_by_derivative_group,
    list_fittable_params,
    get_derivative_group,
    is_spin_param,
    is_dm_param,
    is_astrometry_param,
    is_binary_param,
    get_spin_params_from_list,
    get_dm_params_from_list,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_spec_is_frozen(self):
        """Test: ParameterSpec is immutable."""
        spec = get_spec('F0')
        assert spec is not None

        try:
            spec.name = 'F99'
            assert False, "Should not be able to modify frozen dataclass"
        except AttributeError:
            pass  # Expected

    def test_spec_fields(self):
        """Test: ParameterSpec has required fields."""
        spec = get_spec('F0')

        assert spec.name == 'F0'
        assert spec.group == 'spin'
        assert spec.derivative_group == DerivativeGroup.SPIN
        assert spec.internal_unit == 'Hz'
        assert 'NU' in spec.aliases

    def test_spec_derivative_group(self):
        """Test: DerivativeGroup enum values."""
        assert DerivativeGroup.SPIN.name == 'SPIN'
        assert DerivativeGroup.DM.name == 'DM'
        assert DerivativeGroup.ASTROMETRY.name == 'ASTROMETRY'
        assert DerivativeGroup.BINARY.name == 'BINARY'


class TestParameterRegistry:
    """Tests for PARAMETER_REGISTRY."""

    def test_registry_contains_spin_params(self):
        """Test: Registry contains spin parameters."""
        for param in ['F0', 'F1', 'F2', 'F3']:
            assert param in PARAMETER_REGISTRY

    def test_registry_contains_dm_params(self):
        """Test: Registry contains DM parameters."""
        for param in ['DM', 'DM1', 'DM2']:
            assert param in PARAMETER_REGISTRY

    def test_registry_contains_astrometry_params(self):
        """Test: Registry contains astrometry parameters."""
        for param in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
            assert param in PARAMETER_REGISTRY

    def test_registry_contains_binary_params(self):
        """Test: Registry contains binary parameters."""
        for param in ['PB', 'A1', 'ECC', 'OM', 'T0', 'TASC', 'EPS1', 'EPS2']:
            assert param in PARAMETER_REGISTRY

    def test_registry_contains_epochs(self):
        """Test: Registry contains epoch parameters."""
        for param in ['PEPOCH', 'DMEPOCH', 'POSEPOCH']:
            assert param in PARAMETER_REGISTRY


class TestAliasResolution:
    """Tests for canonicalize_param_name."""

    def test_spin_aliases(self):
        """Test: Spin parameter aliases resolve correctly."""
        assert canonicalize_param_name('NU') == 'F0'
        assert canonicalize_param_name('F') == 'F0'
        assert canonicalize_param_name('NUDOT') == 'F1'
        assert canonicalize_param_name('FDOT') == 'F1'

    def test_dm_aliases(self):
        """Test: DM parameter aliases resolve correctly."""
        assert canonicalize_param_name('DM0') == 'DM'

    def test_astrometry_aliases(self):
        """Test: Astrometry parameter aliases resolve correctly."""
        assert canonicalize_param_name('PMRAC') == 'PMRA'
        assert canonicalize_param_name('PARALLAX') == 'PX'

    def test_binary_aliases(self):
        """Test: Binary parameter aliases resolve correctly."""
        assert canonicalize_param_name('PORB') == 'PB'
        assert canonicalize_param_name('ASINI') == 'A1'
        assert canonicalize_param_name('E') == 'ECC'
        assert canonicalize_param_name('OMEGA') == 'OM'

    def test_canonical_unchanged(self):
        """Test: Canonical names pass through unchanged."""
        for param in ['F0', 'F1', 'DM', 'RAJ', 'PB']:
            assert canonicalize_param_name(param) == param

    def test_unknown_unchanged(self):
        """Test: Unknown names pass through unchanged."""
        assert canonicalize_param_name('UNKNOWN') == 'UNKNOWN'


class TestGetSpec:
    """Tests for get_spec function."""

    def test_get_spec_canonical(self):
        """Test: get_spec with canonical names."""
        spec = get_spec('F0')
        assert spec is not None
        assert spec.name == 'F0'

    def test_get_spec_alias(self):
        """Test: get_spec resolves aliases."""
        spec = get_spec('NU')
        assert spec is not None
        assert spec.name == 'F0'

    def test_get_spec_unknown(self):
        """Test: get_spec returns None for unknown params."""
        spec = get_spec('UNKNOWN')
        assert spec is None


class TestDerivativeGroupRouting:
    """Tests for derivative group routing functions."""

    def test_get_derivative_group_spin(self):
        """Test: Spin params route to SPIN group."""
        for param in ['F0', 'F1', 'F2', 'F3']:
            assert get_derivative_group(param) == DerivativeGroup.SPIN

    def test_get_derivative_group_dm(self):
        """Test: DM params route to DM group."""
        for param in ['DM', 'DM1', 'DM2']:
            assert get_derivative_group(param) == DerivativeGroup.DM

    def test_get_derivative_group_astrometry(self):
        """Test: Astrometry params route to ASTROMETRY group."""
        for param in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
            assert get_derivative_group(param) == DerivativeGroup.ASTROMETRY

    def test_get_derivative_group_binary(self):
        """Test: Binary params route to BINARY group."""
        for param in ['PB', 'A1', 'ECC', 'OM', 'T0']:
            assert get_derivative_group(param) == DerivativeGroup.BINARY

    def test_get_derivative_group_epoch(self):
        """Test: Epoch params route to EPOCH group."""
        for param in ['PEPOCH', 'DMEPOCH', 'POSEPOCH']:
            assert get_derivative_group(param) == DerivativeGroup.EPOCH


class TestParameterTypeChecks:
    """Tests for is_X_param functions."""

    def test_is_spin_param(self):
        """Test: is_spin_param correctly identifies spin params."""
        assert is_spin_param('F0') is True
        assert is_spin_param('F1') is True
        assert is_spin_param('DM') is False
        assert is_spin_param('RAJ') is False
        assert is_spin_param('UNKNOWN') is False

    def test_is_dm_param(self):
        """Test: is_dm_param correctly identifies DM params."""
        assert is_dm_param('DM') is True
        assert is_dm_param('DM1') is True
        assert is_dm_param('F0') is False
        assert is_dm_param('RAJ') is False

    def test_is_astrometry_param(self):
        """Test: is_astrometry_param correctly identifies astrometry params."""
        assert is_astrometry_param('RAJ') is True
        assert is_astrometry_param('DECJ') is True
        assert is_astrometry_param('PMRA') is True
        assert is_astrometry_param('F0') is False

    def test_is_binary_param(self):
        """Test: is_binary_param correctly identifies binary params."""
        assert is_binary_param('PB') is True
        assert is_binary_param('A1') is True
        assert is_binary_param('F0') is False


class TestParameterFiltering:
    """Tests for parameter filtering functions."""

    def test_get_spin_params_from_list(self):
        """Test: get_spin_params_from_list filters correctly."""
        params = ['F0', 'F1', 'DM', 'RAJ', 'PB']
        result = get_spin_params_from_list(params)
        assert result == ['F0', 'F1']

    def test_get_dm_params_from_list(self):
        """Test: get_dm_params_from_list filters correctly."""
        params = ['F0', 'DM', 'DM1', 'RAJ', 'PB']
        result = get_dm_params_from_list(params)
        assert result == ['DM', 'DM1']

    def test_filter_empty_list(self):
        """Test: Filtering empty list returns empty list."""
        assert get_spin_params_from_list([]) == []
        assert get_dm_params_from_list([]) == []

    def test_filter_no_matches(self):
        """Test: Filtering list with no matches returns empty list."""
        params = ['RAJ', 'DECJ', 'PB']
        assert get_spin_params_from_list(params) == []
        assert get_dm_params_from_list(params) == []


class TestListingFunctions:
    """Tests for listing functions."""

    def test_list_params_by_group(self):
        """Test: list_params_by_group returns correct params."""
        spin = list_params_by_group('spin')
        assert 'F0' in spin
        assert 'F1' in spin
        assert 'DM' not in spin

        dm = list_params_by_group('dm')
        assert 'DM' in dm
        assert 'DM1' in dm
        assert 'F0' not in dm

    def test_list_params_by_derivative_group(self):
        """Test: list_params_by_derivative_group returns correct params."""
        spin = list_params_by_derivative_group(DerivativeGroup.SPIN)
        assert 'F0' in spin
        assert 'DM' not in spin

    def test_list_fittable_params(self):
        """Test: list_fittable_params excludes epochs."""
        fittable = list_fittable_params()

        # Should include these
        assert 'F0' in fittable
        assert 'DM' in fittable
        assert 'RAJ' in fittable
        assert 'PB' in fittable

        # Should NOT include epochs
        assert 'PEPOCH' not in fittable
        assert 'DMEPOCH' not in fittable
        assert 'POSEPOCH' not in fittable


class TestSpecAttributes:
    """Tests for specific spec attributes."""

    def test_angle_params_use_radians(self):
        """Test: Angle parameters have radians as internal unit."""
        raj = get_spec('RAJ')
        assert raj.internal_unit == 'rad'

        decj = get_spec('DECJ')
        assert decj.internal_unit == 'rad'

        om = get_spec('OM')
        assert om.internal_unit == 'rad'

    def test_angle_params_have_codecs(self):
        """Test: Angle parameters specify appropriate codecs."""
        raj = get_spec('RAJ')
        assert raj.par_codec_name == 'raj'

        decj = get_spec('DECJ')
        assert decj.par_codec_name == 'decj'

    def test_epoch_params_have_codec(self):
        """Test: Epoch parameters use epoch_mjd codec."""
        pepoch = get_spec('PEPOCH')
        assert pepoch.par_codec_name == 'epoch_mjd'


def run_all_tests():
    """Run all parameter spec tests."""
    print("="*80)
    print("ParameterSpec Test Suite")
    print("="*80)

    tests = [
        TestParameterSpec(),
        TestParameterRegistry(),
        TestAliasResolution(),
        TestGetSpec(),
        TestDerivativeGroupRouting(),
        TestParameterTypeChecks(),
        TestParameterFiltering(),
        TestListingFunctions(),
        TestSpecAttributes(),
    ]

    for test_class in tests:
        class_name = test_class.__class__.__name__
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"[x] {class_name}.{method_name}")
                except AssertionError as e:
                    print(f"[ ] {class_name}.{method_name}: {e}")
                except Exception as e:
                    print(f"[ ] {class_name}.{method_name}: {type(e).__name__}: {e}")

    print()
    print("All tests completed!")


if __name__ == '__main__':
    run_all_tests()
