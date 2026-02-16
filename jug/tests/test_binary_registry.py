#!/usr/bin/env python3
"""
Test the binary model registry.

Verifies that:
1. Built-in models are registered correctly
2. Routing works for all registered models
3. Error handling works for unknown models
4. Non-binary pulsars return zero delay
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_builtin_models_registered():
    """Test that all built-in binary models are registered."""
    from jug.fitting.binary_registry import list_registered_models, is_model_registered

    registered = list_registered_models()
    print(f"\nRegistered binary models: {registered}")

    # DD family
    for model in ['DD', 'DDK', 'DDS', 'DDH', 'DDGR']:
        assert is_model_registered(model), f"{model} not registered"

    # BT family
    for model in ['BT', 'BTX']:
        assert is_model_registered(model), f"{model} not registered"

    # ELL1 family
    for model in ['ELL1', 'ELL1H', 'ELL1K']:
        assert is_model_registered(model), f"{model} not registered"

    # T2 (universal model)
    assert is_model_registered('T2'), "T2 not registered"

    print("  ✓ All expected models are registered")


def test_compute_binary_delay_routing():
    """Test that compute_binary_delay routes to correct model."""
    from jug.fitting.binary_registry import compute_binary_delay

    # Create test TOAs
    toas = np.array([55000.0, 55001.0, 55002.0])

    # DD model params
    dd_params = {
        'BINARY': 'DD',
        'A1': 10.0,
        'PB': 1.0,
        'T0': 55000.0,
        'ECC': 0.1,
        'OM': 45.0,
        'PBDOT': 0.0,
        'GAMMA': 0.0,
        'SINI': 0.8,
        'M2': 0.3,
    }

    # ELL1 model params
    ell1_params = {
        'BINARY': 'ELL1',
        'A1': 10.0,
        'PB': 1.0,
        'TASC': 55000.0,
        'EPS1': 0.01,
        'EPS2': 0.01,
        'PBDOT': 0.0,
        'A1DOT': 0.0,
        'SINI': 0.8,
        'M2': 0.3,
    }

    # Test DD
    dd_delay = compute_binary_delay(toas, dd_params)
    assert len(dd_delay) == 3, "DD delay wrong shape"
    assert not np.allclose(dd_delay, 0), "DD delay should be non-zero"
    print(f"  DD delay range: {dd_delay.min():.6f} to {dd_delay.max():.6f} s")

    # Test ELL1
    ell1_delay = compute_binary_delay(toas, ell1_params)
    assert len(ell1_delay) == 3, "ELL1 delay wrong shape"
    assert not np.allclose(ell1_delay, 0), "ELL1 delay should be non-zero"
    print(f"  ELL1 delay range: {ell1_delay.min():.6f} to {ell1_delay.max():.6f} s")

    # Test case insensitivity
    dd_params_lower = {**dd_params, 'BINARY': 'dd'}
    dd_delay_lower = compute_binary_delay(toas, dd_params_lower)
    np.testing.assert_array_equal(dd_delay, dd_delay_lower)
    print("  ✓ Case-insensitive model names work")

    print("  ✓ Binary delay routing works correctly")


def test_t2_delay_routing_dd_style():
    """Test T2 routes to DD when T0/ECC/OM params are present."""
    from jug.fitting.binary_registry import compute_binary_delay

    toas = np.array([55000.0, 55001.0, 55002.0])

    # DD-style T2 params (has T0, ECC, OM → should dispatch to DD)
    t2_dd_params = {
        'BINARY': 'T2',
        'A1': 10.0,
        'PB': 1.0,
        'T0': 55000.0,
        'ECC': 0.1,
        'OM': 45.0,
        'PBDOT': 0.0,
        'GAMMA': 0.0,
        'SINI': 0.8,
        'M2': 0.3,
    }

    # Equivalent DD params
    dd_params = {**t2_dd_params, 'BINARY': 'DD'}

    t2_delay = compute_binary_delay(toas, t2_dd_params)
    dd_delay = compute_binary_delay(toas, dd_params)

    assert len(t2_delay) == 3, "T2 delay wrong shape"
    assert not np.allclose(t2_delay, 0), "T2 delay should be non-zero"
    np.testing.assert_allclose(t2_delay, dd_delay, rtol=1e-15,
                               err_msg="T2 (DD-style) should match DD exactly")
    print(f"  T2(DD) delay range: {t2_delay.min():.6f} to {t2_delay.max():.6f} s")
    print("  ✓ T2 with T0/ECC/OM matches DD")


def test_t2_delay_routing_ell1_style():
    """Test T2 routes to ELL1 when TASC/EPS1/EPS2 params are present."""
    from jug.fitting.binary_registry import compute_binary_delay

    toas = np.array([55000.0, 55001.0, 55002.0])

    # ELL1-style T2 params (has TASC, EPS1, EPS2 → should dispatch to ELL1)
    t2_ell1_params = {
        'BINARY': 'T2',
        'A1': 10.0,
        'PB': 1.0,
        'TASC': 55000.0,
        'EPS1': 0.01,
        'EPS2': 0.01,
        'PBDOT': 0.0,
        'A1DOT': 0.0,
        'SINI': 0.8,
        'M2': 0.3,
    }

    # Equivalent ELL1 params
    ell1_params = {**t2_ell1_params, 'BINARY': 'ELL1'}

    t2_delay = compute_binary_delay(toas, t2_ell1_params)
    ell1_delay = compute_binary_delay(toas, ell1_params)

    assert len(t2_delay) == 3, "T2 delay wrong shape"
    assert not np.allclose(t2_delay, 0), "T2 delay should be non-zero"
    np.testing.assert_allclose(t2_delay, ell1_delay, rtol=1e-15,
                               err_msg="T2 (ELL1-style) should match ELL1 exactly")
    print(f"  T2(ELL1) delay range: {t2_delay.min():.6f} to {t2_delay.max():.6f} s")
    print("  ✓ T2 with TASC/EPS1/EPS2 matches ELL1")


def test_non_binary_returns_zeros():
    """Test that non-binary pulsars return zero delay."""
    from jug.fitting.binary_registry import compute_binary_delay

    toas = np.array([55000.0, 55001.0, 55002.0])

    # No BINARY parameter
    params_no_binary = {'F0': 100.0}
    delay = compute_binary_delay(toas, params_no_binary)
    np.testing.assert_array_equal(delay, np.zeros(3))

    # Empty BINARY parameter
    params_empty = {'BINARY': '', 'F0': 100.0}
    delay = compute_binary_delay(toas, params_empty)
    np.testing.assert_array_equal(delay, np.zeros(3))

    print("  ✓ Non-binary pulsars return zero delay")


def test_unknown_model_raises():
    """Test that unknown binary models raise ValueError."""
    from jug.fitting.binary_registry import compute_binary_delay

    toas = np.array([55000.0, 55001.0])
    params = {'BINARY': 'UNKNOWN_MODEL', 'A1': 1.0}

    try:
        compute_binary_delay(toas, params)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'UNKNOWN_MODEL' in str(e)
        assert 'Registered models' in str(e)
        print(f"  ✓ Unknown model raises helpful error: {e}")


def test_derivatives_routing():
    """Test that compute_binary_derivatives routes correctly."""
    from jug.fitting.binary_registry import compute_binary_derivatives

    toas = np.array([55000.0, 55001.0, 55002.0])

    # DD model
    dd_params = {
        'BINARY': 'DD',
        'F0': 100.0,
        'A1': 10.0,
        'PB': 1.0,
        'T0': 55000.0,
        'ECC': 0.1,
        'OM': 45.0,
    }

    derivs = compute_binary_derivatives(dd_params, toas, ['A1', 'PB'])
    assert 'A1' in derivs, "Missing A1 derivative"
    assert 'PB' in derivs, "Missing PB derivative"
    assert len(derivs['A1']) == 3, "Wrong derivative shape"
    print(f"  DD derivatives computed for: {list(derivs.keys())}")

    # ELL1 model
    ell1_params = {
        'BINARY': 'ELL1',
        'F0': 100.0,
        'A1': 10.0,
        'PB': 1.0,
        'TASC': 55000.0,
        'EPS1': 0.01,
        'EPS2': 0.01,
    }

    derivs = compute_binary_derivatives(ell1_params, toas, ['A1', 'PB'])
    assert 'A1' in derivs
    assert 'PB' in derivs
    print(f"  ELL1 derivatives computed for: {list(derivs.keys())}")

    print("  ✓ Binary derivatives routing works correctly")


def test_t2_derivatives_routing():
    """Test that T2 derivatives dispatch correctly based on parameterization."""
    from jug.fitting.binary_registry import compute_binary_derivatives

    toas = np.array([55000.0, 55001.0, 55002.0])

    # DD-style T2
    t2_dd_params = {
        'BINARY': 'T2',
        'F0': 100.0,
        'A1': 10.0,
        'PB': 1.0,
        'T0': 55000.0,
        'ECC': 0.1,
        'OM': 45.0,
    }

    derivs_dd = compute_binary_derivatives(t2_dd_params, toas, ['A1', 'PB'])
    assert 'A1' in derivs_dd, "Missing A1 derivative for T2(DD)"
    assert 'PB' in derivs_dd, "Missing PB derivative for T2(DD)"
    assert len(derivs_dd['A1']) == 3, "Wrong derivative shape"
    print(f"  T2(DD) derivatives computed for: {list(derivs_dd.keys())}")

    # ELL1-style T2
    t2_ell1_params = {
        'BINARY': 'T2',
        'F0': 100.0,
        'A1': 10.0,
        'PB': 1.0,
        'TASC': 55000.0,
        'EPS1': 0.01,
        'EPS2': 0.01,
    }

    derivs_ell1 = compute_binary_derivatives(t2_ell1_params, toas, ['A1', 'PB'])
    assert 'A1' in derivs_ell1, "Missing A1 derivative for T2(ELL1)"
    assert 'PB' in derivs_ell1, "Missing PB derivative for T2(ELL1)"
    print(f"  T2(ELL1) derivatives computed for: {list(derivs_ell1.keys())}")

    # Verify T2(DD) matches DD exactly
    dd_params = {**t2_dd_params, 'BINARY': 'DD'}
    derivs_dd_direct = compute_binary_derivatives(dd_params, toas, ['A1', 'PB'])
    np.testing.assert_allclose(derivs_dd['A1'], derivs_dd_direct['A1'], rtol=1e-15)
    np.testing.assert_allclose(derivs_dd['PB'], derivs_dd_direct['PB'], rtol=1e-15)
    print("  ✓ T2(DD) derivatives match DD exactly")

    # Verify T2(ELL1) matches ELL1 exactly
    ell1_params = {**t2_ell1_params, 'BINARY': 'ELL1'}
    derivs_ell1_direct = compute_binary_derivatives(ell1_params, toas, ['A1', 'PB'])
    np.testing.assert_allclose(derivs_ell1['A1'], derivs_ell1_direct['A1'], rtol=1e-15)
    np.testing.assert_allclose(derivs_ell1['PB'], derivs_ell1_direct['PB'], rtol=1e-15)
    print("  ✓ T2(ELL1) derivatives match ELL1 exactly")

    print("  ✓ T2 derivatives routing works correctly")


if __name__ == "__main__":
    print("=" * 70)
    print("Binary Model Registry Tests")
    print("=" * 70)

    all_passed = True
    tests = [
        test_builtin_models_registered,
        test_compute_binary_delay_routing,
        test_t2_delay_routing_dd_style,
        test_t2_delay_routing_ell1_style,
        test_non_binary_returns_zeros,
        test_unknown_model_raises,
        test_derivatives_routing,
        test_t2_derivatives_routing,
    ]

    for test in tests:
        try:
            all_passed &= test()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
