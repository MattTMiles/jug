"""Test script for BT binary model.

This script validates the BT binary model implementation against known
test cases.
"""

import numpy as np
# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax
import jax.numpy as jnp

from jug.delays.binary_bt import bt_binary_delay_vectorized


def test_bt_model():
    """Test BT binary model with synthetic parameters."""
    print("="*60)
    print("Testing BT/DD Binary Model")
    print("="*60)
    
    # Synthetic binary parameters (similar to a typical MSP binary)
    pb = 1.533449474  # days (orbital period)
    a1 = 1.8  # light-seconds (semi-major axis projection)
    ecc = 0.00001  # eccentricity (very small)
    om = 45.0  # degrees (longitude of periastron)
    t0 = 50000.0  # MJD (time of periastron)
    gamma = 0.0  # seconds (Einstein delay - often small)
    pbdot = 0.0  # dimensionless (period derivative)
    m2 = 0.2  # solar masses (companion mass)
    sini = 0.9  # sine of inclination
    omdot = 0.0  # deg/yr (periastron advance - DD only)
    xdot = 0.0  # light-sec/sec (A1 derivative)
    
    # Test times (MJD)
    t_test = jnp.array([50000.0, 50000.5, 50001.0, 50002.0])
    
    # Compute delays
    delays = bt_binary_delay_vectorized(
        t_test, pb, a1, ecc, om, t0, gamma, pbdot, m2, sini, omdot, xdot
    )
    
    print(f"\nTest times (MJD): {t_test}")
    print(f"Binary delays (seconds): {delays}")
    print(f"Binary delays (microseconds): {delays * 1e6}")
    print(f"\nDelay range: {np.min(delays) * 1e6:.3f} to {np.max(delays) * 1e6:.3f} mus")
    
    # Check that delays are reasonable (should be on order of a1 ~ 1.8 light-seconds)
    assert np.all(np.abs(delays) < 2.0 * a1), "Delays exceed maximum possible value!"
    print("[x] Delays are physically reasonable")


def test_t2_model_dd_style():
    """Test T2 model dispatches correctly with DD-style (T0/ECC/OM) parameters.
    
    T2 with Keplerian params should produce identical output to DD.
    """
    from jug.fitting.binary_registry import compute_binary_delay

    print("="*60)
    print("Testing T2 Binary Model (DD-style parameterization)")
    print("="*60)

    t_test = jnp.array([50000.0, 50000.25, 50000.5, 50000.75, 50001.0, 50002.0])

    params = {
        'A1': 1.8,
        'PB': 1.533449474,
        'T0': 50000.0,
        'ECC': 0.00001,
        'OM': 45.0,
        'GAMMA': 0.0,
        'PBDOT': 0.0,
        'SINI': 0.9,
        'M2': 0.2,
        'OMDOT': 0.0,
        'XDOT': 0.0,
        'EDOT': 0.0,
    }

    t2_params = {**params, 'BINARY': 'T2'}
    dd_params = {**params, 'BINARY': 'DD'}

    t2_delays = compute_binary_delay(t_test, t2_params)
    dd_delays = compute_binary_delay(t_test, dd_params)

    print(f"\nT2 delays (mus): {t2_delays * 1e6}")
    print(f"DD delays (mus): {dd_delays * 1e6}")
    print(f"Max difference: {np.max(np.abs(t2_delays - dd_delays)):.2e} s")

    assert np.all(np.abs(t2_delays) < 2.0 * 1.8), "T2 delays exceed physical maximum"
    np.testing.assert_allclose(t2_delays, dd_delays, rtol=1e-15,
                               err_msg="T2 (DD-style) does not match DD")
    print("[x] T2 DD-style matches DD exactly")


def test_t2_model_ell1_style():
    """Test T2 model dispatches correctly with ELL1-style (TASC/EPS1/EPS2) parameters.
    
    T2 with Laplace-Lagrange params should produce identical output to ELL1.
    """
    from jug.fitting.binary_registry import compute_binary_delay

    print("="*60)
    print("Testing T2 Binary Model (ELL1-style parameterization)")
    print("="*60)

    t_test = jnp.array([55000.0, 55000.25, 55000.5, 55000.75, 55001.0, 55002.0])

    params = {
        'A1': 0.025795,
        'PB': 0.14484,
        'TASC': 55000.0,
        'EPS1': 3.5e-06,
        'EPS2': -6.7e-07,
        'PBDOT': 0.0,
        'A1DOT': 0.0,
        'SINI': 0.0,
        'M2': 0.0,
    }

    t2_params = {**params, 'BINARY': 'T2'}
    ell1_params = {**params, 'BINARY': 'ELL1'}

    t2_delays = compute_binary_delay(t_test, t2_params)
    ell1_delays = compute_binary_delay(t_test, ell1_params)

    print(f"\nT2 delays (mus): {t2_delays * 1e6}")
    print(f"ELL1 delays (mus): {ell1_delays * 1e6}")
    print(f"Max difference: {np.max(np.abs(t2_delays - ell1_delays)):.2e} s")

    assert np.all(np.abs(t2_delays) < 2.0 * 0.025795), "T2 delays exceed physical maximum"
    np.testing.assert_allclose(t2_delays, ell1_delays, rtol=1e-15,
                               err_msg="T2 (ELL1-style) does not match ELL1")
    print("[x] T2 ELL1-style matches ELL1 exactly")


if __name__ == "__main__":
    print("\nBinary Model Implementation Tests")
    print("="*60)
    
    # Test BT model
    bt_delays = test_bt_model()
    
    # Test T2 model
    test_t2_model_dd_style()
    test_t2_model_ell1_style()
    
    print("\n" + "="*60)
    print("All Tests Completed Successfully!")
    print("="*60)
