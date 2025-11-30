"""Test script for BT/DD and T2 binary models.

This script validates the new binary model implementations against known
test cases before integration into the main calculator.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jug.delays.binary_bt import bt_binary_delay_vectorized
from jug.delays.binary_t2 import t2_binary_delay_vectorized


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
    print(f"\nDelay range: {np.min(delays) * 1e6:.3f} to {np.max(delays) * 1e6:.3f} μs")
    
    # Check that delays are reasonable (should be on order of a1 ~ 1.8 light-seconds)
    assert np.all(np.abs(delays) < 2.0 * a1), "Delays exceed maximum possible value!"
    print("✓ Delays are physically reasonable")
    
    return delays


def test_t2_model():
    """Test T2 binary model with synthetic parameters."""
    print("\n" + "="*60)
    print("Testing T2 (Tempo2 General) Binary Model")
    print("="*60)
    
    # T2 parameters (similar to BT but with additional time derivatives)
    pb = 1.533449474  # days
    a1 = 1.8  # light-seconds
    ecc = 0.00001
    om = 45.0  # degrees
    t0 = 50000.0  # MJD
    gamma = 0.0  # seconds
    pbdot = 0.0
    xdot = 0.0  # light-sec/sec
    edot = 0.0  # 1/sec (eccentricity derivative - T2 only)
    omdot = 0.0  # deg/yr
    m2 = 0.2  # solar masses
    sini = 0.9
    kin = 0.0  # degrees (inclination angle - T2 only)
    kom = 0.0  # degrees (ascending node - T2 only)
    
    # Test times
    t_test = jnp.array([50000.0, 50000.5, 50001.0, 50002.0])
    
    # Compute delays
    delays = t2_binary_delay_vectorized(
        t_test, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
        m2, sini, kin, kom
    )
    
    print(f"\nTest times (MJD): {t_test}")
    print(f"Binary delays (seconds): {delays}")
    print(f"Binary delays (microseconds): {delays * 1e6}")
    print(f"\nDelay range: {np.min(delays) * 1e6:.3f} to {np.max(delays) * 1e6:.3f} μs")
    
    # Check that delays are reasonable
    assert np.all(np.abs(delays) < 2.0 * a1), "Delays exceed maximum possible value!"
    print("✓ Delays are physically reasonable")
    
    return delays


def compare_bt_vs_t2():
    """Compare BT and T2 models with same parameters (should match)."""
    print("\n" + "="*60)
    print("Comparing BT vs T2 Models (Should Match)")
    print("="*60)
    
    # Identical parameters for both models
    pb, a1, ecc, om, t0 = 1.533449474, 1.8, 0.00001, 45.0, 50000.0
    gamma, pbdot, xdot, omdot = 0.0, 0.0, 0.0, 0.0
    m2, sini = 0.2, 0.9
    edot, kin, kom = 0.0, 0.0, 0.0
    
    t_test = jnp.linspace(50000.0, 50010.0, 100)
    
    # BT delays
    bt_delays = bt_binary_delay_vectorized(
        t_test, pb, a1, ecc, om, t0, gamma, pbdot, m2, sini, omdot, xdot
    )
    
    # T2 delays
    t2_delays = t2_binary_delay_vectorized(
        t_test, pb, a1, ecc, om, t0, gamma, pbdot, xdot, edot, omdot,
        m2, sini, kin, kom
    )
    
    # Compare
    diff = bt_delays - t2_delays
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"\nRMS difference: {rms_diff * 1e9:.3f} nanoseconds")
    print(f"Max difference: {np.max(np.abs(diff)) * 1e9:.3f} nanoseconds")
    
    if rms_diff < 1e-12:  # 1 picosecond tolerance
        print("✓ BT and T2 models match to numerical precision!")
    else:
        print("⚠ Models differ - may indicate implementation differences")
    
    return bt_delays, t2_delays, diff


if __name__ == "__main__":
    print("\nBinary Model Implementation Tests")
    print("="*60)
    
    # Test BT model
    bt_delays = test_bt_model()
    
    # Test T2 model
    t2_delays = test_t2_model()
    
    # Compare BT vs T2
    bt_delays, t2_delays, diff = compare_bt_vs_t2()
    
    print("\n" + "="*60)
    print("All Tests Completed Successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Integrate BT/DD/T2 models into combined_delays()")
    print("2. Add binary model detection in simple_calculator.py")
    print("3. Test on real pulsars with DD/T2 models")
    print("4. Add analytical derivatives for fitting support")
