"""
Angle Codec Round-Trip Tests
============================

Tests that verify RAJ/DECJ codecs correctly convert between
sexagesimal strings and radians with sub-mas precision.

Key requirements:
- Round-trip error < 5e-10 rad (< 0.1 mas)
- Handles edge cases (0, pi, 2*pi, negative DEC)
- Bit-level stability for repeated encoding

Run with:
    pytest jug/tests/test_angle_codecs.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.model.codecs import RAJCodec, DECJCodec, FloatCodec, EpochMJDCodec, CODECS


class TestRAJCodec:
    """Tests for Right Ascension codec."""

    def test_basic_decode(self):
        """Test: Basic sexagesimal decoding."""
        codec = RAJCodec()

        # Standard RA value
        ra = codec.decode("19:09:47.4346970")
        assert 5.0 < ra < 5.1  # Should be ~5.017 rad

        # Midnight (0h)
        ra = codec.decode("00:00:00.0")
        assert ra == 0.0

        # 6h = pi/2 rad
        ra = codec.decode("06:00:00.0")
        assert abs(ra - math.pi / 2) < 1e-10

        # 12h = pi rad
        ra = codec.decode("12:00:00.0")
        assert abs(ra - math.pi) < 1e-10

    def test_basic_encode(self):
        """Test: Basic radian to sexagesimal encoding."""
        codec = RAJCodec()

        # 0 rad = 0h
        s = codec.encode(0.0)
        assert s.startswith("00:00:00")

        # pi/2 rad = 6h
        s = codec.encode(math.pi / 2)
        assert s.startswith("06:00:00")

        # pi rad = 12h
        s = codec.encode(math.pi)
        assert s.startswith("12:00:00")

    def test_roundtrip_precision(self):
        """Test: Round-trip precision < 5e-10 rad (sub-mas)."""
        codec = RAJCodec()

        # Test various RA values
        test_values = [
            0.0,
            math.pi / 4,
            math.pi / 2,
            math.pi,
            3 * math.pi / 2,
            2 * math.pi - 0.001,
            5.016907824863444,  # J1909-3744 RA
        ]

        for original_rad in test_values:
            encoded = codec.encode(original_rad)
            decoded = codec.decode(encoded)
            error = abs(decoded - (original_rad % (2 * math.pi)))

            assert error < 5e-10, \
                f"Round-trip error {error:.2e} rad too large for {original_rad:.10f} rad"

    def test_edge_cases(self):
        """Test: Edge cases near boundaries."""
        codec = RAJCodec()

        # Near midnight wrap
        ra = codec.decode("23:59:59.99999999")
        assert ra < 2 * math.pi

        # Exact 24h should wrap to 0
        s = codec.encode(2 * math.pi)
        assert s.startswith("00:00:00") or s.startswith("24:00:00")

    def test_random_roundtrips(self):
        """Test: Random values for round-trip stability."""
        codec = RAJCodec()
        np.random.seed(42)

        for _ in range(100):
            original = np.random.uniform(0, 2 * math.pi)
            encoded = codec.encode(original)
            decoded = codec.decode(encoded)
            error = abs(decoded - original)
            assert error < 5e-10


class TestDECJCodec:
    """Tests for Declination codec."""

    def test_basic_decode(self):
        """Test: Basic sexagesimal decoding."""
        codec = DECJCodec()

        # Standard DEC value (negative)
        dec = codec.decode("-37:44:14.4662200")
        assert -0.7 < dec < -0.6  # Should be ~-0.659 rad

        # Equator (0 deg)
        dec = codec.decode("+00:00:00.0")
        assert dec == 0.0

        dec = codec.decode("-00:00:00.0")
        assert dec == 0.0

        # North pole (+90 deg = +pi/2 rad)
        dec = codec.decode("+90:00:00.0")
        assert abs(dec - math.pi / 2) < 1e-10

        # South pole (-90 deg = -pi/2 rad)
        dec = codec.decode("-90:00:00.0")
        assert abs(dec + math.pi / 2) < 1e-10

    def test_basic_encode(self):
        """Test: Basic radian to sexagesimal encoding."""
        codec = DECJCodec()

        # 0 rad = 0 deg
        s = codec.encode(0.0)
        assert "00:00:00" in s

        # pi/2 rad = +90 deg
        s = codec.encode(math.pi / 2)
        assert s.startswith("+90:00:00")

        # -pi/2 rad = -90 deg
        s = codec.encode(-math.pi / 2)
        assert s.startswith("-90:00:00")

    def test_roundtrip_precision(self):
        """Test: Round-trip precision < 5e-10 rad (sub-mas)."""
        codec = DECJCodec()

        # Test various DEC values
        test_values = [
            0.0,
            math.pi / 4,
            math.pi / 2 - 0.001,  # Near north pole
            -math.pi / 4,
            -math.pi / 2 + 0.001,  # Near south pole
            -0.6586410386328931,  # J1909-3744 DEC
        ]

        for original_rad in test_values:
            encoded = codec.encode(original_rad)
            decoded = codec.decode(encoded)
            error = abs(decoded - original_rad)

            assert error < 5e-10, \
                f"Round-trip error {error:.2e} rad too large for {original_rad:.10f} rad"

    def test_sign_handling(self):
        """Test: Sign handling for positive and negative DEC."""
        codec = DECJCodec()

        # Positive
        s = codec.encode(0.1)
        assert s.startswith('+')

        # Negative
        s = codec.encode(-0.1)
        assert s.startswith('-')

        # Zero can be either + or -
        s = codec.encode(0.0)
        assert s[0] in ('+', '-')

    def test_random_roundtrips(self):
        """Test: Random values for round-trip stability."""
        codec = DECJCodec()
        np.random.seed(42)

        for _ in range(100):
            original = np.random.uniform(-math.pi / 2 + 0.001, math.pi / 2 - 0.001)
            encoded = codec.encode(original)
            decoded = codec.decode(encoded)
            error = abs(decoded - original)
            assert error < 5e-10


class TestOtherCodecs:
    """Tests for Float and Epoch codecs."""

    def test_float_codec_roundtrip(self):
        """Test: Float codec preserves precision."""
        codec = FloatCodec()

        test_values = [0.0, 1.0, -1.0, 3.141592653589793, 1e-20, 1e20]

        for val in test_values:
            encoded = codec.encode(val)
            decoded = codec.decode(encoded)
            # Float codec should be exact for representable values
            assert decoded == val or abs(decoded - val) < abs(val) * 1e-15

    def test_epoch_codec_roundtrip(self):
        """Test: Epoch MJD codec preserves precision."""
        codec = EpochMJDCodec()

        test_values = [50000.0, 58000.123456789, 60000.0]

        for val in test_values:
            encoded = codec.encode(val)
            decoded = codec.decode(encoded)
            # MJD should preserve to ~1 microsecond (1e-11 days)
            assert abs(decoded - val) < 1e-9

    def test_codec_registry(self):
        """Test: Codec registry contains all expected codecs."""
        expected = ['float', 'epoch_mjd', 'raj', 'decj']

        for name in expected:
            assert name in CODECS, f"Missing codec: {name}"
            assert hasattr(CODECS[name], 'encode')
            assert hasattr(CODECS[name], 'decode')


class TestJ1909Values:
    """Tests using actual J1909-3744 coordinates."""

    def test_j1909_raj(self):
        """Test: J1909-3744 RA round-trip."""
        codec = RAJCodec()

        # Actual J1909-3744 RA from .par file
        raj_str = "19:09:47.4346970"
        raj_rad = codec.decode(raj_str)

        # Should be close to 5.017 rad
        assert 5.016 < raj_rad < 5.018

        # Round-trip
        raj_back = codec.encode(raj_rad)
        raj_check = codec.decode(raj_back)

        assert abs(raj_check - raj_rad) < 5e-10

    def test_j1909_decj(self):
        """Test: J1909-3744 DEC round-trip."""
        codec = DECJCodec()

        # Actual J1909-3744 DEC from .par file
        decj_str = "-37:44:14.4662200"
        decj_rad = codec.decode(decj_str)

        # Should be close to -0.659 rad
        assert -0.66 < decj_rad < -0.65

        # Round-trip
        decj_back = codec.encode(decj_rad)
        decj_check = codec.decode(decj_back)

        assert abs(decj_check - decj_rad) < 5e-10


def run_all_tests():
    """Run all codec tests."""
    print("="*80)
    print("Angle Codec Test Suite")
    print("="*80)

    # RAJ tests
    raj_tests = TestRAJCodec()
    raj_tests.test_basic_decode()
    print("[x] RAJ basic decode")
    raj_tests.test_basic_encode()
    print("[x] RAJ basic encode")
    raj_tests.test_roundtrip_precision()
    print("[x] RAJ round-trip precision")
    raj_tests.test_edge_cases()
    print("[x] RAJ edge cases")
    raj_tests.test_random_roundtrips()
    print("[x] RAJ random round-trips")

    # DECJ tests
    decj_tests = TestDECJCodec()
    decj_tests.test_basic_decode()
    print("[x] DECJ basic decode")
    decj_tests.test_basic_encode()
    print("[x] DECJ basic encode")
    decj_tests.test_roundtrip_precision()
    print("[x] DECJ round-trip precision")
    decj_tests.test_sign_handling()
    print("[x] DECJ sign handling")
    decj_tests.test_random_roundtrips()
    print("[x] DECJ random round-trips")

    # Other codecs
    other_tests = TestOtherCodecs()
    other_tests.test_float_codec_roundtrip()
    print("[x] Float codec round-trip")
    other_tests.test_epoch_codec_roundtrip()
    print("[x] Epoch codec round-trip")
    other_tests.test_codec_registry()
    print("[x] Codec registry")

    # J1909 tests
    j1909_tests = TestJ1909Values()
    j1909_tests.test_j1909_raj()
    print("[x] J1909 RA")
    j1909_tests.test_j1909_decj()
    print("[x] J1909 DEC")

    print()
    print("All tests passed!")


if __name__ == '__main__':
    run_all_tests()
