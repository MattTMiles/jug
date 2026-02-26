"""Tests for white noise models: EFAC, EQUAD, ECORR parsing and application.

Tests cover:
- Parsing T2EFAC/T2EQUAD/ECORR (Tempo2 format)
- Parsing EFAC/EQUAD (Tempo1 format)
- Parsing global (flag-less) EFAC/EQUAD
- Backend mask construction
- EFAC-only, EQUAD-only, EFAC+EQUAD scaling
- Multiple backends with different noise parameters
- Integration with par_reader noise line collection
- Round-trip: par file -> noise_lines -> parse -> apply
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from jug.noise.white import (
    WhiteNoiseEntry,
    parse_noise_lines,
    parse_noise_params_from_file,
    build_backend_mask,
    apply_white_noise,
)


# ===================================================================
# parse_noise_lines
# ===================================================================

class TestParseNoiseLines:
    """Test noise line parsing from par file content."""

    def test_t2efac(self):
        lines = ["T2EFAC -f L-wide_PUPPI 1.5"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EFAC"
        assert e.flag_name == "f"
        assert e.flag_value == "L-wide_PUPPI"
        assert e.value == pytest.approx(1.5)

    def test_t2equad(self):
        lines = ["T2EQUAD -f L-wide_PUPPI 0.35"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EQUAD"
        assert e.flag_name == "f"
        assert e.flag_value == "L-wide_PUPPI"
        assert e.value == pytest.approx(0.35)

    def test_ecorr(self):
        lines = ["ECORR -f 430_ASP 2.1"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "ECORR"
        assert e.flag_name == "f"
        assert e.flag_value == "430_ASP"
        assert e.value == pytest.approx(2.1)

    def test_tempo1_efac(self):
        """Tempo1-style: EFAC <flag_name> <flag_value> <value>"""
        lines = ["EFAC -be GUPPI 1.2"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EFAC"
        assert e.flag_name == "be"
        assert e.flag_value == "GUPPI"
        assert e.value == pytest.approx(1.2)

    def test_tempo1_equad(self):
        lines = ["EQUAD -be GUPPI 0.5"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EQUAD"
        assert e.flag_name == "be"
        assert e.flag_value == "GUPPI"
        assert e.value == pytest.approx(0.5)

    def test_global_efac(self):
        """EFAC with no flag selector applies to all TOAs."""
        lines = ["EFAC 1.1"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EFAC"
        assert e.flag_name == "*"
        assert e.flag_value == "*"
        assert e.value == pytest.approx(1.1)

    def test_global_equad(self):
        lines = ["EQUAD 0.2"]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        e = entries[0]
        assert e.kind == "EQUAD"
        assert e.flag_name == "*"
        assert e.flag_value == "*"
        assert e.value == pytest.approx(0.2)

    def test_multiple_entries(self):
        """Parse multiple mixed noise entries."""
        lines = [
            "T2EFAC -f L-wide_PUPPI 1.1",
            "T2EFAC -f 430_ASP 1.3",
            "T2EQUAD -f L-wide_PUPPI 0.5",
            "T2EQUAD -f 430_ASP 0.8",
            "ECORR -f L-wide_PUPPI 2.0",
        ]
        entries = parse_noise_lines(lines)
        assert len(entries) == 5
        kinds = [e.kind for e in entries]
        assert kinds.count("EFAC") == 2
        assert kinds.count("EQUAD") == 2
        assert kinds.count("ECORR") == 1

    def test_empty_lines(self):
        entries = parse_noise_lines([])
        assert entries == []

    def test_comments_skipped(self):
        lines = [
            "# This is a comment",
            "T2EFAC -f PUPPI 1.0",
        ]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1

    def test_blank_lines_skipped(self):
        lines = ["", "  ", "T2EFAC -f PUPPI 1.0", ""]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1

    def test_malformed_line_skipped(self):
        """Lines with wrong number of tokens or bad float are silently skipped."""
        lines = [
            "T2EFAC -f",                  # too few tokens
            "T2EFAC -f PUPPI notanumber",  # bad float
            "T2EFAC -f PUPPI 1.0",         # good
        ]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        assert entries[0].value == pytest.approx(1.0)

    def test_whitespace_insensitive(self):
        """Extra whitespace shouldn't matter."""
        lines = ["  T2EFAC   -f   PUPPI   1.5  "]
        entries = parse_noise_lines(lines)
        assert len(entries) == 1
        assert entries[0].value == pytest.approx(1.5)

    def test_frozen_dataclass(self):
        """WhiteNoiseEntry should be immutable."""
        e = WhiteNoiseEntry("EFAC", "f", "PUPPI", 1.0)
        with pytest.raises(AttributeError):
            e.value = 2.0  # type: ignore


# ===================================================================
# build_backend_mask
# ===================================================================

class TestBuildBackendMask:
    """Test TOA flag matching."""

    @pytest.fixture
    def sample_flags(self):
        """Sample per-TOA flag dicts."""
        return [
            {"f": "L-wide_PUPPI", "be": "GUPPI"},
            {"f": "L-wide_PUPPI", "be": "GUPPI"},
            {"f": "430_ASP", "be": "ASP"},
            {"f": "430_ASP", "be": "ASP"},
            {"f": "820_GUPPI", "be": "GUPPI"},
        ]

    def test_match_f_flag(self, sample_flags):
        mask = build_backend_mask(sample_flags, "f", "L-wide_PUPPI")
        np.testing.assert_array_equal(mask, [True, True, False, False, False])

    def test_match_be_flag(self, sample_flags):
        mask = build_backend_mask(sample_flags, "be", "GUPPI")
        np.testing.assert_array_equal(mask, [True, True, False, False, True])

    def test_no_match(self, sample_flags):
        mask = build_backend_mask(sample_flags, "f", "NONEXISTENT")
        np.testing.assert_array_equal(mask, [False, False, False, False, False])

    def test_global_wildcard(self, sample_flags):
        mask = build_backend_mask(sample_flags, "*", "*")
        np.testing.assert_array_equal(mask, [True, True, True, True, True])

    def test_missing_flag_key(self, sample_flags):
        """TOAs without the flag key should not match."""
        mask = build_backend_mask(sample_flags, "sys", "value")
        np.testing.assert_array_equal(mask, [False, False, False, False, False])

    def test_empty_flags(self):
        mask = build_backend_mask([], "f", "PUPPI")
        assert len(mask) == 0
        assert mask.dtype == bool


# ===================================================================
# apply_white_noise
# ===================================================================

class TestApplyWhiteNoise:
    """Test EFAC/EQUAD scaling of TOA errors."""

    @pytest.fixture
    def simple_setup(self):
        """4 TOAs, 2 backends."""
        errors_us = np.array([1.0, 2.0, 3.0, 4.0])
        toa_flags = [
            {"f": "A"},
            {"f": "A"},
            {"f": "B"},
            {"f": "B"},
        ]
        return errors_us, toa_flags

    def test_no_entries(self, simple_setup):
        """No noise entries -> errors unchanged."""
        errors_us, toa_flags = simple_setup
        result = apply_white_noise(errors_us, toa_flags, [])
        np.testing.assert_array_almost_equal(result, errors_us)

    def test_efac_only(self, simple_setup):
        """EFAC scales errors multiplicatively."""
        errors_us, toa_flags = simple_setup
        entries = [WhiteNoiseEntry("EFAC", "f", "A", 2.0)]
        result = apply_white_noise(errors_us, toa_flags, entries)
        # Backend A: sigma_eff = 2.0 * sigma,  Backend B: unchanged (EFAC=1)
        expected = np.array([2.0, 4.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_equad_only(self, simple_setup):
        """EQUAD adds in quadrature."""
        errors_us, toa_flags = simple_setup
        entries = [WhiteNoiseEntry("EQUAD", "f", "A", 1.0)]
        result = apply_white_noise(errors_us, toa_flags, entries)
        # Backend A: sigma_eff = sqrt(sigma^2 + 1^2) = sqrt(1+1), sqrt(4+1)
        # Backend B: unchanged
        expected = np.array([
            np.sqrt(1.0 + 1.0),   # sqrt(2)
            np.sqrt(4.0 + 1.0),   # sqrt(5)
            3.0,
            4.0,
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_efac_and_equad(self, simple_setup):
        """EFAC + EQUAD: sigma_eff = EFAC * sqrt(sigma^2 + EQUAD^2)."""
        errors_us, toa_flags = simple_setup
        entries = [
            WhiteNoiseEntry("EFAC", "f", "A", 1.5),
            WhiteNoiseEntry("EQUAD", "f", "A", 0.5),
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)
        # Backend A: sigma_eff = sqrt(1.5^2 * (sigma^2 + 0.5^2))
        expected_A0 = np.sqrt(1.5**2 * (1.0**2 + 0.5**2))
        expected_A1 = np.sqrt(1.5**2 * (2.0**2 + 0.5**2))
        expected = np.array([expected_A0, expected_A1, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_efac(self, simple_setup):
        """Global EFAC applies to all TOAs."""
        errors_us, toa_flags = simple_setup
        entries = [WhiteNoiseEntry("EFAC", "*", "*", 3.0)]
        result = apply_white_noise(errors_us, toa_flags, entries)
        expected = errors_us * 3.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiple_backends(self):
        """Each backend gets its own EFAC/EQUAD."""
        errors_us = np.array([1.0, 1.0, 1.0])
        toa_flags = [{"f": "A"}, {"f": "B"}, {"f": "C"}]
        entries = [
            WhiteNoiseEntry("EFAC", "f", "A", 2.0),
            WhiteNoiseEntry("EFAC", "f", "B", 3.0),
            WhiteNoiseEntry("EQUAD", "f", "A", 1.0),
            WhiteNoiseEntry("EQUAD", "f", "C", 0.5),
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)
        # A: sqrt(2^2 * (1^2 + 1^2)) = sqrt(8)
        # B: sqrt(3^2 * (1^2 + 0^2)) = 3
        # C: sqrt(1^2 * (1^2 + 0.5^2)) = sqrt(1.25)
        expected = np.array([
            np.sqrt(4.0 * 2.0),     # 2sqrt2
            3.0,
            np.sqrt(1.25),
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ecorr_ignored(self, simple_setup):
        """ECORR entries should be silently ignored."""
        errors_us, toa_flags = simple_setup
        entries = [
            WhiteNoiseEntry("ECORR", "f", "A", 10.0),
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)
        np.testing.assert_array_almost_equal(result, errors_us)

    def test_last_match_wins(self):
        """If multiple entries match same backend, last one wins."""
        errors_us = np.array([1.0])
        toa_flags = [{"f": "A"}]
        entries = [
            WhiteNoiseEntry("EFAC", "f", "A", 2.0),
            WhiteNoiseEntry("EFAC", "f", "A", 5.0),  # should override
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)
        expected = np.array([5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_formula_derivation(self):
        """Verify the exact formula: sigma_eff^2 = EFAC^2 * (sigma^2 + EQUAD^2)."""
        sigma = 1.5
        efac = 1.3
        equad = 0.7
        errors_us = np.array([sigma])
        toa_flags = [{"f": "X"}]
        entries = [
            WhiteNoiseEntry("EFAC", "f", "X", efac),
            WhiteNoiseEntry("EQUAD", "f", "X", equad),
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)
        expected = np.sqrt(efac**2 * (sigma**2 + equad**2))
        np.testing.assert_almost_equal(result[0], expected, decimal=14)


# ===================================================================
# parse_noise_params_from_file
# ===================================================================

class TestParseNoiseParamsFromFile:
    """Test reading noise params directly from a par file on disk."""

    def test_round_trip(self, tmp_path):
        """Write a par file with noise lines, parse it back."""
        par_content = """\
PSRJ     J0437-4715
RAJ      04:37:15.883
DECJ     -47:15:09.11
F0       173.6879
PEPOCH   55000.0
T2EFAC -f L-wide_PUPPI 1.2
T2EQUAD -f L-wide_PUPPI 0.4
T2EFAC -f 430_ASP 1.1
ECORR -f L-wide_PUPPI 3.5
"""
        par_file = tmp_path / "test.par"
        par_file.write_text(par_content)

        entries = parse_noise_params_from_file(str(par_file))
        assert len(entries) == 4
        kinds = [e.kind for e in entries]
        assert kinds.count("EFAC") == 2
        assert kinds.count("EQUAD") == 1
        assert kinds.count("ECORR") == 1


# ===================================================================
# Par reader integration
# ===================================================================

class TestParReaderNoiseLine:
    """Test that jug.io.par_reader.parse_par_file stores noise lines."""

    def test_noise_lines_collected(self, tmp_path):
        """Noise keyword lines end up in params['_noise_lines']."""
        par_content = """\
PSRJ     J0437-4715
F0       173.6879
PEPOCH   55000.0
T2EFAC -f PUPPI 1.3
T2EQUAD -f PUPPI 0.2
ECORR -f PUPPI 5.0
"""
        par_file = tmp_path / "test.par"
        par_file.write_text(par_content)

        from jug.io.par_reader import parse_par_file
        params = parse_par_file(par_file)

        assert '_noise_lines' in params
        assert len(params['_noise_lines']) == 3
        # Verify keywords are present
        keywords = [l.split()[0].upper() for l in params['_noise_lines']]
        assert 'T2EFAC' in keywords
        assert 'T2EQUAD' in keywords
        assert 'ECORR' in keywords

    def test_no_noise_lines(self, tmp_path):
        """Par file without noise lines has no _noise_lines key."""
        par_content = """\
PSRJ     J0437-4715
F0       173.6879
PEPOCH   55000.0
"""
        par_file = tmp_path / "test.par"
        par_file.write_text(par_content)

        from jug.io.par_reader import parse_par_file
        params = parse_par_file(par_file)

        assert '_noise_lines' not in params

    def test_noise_lines_not_parsed_as_params(self, tmp_path):
        """Noise lines should NOT appear as regular params."""
        par_content = """\
PSRJ     J0437-4715
F0       173.6879
PEPOCH   55000.0
T2EFAC -f PUPPI 1.3
EQUAD -be GUPPI 0.4
"""
        par_file = tmp_path / "test.par"
        par_file.write_text(par_content)

        from jug.io.par_reader import parse_par_file
        params = parse_par_file(par_file)

        # These should NOT be regular params (they'd be mangled key=value)
        assert 'T2EFAC' not in params
        assert 'EQUAD' not in params
        # But they should be in _noise_lines
        assert len(params['_noise_lines']) == 2


# ===================================================================
# End-to-end integration
# ===================================================================

class TestEndToEndNoise:
    """Integration: par file -> par_reader -> parse_noise_lines -> apply."""

    def test_par_to_scaled_errors(self, tmp_path):
        """Full pipeline: read par, extract noise, scale errors."""
        par_content = """\
PSRJ     J0437-4715
F0       173.6879
PEPOCH   55000.0
T2EFAC -f backend_A 2.0
T2EQUAD -f backend_A 1.0
T2EFAC -f backend_B 1.5
"""
        par_file = tmp_path / "test.par"
        par_file.write_text(par_content)

        from jug.io.par_reader import parse_par_file
        from jug.noise.white import parse_noise_lines, apply_white_noise

        # Step 1: Parse par file
        params = parse_par_file(par_file)
        noise_lines = params['_noise_lines']

        # Step 2: Parse noise entries
        entries = parse_noise_lines(noise_lines)
        assert len(entries) == 3

        # Step 3: Apply to synthetic errors
        errors_us = np.array([1.0, 1.0, 1.0])
        toa_flags = [
            {"f": "backend_A"},
            {"f": "backend_B"},
            {"f": "backend_C"},  # no noise spec -> unchanged
        ]
        result = apply_white_noise(errors_us, toa_flags, entries)

        # backend_A: sqrt(2^2 * (1^2 + 1^2)) = sqrt(8) = 2sqrt2 ~= 2.828
        expected_A = np.sqrt(4.0 * 2.0)
        np.testing.assert_almost_equal(result[0], expected_A)

        # backend_B: sqrt(1.5^2 * (1^2 + 0^2)) = 1.5
        np.testing.assert_almost_equal(result[1], 1.5)

        # backend_C: unchanged (EFAC=1, EQUAD=0)
        np.testing.assert_almost_equal(result[2], 1.0)
