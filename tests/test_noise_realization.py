"""Tests for the generic noise realization pipeline.

Verifies that ``compute_noise_realization()`` and ``realize_noise_generic()``
produce identical results to the legacy per-process realization functions.
"""

import numpy as np
import pytest

from jug.noise.red_noise import (
    RedNoiseProcess,
    DMNoiseProcess,
    ChromaticNoiseProcess,
    realize_red_noise,
    realize_dm_noise,
    realize_chromatic_noise,
    realize_noise_generic,
)
from jug.engine.noise_mode import compute_noise_realization


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def toa_data(rng):
    n = 200
    mjd = np.sort(rng.uniform(50000, 55000, n))
    res = rng.standard_normal(n) * 1e-6
    err = np.full(n, 1e-6)
    freqs = rng.uniform(700, 3000, n)
    return mjd, res, err, freqs


# ---------------------------------------------------------------------------
# realize_noise_generic vs legacy functions
# ---------------------------------------------------------------------------

class TestGenericVsLegacy:
    """Generic function must be bit-identical to the legacy per-process ones."""

    def test_red_noise_parity(self, toa_data):
        mjd, res, err, _ = toa_data
        log10_A, gamma, nharm = -13.5, 4.33, 30

        legacy = realize_red_noise(mjd, res, err, log10_A, gamma, nharm)
        proc = RedNoiseProcess(log10_A, gamma, nharm)
        generic = realize_noise_generic(proc, mjd, res, err)

        np.testing.assert_array_equal(legacy, generic)

    def test_dm_noise_parity(self, toa_data):
        mjd, res, err, freqs = toa_data
        log10_A, gamma, nharm = -14.0, 2.5, 30

        legacy = realize_dm_noise(mjd, freqs, res, err, log10_A, gamma, nharm)
        proc = DMNoiseProcess(log10_A, gamma, nharm)
        generic = realize_noise_generic(proc, mjd, res, err, freq_mhz=freqs)

        np.testing.assert_array_equal(legacy, generic)

    def test_chromatic_noise_parity(self, toa_data):
        mjd, res, err, freqs = toa_data
        log10_A, gamma, nharm, chrom_idx = -14.0, 3.0, 30, 4.0

        legacy = realize_chromatic_noise(
            mjd, freqs, res, err, log10_A, gamma,
            chrom_idx=chrom_idx, n_harmonics=nharm,
        )
        proc = ChromaticNoiseProcess(log10_A, gamma, chrom_idx=chrom_idx, n_harmonics=nharm)
        generic = realize_noise_generic(proc, mjd, res, err, freq_mhz=freqs)

        np.testing.assert_array_equal(legacy, generic)


# ---------------------------------------------------------------------------
# compute_noise_realization (full registry path)
# ---------------------------------------------------------------------------

class TestComputeNoiseRealization:
    """End-to-end tests going through the NOISE_REGISTRY."""

    def test_red_noise_from_params(self, toa_data):
        mjd, res, err, _ = toa_data
        params = {'TNRedAmp': -13.5, 'TNRedGam': 4.33, 'TNRedC': 30}

        legacy = realize_red_noise(mjd, res, err, -13.5, 4.33, 30)
        result = compute_noise_realization('RedNoise', params, mjd, res, err)

        assert result is not None
        np.testing.assert_array_equal(legacy, result)

    def test_dm_noise_from_params(self, toa_data):
        mjd, res, err, freqs = toa_data
        params = {'TNDMAmp': -14.0, 'TNDMGam': 2.5, 'TNDMC': 30}

        # TNDMAmp is in T2 convention; realize_dm_noise takes enterprise
        # convention log10_A, so apply the same conversion.
        from jug.noise.red_noise import TNDM_OFFSET
        log10_A_ent = -14.0 - TNDM_OFFSET

        legacy = realize_dm_noise(mjd, freqs, res, err, log10_A_ent, 2.5, 30)
        result = compute_noise_realization(
            'DMNoise', params, mjd, res, err, freq_mhz=freqs,
        )

        assert result is not None
        np.testing.assert_array_equal(legacy, result)

    def test_chromatic_noise_from_params(self, toa_data):
        mjd, res, err, freqs = toa_data
        params = {
            'TNChromAmp': -14.0, 'TNChromGam': 3.0,
            'TNChromIdx': 4.0, 'TNChromC': 30,
        }

        legacy = realize_chromatic_noise(
            mjd, freqs, res, err, -14.0, 3.0, chrom_idx=4.0, n_harmonics=30,
        )
        result = compute_noise_realization(
            'ChromaticNoise', params, mjd, res, err, freq_mhz=freqs,
        )

        assert result is not None
        np.testing.assert_array_equal(legacy, result)

    def test_unknown_process_returns_none(self, toa_data):
        mjd, res, err, _ = toa_data
        assert compute_noise_realization('EFAC', {}, mjd, res, err) is None

    def test_missing_params_returns_none(self, toa_data):
        mjd, res, err, _ = toa_data
        # RedNoise params missing -> from_par returns None
        assert compute_noise_realization('RedNoise', {}, mjd, res, err) is None

    def test_chromatic_needs_freqs(self, toa_data):
        mjd, res, err, _ = toa_data
        params = {
            'TNChromAmp': -14.0, 'TNChromGam': 3.0,
            'TNChromIdx': 4.0, 'TNChromC': 30,
        }
        with pytest.raises(ValueError, match="requires freq_mhz"):
            compute_noise_realization('ChromaticNoise', params, mjd, res, err)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Robustness checks."""

    def test_no_impl_class_returns_none(self, toa_data):
        """Processes without impl_class (e.g. EFAC/EQUAD) return None."""
        mjd, res, err, _ = toa_data
        assert compute_noise_realization('EFAC', {}, mjd, res, err) is None
        assert compute_noise_realization('EQUAD', {}, mjd, res, err) is None

    def test_output_shape_matches_input(self, toa_data):
        mjd, res, err, freqs = toa_data
        params = {'TNRedAmp': -13.5, 'TNRedGam': 4.33, 'TNRedC': 30}
        result = compute_noise_realization('RedNoise', params, mjd, res, err)
        assert result.shape == mjd.shape

    def test_result_is_finite(self, toa_data):
        mjd, res, err, _ = toa_data
        params = {'TNRedAmp': -13.5, 'TNRedGam': 4.33, 'TNRedC': 30}
        result = compute_noise_realization('RedNoise', params, mjd, res, err)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# GUI param write-back (get_par_key_for_field)
# ---------------------------------------------------------------------------

class TestParKeyMapping:
    """Verify field→par-key mapping used by GUI write-back."""

    def test_red_noise_keys(self):
        from jug.engine.noise_mode import get_par_key_for_field
        assert get_par_key_for_field('RedNoise', 'log10_A') == 'REDAMP'
        assert get_par_key_for_field('RedNoise', 'gamma') == 'REDGAM'
        assert get_par_key_for_field('RedNoise', 'n_harmonics') == 'REDC'

    def test_chromatic_noise_keys(self):
        from jug.engine.noise_mode import get_par_key_for_field
        assert get_par_key_for_field('ChromaticNoise', 'log10_A') == 'CHROMAMP'
        assert get_par_key_for_field('ChromaticNoise', 'chrom_idx') == 'CHROMIDX'

    def test_unknown_returns_none(self):
        from jug.engine.noise_mode import get_par_key_for_field
        assert get_par_key_for_field('EFAC', 'value') is None
        assert get_par_key_for_field('RedNoise', 'nonexistent') is None

    def test_gui_roundtrip(self, toa_data):
        """Simulate GUI add→realise: defaults written via write_noise_params are parseable."""
        from jug.engine.noise_mode import get_impl_param_defs, write_noise_params
        mjd, res, err, freqs = toa_data

        # Simulate _add_process: get defaults, write to params via engine
        params = {}
        field_values = {p['key']: p['value'] for p in get_impl_param_defs('ChromaticNoise')}
        write_noise_params('ChromaticNoise', field_values, params)

        # Now compute_noise_realization should work
        result = compute_noise_realization(
            'ChromaticNoise', params, mjd, res, err, freq_mhz=freqs,
        )
        assert result is not None
        assert result.shape == mjd.shape

    def test_write_noise_params(self):
        """write_noise_params writes correct par keys and numeric values."""
        from jug.engine.noise_mode import write_noise_params
        params = {}
        write_noise_params('RedNoise', {'log10_A': '-13.5', 'gamma': '4.33', 'n_harmonics': '30'}, params)
        assert params['REDAMP'] == -13.5
        assert params['REDGAM'] == 4.33
        assert params['REDC'] == 30.0

    def test_write_noise_params_ignores_unknown(self):
        """write_noise_params silently skips fields with no par key."""
        from jug.engine.noise_mode import write_noise_params
        params = {}
        write_noise_params('EFAC', {'value': '1.0'}, params)
        assert params == {}
        write_noise_params('RedNoise', {'nonexistent': '42'}, params)
        assert params == {}


# ---------------------------------------------------------------------------
# TNECORR parsing convention tests
# ---------------------------------------------------------------------------

class TestTNECORRParsing:
    """Verify that TNECORR is parsed correctly for both known conventions.

    Two conventions exist in real par files:
      - TempoNest (e.g. MPTA/DR3):  value is log10(seconds), typically negative.
        Example: TNECORR -f KAT_MKBF -6.3264  → ecorr = 10^(-6.3264) s = 0.472 µs
      - Tempo2 (e.g. PPTA DR2):     value is directly in microseconds, always positive.
        Example: TNECORR -group CASPSR_40CM 0.1274  → ecorr = 0.1274 µs

    JUG distinguishes the two by sign: negative → log10(s), positive → direct µs.
    """

    def _parse(self, line):
        """Parse a single TNECORR par-file line via JUG's white noise parser."""
        from jug.noise.white import parse_noise_lines
        entries = parse_noise_lines(line.splitlines())
        ecorr_entries = [e for e in entries if e.kind == 'ECORR']
        return ecorr_entries

    def test_temponest_negative_log10s(self):
        """Negative TNECORR value (TempoNest log10(s)) is converted to µs correctly."""
        entries = self._parse("TNECORR -f KAT_MKBF -6.3264270978060475")
        assert len(entries) == 1
        e = entries[0]
        assert e.flag_name == "f"
        assert e.flag_value == "KAT_MKBF"
        expected_us = 10 ** (-6.3264270978060475) * 1e6  # ≈ 0.4716 µs
        assert abs(e.value - expected_us) < 1e-9, (
            f"Expected {expected_us:.6f} µs, got {e.value:.6f} µs"
        )

    def test_tempo2_positive_direct_us(self):
        """Positive TNECORR value (Tempo2 direct µs) is passed through unchanged."""
        entries = self._parse("TNECORR -group CASPSR_40CM 0.127374157869")
        assert len(entries) == 1
        e = entries[0]
        assert e.flag_name == "group"
        assert e.flag_value == "CASPSR_40CM"
        assert abs(e.value - 0.127374157869) < 1e-12, (
            f"Expected 0.127374 µs, got {e.value:.9f} µs"
        )

    def test_tempo2_small_positive_direct_us(self):
        """Small positive TNECORR (Tempo2 direct µs) is not misidentified as log10."""
        entries = self._parse("TNECORR -group PDFB1_1433 0.0137717366787")
        assert len(entries) == 1
        assert abs(entries[0].value - 0.0137717366787) < 1e-12

    def test_temponest_ecorr_physical_magnitude(self):
        """TempoNest TNECORR=-6.33 gives a physically plausible ECORR in µs."""
        entries = self._parse("TNECORR -f KAT_MKBF -6.33")
        assert len(entries) == 1
        ecorr_us = entries[0].value
        # Should be ~0.47 µs — reasonable for a millisecond pulsar
        assert 0.01 < ecorr_us < 10.0, (
            f"ECORR = {ecorr_us:.4f} µs is outside the plausible 0.01–10 µs range"
        )

    def test_multiple_tnecorr_lines(self):
        """Multiple TNECORR lines in a par file are all parsed."""
        par_text = "\n".join([
            "TNECORR -f KAT_MKBF -6.3264270978060475",
            "TNECORR -group CASPSR_40CM 0.127374157869",
            "TNECORR -group CPSR2_20CM 0.207124324536",
        ])
        from jug.noise.white import parse_noise_lines
        entries = [e for e in parse_noise_lines(par_text.splitlines()) if e.kind == 'ECORR']
        assert len(entries) == 3
        # First is TempoNest log10(s)
        assert entries[0].flag_value == "KAT_MKBF"
        assert abs(entries[0].value - 10**(-6.3264270978060475) * 1e6) < 1e-9
        # Second and third are Tempo2 direct µs
        assert abs(entries[1].value - 0.127374157869) < 1e-12
        assert abs(entries[2].value - 0.207124324536) < 1e-12

