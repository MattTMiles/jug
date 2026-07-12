"""Regression tests for EFAC/EQUAD white-noise convention."""

import numpy as np
import pytest

from jug.noise.white import apply_white_noise, parse_noise_lines

pytestmark = pytest.mark.smoke


def test_apply_white_noise_uses_enterprise_measurementnoise_form():
    errors_us = np.array([0.12, 2.325], dtype=float)
    toa_flags = [{"f": "KAT_MKBF"}, {"f": "KAT_MKBF"}]
    entries = parse_noise_lines([
        "EFAC -f KAT_MKBF 1.3",
        "EQUAD -f KAT_MKBF 0.4",
    ])

    scaled_us = apply_white_noise(errors_us, toa_flags, entries)

    expected_var_us2 = 1.3**2 * (errors_us**2 + 0.4**2)
    unscaled_equad_var_us2 = (1.3 * errors_us) ** 2 + 0.4**2
    np.testing.assert_allclose(scaled_us**2, expected_var_us2, rtol=1e-14)
    assert not np.allclose(expected_var_us2, unscaled_equad_var_us2)


def test_t2equad_negative_value_parses_as_log10_seconds_to_microseconds():
    entries = parse_noise_lines(["T2EQUAD -f KAT_MKBF -6.0"])

    assert len(entries) == 1
    assert entries[0].kind == "EQUAD"
    assert entries[0].value == 1.0
