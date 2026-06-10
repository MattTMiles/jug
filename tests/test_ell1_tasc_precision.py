"""Regression tests for sub-float64 ELL1 epoch precision."""

import numpy as np

from jug.fitting.derivatives_binary import compute_ell1_binary_delay


def test_ell1_delay_uses_high_precision_tasc_cache():
    tasc_text = "58258.9529477631550272"
    tasc_float = float(tasc_text)
    toas = np.array([57579.2, 58258.95, 58939.4], dtype=np.longdouble)
    base = {
        "BINARY": "ELL1",
        "A1": 0.6558,
        "TASC": tasc_float,
        "EPS1": 0.0,
        "EPS2": 0.0,
        "FB0": 6.29183023215e-5,
    }

    precise = dict(base)
    precise["_high_precision"] = {"TASC": tasc_text}
    explicit = dict(base)
    explicit["TASC"] = np.longdouble(tasc_text)

    delay_precise = np.asarray(compute_ell1_binary_delay(toas, precise))
    delay_explicit = np.asarray(compute_ell1_binary_delay(toas, explicit))
    delay_truncated = np.asarray(compute_ell1_binary_delay(toas, base))

    np.testing.assert_array_equal(delay_precise, delay_explicit)
    assert np.max(np.abs(delay_precise - delay_truncated)) > 10e-12
