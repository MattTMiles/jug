"""Regression tests for long-span DD orbital phase reduction."""

import numpy as np

from jug.delays.binary_dd import dd_binary_delay_from_tt0
from jug.fitting.derivatives_binary import compute_ell1_binary_delay
from jug.residuals.simple_calculator import _extract_binary_params
from jug.utils.orbit_reduction import reduce_binary_time_sec


def test_dd_reduced_phase_is_continuous_at_large_orbit_boundary():
    pb_days = 1.0
    period = np.longdouble(86400.0)
    boundary = np.longdouble(10000.0) * period
    epsilon = np.longdouble("5e-8")
    tt0_ld = np.array([boundary - epsilon, boundary + epsilon], dtype=np.longdouble)
    tt0_red = reduce_binary_time_sec(tt0_ld, pb_days=pb_days)

    delay = np.asarray(dd_binary_delay_from_tt0(
        np.asarray(tt0_ld, dtype=float),
        pb_days=pb_days,
        a1_lt_sec=3.0,
        ecc=0.2,
        omega_deg=40.0,
        omdot_deg_yr=1000.0,
        tt0_red_sec=tt0_red,
    ))

    assert np.all(np.isfinite(delay))
    assert abs(delay[1] - delay[0]) < 1e-10


def test_ell1_preserves_sub_float64_fb0_refinement():
    fb0_a = np.longdouble(np.float64(1.0 / 8640.0))
    fb0_b = fb0_a + np.longdouble(np.spacing(float(fb0_a))) / 4
    assert float(fb0_a) == float(fb0_b)

    tasc = np.longdouble("58314.106867705951675")
    toas = tasc + np.array([0.0, 1000.0, 3000.0, 6000.0], dtype=np.longdouble)

    def params(fb0):
        return {
            "BINARY": "ELL1",
            "FB0": float(fb0),
            "A1": 3.0,
            "TASC": float(tasc),
            "EPS1": 2e-6,
            "EPS2": 3e-6,
            "_high_precision": {"FB0": str(fb0), "TASC": str(tasc)},
        }

    delay_a = np.asarray(compute_ell1_binary_delay(toas, params(fb0_a)))
    delay_b = np.asarray(compute_ell1_binary_delay(toas, params(fb0_b)))
    assert np.std(delay_a - delay_b) > 1e-14

    bp_a = _extract_binary_params(params(fb0_a), verbose=False)
    bp_b = _extract_binary_params(params(fb0_b), verbose=False)
    assert bp_a["fb0_val"] == bp_b["fb0_val"]
    assert bp_a["fb0_ld"] == fb0_a
    assert bp_b["fb0_ld"] == fb0_b


def test_ell1_preserves_sub_float64_pb_refinement():
    pb_a = np.longdouble("0.69888924332690960384")
    pb_b = np.longdouble("0.6988892433269096468")
    assert float(pb_a) == float(pb_b)

    tasc = np.longdouble("58314.106867705951675")
    toas = tasc + np.array([0.0, 1000.0, 3000.0, 6000.0], dtype=np.longdouble)

    def params(pb):
        return {
            "BINARY": "ELL1",
            "PB": float(pb),
            "A1": 3.7188619642,
            "TASC": float(tasc),
            "EPS1": 2.65e-6,
            "EPS2": 2.09e-6,
            "_high_precision": {"PB": str(pb), "TASC": str(tasc)},
        }

    delay_a = np.asarray(compute_ell1_binary_delay(toas, params(pb_a)))
    delay_b = np.asarray(compute_ell1_binary_delay(toas, params(pb_b)))

    # Values collapse to same float64 PB, but accumulated phase must differ.
    assert np.std(delay_a - delay_b) > 1e-14

    bp_a = _extract_binary_params(params(pb_a), verbose=False)
    bp_b = _extract_binary_params(params(pb_b), verbose=False)
    assert bp_a["pb_val"] == bp_b["pb_val"]
    assert bp_a["pb_ld"] == pb_a
    assert bp_b["pb_ld"] == pb_b
