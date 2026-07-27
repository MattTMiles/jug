"""Absolute-residual reconstruction recipe."""

from __future__ import annotations

import numpy as np
import pytest

from jug.residuals.gauge import (
    ReferenceGauge,
    apply_phase_gauge,
    reconstruct_absolute_residuals,
)


def test_reconstruct_none_is_plain_addition():
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    delta = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    out = reconstruct_absolute_residuals(ref, delta, ReferenceGauge(mode="none"))
    np.testing.assert_allclose(out, ref + delta)


def test_reconstruct_mean_matches_pre_v3_and_naive_differs_by_constant():
    ref = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    # Host reference already mean-gauged.
    ref_g = apply_phase_gauge(ref, ReferenceGauge(mode="mean", weights=None))
    delta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)  # nonzero mean
    gauge = ReferenceGauge(mode="mean", weights=None)

    abs_rec = reconstruct_absolute_residuals(ref_g, delta, gauge)
    centered_delta = apply_phase_gauge(delta, gauge)
    pre_v3 = ref_g + centered_delta
    np.testing.assert_allclose(abs_rec, pre_v3, rtol=0, atol=1e-12)

    naive = ref_g + delta
    diff = naive - abs_rec
    # Wart: naive sum differs by exactly a constant (the mean of delta).
    np.testing.assert_allclose(diff, np.full_like(diff, diff[0]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(diff[0], np.mean(delta), rtol=0, atol=1e-15)
    assert abs(float(diff[0])) > 1e-12


def test_reconstruct_mean_weighted():
    ref = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float64)
    w = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    gauge = ReferenceGauge(mode="mean", weights=w)
    ref_g = apply_phase_gauge(ref, gauge)
    delta = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    abs_rec = reconstruct_absolute_residuals(ref_g, delta, gauge)
    pre_v3 = ref_g + apply_phase_gauge(delta, gauge)
    np.testing.assert_allclose(abs_rec, pre_v3, rtol=0, atol=1e-12)


def test_reconstruct_constant_raises_refphs_tzr():
    with pytest.raises(NotImplementedError, match="REFPHS TZR"):
        reconstruct_absolute_residuals(
            np.ones(2),
            np.zeros(2),
            ReferenceGauge(mode="constant", offset_sec=1e-6),
        )


def test_frozen_model_absolute_residuals_sec_wrapper(tmp_path):
    from jug.engine.session import TimingSession
    from jug.fitting.residual_model import export_frozen_residual_model

    par = tmp_path / "spin.par"
    tim = tmp_path / "spin.tim"
    par.write_text(
        "\n".join(
            [
                "PSRJ J0000+0000",
                "F0 200.0 1",
                "F1 -1e-15 1",
                "PEPOCH 56000",
                "POSEPOCH 56000",
                "DM 10.0 1",
                "DMEPOCH 56000",
                "RAJ 00:00:00",
                "DECJ +00:00:00",
                "UNITS TDB",
                "EPHEM DE440",
                "CLK TT(BIPM2021)",
                "TZRMJD 56000",
                "TZRFRQ 1400",
                "TZRSITE ao",
                "",
            ]
        )
    )
    tim.write_text(
        "FORMAT 1\n" + "\n".join(f"test 1400.0 {56000.0 + 0.1 * i} 1.0 ao" for i in range(8)) + "\n"
    )
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="pint")
    session.compute_residuals(subtract_tzr=True)
    model = export_frozen_residual_model(session, fit_params=["F0", "DM"])
    delta = np.array([1e-8, 1e-3], dtype=np.float64)
    r_abs = model.absolute_residuals_sec(delta)
    r_delta = np.asarray(model.residual_delta_jax(delta), dtype=np.float64)
    expected = reconstruct_absolute_residuals(
        model.reference_residuals_sec, r_delta, model.reference_gauge
    )
    np.testing.assert_allclose(r_abs, expected, rtol=0, atol=1e-15)

    naive = model.reference_residuals_sec + r_delta
    diff = naive - r_abs
    np.testing.assert_allclose(diff, np.full_like(diff, diff[0]), rtol=0, atol=1e-12)
