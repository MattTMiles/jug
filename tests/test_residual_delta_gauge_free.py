"""FrozenResidualModel residual_delta is gauge-free."""

from __future__ import annotations

import numpy as np

from jug.residuals.gauge import ReferenceGauge, apply_phase_gauge


def _write_dd_par(path, ecc=5e-4):
    path.write_text(
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
                "BINARY DD",
                "PB 5.0 1",
                "T0 56000.0 1",
                "A1 10.0 1",
                "OM 45.0 1",
                f"E {ecc} 1",
                "M2 0.2 1",
                "SINI 0.9 1",
                "TZRMJD 56000",
                "TZRFRQ 1400",
                "TZRSITE ao",
                "",
            ]
        )
    )


def _write_minimal_tim(path, n=16):
    lines = ["FORMAT 1"]
    for i in range(n):
        # Varying TOA errors → nonuniform weights for the weighted mean gauge.
        err = 0.5 + 0.1 * (i % 5)
        lines.append(f"test 1400.0 {56000.0 + 0.1 * i} {err} ao")
    path.write_text("\n".join(lines) + "\n")


def test_frozen_residual_delta_has_nonzero_mean_and_gauge_recovers_pre_v3(tmp_path):
    from jug.engine.session import TimingSession
    from jug.fitting.residual_model import export_frozen_residual_model

    par = tmp_path / "dd.par"
    tim = tmp_path / "dd.tim"
    _write_dd_par(par)
    _write_minimal_tim(tim)
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="pint")
    session.compute_residuals(subtract_tzr=True)
    model = export_frozen_residual_model(
        session,
        fit_params=["F0", "DM", "PB"],
    )

    # Multi-parameter delta whose columns have a nonzero mean when applied.
    delta = np.array([1e-8, 1e-3, 1e-8], dtype=np.float64)
    residual_delta = np.asarray(model.residual_delta_jax(delta), dtype=np.float64)

    assert (
        abs(float(np.mean(residual_delta))) > 1e-16
    ), "gauge-free residual_delta must retain a nonzero mean"

    w = model.reference_gauge.weights
    assert w is not None
    gauged = apply_phase_gauge(residual_delta, ReferenceGauge(mode="mean", weights=w))
    # Previously the export subtracted the weighted mean of the delta.
    previously_centered = residual_delta - np.sum(residual_delta * w)
    np.testing.assert_allclose(gauged, previously_centered, rtol=0, atol=1e-15)
    np.testing.assert_allclose(float(np.sum(gauged * w)), 0.0, atol=1e-15)
