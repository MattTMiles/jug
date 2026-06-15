from pathlib import Path

import numpy as np

from jug.scripts.compare_pint_batch import (
    _noise_component_rows,
    _remove_spin_gauge,
    _read_two_column_clock,
    _write_pint_tempo_clock_from_two_column,
)


def test_clock_conversion_preserves_offsets(tmp_path):
    src = tmp_path / "obs2gps.clk"
    dst = tmp_path / "time_obs.dat"
    src.write_text("# comment\n58000 1e-6\n58001 -2e-6\n")

    _write_pint_tempo_clock_from_two_column(src, dst)

    text = dst.read_text()
    assert "synthetic from obs2gps.clk" in text
    assert "1.000" in text
    assert "-2.000" in text
    assert _read_two_column_clock(src) == [(58000.0, 1e-6), (58001.0, -2e-6)]


def test_noise_component_mapping_uses_known_aliases():
    jug = {"RedNoise": np.array([1.0, 2.0]), "ECORR": np.array([3.0, 4.0])}
    pint = {"pl_red_noise": np.array([1.5, 2.5]), "ecorr_noise": np.array([3.5, 4.5])}

    rows = _noise_component_rows(jug, pint)

    assert [row[0] for row in rows] == ["Red noise", "ECORR"]
    np.testing.assert_allclose(rows[0][3] - rows[0][4], [-0.5, -0.5])


def test_remove_spin_gauge_removes_quadratic_only():
    mjd = np.linspace(53000.0, 59000.0, 200)
    x = (mjd - np.mean(mjd)) / np.ptp(mjd)
    structured = 2.0 + 0.3 * x - 0.7 * x**2
    signal = 1e-4 * np.sin(np.linspace(0.0, 12.0, len(mjd)))

    cleaned = _remove_spin_gauge(structured + signal, mjd, np.ones_like(mjd))

    np.testing.assert_allclose(cleaned, _remove_spin_gauge(signal, mjd, np.ones_like(mjd)), atol=1e-12)
