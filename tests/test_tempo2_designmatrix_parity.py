"""Design-matrix parity tests for Tempo2-compatible mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.testing.tempo2_reference import tempo2_reference
from jug.fitting.optimized_fitter import compute_designmatrix

from tempo2_fixtures import get_tempo2_fixture

TARGET_COLUMNS = ("F0", "F1", "DM", "RAJ", "DECJ", "PB", "A1", "EPS1", "EPS2")

FIXTURE_COLUMNS = {
    "epta_j0030_isolated": ("F0", "F1", "DM"),
    "epta_j1909_t2": TARGET_COLUMNS,
    "ppta_j1902_ell1h": TARGET_COLUMNS,
}


def _project_offset(column: np.ndarray) -> np.ndarray:
    """Remove the constant offset column that libstempo carries explicitly."""
    return column - np.mean(column)


def _column_stats(jug_col: np.ndarray, ref_col: np.ndarray) -> dict[str, float]:
    jug_proj = _project_offset(np.asarray(jug_col, dtype=np.float64))
    ref_proj = _project_offset(np.asarray(ref_col, dtype=np.float64))
    finite = np.isfinite(jug_proj) & np.isfinite(ref_proj)
    jug_proj = jug_proj[finite]
    ref_proj = ref_proj[finite]
    ref_norm2 = float(np.dot(ref_proj, ref_proj))
    if ref_norm2 == 0.0:
        scale = np.nan
    else:
        scale = float(np.dot(jug_proj, ref_proj) / ref_norm2)
    if np.std(jug_proj) == 0.0 or np.std(ref_proj) == 0.0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(jug_proj, ref_proj)[0, 1])
    delta = jug_proj - ref_proj
    return {
        "corr": corr,
        "scale": scale,
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _worst_toas_message(param: str, jug_col: np.ndarray, ref_col: np.ndarray, n: int = 5) -> str:
    jug_proj = _project_offset(np.asarray(jug_col, dtype=np.float64))
    ref_proj = _project_offset(np.asarray(ref_col, dtype=np.float64))
    delta = jug_proj - ref_proj
    worst = np.argsort(np.abs(delta))[-n:][::-1]
    rows = [
        f"{idx}: jug={jug_proj[idx]:.9e}, ref={ref_proj[idx]:.9e}, delta={delta[idx]:.9e}"
        for idx in worst
    ]
    return f"{param} worst TOAs after offset projection: " + "; ".join(rows)


def _assert_column_matches(param: str, jug_col: np.ndarray, ref_col: np.ndarray):
    stats = _column_stats(jug_col, ref_col)
    message = (
        f"{param}: corr={stats['corr']:.8f}, scale={stats['scale']:.8f}, "
        f"rms={stats['rms']:.3e}, max={stats['max_abs']:.3e}; "
        f"{_worst_toas_message(param, jug_col, ref_col)}"
    )
    assert stats["corr"] > 0.9998, message
    assert abs(stats["scale"] - 1.0) < 0.02, message


@pytest.mark.tempo2
def test_tempo2_sandbox_designmatrix_smoke():
    fixture = get_tempo2_fixture("epta_j1909_t2")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        include_designmatrix=True,
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert ref.designmatrix.shape[0] == ref.ntoa
    assert ref.designmatrix.shape[1] > 0
    assert ref.designmatrix.shape[1] == len(ref.designmatrix_labels)
    assert np.all(np.isfinite(ref.designmatrix))


@pytest.mark.tempo2
def test_tempo2_designmatrix_column_parity_f0():
    fixture = get_tempo2_fixture("epta_j1909_t2")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=["F0"],
        include_designmatrix=True,
    )

    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        ["F0"],
        compatibility="tempo2",
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert jug.matrix.shape[0] == ref.ntoa
    assert "F0" in jug.labels
    assert ref.designmatrix_labels == ["Offset", "F0"]
    np.testing.assert_allclose(jug.matrix[:, 0], ref.designmatrix[:, 1], rtol=0.0, atol=0.02)


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", ["epta_j0030_isolated", "epta_j1909_t2", "ppta_j1902_ell1h"])
def test_tempo2_designmatrix_columns_match_libstempo(fixture_id):
    """Compare real JUG timing columns against libstempo by label.

    libstempo includes an explicit offset column.  JUG residual columns are
    generated from mean-subtracted residuals, so equality is checked after
    projecting both columns orthogonal to the constant offset.
    """
    fixture = get_tempo2_fixture(fixture_id)
    fit_params = list(FIXTURE_COLUMNS[fixture_id])
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params,
        compatibility="tempo2",
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert jug.matrix.shape[0] == ref.ntoa
    assert jug.labels == fit_params

    ref_label_to_idx = {label: idx for idx, label in enumerate(ref.designmatrix_labels)}
    for jug_idx, param in enumerate(jug.labels):
        assert param in ref_label_to_idx
        ref_col = ref.designmatrix[:, ref_label_to_idx[param]]
        _assert_column_matches(param, jug.matrix[:, jug_idx], ref_col)
