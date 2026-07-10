"""Shared tempo2 fixture parity assertion helpers."""

from __future__ import annotations

import numpy as np


def project_offset(column: np.ndarray) -> np.ndarray:
    """Remove the constant offset column that libstempo carries explicitly."""
    return column - np.mean(column)


def column_stats(jug_col: np.ndarray, ref_col: np.ndarray) -> dict[str, float]:
    jug_proj = project_offset(np.asarray(jug_col, dtype=np.float64))
    ref_proj = project_offset(np.asarray(ref_col, dtype=np.float64))
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


def worst_toas_message(param: str, jug_col: np.ndarray, ref_col: np.ndarray, n: int = 5) -> str:
    jug_proj = project_offset(np.asarray(jug_col, dtype=np.float64))
    ref_proj = project_offset(np.asarray(ref_col, dtype=np.float64))
    delta = jug_proj - ref_proj
    worst = np.argsort(np.abs(delta))[-n:][::-1]
    rows = [
        f"{idx}: jug={jug_proj[idx]:.9e}, ref={ref_proj[idx]:.9e}, delta={delta[idx]:.9e}"
        for idx in worst
    ]
    return f"{param} worst TOAs after offset projection: " + "; ".join(rows)


def assert_column_matches(param: str, jug_col: np.ndarray, ref_col: np.ndarray) -> None:
    stats = column_stats(jug_col, ref_col)
    message = (
        f"{param}: corr={stats['corr']:.8f}, scale={stats['scale']:.8f}, "
        f"rms={stats['rms']:.3e}, max={stats['max_abs']:.3e}; "
        f"{worst_toas_message(param, jug_col, ref_col)}"
    )
    assert stats["corr"] > 0.9998, message
    assert abs(stats["scale"] - 1.0) < 0.02, message


def tempo2_to_pint_vela_scale(param: str) -> float:
    """Scale libstempo units to JUG's exported PINT/Vela fit-unit convention."""
    param_upper = param.upper()
    if param_upper == "RAJ":
        return np.pi / 12.0
    if param_upper == "DECJ":
        return np.pi / 180.0
    return 1.0


def delta_stats_ns(jug_residuals_us, tempo2_residuals_us) -> dict[str, float]:
    delta_ns = (np.asarray(jug_residuals_us) - np.asarray(tempo2_residuals_us)) * 1000.0
    return {
        "rms": float(np.sqrt(np.mean(np.square(delta_ns)))),
        "max_abs": float(np.max(np.abs(delta_ns))),
        "p99_abs": float(np.percentile(np.abs(delta_ns), 99)),
        "mean": float(np.mean(delta_ns)),
    }


def assert_residual_parity(
    jug,
    ref,
    fixture_id: str,
    *,
    rms_delta_ns: float = 5.0,
    max_delta_ns: float = 25.0,
    p99_delta_ns: float = 10.0,
) -> None:
    assert jug["n_toas"] == ref.ntoa
    stats = delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    wrms_delta_ns = abs(jug["weighted_rms_us"] - ref.wrms_us) * 1000.0
    message = (
        f"{fixture_id}: rms={stats['rms']:.3f} ns, "
        f"p99={stats['p99_abs']:.3f} ns, max={stats['max_abs']:.3f} ns, "
        f"mean={stats['mean']:.3f} ns, wrms_delta={wrms_delta_ns:.3f} ns; "
        f"first5_delta_ns={((np.asarray(jug['residuals_us'][:5]) - ref.residuals_us[:5]) * 1000.0).tolist()}"
    )
    assert stats["rms"] < rms_delta_ns, message
    assert stats["p99_abs"] < p99_delta_ns, message
    assert stats["max_abs"] < max_delta_ns, message
    assert wrms_delta_ns < rms_delta_ns, message
