"""Small libstempo oracle wrapper for Tempo2-compatibility tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jug.testing.sandbox_tempo2 import Policy, tempopulsar


@dataclass
class Tempo2Reference:
    """Normalized Tempo2 reference data returned by the sandbox oracle."""

    residuals_us: np.ndarray
    errors_us: np.ndarray
    wrms_us: float
    ntoa: int
    params: dict[str, Any]
    designmatrix: np.ndarray | None = None
    designmatrix_labels: list[str] | None = None


def _weighted_rms(values_us: np.ndarray, errors_us: np.ndarray) -> float:
    weights = 1.0 / np.square(errors_us)
    return float(np.sqrt(np.sum(weights * np.square(values_us)) / np.sum(weights)))


def _param_snapshot(psr: Any) -> dict[str, Any]:
    """Best-effort snapshot of libstempo parameter values and uncertainties."""
    params: dict[str, Any] = {}
    names_obj = getattr(psr, "pars", [])
    values_obj = getattr(psr, "vals", [])
    errors_obj = getattr(psr, "errs", [])
    names = list(names_obj() if callable(names_obj) else names_obj or [])
    values = list(values_obj() if callable(values_obj) else values_obj or [])
    errors = list(errors_obj() if callable(errors_obj) else errors_obj or [])
    for idx, name in enumerate(names):
        entry: dict[str, Any] = {}
        if idx < len(values):
            entry["value"] = values[idx]
        if idx < len(errors):
            entry["error"] = errors[idx]
        params[str(name)] = entry
    return params


def tempo2_reference(
    par: str | Path,
    tim: str | Path,
    *,
    dofit: bool = False,
    fit_params: list[str] | None = None,
    include_designmatrix: bool = False,
    policy: Policy | None = None,
) -> Tempo2Reference:
    """Run libstempo through the sandbox and normalize units for tests.

    libstempo returns residuals in seconds and TOA uncertainties in microseconds.
    This wrapper returns both residuals and uncertainties in microseconds.
    """
    psr = tempopulsar(
        parfile=str(par),
        timfile=str(tim),
        dofit=False,
        policy=policy or Policy(call_timeout_s=120.0),
    )
    fit_param_names: list[str] | None = None
    if fit_params is not None:
        names_obj = getattr(psr, "pars", [])
        for name in list(names_obj() if callable(names_obj) else names_obj or []):
            try:
                psr[str(name)].fit = False
            except Exception:
                pass
        for name in fit_params:
            psr[name].fit = True
        names_obj = getattr(psr, "pars", [])
        fit_param_names = [str(name) for name in list(names_obj() if callable(names_obj) else names_obj or [])]
    if dofit:
        psr.fit()

    residuals_us = np.asarray(psr.residuals(), dtype=np.float64) * 1.0e6
    errors_us = np.asarray(psr.toaerrs, dtype=np.float64)
    designmatrix = None
    designmatrix_labels = None
    if include_designmatrix:
        designmatrix = np.asarray(psr.designmatrix(), dtype=np.float64)
        if fit_param_names is None:
            names_obj = getattr(psr, "pars", [])
            fit_param_names = [str(name) for name in list(names_obj() if callable(names_obj) else names_obj or [])]
        designmatrix_labels = ["Offset"] + fit_param_names

    return Tempo2Reference(
        residuals_us=residuals_us,
        errors_us=errors_us,
        wrms_us=_weighted_rms(residuals_us, errors_us),
        ntoa=int(residuals_us.size),
        params=_param_snapshot(psr),
        designmatrix=designmatrix,
        designmatrix_labels=designmatrix_labels,
    )
