"""
Quick check: does refitting spin parameters explain the Tempo2 vs JUG residual gap?

This script:
1) Loads barycentric TOAs from Tempo2's `temp_pre_components.out`.
2) Computes Tempo2 prefit RMS from `prefit.res`.
3) Computes JUG-style residuals using the par file values.
4) Fits a simple spin model (phi0, F0, F1) by least squares against rounded pulse
   numbers and reports the resulting RMS.

Run with: `python parameter_fit_test.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SECS_PER_DAY = 86400.0


def load_barycentric_toas(path: Path) -> np.ndarray:
    """Column 2 of temp_pre_components.out = barycentric MJD from Tempo2."""
    vals: list[float] = []
    for line in path.read_text().splitlines():
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) >= 2:
            vals.append(float(parts[1]))
    return np.asarray(vals, dtype=float)


def load_tempo2_prefit_residuals(path: Path) -> np.ndarray:
    """Seconds from prefit.res (Tempo2)."""
    vals: list[float] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vals.append(float(parts[1]))
    return np.asarray(vals, dtype=float)


def load_spin_params(par_path: Path) -> tuple[float, float, float]:
    f0 = f1 = pepoch = None
    for line in par_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "F0":
            f0 = float(parts[1])
        elif parts[0] == "F1":
            f1 = float(parts[1])
        elif parts[0] == "PEPOCH":
            pepoch = float(parts[1])
    if f0 is None or f1 is None or pepoch is None:
        raise ValueError("Missing F0/F1/PEPOCH in par file")
    return f0, f1, pepoch


def jug_residuals(t_bary: np.ndarray, f0: float, f1: float, pepoch: float) -> np.ndarray:
    """Phase-based residuals used in the notebook."""
    dt = (t_bary - pepoch) * SECS_PER_DAY
    phase = f0 * dt + 0.5 * f1 * dt**2
    frac = (phase + 0.5) % 1.0 - 0.5
    return frac / f0


def fit_spin_params(t_bary: np.ndarray, f0: float, f1: float, pepoch: float) -> tuple[float, float, float]:
    """
    Rough spin refit: keep pulse numbering from the current model
    and solve for phi0, F0, F1 with linear least squares.
    """
    dt = (t_bary - pepoch) * SECS_PER_DAY
    phase_guess = f0 * dt + 0.5 * f1 * dt**2
    n_cycles = np.round(phase_guess)
    A = np.vstack([np.ones_like(dt), dt, 0.5 * dt**2]).T
    phi0_fit, f0_fit, f1_fit = np.linalg.lstsq(A, n_cycles, rcond=None)[0]
    return phi0_fit, f0_fit, f1_fit


def main() -> None:
    t_bary = load_barycentric_toas(Path("JUG/temp_pre_components.out"))
    tempo2_res = load_tempo2_prefit_residuals(Path("JUG/prefit.res"))
    f0, f1, pepoch = load_spin_params(Path("JUG/temp_model_tdb.par"))

    tempo2_rms_us = np.sqrt(np.mean((tempo2_res * 1e6) ** 2))

    jug_res = jug_residuals(t_bary, f0, f1, pepoch)
    jug_rms_us = np.sqrt(np.mean((jug_res * 1e6) ** 2))

    phi0_fit, f0_fit, f1_fit = fit_spin_params(t_bary, f0, f1, pepoch)
    phase_fit = phi0_fit + f0_fit * (t_bary - pepoch) * SECS_PER_DAY + 0.5 * f1_fit * (
        (t_bary - pepoch) * SECS_PER_DAY
    ) ** 2
    frac_fit = phase_fit - np.round(phase_fit)
    jug_fit_res = frac_fit / f0_fit
    jug_fit_rms_us = np.sqrt(np.mean((jug_fit_res * 1e6) ** 2))

    print("Tempo2 prefit RMS (us): {:.6f}".format(tempo2_rms_us))
    print("JUG-style RMS with par values (us): {:.6f}".format(jug_rms_us))
    print(
        "Fitted spin params: phi0={:.6f} cycles, F0={:.15f}, F1={:.3e}".format(
            phi0_fit, f0_fit, f1_fit
        )
    )
    print("JUG-style RMS after spin refit (us): {:.6f}".format(jug_fit_rms_us))

    if len(tempo2_res) == len(jug_fit_res):
        corr = np.corrcoef(jug_fit_res * 1e6, tempo2_res * 1e6)[0, 1]
        print("Correlation with Tempo2 residuals after refit: {:.6f}".format(corr))


if __name__ == "__main__":
    main()
