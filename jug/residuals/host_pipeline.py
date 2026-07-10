"""Shared host residual finalization (mirror of pint/residuals.py)."""
from __future__ import annotations
import numpy as np
from jug.residuals.phase import compute_phase_residuals


def finalize_pint_host_residuals(
    *,
    dt_sec: np.ndarray,
    params: dict,
    weights_scaled: np.ndarray,
    subtract_mean_in_phase: bool,
    tzr_phase_for_residuals,
    jump_phase: np.ndarray,
    external_pn: np.ndarray | None,
    track_val,
    external_pn_add: np.ndarray | None,
    phase_bbat_mjd,
    phase_torb_sec,
    addsat_sec: np.ndarray | None,
    phase_mean_mode: str,
    use_native_bbat_phase5: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PINT-family host residual finalization via ``compute_phase_residuals``."""
    return compute_phase_residuals(
        dt_sec,
        params,
        weights_scaled,
        subtract_mean=subtract_mean_in_phase,
        tzr_phase=tzr_phase_for_residuals,
        jump_phase=jump_phase,
        external_pulse_numbers=external_pn,
        track_val=int(track_val) if track_val is not None else None,
        external_pn_add=external_pn_add,
        bbat_mjd=phase_bbat_mjd,
        torb_sec=phase_torb_sec,
        use_native_bbat_phase5=use_native_bbat_phase5,
        addsat_sec=addsat_sec,
        mean_mode=phase_mean_mode,
    )
