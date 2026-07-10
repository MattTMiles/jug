"""Step 16-18 probe helpers for native-chain tests (no oracle imports)."""

from __future__ import annotations

import numpy as np

from jug.utils.constants import SECS_PER_DAY


def formbats_correction_tt_sec(
    correction_tt_sec: np.ndarray,
    *,
    utc_to_tdb_sec: np.ndarray | None = None,
    formbats_tt_sec: np.ndarray | None = None,
) -> np.ndarray:
    """Map JUG clock exports to tempo2 ``getCorrectionTT`` for formBats."""
    if formbats_tt_sec is not None:
        return np.asarray(formbats_tt_sec, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    if utc_to_tdb_sec is None:
        return tt
    return tt + np.asarray(utc_to_tdb_sec, dtype=np.float64)


def compute_formbats_effective_shapiro_sec(
    shapiro_sun_sec: np.ndarray,
    shapiro_planets_stored_sec: np.ndarray,
    *,
    planet_shapiro: float = 1.0,
    calc_shapiro: bool = True,
) -> np.ndarray:
    """Tempo2 ``formBats.C`` effective Shapiro (matches pytempo ``shapiro_delay_sec``)."""
    if not calc_shapiro:
        return np.zeros_like(np.asarray(shapiro_sun_sec, dtype=np.float64))
    sun = np.asarray(shapiro_sun_sec, dtype=np.float64)
    planets = np.asarray(shapiro_planets_stored_sec, dtype=np.float64)
    return sun + float(planet_shapiro) * planets


def batcorr_from_model_epoch(
    model_mjd: np.ndarray,
    sat_mjd: np.ndarray,
    prebinary_delay_sec: np.ndarray,
) -> np.ndarray:
    """JUG IFTE model-epoch batCorr identity: (model−sat)×86400 − prebinary."""
    model = np.asarray(model_mjd, dtype=np.float64)
    sat = np.asarray(sat_mjd, dtype=np.float64)
    prebin = np.asarray(prebinary_delay_sec, dtype=np.float64)
    return (model - sat) * SECS_PER_DAY - prebin


def formbats_replay_batcorr_sec(
    correction_tt_sec: np.ndarray,
    correction_tt_tb_sec: np.ndarray,
    tropospheric_sec: np.ndarray,
    roemer_sec: np.ndarray,
    shapiro_sun_sec: np.ndarray,
    shapiro_planets_sec: np.ndarray,
    tdis1_sec: np.ndarray,
    tdis2_sec: np.ndarray,
) -> np.ndarray:
    """Replay formBats.C L67-L71 with JUG roemer sign (negative projection)."""
    roemer_t2 = -np.asarray(roemer_sec, dtype=np.float64)
    shap = np.asarray(shapiro_sun_sec, dtype=np.float64) + np.asarray(
        shapiro_planets_sec, dtype=np.float64
    )
    return formbats_replay_batcorr_tempo2_sec(
        correction_tt_sec,
        correction_tt_tb_sec,
        tropospheric_sec,
        roemer_t2,
        shap,
        tdis1_sec,
        tdis2_sec,
    )


def formbats_replay_batcorr_tempo2_sec(
    correction_tt_sec: np.ndarray,
    correction_tt_tb_sec: np.ndarray,
    tropospheric_sec: np.ndarray,
    roemer_tempo2_sec: np.ndarray,
    shapiro_delay_sec: np.ndarray,
    tdis1_sec: np.ndarray,
    tdis2_sec: np.ndarray,
) -> np.ndarray:
    """Replay formBats.C with tempo2 obsn sign conventions (roemer added)."""
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    tt_tb = np.asarray(correction_tt_tb_sec, dtype=np.float64)
    tropo = np.asarray(tropospheric_sec, dtype=np.float64)
    roemer = np.asarray(roemer_tempo2_sec, dtype=np.float64)
    shap = np.asarray(shapiro_delay_sec, dtype=np.float64)
    tdis1 = np.asarray(tdis1_sec, dtype=np.float64)
    tdis2 = np.asarray(tdis2_sec, dtype=np.float64)
    return tt + tt_tb - tropo + roemer - shap - tdis1 - tdis2


def rms_ns(delta: np.ndarray, *, demean: bool = False, is_mjd: bool = False) -> float:
    arr = np.asarray(delta, dtype=np.float64)
    if is_mjd:
        arr = arr * SECS_PER_DAY
    if demean:
        arr = arr - np.mean(arr)
    return float(np.sqrt(np.mean(arr**2)) * 1e9)
