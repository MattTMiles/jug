"""JAX tempo2 clock corrections (``clkcorr.C`` / ``tt2tdb.C``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jug.utils.constants import C_KM_S, SECS_PER_DAY
from jug.utils.ifteph import IFTE_LC, IFTE_MJD0, IFTE_TEPH0_SEC, ifte_delta_t_mjd
from jug.utils.timescales import IFTE_K, IFTE_KM1


def interpolate_clock_jax(mjd: jnp.ndarray, mjd_table: jnp.ndarray, offset_table: jnp.ndarray) -> jnp.ndarray:
    """Linear clock interpolation on a continuous MJD grid (JAX)."""
    mjd = jnp.asarray(mjd, dtype=jnp.float64)
    mjd_table = jnp.asarray(mjd_table, dtype=jnp.float64)
    offset_table = jnp.asarray(offset_table, dtype=jnp.float64)
    if mjd_table.size == 0:
        return jnp.zeros_like(mjd)
    idx = jnp.searchsorted(mjd_table, mjd, side="right")
    idx = jnp.clip(idx, 1, mjd_table.size - 1)
    mjd0 = mjd_table[idx - 1]
    mjd1 = mjd_table[idx]
    off0 = offset_table[idx - 1]
    off1 = offset_table[idx]
    frac = (mjd - mjd0) / jnp.maximum(mjd1 - mjd0, 1e-30)
    out = off0 + frac * (off1 - off0)
    out = jnp.where(mjd <= mjd_table[0], 0.0, out)
    out = jnp.where(mjd >= mjd_table[-1], 0.0, out)
    return out


def compute_tempo2_get_correction_tt_jax(
    sat_mjd: jnp.ndarray,
    *,
    chain_mjd_tables: tuple[jnp.ndarray, ...],
    chain_offset_tables: tuple[jnp.ndarray, ...],
    bipm_mjd: jnp.ndarray,
    bipm_offset: jnp.ndarray,
    feedback_iters: int = 3,
) -> jnp.ndarray:
    """Sum a pre-resolved UTC→TAI clock chain with ``sat+corr/SECDAY`` feedback."""
    sat = jnp.asarray(sat_mjd, dtype=jnp.float64)

    def one_iter(corr):
        mjd_eval = sat + corr / SECS_PER_DAY
        total = jnp.zeros_like(sat)
        for mjd_tab, off_tab in zip(chain_mjd_tables, chain_offset_tables):
            total = total + interpolate_clock_jax(mjd_eval, mjd_tab, off_tab)
        bipm = interpolate_clock_jax(mjd_eval, bipm_mjd, bipm_offset) - 32.184
        return total + bipm

    corr = jnp.zeros_like(sat)
    for _ in range(max(1, int(feedback_iters))):
        corr = one_iter(corr)
    return corr


def compute_tempo2_correction_tt_tb_jax(
    mjd_tt: jnp.ndarray,
    observatory_earth_km: jnp.ndarray,
    earth_ssb_vel_km_s: jnp.ndarray,
    *,
    units_tdb: bool = True,
    si_units: bool = False,
) -> jnp.ndarray:
    """``tt2tdb.C`` ``correctionTT_TB`` in seconds (JAX)."""
    ifte_k = jnp.asarray(float(IFTE_K), dtype=jnp.float64)
    ifte_km1 = jnp.asarray(float(IFTE_KM1), dtype=jnp.float64)
    mjd = jnp.asarray(mjd_tt, dtype=jnp.float64)
    obs_km = jnp.asarray(observatory_earth_km, dtype=jnp.float64)
    earth_vel = jnp.asarray(earth_ssb_vel_km_s, dtype=jnp.float64)
    delta_t = jnp.asarray(ifte_delta_t_mjd(np.asarray(mjd)), dtype=jnp.float64)
    obs_term = jnp.sum(obs_km * earth_vel, axis=-1) / (C_KM_S**2)
    obs_term = obs_term / (1.0 - IFTE_LC)
    if si_units:
        obs_term = obs_term / (ifte_k * ifte_k)
    else:
        obs_term = obs_term / ifte_k
    correction_teph = IFTE_TEPH0_SEC + obs_term + delta_t / (1.0 - IFTE_LC)
    if units_tdb and not si_units:
        return correction_teph
    linear = ifte_km1 * (mjd - IFTE_MJD0) * SECS_PER_DAY
    return linear + ifte_k * (correction_teph - IFTE_TEPH0_SEC)


def compute_einstein_rate_jax(mjd_tt: jnp.ndarray, *, si_units: bool = False) -> jnp.ndarray:
    """Simplified ``einsteinRate`` when ``DILATEFREQ=Y`` (JAX wrapper)."""
    from jug.delays.barycentric import compute_einstein_rate

    scale = "TCB" if si_units else "TDB"
    arr = np.asarray(jax.device_get(mjd_tt), dtype=np.float64)
    return jnp.asarray(compute_einstein_rate(arr, units=scale), dtype=jnp.float64)


def pack_clock_chain_jax(obs_chain: dict, bipm_clock: dict) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], jnp.ndarray, jnp.ndarray]:
    """Pack parsed clock dicts into JAX tables for ``compute_tempo2_get_correction_tt_jax``."""
    mjd_tables = []
    offset_tables = []
    if "mjd" in obs_chain and "offset" in obs_chain:
        mjd_tables.append(jnp.asarray(obs_chain["mjd"], dtype=jnp.float64))
        offset_tables.append(jnp.asarray(obs_chain["offset"], dtype=jnp.float64))
    for link in obs_chain.get("links", []):
        mjd_tables.append(jnp.asarray(link["mjd"], dtype=jnp.float64))
        offset_tables.append(jnp.asarray(link["offset"], dtype=jnp.float64))
    bipm_mjd = jnp.asarray(bipm_clock["mjd"], dtype=jnp.float64)
    bipm_off = jnp.asarray(bipm_clock["offset"], dtype=jnp.float64)
    return tuple(mjd_tables), tuple(offset_tables), bipm_mjd, bipm_off
