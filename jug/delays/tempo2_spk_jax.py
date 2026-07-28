"""JAX-safe JPL SPK Chebyshev evaluation (tempo2 ``readEphemeris.C``)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_ephemeris import (
    _NAIF_EMB,
    _NAIF_EARTH,
    _NAIF_SUN,
    _PLANET_BARY,
    _SSB,
    _open_spk,
    _pos_vel_km,
    _segment_pair,
    mjd_to_jd,
)
from jug.utils.constants import SECS_PER_DAY

T0 = 2451545.0
S_PER_DAY = SECS_PER_DAY


class SpkSegmentPacked(NamedTuple):
    """Chebyshev coefficients for one SPK segment (type 2/3)."""

    init: jnp.ndarray
    intlen: jnp.ndarray
    coefficients: jnp.ndarray  # (coeff_count, component_count, n_intervals)


class Tempo2SpkPacked(NamedTuple):
    """Static SPK tables for in-graph tempo2 ephemeris."""

    emb_ssb: SpkSegmentPacked
    earth_emb: SpkSegmentPacked
    sun_ssb: SpkSegmentPacked
    planets_ssb: dict[str, SpkSegmentPacked]


def _pack_segment(segment) -> SpkSegmentPacked:
    init, intlen, coefficients = segment._data
    return SpkSegmentPacked(
        init=np.asarray(init, dtype=np.float64),
        intlen=np.asarray(intlen, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
    )


@lru_cache(maxsize=4)
def pack_tempo2_spk_jax(ephem_path: str) -> Tempo2SpkPacked:
    """Load SPK once on host; return immutable JAX coefficient arrays."""
    kernel = _open_spk(ephem_path)
    planets = {
        name: _pack_segment(_segment_pair(kernel, _SSB, bary))
        for name, bary in _PLANET_BARY.items()
    }
    return Tempo2SpkPacked(
        emb_ssb=_pack_segment(_segment_pair(kernel, _SSB, _NAIF_EMB)),
        earth_emb=_pack_segment(_segment_pair(kernel, _NAIF_EMB, _NAIF_EARTH)),
        sun_ssb=_pack_segment(_segment_pair(kernel, _SSB, _NAIF_SUN)),
        planets_ssb=planets,
    )


def _chebyshev_index_offset(jd: jnp.ndarray, init: jnp.ndarray, intlen: jnp.ndarray, n: int):
    """Interval index and offset within interval (jplephem ``Segment.generate``)."""
    tdb = jnp.asarray(jd, dtype=jnp.float64)
    tdb2 = jnp.zeros_like(tdb)
    index1, offset1 = jnp.divmod((tdb - T0) * S_PER_DAY - init, intlen)
    index2, offset2 = jnp.divmod(tdb2 * S_PER_DAY, intlen)
    index3, offset = jnp.divmod(offset1 + offset2, intlen)
    index = (index1 + index2 + index3).astype(jnp.int32)
    omegas = index == n
    index = jnp.where(omegas, index - 1, index)
    offset = jnp.where(omegas, offset + intlen, offset)
    return index, offset


def eval_spk_chebyshev_jax(
    jd: jnp.ndarray,
    segment: SpkSegmentPacked,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate SPK position (km) and velocity (km/s) at Julian Date ``jd``."""
    init = segment.init
    intlen = segment.intlen
    coefficients = segment.coefficients
    coeff_count, component_count, n = coefficients.shape
    index, offset = _chebyshev_index_offset(jd, init, intlen, int(n))
    coeffs = coefficients[:, :, index]

    s = 2.0 * offset / intlen - 1.0
    s2 = 2.0 * s
    w0 = jnp.asarray(0.0, dtype=jnp.float64)
    w1 = jnp.asarray(0.0, dtype=jnp.float64)
    wlist = []
    for i in range(int(coeff_count) - 1):
        coefficient = coeffs[i]
        w2 = w1
        w1 = w0
        w0 = coefficient + (s2 * w1 - w2)
        wlist.append(w1)
    components = coeffs[-1] + (s * w0 - w1)

    dw0 = jnp.asarray(0.0, dtype=jnp.float64)
    dw1 = jnp.asarray(0.0, dtype=jnp.float64)
    for i, w1_val in enumerate(wlist):
        coefficient = coeffs[i]
        dw2 = dw1
        dw1 = dw0
        dw0 = 2.0 * w1_val + dw1 * s2 - dw2
    rates = w0 + s * dw0 - dw1
    rates = rates / intlen * 2.0 * S_PER_DAY
    rates = rates / S_PER_DAY
    return components, rates


def earth_geocenter_from_ssb_jax(
    jd: jnp.ndarray,
    spk: Tempo2SpkPacked,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Earth geocenter position/velocity w.r.t. SSB (km, km/s)."""
    emb_pos, emb_vel = eval_spk_chebyshev_jax(jd, spk.emb_ssb)
    earth_pos, earth_vel = eval_spk_chebyshev_jax(jd, spk.earth_emb)
    return emb_pos + earth_pos, emb_vel + earth_vel


def sun_from_ssb_jax(jd: jnp.ndarray, spk: Tempo2SpkPacked) -> tuple[jnp.ndarray, jnp.ndarray]:
    return eval_spk_chebyshev_jax(jd, spk.sun_ssb)


def planet_from_ssb_jax(
    jd: jnp.ndarray,
    spk: Tempo2SpkPacked,
    planet: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return eval_spk_chebyshev_jax(jd, spk.planets_ssb[planet])


def mjd_to_jd_jax(mjd: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(mjd, dtype=jnp.float64) + 2400000.5


def eval_spk_host_reference(jd: float, segment_packed: SpkSegmentPacked, kernel, center: int, target: int):
    """Host numpy reference for one JD (tests only)."""
    pos, vel = _pos_vel_km(kernel, center, target, jd)
    jax_pos, jax_vel = eval_spk_chebyshev_jax(
        jnp.asarray(jd, dtype=jnp.float64), segment_packed
    )
    return (
        np.asarray(jax.device_get(jax_pos), dtype=np.float64),
        np.asarray(jax.device_get(jax_vel), dtype=np.float64),
        pos,
        vel,
    )
