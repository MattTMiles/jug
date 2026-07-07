"""JAX-safe observatory ITRF→GCRS site motion (Astropy ``get_gcrs_posvel``)."""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jug.utils.constants import SECS_PER_DAY

OMEGA_EARTH = 7.2921150e-5  # rad/s


class IersEopPacked(NamedTuple):
    """Static IERS Earth-orientation table for in-graph interpolation."""

    mjd: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    dut1: np.ndarray


@lru_cache(maxsize=1)
def pack_iers_eop_jax() -> IersEopPacked:
    """Load Astropy IERS-B table once; pack as JAX arrays."""
    from astropy.utils.iers import IERS_B

    table = IERS_B.open()
    mjd = np.asarray(table["MJD"].value, dtype=np.float64)
    xp = np.asarray(table["PM_x"].value, dtype=np.float64)  # arcsec
    yp = np.asarray(table["PM_y"].value, dtype=np.float64)
    dut1 = np.asarray(table["UT1_UTC"].value, dtype=np.float64)  # seconds
    return IersEopPacked(
        mjd=np.asarray(mjd, dtype=np.float64),
        xp=np.asarray(xp, dtype=np.float64),
        yp=np.asarray(yp, dtype=np.float64),
        dut1=np.asarray(dut1, dtype=np.float64),
    )


def _interp_eop(mjd: jnp.ndarray, table_mjd: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    mjd = jnp.asarray(mjd, dtype=jnp.float64)
    idx = jnp.searchsorted(table_mjd, mjd, side="right")
    idx = jnp.clip(idx, 1, table_mjd.size - 1)
    m0 = table_mjd[idx - 1]
    m1 = table_mjd[idx]
    v0 = values[idx - 1]
    v1 = values[idx]
    frac = (mjd - m0) / jnp.maximum(m1 - m0, 1e-30)
    out = v0 + frac * (v1 - v0)
    return jnp.where(mjd <= table_mjd[0], values[0], out)


def _mjd_to_jd12(mjd: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    jd = jnp.asarray(mjd, dtype=jnp.float64) + 2400000.5
    jd1 = jnp.floor(jd)
    jd2 = jd - jd1
    return jd1, jd2


def _gcrs_to_cirs_mat_jax(jd1: jnp.ndarray, jd2: jnp.ndarray) -> jnp.ndarray:
    """Batched celestial-to-intermediate matrix via host ERFA."""
    shape = jd1.shape
    jd1_flat = jnp.reshape(jd1, (-1,))
    jd2_flat = jnp.reshape(jd2, (-1,))
    n = jd1_flat.size

    def callback(j1, j2):
        import erfa

        out = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            out[i] = erfa.c2i06a(float(j1[i]), float(j2[i]))
        return out

    mats = jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((n, 3, 3), jnp.float64),
        jd1_flat,
        jd2_flat,
        vmap_method="broadcast_all",
    )
    return jnp.reshape(mats, shape + (3, 3))


def _mjd_tt_to_ut1_jd12_jax(mjd_tt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """TT MJD → UT1 two-part JD (Astropy scale chain)."""
    mjd = jnp.atleast_1d(jnp.asarray(mjd_tt, dtype=jnp.float64))
    n = mjd.size

    def callback(mjd_v):
        from astropy.coordinates.builtin_frames.utils import get_jd12
        from astropy.time import Time

        jd1 = np.zeros(n, dtype=np.float64)
        jd2 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            j1, j2 = get_jd12(Time(float(mjd_v[i]), format="mjd", scale="tt"), "ut1")
            jd1[i] = j1
            jd2[i] = j2
        return jd1, jd2

    jd1, jd2 = jax.pure_callback(
        callback,
        (
            jax.ShapeDtypeStruct((n,), jnp.float64),
            jax.ShapeDtypeStruct((n,), jnp.float64),
        ),
        mjd,
        vmap_method="broadcast_all",
    )
    return jd1, jd2


def _polar_motion_jax(mjd_tt: jnp.ndarray, eop: IersEopPacked) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Polar motion at TT MJD via Astropy (matches ``get_polar_motion``)."""
    mjd = jnp.atleast_1d(jnp.asarray(mjd_tt, dtype=jnp.float64))
    n = mjd.size

    def callback(mjd_v):
        from astropy.coordinates.builtin_frames.utils import get_polar_motion
        from astropy.time import Time

        xp = np.zeros(n, dtype=np.float64)
        yp = np.zeros(n, dtype=np.float64)
        for i in range(n):
            xpi, ypi = get_polar_motion(Time(float(mjd_v[i]), format="mjd", scale="tt"))
            xp[i] = xpi
            yp[i] = ypi
        return xp, yp

    return jax.pure_callback(
        callback,
        (
            jax.ShapeDtypeStruct((n,), jnp.float64),
            jax.ShapeDtypeStruct((n,), jnp.float64),
        ),
        mjd,
        vmap_method="broadcast_all",
    )


def cirs_to_itrs_mat_jax(
    mjd_tt: jnp.ndarray,
    eop: IersEopPacked,
) -> jnp.ndarray:
    """CIRS-to-ITRS rotation matrices at TT MJD (Astropy convention)."""
    mjd = jnp.atleast_1d(jnp.asarray(mjd_tt, dtype=jnp.float64))
    jd1_tt, jd2_tt = _mjd_to_jd12(mjd)
    jd1_ut1, jd2_ut1 = _mjd_tt_to_ut1_jd12_jax(mjd)
    xp, yp = _polar_motion_jax(mjd, eop)
    n = mjd.size

    def callback(j1_tt, j2_tt, j1_ut1, j2_ut1, xp_r, yp_r):
        import erfa

        out = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            sp = erfa.sp00(float(j1_tt[i]), float(j2_tt[i]))
            rpom = erfa.pom00(float(xp_r[i]), float(yp_r[i]), sp)
            era = erfa.era00(float(j1_ut1[i]), float(j2_ut1[i]))
            out[i] = erfa.c2tcio(np.eye(3), era, rpom)
        return out

    return jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((n, 3, 3), jnp.float64),
        jnp.reshape(jd1_tt, (-1,)),
        jnp.reshape(jd2_tt, (-1,)),
        jnp.reshape(jd1_ut1, (-1,)),
        jnp.reshape(jd2_ut1, (-1,)),
        jnp.reshape(xp, (-1,)),
        jnp.reshape(yp, (-1,)),
        vmap_method="broadcast_all",
    )


def gcrs_to_cirs_mat_jax(mjd_tt: jnp.ndarray) -> jnp.ndarray:
    mjd = jnp.atleast_1d(jnp.asarray(mjd_tt, dtype=jnp.float64))
    jd1, jd2 = _mjd_to_jd12(mjd)
    return _gcrs_to_cirs_mat_jax(jd1, jd2)


def observatory_earth_km_jax(
    site_mjd: jnp.ndarray,
    obs_itrf_km: jnp.ndarray,
    *,
    eop: IersEopPacked,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GCRS observatory position/velocity (km, km/s) at ``site_mjd`` (TT).

    Mirrors Astropy ``EarthLocation.get_gcrs_posvel`` (CIRS route).
    """
    obs = jnp.asarray(obs_itrf_km, dtype=jnp.float64).reshape(3)
    mjd = jnp.atleast_1d(jnp.asarray(site_mjd, dtype=jnp.float64))
    ref_to_itrs = cirs_to_itrs_mat_jax(mjd, eop)
    gcrs_to_ref = gcrs_to_cirs_mat_jax(mjd)
    ref_to_gcrs = jnp.swapaxes(gcrs_to_ref, -1, -2)
    itrs_to_gcrs = ref_to_gcrs @ jnp.swapaxes(ref_to_itrs, -1, -2)

    pos = jnp.einsum("...ij,j->...i", itrs_to_gcrs, obs)
    rot_vec = ref_to_gcrs[..., 2, :] * OMEGA_EARTH
    vel = jnp.cross(rot_vec, pos)
    return pos, vel


def observatory_earth_state_jax(
    site_mjd: jnp.ndarray,
    obs_itrf_km: jnp.ndarray,
    *,
    eop: IersEopPacked,
) -> jnp.ndarray:
    """Return ``(N, 6)`` observatory geocenter state in km / km/s."""
    pos, vel = observatory_earth_km_jax(site_mjd, obs_itrf_km, eop=eop)
    return jnp.concatenate([pos, vel], axis=-1)
