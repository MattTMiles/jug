"""In-graph Tempo2 ephemeris geometry (SPK + site motion + Teph bootstrap)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from jug.delays.tempo2_ephemeris import tempo2_read_ephemeris_au_scale
from jug.delays.tempo2_spk_jax import (
    Tempo2SpkPacked,
    earth_geocenter_from_ssb_jax,
    mjd_to_jd_jax,
    planet_from_ssb_jax,
    sun_from_ssb_jax,
)
from jug.delays.tempo2_site_jax import IersEopPacked, observatory_earth_state_jax
from jug.utils.constants import C_KM_S, SECS_PER_DAY


class Tempo2ObservatoryStateJax(NamedTuple):
    """In-graph observatory/ephemeris vectors (km / km/s + light-seconds)."""

    earth_ssb_km: jnp.ndarray
    observatory_earth_km: jnp.ndarray
    sun_ssb_km: jnp.ndarray
    planet_ssb_km: dict[str, jnp.ndarray]
    site_vel_km_s: jnp.ndarray
    ssb_obs_ls: jnp.ndarray
    obs_sun_ls: jnp.ndarray
    obs_jupiter_ls: jnp.ndarray


def _scale_ephemeris_km(arr: jnp.ndarray, *, si_units: bool) -> jnp.ndarray:
    scale = jnp.asarray(tempo2_read_ephemeris_au_scale(si_units=si_units), dtype=jnp.float64)
    return arr * scale


def _stack_planet_shapiro_rsa_jax(
    geom: "Tempo2ObservatoryStateJax",
    names: tuple[str, ...] = ("venus", "jupiter", "saturn", "uranus", "neptune"),
) -> tuple[jnp.ndarray, ...]:
    """Tempo2 ``shapiro_delay.C`` rsa vectors (body→observatory) in light-seconds.

    Uses geocentric ``planet_earth`` (``jpl_pleph(N,3)``), not ``planet_ssb - ssb_obs``:
    ``rsa = observatory_earth - (planet_ssb - earth_ssb)``.
    """
    obs = geom.observatory_earth_km[:, :3]
    earth = geom.earth_ssb_km[:, :3]
    out = []
    for name in names:
        pv = geom.planet_ssb_km.get(name)
        if pv is None:
            out.append(jnp.zeros((obs.shape[0], 3), dtype=jnp.float64))
        else:
            planet_geo = pv[:, :3] - earth
            out.append((obs - planet_geo) / C_KM_S)
    return tuple(out)


def _stack_planet_obs_ls_jax(
    geom: "Tempo2ObservatoryStateJax",
    names: tuple[str, ...] = ("venus", "jupiter", "saturn", "uranus", "neptune"),
) -> tuple[jnp.ndarray, ...]:
    """Alias retained for callers; vectors are Tempo2 rsa (see ``_stack_planet_shapiro_rsa_jax``)."""
    return _stack_planet_shapiro_rsa_jax(geom, names=names)


def observatory_chain_vectors_jax(
    earth_ssb_km: jnp.ndarray,
    observatory_earth_km: jnp.ndarray,
    sun_ssb_km: jnp.ndarray,
    planet_ssb_km: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ssb_obs_km = earth_ssb_km[:, :3] + observatory_earth_km[:, :3]
    ssb_obs_ls = ssb_obs_km / C_KM_S
    obs_sun_ls = (sun_ssb_km[:, :3] - ssb_obs_km) / C_KM_S
    jup = planet_ssb_km.get("jupiter")
    if jup is None:
        obs_jupiter_ls = jnp.zeros((ssb_obs_km.shape[0], 3), dtype=jnp.float64)
    else:
        obs_jupiter_ls = (jup[:, :3] - ssb_obs_km) / C_KM_S
    return ssb_obs_ls, obs_sun_ls, obs_jupiter_ls


def compute_tempo2_observatory_state_jax(
    ephem_mjd: jnp.ndarray,
    *,
    site_mjd: jnp.ndarray,
    obs_itrf_km: jnp.ndarray,
    spk: Tempo2SpkPacked,
    eop: IersEopPacked,
    si_units: bool = True,
    planet_names: tuple[str, ...] = (
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ),
) -> Tempo2ObservatoryStateJax:
    """Compute Earth/Sun/planet and site vectors inside the JIT graph."""
    jd = mjd_to_jd_jax(ephem_mjd)

    def one_epoch(jd_i):
        earth_pos, earth_vel = earth_geocenter_from_ssb_jax(jd_i, spk)
        sun_pos, sun_vel = sun_from_ssb_jax(jd_i, spk)
        return jnp.concatenate([earth_pos, earth_vel]), jnp.concatenate([sun_pos, sun_vel])

    earth_ssb, sun_ssb = jax.vmap(one_epoch)(jd)
    earth_ssb = _scale_ephemeris_km(earth_ssb, si_units=si_units)
    sun_ssb = _scale_ephemeris_km(sun_ssb, si_units=si_units)

    planet_ssb: dict[str, jnp.ndarray] = {}
    for name in planet_names:
        if name not in spk.planets_ssb:
            continue

        def one_planet(jd_i, planet=name):
            pos, vel = planet_from_ssb_jax(jd_i, spk, planet)
            return jnp.concatenate([pos, vel])

        planet_ssb[name] = _scale_ephemeris_km(jax.vmap(one_planet)(jd), si_units=si_units)

    observatory_earth = observatory_earth_state_jax(site_mjd, obs_itrf_km, eop=eop)
    site_vel = observatory_earth[:, 3:6]
    ssb_obs_ls, obs_sun_ls, obs_jupiter_ls = observatory_chain_vectors_jax(
        earth_ssb, observatory_earth, sun_ssb, planet_ssb
    )
    return Tempo2ObservatoryStateJax(
        earth_ssb_km=earth_ssb,
        observatory_earth_km=observatory_earth,
        sun_ssb_km=sun_ssb,
        planet_ssb_km=planet_ssb,
        site_vel_km_s=site_vel,
        ssb_obs_ls=ssb_obs_ls,
        obs_sun_ls=obs_sun_ls,
        obs_jupiter_ls=obs_jupiter_ls,
    )


def bootstrap_tempo2_geometry_jax(
    sat_mjd: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    *,
    obs_itrf_km: jnp.ndarray,
    spk: Tempo2SpkPacked,
    eop: IersEopPacked,
    ifte_records: jnp.ndarray,
    ifte_start_jd: jnp.ndarray,
    ifte_end_jd: jnp.ndarray,
    ifte_step_jd: jnp.ndarray,
    ifte_coef_offset: int,
    ifte_ncf: int,
    ifte_na: int,
    si_units: bool = True,
    units_tdb: bool = True,
    max_iter: int = 8,
    tol: float = 1.0e-15,
) -> tuple[jnp.ndarray, Tempo2ObservatoryStateJax]:
    """Fixed-point Teph ↔ ephemeris bootstrap inside the JIT graph."""
    from jug.residuals.tempo2_native.clock_jax import compute_tempo2_correction_tt_tb_jax

    site_mjd = sat_mjd + correction_tt_sec / SECS_PER_DAY
    mjd_tt = site_mjd

    def geom_and_tt_tb(tt_teph_val):
        state = compute_tempo2_observatory_state_jax(
            site_mjd + tt_teph_val / SECS_PER_DAY,
            site_mjd=site_mjd,
            obs_itrf_km=obs_itrf_km,
            spk=spk,
            eop=eop,
            si_units=si_units,
        )
        tt_tb, tt_teph_new = compute_tempo2_correction_tt_tb_jax(
            mjd_tt,
            state.observatory_earth_km[:, :3],
            state.earth_ssb_km[:, 3:6],
            ifte_records=ifte_records,
            ifte_start_jd=ifte_start_jd,
            ifte_end_jd=ifte_end_jd,
            ifte_step_jd=ifte_step_jd,
            ifte_coef_offset=ifte_coef_offset,
            ifte_ncf=ifte_ncf,
            ifte_na=ifte_na,
            units_tdb=units_tdb,
            si_units=si_units,
        )
        return tt_tb, tt_teph_new, state

    def cond_fn(carry):
        _tt_teph, _tt_tb, _state, delta, n = carry
        return (delta >= tol) & (n < max_iter)

    def body_fn(carry):
        tt_teph_val, _tt_tb, _state, _delta, n = carry
        tt_tb, tt_teph_new, state = geom_and_tt_tb(tt_teph_val)
        delta = jnp.max(jnp.abs(tt_teph_new - tt_teph_val))
        return tt_teph_new, tt_tb, state, delta, n + 1

    tt_tb0, tt_teph0, state0 = geom_and_tt_tb(jnp.zeros_like(sat_mjd))
    tt_teph_f, tt_tb_f, state_f, _delta, _n = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            tt_teph0,
            tt_tb0,
            state0,
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    tt_tb_final, _tt_teph_final, state_final = geom_and_tt_tb(tt_teph_f)
    return tt_tb_final, state_final
