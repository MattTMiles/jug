"""BBAT lite kernel for ``fixed_state_stripped`` tempo2 JAX fitting."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jug.residuals.tempo2.calculate_bclt_jax import compute_bclt_terms_fixed_state_jax
from jug.residuals.tempo2.formbats_jax import (
    compute_formbats_daysec,
    compute_shklovskii_sec_jax_pure_daysec,
)
from jug.residuals.tempo2.model.static import compute_dm_vals_jax, planet_rsa_tuple_jax_from_dict

if TYPE_CHECKING:
    from jug.residuals.tempo2.common import NativeDeltaPack


@partial(
    jax.jit,
    static_argnames=(
        "ne_sw",
        "posepoch_mjd",
        "pmrv_rad_century",
        "dilate_freq",
        "si_units",
        "units_tdb",
        "planet_shapiro_enabled",
        "dm_epoch",
        "dm_coeffs",
        "use_native_ecliptic",
        "shk_posepoch",
    ),
)
def compute_bbat_lite_daysec_jax(
    *,
    params_pos: jnp.ndarray,
    params_vel: jnp.ndarray,
    params_acc: jnp.ndarray,
    sat_mjd: jnp.ndarray,
    sat_int_day: jnp.ndarray,
    sat_sec_in_day: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    tropo_sec: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    correction_tt_tb_sec: jnp.ndarray,
    einstein_rate: jnp.ndarray,
    dt_ssb_ref_sec: jnp.ndarray,
    earth_ssb_km: jnp.ndarray,
    site_vel_km_s: jnp.ndarray,
    ssb_obs_ls: jnp.ndarray,
    obs_sun_ls: jnp.ndarray,
    obs_jupiter_ls: jnp.ndarray,
    planet_obs_ls: dict[str, jnp.ndarray] | None,
    dm_vals: jnp.ndarray | None,
    dm_epoch: float,
    dm_coeffs: tuple[float, ...],
    ne_sw: float,
    posepoch_mjd: float,
    parallax_mas: jnp.ndarray,
    pmrv_rad_century: float,
    dilate_freq: bool,
    si_units: bool,
    units_tdb: bool,
    planet_shapiro_enabled: bool,
    use_native_ecliptic: bool,
    dshk: jnp.ndarray,
    pmra: jnp.ndarray,
    pmdec: jnp.ndarray,
    shk_posepoch: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-pass BCLT + tempo2 ``formBats``/``secularMotion`` BBAT daysec (no phase5)."""
    if dm_vals is None:
        dm_vals = compute_dm_vals_jax(sat_mjd, dm_epoch=dm_epoch, dm_coeffs=dm_coeffs)
    tt = jnp.asarray(correction_tt_sec, dtype=jnp.float64)
    tt_tb = jnp.asarray(correction_tt_tb_sec, dtype=jnp.float64)
    einstein = jnp.asarray(einstein_rate, dtype=jnp.float64)
    if planet_obs_ls is None:
        planet_obs_ls = {"jupiter": obs_jupiter_ls}
    planet_rsa = planet_rsa_tuple_jax_from_dict(
        planet_obs_ls,
        n_toa=int(sat_mjd.shape[0]),
        obs_jupiter_ls=obs_jupiter_ls,
    )
    bclt = compute_bclt_terms_fixed_state_jax(
        sat_mjd=sat_mjd,
        correction_tt_sec=tt,
        correction_tt_tb_sec=tt_tb,
        dt_ssb_ref_sec=dt_ssb_ref_sec,
        ssb_obs_ls=ssb_obs_ls,
        obs_sun_ls=obs_sun_ls,
        freq_mhz=freq_mhz,
        earth_ssb_vel_km_s=earth_ssb_km[:, 3:6],
        site_vel_km_s=site_vel_km_s,
        dm_vals=dm_vals,
        pos_pulsar=params_pos,
        vel_pulsar=params_vel,
        acc_pulsar=params_acc,
        posepoch_mjd=posepoch_mjd,
        parallax_mas=parallax_mas,
        pmrv_rad_century=pmrv_rad_century,
        ne_sw=ne_sw,
        einstein_rate=einstein,
        dilate_freq=dilate_freq,
        planet_shapiro_enabled=planet_shapiro_enabled,
        obs_jupiter_ls=obs_jupiter_ls,
        planet_obs_ls=planet_rsa,
    )
    shap_delay = bclt.shapiro_sun_sec + jnp.where(
        planet_shapiro_enabled,
        bclt.shapiro_planets_sec,
        0.0,
    )
    tropo = jnp.asarray(tropo_sec, dtype=jnp.float64)
    shk_pep_int = jnp.floor(jnp.asarray(shk_posepoch, dtype=jnp.float64))
    shk_pep_frac = jnp.asarray(shk_posepoch, dtype=jnp.float64) - shk_pep_int
    _, _, bat_int, bat_sec, _, _ = compute_formbats_daysec(
        sat_int_day,
        sat_sec_in_day,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        jnp.zeros_like(sat_mjd),
    )
    shk = compute_shklovskii_sec_jax_pure_daysec(
        bat_int,
        bat_sec,
        shk_pep_int,
        shk_pep_frac,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
    )
    _, _, _, _, bbat_int, bbat_sec = compute_formbats_daysec(
        sat_int_day,
        sat_sec_in_day,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        shk,
    )
    return bbat_int, bbat_sec


def bbat_lite_daysec_from_pack(params: dict, pack: "NativeDeltaPack") -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the BBAT lite kernel from a ``NativeDeltaPack`` and parameter dict."""
    from jug.residuals.tempo2.common import (
        _dm_coeffs_jax,
        _param_scalar_jax,
        pulsar_vectors_from_params_jax,
    )

    pos, vel, acc = pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    return compute_bbat_lite_daysec_jax(
        params_pos=jnp.asarray(pos, dtype=jnp.float64),
        params_vel=jnp.asarray(vel, dtype=jnp.float64),
        params_acc=jnp.asarray(acc, dtype=jnp.float64),
        sat_mjd=pack.sat_mjd,
        sat_int_day=pack.sat_int_day,
        sat_sec_in_day=pack.sat_sec_in_day,
        freq_mhz=pack.freq_mhz,
        tropo_sec=pack.tropo_sec,
        correction_tt_sec=pack.correction_tt_sec,
        correction_tt_tb_sec=pack.correction_tt_tb_sec,
        einstein_rate=pack.einstein_rate,
        dt_ssb_ref_sec=pack.dt_ssb_ref_sec,
        earth_ssb_km=pack.earth_ssb_km,
        site_vel_km_s=pack.site_vel_km_s,
        ssb_obs_ls=pack.ssb_obs_ls,
        obs_sun_ls=pack.obs_sun_ls,
        obs_jupiter_ls=pack.obs_jupiter_ls,
        planet_obs_ls=pack.planet_obs_ls,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        ne_sw=float(pack.ne_sw),
        posepoch_mjd=float(pack.posepoch_mjd),
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=float(pack.pmrv_rad_century),
        dilate_freq=bool(pack.dilate_freq),
        si_units=bool(pack.si_units),
        units_tdb=bool(pack.units_tdb),
        planet_shapiro_enabled=bool(pack.planet_shapiro_enabled),
        use_native_ecliptic=bool(pack.use_native_ecliptic),
        dshk=jnp.asarray(_param_scalar_jax(params, "DSHK", pack.dshk), dtype=jnp.float64),
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=float(pack.shk_posepoch),
    )