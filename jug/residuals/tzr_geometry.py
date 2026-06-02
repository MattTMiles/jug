"""Mode-specific TZR astrometry helpers (Phase C).

Splits absolute-phase geometry into pint-family and tempo2-native paths so
``_compute_tzr_phase`` can mirror the main-TOA provider dispatch from Phase B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

from jug.delays.barycentric import (
    compute_barycentric_freq,
    compute_ecliptic_pulsar_direction,
    compute_einstein_rate,
    compute_pulsar_direction,
    compute_roemer_delay,
    compute_shapiro_delay,
    compute_ssb_obs_pos_vel,
    rotate_equatorial_to_ecliptic,
)
from jug.delays.tempo2_ephemeris import (
    compute_tempo2_ephemeris_state,
    resolve_tempo2_ephemeris_path,
)
from jug.delays.tempo2_geometry import (
    build_pulsar_direction,
    compute_tempo2_roemer_sec,
    compute_tempo2_shapiro_sec,
    ecliptic_obliquity_rad,
    planet_shapiro_sec,
    ssb_obs_light_seconds,
    tempo2_equ2ecl,
)
from jug.io.par_reader import OBLIQUITY_ARCSEC
from jug.io.tim_reader import compute_tdb_standalone_vectorized
from jug.residuals.engine_conventions import EngineConventionProfile
from jug.utils.constants import C_KM_S, SECS_PER_DAY, T_PLANET, T_SUN_SEC
from jug.io.par_reader import get_longdouble


@dataclass
class TzrEpochs:
    """Resolved TZRMJD epochs for TZR geometry and phase."""

    tzrmjd_raw: np.longdouble
    tzrmjd_tdb: np.longdouble
    tzrmjd_model: np.longdouble
    tzrmjd_scale_resolved: str
    delta_tzr_sec: float


@dataclass
class TzrAstrometryTerms:
    """Astrometric delay terms at the TZR reference epoch."""

    L_hat: np.ndarray
    roemer_shapiro_sec: float
    freq_bary_mhz: float
    obs_sun_delay_km: np.ndarray
    ssb_obs_pos_km: np.ndarray
    use_native_ecliptic: bool
    obl_rad: float
    geometry_backend: str


def resolve_tzrmjd_epochs(
    *,
    params: dict[str, Any],
    tzrmjd_scale: str,
    tzr_is_ssb: bool,
    tzr_site: str,
    tzr_clock: Any,
    bipm_clock: Any,
    tzr_location: EarthLocation,
    model_timescale: str,
    engine_profile: EngineConventionProfile,
    verbose: bool,
) -> TzrEpochs:
    """Convert TZRMJD to TDB and model timescale with mode-specific AUTO rules.

    ``AUTO`` semantics:

    - ``compatibility="tempo2"`` and par ``UNITS=TDB``: TZRMJD is TDB (no conversion).
    - Otherwise: TZRMJD is UTC site arrival, converted via ``TZRSITE`` clock chain.
    """
    tzrmjd_raw = get_longdouble(params, "TZRMJD")
    tzrmjd_scale_upper = tzrmjd_scale.upper()
    model_timescale = str(model_timescale).upper()
    par_timescale = str(params.get("_par_timescale", "TDB")).upper()

    if tzr_is_ssb:
        if model_timescale == "TCB":
            from jug.utils.timescales import convert_tcb_epoch_to_tdb

            tzrmjd_model = tzrmjd_raw
            tzrmjd_tdb = convert_tcb_epoch_to_tdb(tzrmjd_raw)
        else:
            tzrmjd_model = tzrmjd_raw
            tzrmjd_tdb = tzrmjd_raw
        if verbose:
            print(f"   TZRMJD treated as {model_timescale} (TZRSITE=ssb, barycentric)")
        return TzrEpochs(
            tzrmjd_raw=tzrmjd_raw,
            tzrmjd_tdb=tzrmjd_tdb,
            tzrmjd_model=tzrmjd_model,
            tzrmjd_scale_resolved=model_timescale,
            delta_tzr_sec=0.0,
        )

    if tzrmjd_scale_upper == "TDB":
        tzrmjd_tdb = tzrmjd_raw
        if model_timescale == "TCB":
            from jug.utils.timescales import convert_tdb_epoch_to_tempo2_tcb

            tzrmjd_model = convert_tdb_epoch_to_tempo2_tcb(tzrmjd_tdb)
        else:
            tzrmjd_model = tzrmjd_tdb
        if verbose:
            print("   TZRMJD treated as TDB (no conversion)")
        return TzrEpochs(
            tzrmjd_raw=tzrmjd_raw,
            tzrmjd_tdb=tzrmjd_tdb,
            tzrmjd_model=tzrmjd_model,
            tzrmjd_scale_resolved="TDB",
            delta_tzr_sec=0.0,
        )

    if tzrmjd_scale_upper == "UTC":
        if verbose:
            print(
                f"   TZRMJD scale: UTC (explicit override, converting via {tzr_site} clock)"
            )
        if par_timescale == "TDB":
            print(
                "   [!] WARNING: tzrmjd_scale='UTC' contradicts par file UNITS=TDB. "
                "This will apply a ~69 s clock correction to TZRMJD. "
                "Use only for legacy par files with UTC TZRMJD values."
            )
    elif tzrmjd_scale_upper == "AUTO":
        if engine_profile.is_tempo2 and par_timescale == "TDB":
            tzrmjd_tdb = tzrmjd_raw
            tzrmjd_model = tzrmjd_raw
            if verbose:
                print(
                    "   TZRMJD scale: AUTO -> TDB (tempo2 mode, UNITS=TDB, no conversion)"
                )
            return TzrEpochs(
                tzrmjd_raw=tzrmjd_raw,
                tzrmjd_tdb=tzrmjd_tdb,
                tzrmjd_model=tzrmjd_model,
                tzrmjd_scale_resolved="TDB",
                delta_tzr_sec=0.0,
            )
        if verbose:
            print(
                f"   TZRMJD scale: AUTO -> UTC (site arrival, converting via {tzr_site} clock)"
            )
    else:
        raise ValueError(
            f"Invalid tzrmjd_scale '{tzrmjd_scale}'. Must be 'AUTO', 'TDB', or 'UTC'."
        )

    tzrmjd_tdb_ld = compute_tdb_standalone_vectorized(
        [int(tzrmjd_raw)],
        [float(tzrmjd_raw - int(tzrmjd_raw))],
        tzr_clock,
        bipm_clock,
        tzr_location,
    )[0]
    tzrmjd_tdb = np.longdouble(tzrmjd_tdb_ld)
    if model_timescale == "TCB":
        from jug.utils.timescales import convert_tdb_epoch_to_tempo2_tcb

        tzrmjd_model = convert_tdb_epoch_to_tempo2_tcb(tzrmjd_tdb)
    else:
        tzrmjd_model = tzrmjd_tdb
    delta_tzr_sec = float(tzrmjd_tdb - tzrmjd_raw) * SECS_PER_DAY
    if verbose:
        print(f"   TZRMJD converted from UTC to TDB (delta = {delta_tzr_sec:.3f} s)")
    return TzrEpochs(
        tzrmjd_raw=tzrmjd_raw,
        tzrmjd_tdb=tzrmjd_tdb,
        tzrmjd_model=tzrmjd_model,
        tzrmjd_scale_resolved="UTC",
        delta_tzr_sec=delta_tzr_sec,
    )


def _tzr_use_native_ecliptic(
    params: dict[str, Any],
    engine_profile: EngineConventionProfile,
) -> bool:
    """Native ecliptic Roemer at TZR is tempo2-only (Phase C)."""
    return engine_profile.is_tempo2 and bool(params.get("_ecliptic_coords", False))


def _pint_ecliptic_obliquity_rad(params: dict[str, Any]) -> float:
    ecl_frame = str(params.get("_ecliptic_frame", "IERS2010")).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC["IERS2010"])
    return obl_arcsec * np.pi / (180.0 * 3600.0)


def compute_tzr_astrometry_pint(
    *,
    params: dict[str, Any],
    epochs: TzrEpochs,
    tzr_obs_itrf_km: np.ndarray,
    tzr_is_ssb: bool,
    ephem: str,
    ra_rad: float,
    dec_rad: float,
    pmra_rad_day: float,
    pmdec_rad_day: float,
    posepoch: float,
    parallax_mas: float,
    planet_shapiro_enabled: bool,
    model_timescale: str,
    verbose: bool,
) -> TzrAstrometryTerms:
    """PINT-family TZR Roemer/Shapiro (Astropy ephemeris, unchanged Phase B pint path)."""
    tzr_tdb_arr = np.array([float(epochs.tzrmjd_tdb)])
    if tzr_is_ssb:
        tzr_ssb_obs_pos = np.zeros((1, 3))
        tzr_ssb_obs_vel = np.zeros((1, 3))
    else:
        tzr_ssb_obs_pos, tzr_ssb_obs_vel = compute_ssb_obs_pos_vel(
            tzr_tdb_arr, tzr_obs_itrf_km, ephemeris=ephem
        )

    tzr_model_arr = np.array([float(epochs.tzrmjd_model)])
    use_native_ecliptic = bool(params.get("_ecliptic_coords", False))
    obl_rad = _pint_ecliptic_obliquity_rad(params) if use_native_ecliptic else 0.0

    if use_native_ecliptic:
        tzr_L_hat = compute_ecliptic_pulsar_direction(
            float(params["_ecliptic_lon_deg"]),
            float(params["_ecliptic_lat_deg"]),
            float(params.get("_ecliptic_pm_lon", 0.0)),
            float(params.get("_ecliptic_pm_lat", 0.0)),
            posepoch,
            tzr_model_arr,
        )
        tzr_ssb_obs_pos_delay = rotate_equatorial_to_ecliptic(tzr_ssb_obs_pos, obl_rad)
        tzr_ssb_obs_vel_delay = rotate_equatorial_to_ecliptic(tzr_ssb_obs_vel, obl_rad)
    else:
        tzr_L_hat = compute_pulsar_direction(
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tzr_model_arr
        )
        tzr_ssb_obs_pos_delay = tzr_ssb_obs_pos
        tzr_ssb_obs_vel_delay = tzr_ssb_obs_vel

    if model_timescale == "TCB":
        from jug.utils.timescales import IFTE_K

        ifte = float(IFTE_K)
        tzr_ssb_obs_pos_delay = tzr_ssb_obs_pos_delay * ifte
        tzr_ssb_obs_vel_delay = tzr_ssb_obs_vel_delay * ifte

    tzr_roemer = compute_roemer_delay(tzr_ssb_obs_pos_delay, tzr_L_hat, parallax_mas)[0]

    tzr_times = Time(tzr_tdb_arr, format="mjd", scale="tdb")
    with solar_system_ephemeris.set(ephem):
        tzr_sun_pos = get_body_barycentric_posvel("sun", tzr_times)[0].xyz.to(u.km).value.T
    tzr_obs_sun = tzr_sun_pos - tzr_ssb_obs_pos
    tzr_obs_sun_delay = (
        rotate_equatorial_to_ecliptic(tzr_obs_sun, obl_rad)
        if use_native_ecliptic
        else tzr_obs_sun
    )
    if model_timescale == "TCB":
        from jug.utils.timescales import IFTE_K

        tzr_obs_sun_delay = tzr_obs_sun_delay * float(IFTE_K)

    tzr_sun_shapiro = compute_shapiro_delay(tzr_obs_sun_delay, tzr_L_hat, T_SUN_SEC)[0]

    tzr_planet_shapiro = 0.0
    if planet_shapiro_enabled:
        with solar_system_ephemeris.set(ephem):
            for planet in ["jupiter", "saturn", "uranus", "neptune", "venus"]:
                tzr_planet_pos = get_body_barycentric_posvel(planet, tzr_times)[0].xyz.to(u.km).value.T
                tzr_obs_planet = tzr_planet_pos - tzr_ssb_obs_pos
                tzr_obs_planet_delay = (
                    rotate_equatorial_to_ecliptic(tzr_obs_planet, obl_rad)
                    if use_native_ecliptic
                    else tzr_obs_planet
                )
                if model_timescale == "TCB":
                    from jug.utils.timescales import IFTE_K

                    tzr_obs_planet_delay = tzr_obs_planet_delay * float(IFTE_K)
                tzr_planet_shapiro += compute_shapiro_delay(
                    tzr_obs_planet_delay, tzr_L_hat, T_PLANET[planet]
                )[0]

    dilate_freq = False
    if "DILATEFREQ" in params:
        flag = str(params["DILATEFREQ"]).upper().strip()
        dilate_freq = flag in ("Y", "1", "TRUE", "T")
    tzr_einstein_rate = None
    if dilate_freq:
        tzr_einstein_rate = compute_einstein_rate(
            tzr_tdb_arr,
            units=params.get("_timescale_in", params.get("_par_timescale", "TDB")),
        )
    tzr_freq = float(params.get("TZRFRQ", 1400.0))
    if not np.isfinite(tzr_freq):
        tzr_freq = 1e12
    tzr_freq_bary = compute_barycentric_freq(
        np.array([tzr_freq]),
        tzr_ssb_obs_vel_delay,
        tzr_L_hat,
        einstein_rate=tzr_einstein_rate,
    )[0]

    return TzrAstrometryTerms(
        L_hat=tzr_L_hat,
        roemer_shapiro_sec=float(tzr_roemer + tzr_sun_shapiro + tzr_planet_shapiro),
        freq_bary_mhz=float(tzr_freq_bary),
        obs_sun_delay_km=np.asarray(tzr_obs_sun_delay[0], dtype=np.float64),
        ssb_obs_pos_km=np.asarray(tzr_ssb_obs_pos[0], dtype=np.float64),
        use_native_ecliptic=use_native_ecliptic,
        obl_rad=obl_rad,
        geometry_backend="astropy_jpl",
    )


def compute_tzr_astrometry_tempo2(
    *,
    params: dict[str, Any],
    epochs: TzrEpochs,
    tzr_obs_itrf_km: np.ndarray,
    tzr_is_ssb: bool,
    ephem: str,
    parallax_mas: float,
    engine_profile: EngineConventionProfile,
    model_timescale: str,
    verbose: bool,
) -> TzrAstrometryTerms:
    """Tempo2-native TZR Roemer/Shapiro (jplephem on TDB; IFTE Astropy on TCB)."""
    tzr_tdb_arr = np.array([float(epochs.tzrmjd_tdb)])
    if tzr_is_ssb:
        tzr_ssb_obs_pos = np.zeros((1, 3))
        tzr_ssb_obs_vel = np.zeros((1, 3))
    else:
        tzr_ssb_obs_pos, tzr_ssb_obs_vel = compute_ssb_obs_pos_vel(
            tzr_tdb_arr, tzr_obs_itrf_km, ephemeris=ephem
        )

    tzr_model_arr = np.array([float(epochs.tzrmjd_model)], dtype=np.longdouble)
    use_native_ecliptic = _tzr_use_native_ecliptic(params, engine_profile)
    obl_rad = ecliptic_obliquity_rad(params, use_native_ecliptic)

    tzr_ssb_obs_pos_delay = (
        rotate_equatorial_to_ecliptic(tzr_ssb_obs_pos, obl_rad)
        if use_native_ecliptic
        else tzr_ssb_obs_pos
    )
    tzr_ssb_obs_vel_delay = (
        rotate_equatorial_to_ecliptic(tzr_ssb_obs_vel, obl_rad)
        if use_native_ecliptic
        else tzr_ssb_obs_vel
    )

    if model_timescale == "TCB":
        from jug.utils.timescales import IFTE_K

        ifte = float(IFTE_K)
        tzr_ssb_obs_pos_delay = tzr_ssb_obs_pos_delay * ifte
        tzr_ssb_obs_vel_delay = tzr_ssb_obs_vel_delay * ifte

    L_hat, _pos, _vel = build_pulsar_direction(
        params,
        tzr_model_arr,
        use_native_ecliptic=use_native_ecliptic,
    )

    if engine_profile.is_tcb or model_timescale == "TCB":
        from jug.utils.timescales import IFTE_K

        ifte = float(IFTE_K)
        tzr_roemer = compute_roemer_delay(tzr_ssb_obs_pos_delay, L_hat, parallax_mas)[0]
        tzr_times = Time(tzr_tdb_arr, format="mjd", scale="tdb")
        with solar_system_ephemeris.set(ephem):
            tzr_sun_pos = get_body_barycentric_posvel("sun", tzr_times)[0].xyz.to(u.km).value.T
        tzr_obs_sun = tzr_sun_pos - tzr_ssb_obs_pos
        tzr_obs_sun_delay = (
            tempo2_equ2ecl(tzr_obs_sun) if use_native_ecliptic else tzr_obs_sun
        )
        tzr_obs_sun_delay = tzr_obs_sun_delay * ifte
        tzr_sun_shapiro = (
            compute_shapiro_delay(tzr_obs_sun_delay, L_hat, T_SUN_SEC)[0]
            if engine_profile.solar_shapiro
            else 0.0
        )
        tzr_planet_shapiro = 0.0
        if engine_profile.planet_shapiro:
            with solar_system_ephemeris.set(ephem):
                for planet in ["jupiter", "saturn", "uranus", "neptune", "venus"]:
                    tzr_planet_pos = get_body_barycentric_posvel(planet, tzr_times)[
                        0
                    ].xyz.to(u.km).value.T
                    tzr_obs_planet = tzr_planet_pos - tzr_ssb_obs_pos
                    tzr_obs_planet_delay = (
                        tempo2_equ2ecl(tzr_obs_planet)
                        if use_native_ecliptic
                        else tzr_obs_planet
                    )
                    tzr_planet_shapiro += compute_shapiro_delay(
                        tzr_obs_planet_delay * ifte, L_hat, T_PLANET[planet]
                    )[0]
        backend = "tempo2_tcb_native"
    else:
        ephem_path = resolve_tempo2_ephemeris_path(engine_profile.ephem or ephem)
        if verbose:
            print(f"   Tempo2-native TZR ephemeris: {ephem_path}")
        eph_state = compute_tempo2_ephemeris_state(
            tzr_tdb_arr,
            tzr_ssb_obs_pos,
            ephem_path=ephem_path,
        )
        ssb_obs_ls = ssb_obs_light_seconds(tzr_ssb_obs_pos_delay)
        tzr_roemer = float(
            compute_tempo2_roemer_sec(
                ssb_obs_ls,
                L_hat,
                parallax_mas=parallax_mas,
                pmrv_rad_century=0.0,
                vel_pulsar=None,
                delt_centuries=None,
            )[0]
        )
        obs_sun_ls = eph_state.obs_sun_ls
        if use_native_ecliptic:
            obs_sun_ls = tempo2_equ2ecl(obs_sun_ls)
        tzr_obs_sun_delay = obs_sun_ls * C_KM_S
        tzr_sun_shapiro = (
            float(compute_tempo2_shapiro_sec(obs_sun_ls, L_hat, T_SUN_SEC)[0])
            if engine_profile.solar_shapiro
            else 0.0
        )
        planets_obs = dict(eph_state.planets_obs_ls)
        if use_native_ecliptic:
            planets_obs = {k: tempo2_equ2ecl(v) for k, v in planets_obs.items()}
        tzr_planet_shapiro = float(
            planet_shapiro_sec(
                planets_obs,
                L_hat,
                enabled=engine_profile.planet_shapiro,
            )[0]
        )
        backend = "tempo2_tdb_native"

    tzr_einstein_rate = None
    if engine_profile.dilatefreq:
        units = "TCB" if model_timescale == "TCB" else "TDB"
        tzr_einstein_rate = compute_einstein_rate(tzr_tdb_arr, units=units)
    tzr_freq = float(params.get("TZRFRQ", 1400.0))
    if not np.isfinite(tzr_freq):
        tzr_freq = 1e12
    tzr_freq_bary = compute_barycentric_freq(
        np.array([tzr_freq]),
        tzr_ssb_obs_vel_delay,
        L_hat,
        einstein_rate=tzr_einstein_rate,
    )[0]

    return TzrAstrometryTerms(
        L_hat=L_hat,
        roemer_shapiro_sec=float(tzr_roemer + tzr_sun_shapiro + tzr_planet_shapiro),
        freq_bary_mhz=float(tzr_freq_bary),
        obs_sun_delay_km=np.asarray(tzr_obs_sun_delay[0], dtype=np.float64),
        ssb_obs_pos_km=np.asarray(tzr_ssb_obs_pos[0], dtype=np.float64),
        use_native_ecliptic=use_native_ecliptic,
        obl_rad=obl_rad,
        geometry_backend=backend,
    )
