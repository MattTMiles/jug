"""Delay providers for PINT-family compatibility mode.

Runtime physics conventions come from ``EngineConventionProfile``; comparison-only
knobs remain in ``DiagnosticConventions``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time

from jug.delays.barycentric import (
    compute_barycentric_freq,
    compute_einstein_rate,
    compute_roemer_delay,
    compute_shapiro_delay,
    compute_ssb_obs_pos_vel,
    rotate_equatorial_to_ecliptic,
    ecliptic_obliquity_rad,
)
from jug.io.par_reader import parse_dec, parse_ra
from jug.residuals.diagnostic_conventions import DiagnosticConventions, TermDiagnosticMetadata
from jug.residuals.engine_conventions import (
    EngineConventionProfile,
    default_engine_profile,
    normalize_compatibility_mode,
    validate_engine_profile_matches_compatibility,
)
from jug.utils.constants import OBSERVATORIES, T_PLANET, T_SUN_SEC


@dataclass
class GeometryTerms:
    """Per-TOA astrometric delay terms returned by compatibility providers."""

    model_mjd: np.ndarray
    model_timescale: str
    L_hat: np.ndarray
    ssb_obs_pos_km: np.ndarray
    ssb_obs_vel_km_s: np.ndarray
    ssb_obs_pos_delay_km: np.ndarray
    ssb_obs_vel_delay_km_s: np.ndarray
    roemer_sec: np.ndarray
    sun_shapiro_sec: np.ndarray
    planet_shapiro_sec: np.ndarray
    roemer_shapiro_sec: np.ndarray
    obs_sun_pos_km: np.ndarray
    obs_sun_pos_delay_km: np.ndarray
    obs_planet_pos_ls_cached: dict[str, np.ndarray] | None
    freq_bary_mhz: np.ndarray
    use_native_ecliptic: bool
    obl_rad: float
    metadata: TermDiagnosticMetadata


class DelayProvider(ABC):
    """Backend-specific astrometry and propagation delay provider."""

    profile: EngineConventionProfile
    diagnostics: DiagnosticConventions

    def __init__(
        self,
        compatibility: str,
        profile: EngineConventionProfile | None = None,
        diagnostics: DiagnosticConventions | None = None,
    ):
        mode = normalize_compatibility_mode(compatibility)
        self.profile = profile or default_engine_profile(mode)
        self.diagnostics = diagnostics or DiagnosticConventions()
        validate_engine_profile_matches_compatibility(compatibility, self.profile)

    @abstractmethod
    def compute_geometry_terms(
        self,
        *,
        params: dict[str, Any],
        tdb_mjd: np.ndarray,
        toas: list[Any],
        obs_itrf_km: np.ndarray,
        all_obs_codes: list[str],
        ephem: str,
        geometry_cache: dict | None,
        geo_hit: bool,
        verbose: bool,
    ) -> GeometryTerms:
        """Compute Roemer/Shapiro geometry and barycentric frequency."""

    @property
    def compatibility(self) -> str:
        return self.profile.compatibility

    @property
    def phase_mean_mode(self) -> str:
        if self.diagnostics.phase_mean_mode is not None:
            return self.diagnostics.phase_mean_mode
        return self.profile.phase_mean_mode

    @property
    def provider_name(self) -> str:
        return f"{self.compatibility}_delay_provider"


class PintDelayProvider(DelayProvider):
    """PINT-family geometry: Astropy ephemeris, equatorial or PINT ecliptic path."""

    def compute_geometry_terms(
        self,
        *,
        params: dict[str, Any],
        tdb_mjd: np.ndarray,
        toas: list[Any],
        obs_itrf_km: np.ndarray,
        all_obs_codes: list[str],
        ephem: str,
        geometry_cache: dict | None,
        geo_hit: bool,
        verbose: bool,
    ) -> GeometryTerms:
        return _compute_pint_geometry_terms(
            provider=self,
            params=params,
            tdb_mjd=tdb_mjd,
            toas=toas,
            obs_itrf_km=obs_itrf_km,
            all_obs_codes=all_obs_codes,
            ephem=ephem,
            geometry_cache=geometry_cache,
            geo_hit=geo_hit,
            verbose=verbose,
        )


def get_delay_provider(
    compatibility: str,
    profile: EngineConventionProfile | None = None,
    diagnostics: DiagnosticConventions | None = None,
    *,
    conventions: DiagnosticConventions | None = None,
) -> DelayProvider:
    """Factory for the PINT-family delay provider."""
    diag = diagnostics or conventions or DiagnosticConventions()
    normalize_compatibility_mode(compatibility)
    return PintDelayProvider(compatibility, profile, diag)


def _compute_ssb_obs_for_toas(
    tdb_mjd: np.ndarray,
    toas: list[Any],
    obs_itrf_km: np.ndarray,
    all_obs_codes: list[str],
    ephem: str,
    geometry_cache: dict | None,
    geo_hit: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    if geo_hit and geometry_cache is not None:
        return (
            geometry_cache["ssb_obs_pos_km"],
            geometry_cache["ssb_obs_vel_km_s"],
            geometry_cache["obs_sun_pos_km"],
            geometry_cache.get("obs_planet_pos_ls"),
        )

    is_multi_obs = len(all_obs_codes) > 1
    if not is_multi_obs:
        ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(
            tdb_mjd, obs_itrf_km, ephemeris=ephem
        )
    else:
        ssb_obs_pos_km = np.zeros((len(toas), 3))
        ssb_obs_vel_km_s = np.zeros((len(toas), 3))
        for obs_code in all_obs_codes:
            idxs = [i for i, toa in enumerate(toas) if toa.observatory.lower() == obs_code]
            obs_loc_km = OBSERVATORIES.get(obs_code, obs_itrf_km)
            pos, vel = compute_ssb_obs_pos_vel(tdb_mjd[idxs], obs_loc_km, ephemeris=ephem)
            ssb_obs_pos_km[idxs] = pos
            ssb_obs_vel_km_s[idxs] = vel

    times = Time(tdb_mjd, format="mjd", scale="tdb")
    with solar_system_ephemeris.set(ephem):
        sun_pos = get_body_barycentric_posvel("sun", times)[0].xyz.to(u.km).value.T
    obs_sun_pos_km = sun_pos - ssb_obs_pos_km
    return ssb_obs_pos_km, ssb_obs_vel_km_s, obs_sun_pos_km, None


def _compute_pint_geometry_terms(
    *,
    provider: DelayProvider,
    params: dict[str, Any],
    tdb_mjd: np.ndarray,
    toas: list[Any],
    obs_itrf_km: np.ndarray,
    all_obs_codes: list[str],
    ephem: str,
    geometry_cache: dict | None,
    geo_hit: bool,
    verbose: bool,
) -> GeometryTerms:
    """PINT-family Astropy geometry kernel."""
    from jug.delays.barycentric import compute_ecliptic_pulsar_direction, compute_pulsar_direction

    model_timescale = str(
        params.get("_timescale_in", params.get("_par_timescale", "TDB"))
    ).upper()
    model_mjd = np.array(tdb_mjd, dtype=np.longdouble)
    use_native_ecliptic = False
    obl_rad = 0.0

    ssb_obs_pos_km, ssb_obs_vel_km_s, obs_sun_pos_km, obs_planet_pos_ls_cached = (
        _compute_ssb_obs_for_toas(
            tdb_mjd, toas, obs_itrf_km, all_obs_codes, ephem, geometry_cache, geo_hit
        )
    )

    ra_rad = float(params.get("_raj_rad", parse_ra(params["RAJ"])))
    dec_rad = float(params.get("_decj_rad", parse_dec(params["DECJ"])))
    pmra_rad_day = params.get("PMRA", 0.0) * (np.pi / 180 / 3600000) / 365.25
    pmdec_rad_day = params.get("PMDEC", 0.0) * (np.pi / 180 / 3600000) / 365.25
    posepoch = params.get("POSEPOCH", params["PEPOCH"])
    parallax_mas = params.get("PX", 0.0)

    if params.get("_ecliptic_coords", False) and provider.profile.native_ecliptic:
        use_native_ecliptic = True
        obl_rad = ecliptic_obliquity_rad(params, True)
        L_hat = compute_ecliptic_pulsar_direction(
            float(params["_ecliptic_lon_deg"]),
            float(params["_ecliptic_lat_deg"]),
            float(params.get("_ecliptic_pm_lon", 0.0)),
            float(params.get("_ecliptic_pm_lat", 0.0)),
            posepoch,
            model_mjd,
        )
        ssb_obs_pos_delay_km = rotate_equatorial_to_ecliptic(ssb_obs_pos_km, obl_rad)
        ssb_obs_vel_delay_km_s = rotate_equatorial_to_ecliptic(ssb_obs_vel_km_s, obl_rad)
    else:
        L_hat = compute_pulsar_direction(
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, model_mjd
        )
        ssb_obs_pos_delay_km = ssb_obs_pos_km
        ssb_obs_vel_delay_km_s = ssb_obs_vel_km_s

    roemer_sec = compute_roemer_delay(ssb_obs_pos_delay_km, L_hat, parallax_mas)
    obs_sun_pos_delay_km = (
        rotate_equatorial_to_ecliptic(obs_sun_pos_km, obl_rad) if use_native_ecliptic else obs_sun_pos_km
    )
    sun_shapiro_sec = (
        compute_shapiro_delay(obs_sun_pos_delay_km, L_hat, T_SUN_SEC)
        if provider.profile.solar_shapiro
        else np.zeros(len(tdb_mjd), dtype=np.float64)
    )

    planet_shapiro_enabled = provider.profile.planet_shapiro
    planet_shapiro_sec = np.zeros(len(tdb_mjd), dtype=np.float64)
    if planet_shapiro_enabled:
        need_planet_compute = not geo_hit or obs_planet_pos_ls_cached is None
        if need_planet_compute:
            if verbose:
                print("   Computing planetary Shapiro delays (pint path)...")
            times = Time(tdb_mjd, format="mjd", scale="tdb")
            obs_planet_pos_ls_cached = {}
            with solar_system_ephemeris.set(ephem):
                for planet in ["jupiter", "saturn", "uranus", "neptune", "venus"]:
                    planet_pos = get_body_barycentric_posvel(planet, times)[0].xyz.to(u.km).value.T
                    obs_planet_pos_ls_cached[planet] = planet_pos - ssb_obs_pos_km
        if obs_planet_pos_ls_cached:
            for planet, obs_planet_km in obs_planet_pos_ls_cached.items():
                obs_planet_delay_km = (
                    rotate_equatorial_to_ecliptic(obs_planet_km, obl_rad)
                    if use_native_ecliptic
                    else obs_planet_km
                )
                planet_shapiro_sec += compute_shapiro_delay(
                    obs_planet_delay_km, L_hat, T_PLANET[planet]
                )

    roemer_shapiro_sec = roemer_sec + sun_shapiro_sec + planet_shapiro_sec

    if geometry_cache is not None and not geo_hit:
        geometry_cache["tdb_mjd"] = tdb_mjd
        geometry_cache["ssb_obs_pos_km"] = ssb_obs_pos_km
        geometry_cache["ssb_obs_vel_km_s"] = ssb_obs_vel_km_s
        geometry_cache["obs_sun_pos_km"] = obs_sun_pos_km
        geometry_cache["obs_planet_pos_ls"] = (
            obs_planet_pos_ls_cached if planet_shapiro_enabled else None
        )

    einstein_rate = None
    if provider.profile.dilatefreq:
        units = params.get("_timescale_in", params.get("_par_timescale", "TDB"))
        einstein_rate = compute_einstein_rate(tdb_mjd, units=units)
    freq_mhz = np.array([toa.freq_mhz for toa in toas])
    freq_bary_mhz = compute_barycentric_freq(
        freq_mhz,
        ssb_obs_vel_delay_km_s,
        L_hat,
        einstein_rate=einstein_rate,
    )

    metadata = TermDiagnosticMetadata(
        compatibility=provider.compatibility,
        provider=provider.provider_name,
        geometry_backend="astropy_jpl",
        term_sources={
            "roemer_sec": "astropy_jpl",
            "sun_shapiro_sec": "astropy_jpl",
            "planet_shapiro_sec": "astropy_jpl",
            "freq_bary_mhz": "astropy_jpl",
        },
    )

    return GeometryTerms(
        model_mjd=np.array(model_mjd, dtype=np.longdouble),
        model_timescale=model_timescale,
        L_hat=L_hat,
        ssb_obs_pos_km=ssb_obs_pos_km,
        ssb_obs_vel_km_s=ssb_obs_vel_km_s,
        ssb_obs_pos_delay_km=ssb_obs_pos_delay_km,
        ssb_obs_vel_delay_km_s=ssb_obs_vel_delay_km_s,
        roemer_sec=np.asarray(roemer_sec, dtype=np.float64),
        sun_shapiro_sec=np.asarray(sun_shapiro_sec, dtype=np.float64),
        planet_shapiro_sec=np.asarray(planet_shapiro_sec, dtype=np.float64),
        roemer_shapiro_sec=np.asarray(roemer_shapiro_sec, dtype=np.float64),
        obs_sun_pos_km=obs_sun_pos_km,
        obs_sun_pos_delay_km=obs_sun_pos_delay_km,
        obs_planet_pos_ls_cached=obs_planet_pos_ls_cached,
        freq_bary_mhz=np.asarray(freq_bary_mhz, dtype=np.float64),
        use_native_ecliptic=use_native_ecliptic,
        obl_rad=obl_rad,
        metadata=metadata,
    )