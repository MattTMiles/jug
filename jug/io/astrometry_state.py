"""Central astrometry parameter synchronization helpers.

JUG keeps par-file native coordinates (for round-trip I/O) and canonical
equatorial working coordinates (for delay evaluation).  These helpers are the
single Python-side place that keeps those mirrors synchronized.  They are not
part of the JAX timing kernels; they only update small parameter dictionaries
when parsing, fitting, or writing temporary par files.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set

import numpy as np


ECLIPTIC_PUBLIC_TO_INTERNAL = {
    'ELONG': '_ecliptic_lon_deg',
    'ELAT': '_ecliptic_lat_deg',
    'PMELONG': '_ecliptic_pm_lon',
    'PMELAT': '_ecliptic_pm_lat',
}

EQUATORIAL_DERIVED_KEYS = {'RAJ', 'DECJ', 'PMRA', 'PMDEC'}
LAMBDA_DERIVED_KEYS = {'LAMBDA', 'BETA', 'PMLAMBDA', 'PMBETA'}
ELONG_DERIVED_KEYS = {'ELONG', 'ELAT', 'PMELONG', 'PMELAT'}


def native_ecliptic_family(initial_params: Mapping[str, Any]) -> Optional[str]:
    """Return the native ecliptic family used by the original par data.

    ``ELONG/ELAT`` and ``LAMBDA/BETA`` are equivalent physical coordinates, but
    writing both to a temp par file is dangerous because the parser must choose
    one.  Preserve the original family whenever possible.
    """
    if 'LAMBDA' in initial_params and 'ELONG' not in initial_params:
        return 'lambda'
    if 'ELONG' in initial_params:
        return 'elong'
    return None


def temp_par_skip_keys(params: Mapping[str, Any], initial_params: Mapping[str, Any]) -> Set[str]:
    """Coordinate keys that should not be added to a temporary par file."""
    if not params.get('_ecliptic_coords'):
        return set()

    family = native_ecliptic_family(initial_params)
    skip = set(EQUATORIAL_DERIVED_KEYS)
    if family == 'lambda':
        skip |= ELONG_DERIVED_KEYS
    else:
        skip |= LAMBDA_DERIVED_KEYS
    return skip


def sync_ecliptic_public_to_internal(
    params: Dict[str, Any],
    updates: Optional[Mapping[str, Any]] = None,
) -> None:
    """Apply fitted ecliptic values and refresh equatorial working keys.

    Updates are in par-file units: degrees for ELONG/ELAT and mas/yr for
    PMELONG/PMELAT.  The function mutates ``params`` in place.
    """
    if updates:
        for public_key, internal_key in ECLIPTIC_PUBLIC_TO_INTERNAL.items():
            if public_key in updates:
                value = float(updates[public_key])
                params[public_key] = value
                params[internal_key] = value

    if any(k in params for k in ECLIPTIC_PUBLIC_TO_INTERNAL.values()):
        reconvert_ecliptic_to_equatorial(params)


def reconvert_ecliptic_to_equatorial(params: Dict[str, Any]) -> None:
    """Refresh RAJ/DECJ/PMRA/PMDEC from internal ecliptic coordinates."""
    from jug.io.par_reader import OBLIQUITY_ARCSEC, format_dec, format_ra

    ecl_lon_deg = float(params.get('_ecliptic_lon_deg', params.get('ELONG', 0.0)))
    ecl_lat_deg = float(params.get('_ecliptic_lat_deg', params.get('ELAT', 0.0)))
    ecl_frame = str(params.get('_ecliptic_frame', params.get('ECL', 'IERS2010'))).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010'])
    obl_rad = obl_arcsec * np.pi / (180.0 * 3600.0)

    lon_rad = np.radians(ecl_lon_deg)
    lat_rad = np.radians(ecl_lat_deg)
    cos_lon, sin_lon = np.cos(lon_rad), np.sin(lon_rad)
    cos_lat, sin_lat = np.cos(lat_rad), np.sin(lat_rad)
    cos_obl, sin_obl = np.cos(obl_rad), np.sin(obl_rad)

    x = cos_lon * cos_lat
    y = sin_lon * cos_lat * cos_obl - sin_lat * sin_obl
    z = sin_lon * cos_lat * sin_obl + sin_lat * cos_obl

    ra_rad = np.arctan2(y, x) % (2 * np.pi)
    dec_rad = np.arctan2(z, np.sqrt(x**2 + y**2))

    params['RAJ'] = format_ra(ra_rad)
    params['DECJ'] = format_dec(dec_rad)
    params['_raj_rad'] = float(ra_rad)
    params['_decj_rad'] = float(dec_rad)

    pm_lon = float(params.get('_ecliptic_pm_lon', params.get('PMELONG', 0.0)))
    pm_lat = float(params.get('_ecliptic_pm_lat', params.get('PMELAT', 0.0)))
    if pm_lon != 0.0 or pm_lat != 0.0:
        dx = -sin_lon * pm_lon - cos_lon * sin_lat * pm_lat
        dy = cos_lon * pm_lon - sin_lon * sin_lat * pm_lat
        dz = cos_lat * pm_lat

        dx_eq = dx
        dy_eq = dy * cos_obl - dz * sin_obl
        dz_eq = dy * sin_obl + dz * cos_obl

        cos_ra, sin_ra = np.cos(ra_rad), np.sin(ra_rad)
        cos_dec, sin_dec = np.cos(dec_rad), np.sin(dec_rad)
        params['PMRA'] = -sin_ra * dx_eq + cos_ra * dy_eq
        params['PMDEC'] = (
            -cos_ra * sin_dec * dx_eq
            - sin_ra * sin_dec * dy_eq
            + cos_dec * dz_eq
        )


def ecliptic_aliases_from_equatorial(
    params: Mapping[str, Any],
    updated_params: Mapping[str, Any],
) -> Dict[str, float]:
    """Return LAMBDA/BETA/PMLAMBDA/PMBETA aliases from current RAJ/DECJ."""
    from jug.io.par_reader import convert_equatorial_to_ecliptic, parse_dec, parse_ra

    ra_val = updated_params.get('RAJ', params.get('RAJ'))
    dec_val = updated_params.get('DECJ', params.get('DECJ'))
    if ra_val is None or dec_val is None:
        return {}

    ra_rad = parse_ra(ra_val) if isinstance(ra_val, str) else float(ra_val)
    dec_rad = parse_dec(dec_val) if isinstance(dec_val, str) else float(dec_val)
    ecl_frame = params.get('_ecliptic_frame', params.get('ECL', 'IERS2010'))
    ecl = convert_equatorial_to_ecliptic(
        ra_rad,
        dec_rad,
        pmra=updated_params.get('PMRA', params.get('PMRA', 0.0)),
        pmdec=updated_params.get('PMDEC', params.get('PMDEC', 0.0)),
        ecl_frame=ecl_frame,
    )
    out = {'LAMBDA': ecl['LAMBDA'], 'BETA': ecl['BETA']}
    if 'PMLAMBDA' in params:
        out['PMLAMBDA'] = ecl['PMLAMBDA']
    if 'PMBETA' in params:
        out['PMBETA'] = ecl['PMBETA']
    return out
