"""Regression tests for centralized astrometry parameter state."""

from pathlib import Path
import re

from jug.io.astrometry_state import (
    sync_ecliptic_public_to_internal,
    temp_par_skip_keys,
)
from jug.io.par_writer import write_par_file


def _ecliptic_params():
    return {
        'PSR': 'TEST',
        'ECL': 'IERS2010',
        'ELONG': 286.0,
        'ELAT': 32.0,
        'PMELONG': -3.0,
        'PMELAT': -5.0,
        '_ecliptic_coords': True,
        '_ecliptic_frame': 'IERS2010',
        '_ecliptic_lon_deg': 286.0,
        '_ecliptic_lat_deg': 32.0,
        '_ecliptic_pm_lon': -3.0,
        '_ecliptic_pm_lat': -5.0,
    }


def test_ecliptic_sync_updates_public_internal_and_equatorial():
    params = _ecliptic_params()

    sync_ecliptic_public_to_internal(
        params,
        {'ELONG': 286.1, 'ELAT': 32.2, 'PMELONG': -3.3, 'PMELAT': -5.4},
    )

    assert params['ELONG'] == 286.1
    assert params['_ecliptic_lon_deg'] == 286.1
    assert params['ELAT'] == 32.2
    assert params['_ecliptic_lat_deg'] == 32.2
    assert params['PMELONG'] == -3.3
    assert params['_ecliptic_pm_lon'] == -3.3
    assert params['PMELAT'] == -5.4
    assert params['_ecliptic_pm_lat'] == -5.4
    assert 'RAJ' in params and 'DECJ' in params
    assert '_raj_rad' in params and '_decj_rad' in params
    assert 'PMRA' in params and 'PMDEC' in params


def test_temp_par_preserves_native_elong_family():
    params = _ecliptic_params()
    initial = {'ELONG': params['ELONG'], 'ELAT': params['ELAT']}

    skip = temp_par_skip_keys(params, initial)

    assert {'RAJ', 'DECJ', 'PMRA', 'PMDEC'} <= skip
    assert {'LAMBDA', 'BETA', 'PMLAMBDA', 'PMBETA'} <= skip
    assert 'ELONG' not in skip
    assert 'PMELONG' not in skip


def test_par_writer_uses_central_ecliptic_pm_keys(tmp_path: Path):
    params = _ecliptic_params()
    params['_ecliptic_pm_lon'] = -3.123
    params['_ecliptic_pm_lat'] = -5.456

    out = tmp_path / 'test.par'
    write_par_file(params, out, fit_params={'PMELONG', 'PMELAT'})
    text = out.read_text()

    assert re.search(r'^PMELONG\s+-3\.123\s+1$', text, re.MULTILINE)
    assert re.search(r'^PMELAT\s+-5\.456\s+1$', text, re.MULTILINE)
