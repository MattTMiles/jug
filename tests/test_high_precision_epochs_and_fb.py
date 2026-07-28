"""Regression coverage for epoch precision and arbitrary FB indices."""

from pathlib import Path

import numpy as np
import pytest

from jug.io.par_reader import get_longdouble, parse_par_file
from jug.io.par_writer import write_par_file
from jug.model.parameter_spec import get_binary_params_from_list, get_spec, validate_fit_param
from jug.utils.timescales import convert_par_params_to_tdb, scale_parameter_tcb_to_tdb

pytestmark = pytest.mark.smoke


def test_arbitrary_fb_is_binary_fittable_and_writable(tmp_path):
    assert validate_fit_param('FB40')
    assert get_spec('FB40').tcb_scaling_dim == -41
    assert get_binary_params_from_list(['F0', 'FB40']) == ['FB40']

    output = tmp_path / 'fb40.par'
    write_par_file(
        {'PSR': 'JTEST', 'BINARY': 'ELL1', 'FB0': 1e-4, 'FB40': 2e-80},
        output,
        fit_params={'FB40'},
    )
    text = output.read_text()
    assert 'FB40' in text


def test_production_kernel_collects_all_contiguous_fb_terms():
    from jug.residuals.simple_calculator import _extract_binary_params

    params = {
        'BINARY': 'ELL1', 'PEPOCH': 58000.0, 'TASC': 58000.0,
        'A1': 1.0, 'EPS1': 0.0, 'EPS2': 0.0,
        **{f'FB{i}': (1e-4 if i == 0 else 0.0) for i in range(41)},
    }
    binary = _extract_binary_params(params, verbose=False)
    assert binary['fb_coeffs_jax'].shape == (41,)
    assert binary['fb_factorials_jax'].shape == (41,)


def test_arbitrary_fb_tcb_scaling():
    params = {
        'UNITS': 'TCB', '_par_timescale': 'TCB',
        'FB40': 2e-80, '_high_precision': {},
    }
    original = params['FB40']
    convert_par_params_to_tdb(params, verbose=False)
    expected = scale_parameter_tcb_to_tdb(original, -41)
    assert params['FB40'] == expected


def test_indexed_epochs_are_cached_at_source_precision(tmp_path):
    par = tmp_path / 'epochs.par'
    par.write_text(
        'PSR JTEST\n'
        'UNITS TDB\n'
        'F0 100\n'
        'PEPOCH 58000\n'
        'GLEP_12 58258.9529477631550272\n'
        'EXPEP_23 58259.1234567890123456\n'
        'DMXR1_0042 57000.0000000000123456\n'
    )
    params = parse_par_file(par)
    assert get_longdouble(params, 'GLEP_12') == np.longdouble('58258.9529477631550272')
    assert get_longdouble(params, 'EXPEP_23') == np.longdouble('58259.1234567890123456')
    assert get_longdouble(params, 'DMXR1_0042') == np.longdouble('57000.0000000000123456')


def test_dd_and_bt_use_cached_t0_precision():
    from jug.delays.binary_bt import compute_bt_binary_delay
    from jug.fitting.derivatives_dd import compute_dd_binary_delay

    t0_text = '58258.9529477631550272'
    toas = np.array([58258.8, 58259.1], dtype=np.longdouble)
    base = {
        'PB': 0.5, 'A1': 1.2, 'T0': float(t0_text), 'ECC': 0.1,
        'OM': 40.0, 'PBDOT': 0.0, 'GAMMA': 0.0,
    }
    precise = dict(base, _high_precision={'T0': t0_text})
    explicit = dict(base, T0=np.longdouble(t0_text))

    for compute in (compute_dd_binary_delay, compute_bt_binary_delay):
        cached = np.asarray(compute(toas, precise))
        direct = np.asarray(compute(toas, explicit))
        truncated = np.asarray(compute(toas, base))
        np.testing.assert_array_equal(cached, direct)
        assert np.max(np.abs(cached - truncated)) > 1e-12


def test_spin_and_dm_derivatives_use_cached_epochs():
    from jug.fitting.derivatives_dm import compute_dm_derivatives
    from jug.fitting.derivatives_spin import compute_spin_derivatives

    epoch_text = '58258.9529477631550272'
    toas = np.array([58258.8, 58259.1], dtype=np.longdouble)

    spin_base = {'F0': 100.0, 'PEPOCH': float(epoch_text)}
    spin_cached = dict(spin_base, _high_precision={'PEPOCH': epoch_text})
    spin_direct = dict(spin_base, PEPOCH=np.longdouble(epoch_text))
    np.testing.assert_array_equal(
        np.asarray(compute_spin_derivatives(spin_cached, toas, ['F0'])['F0']),
        np.asarray(compute_spin_derivatives(spin_direct, toas, ['F0'])['F0']),
    )

    dm_base = {'DM': 10.0, 'DM1': 1.0, 'DMEPOCH': float(epoch_text)}
    dm_cached = dict(dm_base, _high_precision={'DMEPOCH': epoch_text})
    dm_direct = dict(dm_base, DMEPOCH=np.longdouble(epoch_text))
    freq = np.array([800.0, 1400.0])
    np.testing.assert_array_equal(
        np.asarray(compute_dm_derivatives(dm_cached, toas, freq, ['DM1'])['DM1']),
        np.asarray(compute_dm_derivatives(dm_direct, toas, freq, ['DM1'])['DM1']),
    )
