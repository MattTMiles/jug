import numpy as np
import pytest

from jug.delays.binary_bt import (
    compute_binary_derivatives_bt,
    compute_bt_binary_delay,
)
from jug.fitting.binary_registry import (
    compute_binary_delay,
    get_binary_delay_func,
)


PARAMS = {
    "BINARY": "BT",
    "PB": 38.50383278851616,
    "A1": 14.74988754352287,
    "ECC": 0.00023823946365430758,
    "T0": 58204.159397509715,
    "OM": -140.92051062087979,
    "PBDOT": 0.0,
    "XDOT": 0.0,
    "EDOT": 0.0,
    "OMDOT": 0.0,
    "GAMMA": 0.0,
}


def _pint_bt_delay(toas):
    pytest.importorskip("pint")
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.BT_model import BTmodel

    model = BTmodel()
    model.update_input(
        barycentric_toa=np.asarray(toas, dtype=np.longdouble) * u.day,
        PB=PARAMS["PB"] * u.day,
        A1=PARAMS["A1"] * u.lightsecond,
        ECC=PARAMS["ECC"] * u.dimensionless_unscaled,
        T0=np.longdouble(str(PARAMS["T0"])) * u.day,
        OM=PARAMS["OM"] * u.deg,
        PBDOT=PARAMS["PBDOT"] * u.day / u.day,
        A1DOT=PARAMS["XDOT"] * u.lightsecond / u.second,
        EDOT=PARAMS["EDOT"] / u.second,
        OMDOT=PARAMS["OMDOT"] * u.deg / u.year,
        GAMMA=PARAMS["GAMMA"] * u.second,
    )
    return model.BTdelay().to_value(u.second)


def test_bt_registry_uses_bt_model():
    assert get_binary_delay_func("BT") is compute_bt_binary_delay


def test_bt_delay_matches_pint():
    toas = np.linspace(54000.0, 60000.0, 257, dtype=np.longdouble)
    jug = np.asarray(compute_binary_delay(toas, PARAMS))
    pint = _pint_bt_delay(toas)
    assert np.max(np.abs(jug - pint)) < 1e-11


@pytest.mark.parametrize(
    ("param", "step"),
    [
        ("PB", 1e-7),
        ("A1", 1e-7),
        ("ECC", 1e-8),
        ("OM", 1e-5),
        ("T0", 1e-7),
        ("PBDOT", 1e-11),
        ("XDOT", 1e-15),
        ("EDOT", 1e-16),
        ("OMDOT", 1e-7),
        ("GAMMA", 1e-7),
    ],
)
def test_bt_derivatives_match_finite_difference(param, step):
    toas = np.linspace(57000.0, 59000.0, 31, dtype=np.longdouble)
    analytic = np.asarray(
        compute_binary_derivatives_bt(PARAMS, toas, [param])[param]
    )
    plus = dict(PARAMS)
    minus = dict(PARAMS)
    plus[param] += step
    minus[param] -= step
    numeric = (
        np.asarray(compute_bt_binary_delay(toas, plus))
        - np.asarray(compute_bt_binary_delay(toas, minus))
    ) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=5e-4, atol=3e-7)
