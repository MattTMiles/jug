"""PINT vs JUG residual comparison on J1713+0747 (binary delay patch check).

Environment variables for CI:
    JUG_TEST_J1713_PAR=/path/to/J1713+0747.par
    JUG_TEST_J1713_TIM=/path/to/J1713+0747.tim

The bundled J1713 par file uses UNITS TCB, which PINT does not support, so
this test skips unless a TDB fixture is provided via the env vars above.
"""

import numpy as np
import pytest

try:
    from tests.test_paths import get_j1713_paths, skip_if_missing
except ImportError:
    from test_paths import get_j1713_paths, skip_if_missing


def test_binary_patch_pint_vs_jug():
    pytest.importorskip("pint")
    from pint.models import get_model
    from pint.residuals import Residuals
    from pint.toa import get_TOAs

    from jug.residuals.simple_calculator import compute_residuals_simple

    par_path, tim_path = get_j1713_paths()
    if not skip_if_missing(par_path, tim_path, "binary_patch"):
        pytest.skip("J1713+0747 test data not available")

    try:
        model = get_model(str(par_path))
    except ValueError as exc:
        if "TCB" in str(exc):
            pytest.skip(f"PINT cannot load this par file: {exc}")
        raise
    toas = get_TOAs(str(tim_path), planets=True, ephem="de440")
    resid_us_pint = Residuals(toas, model).time_resids.to_value("us")

    result = compute_residuals_simple(str(par_path), str(tim_path), verbose=False)
    resid_us_jug = result["residuals_us"]

    diff = (resid_us_jug - np.mean(resid_us_jug)) - (
        resid_us_pint - np.mean(resid_us_pint)
    )
    assert np.std(diff) < 1.0, (
        f"JUG vs PINT residual RMS {np.std(diff):.3f} us "
        f"(max |diff| {np.max(np.abs(diff)):.3f} us)"
    )
