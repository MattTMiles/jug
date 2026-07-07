"""PINT vs JUG comparison on J0613-0200 with NE_SW removed.

The bundled J0613 par file uses UNITS TCB, which PINT does not support, so
this test skips unless a TDB-convertible fixture is available.  It exists to
document that NE_SW handling was ruled out as a PINT/JUG divergence source.
"""

import numpy as np
import pytest

try:
    from tests.test_paths import get_j0613_paths, skip_if_missing
except ImportError:
    from test_paths import get_j0613_paths, skip_if_missing


def test_j0613_no_sw_pint_vs_jug(tmp_path):
    pint = pytest.importorskip("pint")
    import pint.fitter
    import pint.logging
    from pint.models import get_model_and_toas

    from jug.residuals.simple_calculator import compute_residuals_simple

    pint.logging.setup(level="WARNING")

    par_path, tim_path = get_j0613_paths()
    if not skip_if_missing(par_path, tim_path, "j0613_no_sw"):
        pytest.skip("J0613-0200 test data not available")

    no_sw_par = tmp_path / "J0613-0200_no_sw.par"
    with open(par_path) as f:
        lines = f.readlines()
    with open(no_sw_par, "w") as f:
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("NE_SW") and not stripped.startswith(
                "TNsubtractPoly"
            ):
                f.write(line)

    try:
        m, t = get_model_and_toas(str(no_sw_par), str(tim_path))
    except ValueError as exc:
        if "TCB" in str(exc):
            pytest.skip(f"PINT cannot load this par file: {exc}")
        raise
    fitter = pint.fitter.WLSFitter(t, m)
    pint_resids = fitter.resids.time_resids.to("s").value

    jug_res = compute_residuals_simple(str(no_sw_par), str(tim_path), verbose=False)
    jug_resids = jug_res["residuals_us"] * 1e-6

    rms_diff_ns = np.std(pint_resids - jug_resids) * 1e9
    assert rms_diff_ns < 10.0
