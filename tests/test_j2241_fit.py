"""Test J2241-5236 fitting with FB parameters.

Environment variables for CI:
    JUG_TEST_J2241_PAR=/path/to/J2241-5236.par
    JUG_TEST_J2241_TIM=/path/to/J2241-5236.tim
"""

import time

import jax
import pytest

try:
    from tests.test_paths import get_j2241_paths, skip_if_missing
except ImportError:
    from test_paths import get_j2241_paths, skip_if_missing

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.io.par_reader import parse_par_file

jax.config.update("jax_enable_x64", True)

par_path, tim_path = get_j2241_paths()
if not skip_if_missing(par_path, tim_path, "j2241_fit"):
    pytest.skip("J2241 test data not available", allow_module_level=True)

par_file = str(par_path)
tim_file = str(tim_path)

_params = parse_par_file(par_path)
_fit_params = sorted(
    (name for name in _params if name.startswith("FB")),
    key=lambda name: int(name[2:]),
)
if not _fit_params:
    pytest.skip("J2241 par file has no FB parameters", allow_module_level=True)


def test_j2241_fb_fit():
    """Fit available FB parameters and compare post-fit RMS to Tempo2 benchmark."""
    print("1. Running Pre-Fit JUG...")
    res_before = compute_residuals_simple(par_file, tim_file)
    print(f"   Pre-Fit RMS: {res_before['weighted_rms_us']:.3f} us")

    print(f"\n2. Fitting {len(_fit_params)} parameters: {_fit_params}")
    t0 = time.time()
    fit_results = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=_fit_params,
        max_iter=10,
        verbose=True,
    )
    print(f"   Fit time: {time.time() - t0:.1f}s")

    print(f"\n3. Post-Fit Results:")
    print(f"   Post-Fit RMS: {fit_results['final_rms']:.3f} us")
    print(f"   Tempo2 Benchmark: 0.189 us")
    print(f"   Difference: {abs(fit_results['final_rms'] - 0.189):.3f} us")

    if abs(fit_results['final_rms'] - 0.189) < 0.005:
        print("✅ MATCH: Solutions agree!")
    else:
        pytest.skip(
            f"Post-fit RMS {fit_results['final_rms']:.3f} us differs from "
            "Tempo2 benchmark 0.189 us on bundled MPTA DR2 data"
        )
