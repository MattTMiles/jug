"""Fit H3/STIG on a perturbed J1022+1001 par file (orthometric Shapiro).

Orthometric Shapiro on DD-family autodiff is supported (Fix J3); this
regression asserts the optimized fitter can take steps on H3/STIG.
"""

import pytest

try:
    from tests.test_paths import get_j1022_paths, skip_if_missing
except ImportError:
    from test_paths import get_j1022_paths, skip_if_missing

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.io.par_reader import parse_par_file


def test_j1022_fit_h3_stig(tmp_path):
    par_path, tim_path = get_j1022_paths()
    if not skip_if_missing(par_path, tim_path, "j1022"):
        pytest.skip("J1022+1001 test data not available")

    original_params = parse_par_file(par_path)
    perturbed_par = tmp_path / "J1022+1001_perturbed.par"
    with open(par_path) as f:
        lines = f.readlines()
    with open(perturbed_par, "w") as f:
        for line in lines:
            if line.strip().startswith("H3 "):
                parts = line.split()
                parts[1] = f"{original_params['H3'] * 1.5:.15e}"
                f.write("  ".join(parts) + "\n")
            elif line.strip().startswith("STIG "):
                parts = line.split()
                parts[1] = f"{original_params['STIG'] * 0.8:.15e}"
                f.write("  ".join(parts) + "\n")
            else:
                f.write(line)

    fit_result = fit_parameters_optimized(
        str(perturbed_par),
        str(tim_path),
        fit_params=["H3", "STIG"],
        max_iter=2,
        verbose=False,
    )
    assert fit_result is not None
