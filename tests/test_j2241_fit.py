"""Test J2241-5236 fitting with FB parameters.

Environment variables for CI:
    JUG_TEST_J2241_PAR=/path/to/J2241-5236.par
    JUG_TEST_J2241_TIM=/path/to/J2241-5236.tim
"""

import numpy as np
import sys
import jax
import time

import pytest

# Import test path utilities
try:
    from tests.test_paths import get_j2241_paths, skip_if_missing
except ImportError:
    from test_paths import get_j2241_paths, skip_if_missing

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized

# Enable x64 precision
jax.config.update("jax_enable_x64", True)

# Get paths from environment or defaults
par_path, tim_path = get_j2241_paths()
if not skip_if_missing(par_path, tim_path, "j2241_fit"):
    pytest.skip("J2241 test data not available", allow_module_level=True)

par_file = str(par_path)
tim_file = str(tim_path)

print("1. Running Pre-Fit JUG...")

# Compute pre-fit residuals
res_before = compute_residuals_simple(par_file, tim_file)
print(f"   Pre-Fit RMS: {res_before['weighted_rms_us']:.3f} us")

# 2. Setup fitting for FB parameters
fit_params = [f'FB{i}' for i in range(18)]
print(f"\n2. Fitting 18 parameters: {fit_params}")

# Run fit
t0 = time.time()
fit_results = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=fit_params,
    max_iter=10,
    verbose=True
)

print(f"\n3. Post-Fit Results:")
print(f"   Post-Fit RMS: {fit_results['final_rms']:.3f} us")
print(f"   Tempo2 Benchmark: 0.189 us")
print(f"   Difference: {abs(fit_results['final_rms'] - 0.189):.3f} us")

if abs(fit_results['final_rms'] - 0.189) < 0.005:
    print("✅ MATCH: Solutions agree!")
else:
    print("❌ MISMATCH: Solutions differ")
