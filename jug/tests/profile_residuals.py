#!/usr/bin/env python
"""Profile where time is spent in compute_residuals_simple.

Uses cProfile to get a proper breakdown.

Usage:
    JAX_PLATFORMS=cpu conda run -n discotech python jug/tests/profile_residuals.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import cProfile
import pstats
import io

PAR = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327_tdb.par"
TIM = "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads/J0125-2327.tim"

from jug.residuals.simple_calculator import compute_residuals_simple

# Warmup (first call includes import & JIT overhead)
print("=== Warmup call ===")
_ = compute_residuals_simple(PAR, TIM, verbose=False, subtract_tzr=False)

# Profile second call (steady-state)
print("\n=== Profiled call ===")
pr = cProfile.Profile()
pr.enable()
result = compute_residuals_simple(PAR, TIM, verbose=False, subtract_tzr=False)
pr.disable()

# Print top functions by cumulative time
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

# Also show by internal (tottime)
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(30)
print("\n=== By total internal time ===")
print(s2.getvalue())
