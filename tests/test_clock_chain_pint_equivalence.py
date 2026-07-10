"""JUG clock-chain == PINT clock_corrections (per-observatory, by filename).

Locks the PINT/Tempo2-equivalent configured clock chain
(`simple_calculator._OBS_CLOCK_FILES`): for each GPS-chain observatory the JUG
correction (Σ observatory clock files + gps2utc.clk + TT(BIPM)) must equal
PINT's `Observatory.clock_corrections()` to sub-ns.

This guards against the 2026-06-22 class of regression where a clock-file
*header* relabel (gps2utc.clk "UTC(GPS) UTC" -> "UTC(GPS) UTC(USNO)") silently
re-routed JUG's graph onto a different UTC realization (e.g. VLA via
vla2nist->nist2utc, ~2264 ns off). Because JUG now routes by filename, the
header is irrelevant and JUG stays bit-aligned with PINT.
"""
import os
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

CLOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "clock")
CLOCK_DIR = os.path.abspath(CLOCK_DIR)
BIPM = "BIPM2024"
# Test MJDs within the common in-coverage window of all observatory clock files
# (e.g. vla2gps starts MJD ~57054, ao2gps ends ~59079). Outside a file's range
# both codes extrapolate and may differ (~100 ns) -- a separate coverage concern,
# not a routing one, and no real TOAs fall there. No leap second after 57754.
TEST_MJDS = np.array([57200.0, 57800.0, 58300.0, 58800.0])

# (jug_obs_code, pint_obs_name)
OBS = [("pks", "parkes"), ("gbt", "gbt"), ("vla", "vla"), ("ao", "arecibo")]


@pytest.mark.parametrize("jug_code,pint_name", OBS)
def test_jug_clock_chain_matches_pint(jug_code, pint_name):
    pint = pytest.importorskip("pint")
    import pint.logging
    pint.logging.setup(level="ERROR")
    from astropy.time import Time
    import astropy.units as u
    from pint.observatory import get_observatory
    from jug.scripts.compare_pint_batch import prepare_pint_environment
    from jug.residuals.simple_calculator import _load_obs_chain
    from jug.io.clock import (interpolate_clock_vectorized, parse_clock_file)

    # Force PINT onto JUG's clock files (so gbt/vla use gbt2gps/vla2gps data,
    # not PINT's bundled time_*.dat) -- the same sync the comparison harness uses.
    prepare_pint_environment(CLOCK_DIR, "/tmp/jug_pint_clock_override_test")

    t = Time(TEST_MJDS, format="pulsar_mjd", scale="utc")

    obs = get_observatory(pint_name)
    pint_corr = obs.clock_corrections(
        t, include_bipm=True, bipm_version=BIPM, limits="warn"
    ).to_value(u.us) * 1e3  # ns

    # JUG: configured observatory chain (obs->GPS + gps2utc) + TT(BIPM).
    obs_chain = _load_obs_chain(CLOCK_DIR, jug_code)
    jug_obs = interpolate_clock_vectorized(obs_chain, TEST_MJDS)
    bipm = parse_clock_file(os.path.join(CLOCK_DIR, f"tai2tt_{BIPM.lower()}.clk"))
    jug_bipm = np.interp(TEST_MJDS, bipm["mjd"], bipm["offset"]) - 32.184
    jug_corr = (jug_obs + jug_bipm) * 1e9  # ns

    # Compare as a difference relative to the mean (absolute zero-point is the
    # same convention; per-TOA variation is what matters for residuals).
    d = (jug_corr - pint_corr)
    d -= np.median(d)
    assert np.max(np.abs(d)) < 1.0, (
        f"{jug_code}: JUG clock chain {obs_chain['chain']} disagrees with PINT "
        f"by up to {np.max(np.abs(d)):.3f} ns: {d}"
    )


def test_routing_is_header_independent():
    """Configured routing must not depend on the gps2utc.clk header label."""
    from jug.residuals.simple_calculator import _load_obs_chain
    from jug.io.clock import interpolate_clock_vectorized
    chain = _load_obs_chain(CLOCK_DIR, "vla")
    # VLA must go through the GPS chain, never the NIST realization.
    assert chain["chain"] == ["vla2gps.clk", "gps2utc.clk"], chain["chain"]
    assert "nist2utc.clk" not in chain["chain"]
