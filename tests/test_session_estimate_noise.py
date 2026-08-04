"""Tests for TimingSession.estimate_noise().

The wrapper must (a) reproduce a direct estimate_noise_parameters() call built
from the same session, and (b) forward component toggles so users can choose
what is estimated rather than always estimating everything.

Kept deliberately short (few SVI steps): these assert plumbing, not convergence
-- amplitude-convention accuracy is covered by test_map_noise_convention.py.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("optax")

from jug.engine.session import TimingSession
from jug.noise.map_estimator import estimate_noise_parameters

DATA = Path(__file__).parent / "data_golden"
FAST = dict(batch_size=100, max_num_batches=2, seed=7)


@pytest.fixture(scope="module")
def session():
    return TimingSession(DATA / "J1909_proper.par", DATA / "J1909_proper.tim")


def test_estimate_noise_matches_direct_call(session):
    """The wrapper is a pure convenience layer over the estimator."""
    pre = session.compute_residuals()
    direct = estimate_noise_parameters(
        residuals_sec=pre["residuals_us"] * 1e-6,
        errors_sec=pre["errors_us"] * 1e-6,
        toas_mjd=pre["tdb_mjd"],
        freq_mhz=pre["freq_bary_mhz"],
        toa_flags=pre["toa_flags"],
        params=session.params,
        **FAST,
    )
    wrapped = session.estimate_noise(**FAST)

    assert set(wrapped.params) == set(direct.params)
    for key, value in direct.params.items():
        assert wrapped.params[key] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_estimate_noise_component_toggles_are_forwarded(session):
    """Users can switch individual processes off."""
    no_red = session.estimate_noise(include_red_noise=False, **FAST)
    assert not any(k.startswith("TNRED") for k in no_red.params)
    assert any(k.startswith("TNDM") for k in no_red.params)

    white_only = session.estimate_noise(
        include_red_noise=False, include_dm_noise=False, include_ecorr=False, **FAST
    )
    assert not any(k.startswith(("TNRED", "TNDM")) for k in white_only.params)
    assert not any(k.startswith("ECORR") for k in white_only.params)
    # EFAC/EQUAD are always estimated.
    assert any(k.startswith("EFAC") for k in white_only.params)
    assert any(k.startswith("EQUAD") for k in white_only.params)


def test_estimate_noise_accepts_supplied_residuals(session):
    """Post-fit residuals can be passed in instead of the session's own."""
    pre = session.compute_residuals()
    est = session.estimate_noise(residuals_us=pre["residuals_us"] * 0.5, **FAST)
    assert any(k.startswith("EFAC") for k in est.params)

    with pytest.raises(ValueError, match="TOAs"):
        session.estimate_noise(residuals_us=np.zeros(3), **FAST)
