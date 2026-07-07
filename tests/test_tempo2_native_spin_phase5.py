"""DEV ORACLE — native spin counterfactual vs pytempo (Phase 3).

Native spin uses ``phase5(bbat, torb)`` with ``torb`` defined as a closure against
``bbat``; a ~304 ns ``bbat_mjd`` assembly offset vs tempo2 may partially cancel in
``deltaT``. This test is not a substitute for ``acceptance_residual_sec`` parity.
See ``TEMPO2_PARITY.md`` § "formBats bat_mjd / bbat_mjd assembly".
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.tempo2_native.chain_jax import compute_tempo2_native_residuals_jax


def test_native_spin_wsrt167_vs_pytempo_acceptance(
    wsrt167_fixture, wsrt167_native_terms, wsrt167_pytempo_oracle
):
    params = parse_par_file(wsrt167_fixture["par_path"])
    toas = parse_tim_file_mjds(wsrt167_fixture["tim_path"])
    pn = np.array([int(t.flags["pn"]) for t in toas], dtype=np.int64)
    pn_add = np.full(len(toas), -1, dtype=np.int64)
    running = np.int64(-1)
    for i, toa in enumerate(toas):
        pn_add[i] = running
        if toa.flags.get("pnadd") is not None:
            running += np.int64(int(toa.flags["pnadd"]))
    residuals_sec, _, _ = compute_tempo2_native_residuals_jax(
        native_terms=wsrt167_native_terms,
        params=params,
        weights=np.ones(len(toas)),
        pulse_numbers=pn,
        pn_add=pn_add,
        jump_phase=None,
        tzr_phase=None,
        subtract_mean=True,
    )
    jug_sec = np.asarray(jax.device_get(residuals_sec), dtype=np.float64)
    pt_sec = wsrt167_pytempo_oracle.fields["acceptance_residual_sec"]
    delta_ns = (jug_sec - pt_sec) * 1e9
    # Native spin counterfactual; loose gate until full acceptance-residual closure.
    assert np.sqrt(np.mean(delta_ns**2)) < 20000.0
