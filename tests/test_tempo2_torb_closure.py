"""DEV ORACLE — native torb closure vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file
from jug.residuals.tempo2.compensated import split_mjd_to_daysec
from jug.residuals.tempo2.formbats_jax import compute_torb_closure_daysec
from jug.residuals.tempo2.spin_jax import pepoch_parts_from_value
from tempo2_test_helpers import delta_ns


def _torb_from_native(native, pepoch: float) -> np.ndarray:
    bbat_int, bbat_sec = split_mjd_to_daysec(native.bbat_mjd)
    _, pep_int, pep_frac = pepoch_parts_from_value(
        jnp.asarray(pepoch, dtype=jnp.float64)
    )
    return np.asarray(
        jax.device_get(
            compute_torb_closure_daysec(
                bbat_int,
                bbat_sec,
                native.dt_emission_sec,
                pep_int,
                pep_frac,
            )
        ),
        dtype=np.float64,
    )


def test_native_torb_closure_strict_formbats_interim_wsrt167(
    wsrt167_fixture, wsrt167_native_terms, wsrt167_pytempo_oracle
):
    pepoch = float(parse_par_file(wsrt167_fixture["par_path"])["PEPOCH"])
    torb = _torb_from_native(wsrt167_native_terms, pepoch)
    delta = delta_ns(torb, wsrt167_pytempo_oracle.fields["torb_sec"])
    assert np.sqrt(np.mean(delta**2)) < 25e6
