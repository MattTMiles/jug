"""IFTE time-ephemeris reader (tempo2 ``ifteph.C`` port)."""

from __future__ import annotations

import numpy as np
import pytest

from jug.utils.ifteph import (
    IFTE_LC,
    IFTE_TEPH0_SEC,
    ifte_close,
    ifte_delta_t_sec,
    ifte_delta_t_sec_jax,
    ifte_init,
    load_ifte_coeff_tables,
)
from jug.utils.timescales import IFTE_K


@pytest.fixture(autouse=True)
def _reset_ifte():
    ifte_close()
    yield
    ifte_close()


def test_ifte_init_reads_te405_header():
    ifte_init()
    from jug.utils import ifteph

    state = ifteph._STATE
    assert state.start_jd == pytest.approx(2438736.5)
    assert state.end_jd == pytest.approx(2469808.5)
    assert state.step_jd == pytest.approx(32.0)
    assert state.reclen == 1808
    assert state.swap_endian is True


def test_ifte_delta_t_sec_scales_days_to_seconds():
    ifte_init()
    sec = ifte_delta_t_sec(54100.0)
    assert abs(sec) < 1.0
    assert sec != 0.0


def test_ifte_jax_matches_host():
    tables = load_ifte_coeff_tables()
    mjds = np.array([54100.0, 55000.123456, 56000.987654], dtype=np.float64)
    host = np.array([ifte_delta_t_sec(float(m)) for m in mjds], dtype=np.float64)
    import jax
    import jax.numpy as jnp

    jax_delta = np.asarray(
        jax.device_get(
            ifte_delta_t_sec_jax(
                jnp.asarray(mjds, dtype=jnp.float64),
                ifte_records=jnp.asarray(tables.records, dtype=jnp.float64),
                ifte_start_jd=jnp.asarray(tables.start_jd, dtype=jnp.float64),
                ifte_end_jd=jnp.asarray(tables.end_jd, dtype=jnp.float64),
                ifte_step_jd=jnp.asarray(tables.step_jd, dtype=jnp.float64),
                ifte_coef_offset=int(tables.coef_offset),
                ifte_ncf=int(tables.ncf),
                ifte_na=int(tables.na),
            )
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(jax_delta, host, rtol=0.0, atol=1e-15)


@pytest.mark.dev_oracle
@pytest.mark.tempo2
def test_ifte_delta_matches_pytempo_implied_wsrt167():
    pytest.importorskip("pytempo")
    from tempo2_fixtures import get_tempo2_fixture
    from pytempo.sandbox import tempopulsar

    fixture = get_tempo2_fixture("wsrt167")
    psr = tempopulsar(
        parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False
    )
    diag = psr.toa_diagnostics(removemean=False)
    ifte_init()
    trace = [0, 42, 85, 166]
    for i in trace:
        tt_pt = float(diag["correction_tt_sec"][i])
        sat = float(diag["sat"][i]) if "sat" in diag else None
        if sat is None:
            from jug.io.tim_reader import parse_tim_file_mjds

            toas = parse_tim_file_mjds(fixture["tim_path"])
            sat = toas[i].mjd_int + toas[i].mjd_frac
        mjd_tt = sat + tt_pt / 86400.0
        obs_scaled = float(diag["obs_term_raw_sec"][i]) / IFTE_K
        delta_implied_pt = (
            float(diag["correction_tt_teph_sec"][i]) - IFTE_TEPH0_SEC - obs_scaled
        ) * (1.0 - IFTE_LC)
        delta_jug = ifte_delta_t_sec(mjd_tt)
        assert abs(delta_jug - delta_implied_pt) < 1e-12
