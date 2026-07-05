"""IFTE time-ephemeris reader (tempo2 ``ifteph.C`` port)."""

from __future__ import annotations

import pytest

from jug.utils.ifteph import ifte_close, ifte_delta_t_sec, ifte_init


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
