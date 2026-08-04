"""Tests for TimingSession.parameter_table() presentation.

Guards three past defects:
  1. RAJ/DECJ printed sexagesimal pre-fit but radians post-fit (mixed units).
  2. Delta/sigma blank for RAJ/DECJ (string minus float is not computable).
  3. Rows emitted in dict order, so the table order varied between datasets.
"""

from pathlib import Path

import numpy as np
import pytest

from jug.engine.session import TimingSession
from jug.io.par_writer import canonical_param_order

DATA = Path(__file__).parent / "data_golden"


@pytest.fixture(scope="module")
def session():
    return TimingSession(DATA / "J1909_proper.par", DATA / "J1909_proper.tim")


def test_canonical_order_is_sectioned_and_numeric_aware():
    names = ["XDOT", "F1", "DECJ", "DM2", "F0", "PX", "RAJ", "FD10", "FD2",
             "JUMP10", "JUMP2", "PB", "A1", "DM1", "ZZUNKNOWN", "PMRA"]
    ordered = canonical_param_order(names)

    assert ordered.index("RAJ") < ordered.index("DECJ") < ordered.index("PMRA")
    assert ordered.index("PMRA") < ordered.index("F0") < ordered.index("DM1")
    assert ordered.index("DM2") < ordered.index("PX") < ordered.index("PB")
    assert ordered.index("PB") < ordered.index("A1") < ordered.index("XDOT")
    assert ordered.index("XDOT") < ordered.index("FD2") < ordered.index("FD10")
    # Numbered params sort numerically, not lexicographically.
    assert ordered.index("JUMP2") < ordered.index("JUMP10")
    # Unrecognised names go last, and nothing is dropped or duplicated.
    assert ordered[-1] == "ZZUNKNOWN"
    assert sorted(ordered) == sorted(set(names))
    assert canonical_param_order(ordered) == ordered


def test_parameter_table_rows_follow_canonical_order(session, capsys):
    session.parameter_table()
    rows = [ln.split()[0] for ln in capsys.readouterr().out.splitlines()[2:] if ln.strip()]
    assert rows == canonical_param_order(rows)


def test_raj_decj_printed_in_radians_with_delta_sigma(session, capsys):
    """Both columns numeric in radians, and Delta/sigma populated."""
    fit = session.fit_parameters(fit_params=["RAJ", "DECJ"], max_iter=1)
    session.parameter_table(fit)
    out = capsys.readouterr().out

    for name in ("RAJ", "DECJ"):
        row = next(ln for ln in out.splitlines() if ln.startswith(name))
        fields = row.split()
        assert ":" not in row, f"{name} still printed sexagesimal: {row}"
        pre, post, unc, delta_sigma = (float(f) for f in fields[1:5])
        # J1909-3744 is at RA ~5.02 rad, Dec ~-0.66 rad.
        assert abs(pre) < 2 * np.pi and abs(post) < 2 * np.pi
        assert pre == pytest.approx(post, abs=1e-6)
        assert unc > 0
        assert delta_sigma == pytest.approx(abs(post - pre) / unc, rel=1e-2)
