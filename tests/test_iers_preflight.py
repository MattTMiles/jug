"""IERS/EOP preflight checks for geometry (ITRF→GCRS)."""

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from jug.io.clock import (
    check_iers_coverage,
    iers_strict_enabled,
    raise_on_iers_failure,
    warn_on_iers_failure,
)


def test_check_iers_coverage_ok_when_probe_succeeds():
    valid, issues = check_iers_coverage(56000.0, 56500.0, verbose=False)
    assert valid is True
    assert not any(i.get("severity") == "error" for i in issues)


def test_check_iers_coverage_fails_on_empty_table(monkeypatch):
    class EmptyTable:
        colnames = ["MJD", "UT1_UTC"]

        def __getitem__(self, key):
            return np.array([])

    import astropy.utils.iers as astropy_iers

    monkeypatch.setattr(
        astropy_iers.earth_orientation_table, "get", lambda: EmptyTable()
    )

    valid, issues = check_iers_coverage(56000.0, 56500.0, verbose=False)
    assert valid is False
    assert any(i.get("severity") == "error" for i in issues)
    assert "IERS" in issues[0]["message"]


def test_check_iers_coverage_fails_on_probe_error(monkeypatch):
    import astropy.utils.iers as astropy_iers
    from jug.io import clock as clock_mod

    real_get = astropy_iers.earth_orientation_table.get

    def probe_raises(_mjd):
        raise ValueError("not enough values to unpack (expected 3, got 0)")

    monkeypatch.setattr(astropy_iers.earth_orientation_table, "get", real_get)
    monkeypatch.setattr(clock_mod, "_probe_iers_gcrs_transform", probe_raises)

    valid, issues = check_iers_coverage(56000.0, 56500.0, verbose=False)
    assert valid is False
    assert any("ITRF→GCRS transform failed" in i["message"] for i in issues)


def test_raise_on_iers_failure():
    with pytest.raises(RuntimeError, match="IERS"):
        raise_on_iers_failure(
            False,
            [{"severity": "error", "message": "EOP/IERS ERROR: table is empty."}],
        )


def test_iers_strict_enabled_under_pytest():
    assert iers_strict_enabled() is True
    assert iers_strict_enabled(iers_policy="warn") is True
    assert iers_strict_enabled(iers_policy="strict") is True


def test_warn_on_iers_failure_emits_user_warning():
    with pytest.warns(UserWarning, match="IERS"):
        warn_on_iers_failure(
            False,
            [{"severity": "error", "message": "EOP/IERS ERROR: table is empty."}],
        )


def test_load_clock_corrections_warns_on_iers_failure_non_strict(monkeypatch):
    from jug.residuals.simple_calculator import _load_clock_corrections

    def fake_iers(*args, **kwargs):
        return False, [{"severity": "error", "message": "EOP/IERS ERROR: broken cache."}]

    monkeypatch.setattr("jug.io.clock.check_iers_coverage", fake_iers)
    monkeypatch.setattr("jug.io.clock.iers_strict_enabled", lambda *args, **kwargs: False)

    clock_dir = repo_root / "data" / "clock"
    params = {"CLK": "TT(BIPM2024)"}
    mjd_utc = np.array([56000.0, 56500.0])

    with pytest.warns(UserWarning, match="IERS"):
        out = _load_clock_corrections(
            "ao", ["ao"], clock_dir, params, mjd_utc, verbose=False
        )
    assert out["obs_clock"] is not None


    from jug.residuals.simple_calculator import _load_clock_corrections

    def fake_iers(*args, **kwargs):
        return False, [{"severity": "error", "message": "EOP/IERS ERROR: broken cache."}]

    monkeypatch.setattr("jug.io.clock.check_iers_coverage", fake_iers)
    monkeypatch.setattr("jug.io.clock.iers_strict_enabled", lambda *args, **kwargs: True)

    clock_dir = repo_root / "data" / "clock"
    params = {"CLK": "TT(BIPM2024)"}
    mjd_utc = np.array([56000.0, 56500.0])

    with pytest.raises(RuntimeError, match="IERS"):
        _load_clock_corrections(
            "ao", ["ao"], clock_dir, params, mjd_utc, verbose=False
        )
