"""Unit tests for FDJUMPDM support (Tempo2 fdjumpIdx == -2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jug.fitting.derivatives_fdjump import (
    compute_fdjump_delay,
    compute_fdjump_derivatives,
)
from jug.fitting.forward_delay import _fdjump_delay
from jug.io.par_reader import parse_par_file
from jug.io.par_writer import write_par_file
from jug.utils.constants import K_DM_SEC


_MINI_PAR = """\
PSRJ           J0000+0000
RAJ            00:00:00.0
DECJ           +00:00:00.0
F0             100.0
F1             0.0
PEPOCH         55000.0
DM             10.0
EPHEM          DE436
CLK            TT(BIPM2021)
UNITS          TDB
FDJUMPLOG Y
FDJUMP_SCALE LOG
FDJUMP1 -pta EPTA 1.0e-4 1
FDJUMPDM -pta EPTA 0.012345 1
FDJUMPDM -pta PPTA -0.001 0
"""


@pytest.fixture
def fdjumpdm_par(tmp_path: Path) -> Path:
    path = tmp_path / "mini.par"
    path.write_text(_MINI_PAR)
    return path


def test_parse_fdjumpdm(fdjumpdm_par: Path):
    params = parse_par_file(fdjumpdm_par)

    assert params["FDJUMPDM_1"] == pytest.approx(0.012345)
    assert params["FDJUMPDM_2"] == pytest.approx(-0.001)
    assert params["_fit_flags"]["FDJUMPDM_1"] is True
    assert "FDJUMPDM_2" not in params["_fit_flags"]

    meta1 = params["_fdjump_meta_FDJUMPDM_1"]
    assert meta1["fd_index"] == -2
    assert meta1["kind"] == "dm"
    assert meta1["flag_name"] == "pta"
    assert meta1["flag_value"] == "EPTA"

    meta2 = params["_fdjump_meta_FDJUMPDM_2"]
    assert meta2["flag_value"] == "PPTA"

    # Regular FDJUMP still parsed
    fdjump1_keys = [k for k in params if k.startswith("FDJUMP1_")]
    assert len(fdjump1_keys) == 1
    assert params["_fdjump_log"] is True


def test_fdjumpdm_delay_and_derivatives_match_dm_kernel():
    freq_mhz = np.array([400.0, 800.0, 1400.0], dtype=np.float64)
    mask = np.array([True, True, False])
    params = {
        "FDJUMPDM_1": 0.02,
        "_fdjump_meta_FDJUMPDM_1": {
            "fd_index": -2,
            "kind": "dm",
            "flag_name": "pta",
            "flag_value": "EPTA",
            "log_scale": True,
        },
    }
    masks = {"FDJUMPDM_1": mask}

    delay = compute_fdjump_delay(params, freq_mhz, ["FDJUMPDM_1"], masks)
    expected = np.where(mask, 0.02 * K_DM_SEC / (freq_mhz ** 2), 0.0)
    np.testing.assert_allclose(delay, expected, rtol=1e-14)

    derivs = compute_fdjump_derivatives(
        params, freq_mhz, ["FDJUMPDM_1"], fdjump_masks=masks
    )
    np.testing.assert_allclose(
        derivs["FDJUMPDM_1"], np.where(mask, K_DM_SEC / (freq_mhz ** 2), 0.0), rtol=1e-14
    )


def test_forward_delay_fdjumpdm_traceable():
    jnp = pytest.importorskip("jax.numpy")
    freq_mhz = np.array([500.0, 1000.0], dtype=np.float64)
    mask = np.array([True, False])
    params = {
        "FDJUMPDM_1": 0.01,
        "_fdjump_meta_FDJUMPDM_1": {
            "fd_index": -2,
            "kind": "dm",
            "flag_name": "pta",
            "flag_value": "EPTA",
            "log_scale": True,
        },
    }
    delay = np.asarray(
        _fdjump_delay(jnp, params, freq_mhz, ["FDJUMPDM_1"], {"FDJUMPDM_1": mask})
    )
    expected = np.array([0.01 * K_DM_SEC / (500.0 ** 2), 0.0])
    np.testing.assert_allclose(delay, expected, rtol=1e-12)


def test_write_fdjumpdm_roundtrip(fdjumpdm_par: Path, tmp_path: Path):
    params = parse_par_file(fdjumpdm_par)
    params["FDJUMPDM_1"] = 0.05
    out = tmp_path / "out.par"
    fdjump1_key = next(k for k in params if k.startswith("FDJUMP1_"))
    write_par_file(params, out, fit_params={"FDJUMPDM_1", fdjump1_key})

    text = out.read_text()
    assert "FDJUMPDM" in text
    assert "0.05" in text or "5e-02" in text.lower() or "5.0e-02" in text.lower()

    reparsed = parse_par_file(out)
    assert reparsed["FDJUMPDM_1"] == pytest.approx(0.05)
    assert reparsed["FDJUMPDM_2"] == pytest.approx(-0.001)
    assert reparsed["_fdjump_meta_FDJUMPDM_1"]["kind"] == "dm"


_PINT_DIALECT_PAR = """\
PSRJ           J0000+0000
RAJ            00:00:00.0
DECJ           +00:00:00.0
F0             100.0
F1             0.0
PEPOCH         55000.0
DM             10.0
EPHEM          DE436
CLK            TT(BIPM2021)
UNITS          TDB
FDJUMPLOG Y
FD1JUMP -sys GM_GWB_500_100_b1 0.01 1
FD1JUMP -sys GM_GWB_1460_100_b1 0.02 1
FD2JUMP -sys GM_GWB_500_100_b1 0.001 1
FDJUMPDM -sys GM_GWB_500_100_b1 0.00002 1
"""


def test_parse_pint_fdxjump_dialect(tmp_path: Path):
    path = tmp_path / "pint_fdjump.par"
    path.write_text(_PINT_DIALECT_PAR)
    params = parse_par_file(path)

    fd1 = sorted(k for k in params if k.startswith("FDJUMP1_"))
    fd2 = sorted(k for k in params if k.startswith("FDJUMP2_"))
    assert len(fd1) == 2
    assert len(fd2) == 1
    assert params[fd1[0]] == pytest.approx(0.01)
    assert params[fd1[1]] == pytest.approx(0.02)
    assert params[fd2[0]] == pytest.approx(0.001)
    assert params["_fdjump_meta_" + fd1[0]]["dialect"] == "pint"
    assert params["_fdjump_meta_" + fd1[0]]["fd_index"] == 1
    assert params["_fdjump_meta_" + fd2[0]]["fd_index"] == 2
    assert params["FDJUMPDM_1"] == pytest.approx(0.00002)


def test_write_preserves_pint_fdxjump_dialect(tmp_path: Path):
    path = tmp_path / "pint_fdjump.par"
    path.write_text(_PINT_DIALECT_PAR)
    params = parse_par_file(path)
    fd1 = sorted(k for k in params if k.startswith("FDJUMP1_"))
    params[fd1[0]] = 0.03

    out = tmp_path / "out.par"
    write_par_file(params, out, fit_params=set(fd1))
    text = out.read_text()
    assert "FD1JUMP" in text
    assert "FD2JUMP" in text
    assert "FDJUMP1" not in text  # keep PINT spelling on rewrite

    reparsed = parse_par_file(out)
    fd1_out = sorted(k for k in reparsed if k.startswith("FDJUMP1_"))
    assert reparsed[fd1_out[0]] == pytest.approx(0.03)
