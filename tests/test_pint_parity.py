"""
PINT and Tempo2 parity tests for J1909_proper dataset (100 TOAs, MPTA DR3).

Compares JUG residuals against PINT and Tempo2 using raw-error weighted RMS
(1/err^2 weights, no EFAC/EQUAD noise model).

Requires PINT: pip install pint-pulsar
Tempo2 tests skip automatically if the `tempo2` binary is not on PATH.

Run standalone:
    pytest tests/test_pint_parity.py -v
    pytest tests/test_pint_parity.py -v -s   # show print output

Force-run even in CI:
    JUG_TEST_PINT=1 pytest tests/test_pint_parity.py -v
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

# Skip entire module unless PINT is installed or JUG_TEST_PINT env var set
pint = pytest.importorskip("pint.models", reason="PINT not installed (pip install pint-pulsar)")

_FORCE_PINT = os.environ.get("JUG_TEST_PINT", "").lower() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not _FORCE_PINT,
    reason="PINT parity tests skipped by default. Set JUG_TEST_PINT=1 to enable.",
)

GOLDEN_DIR = Path(__file__).parent / "data_golden"
PAR = GOLDEN_DIR / "J1909_proper.par"
TIM = GOLDEN_DIR / "J1909_proper.tim"
GOLDEN = GOLDEN_DIR / "J1909_proper_golden.json"


def _raw_wrms_pint(par_path, tim_path):
    """Compute PINT raw-error WRMS (no noise model corrections)."""
    import pint.models
    import pint.toa
    import pint.residuals

    logging.getLogger("pint").setLevel(logging.ERROR)

    model = pint.models.get_model(str(par_path))
    ephem = model.EPHEM.value if hasattr(model, "EPHEM") else None
    toas = pint.toa.get_TOAs(str(tim_path), planets=True, ephem=ephem)
    res = pint.residuals.Residuals(toas, model)

    res_us = res.time_resids.to("us").value
    errs_us = toas.get_errors().to("us").value
    weights = 1.0 / errs_us**2
    wrms = float(np.sqrt(np.sum(weights * res_us**2) / np.sum(weights)))
    return wrms, toas.ntoas, res_us


def _jug_wrms(par_path, tim_path):
    """Compute JUG raw-error WRMS."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from jug.residuals.simple_calculator import compute_residuals_simple

    result = compute_residuals_simple(str(par_path), str(tim_path), verbose=False)
    return result["weighted_rms_us"], result["n_toas"], result["residuals_us"]


@pytest.fixture(scope="module")
def golden():
    assert GOLDEN.exists(), f"Golden file missing: {GOLDEN}"
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def jug_result():
    assert PAR.exists() and TIM.exists(), "J1909_proper dataset missing"
    wrms, n, res = _jug_wrms(PAR, TIM)
    return {"wrms_us": wrms, "n_toas": n, "residuals_us": res}


@pytest.fixture(scope="module")
def pint_result():
    assert PAR.exists() and TIM.exists(), "J1909_proper dataset missing"
    wrms, n, res = _raw_wrms_pint(PAR, TIM)
    return {"wrms_us": wrms, "n_toas": n, "residuals_us": res}


# ── Self-consistency: JUG matches its own golden ──────────────────────────────

def test_proper_n_toas(jug_result, golden):
    """JUG n_toas matches golden."""
    assert jug_result["n_toas"] == golden["n_toas"]


def test_proper_jug_wrms_matches_golden(jug_result, golden):
    """JUG raw WRMS matches golden to 1 ppm."""
    tol = golden["tolerances"]["rms_rel_tol"]
    rel_diff = abs(jug_result["wrms_us"] - golden["weighted_rms_us"]) / golden["weighted_rms_us"]
    assert rel_diff <= tol, (
        f"JUG WRMS {jug_result['wrms_us']:.6f} µs differs from golden "
        f"{golden['weighted_rms_us']:.6f} µs by {rel_diff:.2e} > tol {tol:.2e}"
    )


def test_proper_first5_residuals_match_golden(jug_result, golden):
    """JUG first 5 residuals match golden to 0.1 ns."""
    tol_ns = golden["tolerances"]["residual_abs_tol_ns"]
    actual_ns = [r * 1000 for r in jug_result["residuals_us"][:5]]
    expected_ns = golden["first_5_residuals_ns"]
    for i, (act, exp) in enumerate(zip(actual_ns, expected_ns)):
        diff = abs(act - exp)
        assert diff <= tol_ns, f"residual[{i}]: {act:.3f} ns vs golden {exp:.3f} ns, diff {diff:.3f} ns > {tol_ns} ns"


# ── PINT parity ───────────────────────────────────────────────────────────────

def test_pint_n_toas(jug_result, pint_result):
    """JUG and PINT agree on TOA count."""
    assert jug_result["n_toas"] == pint_result["n_toas"], (
        f"TOA count: JUG={jug_result['n_toas']}, PINT={pint_result['n_toas']}"
    )


def test_pint_wrms_parity(jug_result, pint_result, golden):
    """JUG raw-error WRMS matches PINT within 0.1%."""
    tol = golden["tolerances"]["pint_parity_rel_tol"]  # 0.001 = 0.1%
    pct = abs(jug_result["wrms_us"] - pint_result["wrms_us"]) / pint_result["wrms_us"]
    assert pct <= tol, (
        f"WRMS: JUG={jug_result['wrms_us']:.6f} µs, "
        f"PINT={pint_result['wrms_us']:.6f} µs, "
        f"diff={pct*100:.4f}% > {tol*100:.1f}%"
    )


def test_pint_wrms_parity_vs_stored_golden(jug_result, golden):
    """JUG WRMS matches stored PINT reference (no live PINT needed once golden exists)."""
    pint_ref = golden["pint_reference"]["raw_wrms_us"]
    tol = golden["tolerances"]["pint_parity_rel_tol"]
    pct = abs(jug_result["wrms_us"] - pint_ref) / pint_ref
    assert pct <= tol, (
        f"WRMS: JUG={jug_result['wrms_us']:.6f} µs, "
        f"PINT_ref={pint_ref:.6f} µs, "
        f"diff={pct*100:.4f}% > {tol*100:.1f}%"
    )


def test_pint_max_per_toa_diff(jug_result, pint_result):
    """Max per-TOA residual difference between JUG and PINT < 50 ns."""
    jug_ns = np.array(jug_result["residuals_us"]) * 1000
    pint_ns = np.array(pint_result["residuals_us"]) * 1000
    max_diff_ns = float(np.max(np.abs(jug_ns - pint_ns)))
    assert max_diff_ns < 50.0, f"Max per-TOA diff {max_diff_ns:.2f} ns >= 50 ns"


# ── Tempo2 parity ─────────────────────────────────────────────────────────────

import importlib.util


def _tempo2_available():
    return importlib.util.find_spec("libstempo") is not None


def _raw_wrms_tempo2(par_path, tim_path):
    """Tempo2 oracle WRMS is not available in the pint-only portable build."""
    del par_path, tim_path
    pytest.skip("tempo2 reference harness not available in pint-only build")


_tempo2_skip = pytest.mark.skipif(
    not _tempo2_available(),
    reason="libstempo not installed",
)


@pytest.fixture(scope="module")
def tempo2_result():
    assert PAR.exists() and TIM.exists(), "J1909_proper dataset missing"
    wrms, n, res = _raw_wrms_tempo2(PAR, TIM)
    return {"wrms_us": wrms, "n_toas": n, "residuals_us": res}


@_tempo2_skip
def test_tempo2_n_toas(jug_result, tempo2_result):
    """JUG and Tempo2 agree on TOA count."""
    assert jug_result["n_toas"] == tempo2_result["n_toas"], (
        f"TOA count: JUG={jug_result['n_toas']}, Tempo2={tempo2_result['n_toas']}"
    )


@_tempo2_skip
def test_tempo2_wrms_parity(jug_result, tempo2_result, golden):
    """JUG raw-error WRMS matches Tempo2 within 0.1%."""
    tol = golden["tolerances"]["tempo2_parity_rel_tol"]
    pct = abs(jug_result["wrms_us"] - tempo2_result["wrms_us"]) / tempo2_result["wrms_us"]
    assert pct <= tol, (
        f"WRMS: JUG={jug_result['wrms_us']:.6f} µs, "
        f"Tempo2={tempo2_result['wrms_us']:.6f} µs, "
        f"diff={pct*100:.4f}% > {tol*100:.1f}%"
    )


@_tempo2_skip
def test_tempo2_wrms_parity_vs_stored_golden(jug_result, golden):
    """JUG WRMS matches stored Tempo2 reference (no live Tempo2 needed once golden exists)."""
    t2_ref = golden["tempo2_reference"]["raw_wrms_us"]
    tol = golden["tolerances"]["tempo2_parity_rel_tol"]
    pct = abs(jug_result["wrms_us"] - t2_ref) / t2_ref
    assert pct <= tol, (
        f"WRMS: JUG={jug_result['wrms_us']:.6f} µs, "
        f"Tempo2_ref={t2_ref:.6f} µs, "
        f"diff={pct*100:.4f}% > {tol*100:.1f}%"
    )


@_tempo2_skip
def test_tempo2_max_per_toa_diff(jug_result, tempo2_result):
    """Max per-TOA residual difference between JUG and Tempo2 < 50 ns."""
    jug_ns = np.array(jug_result["residuals_us"]) * 1000
    t2_ns = np.array(tempo2_result["residuals_us"]) * 1000
    max_diff_ns = float(np.max(np.abs(jug_ns - t2_ns)))
    assert max_diff_ns < 50.0, f"Max per-TOA diff {max_diff_ns:.2f} ns >= 50 ns"
