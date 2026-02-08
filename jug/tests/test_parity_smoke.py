"""
Parity smoke pack — cross-pulsar regression tests.
====================================================

Runs the JUG residuals engine on multiple pulsars and verifies:
1. WRMS is within a physically-reasonable range.
2. Results are deterministic (two runs agree bit-for-bit).
3. Config fingerprint is JUG-compatible (TDB, DE440, BIPM2024).
4. (When Tempo2 is available) per-TOA parity within golden thresholds.

Pulsars covered:
    J1909-3744  — ELL1 binary, SINI+M2, DM1/DM2, FD1–FD9, NE_SW
    J0614-3329  — DD binary, SINI+M2, DM1/DM2, PBDOT, FD1

Run with:
    pytest jug/tests/test_parity_smoke.py -v -o "addopts="
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.engine.session import TimingSession
from jug.testing.fingerprint import extract_fingerprint, validate_jug_compatible

JUG_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PULSARS = JUG_ROOT / "data" / "pulsars"
DATA_GOLDEN = JUG_ROOT / "tests" / "data_golden"


def _skip_no_data(pulsar):
    par = DATA_PULSARS / f"{pulsar}_tdb.par"
    tim = DATA_PULSARS / f"{pulsar}.tim"
    if not par.exists() or not tim.exists():
        pytest.skip(f"{pulsar} data not found")
    return par, tim


def _has_tempo2():
    return shutil.which("tempo2") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks — WRMS in physical range
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeSanity:
    """Verify each pulsar produces residuals with WRMS in a plausible range."""

    @pytest.mark.parametrize(
        "pulsar,max_wrms_us",
        [
            ("J1909-3744", 1.0),   # TRES ≈ 0.40 μs
            ("J0614-3329", 5.0),   # TRES ≈ 2.34 μs
        ],
    )
    def test_wrms_in_range(self, pulsar, max_wrms_us):
        par, tim = _skip_no_data(pulsar)
        result = compute_residuals_simple(par, tim, verbose=False)
        wrms = result["weighted_rms_us"]
        assert wrms > 0, f"{pulsar}: WRMS is zero or negative"
        assert wrms < max_wrms_us, (
            f"{pulsar}: WRMS {wrms:.3f} μs exceeds {max_wrms_us} μs"
        )

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_n_toas_match_par(self, pulsar):
        """Number of TOAs must match NTOA in par file."""
        par, tim = _skip_no_data(pulsar)
        result = compute_residuals_simple(par, tim, verbose=False)
        from jug.io.par_reader import parse_par_file

        params = parse_par_file(par)
        ntoa_par = int(params.get("NTOA", 0))
        if ntoa_par > 0:
            assert result["n_toas"] == ntoa_par, (
                f"{pulsar}: n_toas={result['n_toas']} != NTOA={ntoa_par}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeDeterminism:
    """Two independent runs on the same pulsar must agree bit-for-bit."""

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_bitexact(self, pulsar):
        par, tim = _skip_no_data(pulsar)
        r1 = compute_residuals_simple(par, tim, verbose=False)
        r2 = compute_residuals_simple(par, tim, verbose=False)
        np.testing.assert_array_equal(r1["residuals_us"], r2["residuals_us"])


# ──────────────────────────────────────────────────────────────────────────────
# Config fingerprinting
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeFingerprint:
    """Verify par files are JUG-compatible (TDB, DE440, BIPM2024)."""

    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_jug_compatible(self, pulsar):
        par, _ = _skip_no_data(pulsar)
        fp = extract_fingerprint(par)
        ok, issues = validate_jug_compatible(fp)
        assert ok, f"{pulsar} par file not JUG-compatible: {issues}"


# ──────────────────────────────────────────────────────────────────────────────
# Fitting smoke — converges, RMS improves
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeFitting:
    """Verify fitting converges and improves RMS for a few core params."""

    @pytest.mark.parametrize(
        "pulsar,fit_params",
        [
            ("J1909-3744", ["F0", "F1", "DM"]),
            ("J0614-3329", ["F0", "F1", "DM"]),
        ],
    )
    def test_fit_converges(self, pulsar, fit_params):
        par, tim = _skip_no_data(pulsar)
        session = TimingSession(par, tim, verbose=False)
        prefit = session.compute_residuals(subtract_tzr=True)
        prefit_rms = prefit["rms_us"]

        # Reset session for clean fit
        session2 = TimingSession(par, tim, verbose=False)
        _ = session2.compute_residuals(subtract_tzr=False)
        result = session2.fit_parameters(
            fit_params=fit_params,
            max_iter=25,
            convergence_threshold=1e-14,
            solver_mode="exact",
            verbose=False,
        )

        assert result["converged"], f"{pulsar}: fit did not converge"
        assert result["final_rms"] <= prefit_rms * 1.001, (
            f"{pulsar}: postfit RMS {result['final_rms']:.4f} > prefit {prefit_rms:.4f}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Binary-model coverage
# ──────────────────────────────────────────────────────────────────────────────

class TestSmokeBinaryModels:
    """Verify different binary models produce reasonable binary delays."""

    def test_ell1_j1909(self):
        """ELL1 binary model for J1909-3744 (short-period, very circular)."""
        par, tim = _skip_no_data("J1909-3744")
        result = compute_residuals_simple(par, tim, verbose=False)
        assert result["prebinary_delay_sec"] is not None
        # Binary delay should have sub-μs variation for well-timed pulsar
        binary_contribution = np.std(result["prebinary_delay_sec"])
        assert binary_contribution > 0, "Binary delay has zero variation"

    def test_dd_j0614(self):
        """DD binary model for J0614-3329 (long-period, small eccentricity)."""
        par, tim = _skip_no_data("J0614-3329")
        result = compute_residuals_simple(par, tim, verbose=False)
        assert result["prebinary_delay_sec"] is not None
        binary_contribution = np.std(result["prebinary_delay_sec"])
        assert binary_contribution > 0, "Binary delay has zero variation"


# ──────────────────────────────────────────────────────────────────────────────
# Tempo2 parity (integration, requires tempo2)
# ──────────────────────────────────────────────────────────────────────────────

def _run_tempo2(par, tim):
    """Run Tempo2 and return per-TOA residuals."""
    fmt = "{bat} {freq} {post} {err}\\n"
    cmd = [
        "tempo2", "-f", str(par), str(tim),
        "-output", "general2",
        "-s", fmt,
        "-nobs", "100000",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"Tempo2 failed: {proc.stderr[:200]}")

    lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
    data_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 4:
            try:
                float(parts[0])
                data_lines.append(parts)
            except ValueError:
                continue

    bat = np.array([float(p[0]) for p in data_lines])
    post = np.array([float(p[2]) for p in data_lines])
    return bat, post


@pytest.mark.skipif(not _has_tempo2(), reason="tempo2 not on PATH")
class TestTempo2Parity:
    """Per-TOA comparison against Tempo2.

    Thresholds are per-pulsar because the clock-chain systematic offset
    varies with observation span and frequency coverage.

    - J1909-3744: max |Δ| < 100 ns  (ELL1, short PB, 10k TOAs)
    - J0614-3329: max |Δ| < 500 ns  (DD, long PB, Shapiro + OM)
    """

    _THRESHOLDS = {
        "J1909-3744": {"max_abs_ns": 100.0, "wrms_diff_ns": 2.0},
        "J0614-3329": {"max_abs_ns": 500.0, "wrms_diff_ns": 5.0},
    }

    @pytest.mark.integration
    @pytest.mark.parametrize("pulsar", ["J1909-3744", "J0614-3329"])
    def test_per_toa_parity(self, pulsar):
        par, tim = _skip_no_data(pulsar)
        thresholds = self._THRESHOLDS[pulsar]

        # JUG
        result = compute_residuals_simple(par, tim, verbose=False)
        jug_res = result["residuals_us"] * 1e-6  # → seconds

        # Tempo2
        t2_bat, t2_post = _run_tempo2(par, tim)

        # Match by index (same input file → same ordering)
        n = min(len(jug_res), len(t2_post))
        assert n > 0, f"{pulsar}: no matched TOAs"

        delta_ns = (jug_res[:n] - t2_post[:n]) * 1e9
        max_abs = float(np.max(np.abs(delta_ns)))

        assert max_abs < thresholds["max_abs_ns"], (
            f"{pulsar}: max per-TOA |Δ| = {max_abs:.1f} ns > {thresholds['max_abs_ns']} ns threshold"
        )

        # WRMS comparison
        errs = result["errors_us"][:n]
        w = 1.0 / (errs ** 2)
        sw = np.sum(w)

        def wrms(res):
            res_us = res * 1e6
            wm = np.sum(res_us * w) / sw
            return np.sqrt(np.sum(w * (res_us - wm) ** 2) / sw)

        wrms_jug = wrms(jug_res[:n])
        wrms_t2 = wrms(t2_post[:n])
        wrms_diff_ns = (wrms_jug - wrms_t2) * 1e3

        assert abs(wrms_diff_ns) < thresholds["wrms_diff_ns"], (
            f"{pulsar}: |ΔWRMS| = {abs(wrms_diff_ns):.3f} ns > {thresholds['wrms_diff_ns']} ns threshold"
        )
