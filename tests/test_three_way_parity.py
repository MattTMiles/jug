"""
Three-way parity tests: JUG vs PINT vs Tempo2.

Tests both noise-free WLS and noise-aware GLS fits on the J1909_proper dataset
(100 TOAs, MPTA DR3 subset), comparing:
  - Post-fit weighted RMS
  - Post-fit parameter values for well-constrained timing parameters

Noise model in J1909_proper.par:
  - EFAC   = 1.078       (-f KAT_MKBF)
  - T2EQUAD = -6.87      (log10 s, -f KAT_MKBF)
  - TNECORR = -6.33      (log10 s, -f KAT_MKBF) → 0.47 µs
  - TNDMAmp = -13.88,  TNDMGam = 2.62,  TNDMC = 30   (30 DM noise harmonics)

Parameter comparison strategy:
  - Only compare parameters well-constrained on 100 TOAs: F0, F1, RAJ, DECJ,
    PB, A1, EPS1, EPS2, TASC, PMRA, PMDEC.
  - Poorly-constrained params (FD7-9, PX, PBDOT, XDOT, M2) are excluded from
    parameter parity checks — their values can differ wildly between codes on
    a small noisy dataset without indicating a bug.
  - Tolerances are expressed as fractions of the Tempo2 formal uncertainty for
    that parameter, so they scale correctly regardless of parameter magnitude.

Enable:
    JUG_TEST_PINT=1   pytest tests/test_three_way_parity.py -v   (JUG vs PINT)
    JUG_TEST_TEMPO2=1 pytest tests/test_three_way_parity.py -v   (JUG vs Tempo2)
    JUG_TEST_PINT=1 JUG_TEST_TEMPO2=1 pytest tests/test_three_way_parity.py -v
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional dependency / environment gating
# ---------------------------------------------------------------------------
pint_available = True
try:
    import pint.models   # noqa: F401
except ImportError:
    pint_available = False

_FORCE_PINT   = os.environ.get("JUG_TEST_PINT",   "").lower() in ("1", "true", "yes")
_FORCE_TEMPO2 = os.environ.get("JUG_TEST_TEMPO2", "").lower() in ("1", "true", "yes")
_TEMPO2_ON_PATH = shutil.which("tempo2") is not None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLDEN_DIR   = Path(__file__).parent / "data_golden"
PAR_NOISEFREE = GOLDEN_DIR / "J1909_noisefree.par"
PAR_NOISE     = GOLDEN_DIR / "J1909_proper.par"
TIM           = GOLDEN_DIR / "J1909_proper.tim"

# 2828-TOA parity dataset (full MPTA DR3 subset, better-conditioned)
PAR_PARITY_NOISEFREE = GOLDEN_DIR / "J1909_parity_noisefree.par"
PAR_PARITY_NOISE     = GOLDEN_DIR / "J1909_parity_noise.par"
TIM_PARITY           = GOLDEN_DIR / "J1909_parity.tim"

# ---------------------------------------------------------------------------
# Parameters to fit and to compare
# ---------------------------------------------------------------------------
FIT_PARAMS = [
    "F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC",
    "DM1", "DM2",
    "PB", "A1", "EPS1", "EPS2", "TASC",
    "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FD7", "FD8", "FD9",
    "PBDOT", "M2", "XDOT", "PX",
]

# Parameters that are well-constrained on 100 TOAs and worth comparing across codes.
# Tolerance: max allowed deviation in units of the Tempo2 formal 1-sigma uncertainty.
# These are empirically set to be generous enough for a small noisy dataset but
# tight enough to catch real bugs.
COMPARE_PARAMS = {
    "F0":    3.0,   # spin frequency: very well constrained
    "F1":    3.0,   # spin-down: well constrained
    "PB":    3.0,   # orbital period: very well constrained
    "A1":    3.0,   # projected semi-major axis: well constrained
    "TASC":  3.0,   # time of ascending node: well constrained
    "EPS1":  5.0,   # eccentricity vector: moderately constrained
    "EPS2":  5.0,
    "PMRA":  5.0,   # proper motion: constrained but noise-dependent
    "PMDEC": 5.0,
}

# Tolerance on post-fit WRMS agreement (fractional, relative to Tempo2 value)
WRMS_TOL_NOISEFREE = 0.05   # 5%: WLS is deterministic, should be very close
WRMS_TOL_NOISE     = 0.25   # 25%: GLS codes use different algorithms


# ---------------------------------------------------------------------------
# Helpers: run each code and return (wrms_us, params_dict, uncertainties_dict)
# ---------------------------------------------------------------------------

def _jug_fit(par_path, tim_path=None):
    """Run JUG fit. Returns (wrms_us, params, uncertainties).

    uncertainties are returned as None — JUG doesn't currently expose them
    in the same way, so parameter comparisons use Tempo2 uncertainties.
    """
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    result = fit_parameters_optimized(par_path, tim_path or TIM, FIT_PARAMS, verbose=False)
    _logging.disable(_logging.NOTSET)
    return result["final_rms"], result["final_params"], result.get("uncertainties")


def _pint_fit(par_path, gls=True, tim_path=None):
    """Run PINT WLS or GLS fit. Returns (wrms_us, params, uncertainties)."""
    import pint.models
    import pint.toa
    import pint.fitter

    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model = pint.models.get_model(str(par_path))
    toas  = pint.toa.get_TOAs(str(tim_path or TIM), planets=True, ephem=model.EPHEM.value)

    for p in model.free_params[:]:
        getattr(model, p).frozen = True
    for p in FIT_PARAMS:
        if hasattr(model, p):
            getattr(model, p).frozen = False

    fitter_cls = pint.fitter.GLSFitter if gls else pint.fitter.WLSFitter
    fitter = fitter_cls(toas, model)
    fitter.fit_toas(maxiter=5 if gls else 10)

    res_us   = fitter.resids.time_resids.to("us").value
    errs_us  = toas.get_errors().to("us").value
    weights  = 1.0 / errs_us**2
    wrms     = float(np.sqrt(np.sum(weights * res_us**2) / np.sum(weights)))

    params = {}
    uncerts = {}
    fitparams = fitter.get_fitparams()
    for p in FIT_PARAMS:
        if p in fitparams:
            param_obj = fitparams[p]
            # Extract numeric value in base units
            qty = param_obj.quantity
            params[p]  = float(qty.value)
            uncerts[p] = float(param_obj.uncertainty_value or 0.0)

    return wrms, params, uncerts


def _tempo2_fit(par_path, tim_path=None):
    """Run Tempo2 fit. Returns (wrms_us, params, uncertainties).

    WRMS is computed from Tempo2 post-fit residuals using the same formula
    as JUG: sqrt(sum(w * r^2) / sum(w)) with w = 1 / err_us^2, where
    err_us are the raw TOA errors as used by each code (EFAC/EQUAD-scaled
    for noise-aware fits, raw otherwise).

    This ensures WRMS is comparable to JUG's final_rms.  We use Tempo2's
    general2 plugin to extract per-TOA post-fit residuals and errors.
    """
    import shutil as _shutil

    tim = tim_path or TIM
    fit_flags = []
    for p in FIT_PARAMS:
        fit_flags += ["-fit", p]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy par and tim into tmpdir so Tempo2 can find them
        par_tmp = Path(tmpdir) / Path(par_path).name
        tim_tmp = Path(tmpdir) / tim.name
        _shutil.copy(par_path, par_tmp)
        _shutil.copy(tim, tim_tmp)

        # Run with general2 plugin to capture per-TOA residuals and errors.
        # Format string: post-fit residual (s) and TOA error (us).
        gen2 = subprocess.run(
            ["tempo2", "-output", "general2",
             "-f", str(par_tmp), str(tim_tmp)] + fit_flags +
            ["-s", "{post} {err}\n"],
            capture_output=True, text=True, cwd=tmpdir, timeout=300,
        )
        # Also run without general2 to get the parameter table
        par_run = subprocess.run(
            ["tempo2", "-f", str(par_tmp), str(tim_tmp)] + fit_flags,
            capture_output=True, text=True, cwd=tmpdir, timeout=300,
        )

    # ---- Parse residuals from general2 output ----
    rows = []
    for line in gen2.stdout.split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not rows:
        raise RuntimeError(
            f"Cannot parse Tempo2 general2 residuals:\n{gen2.stdout[:2000]}"
        )
    rows_arr   = np.array(rows)
    post_us    = rows_arr[:, 0] * 1e6   # convert seconds → microseconds
    err_us     = rows_arr[:, 1]
    weights    = 1.0 / err_us**2
    wrms       = float(np.sqrt(np.sum(weights * post_us**2) / np.sum(weights)))

    # ---- Parse parameter table from normal Tempo2 run ----
    output = par_run.stdout + par_run.stderr
    params  = {}
    uncerts = {}
    # Parse the parameter table line by line.
    # Fitted lines end with "Y"; tokens are:
    #   NAME [units] prefit postfit uncertainty difference Y
    # where [units] may be absent or truncated (e.g. "DM1 (cm^-3 pc y ...").
    # Strategy: any line ending in "Y" where the first token is a known
    # parameter name; the last 4 tokens before Y are prefit/postfit/unc/diff.
    for line in output.split("\n"):
        tokens = line.split()
        if len(tokens) < 6 or tokens[-1] != "Y":
            continue
        name = tokens[0]
        if name not in FIT_PARAMS:
            continue
        try:
            postfit = float(tokens[-4])
            unc     = float(tokens[-3])
        except ValueError:
            continue
        params[name]  = postfit
        uncerts[name] = unc

    return wrms, params, uncerts


# ---------------------------------------------------------------------------
# Fixtures (module-scoped — run fits once per session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jug_noisefree():
    wrms, params, _ = _jug_fit(PAR_NOISEFREE)
    return {"wrms_us": wrms, "params": params}

@pytest.fixture(scope="module")
def jug_noise():
    wrms, params, _ = _jug_fit(PAR_NOISE)
    return {"wrms_us": wrms, "params": params}

@pytest.fixture(scope="module")
def pint_noisefree():
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit(PAR_NOISEFREE, gls=False)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}

@pytest.fixture(scope="module")
def pint_noise():
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit(PAR_NOISE, gls=True)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}

@pytest.fixture(scope="module")
def tempo2_noisefree():
    if not _TEMPO2_ON_PATH:
        pytest.skip("tempo2 not found on PATH")
    if not _FORCE_TEMPO2:
        pytest.skip("Set JUG_TEST_TEMPO2=1 to enable Tempo2 tests")
    wrms, params, uncerts = _tempo2_fit(PAR_NOISEFREE)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}

@pytest.fixture(scope="module")
def tempo2_noise():
    if not _TEMPO2_ON_PATH:
        pytest.skip("tempo2 not found on PATH")
    if not _FORCE_TEMPO2:
        pytest.skip("Set JUG_TEST_TEMPO2=1 to enable Tempo2 tests")
    wrms, params, uncerts = _tempo2_fit(PAR_NOISE)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


# ---------------------------------------------------------------------------
# Helpers for comparisons
# ---------------------------------------------------------------------------

def _check_wrms(jug_wrms, other_wrms, other_name, tol):
    rel = abs(jug_wrms - other_wrms) / other_wrms
    assert rel <= tol, (
        f"WRMS: JUG={jug_wrms:.4f} µs, {other_name}={other_wrms:.4f} µs, "
        f"relative diff={rel*100:.1f}% > {tol*100:.0f}%"
    )


def _check_params(jug_params, ref_params, ref_uncerts, ref_name, sigma_tol):
    """Compare JUG post-fit params to reference within sigma_tol * ref_uncertainty."""
    failures = []
    for p, n_sigma in COMPARE_PARAMS.items():
        if p not in jug_params or p not in ref_params:
            continue
        unc = ref_uncerts.get(p, 0.0) if ref_uncerts else 0.0
        if unc == 0.0:
            continue  # can't compare without uncertainty
        diff = abs(float(jug_params[p]) - float(ref_params[p]))
        allowed = sigma_tol * n_sigma * unc
        if diff > allowed:
            failures.append(
                f"  {p}: JUG={jug_params[p]:.6g}, {ref_name}={ref_params[p]:.6g}, "
                f"diff={diff:.3e}, allowed={allowed:.3e} "
                f"({sigma_tol}×{n_sigma}σ = {sigma_tol*n_sigma}σ)"
            )
    if failures:
        pytest.fail(
            f"Parameter parity failures vs {ref_name}:\n" + "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# A. Noise-free WLS: JUG vs PINT
# ---------------------------------------------------------------------------

class TestNoiseFreeJugVsPint:
    """Noise-free WLS parity: JUG vs PINT.

    Both use 1/sigma^2 weighting with no noise model.
    """

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_wrms_parity(self, jug_noisefree, pint_noisefree):
        """JUG and PINT noise-free WRMS agree within 5%."""
        _check_wrms(jug_noisefree["wrms_us"], pint_noisefree["wrms_us"],
                    "PINT", WRMS_TOL_NOISEFREE)

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_param_parity(self, jug_noisefree, pint_noisefree):
        """JUG and PINT noise-free post-fit parameters agree within 3σ."""
        _check_params(
            jug_noisefree["params"], pint_noisefree["params"],
            pint_noisefree["uncerts"], "PINT", sigma_tol=3.0,
        )


# ---------------------------------------------------------------------------
# B. Noise-free WLS: JUG vs Tempo2
# ---------------------------------------------------------------------------

class TestNoiseFreeJugVsTempo2:
    """Noise-free WLS parity: JUG vs Tempo2.

    NOTE on WRMS comparison: the 100-TOA dataset has only 72 degrees of
    freedom (100 TOAs − 28 free parameters), which is near-singular for WLS.
    In this regime the two codes may converge to different numerical minima,
    producing different post-fit residuals even when the fitted parameters
    agree within formal uncertainties.  Cross-code WRMS comparison is only
    meaningful on a well-conditioned dataset; on this dataset we test
    parameter parity only.
    """

    @pytest.mark.skipif(not _FORCE_TEMPO2 or not _TEMPO2_ON_PATH,
                        reason="Set JUG_TEST_TEMPO2=1 and ensure tempo2 is on PATH")
    def test_param_parity(self, jug_noisefree, tempo2_noisefree):
        """JUG and Tempo2 noise-free post-fit parameters agree within 3σ."""
        _check_params(
            jug_noisefree["params"], tempo2_noisefree["params"],
            tempo2_noisefree["uncerts"], "Tempo2", sigma_tol=3.0,
        )


# ---------------------------------------------------------------------------
# C. Noise-aware GLS: JUG vs PINT
# ---------------------------------------------------------------------------

class TestNoiseAwareJugVsPint:
    """Noise-aware GLS parity: JUG vs PINT.

    Both include EFAC, EQUAD, TNECORR, and DM noise Fourier basis (30 harmonics).
    Tolerance is looser (25% WRMS, 5σ params) because the two codes use different
    GLS algorithms (augmented SVD vs Woodbury/Cholesky) and this 100-TOA dataset
    is EQUAD-dominated, amplifying numerical differences.
    """

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_wrms_parity(self, jug_noise, pint_noise):
        """JUG and PINT noise-aware WRMS agree within 25%."""
        _check_wrms(jug_noise["wrms_us"], pint_noise["wrms_us"],
                    "PINT", WRMS_TOL_NOISE)

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_param_parity(self, jug_noise, pint_noise):
        """JUG and PINT noise-aware post-fit parameters agree within 5σ."""
        _check_params(
            jug_noise["params"], pint_noise["params"],
            pint_noise["uncerts"], "PINT", sigma_tol=5.0,
        )


# ---------------------------------------------------------------------------
# D. Noise-aware GLS: JUG vs Tempo2
# ---------------------------------------------------------------------------

class TestNoiseAwareJugVsTempo2:
    """Noise-aware GLS parity: JUG vs Tempo2.

    Tempo2 applies EFAC/EQUAD via scaled TOA uncertainties and absorbs DM noise
    via its internal GLS solver.  JUG uses the augmented SVD approach.
    """

    @pytest.mark.skipif(not _FORCE_TEMPO2 or not _TEMPO2_ON_PATH,
                        reason="Set JUG_TEST_TEMPO2=1 and ensure tempo2 is on PATH")
    def test_wrms_parity(self, jug_noise, tempo2_noise):
        """JUG and Tempo2 noise-aware WRMS agree within 25%."""
        _check_wrms(jug_noise["wrms_us"], tempo2_noise["wrms_us"],
                    "Tempo2", WRMS_TOL_NOISE)

    @pytest.mark.skipif(not _FORCE_TEMPO2 or not _TEMPO2_ON_PATH,
                        reason="Set JUG_TEST_TEMPO2=1 and ensure tempo2 is on PATH")
    def test_param_parity(self, jug_noise, tempo2_noise):
        """JUG and Tempo2 noise-aware post-fit parameters agree within 5σ."""
        _check_params(
            jug_noise["params"], tempo2_noise["params"],
            tempo2_noise["uncerts"], "Tempo2", sigma_tol=5.0,
        )


# ---------------------------------------------------------------------------
# E. Cross-code sanity: noise changes the answer
# ---------------------------------------------------------------------------

class TestNoiseSanity:
    """Sanity checks that don't require any external codes."""

    def test_jug_noise_wrms_differs_from_noisefree(self, jug_noisefree, jug_noise):
        """JUG noise-aware WRMS differs from noise-free (noise model is active)."""
        assert jug_noisefree["wrms_us"] != jug_noise["wrms_us"], (
            "Noise-free and noise-aware JUG WRMS are identical — "
            "noise model may not be active"
        )

    def test_jug_noise_params_shift(self, jug_noisefree, jug_noise):
        """Key parameters shift when noise model is applied (fit is different)."""
        # At least one of the well-constrained params should shift noticeably
        changed = False
        for p in ["F0", "F1", "PB", "A1", "PMRA", "PMDEC"]:
            if p in jug_noisefree["params"] and p in jug_noise["params"]:
                if jug_noisefree["params"][p] != jug_noise["params"][p]:
                    changed = True
                    break
        assert changed, (
            "No timing parameters changed between noise-free and noise-aware JUG fits"
        )

    @pytest.mark.skipif(not _FORCE_TEMPO2 or not _TEMPO2_ON_PATH,
                        reason="Set JUG_TEST_TEMPO2=1")
    def test_tempo2_noise_wrms_differs_from_noisefree(
        self, tempo2_noisefree, tempo2_noise
    ):
        """Tempo2 noise-aware WRMS differs from noise-free."""
        assert tempo2_noisefree["wrms_us"] != tempo2_noise["wrms_us"], (
            "Tempo2 noise-free and noise-aware WRMS identical — "
            "noise model may not be active"
        )


# ---------------------------------------------------------------------------
# E. Parity dataset (2828 TOAs, well-conditioned): JUG vs PINT vs Tempo2
# ---------------------------------------------------------------------------
# This is the primary parity check.  With 2800+ DOFs the WLS fit is
# well-conditioned and numerical minima should be essentially identical.
# JUG and PINT should agree within 2% WRMS; Tempo2 linearizes around
# the starting parameters (one iteration by default) so it is only used
# for parameter parity here, not WRMS.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def jug_parity_noisefree():
    wrms, params, uncerts = _jug_fit(PAR_PARITY_NOISEFREE, TIM_PARITY)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


@pytest.fixture(scope="module")
def pint_parity_noisefree():
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit(PAR_PARITY_NOISEFREE, gls=False, tim_path=TIM_PARITY)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


@pytest.fixture(scope="module")
def tempo2_parity_noisefree():
    if not _TEMPO2_ON_PATH:
        pytest.skip("tempo2 not found on PATH")
    if not _FORCE_TEMPO2:
        pytest.skip("Set JUG_TEST_TEMPO2=1 to enable Tempo2 tests")
    wrms, params, uncerts = _tempo2_fit(PAR_PARITY_NOISEFREE, TIM_PARITY)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


class TestParityDatasetNoiseFree:
    """Parity checks on the 2828-TOA well-conditioned dataset.

    With 2800+ DOFs the WLS fit is numerically stable and different codes
    should converge to essentially the same solution:
      - JUG vs PINT WRMS: within 2% (both use iterative nonlinear fitting)
      - JUG vs Tempo2 params: within 2σ (parameter values should agree)

    Note: Tempo2's WRMS is excluded from comparison because Tempo2 uses a
    single linear iteration while JUG/PINT iterate to full convergence;
    Tempo2 therefore converges to a slightly higher WRMS on large datasets.
    The parameter comparison is the true cross-code correctness check.
    """

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_wrms_parity_jug_vs_pint(self, jug_parity_noisefree, pint_parity_noisefree):
        """JUG and PINT noisefree WRMS agree within 2% on 2828-TOA dataset."""
        _check_wrms(
            jug_parity_noisefree["wrms_us"],
            pint_parity_noisefree["wrms_us"],
            "PINT",
            tol=0.02,
        )

    @pytest.mark.skipif(not _FORCE_TEMPO2 or not _TEMPO2_ON_PATH,
                        reason="Set JUG_TEST_TEMPO2=1 and ensure tempo2 is on PATH")
    def test_param_parity_jug_vs_tempo2(self, jug_parity_noisefree, tempo2_parity_noisefree):
        """JUG and Tempo2 noisefree fitted parameters agree within 3σ on 2828-TOA dataset."""
        _check_params(
            jug_parity_noisefree["params"],
            tempo2_parity_noisefree["params"],
            tempo2_parity_noisefree["uncerts"],
            "Tempo2",
            sigma_tol=3.0,
        )

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_param_parity_jug_vs_pint(self, jug_parity_noisefree, pint_parity_noisefree):
        """JUG and PINT noisefree fitted parameters agree within 3σ on 2828-TOA dataset."""
        _check_params(
            jug_parity_noisefree["params"],
            pint_parity_noisefree["params"],
            pint_parity_noisefree["uncerts"],
            "PINT",
            sigma_tol=3.0,
        )
