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
import importlib.util
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
_TEMPO2_ON_PATH = importlib.util.find_spec("libstempo") is not None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLDEN_DIR   = Path(__file__).parent / "data_golden"
PAR_NOISEFREE = GOLDEN_DIR / "J1909_noisefree.par"
PAR_NOISE     = GOLDEN_DIR / "J1909_proper.par"
TIM           = GOLDEN_DIR / "J1909_proper.tim"

# J0437-4715 paths (converted TDB par in golden dir; TIM from PPTA DR4 MTM dataset)
PAR_J0437 = GOLDEN_DIR / "J0437_tdb.par"
PAR_J0437_NOISEFREE = GOLDEN_DIR / "J0437_tdb_noisefree.par"
_J0437_TIM_CANDIDATES = [
    Path(__file__).parent / "data_mpta" / "j0437" / "J0437-4715.tim",
    Path("/home/mattm/soft/JUG/data/pulsars/PPTA_data/ppta_dr4-data_dev-data-partim-MTM/data/partim/MTM/J0437-4715.tim"),
]
TIM_J0437 = next((p for p in _J0437_TIM_CANDIDATES if p.exists()), None)

# Local clock directory shipped with PPTA DR4 dataset
_J0437_CLK_CANDIDATES = [
    Path(__file__).parent.parent / "data" / "clock",
    Path("/home/mattm/soft/JUG/data/pulsars/PPTA_data/ppta_dr4-data_dev-data-partim-MTM/data/partim/clock"),
]
CLK_J0437 = next((p for p in _J0437_CLK_CANDIDATES if p.exists()), None)

# 2828-TOA parity dataset (full MPTA DR3 subset, better-conditioned)
PAR_PARITY_NOISEFREE = GOLDEN_DIR / "J1909_parity_noisefree.par"
PAR_PARITY_NOISE     = GOLDEN_DIR / "J1909_parity_noise.par"
TIM_PARITY           = GOLDEN_DIR / "J1909_parity.tim"

# ---------------------------------------------------------------------------
# Parameters to fit and to compare
# ---------------------------------------------------------------------------
# Parameters to fit on the 100-TOA dataset (SINI excluded: too poorly constrained
# on 100 TOAs and can push PINT's SINI > 1, causing an error).
FIT_PARAMS = [
    "F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC",
    "DM1", "DM2",
    "PB", "A1", "EPS1", "EPS2", "TASC",
    "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FD7", "FD8", "FD9",
    "PBDOT", "M2", "XDOT", "PX",
]

# Parameters to fit on the 2828-TOA parity dataset (fully constrained; includes SINI).
FIT_PARAMS_PARITY = [
    "F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC",
    "DM1", "DM2",
    "PB", "A1", "EPS1", "EPS2", "TASC",
    "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FD7", "FD8", "FD9",
    "PBDOT", "M2", "SINI", "XDOT", "PX",
]

# PINT uses different internal names for some parameters.
# This mapping translates JUG/Tempo2 par-file names to PINT attribute names.
_PINT_PARAM_ALIASES = {
    "XDOT": "A1DOT",   # XDOT in par file = A1DOT in PINT
}

# Parameters to compare across codes, with per-parameter sigma tolerances.
# Tolerance: max allowed deviation in units of the reference code's formal 1-sigma
# uncertainty, multiplied by the sigma_tol passed to _check_params.
# For the 100-TOA dataset these are looser (few-sigma) because the dataset is
# small and noisy. For the 2828-TOA parity dataset use sigma_tol=1 (tight).
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

# Tighter set for the 2828-TOA parity dataset: all fitted params, 1σ tolerance.
COMPARE_PARAMS_PARITY = {
    "F0":    1.0,
    "F1":    1.0,
    "PB":    1.0,
    "A1":    1.0,
    "TASC":  1.0,
    "EPS1":  1.0,
    "EPS2":  1.0,
    "PMRA":  1.0,
    "PMDEC": 1.0,
    "SINI":  1.0,
    "XDOT":  1.0,
    "PBDOT": 1.0,
    "M2":    1.0,
    "PX":    1.0,
}

# Tempo2 does only a single linearised step, so it does not converge to the
# same minimum as JUG/PINT for weakly-constrained parameters.  We therefore
# only compare the well-constrained (bright-line) parameters against Tempo2.
COMPARE_PARAMS_PARITY_TEMPO2 = {
    "F0":    1.0,
    "F1":    1.0,
    "PMRA":  1.0,
    "PMDEC": 1.0,
}

# Tolerance on post-fit WRMS agreement (fractional, relative to Tempo2 value)
WRMS_TOL_NOISEFREE = 0.05   # 5%: WLS is deterministic, should be very close
WRMS_TOL_NOISE     = 0.25   # 25%: GLS codes use different algorithms


# ---------------------------------------------------------------------------
# Helpers: run each code and return (wrms_us, params_dict, uncertainties_dict)
# ---------------------------------------------------------------------------

def _jug_fit(par_path, tim_path=None, fit_params=None):
    """Run JUG fit. Returns (wrms_us, params, uncertainties).

    uncertainties are returned as None — JUG doesn't currently expose them
    in the same way, so parameter comparisons use Tempo2 uncertainties.
    """
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    result = fit_parameters_optimized(par_path, tim_path or TIM, fit_params or FIT_PARAMS, verbose=False)
    _logging.disable(_logging.NOTSET)
    return result["final_rms"], result["final_params"], result.get("uncertainties")


def _pint_fit(par_path, gls=True, tim_path=None, fit_params=None):
    """Run PINT WLS or GLS fit. Returns (wrms_us, params, uncertainties)."""
    import pint.models
    import pint.toa
    import pint.fitter

    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    _fp = fit_params or FIT_PARAMS
    model = pint.models.get_model(str(par_path))
    bipm_ver = model.CLOCK.value.replace("TT(", "").replace(")", "") if hasattr(model, "CLOCK") else "BIPM2023"
    toas  = pint.toa.get_TOAs(str(tim_path or TIM), planets=True, ephem=model.EPHEM.value,
                               bipm_version=bipm_ver)

    for p in model.free_params[:]:
        getattr(model, p).frozen = True
    for p in _fp:
        pint_name = _PINT_PARAM_ALIASES.get(p, p)
        if hasattr(model, pint_name):
            getattr(model, pint_name).frozen = False

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
    # Build reverse alias map: PINT name -> par-file name
    _pint_to_jug = {v: k for k, v in _PINT_PARAM_ALIASES.items()}
    for p in _fp:
        pint_name = _PINT_PARAM_ALIASES.get(p, p)
        if pint_name in fitparams:
            param_obj = fitparams[pint_name]
            # Extract numeric value in base units
            qty = param_obj.quantity
            params[p]  = float(qty.value)
            uncerts[p] = float(param_obj.uncertainty_value or 0.0)

    return wrms, params, uncerts


def _tempo2_fit(par_path, tim_path=None):
    """Tempo2 oracle fits are not available in the pint-only portable build."""
    del par_path, tim_path
    pytest.skip("tempo2 reference harness not available in pint-only build")


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


def _check_params(jug_params, ref_params, ref_uncerts, ref_name, sigma_tol,
                  compare_dict=None):
    """Compare JUG post-fit params to reference within sigma_tol * ref_uncertainty.

    Parameters
    ----------
    compare_dict : dict, optional
        Maps parameter name → per-parameter n_sigma multiplier.
        Defaults to COMPARE_PARAMS if not provided.
    """
    if compare_dict is None:
        compare_dict = COMPARE_PARAMS
    failures = []
    for p, n_sigma in compare_dict.items():
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
    @pytest.mark.xfail(
        reason=(
            "JUG reports noise-subtracted WRMS; PINT reports raw post-fit WRMS. "
            "These definitions differ fundamentally and are not directly comparable."
        ),
        strict=False,
    )
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
    @pytest.mark.xfail(
        reason=(
            "JUG reports noise-subtracted WRMS; Tempo2 reports raw post-fit WRMS. "
            "These definitions differ fundamentally and are not directly comparable."
        ),
        strict=False,
    )
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
    wrms, params, uncerts = _jug_fit(PAR_PARITY_NOISEFREE, TIM_PARITY, fit_params=FIT_PARAMS_PARITY)
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


@pytest.fixture(scope="module")
def pint_parity_noisefree():
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit(PAR_PARITY_NOISEFREE, gls=False, tim_path=TIM_PARITY,
                                      fit_params=FIT_PARAMS_PARITY)
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
        """JUG and Tempo2 noisefree fitted parameters agree within 2σ on 2828-TOA dataset."""
        _check_params(
            jug_parity_noisefree["params"],
            tempo2_parity_noisefree["params"],
            tempo2_parity_noisefree["uncerts"],
            "Tempo2",
            sigma_tol=2.0,
            compare_dict=COMPARE_PARAMS_PARITY_TEMPO2,
        )

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    def test_param_parity_jug_vs_pint(self, jug_parity_noisefree, pint_parity_noisefree):
        """JUG and PINT noisefree fitted parameters agree within 1σ on 2828-TOA dataset."""
        _check_params(
            jug_parity_noisefree["params"],
            pint_parity_noisefree["params"],
            pint_parity_noisefree["uncerts"],
            "PINT",
            sigma_tol=1.0,
            compare_dict=COMPARE_PARAMS_PARITY,
        )


# ---------------------------------------------------------------------------
# J0437-4715 PPTA DR4 MTM: noise-aware GLS parity (JUG vs PINT)
# ---------------------------------------------------------------------------
#
# J0437-4715 is a stringent test because it has:
#   - 14783 TOAs over ~17 yr
#   - Complex JUMP structure (~40 JUMPs, many -group / -h / -j flags)
#   - DDK binary model (Kopeikin kinematic aberration)
#   - Three noise processes: red noise, DM noise, chromatic noise (idx=8),
#     each with 160 harmonics → 960 noise basis columns
#   - Parkes observatory (clock chain: pks2gps.clk + gps2gpst.clk + gpst2utc.clk)
#
# The par file is the PPTA DR4 TCB+BINARY T2 par converted to TDB+DDK via
# `tcb2tdb --allow_T2`, saved at tests/data_golden/J0437_tdb.par.
# The TIM file is the original PPTA DR4 MTM dataset.
#
# Tolerance: JUG and PINT GLS fitted F0, F1 must agree within 3σ of the
# larger of the two formal uncertainties.  RAJ/DECJ are excluded because PINT
# returns them in different units (radians vs hour/deg strings).

_J0437_FIT_PARAMS = ["F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC", "PX"]
_J0437_COMPARE_PARAMS = {"F0": 3.0, "F1": 3.0}

_j0437_available = PAR_J0437.exists() and TIM_J0437 is not None

# Alias map for J0437 PINT parameter names
_J0437_PINT_ALIASES = {}


def _jug_fit_j0437():
    """Run JUG GLS fit on J0437-4715. Returns (wrms_us, params, uncertainties)."""
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    result = fit_parameters_optimized(
        PAR_J0437, TIM_J0437,
        fit_params=_J0437_FIT_PARAMS,
        verbose=False,
    )
    _logging.disable(_logging.NOTSET)
    return result["final_rms"], result["final_params"], result.get("uncertainties")


def _pint_fit_j0437():
    """Run PINT GLS fit on J0437-4715. Returns (wrms_us, params, uncertainties).

    Notes
    -----
    - ``find_empty_masks`` is called before fitting to freeze JUMPs that have
      no matching TOAs (the DR4 par has legacy backends not in MTM subset).
    - Fitted params read from ``get_fitparams()`` (not ``m.F1.value`` which
      retains the par-file value in longdouble form).
    """
    import pint.models
    import pint.toa
    import pint.fitter

    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model = pint.models.get_model(str(PAR_J0437))
    bipm_ver = (model.CLOCK.value.replace("TT(", "").replace(")", "")
                if hasattr(model, "CLOCK") else "BIPM2023")
    toas = pint.toa.get_TOAs(
        str(TIM_J0437), planets=True,
        ephem=model.EPHEM.value, bipm_version=bipm_ver,
    )

    # Freeze JUMPs whose flag values are absent from the TOA set
    model.find_empty_masks(toas, freeze=True)

    # Set exactly the params we want to fit
    for p in model.free_params[:]:
        getattr(model, p).frozen = True
    for p in _J0437_FIT_PARAMS:
        pint_name = _J0437_PINT_ALIASES.get(p, p)
        if hasattr(model, pint_name):
            getattr(model, pint_name).frozen = False

    fitter = pint.fitter.GLSFitter(toas, model)
    fitter.fit_toas(maxiter=5)

    res_us  = fitter.resids.time_resids.to("us").value
    errs_us = toas.get_errors().to("us").value
    weights = 1.0 / errs_us ** 2
    wrms    = float(np.sqrt(np.sum(weights * res_us ** 2) / np.sum(weights)))

    params  = {}
    uncerts = {}
    fitparams = fitter.get_fitparams()
    for p in _J0437_FIT_PARAMS:
        pint_name = _J0437_PINT_ALIASES.get(p, p)
        if pint_name in fitparams:
            obj = fitparams[pint_name]
            params[p]  = float(obj.quantity.value)
            uncerts[p] = float(obj.uncertainty_value or 0.0)

    return wrms, params, uncerts


@pytest.fixture(scope="module")
def jug_j0437():
    if not _j0437_available:
        pytest.skip("J0437 data not available")
    wrms, params, uncerts = _jug_fit_j0437()
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


@pytest.fixture(scope="module")
def pint_j0437():
    if not _j0437_available:
        pytest.skip("J0437 data not available")
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit_j0437()
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


class TestJ0437ParityNoiseAware:
    """JUG vs PINT GLS parity on J0437-4715 PPTA DR4 MTM (14783 TOAs).

    Both codes use:
      - TDB+DDK par converted from original TCB+BINARY T2 par
      - Parkes clock chain: pks2gps.clk → gps2gpst.clk → gpst2utc.clk
      - Three power-law noise processes (red, DM, chromatic) × 160 harmonics

    Tolerance: spin parameters (F0, F1) must agree within 3σ of PINT's
    formal GLS uncertainty.  WRMS is not compared because JUG reports
    noise-subtracted residuals while PINT reports raw post-fit residuals.
    """

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    @pytest.mark.skipif(not _j0437_available,
                        reason="J0437 PPTA DR4 TIM not available at expected path")
    def test_param_parity_gls(self, jug_j0437, pint_j0437):
        """JUG and PINT GLS fitted F0, F1 agree within 3σ on J0437-4715."""
        _check_params(
            jug_j0437["params"],
            pint_j0437["params"],
            pint_j0437["uncerts"],
            "PINT",
            sigma_tol=3.0,
            compare_dict=_J0437_COMPARE_PARAMS,
        )


# ---------------------------------------------------------------------------
# J0437-4715 noise-free WLS parity (43-parameter DDK fit)
# ---------------------------------------------------------------------------
#
# Full 43-parameter WLS fit on J0437-4715 PPTA DR4 MTM dataset (14783 TOAs).
# Uses local PPTA DR4 clock files (pks2gps, gps2utc, tai2tt_bipm2024).
# PINT requires find_empty_masks() to freeze 14 JUMPs with no matching TOAs.
#
# Near-circular orbit (ECC ~ 2e-5) causes exact degeneracy:
#   corr(T0, OM) = +1.000  and  corr(PB, OMDOT) = +1.000
# All three codes (JUG, PINT, Tempo2) land on different points of this
# degenerate ridge with WRMS differences < 0.1 ns.
#
# Test strategy:
#   - WRMS: JUG vs PINT within 0.5 µs absolute (< 2 ns in practice)
#   - Non-degenerate params: KIN, A1, ECC, PBDOT, FD1-FD6 within 5σ
#   - Degenerate params (T0, OM, PB, OMDOT, KOM, M2, PMRA, PMDEC, PX)
#     NOT tested for sigma-parity (ridge degeneracy, not a bug)

_J0437_WLS_FIT_PARAMS = [
    "PX", "RAJ", "DECJ", "PMRA", "PMDEC", "F0", "F1", "DM", "DM1", "DM2",
    "PB", "PBDOT", "A1", "ECC", "T0", "OM", "OMDOT", "M2", "KIN", "KOM",
    "FD1", "FD2", "FD3", "FD4", "FD5", "FD6",
    "JUMP3", "JUMP5", "JUMP6", "JUMP7", "JUMP8", "JUMP9", "JUMP10",
    "JUMP49", "JUMP50", "JUMP51", "JUMP52", "JUMP53",
    "JUMP55", "JUMP56", "JUMP57", "JUMP58", "JUMP59",
]

# Only non-degenerate params tested for sigma-parity
_J0437_WLS_COMPARE_PARAMS = {
    "KIN":   5.0,
    "A1":    5.0,
    "ECC":   5.0,
    "PBDOT": 5.0,
    "FD1":   5.0,
    "FD2":   5.0,
    "FD3":   5.0,
    "FD4":   5.0,
    "FD5":   5.0,
    "FD6":   5.0,
}

_j0437_wls_available = PAR_J0437_NOISEFREE.exists() and TIM_J0437 is not None and CLK_J0437 is not None


def _jug_fit_j0437_wls():
    """Run JUG WLS fit on J0437-4715 (43 params). Returns (wrms_us, params, uncertainties)."""
    from jug.fitting.optimized_fitter import fit_parameters_optimized
    import logging as _logging
    _logging.disable(_logging.CRITICAL)
    result = fit_parameters_optimized(
        PAR_J0437_NOISEFREE, TIM_J0437,
        fit_params=_J0437_WLS_FIT_PARAMS,
        clock_dir=str(CLK_J0437),
        verbose=False,
    )
    _logging.disable(_logging.NOTSET)
    return result["final_rms"], result["final_params"], result.get("uncertainties")


def _pint_fit_j0437_wls():
    """Run PINT WLS fit on J0437-4715 (43 params). Returns (wrms_us, params, uncertainties)."""
    import pint.models
    import pint.toa
    import pint.fitter

    if CLK_J0437 is not None:
        os.environ["PINT_CLOCK_OVERRIDE"] = str(CLK_J0437)

    logging.getLogger("pint").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model = pint.models.get_model(str(PAR_J0437_NOISEFREE))
    toas = pint.toa.get_TOAs(str(TIM_J0437), model=model)

    # Freeze JUMPs with no matching TOAs (14 in DR4 par vs MTM subset)
    model.find_empty_masks(toas, freeze=True)

    # Set exactly the params we want to fit (XDOT alias not needed for J0437)
    for p in model.free_params[:]:
        getattr(model, p).frozen = True
    for p in _J0437_WLS_FIT_PARAMS:
        if hasattr(model, p):
            getattr(model, p).frozen = False

    fitter = pint.fitter.WLSFitter(toas, model)
    fitter.fit_toas(maxiter=20)

    res_us  = fitter.resids.time_resids.to("us").value
    errs_us = toas.get_errors().to("us").value
    weights = 1.0 / errs_us ** 2
    wrms    = float(np.sqrt(np.sum(weights * res_us ** 2) / np.sum(weights)))

    params  = {}
    uncerts = {}
    fitparams = fitter.get_fitparams()
    for p in _J0437_WLS_FIT_PARAMS:
        if p in fitparams:
            obj = fitparams[p]
            params[p]  = float(obj.quantity.value)
            uncerts[p] = float(obj.uncertainty_value or 0.0)

    return wrms, params, uncerts


@pytest.fixture(scope="module")
def jug_j0437_wls():
    if not _j0437_wls_available:
        pytest.skip("J0437 noisefree data not available")
    wrms, params, uncerts = _jug_fit_j0437_wls()
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


@pytest.fixture(scope="module")
def pint_j0437_wls():
    if not _j0437_wls_available:
        pytest.skip("J0437 noisefree data not available")
    if not pint_available:
        pytest.skip("PINT not installed")
    if not _FORCE_PINT:
        pytest.skip("Set JUG_TEST_PINT=1 to enable PINT tests")
    wrms, params, uncerts = _pint_fit_j0437_wls()
    return {"wrms_us": wrms, "params": params, "uncerts": uncerts}


class TestJ0437WLSNoiseFree:
    """JUG vs PINT WLS parity on J0437-4715 (43-param DDK, 14783 TOAs).

    Near-circular orbit (ECC~2e-5) creates exact degeneracy between T0/OM
    and PB/OMDOT. All codes land on different ridge points; WRMS difference
    is < 0.1 ns. Only non-degenerate params (KIN, A1, ECC, PBDOT, FD1-6)
    are tested for sigma-parity.
    """

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    @pytest.mark.skipif(not _j0437_wls_available,
                        reason="J0437 noisefree par or TIM not available")
    def test_wrms_parity(self, jug_j0437_wls, pint_j0437_wls):
        """JUG and PINT WLS WRMS agree within 0.5 µs absolute on J0437-4715."""
        diff = abs(jug_j0437_wls["wrms_us"] - pint_j0437_wls["wrms_us"])
        assert diff < 0.5, (
            f"WRMS: JUG={jug_j0437_wls['wrms_us']:.4f} µs, "
            f"PINT={pint_j0437_wls['wrms_us']:.4f} µs, diff={diff:.4f} µs"
        )

    @pytest.mark.skipif(not _FORCE_PINT or not pint_available,
                        reason="Set JUG_TEST_PINT=1 and install PINT")
    @pytest.mark.skipif(not _j0437_wls_available,
                        reason="J0437 noisefree par or TIM not available")
    def test_non_degenerate_param_parity(self, jug_j0437_wls, pint_j0437_wls):
        """JUG and PINT non-degenerate params (KIN, A1, ECC, PBDOT, FD1-6) agree within 5σ."""
        _check_params(
            jug_j0437_wls["params"],
            pint_j0437_wls["params"],
            pint_j0437_wls["uncerts"],
            "PINT",
            sigma_tol=1.0,
            compare_dict=_J0437_WLS_COMPARE_PARAMS,
        )
