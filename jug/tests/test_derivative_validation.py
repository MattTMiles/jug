"""
Derivative / design-matrix validation suite.
=============================================

For every fitted parameter in the timing model, this test compares the
**analytic** derivative (used in the WLS design matrix) against a
**finite-difference** derivative of the *same* forward-model function.

The comparison uses **relative error** (max |analytic - FD| / max |analytic|)
plus a correlation check.  This avoids threshold-tuning problems caused by
different parameter magnitudes.

Run with:
    pytest jug/tests/test_derivative_validation.py -v -o "addopts="
"""

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.derivatives_spin import compute_spin_derivatives, taylor_horner
from jug.fitting.derivatives_dm import compute_dm_derivatives
from jug.fitting.derivatives_fd import compute_fd_derivatives, compute_fd_delay
from jug.fitting.derivatives_sw import compute_sw_derivatives
from jug.fitting.derivatives_astrometry import (
    compute_astrometry_derivatives,
    compute_astrometric_delay,
)
from jug.fitting.binary_registry import (
    compute_binary_delay,
    compute_binary_derivatives,
)
from jug.fitting.optimized_fitter import compute_dm_delay_fast
from jug.io.par_reader import parse_par_file, parse_ra, parse_dec

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
JUG_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PULSARS = JUG_ROOT / "data" / "pulsars"

from jug.utils.constants import SECS_PER_DAY, K_DM_SEC

# Thresholds
CORR_THRESHOLD = 0.999_99
REL_ERR_THRESHOLD = 1e-5       # default max relative error
REL_ERR_THRESHOLD_LOOSE = 1e-3  # for params where FD is inherently noisy

# Finite-difference step sizes - chosen so that h is small enough to stay in
# the linear regime yet large enough to avoid catastrophic cancellation.
_STEPS = {
    "F0": 1e-6, "F1": 1e-14,
    "DM": 1e-4, "DM1": 1e-6, "DM2": 1e-6,
    "RAJ": 1e-9, "DECJ": 1e-9,
    "PMRA": 1e-2, "PMDEC": 1e-2, "PX": 1e-2,
    "PB": 1e-8, "A1": 1e-6, "TASC": 1e-6,
    "EPS1": 1e-6, "EPS2": 1e-6,
    "PBDOT": 1e-12, "XDOT": 1e-18,
    "M2": 1e-4, "SINI": 1e-6, "H3": 1e-8,
    "T0": 1e-6, "ECC": 1e-8, "OM": 1e-4,
    "FD1": 1e-8, "FD2": 1e-8, "FD3": 1e-8,
    "FD4": 1e-8, "FD5": 1e-8, "FD6": 1e-8,
    "FD7": 1e-8, "FD8": 1e-8, "FD9": 1e-8,
    "NE_SW": 1e-2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_float(params, key):
    """Return float value of params[key], parsing sexagesimal if needed."""
    v = params[key]
    if isinstance(v, str):
        if key == "RAJ":
            return parse_ra(v)
        elif key == "DECJ":
            return parse_dec(v)
        return float(v)
    return float(v)


def _params_with_floats(params):
    """Return a copy with RAJ/DECJ converted to float (radians)."""
    p = dict(params)
    for k in ("RAJ", "DECJ"):
        if k in p:
            p[k] = _ensure_float(p, k)
    return p


def _rel_err(analytic, fd_deriv):
    """Compute max relative error, avoiding division by zero."""
    a = np.asarray(analytic, dtype=np.float64).ravel()
    f = np.asarray(fd_deriv, dtype=np.float64).ravel()
    denom = np.max(np.abs(a))
    if denom < 1e-30:
        denom = np.max(np.abs(f))
    if denom < 1e-30:
        return 0.0  # both zero
    return float(np.max(np.abs(a - f))) / denom


def _corr(a, b):
    """Pearson correlation, returns 1.0 if both constant."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) == 0 and np.std(b) == 0:
        return 1.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Forward-model wrappers that accept float params and return delay arrays
# ---------------------------------------------------------------------------

def _spin_phase(params, dt_sec):
    """Spin phase (cycles) as function of Fn parameters.

    phase = F0*dt + F1*dt^2/2! + F2*dt^3/3! + ...
          = taylor_horner(dt, [0, F0, F1, F2, ...])

    The leading zero is the constant term (phase offset) which is
    conventionally zero at the reference epoch.
    """
    f_terms = []
    for i in range(10):
        k = f"F{i}"
        if k in params:
            f_terms.append(float(params[k]))
        else:
            break
    # Prepend zero for the constant term so that
    # taylor_horner evaluates  0 + F0*dt/1! + F1*dt^2/2! + ...
    return taylor_horner(dt_sec, [0.0] + f_terms)


def _spin_deriv_fd(params, param_name, step, dt_sec):
    """FD of spin derivative matching the analytic convention.
    
    analytic = -d_phase_d_F / F0
    FD:  (-phase(Fn+h) + phase(Fn-h)) / (2h * F0)
    """
    f0 = float(params["F0"])
    v0 = float(params[param_name])

    p_plus = dict(params)
    p_plus[param_name] = v0 + step
    p_minus = dict(params)
    p_minus[param_name] = v0 - step

    phase_plus = _spin_phase(p_plus, dt_sec)
    phase_minus = _spin_phase(p_minus, dt_sec)

    # d(delay)/d(Fn) = -d(phase)/d(Fn) / F0
    return -(phase_plus - phase_minus) / (2 * step * f0)


def _dm_delay_seconds(params, toas_mjd, freq_mhz):
    dm_epoch = float(params.get("DMEPOCH", params.get("PEPOCH", 55000.0)))
    dm_pars = {}
    for k in ("DM", "DM1", "DM2"):
        if k in params:
            dm_pars[k] = float(params[k])
    return compute_dm_delay_fast(toas_mjd, freq_mhz, dm_pars, dm_epoch)


def _fd_delay_seconds(params, freq_mhz):
    fd_pars = {k: float(v) for k, v in params.items() if k.startswith("FD")}
    if not fd_pars:
        return np.zeros_like(freq_mhz)
    return compute_fd_delay(freq_mhz, fd_pars)


def _sw_delay_seconds(params, freq_mhz, sw_geometry_pc):
    ne_sw = float(params.get("NE_SW", params.get("NE1AU", 0.0)))
    return K_DM_SEC * ne_sw * sw_geometry_pc / (freq_mhz ** 2)


def _binary_delay_seconds(params, toas_bary):
    return np.asarray(compute_binary_delay(toas_bary, params))


def _astro_delay_seconds(params, toas_mjd, ssb):
    return np.asarray(compute_astrometric_delay(params, toas_mjd, ssb))


# ---------------------------------------------------------------------------
# Central finite-difference
# ---------------------------------------------------------------------------

def _fd_derivative(fwd_fn, params, param_name, step, **kwargs):
    """Central FD of fwd_fn w.r.t. param_name."""
    v0 = float(params[param_name])
    p_plus = dict(params)
    p_plus[param_name] = v0 + step
    p_minus = dict(params)
    p_minus[param_name] = v0 - step
    return (fwd_fn(p_plus, **kwargs) - fwd_fn(p_minus, **kwargs)) / (2 * step)


# ---------------------------------------------------------------------------
# Pulsar test data
# ---------------------------------------------------------------------------

class PulsarTestData:
    def __init__(self, name):
        par = DATA_PULSARS / f"{name}_tdb.par"
        tim = DATA_PULSARS / f"{name}.tim"
        if not par.exists() or not tim.exists():
            pytest.skip(f"{name} data not found")

        self.params = _params_with_floats(parse_par_file(par))
        result = compute_residuals_simple(par, tim, verbose=False)

        self.toas_mjd = result["tdb_mjd"]
        self.freq_mhz = result["freq_bary_mhz"]
        self.ssb_obs_pos = result["ssb_obs_pos_ls"]
        self.prebinary = result.get("prebinary_delay_sec")
        self.sw_geom = result.get("sw_geometry_pc")

        pepoch = float(self.params.get("PEPOCH", self.toas_mjd[0]))
        self.dt_sec = (self.toas_mjd - pepoch) * SECS_PER_DAY

        if self.prebinary is not None:
            self.toas_bary = self.toas_mjd - self.prebinary / SECS_PER_DAY
        else:
            self.toas_bary = self.toas_mjd


# ---------------------------------------------------------------------------
# J1909-3744 (ELL1)
# ---------------------------------------------------------------------------

class TestDerivJ1909:
    """Derivative validation for J1909-3744 (ELL1 binary)."""

    @pytest.fixture(scope="class")
    def d(self):
        return PulsarTestData("J1909-3744")

    # -- Spin --
    @pytest.mark.parametrize("p", ["F0", "F1"])
    def test_spin(self, d, p):
        analytic = compute_spin_derivatives(d.params, d.toas_mjd, [p])[p]
        fd = _spin_deriv_fd(d.params, p, _STEPS[p], dt_sec=d.dt_sec)
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    # -- DM --
    @pytest.mark.parametrize("p", ["DM1", "DM2"])
    def test_dm(self, d, p):
        if p not in d.params:
            pytest.skip(f"{p} missing")
        analytic = compute_dm_derivatives(
            d.params, d.toas_mjd, d.freq_mhz, [p]
        )[p]
        fd = _fd_derivative(
            _dm_delay_seconds, d.params, p, _STEPS[p],
            toas_mjd=d.toas_mjd, freq_mhz=d.freq_mhz,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    # -- Astrometry --
    @pytest.mark.parametrize("p", ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"])
    def test_astro(self, d, p):
        analytic = compute_astrometry_derivatives(
            d.params, d.toas_mjd, d.ssb_obs_pos, [p]
        )[p]
        fd = _fd_derivative(
            _astro_delay_seconds, d.params, p, _STEPS[p],
            toas_mjd=d.toas_mjd, ssb=d.ssb_obs_pos,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    # -- Binary (ELL1) --
    @pytest.mark.parametrize("p", [
        "PB", "A1", "TASC", "EPS1", "EPS2", "PBDOT", "XDOT", "SINI", "M2",
    ])
    def test_binary(self, d, p):
        if p not in d.params:
            pytest.skip(f"{p} missing")
        analytic = compute_binary_derivatives(
            d.params, d.toas_bary, [p]
        )[p]
        fd = _fd_derivative(
            _binary_delay_seconds, d.params, p, _STEPS[p],
            toas_bary=d.toas_bary,
        )
        thresh = REL_ERR_THRESHOLD_LOOSE if p in ("PBDOT", "XDOT") else REL_ERR_THRESHOLD
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < thresh, f"{p} rel_err"

    # -- FD --
    @pytest.mark.parametrize("p", [
        "FD1", "FD2", "FD3", "FD4", "FD5", "FD6", "FD7", "FD8", "FD9",
    ])
    def test_fd(self, d, p):
        if p not in d.params:
            pytest.skip(f"{p} missing")
        analytic = compute_fd_derivatives(d.params, d.freq_mhz, [p])[p]
        fd = _fd_derivative(
            _fd_delay_seconds, d.params, p, _STEPS[p], freq_mhz=d.freq_mhz,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    # -- SW --
    def test_ne_sw(self, d):
        if d.sw_geom is None:
            pytest.skip("No SW geometry")
        analytic = compute_sw_derivatives(
            d.sw_geom, d.freq_mhz, ["NE_SW"]
        )["NE_SW"]
        fd = _fd_derivative(
            _sw_delay_seconds, d.params, "NE_SW", _STEPS["NE_SW"],
            freq_mhz=d.freq_mhz, sw_geometry_pc=d.sw_geom,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, "NE_SW corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, "NE_SW rel_err"


# ---------------------------------------------------------------------------
# J0614-3329 (DD binary)
# ---------------------------------------------------------------------------

class TestDerivJ0614:
    """Derivative validation for J0614-3329 (DD binary)."""

    @pytest.fixture(scope="class")
    def d(self):
        return PulsarTestData("J0614-3329")

    @pytest.mark.parametrize("p", ["F0", "F1"])
    def test_spin(self, d, p):
        analytic = compute_spin_derivatives(d.params, d.toas_mjd, [p])[p]
        fd = _spin_deriv_fd(d.params, p, _STEPS[p], dt_sec=d.dt_sec)
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    @pytest.mark.parametrize("p", ["DM", "DM1", "DM2"])
    def test_dm(self, d, p):
        if p not in d.params:
            pytest.skip(f"{p} missing")
        analytic = compute_dm_derivatives(
            d.params, d.toas_mjd, d.freq_mhz, [p]
        )[p]
        fd = _fd_derivative(
            _dm_delay_seconds, d.params, p, _STEPS[p],
            toas_mjd=d.toas_mjd, freq_mhz=d.freq_mhz,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    @pytest.mark.parametrize("p", ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"])
    def test_astro(self, d, p):
        analytic = compute_astrometry_derivatives(
            d.params, d.toas_mjd, d.ssb_obs_pos, [p]
        )[p]
        fd = _fd_derivative(
            _astro_delay_seconds, d.params, p, _STEPS[p],
            toas_mjd=d.toas_mjd, ssb=d.ssb_obs_pos,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, f"{p} rel_err"

    @pytest.mark.parametrize("p", [
        "PB", "A1", "T0", "ECC", "OM", "SINI", "M2", "PBDOT",
    ])
    def test_binary_dd(self, d, p):
        if p not in d.params:
            pytest.skip(f"{p} missing")
        analytic = compute_binary_derivatives(
            d.params, d.toas_bary, [p]
        )[p]
        fd = _fd_derivative(
            _binary_delay_seconds, d.params, p, _STEPS[p],
            toas_bary=d.toas_bary,
        )
        thresh = REL_ERR_THRESHOLD_LOOSE if p in ("PBDOT",) else REL_ERR_THRESHOLD
        assert _corr(analytic, fd) > CORR_THRESHOLD, f"{p} corr"
        assert _rel_err(analytic, fd) < thresh, f"{p} rel_err"

    def test_fd1(self, d):
        if "FD1" not in d.params:
            pytest.skip("FD1 missing")
        analytic = compute_fd_derivatives(d.params, d.freq_mhz, ["FD1"])["FD1"]
        fd = _fd_derivative(
            _fd_delay_seconds, d.params, "FD1", _STEPS["FD1"],
            freq_mhz=d.freq_mhz,
        )
        assert _corr(analytic, fd) > CORR_THRESHOLD, "FD1 corr"
        assert _rel_err(analytic, fd) < REL_ERR_THRESHOLD, "FD1 rel_err"
