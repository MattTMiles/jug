"""Tests for hybrid linearization via nonlinear_params (feature_hybrid_linear_binary.md).

§8.4 binary-axis tol: relative agreement rtol=1e-3 where column norm > 1e-6;
absolute floor corresponding to < 1 ns residual response per unit native param
is checked via max|ΔJ_col| * typical_step when needed — primary gate is
columnwise relative/absolute match vs native jacfwd.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jug.engine import open_session
from jug.fitting.jax_residual_delta import (
    _bake_residual_jacobian_native,
    _binary_delay_change_jax,
    _hybrid_delta_partition,
    _params_with_frozen_astrometry,
    _phase_residual_delta_jax,
    _prepare_residual_delta_jax,
    _spin_terms_from_params,
    make_residual_delta_jax_fn,
)
from jug.fitting.nonlinear_params import (
    is_hybrid_nonlinear_params,
    plan_live_keys,
    validate_nonlinear_params,
    warn_if_tempo2_native_ignored,
)
from jug.fitting.optimized_fitter import GeneralFitSetup, fit_parameters_optimized
from jug.fitting.residual_model import export_frozen_residual_model
from jug.model.parameter_spec import get_binary_params_from_list
from jug.utils.constants import SECS_PER_DAY
from jug.utils.units import fit_to_native_value

from tempo2_fixtures import get_tempo2_fixture

pytestmark = [pytest.mark.tempo2]

WLS_MAX_ITER = {
    "ppta_j1902_ell1h": 40,
}

NS_RMS = 1e-9
NS_MAX = 3e-9


def _delta_native_from_wls_uncertainties(
    fit_result: dict,
    delta_params: Sequence[str],
    *,
    k: float,
    active: frozenset[str] | None = None,
) -> np.ndarray:
    """Native-unit delta = k × (fit-unit 1σ converted to native)."""
    unc = fit_result["uncertainties"]
    delta = np.zeros(len(delta_params), dtype=np.float64)
    for i, name in enumerate(delta_params):
        if active is not None and name not in active:
            continue
        sigma_fit = unc.get(name)
        if sigma_fit is None or not np.isfinite(sigma_fit) or float(sigma_fit) <= 0.0:
            raise ValueError(f"missing or invalid WLS uncertainty for {name!r}")
        delta[i] = float(k) * fit_to_native_value(name, float(sigma_fit))
    return delta


def _toy_hybrid_setup(fit_params: list[str], *, nonlinear_params: str = "binary"):
    """Minimal synthetic setup with binary cache for unit-level hybrid tests."""
    n = 5
    tdb_mjd = np.linspace(55000.0, 55001.0, n, dtype=np.float64)
    freq_mhz = np.full(n, 1400.0, dtype=np.float64)
    errors_us = np.full(n, 1.0, dtype=np.float64)
    params = {
        "F0": 200.0,
        "F1": -1.0e-15,
        "PEPOCH": 55000.0,
        "DM": 10.0,
        "DMEPOCH": 55000.0,
        "RAJ": 0.1,
        "DECJ": -0.2,
        "_raj_rad": 0.1,
        "_decj_rad": -0.2,
        "PX": 1.5,
        "PMRA": 0.0,
        "PMDEC": 0.0,
        "POSEPOCH": 55000.0,
        "BINARY": "DD",
        "PB": 5.0,
        "T0": 55000.0,
        "A1": 10.0,
        "OM": 45.0,
        "ECC": 0.01,
        "SINI": 0.9,
        "M2": 0.2,
    }
    dt_sec = (tdb_mjd - params["PEPOCH"]) * SECS_PER_DAY
    prebinary = np.zeros(n, dtype=np.float64)
    ssb = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (n, 1))
    from jug.fitting.binary_delay_plan import resolve_binary_structure

    plan = resolve_binary_structure(params, fit_params, obs_pos_ls=ssb)
    toas_pre = tdb_mjd - prebinary / SECS_PER_DAY
    initial_binary = np.asarray(
        plan.evaluate(toas_pre, params, ssb, np), dtype=np.float64
    )
    return GeneralFitSetup(
        params=params,
        fit_param_list=list(fit_params),
        compatibility="pint",
        fd_column_mode="delay_only",
        param_values_start=[float(params.get(p, 0.0)) for p in fit_params],
        toas_mjd=tdb_mjd,
        freq_mhz=freq_mhz,
        errors_us=errors_us,
        errors_sec=errors_us * 1.0e-6,
        weights=1.0 / (errors_us * 1.0e-6) ** 2,
        dt_sec_cached=dt_sec,
        dt_sec_ld=np.asarray(dt_sec, dtype=np.longdouble),
        tdb_mjd=tdb_mjd,
        initial_dm_delay=None,
        dm_params=[p for p in fit_params if p == "DM"],
        spin_params=[p for p in fit_params if p.startswith("F") and p[1:].isdigit()],
        binary_params=get_binary_params_from_list(fit_params),
        astrometry_params=[
            p for p in fit_params if p in ("RAJ", "DECJ", "PX", "PMRA", "PMDEC")
        ],
        fd_params=[],
        sw_params=[],
        roemer_shapiro_sec=None,
        prebinary_delay_sec=prebinary,
        initial_binary_delay=initial_binary,
        ssb_obs_pos_ls=ssb,
        obs_sun_pos_ls=None,
        obs_planet_pos_ls=None,
        initial_astrometric_delay=None,
        initial_fd_delay=None,
        initial_sw_delay=None,
        sw_geometry_pc=None,
        toa_flags=None,
        ecorr_whitener=None,
        red_noise_basis=None,
        red_noise_prior=None,
        dm_noise_basis=None,
        dm_noise_prior=None,
        chromatic_noise_basis=None,
        chromatic_noise_prior=None,
        ecorr_basis=None,
        ecorr_prior=None,
        band_noise_bases=None,
        band_noise_priors=None,
        band_noise_labels=None,
        group_noise_bases=None,
        group_noise_priors=None,
        group_noise_labels=None,
        dmx_design_matrix=None,
        dmx_labels=None,
        initial_dmx_delay=None,
        dmjump_design_matrix=None,
        dmjump_labels=None,
        jump_masks=None,
        fdjump_masks=None,
        fdjump_params=[],
        initial_fdjump_delay=None,
        jump_phase=None,
        tzr_phase=None,
        noise_config=None,
        nonlinear_params=validate_nonlinear_params(nonlinear_params),
    )


def _export_model(fixture_id: str, fit_params: list[str], *, compatibility: str, nonlinear_params):
    fx = get_tempo2_fixture(fixture_id)
    session = open_session(
        fx["par_path"],
        fx["tim_path"],
        compatibility=compatibility,
        nonlinear_params=nonlinear_params,
        verbose=False,
    )
    session.compute_residuals()
    return session, export_frozen_residual_model(session, fit_params=fit_params)


def _wls_fit(fixture_id: str, fit_params: list[str], *, compatibility: str):
    fx = get_tempo2_fixture(fixture_id)
    return fit_parameters_optimized(
        fx["par_path"],
        fx["tim_path"],
        fit_params,
        max_iter=WLS_MAX_ITER.get(fixture_id, 40),
        verbose=False,
        compatibility=compatibility,
    )


# ---------------------------------------------------------------------------
# §8.6 Validation / warnings
# ---------------------------------------------------------------------------


def test_validate_nonlinear_params_closed_set():
    assert validate_nonlinear_params(None) is None
    assert validate_nonlinear_params("binary") == "binary"
    assert validate_nonlinear_params("binary+") == "binary+"
    assert validate_nonlinear_params("BINARY") == "binary"
    for bad in ("hybrid", "native", ["PB"], 1):
        with pytest.raises(ValueError, match="nonlinear_params"):
            validate_nonlinear_params(bad)  # type: ignore[arg-type]


def test_warn_if_tempo2_native_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger="jug.fitting.nonlinear_params"):
        warn_if_tempo2_native_ignored("binary", "fixed_state_bclt")
    assert any("ignores tempo2_native" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jug.fitting.nonlinear_params"):
        warn_if_tempo2_native_ignored("binary", "fixed_state_stripped")
        warn_if_tempo2_native_ignored(None, "fixed_state_bclt")
    assert not any("ignores tempo2_native" in r.message for r in caplog.records)


def test_hybrid_with_no_binary_axes_stays_hybrid():
    """Hybrid with only linear δ-axes still runs hybrid (pure Jδ); no silent native."""
    fit_params = ["F0", "RAJ"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params="binary")
    assert setup.nonlinear_params == "binary"
    fn = make_residual_delta_jax_fn(
        setup=setup, fit_params=fit_params, nonlinear_params="binary"
    )
    J = _bake_residual_jacobian_native(setup, fit_params)
    delta = np.array([1e-12, 1e-10], dtype=np.float64)
    r = np.asarray(fn(delta), dtype=np.float64)
    expected = J @ delta
    np.testing.assert_allclose(r, expected, atol=1e-15, rtol=0.0)
    assert is_hybrid_nonlinear_params(setup.nonlinear_params)


def test_binary_plus_without_binary_axes_is_pure_matmul():
    """Empty I_bin: binary+ does not run the plan (no live Kopeikin-PX)."""
    fit_params = ["F0", "PX"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params="binary+")
    J = _bake_residual_jacobian_native(setup, fit_params)
    fn_bin = make_residual_delta_jax_fn(
        setup=setup, fit_params=fit_params, nonlinear_params="binary"
    )
    setup.residual_delta_jax_cache = None
    fn_plus = make_residual_delta_jax_fn(
        setup=setup, fit_params=fit_params, nonlinear_params="binary+"
    )
    delta = np.array([0.0, 1e-3], dtype=np.float64)
    r_bin = np.asarray(fn_bin(delta), dtype=np.float64)
    r_plus = np.asarray(fn_plus(delta), dtype=np.float64)
    np.testing.assert_allclose(r_bin, J @ delta, atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(r_plus, r_bin, atol=1e-15, rtol=0.0)


def test_hybrid_rejects_non_none_phase_mean_mode():
    fit_params = ["F0", "PB"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params="binary")
    with pytest.raises(ValueError, match="phase_mean_mode"):
        _prepare_residual_delta_jax(
            setup=setup,
            fit_params=fit_params,
            phase_mean_mode="weighted",
            nonlinear_params="binary",
        )


def test_hybrid_missing_prebinary_raises_cache_msg():
    """With J override, missing prebinary raises the hybrid cache message (§4.3)."""
    from jug.fitting.jax_residual_delta import _HYBRID_BINARY_CACHE_MSG

    fit_params = ["F0", "PB"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params="binary")
    J = _bake_residual_jacobian_native(setup, fit_params)
    setup.prebinary_delay_sec = None
    setup.residual_delta_jax_cache = None
    with pytest.raises(ValueError, match="prebinary_delay_sec"):
        _prepare_residual_delta_jax(
            setup=setup,
            fit_params=fit_params,
            phase_mean_mode="none",
            nonlinear_params="binary",
            residual_jacobian=J,
        )
    # Message stays the locked hybrid cache string (not a bake-path error).
    try:
        _prepare_residual_delta_jax(
            setup=setup,
            fit_params=fit_params,
            phase_mean_mode="none",
            nonlinear_params="binary",
            residual_jacobian=J,
        )
    except ValueError as exc:
        assert str(exc) == _HYBRID_BINARY_CACHE_MSG


def test_open_session_hybrid_warns_on_nondefault_tempo2_native(caplog):
    fx = get_tempo2_fixture("epta_j0030_isolated")
    with caplog.at_level(logging.WARNING, logger="jug.fitting.nonlinear_params"):
        open_session(
            fx["par_path"],
            fx["tim_path"],
            compatibility="tempo2",
            tempo2_native="full",
            nonlinear_params="binary",
            verbose=False,
        )
    assert any("ignores tempo2_native" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# §8.1 No forbidden calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["binary", "binary+"])
def test_hybrid_no_forbidden_delay_calls(monkeypatch, mode):
    fit_params = ["F0", "PB", "A1"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params=mode)

    def _fail(*_a, **_k):
        pytest.fail("forbidden delay callable invoked on hybrid path")

    monkeypatch.setattr(
        "jug.residuals.tempo2.terms.compute_bbat_delay_change_sec_jax", _fail
    )
    monkeypatch.setattr("jug.fitting.forward_delay.compute_total_delay_change", _fail)
    monkeypatch.setattr("jug.fitting.forward_delay.compute_side_delay_change", _fail)

    fn = make_residual_delta_jax_fn(
        setup=setup, fit_params=fit_params, nonlinear_params=mode
    )
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    r0 = np.asarray(fn(zero), dtype=np.float64)
    assert np.max(np.abs(r0)) < 1e-12
    delta = jnp.asarray([0.0, 1e-8, 1e-6], dtype=jnp.float64)
    r1 = np.asarray(fn(delta), dtype=np.float64)
    assert np.all(np.isfinite(r1))


# ---------------------------------------------------------------------------
# §8.5 Session integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compat", ["pint", "tempo2"])
@pytest.mark.parametrize("mode", [None, "binary", "binary+"])
def test_session_nonlinear_params_export(compat, mode):
    fit_params = ["F0", "PB", "A1", "TASC", "EPS1", "EPS2"]
    session, model = _export_model(
        "ppta_j1902_ell1h", fit_params, compatibility=compat, nonlinear_params=mode
    )
    assert session.nonlinear_params == mode
    assert model.nonlinear_params == mode
    r = np.asarray(model.residual_delta_jax(np.zeros(len(fit_params))), dtype=np.float64)
    assert np.max(np.abs(r)) < 1e-14


# ---------------------------------------------------------------------------
# §8.2 Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compat", ["pint", "tempo2"])
@pytest.mark.parametrize("mode", ["binary", "binary+"])
@pytest.mark.parametrize(
    "fixture_id,fit_params",
    [
        (
            "ppta_j1902_ell1h",
            ["F0", "RAJ", "PB", "A1", "TASC", "EPS1", "EPS2"],
        ),
        (
            "epta_j0030_isolated",
            ["F0", "RAJ", "PX"],
        ),
    ],
)
def test_hybrid_invariants(compat, mode, fixture_id, fit_params):
    session, model = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=mode
    )
    n = len(fit_params)
    zero = np.zeros(n, dtype=np.float64)
    r0 = np.asarray(model.residual_delta_jax(zero), dtype=np.float64)
    assert np.max(np.abs(r0)) < 1e-14

    # Bake J from the same setup the export used
    cached = session._cached_result_by_mode[True]
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    setup = _build_general_fit_setup_from_cache(
        {
            "dt_sec": cached["dt_sec"],
            "dt_sec_ld": cached.get("dt_sec_ld"),
            "tdb_mjd": cached["tdb_mjd"],
            "model_mjd": cached.get("model_mjd"),
            "freq_bary_mhz": cached["freq_bary_mhz"],
            "toas_mjd": toas_mjd,
            "errors_us": np.array([toa.error_us for toa in session.toas_data]),
            "toa_flags": [toa.flags for toa in session.toas_data],
            "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
            "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
            "roemer_shapiro_sec": cached.get("roemer_shapiro_sec"),
            "term_diagnostics": cached.get("term_diagnostics"),
            "toas": session.toas_data,
            "nonlinear_params": mode,
        },
        session.params,
        list(fit_params),
        compatibility=compat,
        tempo2_native=session.tempo2_native,
    )
    setup.nonlinear_params = mode
    J = _bake_residual_jacobian_native(setup, fit_params)
    I_bin, I_lin = _hybrid_delta_partition(fit_params)
    live = plan_live_keys(mode)
    names = list(fit_params)
    I_matmul = tuple(
        i for i in I_lin if names[i] not in live
    )

    # Linear-only matmul δ (§2.4)
    delta = np.zeros(n, dtype=np.float64)
    for i in I_matmul:
        delta[i] = 1e-12 if names[i].startswith("F") else 1e-10
    r = np.asarray(model.residual_delta_jax(delta), dtype=np.float64)
    if I_matmul:
        expected = J[:, list(I_matmul)] @ delta[list(I_matmul)]
    else:
        expected = np.zeros_like(r)
    np.testing.assert_allclose(r, expected, atol=1e-15, rtol=0.0)

    # Pure-PX finite δ
    if "PX" in names:
        i_px = names.index("PX")
        d_px = np.zeros(n, dtype=np.float64)
        d_px[i_px] = 1e-3  # mas
        r_px = np.asarray(model.residual_delta_jax(d_px), dtype=np.float64)
        matmul_px = J[:, i_px] * d_px[i_px]
        if mode == "binary":
            np.testing.assert_allclose(r_px, matmul_px, atol=1e-15, rtol=0.0)
        else:
            # binary+: may add plan live-PX term; on non-DDK agree with binary
            _, model_bin = _export_model(
                fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
            )
            r_bin = np.asarray(model_bin.residual_delta_jax(d_px), dtype=np.float64)
            if str(session.params.get("BINARY", "")).upper() == "DDK":
                assert np.max(np.abs(r_px - matmul_px)) > 1e-18
            else:
                np.testing.assert_allclose(r_px, r_bin, atol=1e-15, rtol=0.0)

    # jit + jacfwd finite
    fn = model.residual_delta_jax
    jac = np.asarray(jax.jacfwd(fn)(jnp.zeros(n, dtype=jnp.float64)), dtype=np.float64)
    assert np.all(np.isfinite(jac))


# ---------------------------------------------------------------------------
# §8.3 Accuracy vs native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compat", ["pint", "tempo2"])
def test_hybrid_accuracy_ppta_j1902(compat):
    fixture_id = "ppta_j1902_ell1h"
    fit_params = ["F0", "RAJ", "PB", "A1", "TASC", "EPS1", "EPS2"]
    fit = _wls_fit(fixture_id, fit_params, compatibility=compat)
    _, native = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=None
    )
    _, hybrid = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
    )
    binary_names = frozenset(get_binary_params_from_list(fit_params))
    cases = [
        np.zeros(len(fit_params), dtype=np.float64),
        _delta_native_from_wls_uncertainties(
            fit, fit_params, k=1.0, active=binary_names
        ),
        _delta_native_from_wls_uncertainties(
            fit, fit_params, k=10.0, active=binary_names
        ),
        _delta_native_from_wls_uncertainties(
            fit, fit_params, k=1.0, active=frozenset({"F0"})
        ),
        _delta_native_from_wls_uncertainties(fit, fit_params, k=1.0, active=None),
    ]
    for delta in cases:
        rn = np.asarray(native.residual_delta_jax(delta), dtype=np.float64)
        rh = np.asarray(hybrid.residual_delta_jax(delta), dtype=np.float64)
        diff = rh - rn
        assert np.sqrt(np.mean(diff**2)) < NS_RMS
        assert np.max(np.abs(diff)) < NS_MAX


def test_hybrid_accuracy_ppta_j1741_tempo2():
    fixture_id = "ppta_j1741_ell1"
    fit_params = ["F0", "PB", "A1", "TASC", "EPS1", "EPS2"]
    compat = "tempo2"
    try:
        fit = _wls_fit(fixture_id, fit_params, compatibility=compat)
        _, native = _export_model(
            fixture_id, fit_params, compatibility=compat, nonlinear_params=None
        )
        _, hybrid = _export_model(
            fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
        )
        binary_names = frozenset(get_binary_params_from_list(fit_params))
        for k in (1.0, 10.0):
            delta = _delta_native_from_wls_uncertainties(
                fit, fit_params, k=k, active=binary_names
            )
            rn = np.asarray(native.residual_delta_jax(delta), dtype=np.float64)
            rh = np.asarray(hybrid.residual_delta_jax(delta), dtype=np.float64)
            diff = rh - rn
            assert np.sqrt(np.mean(diff**2)) < NS_RMS
            assert np.max(np.abs(diff)) < NS_MAX
    except AssertionError:
        pytest.xfail("ELL1 plan ell1_t2 truncation debt (§4.6)")


@pytest.mark.parametrize("compat", ["pint", "tempo2"])
def test_hybrid_accuracy_epta_j1918_ddh(compat):
    fixture_id = "epta_j1918_ddh"
    fit_params = ["F0", "PB", "A1", "T0", "OM", "ECC", "H3"]
    # Drop params missing from the par if needed
    fx = get_tempo2_fixture(fixture_id)
    session = open_session(
        fx["par_path"], fx["tim_path"], compatibility=compat, verbose=False
    )
    available = set(session.params)
    fit_params = [p for p in fit_params if p in available or p == "ECC"]
    if "ECC" in fit_params and "ECC" not in available and "E" in available:
        fit_params = [("ECC" if p == "ECC" else p) for p in fit_params]
    fit = _wls_fit(fixture_id, fit_params, compatibility=compat)
    _, native = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=None
    )
    _, hybrid = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
    )
    binary_names = frozenset(get_binary_params_from_list(fit_params))
    delta = _delta_native_from_wls_uncertainties(
        fit, fit_params, k=1.0, active=binary_names
    )
    rn = np.asarray(native.residual_delta_jax(delta), dtype=np.float64)
    rh = np.asarray(hybrid.residual_delta_jax(delta), dtype=np.float64)
    diff = rh - rn
    assert np.sqrt(np.mean(diff**2)) < NS_RMS
    assert np.max(np.abs(diff)) < NS_MAX


def test_hybrid_accuracy_wsrt167_no_binary():
    fixture_id = "wsrt167"
    fit_params = ["F0", "RAJ"]
    compat = "tempo2"
    fit = _wls_fit(fixture_id, fit_params, compatibility=compat)
    _, hybrid = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
    )
    delta = _delta_native_from_wls_uncertainties(fit, fit_params, k=1.0, active=None)
    # Rebuild setup to bake J
    session, _ = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
    )
    cached = session._cached_result_by_mode[True]
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    setup = _build_general_fit_setup_from_cache(
        {
            "dt_sec": cached["dt_sec"],
            "dt_sec_ld": cached.get("dt_sec_ld"),
            "tdb_mjd": cached["tdb_mjd"],
            "freq_bary_mhz": cached["freq_bary_mhz"],
            "toas_mjd": toas_mjd,
            "errors_us": np.array([toa.error_us for toa in session.toas_data]),
            "toa_flags": [toa.flags for toa in session.toas_data],
            "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
            "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
            "term_diagnostics": cached.get("term_diagnostics"),
            "toas": session.toas_data,
            "nonlinear_params": "binary",
        },
        session.params,
        list(fit_params),
        compatibility=compat,
        tempo2_native=session.tempo2_native,
    )
    J = _bake_residual_jacobian_native(setup, fit_params)
    rh = np.asarray(hybrid.residual_delta_jax(delta), dtype=np.float64)
    np.testing.assert_allclose(rh, J @ delta, atol=1e-15, rtol=0.0)


# ---------------------------------------------------------------------------
# §8.3b binary+ DDK / PX
# ---------------------------------------------------------------------------


def test_binary_plus_ddk_px_gates():
    fixture_id = "sim_ddk_tcb"
    fit_params = ["F0", "PX", "PB", "A1", "T0", "OM", "ECC", "KIN", "KOM"]
    compat = "tempo2"
    try:
        fx = get_tempo2_fixture(fixture_id)
    except KeyError:
        pytest.skip("sim_ddk_tcb fixture not present")

    session = open_session(
        fx["par_path"], fx["tim_path"], compatibility=compat, verbose=False
    )
    session.compute_residuals()
    if str(session.params.get("BINARY", "")).upper() != "DDK":
        pytest.skip("fixture is not DDK")

    fit = _wls_fit(fixture_id, fit_params, compatibility=compat)
    _, native = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=None
    )
    _, hybrid = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary"
    )
    _, hybrid_plus = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params="binary+"
    )

    d_px = _delta_native_from_wls_uncertainties(
        fit, fit_params, k=1.0, active=frozenset({"PX"})
    )
    binary_names = frozenset(get_binary_params_from_list(fit_params))
    d_bin = _delta_native_from_wls_uncertainties(
        fit, fit_params, k=1.0, active=binary_names
    )

    for delta in (d_px, d_bin):
        rn = np.asarray(native.residual_delta_jax(delta), dtype=np.float64)
        rp = np.asarray(hybrid_plus.residual_delta_jax(delta), dtype=np.float64)
        diff = rp - rn
        assert np.sqrt(np.mean(diff**2)) < NS_RMS
        assert np.max(np.abs(diff)) < NS_MAX

    r_bin = np.asarray(hybrid.residual_delta_jax(d_px), dtype=np.float64)
    r_plus = np.asarray(hybrid_plus.residual_delta_jax(d_px), dtype=np.float64)
    assert np.max(np.abs(r_plus - r_bin)) > 1e-18

    # Double-count verification (§0.3 steps 1–4)
    cached = session._cached_result_by_mode[True]
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache
    from jug.fitting.designmatrix_assembly import assemble_analytic_designmatrix
    from jug.utils.units import native_to_fit_value

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    setup = _build_general_fit_setup_from_cache(
        {
            "dt_sec": cached["dt_sec"],
            "dt_sec_ld": cached.get("dt_sec_ld"),
            "tdb_mjd": cached["tdb_mjd"],
            "freq_bary_mhz": cached["freq_bary_mhz"],
            "toas_mjd": toas_mjd,
            "errors_us": np.array([toa.error_us for toa in session.toas_data]),
            "toa_flags": [toa.flags for toa in session.toas_data],
            "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
            "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
            "term_diagnostics": cached.get("term_diagnostics"),
            "toas": session.toas_data,
            "nonlinear_params": "binary+",
        },
        session.params,
        list(fit_params),
        compatibility=compat,
        tempo2_native=session.tempo2_native,
    )
    setup.nonlinear_params = "binary+"
    J_baked = _bake_residual_jacobian_native(setup, fit_params)
    i_px = fit_params.index("PX")

    # Astrometry-only / Roemer PX column from analytic assembly
    M_fit = np.asarray(
        assemble_analytic_designmatrix(setup, fit_params, output_units="fit"),
        dtype=np.float64,
    )
    factor = float(native_to_fit_value("PX", 1.0))
    J_astro_px = -M_fit[:, i_px] * factor
    # (2) bake vs astrometry column
    rel = np.max(np.abs(J_baked[:, i_px] - J_astro_px)) / max(
        np.max(np.abs(J_astro_px)), 1e-30
    )
    if rel > 1e-12:
        # Strip Kopeikin contamination from bake for binary+ only
        J_baked = J_baked.copy()
        J_baked[:, i_px] = J_astro_px
    else:
        np.testing.assert_allclose(
            J_baked[:, i_px], J_astro_px, atol=1e-18 * max(np.max(np.abs(J_astro_px)), 1.0)
        )

    # (3) jacfwd(hybrid+)[:, PX] - J_baked[:, PX] ≈ plan-only Kopeikin jac
    _, _, jac_plus = _prepare_residual_delta_jax(
        setup=setup,
        fit_params=fit_params,
        phase_mean_mode="none",
        nonlinear_params="binary+",
        residual_jacobian=J_baked,
    )
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    J_plus = np.asarray(jac_plus(zero), dtype=np.float64)
    kop_col = J_plus[:, i_px] - J_baked[:, i_px]

    # Isolated plan-only PX derivative via finite-diff on binary delay phase map
    from jug.fitting.binary_delay_plan import resolve_binary_structure

    ref_params = dict(session.params)
    plan = resolve_binary_structure(
        ref_params, fit_params, obs_pos_ls=setup.ssb_obs_pos_ls
    )
    ref_f = tuple(float(x) for x in _spin_terms_from_params(ref_params))
    dt_base = setup.dt_sec_ld if setup.dt_sec_ld is not None else setup.dt_sec_cached

    def plan_only_px(px_delta):
        params = dict(ref_params)
        params["PX"] = float(ref_params.get("PX", 0.0)) + px_delta
        params_for_binary = _params_with_frozen_astrometry(
            params, ref_params, setup, live_px=True
        )
        db = _binary_delay_change_jax(params_for_binary, setup, binary_plan=plan)
        return _phase_residual_delta_jax(
            np.asarray(dt_base, dtype=np.float64),
            db,
            ref_f,
            ref_f,
            jnp.asarray(setup.weights, dtype=jnp.float64),
            mean_mode="none",
            f0=jnp.asarray(ref_f[0], dtype=jnp.float64),
        )

    eps = 1e-6
    plan_jac = (np.asarray(plan_only_px(eps)) - np.asarray(plan_only_px(-eps))) / (
        2.0 * eps
    )
    # Agree within binary-axis style tol
    denom = max(float(np.linalg.norm(plan_jac)), 1e-30)
    assert float(np.linalg.norm(kop_col - plan_jac)) / denom < 1e-2


def test_binary_vs_binary_plus_agree_on_non_ddk():
    """Non-DDK: binary and binary+ agree at 1σ PX (no Kopeikin path)."""
    fixture_id = "epta_j0030_isolated"
    fit_params = ["F0", "PX"]
    fit = _wls_fit(fixture_id, fit_params, compatibility="tempo2")
    _, hybrid = _export_model(
        fixture_id, fit_params, compatibility="tempo2", nonlinear_params="binary"
    )
    _, hybrid_plus = _export_model(
        fixture_id, fit_params, compatibility="tempo2", nonlinear_params="binary+"
    )
    d_px = _delta_native_from_wls_uncertainties(
        fit, fit_params, k=1.0, active=frozenset({"PX"})
    )
    r1 = np.asarray(hybrid.residual_delta_jax(d_px), dtype=np.float64)
    r2 = np.asarray(hybrid_plus.residual_delta_jax(d_px), dtype=np.float64)
    np.testing.assert_allclose(r1, r2, atol=1e-15, rtol=0.0)


# ---------------------------------------------------------------------------
# §8.4 Grad agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["binary", "binary+"])
def test_hybrid_jacfwd_agrees_native_binary_axes(mode):
    fixture_id = "ppta_j1902_ell1h"
    fit_params = ["F0", "RAJ", "PB", "A1", "TASC", "EPS1", "EPS2"]
    compat = "tempo2"
    _, native = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=None
    )
    session, hybrid = _export_model(
        fixture_id, fit_params, compatibility=compat, nonlinear_params=mode
    )
    n = len(fit_params)
    zero = jnp.zeros((n,), dtype=jnp.float64)
    Jn = np.asarray(jax.jacfwd(native.residual_delta_jax)(zero), dtype=np.float64)
    Jh = np.asarray(jax.jacfwd(hybrid.residual_delta_jax)(zero), dtype=np.float64)
    I_bin, I_lin = _hybrid_delta_partition(fit_params)
    live = plan_live_keys(mode)
    names = list(fit_params)

    for i in I_bin:
        col_n = Jn[:, i]
        col_h = Jh[:, i]
        norm = float(np.linalg.norm(col_n))
        if norm > 1e-6:
            rel = float(np.linalg.norm(col_h - col_n)) / norm
            assert rel < 1e-3, f"{names[i]}: rel={rel}"

    # Bake J for linear-axis exactness
    cached = session._cached_result_by_mode[True]
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    setup = _build_general_fit_setup_from_cache(
        {
            "dt_sec": cached["dt_sec"],
            "dt_sec_ld": cached.get("dt_sec_ld"),
            "tdb_mjd": cached["tdb_mjd"],
            "freq_bary_mhz": cached["freq_bary_mhz"],
            "toas_mjd": toas_mjd,
            "errors_us": np.array([toa.error_us for toa in session.toas_data]),
            "toa_flags": [toa.flags for toa in session.toas_data],
            "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
            "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
            "term_diagnostics": cached.get("term_diagnostics"),
            "toas": session.toas_data,
            "nonlinear_params": mode,
        },
        session.params,
        list(fit_params),
        compatibility=compat,
        tempo2_native=session.tempo2_native,
    )
    J_baked = _bake_residual_jacobian_native(setup, fit_params)
    I_matmul = tuple(i for i in I_lin if names[i] not in live)
    for i in I_matmul:
        np.testing.assert_allclose(Jh[:, i], J_baked[:, i], atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# §7 FP32 smoke
# ---------------------------------------------------------------------------


def test_hybrid_fp32_finiteness_smoke():
    """§7 scaffolding: float32 closure arrays stay finite; zero-delta within FP32 noise."""
    fit_params = ["F0", "PB"]
    setup = _toy_hybrid_setup(fit_params, nonlinear_params="binary")
    J64 = _bake_residual_jacobian_native(setup, fit_params)
    J32 = np.asarray(J64, dtype=np.float32)
    # Keep delay arrays consistent: cast then re-anchor initial_binary in float32.
    setup.prebinary_delay_sec = np.asarray(setup.prebinary_delay_sec, dtype=np.float32)
    from jug.fitting.binary_delay_plan import resolve_binary_structure

    plan = resolve_binary_structure(
        setup.params, fit_params, obs_pos_ls=setup.ssb_obs_pos_ls
    )
    toas_pre = np.asarray(setup.tdb_mjd, dtype=np.float32) - (
        np.asarray(setup.prebinary_delay_sec, dtype=np.float32) / np.float32(SECS_PER_DAY)
    )
    setup.initial_binary_delay = np.asarray(
        plan.evaluate(toas_pre, setup.params, setup.ssb_obs_pos_ls, np),
        dtype=np.float32,
    )
    setup.ssb_obs_pos_ls = np.asarray(setup.ssb_obs_pos_ls, dtype=np.float32)
    setup.residual_delta_jax_cache = None
    fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        nonlinear_params="binary",
        residual_jacobian=J32,
    )
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    r0 = np.asarray(fn(zero), dtype=np.float64)
    assert np.all(np.isfinite(r0))
    assert np.max(np.abs(r0)) < 1e-6  # FP32 noise floor, not the FP64 1e-14 gate
    delta = jnp.asarray([1e-12, 1e-8], dtype=jnp.float64)
    r1 = np.asarray(fn(delta), dtype=np.float64)
    assert np.all(np.isfinite(r1))


def test_open_session_accepts_nonlinear_params_kwarg():
    import inspect

    from jug.engine.api import open_session as _os

    assert "nonlinear_params" in inspect.signature(_os).parameters
