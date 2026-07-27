"""DD-family orthometric Shapiro autodiff regressions (Fix J3)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jug.fitting.binary_delay_plan import resolve_binary_structure
from jug.fitting.derivatives_dd import (
    _d_delay_d_H3,
    _extract_dd_params,
    _orthometric_values_active,
    compute_binary_derivatives_ddk,
    compute_ddk_binary_delay,
)
from jug.utils.constants import T_SUN


def _ddh_base(**overrides):
    params = {
        "BINARY": "DDH",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "H3": 1e-6,
        "STIG": 0.5,
    }
    params.update(overrides)
    return params


def test_extract_ecc_alias_and_signed_h3():
    assert _extract_dd_params({"E": 0.44, "A1": 1.0, "PB": 1.0, "T0": 55000.0, "OM": 0.0})[
        "ecc"
    ] == pytest.approx(0.44)
    p = _extract_dd_params(_ddh_base(H3=-1e-7, STIG=0.5, SINI=0.0, M2=0.0))
    assert p["m2"] < 0.0
    assert p["sini"] == pytest.approx(2 * 0.5 / (1 + 0.5**2))


def test_live_h3_stig_plan_and_signed_through_zero():
    params = _ddh_base(H3=0.0, STIG=0.8)
    plan = resolve_binary_structure(params, ["H3", "STIG"])
    assert plan.shapiro_param == "h3_stig"
    assert plan.has_shapiro is True
    toas = jnp.linspace(55000.0, 55001.0, 32)
    eps = 1e-9
    d_plus = np.asarray(plan.evaluate(toas, {**params, "H3": eps}, None, jnp))
    d_zero = np.asarray(plan.evaluate(toas, {**params, "H3": 0.0}, None, jnp))
    d_minus = np.asarray(plan.evaluate(toas, {**params, "H3": -eps}, None, jnp))
    np.testing.assert_allclose(d_plus - d_zero, d_zero - d_minus, atol=1e-18)


def test_live_h4_and_h3_with_h3h4_reference_rejected():
    with pytest.raises(NotImplementedError, match="Fitting H4"):
        resolve_binary_structure(_ddh_base(H4=1e-7, STIG=0.0), ["H3", "H4"])
    with pytest.raises(NotImplementedError, match="H3/H4 reference"):
        resolve_binary_structure(_ddh_base(H4=5e-7, STIG=0.0), ["H3"])


def test_nonfinite_standalone_h3_rejected():
    with pytest.raises(ValueError, match="Reference H3"):
        resolve_binary_structure(_ddh_base(H3=np.nan, STIG=0.0, H4=0.0), [])


def test_reference_only_h3h4_positive_and_negative_parity():
    toas = jnp.linspace(55000.0, 55001.0, 16)
    for h3, h4 in ((1e-6, 5e-7), (-1e-6, -5e-7)):
        params = _ddh_base(H3=h3, H4=h4, STIG=0.0, SINI=0.0, M2=0.0)
        plan = resolve_binary_structure(params, [])
        assert plan.shapiro_param == "m2_sini"
        converted = _extract_dd_params(params)
        explicit = {
            "BINARY": "DD",
            "A1": 10.0,
            "PB": 1.0,
            "T0": 55000.0,
            "ECC": 1e-4,
            "OM": 30.0,
            "M2": converted["m2"],
            "SINI": converted["sini"],
        }
        plan_m2 = resolve_binary_structure(explicit, [])
        d1 = np.asarray(plan.evaluate(toas, params, None, jnp))
        d2 = np.asarray(plan_m2.evaluate(toas, explicit, None, jnp))
        np.testing.assert_allclose(d1, d2, atol=1e-15)


def test_stig_above_one_matches_converted_m2_sini():
    stig = 1.154
    h3 = 1e-6
    params = _ddh_base(H3=h3, STIG=stig)
    plan = resolve_binary_structure(params, ["H3", "STIG"])
    sini = 2 * stig / (1 + stig**2)
    m2 = h3 / (stig**3 * T_SUN)
    explicit = {
        "BINARY": "DD",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "M2": m2,
        "SINI": sini,
    }
    plan_m2 = resolve_binary_structure(explicit, [])
    toas = jnp.linspace(55000.0, 55001.0, 16)
    d1 = np.asarray(plan.evaluate(toas, params, None, jnp))
    d2 = np.asarray(plan_m2.evaluate(toas, explicit, None, jnp))
    np.testing.assert_allclose(d1, d2, atol=1e-15)


def test_m2_zero_reference_live_m2_has_nonzero_column():
    params = {
        "BINARY": "DD",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "M2": 0.0,
        "SINI": 0.9,
    }
    plan = resolve_binary_structure(params, ["M2"])
    assert plan.has_shapiro is True
    toas = jnp.linspace(55000.0, 55001.0, 32)

    def delay(m2):
        return plan.evaluate(toas, {**params, "M2": m2}, None, jnp).sum()

    col = jax.jacfwd(delay)(0.0)
    assert np.isfinite(col)
    assert abs(float(col)) > 0.0


def test_ddk_orthometric_rejection_three_paths():
    base = {
        "BINARY": "DDK",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "KIN": 60.0,
        "KOM": 0.0,
        "M2": 0.2,
        "SINI": "KIN",
        "RAJ": 1.0,
        "DECJ": -0.5,
        "H3": 0.0,
        "STIG": 0.0,
    }
    toas = np.linspace(55000.0, 55001.0, 8)
    # Zero-only orthometric lines pass all three for non-orthometric derivatives.
    assert _orthometric_values_active(base) is False
    resolve_binary_structure(base, ["KIN"])
    compute_ddk_binary_delay(toas, base)
    compute_binary_derivatives_ddk(base, toas, ["KIN"])

    # Requested H3 still rejects even with all-zero stored values.
    with pytest.raises(NotImplementedError, match="DDK/Kopeikin"):
        compute_binary_derivatives_ddk(base, toas, ["H3"])

    active = dict(base)
    active["H3"] = 1e-6
    with pytest.raises(NotImplementedError, match="DDK/Kopeikin"):
        resolve_binary_structure(active, [])
    with pytest.raises(NotImplementedError, match="DDK/Kopeikin"):
        compute_ddk_binary_delay(toas, active)
    with pytest.raises(NotImplementedError, match="DDK/Kopeikin"):
        compute_binary_derivatives_ddk(active, toas, ["A1"])


def test_h3_analytic_autodiff_parity():
    params = _ddh_base(H3=1e-6, STIG=0.5)
    plan = resolve_binary_structure(params, ["H3", "STIG"])
    toas = jnp.linspace(55000.0, 55002.0, 24)
    xp = float(params["H3"])
    h = 1e-5 * max(abs(xp), 1e-7)

    def delay_h3(h3):
        return plan.evaluate(toas, {**params, "H3": h3}, None, jnp)

    jad = np.asarray(jax.jacfwd(delay_h3)(xp))
    jfd = (np.asarray(delay_h3(xp + h)) - np.asarray(delay_h3(xp - h))) / (2 * h)
    assert np.max(np.abs(jad)) > 0
    np.testing.assert_allclose(
        jad, jfd, rtol=2e-5, atol=1e-30 + 1e-6 * np.max(np.abs(jad))
    )
    # Analytic column agrees in shape/finiteness.
    analytic = np.asarray(
        _d_delay_d_H3(toas, params["PB"], params["T0"], params["ECC"], jnp.deg2rad(params["OM"]), 0.0, params["STIG"])
    )
    assert np.all(np.isfinite(analytic))
    assert np.max(np.abs(analytic)) > 0


def test_stigma_alias_and_mixed_m2_rejection():
    params = _ddh_base(H3=1e-6, STIG=0.0)
    params["STIGMA"] = 0.6
    del params["STIG"]
    plan = resolve_binary_structure(params, ["H3", "STIGMA"])
    assert plan.shapiro_param == "h3_stig"
    with pytest.raises(ValueError, match="mixes M2/SINI"):
        resolve_binary_structure(_ddh_base(M2=0.1, SINI=0.9), ["H3", "STIG", "M2"])


def test_invalid_orthometric_references():
    with pytest.raises(ValueError, match="Reference H3"):
        resolve_binary_structure(_ddh_base(H3=np.inf, STIG=0.0, H4=0.0), [])
    with pytest.raises(ValueError, match="Reference H3"):
        resolve_binary_structure(_ddh_base(H3=np.nan, STIG=0.0, H4=0.0), [])
    with pytest.raises(ValueError, match="Reference STIG"):
        resolve_binary_structure(_ddh_base(STIG=-0.1), [])
    with pytest.raises(ValueError, match="Reference STIG"):
        resolve_binary_structure(_ddh_base(H3=0.0, STIG=0.0), ["H3", "STIG"])
    with pytest.raises(ValueError, match="H4/H3"):
        resolve_binary_structure(_ddh_base(H3=1e-6, H4=-5e-7, STIG=0.0), [])


def test_shapmax_live_rejected():
    params = {
        "BINARY": "DDS",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "M2": 0.2,
        "SHAPMAX": 3.0,
    }
    with pytest.raises(NotImplementedError, match="SHAPMAX"):
        resolve_binary_structure(params, ["SHAPMAX"])


def test_signed_h3_column_linear_through_zero():
    params = _ddh_base(H3=0.0, STIG=0.7)
    plan = resolve_binary_structure(params, ["H3"])
    toas = jnp.linspace(55000.0, 55001.0, 32)

    def delay(h3):
        return plan.evaluate(toas, {**params, "H3": h3}, None, jnp)

    cols = [np.asarray(jax.jacfwd(delay)(h3)) for h3 in (-1e-7, 0.0, 1e-7)]
    for col in cols:
        assert np.all(np.isfinite(col))
        assert np.max(np.abs(col)) > 0
    np.testing.assert_allclose(cols[0], cols[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cols[2], cols[1], rtol=0, atol=1e-12)


def test_no_shapiro_sector_matches_zero_shapiro():
    params = {
        "BINARY": "DD",
        "A1": 10.0,
        "PB": 1.0,
        "T0": 55000.0,
        "ECC": 1e-4,
        "OM": 30.0,
        "M2": 0.0,
        "SINI": 0.0,
    }
    plan = resolve_binary_structure(params, ["A1"])
    assert plan.has_shapiro is False
    toas = jnp.linspace(55000.0, 55001.0, 16)
    d = np.asarray(plan.evaluate(toas, params, None, jnp))
    # Equivalent explicit call with has_shapiro False path: delay must be finite
    # and unchanged if we inject a nonzero M2 while keeping the static flag False
    # is not possible through the plan API; instead compare to m2=sini=0 kernel.
    from jug.fitting.derivatives_dd import _compute_dd_binary_delay_jit

    tt0 = (toas - params["T0"]) * 86400.0
    d_kernel = np.asarray(
        _compute_dd_binary_delay_jit(
            tt0,
            params["A1"],
            params["PB"],
            params["ECC"],
            params["OM"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            has_shapiro=False,
        )
    )
    np.testing.assert_allclose(d, d_kernel, rtol=0, atol=0)


@pytest.mark.parametrize("stig_key", ["STIG", "STIGMA"])
def test_orthometric_fd_parity_and_residual_deltas(stig_key):
    import dataclasses

    from jug.fitting.binary_registry import compute_binary_delay
    from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
    from jug.utils.constants import SECS_PER_DAY
    from test_designmatrix_autodiff import _setup

    h3 = 1e-6
    stig = 0.55
    params = _ddh_base(H3=h3, STIG=0.0)
    params[stig_key] = stig
    if stig_key != "STIG":
        params.pop("STIG", None)
    fit = ["H3", stig_key]
    plan = resolve_binary_structure(params, fit)
    toas = jnp.linspace(55000.0, 55002.0, 20)
    scales = {"H3": 1e-7, "STIG": 0.1, "STIGMA": 0.1}
    for name in fit:
        xp = float(params[name])
        h = 1e-5 * max(abs(xp), scales[name])

        def delay(val, _name=name):
            return plan.evaluate(toas, {**params, _name: val}, None, jnp)

        jad = np.asarray(jax.jacfwd(delay)(xp))
        jfd = (np.asarray(delay(xp + h)) - np.asarray(delay(xp - h))) / (2 * h)
        assert np.max(np.abs(jad)) > 0
        np.testing.assert_allclose(
            jad, jfd, rtol=2e-5, atol=1e-30 + 1e-6 * np.max(np.abs(jad))
        )

    base = _setup(["F0"])
    n = len(base.tdb_mjd)
    obs = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    prebinary = np.zeros(n)
    toas_pre = base.tdb_mjd - prebinary / SECS_PER_DAY
    full = dict(params, F0=200.0, PEPOCH=55000.0)
    if stig_key == "STIGMA":
        full["STIG"] = stig  # canonical copy for residual writes
    init = np.asarray(compute_binary_delay(toas_pre, full, obs_pos_ls=obs), dtype=float)
    setup = dataclasses.replace(
        base,
        params=full,
        fit_param_list=fit,
        param_values_start=[float(full[p]) for p in fit],
        binary_params=fit,
        prebinary_delay_sec=prebinary,
        initial_binary_delay=init,
        ssb_obs_pos_ls=obs,
        dm_params=[],
        spin_params=[],
        initial_dm_delay=None,
        binary_plan=None,
    )
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit)
    for i, name in enumerate(fit):
        xp = float(params[name] if name in params else full[name])
        h = 1e-5 * max(abs(xp), scales[name])
        delta_p = np.zeros(len(fit))
        delta_m = np.zeros(len(fit))
        delta_p[i] = h
        delta_m[i] = -h
        jfd = (np.asarray(fn(delta_p)) - np.asarray(fn(delta_m))) / (2 * h)
        assert np.max(np.abs(jfd)) > 0
