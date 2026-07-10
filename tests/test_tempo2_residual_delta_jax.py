"""DEV ORACLE — native residual_delta uses staged BCLT tail by default."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax
import jax.numpy as jnp

from jug.fitting.jax_residual_delta import (
    compute_autodiff_designmatrix_from_setup,
    make_residual_delta_jax_fn,
)
from tempo2_fixture_assertions import assert_column_matches, tempo2_to_pint_vela_scale
from tempo2_test_helpers import load_wsrt167_fixture


@pytest.fixture
def wsrt167_setup(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(["F0"], tempo2_native="staged_bclt")


@pytest.fixture
def wsrt167_setup_multiparam(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(
        ["RAJ", "DECJ", "F0", "DM"], tempo2_native="staged_bclt"
    )


@pytest.fixture
def wsrt167_fixed_state_setup(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(
        ["RAJ", "DECJ", "F0", "DM"], tempo2_native="fixed_state_bclt"
    )


def test_native_bclt_converges_within_fixed_iter(wsrt167_setup):
    """Fixed scan length must cover tempo2 convergence on wsrt167."""
    from jug.residuals.tempo2.calculate_bclt_jax import DEFAULT_BCLT_JAX_FIXED_ITER
    from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.residuals.simple_calculator import compute_residuals_simple

    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=True,
    )
    terms = prepare_tempo2_chain_from_simple_result(jug, params, toas)
    iters = np.asarray(jax.device_get(terms.bclt_iterations), dtype=np.int32)
    converged = np.asarray(jax.device_get(terms.converged), dtype=bool)
    assert bool(np.all(converged)), (
        f"BCLT did not converge within {DEFAULT_BCLT_JAX_FIXED_ITER} fixed iterations; "
        f"max iter seen {int(iters.max())}"
    )
    assert int(iters.max()) <= DEFAULT_BCLT_JAX_FIXED_ITER


def test_native_residual_delta_reverse_mode_grad_finite(wsrt167_setup_multiparam):
    """Reverse-mode AD through native staged chain must be finite at theta=0."""
    setup = wsrt167_setup_multiparam
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)

    def loss(x):
        r = fn(x)
        return jnp.sum(r * r)

    grad = jax.grad(loss)(zero)
    grad_np = np.asarray(grad, dtype=np.float64)
    assert grad_np.shape == (len(fit_params),)
    assert np.all(np.isfinite(grad_np))
    assert float(np.max(np.abs(grad_np))) > 0.0


def test_fixed_state_residual_delta_reverse_mode_grad_finite(wsrt167_fixed_state_setup):
    """Reverse-mode AD through fixed-state nonlinear chain must be finite at theta=0."""
    setup = wsrt167_fixed_state_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)

    def loss(x):
        r = fn(x)
        return jnp.sum(r * r)

    grad = jax.grad(loss)(zero)
    grad_np = np.asarray(grad, dtype=np.float64)
    assert grad_np.shape == (len(fit_params),)
    assert np.all(np.isfinite(grad_np))
    assert float(np.max(np.abs(grad_np))) > 0.0


def test_native_jacfwd_jacrev_agreement(wsrt167_setup_multiparam):
    """Forward- and reverse-mode Jacobians must agree at theta=0."""
    setup = wsrt167_setup_multiparam
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    jac_fwd = np.asarray(jax.jacfwd(fn)(zero), dtype=np.float64)
    jac_rev = np.asarray(jax.jacrev(fn)(zero), dtype=np.float64)
    scale = max(float(np.max(np.abs(jac_fwd))), 1.0)
    assert float(np.max(np.abs(jac_fwd - jac_rev))) / scale < 1e-6


def test_native_residual_delta_uses_bbat_displacement_not_pint_delay(
    wsrt167_setup, monkeypatch
):
    """Tempo2 autodiff must use native bbat displacement, not the PINT delay path."""
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")

    calls = {"bbat": 0}

    def _pint_delay(*args, **kwargs):
        raise AssertionError(
            "compute_total_delay_change must not run in tempo2 native mode"
        )

    def _bbat_delay_change(*args, **kwargs):
        calls["bbat"] += 1
        from jug.residuals.tempo2.terms import compute_bbat_delay_change_sec_jax

        return compute_bbat_delay_change_sec_jax(*args, **kwargs)

    def _side_ok(*args, **kwargs):
        from jug.fitting.forward_delay import compute_side_delay_change

        return compute_side_delay_change(*args, **kwargs)

    monkeypatch.setattr(
        "jug.fitting.forward_delay.compute_total_delay_change",
        _pint_delay,
    )
    monkeypatch.setattr(
        "jug.fitting.jax_residual_delta.compute_side_delay_change",
        _side_ok,
    )
    monkeypatch.setattr(
        "jug.fitting.jax_residual_delta.compute_bbat_delay_change_sec_jax",
        _bbat_delay_change,
    )
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0"])
    delta = fn(jnp.zeros(1, dtype=jnp.float64))
    assert delta.shape[0] == setup.toas_mjd.shape[0]
    assert calls["bbat"] == 1


def test_native_autodiff_designmatrix_f0_matches_libstempo(wsrt167_setup):
    """Phase 5 gate: jacfwd(residual_delta) F0 column vs libstempo design matrix."""
    pytest.importorskip("libstempo")
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fixture = load_wsrt167_fixture()
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, ["F0"])
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=["F0"],
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    ref_col = ref.designmatrix[:, ref.designmatrix_labels.index("F0")]
    assert_column_matches("F0", matrix[:, 0], ref_col)


def test_native_f0_jacfwd_finite_difference_spot_check(wsrt167_setup):
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0"])
    jac = np.asarray(jax.jacfwd(fn)(jnp.zeros(1, dtype=jnp.float64)), dtype=np.float64).reshape(-1)
    # Full-chain phase5 spans decades; keep FD in the local linear regime (trunc wraps
    # outside ~1e-10 Hz for wsrt167).
    eps = 1e-10
    fd = (np.asarray(fn(jnp.asarray([eps]))) - np.asarray(fn(jnp.asarray([-eps])))) / (2 * eps)
    fd = fd.reshape(-1)
    scale = max(float(np.max(np.abs(fd))), 1.0)
    assert float(np.max(np.abs(jac - fd))) / scale < 0.05


@pytest.mark.parametrize(
    "param,eps",
    [
        ("RAJ", 1e-10),
        ("DECJ", 1e-10),
        ("PX", 1e-4),
        ("DM", 1e-5),
    ],
)
def test_native_delta_recomputes_delay_terms(wsrt167_setup, param, eps):
    """Phase 5 gate: delay terms must move when astrometry/DM are perturbed."""
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=[param])
    delta = np.asarray(fn(jnp.asarray([eps], dtype=jnp.float64)))
    assert np.max(np.abs(delta)) > 0.0, (
        f"{param} produced zero native residual_delta; delay terms are frozen"
    )


@pytest.mark.parametrize(
    "param,eps",
    [
        ("RAJ", 1e-10),
        ("DECJ", 1e-10),
        ("PX", 1e-4),
        ("DM", 1e-5),
    ],
)
def test_fixed_state_delta_recomputes_delay_terms(wsrt167_fixed_state_setup, param, eps):
    setup = wsrt167_fixed_state_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=[param])
    delta = np.asarray(fn(jnp.asarray([eps], dtype=jnp.float64)))
    assert np.max(np.abs(delta)) > 0.0, (
        f"{param} produced zero fixed-state residual_delta; delay terms are frozen"
    )


def test_fixed_state_close_to_staged_for_small_pta_perturbation(
    wsrt167_setup_multiparam,
    wsrt167_fixed_state_setup,
):
    """Fixed-state nonlinear should track staged BCLT for PTA-scale perturbations."""
    staged = wsrt167_setup_multiparam
    fixed = wsrt167_fixed_state_setup
    if staged.native_chain_static is None or fixed.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    eps = jnp.asarray([1e-10, 1e-10, 1e-10, 1e-5], dtype=jnp.float64)
    staged_fn = make_residual_delta_jax_fn(setup=staged, fit_params=fit_params)
    fixed_fn = make_residual_delta_jax_fn(setup=fixed, fit_params=fit_params)
    staged_delta = np.asarray(staged_fn(eps), dtype=np.float64)
    fixed_delta = np.asarray(fixed_fn(eps), dtype=np.float64)
    assert float(np.max(np.abs(staged_delta - fixed_delta))) < 1e-9


@pytest.fixture
def wsrt167_stripped_setup(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(
        ["RAJ", "DECJ", "F0", "DM"], tempo2_native="fixed_state_stripped"
    )


@pytest.mark.parametrize("param", ["F0", "RAJ", "DECJ", "DM"])
def test_native_autodiff_designmatrix_column_matches_libstempo(
    wsrt167_setup_multiparam, param
):
    """Autodiff design-matrix columns vs libstempo on wsrt167 (staged_bclt)."""
    pytest.importorskip("libstempo")
    setup = wsrt167_setup_multiparam
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fixture = load_wsrt167_fixture()
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    col_idx = fit_params.index(param)
    ref_col = (
        ref.designmatrix[:, ref.designmatrix_labels.index(param)]
        * tempo2_to_pint_vela_scale(param)
    )
    assert_column_matches(param, matrix[:, col_idx], ref_col)


@pytest.mark.parametrize("param", ["F0", "RAJ", "DECJ", "DM"])
def test_fixed_state_autodiff_designmatrix_column_matches_libstempo(
    wsrt167_fixed_state_setup, param
):
    """Autodiff design-matrix columns vs libstempo on wsrt167 (fixed_state_bclt)."""
    pytest.importorskip("libstempo")
    setup = wsrt167_fixed_state_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fixture = load_wsrt167_fixture()
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    col_idx = fit_params.index(param)
    ref_col = (
        ref.designmatrix[:, ref.designmatrix_labels.index(param)]
        * tempo2_to_pint_vela_scale(param)
    )
    assert_column_matches(param, matrix[:, col_idx], ref_col)


@pytest.mark.parametrize("param", ["F0", "RAJ", "DECJ", "DM"])
def test_stripped_autodiff_designmatrix_column_matches_libstempo(
    wsrt167_stripped_setup, param
):
    """Autodiff design-matrix columns vs libstempo on wsrt167 (fixed_state_stripped)."""
    pytest.importorskip("libstempo")
    setup = wsrt167_stripped_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    fixture = load_wsrt167_fixture()
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    col_idx = fit_params.index(param)
    ref_col = (
        ref.designmatrix[:, ref.designmatrix_labels.index(param)]
        * tempo2_to_pint_vela_scale(param)
    )
    assert_column_matches(param, matrix[:, col_idx], ref_col)


def test_stripped_envelope_vs_staged_bclt(wsrt167_setup_multiparam, wsrt167_stripped_setup):
    """Stripped tangent must track staged BCLT at PTA-scale perturbations."""
    staged = wsrt167_setup_multiparam
    stripped = wsrt167_stripped_setup
    if staged.native_chain_static is None or stripped.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    eps = jnp.asarray([1e-10, 1e-10, 1e-10, 1e-5], dtype=jnp.float64)
    staged_fn = make_residual_delta_jax_fn(setup=staged, fit_params=fit_params)
    stripped_fn = make_residual_delta_jax_fn(setup=stripped, fit_params=fit_params)
    staged_delta = np.asarray(staged_fn(eps), dtype=np.float64)
    stripped_delta = np.asarray(stripped_fn(eps), dtype=np.float64)
    diff = staged_delta - stripped_delta
    rms_ns = float(np.sqrt(np.mean((diff * 1e9) ** 2)))
    assert rms_ns < 1.0


@pytest.fixture(scope="module")
def epta_j1909_setup():
    from tempo2_fixtures import get_tempo2_fixture
    from tempo2_test_helpers import build_fit_setup_for_fixture

    fixture = get_tempo2_fixture("epta_j1909_t2")
    return build_fit_setup_for_fixture(
        fixture,
        ["F0", "PB", "A1", "EPS1", "EPS2"],
        tempo2_native="staged_bclt",
    )


@pytest.fixture(scope="module")
def epta_j0613_addsat_setup():
    from tempo2_fixtures import get_tempo2_fixture
    from tempo2_test_helpers import build_fit_setup_for_fixture

    fixture = get_tempo2_fixture("epta_j0613_addsat_min")
    return build_fit_setup_for_fixture(
        fixture,
        ["F0", "DM"],
        tempo2_native="staged_bclt",
    )


@pytest.mark.parametrize("param", ["PB", "A1", "EPS1", "EPS2"])
def test_binary_autodiff_designmatrix_column_matches_libstempo(epta_j1909_setup, param):
    """Binary DD autodiff columns vs libstempo on epta_j1909_t2."""
    pytest.importorskip("libstempo")
    setup = epta_j1909_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["F0", "PB", "A1", "EPS1", "EPS2"]
    from tempo2_fixtures import get_tempo2_fixture

    fixture = get_tempo2_fixture("epta_j1909_t2")
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    col_idx = fit_params.index(param)
    ref_col = (
        ref.designmatrix[:, ref.designmatrix_labels.index(param)]
        * tempo2_to_pint_vela_scale(param)
    )
    assert_column_matches(param, matrix[:, col_idx], ref_col)


def test_addsat_autodiff_f0_matches_libstempo(epta_j0613_addsat_setup):
    """TRACK −2 ``-addsat`` mini fixture: F0 autodiff vs libstempo."""
    pytest.importorskip("libstempo")
    setup = epta_j0613_addsat_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    from tempo2_fixtures import get_tempo2_fixture

    fixture = get_tempo2_fixture("epta_j0613_addsat_min")
    from jug.testing.tempo2_reference import tempo2_reference

    matrix = compute_autodiff_designmatrix_from_setup(setup, ["F0"])
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=["F0"],
        include_designmatrix=True,
    )
    assert ref.designmatrix is not None
    ref_col = ref.designmatrix[:, ref.designmatrix_labels.index("F0")]
    assert_column_matches("F0", matrix[:, 0], ref_col)
