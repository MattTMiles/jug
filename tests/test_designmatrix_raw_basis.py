"""Raw analytic fitter-basis contract for compute_designmatrix."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from jug.fitting.designmatrix_assembly import assemble_analytic_designmatrix
from jug.fitting.jax_residual_delta import _simplified_residual_jacobian_oracle
from jug.fitting.optimized_fitter import (
    GeneralFitSetup,
    _build_general_fit_setup_from_files,
    _compute_designmatrix_from_setup,
    compute_designmatrix,
)
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY
from jug.utils.units import native_to_fit_value

GOLDEN_DIR = Path(__file__).parent / "data_golden"

# Per-family tolerances for J_fit ≈ -M_analytic (no centering transform).
# Binary analytic blocks are simplified tangents — keep a family-local rtol.
_FAMILY_TOL = {
    "spin": (2.0e-8, 1.0e-13),
    "astrometry": (5.0e-6, 1.0e-12),
    "binary": (2.0e-3, 1.0e-9),
    "DM": (2.0e-8, 1.0e-13),
}


@pytest.fixture(scope="module")
def j1909_paths():
    par = GOLDEN_DIR / "J1909_mini.par"
    tim = GOLDEN_DIR / "J1909_mini.tim"
    if not par.exists() or not tim.exists():
        pytest.skip("golden J1909 mini dataset not found")
    return par, tim


def test_compute_designmatrix_is_raw_fitter_basis(j1909_paths):
    par, tim = j1909_paths
    labels = ["F0", "RAJ", "DM"]
    result = compute_designmatrix(par, tim, labels)

    assert result.construction == "analytic-fitter"
    assert result.unit_convention == "pint-vela"
    assert result.compatibility == "pint"
    assert not result.matrix.flags.writeable
    assert not result.residuals_us.flags.writeable
    assert not result.errors_us.flags.writeable
    assert len(result.row_tokens) == result.matrix.shape[0]
    assert result.row_tokens[0].startswith("000000|")

    setup = _build_general_fit_setup_from_files(
        Path(par),
        Path(tim),
        labels,
        clock_dir=None,
        verbose=False,
        compatibility="pint",
    )
    np.testing.assert_allclose(
        result.matrix,
        assemble_analytic_designmatrix(setup, labels, output_units="fit"),
        rtol=0,
        atol=0,
    )
    assert result.reference_fit_values == tuple(
        float(native_to_fit_value(p, v))
        for p, v in zip(result.labels, setup.param_values_start, strict=True)
    )


def test_tzr_params_rejected(j1909_paths):
    par, tim = j1909_paths
    with pytest.raises(ValueError):
        compute_designmatrix(par, tim, ["TZRMJD"])


def test_compute_designmatrix_rejects_setup_filtered_parameter(j1909_paths):
    par, tim = j1909_paths
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        par_text = Path(par).read_text()
        # Flag matches no TOAs in the mini fixture -> empty JUMP mask filtered.
        par_with_empty_jump = tmp_path / "empty_jump.par"
        par_with_empty_jump.write_text(
            par_text + "\nJUMP -fe NONEXISTENT_BACKEND 0.0 1\n"
        )
        with pytest.raises(
            ValueError, match="does not expose a reduced or expanded fitter basis"
        ):
            compute_designmatrix(
                par_with_empty_jump, tim, ["F0", "JUMP1"]
            )


def _toy_setup(fit_params):
    tdb_mjd = np.array([55000.0, 55000.25, 55000.5, 55000.75, 55001.0], dtype=float)
    freq_mhz = np.array([820.0, 900.0, 1100.0, 1400.0, 1600.0], dtype=float)
    errors_us = np.full(len(tdb_mjd), 1.0, dtype=float)
    params = {
        "F0": 200.0,
        "F1": -1.0e-15,
        "PEPOCH": 55000.0,
        "DM": 10.0,
        "DMEPOCH": 55000.0,
        "RAJ": 0.0,
        "DECJ": 0.0,
        "PX": 1.0,
        "PMRA": 0.0,
        "PMDEC": 0.0,
        "POSEPOCH": 55000.0,
        "BINARY": "DD",
        "PB": 5.0,
        "T0": 55000.0,
        "A1": 10.0,
        "OM": 45.0,
        "ECC": 0.1,
    }
    dt_sec = (tdb_mjd - params["PEPOCH"]) * SECS_PER_DAY
    initial_dm_delay = K_DM_SEC * params["DM"] / (freq_mhz**2)
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
        initial_dm_delay=initial_dm_delay,
        dm_params=[p for p in fit_params if p == "DM" or p.startswith("DM")],
        spin_params=[p for p in fit_params if p.startswith("F") and p[1:].isdigit()],
        binary_params=[p for p in fit_params if p in ("PB", "T0", "A1", "OM", "ECC")],
        astrometry_params=[
            p for p in fit_params if p in ("RAJ", "DECJ", "PX", "PMRA", "PMDEC")
        ],
        fd_params=[],
        sw_params=[],
        roemer_shapiro_sec=None,
        prebinary_delay_sec=None,
        initial_binary_delay=None,
        ssb_obs_pos_ls=None,
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
    )


def _family_for(param: str) -> str:
    if param.startswith("F") and param[1:].isdigit():
        return "spin"
    if param in ("DM",) or param.startswith("DM"):
        return "DM"
    if param in ("RAJ", "DECJ", "PX", "PMRA", "PMDEC", "ELONG", "ELAT"):
        return "astrometry"
    return "binary"


@pytest.mark.parametrize(
    "fit_params",
    [
        ["F0", "F1"],
        ["DM"],
    ],
)
def test_oracle_j_equals_minus_m_without_centering(fit_params):
    """J_fit ≈ -M_analytic column-wise, no C transform."""
    setup = _toy_setup(fit_params)
    analytic = _compute_designmatrix_from_setup(setup, fit_params)
    j_fit = _simplified_residual_jacobian_oracle(setup, fit_params)
    for j, name in enumerate(fit_params):
        rtol, atol = _FAMILY_TOL[_family_for(name)]
        np.testing.assert_allclose(
            j_fit[:, j],
            -analytic[:, j],
            rtol=rtol,
            atol=atol,
            err_msg=f"{name}: expected J ≈ -M (gauge-free, no C)",
        )
        # Explicitly reject the old centered contract for columns with mean.
        col_mean = float(np.mean(analytic[:, j]))
        if abs(col_mean) > 1e-18:
            centered = analytic[:, j] - col_mean
            assert not np.allclose(
                j_fit[:, j], -centered, rtol=rtol, atol=atol
            ), f"{name}: J unexpectedly matches -C(M)"


# Binary-family J≈-M (no C) is covered by
# tests/test_designmatrix_autodiff.py::test_oracle_binary_column_matches_raw_analytic
# against a proper binary delay ledger; a zero-ledger toy setup is not meaningful.
