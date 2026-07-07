"""Fresh wsrt167 runtime evidence for Phase 2 IFTE / tt_tb gates (dev_oracle)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2_native.chain_jax import _load_model_static_for_native_chain
from jug.residuals.tempo2_native.clock_jax import (
    compute_tempo2_correction_tt_tb_jax,
    compute_tempo2_get_correction_tt_jax,
)
from jug.testing.tempo2_formbats_closure import compare_formbats_components
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from jug.utils.constants import SECS_PER_DAY
from jug.utils.ifteph import IFTE_LC, IFTE_TEPH0_SEC, ifte_delta_t_mjd, load_ifte_coeff_tables
from jug.utils.timescales import IFTE_K, is_tempo2_si_units, parse_timescale
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    load_wsrt167_fixture,
    rms_ns,
)


def test_phase2_runtime_evidence_wsrt167():
    """Collect numeric gates and write /tmp/jug_phase2_runtime_evidence.txt."""
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    from pytempo.sandbox import tempopulsar

    diag = tempopulsar(
        parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False
    ).toa_diagnostics(removemean=False)
    teph_pt = np.asarray(diag["correction_tt_teph_sec"], dtype=np.float64)
    static = _load_model_static_for_native_chain(params, toas, jug)
    sat = np.asarray(jug["term_diagnostics"]["sat_mjd"], dtype=np.float64)
    td = jug["term_diagnostics"]
    obs = td["tempo2_obs_state"]
    obs_earth = np.asarray(obs["observatory_earth_km"], dtype=np.float64)[:, :3]
    earth_vel = np.asarray(obs["earth_ssb_km"], dtype=np.float64)[:, 3:6]

    tt_jax = np.asarray(
        jax.device_get(
            compute_tempo2_get_correction_tt_jax(
                jnp.asarray(sat, dtype=jnp.float64),
                chain_mjd_tables=tuple(
                    jnp.asarray(t, dtype=jnp.float64) for t in static.chain_mjd_tables
                ),
                chain_offset_tables=tuple(
                    jnp.asarray(t, dtype=jnp.float64) for t in static.chain_offset_tables
                ),
                bipm_mjd=jnp.asarray(static.bipm_mjd, dtype=jnp.float64),
                bipm_offset=jnp.asarray(static.bipm_offset, dtype=jnp.float64),
            )
        ),
        dtype=np.float64,
    )
    mjd_tt = sat + tt_jax / SECS_PER_DAY
    ifte_tables = load_ifte_coeff_tables()
    units = parse_timescale(params)
    tt_tb_jax, teph_jax = compute_tempo2_correction_tt_tb_jax(
        jnp.asarray(mjd_tt, dtype=jnp.float64),
        jnp.asarray(obs_earth, dtype=jnp.float64),
        jnp.asarray(earth_vel, dtype=jnp.float64),
        ifte_records=jnp.asarray(ifte_tables.records, dtype=jnp.float64),
        ifte_start_jd=jnp.asarray(ifte_tables.start_jd, dtype=jnp.float64),
        ifte_end_jd=jnp.asarray(ifte_tables.end_jd, dtype=jnp.float64),
        ifte_step_jd=jnp.asarray(ifte_tables.step_jd, dtype=jnp.float64),
        ifte_coef_offset=int(ifte_tables.coef_offset),
        ifte_ncf=int(ifte_tables.ncf),
        ifte_na=int(ifte_tables.na),
        units_tdb=str(units).upper() == "TDB",
        si_units=is_tempo2_si_units(units),
    )
    tt_tb_jax = np.asarray(jax.device_get(tt_tb_jax), dtype=np.float64)
    teph_jax = np.asarray(jax.device_get(teph_jax), dtype=np.float64)

    native = compute_native_terms_for_fixture(fixture)
    native_tt = np.asarray(jax.device_get(native.correction_tt_sec), dtype=np.float64)
    native_tt_tb = np.asarray(jax.device_get(native.correction_tt_tb_sec), dtype=np.float64)

    report = compare_formbats_components(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )

    rms = {
        "tt_jax": rms_ns(tt_jax, oracle.fields["correction_tt_sec"]),
        "teph_jax": rms_ns(teph_jax, teph_pt),
        "tt_tb_jax": rms_ns(tt_tb_jax, oracle.fields["correction_tt_tb_sec"]),
        "native_tt": rms_ns(native_tt, oracle.fields["correction_tt_sec"]),
        "native_tt_tb": rms_ns(native_tt_tb, oracle.fields["correction_tt_tb_sec"]),
        "host_tt_tb_slot": report.component_rms_ns["tt_tb"],
    }

    trace_lines = []
    for i in [0, 42, 85, 166]:
        tt_pt = float(diag["correction_tt_sec"][i])
        mjd_tt_pt = float(sat[i] + tt_pt / SECS_PER_DAY)
        delta_jug = float(ifte_delta_t_mjd(mjd_tt_pt))
        obs_scaled = float(diag["obs_term_raw_sec"][i]) / IFTE_K
        delta_implied = (
            float(diag["correction_tt_teph_sec"][i]) - IFTE_TEPH0_SEC - obs_scaled
        ) * (1.0 - IFTE_LC)
        trace_lines.append(
            f"  TOA {i}: jug={delta_jug:.12e} implied={delta_implied:.12e} "
            f"diff={(delta_jug - delta_implied) * 1e9:.6f} ns"
        )

    lines = [
        "=== wsrt167 Phase 2 runtime evidence (fresh run) ===",
        "",
        "Phase 2 gates (pytempo oracle, wsrt167, TCB/SI):",
        f"  tt_jax vs pytempo:          {rms['tt_jax']:.9f} ns RMS",
        f"  teph_jax vs pytempo:        {rms['teph_jax']:.9f} ns RMS",
        f"  tt_tb_jax vs pytempo:       {rms['tt_tb_jax']:.9f} ns RMS",
        f"  native chain tt vs pytempo: {rms['native_tt']:.9f} ns RMS",
        f"  native chain tt_tb vs pytempo: {rms['native_tt_tb']:.9f} ns RMS",
        f"  host tt_tb slot vs pytempo: {rms['host_tt_tb_slot']:.9f} ns RMS",
        "",
        "IF_deltaT at pytempo mjd_tt (trace TOAs):",
        *trace_lines,
        "",
        "=== Follow-on work (NOT Phase 2 IFTE scope) ===",
        f"  formbats component tt:    {report.component_rms_ns['tt']:.3f} ns RMS",
        f"  formbats component tdis1: {report.component_rms_ns['tdis1']:.3f} ns RMS",
        f"  formbats component roemer:{report.component_rms_ns['roemer']:.3f} ns RMS",
    ]
    Path("/tmp/jug_phase2_runtime_evidence.txt").write_text("\n".join(lines) + "\n")

    for key in ("tt_jax", "teph_jax", "tt_tb_jax", "native_tt_tb"):
        assert rms[key] < 1.0, f"{key} RMS {rms[key]:.6f} ns"
    assert rms["host_tt_tb_slot"] < 1.0
