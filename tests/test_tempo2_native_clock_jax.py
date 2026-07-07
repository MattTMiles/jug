"""DEV ORACLE — JAX getCorrectionTT vs host Astropy and pytempo."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.tempo2_clock import compute_get_correction_tt_sec
from jug.residuals.tempo2_native.chain_jax import _load_model_static_for_native_chain
from jug.residuals.tempo2_native.clock_jax import compute_tempo2_get_correction_tt_jax
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    delta_ns,
    load_wsrt167_fixture,
)


def _wsrt167_clock_inputs():
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    static = _load_model_static_for_native_chain(params, toas, jug)
    sat = np.asarray(jug["term_diagnostics"]["sat_mjd"], dtype=np.float64)
    return fixture, sat, static


def test_jax_get_correction_tt_matches_host_astropy():
    """Astropy UTC→TT includes topocentric terms absent from tempo2 clkcorr.C."""
    fixture, sat, static = _wsrt167_clock_inputs()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    from jug.residuals.simple_calculator import _load_clock_corrections
    from jug.residuals.tempo2_native.chain_jax import _load_model_static_for_native_chain

    static = _load_model_static_for_native_chain(params, toas, jug)
    all_obs = sorted(set(t.observatory.lower() for t in toas))
    mjd_utc = np.array([t.mjd_int + t.mjd_frac for t in toas], dtype=np.float64)
    from pathlib import Path

    clock_dir = Path(__file__).resolve().parents[1] / "data" / "clock"
    clk = _load_clock_corrections(
        toas[0].observatory, all_obs, clock_dir, params, mjd_utc, verbose=False
    )
    host = compute_get_correction_tt_sec(
        toas,
        obs_clocks=clk["obs_clocks"],
        obs_clock_default=clk["obs_clock"],
        bipm_clock=clk["bipm_clock"],
        all_obs_codes=all_obs,
    )
    jax_tt = np.asarray(
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
    delta = delta_ns(jax_tt, host)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms > 100.0, "expected Astropy topocentric offset vs tempo2-native JAX clock"


def test_jax_get_correction_tt_matches_pytempo_wsrt167():
    fixture, sat, static = _wsrt167_clock_inputs()
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    jax_tt = np.asarray(
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
    delta = delta_ns(jax_tt, oracle.fields["correction_tt_sec"])
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0, f"JAX getCorrectionTT RMS {rms:.3f} ns vs pytempo"


def test_wsrt167_parity_probe_writes_report():
    """Write /tmp/jug_wsrt167_parity_probe.txt for sprint diagnostics."""
    from pathlib import Path

    fixture, _, _ = _wsrt167_clock_inputs()
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    native = compute_native_terms_for_fixture(fixture)
    tt = np.asarray(jax.device_get(native.correction_tt_sec), dtype=np.float64)
    tt_tb = np.asarray(jax.device_get(native.correction_tt_tb_sec), dtype=np.float64)
    roemer = np.asarray(jax.device_get(native.roemer_sec), dtype=np.float64)
    tdis1 = np.asarray(jax.device_get(native.tdis1_sec), dtype=np.float64)
    tdis2 = np.asarray(jax.device_get(native.tdis2_sec), dtype=np.float64)
    dt_ssb = np.asarray(jax.device_get(native.dt_ssb_sec), dtype=np.float64)
    from tempo2_native_test_helpers import native_batcorr_days

    batcorr = native_batcorr_days(native)

    def _rms(a, b, *, is_mjd=False):
        d = delta_ns(a, b, is_mjd=is_mjd)
        return float(np.sqrt(np.mean(d**2)))

    lines = [
        "wsrt167 JUG native vs pytempo parity probe",
        "",
        "Component RMS (ns):",
        f"  tt              {_rms(tt, oracle.fields['correction_tt_sec']):.6f}",
        f"  tt_tb           {_rms(tt_tb, oracle.fields['correction_tt_tb_sec']):.6f}",
        f"  roemer          {_rms(roemer, oracle.fields['roemer_sec']):.6f}",
        f"  tdis1           {_rms(tdis1, oracle.fields['tdis1_sec']):.6f}",
        f"  tdis2           {_rms(tdis2, oracle.fields['tdis2_sec']):.6f}",
        f"  dt_ssb          {_rms(dt_ssb, oracle.fields['dt_ssb_sec']):.6f}",
        f"  bat_corr        {_rms(batcorr, oracle.fields['bat_corr_days'], is_mjd=True):.6f}",
    ]
    Path("/tmp/jug_wsrt167_parity_probe.txt").write_text("\n".join(lines) + "\n")


def test_clock_tt_probe_writes_report():
    """Write /tmp/jug_clock_tt_probe.txt — stepwise tt_tb vs pytempo IFTE diagnostics."""
    from pathlib import Path

    from jug.residuals.tempo2_clock import compute_correction_tt_tb_sec
    from jug.residuals.tempo2_native.clock_jax import compute_tempo2_correction_tt_tb_jax
    from jug.utils.constants import C_KM_S, SECS_PER_DAY
    from jug.utils.ifteph import IFTE_LC, IFTE_TEPH0_SEC, ifte_delta_t_mjd, load_ifte_coeff_tables
    from jug.utils.timescales import IFTE_K, is_tempo2_si_units, parse_timescale

    fixture, sat, static = _wsrt167_clock_inputs()
    params = parse_par_file(fixture["par_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
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
    ifte_delta = np.asarray(ifte_delta_t_mjd(mjd_tt), dtype=np.float64)
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
    tt_tb_host, teph_host = compute_correction_tt_tb_sec(
        mjd_tt,
        observatory_earth_km=obs_earth,
        earth_ssb_vel_km_s=earth_vel,
        params=params,
    )

    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(
        parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False
    )
    diag = psr.toa_diagnostics(removemean=False)

    def _rms(a, b):
        return float(np.sqrt(np.mean(((np.asarray(a) - np.asarray(b)) * 1e9) ** 2)))

    trace = [0, 42, 85, 166]
    lines = [
        "wsrt167 tt_tb step probe",
        f"psr_units={diag.get('psr_units')} psr_time_ephemeris={diag.get('psr_time_ephemeris')}",
        f"parse_timescale={units} si={is_tempo2_si_units(units)}",
        "",
        "RMS (ns):",
        f"  teph_jax vs pytempo  {_rms(teph_jax, diag['correction_tt_teph_sec']):.6f}",
        f"  teph_host vs pytempo {_rms(teph_host, diag['correction_tt_teph_sec']):.6f}",
        f"  tt_tb_jax vs pytempo  {_rms(tt_tb_jax, diag['correction_tt_tb_sec']):.6f}",
        f"  tt_tb_host vs pytempo {_rms(tt_tb_host, diag['correction_tt_tb_sec']):.6f}",
        f"  tt_tb_jax vs host     {_rms(tt_tb_jax, tt_tb_host):.6f}",
        "",
    ]
    for i in trace:
        obs_raw = float(diag["obs_term_raw_sec"][i])
        obs_scaled = obs_raw / IFTE_K
        obs_jug = float(
            np.dot(obs_earth[i], earth_vel[i]) / (C_KM_S**2) / (1.0 - IFTE_LC) / IFTE_K
        )
        tt_pt = float(diag["correction_tt_sec"][i])
        mjd_tt_pt = float(sat[i] + tt_pt / SECS_PER_DAY)
        delta_at_pt_mjd = float(ifte_delta_t_mjd(mjd_tt_pt))
        delta_implied_pt = (
            float(diag["correction_tt_teph_sec"][i]) - IFTE_TEPH0_SEC - obs_scaled
        ) * (1.0 - IFTE_LC)
        lines.extend(
            [
                f"=== TOA {i} ===",
                f"  tt_jax={tt_jax[i]:.12f} s  tt_pytempo={tt_pt:.12f} s",
                f"  ifte_delta_host={ifte_delta[i]:.12e} at_pt_mjd={delta_at_pt_mjd:.12e} "
                f"implied_pt={delta_implied_pt:.12e} s",
                f"  pytempo correction_tt_teph={diag['correction_tt_teph_sec'][i]:.12e} s",
                f"  JUG teph_jax={teph_jax[i]:.12e} host={teph_host[i]:.12e} s",
                f"  pytempo obs_term_raw={obs_raw:.12e} /IFTE_K={obs_scaled:.12e}",
                f"  JUG obs_term={obs_jug:.12e} s",
                f"  expected teph={IFTE_TEPH0_SEC + obs_scaled + ifte_delta[i] / (1 - IFTE_LC):.12e}",
                f"  pytempo tt_tb={diag['correction_tt_tb_sec'][i]:.12e} s",
                f"  JUG tt_tb={tt_tb_jax[i]:.12e} host={tt_tb_host[i]:.12e} s",
                f"  tt_tb delta ns vs pytempo={(tt_tb_jax[i] - diag['correction_tt_tb_sec'][i]) * 1e9:.3f}",
                "",
            ]
        )
    Path("/tmp/jug_clock_tt_probe.txt").write_text("\n".join(lines))


def test_wsrt167_host_tt_tb_component_gate():
    """Host ``correction_tt_tb_sec`` export matches pytempo after IFTE fix."""
    from jug.testing.tempo2_formbats_closure import compare_formbats_components

    fixture, _, _ = _wsrt167_clock_inputs()
    report = compare_formbats_components(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    assert report.component_rms_ns["tt_tb"] < 1.0
