"""DEV ORACLE — Roemer subterm trace probe (writes /tmp/jug_roemer_term_probe.txt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.probe]

import jax

from jug.delays.tempo2_geometry import (
    Tempo2ObservatoryState,
    build_tempo2_pulsar_vectors,
    pmrv_rad_per_century,
    tempo2_observatory_chain_vectors,
)
from jug.io.par_reader import parse_par_file
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.constants import C_KM_S, SECS_PER_DAY
from tempo2_test_helpers import compute_native_terms_for_fixture, load_wsrt167_fixture, rms_cm, rms_ns

TRACE = [0, 42, 85, 166]


def _roemer_subterms(rca_ls, pos, vel, acc, delt, parallax_mas, pmrv):
    from jug.delays.tempo2_geometry import AULTSC, PX_CONV

    rcos1 = float(np.dot(pos, rca_ls))
    rr = float(np.dot(rca_ls, rca_ls))
    pmtrans_rcos2 = float(np.dot(vel, rca_ls))
    pmtrans = float(np.linalg.norm(vel))
    dt_pm = delt * pmtrans_rcos2
    dt_pmtt = -0.5 * pmtrans * pmtrans * delt * delt * rcos1
    dt_acctrans = 0.5 * delt * delt * float(np.dot(acc, rca_ls))
    dt_px = 0.0
    if parallax_mas != 0.0:
        dt_px = -0.5 * parallax_mas * PX_CONV * (rr - rcos1 * rcos1) / AULTSC
    dt_pmtr = -delt * delt * pmrv * pmtrans_rcos2
    roemer_ls = rcos1 + dt_pm + dt_pmtt + dt_px + dt_pmtr + dt_acctrans
    return {
        "rcos1_ls": rcos1,
        "dt_pm_ls": dt_pm,
        "dt_pmtt_ls": dt_pmtt,
        "dt_px_ls": dt_px,
        "dt_pmtr_ls": dt_pmtr,
        "dt_acctrans_ls": dt_acctrans,
        "roemer_ls": roemer_ls,
    }


def test_roemer_term_probe_writes_report():
    """Per-TOA Roemer trace for wsrt167 — explains rca vs roemer gap."""
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    pos, vel, acc = build_tempo2_pulsar_vectors(params, use_native_ecliptic=False)
    pmrv = pmrv_rad_per_century(float(params.get("PMRV", 0.0)))
    px = float(params.get("PX", 0.0))
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))

    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )
    td = jug["term_diagnostics"]
    obs_state = td["tempo2_obs_state"]
    state = Tempo2ObservatoryState(
        earth_ssb_km=np.asarray(obs_state["earth_ssb_km"], dtype=np.float64),
        observatory_earth_km=np.asarray(obs_state["observatory_earth_km"], dtype=np.float64),
        sun_ssb_km=np.asarray(obs_state["sun_ssb_km"], dtype=np.float64),
        planet_ssb_km={
            k: np.asarray(v, dtype=np.float64) for k, v in obs_state["planet_ssb_km"].items()
        },
        site_vel_km_s=np.asarray(obs_state["site_vel_km_s"], dtype=np.float64),
    )
    _, jug_rca_ls, _, _ = tempo2_observatory_chain_vectors(state)

    native = compute_native_terms_for_fixture(fixture)
    jug_roemer = np.asarray(jax.device_get(native.roemer_sec), dtype=np.float64)

    from pytempo.sandbox import tempopulsar

    psr = tempopulsar(
        parfile=str(fixture["par_path"]), timfile=str(fixture["tim_path"]), dofit=False
    )
    diag = psr.toa_diagnostics(removemean=False)
    pt_roemer = np.asarray(diag["roemer_sec"], dtype=np.float64)
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt = np.asarray(jax.device_get(native.correction_tt_sec), dtype=np.float64)
    tt_tb = np.asarray(jax.device_get(native.correction_tt_tb_sec), dtype=np.float64)
    dt_ssb = np.asarray(jax.device_get(native.dt_ssb_sec), dtype=np.float64)

    pt_earth = np.asarray(psr.earth_ssb, dtype=np.float64)
    pt_obs = np.asarray(psr.observatory_earth, dtype=np.float64)
    pt_rca = pt_earth[:, :3] + pt_obs[:, :3]

    lines = [
        "wsrt167 Roemer term probe",
        f"pos_pulsar={pos.tolist()}",
        f"vel_pulsar={vel.tolist()}",
        f"acc_pulsar={acc.tolist()}",
        f"IFTE_K ephemeris scale={float(__import__('jug.utils.timescales', fromlist=['IFTE_K']).IFTE_K)}",
        "",
    ]
    for i in TRACE:
        delt = (sat[i] - posepoch + (tt[i] + tt_tb[i] + dt_ssb[i]) / SECS_PER_DAY) / 36525.0
        jug_terms = _roemer_subterms(jug_rca_ls[i], pos, vel, acc, delt, px, pmrv)
        pt_terms = _roemer_subterms(pt_rca[i], pos, vel, acc, delt, px, pmrv)
        rca_delta_ls = jug_rca_ls[i] - pt_rca[i]
        lines.extend(
            [
                f"=== TOA index {i} sat={sat[i]:.12f} ===",
                f"  delt_centuries={delt:.12e}",
                f"  pos_pulsar={pos.tolist()}",
                f"  vel_pulsar={vel.tolist()}",
                f"  acc_pulsar={acc.tolist()}",
                f"  JUG rca_ls={jug_rca_ls[i].tolist()} norm={np.linalg.norm(jug_rca_ls[i]):.12e} "
                f"dot(pos,rca)={jug_terms['rcos1_ls']:.12e}",
                f"  PT  rca_ls={pt_rca[i].tolist()} norm={np.linalg.norm(pt_rca[i]):.12e} "
                f"dot(pos,rca)={pt_terms['rcos1_ls']:.12e}",
                f"  rca delta norm (ls)={np.linalg.norm(rca_delta_ls):.6e}",
                f"  rca delta norm (cm)={np.linalg.norm(rca_delta_ls) * C_KM_S * 100:.3f}",
                f"  JUG roemer subterms: {jug_terms}",
                f"  PT  roemer subterms: {pt_terms}",
                f"  JUG native roemer_sec={jug_roemer[i]:.12e} PT roemer={pt_roemer[i]:.12e}",
                f"  delta_ns={(jug_roemer[i] - pt_roemer[i]) * 1e9:.3f}",
                "",
            ]
        )
    jug_earth_ls = np.asarray(obs_state["earth_ssb_km"], dtype=np.float64)[:, :3] / C_KM_S
    jug_obs_ls = np.asarray(obs_state["observatory_earth_km"], dtype=np.float64)[:, :3] / C_KM_S
    pt_earth = np.asarray(psr.earth_ssb[:, :3], dtype=np.float64)
    pt_obs = np.asarray(psr.observatory_earth[:, :3], dtype=np.float64)
    pt_dt_ssb = np.asarray(diag.get("dt_ssb_sec", []), dtype=np.float64)
    lines.extend(
        [
            "=== decomposition RMS vs pytempo ===",
            f"earth_ssb_cm={rms_cm(jug_earth_ls, pt_earth):.4f}",
            f"observatory_earth_cm={rms_cm(jug_obs_ls, pt_obs):.4f}",
            f"rca_cm={rms_cm(jug_earth_ls + jug_obs_ls, pt_earth + pt_obs):.4f}",
            f"roemer_ns={rms_ns(jug_roemer, pt_roemer):.4f}",
            f"dt_ssb_ns={rms_ns(dt_ssb, pt_dt_ssb):.4f}"
            if pt_dt_ssb.size
            else "dt_ssb_ns=(not in diag)",
            "",
        ]
    )
    Path("/tmp/jug_roemer_term_probe.txt").write_text("\n".join(lines))
