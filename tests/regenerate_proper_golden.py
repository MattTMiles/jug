#!/usr/bin/env python3
"""Regenerate tests/data_golden/J1909_proper_golden.json.

Recomputes JUG self-reference + live PINT and Tempo2 raw-error WRMS references.
Run after intentional precision changes OR when clock/ephemeris files update
(PINT/astropy re-download gps2utc/BIPM/observatory clocks over time, which shifts
both JUG and PINT residuals by ~tens of ns — the golden must track that drift).

Usage:  python tests/regenerate_proper_golden.py
"""
import json, sys, re, subprocess, logging
from pathlib import Path
import numpy as np
G = Path(__file__).parent / "data_golden"
PAR, TIM, OUT = G/"J1909_proper.par", G/"J1909_proper.tim", G/"J1909_proper_golden.json"
sys.path.insert(0, str(Path(__file__).parent.parent))
from jug.residuals.simple_calculator import compute_residuals_simple

jr = compute_residuals_simple(str(PAR), str(TIM), verbose=False)
jw = float(jr["weighted_rms_us"]); jres = np.array(jr["residuals_us"])

import pint.models, pint.toa, pint.residuals
logging.getLogger("pint").setLevel(logging.ERROR)
m = pint.models.get_model(str(PAR)); ephem = m.EPHEM.value if hasattr(m,"EPHEM") else None
t = pint.toa.get_TOAs(str(TIM), planets=True, ephem=ephem)
pres = pint.residuals.Residuals(t, m).time_resids.to("us").value
perr = t.get_errors().to("us").value
pw = float(np.sqrt(np.sum(pres**2/perr**2)/np.sum(1/perr**2)))
pint_max_ns = float(np.max(np.abs(jres*1e3 - pres*1e3)))

r = subprocess.run(["tempo2","-f",str(PAR),str(TIM),"-nofit","-output","general2","-s","{pre}\n"],
                   capture_output=True, text=True, timeout=120)
t2 = np.array([float(l) for l in r.stdout.splitlines() if re.match(r"^-?[0-9]",l.strip())])*1e6
raw=[]
for line in open(TIM):
    if line.startswith(("FORMAT","C ","#")): continue
    p=line.split()
    if len(p)>=4:
        try: raw.append(float(p[3]))
        except ValueError: pass
raw=np.array(raw); t2w=float(np.sqrt(np.sum(t2**2/raw**2)/np.sum(1/raw**2)))
t2_max_ns=float(np.max(np.abs(jres*1e3 - t2[:len(jres)]*1e3)))

golden = {
  "_comment": "Golden reference for J1909_proper (100 TOAs, MPTA DR3). JUG self-ref + PINT/Tempo2 raw-error parity.",
  "_generated": "python tests/regenerate_proper_golden.py",
  "_parity": f"JUG vs PINT raw WRMS {abs(jw-pw)/pw*100:.3f}% (max per-TOA {pint_max_ns:.1f} ns); "
             f"JUG vs Tempo2 {abs(jw-t2w)/t2w*100:.3f}% (max per-TOA {t2_max_ns:.1f} ns).",
  "_note": "Absolute residuals/WRMS drift ~tens of ns as PINT/astropy clock & ephemeris files update; "
           "the strict correctness guard is the <50 ns max per-TOA JUG-PINT/Tempo2 agreement, not the "
           "WRMS ppm. Tolerances are set accordingly. Regenerate if clock files change.",
  "n_toas": jr["n_toas"],
  "weighted_rms_us": jw,
  "weighted_rms_scaled_us": float(jr.get("weighted_rms_scaled_us", jw)),
  "unweighted_rms_us": float(jr["unweighted_rms_us"]),
  "first_5_residuals_ns": [round(float(x)*1e3,3) for x in jres[:5]],
  "pint_reference": {
    "_comment": "PINT raw-error WRMS (1/err^2 weights, no EFAC/EQUAD).",
    "raw_wrms_us": pw, "max_per_toa_diff_ns": pint_max_ns, "ephem": ephem,
  },
  "tempo2_reference": {
    "_comment": "Tempo2 raw-error WRMS: tempo2 -nofit pre-fit residuals, weights from raw .tim errors.",
    "raw_wrms_us": t2w, "max_per_toa_diff_ns": t2_max_ns,
  },
  "tolerances": {
    "_comment": "rms_rel_tol/residual_abs_tol_ns: JUG self-consistency, loose enough to survive clock-file "
                "updates. pint/tempo2_parity_rel_tol: WRMS-ratio guard; the real correctness check is the "
                "50 ns max per-TOA test (see test_pint_parity.py).",
    "rms_rel_tol": 1e-4,
    "residual_abs_tol_ns": 5.0,
    "pint_parity_rel_tol": 0.005,
    "tempo2_parity_rel_tol": 0.005,
  },
}
OUT.write_text(json.dumps(golden, indent=2))
print(f"wrote {OUT}")
print(f"  JUG  WRMS {jw:.6f} µs  | PINT {pw:.6f} ({abs(jw-pw)/pw*100:.3f}%, max {pint_max_ns:.1f} ns)"
      f" | T2 {t2w:.6f} ({abs(jw-t2w)/t2w*100:.3f}%, max {t2_max_ns:.1f} ns)")
