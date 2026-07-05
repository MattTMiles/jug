# Tempo2 native clock / spin parity — not done

**Status:** work in progress. **Not at parity** with tempo2/libstempo under the
project’s strict ns-level gate (5 / 25 / 10 ns).

Policy and architecture: [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md).
Broader parity tracker: [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md).

---

## Parity review (2026-07-05) — primary report

This section records the evidence-based review of the IFTE / formBats / longdouble /
native ``phase5`` session. **Read this before investing more effort in the native
clock/spin stack.**

### Executive summary

The ~16 ns residual floor on failing fixtures (`wsrt167`, `epta_j0030_isolated`) is
**not** closed by the native clock/spin work. Binary models, Roemer, and site clocks
already match libstempo at the ns level on most fixtures. The remaining gap is a
**small number of outlier TOAs** whose cause is **not Roemer or site-clock mismatch**.

| Action | Verdict |
|--------|---------|
| Longdouble pass (`tempo2_clock.py`, `tempo2_spin.py`) | **Reverted** — bit-identical RMS (35.74 ns native; 16.43 ns production) |
| Native ``phase5`` + ``track_minus2_frac_phase`` | **Quarantined** — worse than production (~36 ns vs ~16 ns on wsrt167) |
| IFTE + formBats in `tempo2_clock.py` | **Diagnostic-only** — production spin uses geometry `model_mjd`, not formBats `model_clock` |
| Next parity work | Clock/Roemer harness + astrometry/TZR at early epochs (see below) |

### Measured fixture survey (production path, 2026-07-05)

| Fixture | N | RMS Δ | max \|Δ\| | binary | Gate |
|---------|---|-------|-----------|--------|------|
| epta_j1909_t2 | 27 | **3.2 ns** | 5.5 ns | T2 | pass |
| epta_j1918_ddh | 12 | **3.0 ns** | 7.7 ns | DDH | pass |
| ppta_j1902_ell1h | 120 | **2.3 ns** | 5.8 ns | ELL1H | pass |
| ng5_j1600 (both) | 625 | **4.1 ns** | 10 ns | DD | pass |
| ppta_j1741_ell1 | 111 | 5.8 ns | 12.7 ns | ELL1 | close |
| epta_j0613_t2_nrt1400 | 120 | 5.9 ns | 17.2 ns | T2 | close |
| **epta_j0030_isolated** | 10 | **15.9 ns** | 38 ns | none | **fail** (2 outlier TOAs) |
| **wsrt167** | 167 | **16.4 ns** | 110 ns | T2 | **fail** |
| epta_j0613_t2_ipta_all | 1369 | 36 ns | 720 ns | T2 | fail (clock-file extrapolation) |

**Binary models are fine.** T2, DD, DDH, ELL1H all sit at 2–4 ns. Do not chase binary
kernels for the ~16 ns floor.

### Longdouble pass — not necessary

Controlled experiment (native path, wsrt167):

```
NATIVE (longdouble clock) RMS: 35.73983864859414 ns
NATIVE (float64 clock)    RMS: 35.73983864859414 ns   ← bit-identical
PRODUCTION                RMS: 16.426632571201743 ns
```

Tempo2’s own ``phase5`` uses ``(int)F0`` / fractional-day decomposition in **double**
by design. Clock corrections are milliseconds; float64 is already sub-ps. The longdouble
pass added complexity with zero measurable benefit and has been **reverted**.

### Native phase5 stack — quarantined

Production (`USE_NATIVE_BBAT_PHASE5 = False` in
`jug/residuals/tempo2_native_quarantine.py`):

- Spin: emission-time Taylor at geometry **`model_mjd`** (TCB epoch map)
- TRACK −2: legacy `-pn_add` wrapping

Quarantined native path (`USE_NATIVE_BBAT_PHASE5 = True`):

- Spin: ``compute_tempo2_phase5`` at formBats ``bbat`` + ``track_minus2_frac_phase``
- **~36 ns RMS** on wsrt167 — **2× worse** than production

The old ~710 ms failure was wrong ``torb`` (`total − prebinary` with sign flip). With
``compute_tempo2_torb_sec`` the native path is functional but **not the route to <5 ns**.

**Do not enable the quarantined path for parity gates.**

### Where the ~16 ns actually lives

#### epta_j0030_isolated — two outlier TOAs, not Roemer/clock

| Metric | All 10 TOAs | Drop 2 worst TOAs |
|--------|-------------|-------------------|
| RMS Δ | **15.9 ns** | **1.97 ns** |

Outliers: indices 0–1, MJD **51275–51276 (1999)**; remaining 8 TOAs are 2008–2009,
same backend (`EFF.EBPP.1410`), ~1400 MHz. Solar elongation ~174° (anti-solar) — **not**
solar-wind/conjunction.

Per-TOA harness (`jug/testing/tempo2_outlier_diff.py`):

| Term | Outlier TOAs | Good TOAs |
|------|--------------|-----------|
| Roemer diff (JUG + libstempo) | ~8–10 ns | ~5–10 ns |
| Site arrival (sat vs stoas) | **0 ns** | **0 ns** |
| Residual diff | **±32–38 ns** | **<6 ns** |

**Conclusion:** outliers are **not** Roemer or site-clock dominated. Likely astrometry /
TZR / early-epoch clock-chain edge (1999 vs 2008–2009 span), not formBats or phase5.

#### wsrt167 — low-band scatter + a few large outliers

- 324–382 MHz; `BINARY T2`; RMS **16.4 ns**, max **110 ns**
- Weak correlation with `1/f²` (DM band); not a flat clock offset
- Roemer matches libstempo to **~0.8 ns RMS** (harness)
- Dropping 3 worst TOAs → **13.4 ns RMS**

### formBats diagnostic gap (does not drive production residuals)

JUG formBats ``bat_mjd`` differs from libstempo ``stoas + batCorrs`` by **~64 s**
(~10¹⁰ ns if reported naïvely). Production **does not use** formBats ``bat``/``bbat``
for spin — it uses geometry ``model_mjd``. This gap is real for the diagnostic path but
**off the production critical path**.

### Recommended path to <5 ns

**Priority 1 — early-epoch / astrometry (J0030 outliers)**

1. Run harness on 1999 TOAs vs 2008–2009 cohort:
   ```bash
   cd ref-packages/jug
   PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
     /opt/venvs/pta/bin/python tools/run_tempo2_outlier_clock_roemer_diff.py \
     tests/data_tempo2/epta_j0030_isolated/epta_j0030_isolated.par \
     tests/data_tempo2/epta_j0030_isolated/epta_j0030_isolated.tim \
     --fixture-id epta_j0030_isolated
   ```
2. Diff Roemer delay decomposition at MJD ≈ 51275: proper motion at POSEPOCH,
   parallax, equatorial vs ecliptic PM, TZR phase at `TZRMJD`.
3. Compare JUG geometric delay vs libstempo per TOA (not just scalar Roemer property).

**Priority 2 — wsrt167 outliers**

1. Same harness; identify TOAs with \|Δ\| > 30 ns.
2. Check DM / FD / `-padd` / TRACK −2 on low-band (324 MHz) TOAs.
3. Do **not** spend cycles on native phase5 or longdouble.

**Priority 3 — data coverage**

- `epta_j0613_t2_ipta_all`: **clock-file extrapolation** (BIPM ends MJD 56289, data to
  56795). Update clock files — not an algorithm bug.

**Defer**

- Native ``phase5`` / formBats production switch
- tt2tb `observatory_earth` Earth-rotation frame (static ITRF vs tempo2 SSB equatorial
  per-TOA vector) — secondary until outlier TOAs are closed

---

## Code layout (post-review)

| Module | Role |
|--------|------|
| `jug/residuals/tempo2_clock.py` | IFTE + formBats — **diagnostics only** (`term_diagnostics`) |
| `jug/residuals/tempo2_spin.py` | Quarantined ``phase5`` / TRACK −2 helpers |
| `jug/residuals/tempo2_native_quarantine.py` | `USE_NATIVE_BBAT_PHASE5 = False` |
| `jug/testing/tempo2_outlier_diff.py` | Per-TOA clock + Roemer diff harness |
| `tools/run_tempo2_outlier_clock_roemer_diff.py` | CLI for harness |

---

## Verification

```bash
cd ref-packages/jug

# Strict parity gates (expected fail on wsrt167 / j0030 until outliers fixed)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_residual_parity.py -q

# Outlier clock / Roemer harness (dev oracle)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_outlier_clock_roemer_diff.py -m dev_oracle -q

# CLI report
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python tools/run_tempo2_outlier_clock_roemer_diff.py \
  tests/data_tempo2/wsrt167/wsrt167.par tests/data_tempo2/wsrt167/wsrt167.tim \
  --fixture-id wsrt167 --outlier-threshold-ns 25
```

Strict gates should fail until outlier TOAs are closed (~16 ns → <5 ns).
