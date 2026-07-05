# Tempo2 native clock / spin parity — not done

**Status:** work in progress. **Not at parity** with tempo2/libstempo under the
project’s strict ns-level gate (5 / 25 / 10 ns) on **wsrt167** (~16 ns production).

Policy and architecture: [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md).
Broader parity tracker: [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md).

**Where we are (2026-07-05):**

| Fix | Fixture | Status |
|-----|---------|--------|
| **#1 Phase C — TZR** | `epta_j0030_isolated` | **Done** — 15.9 → **~4.7 ns RMS** |
| **#2 Phase D Step 1 — pnNew** | `wsrt167` | **Done** — relative ``-pn`` convention; tests added |
| **#2 Phase D Step 2 — bbat spin** | `wsrt167` | **Open** — production still ~16 ns |

---

## Parity review (2026-07-05) — primary report

This section records the evidence-based review of the IFTE / formBats / longdouble /
native ``phase5`` session. **Read this before investing more effort in the native
clock/spin stack.**

### Executive summary

The ~16 ns residual floor on **wsrt167** is **not** closed by toggling the quarantined
native stack. Binary models, Roemer, and site clocks match libstempo at the ns level on
most fixtures. **epta_j0030_isolated** is largely closed by Phase C TZR (~4.7 ns RMS).

| Action | Verdict |
|--------|---------|
| Longdouble pass (`tempo2_clock.py`, `tempo2_spin.py`) | **Reverted** — bit-identical RMS (35.74 ns native; 16.43 ns production) |
| Native ``phase5`` + formBats ``bbat`` | **Quarantined** — ~36 ns; formBats ``bbat`` **~65 s** off pytempo |
| IFTE + formBats in `tempo2_clock.py` | **Diagnostic-only** — production spin uses geometry `model_mjd` |
| Phase C TZR (`tzr_geometry.py`) | **Done** — J0030 passes strict RMS gate |
| Phase D Step 1 pnNew | **Done** — ``pnAct = (pn[i]−pn[0]) + pnAdd``; oracle tests |
| Next parity work | Phase D Step 2: oracle ``bbat`` for production spin on wsrt167 |

### Measured fixture survey (production path, 2026-07-05)

| Fixture | N | RMS Δ | max \|Δ\| | binary | Gate |
|---------|---|-------|-----------|--------|------|
| epta_j1909_t2 | 27 | **3.2 ns** | 5.5 ns | T2 | pass |
| epta_j1918_ddh | 12 | **3.0 ns** | 7.7 ns | DDH | pass |
| ppta_j1902_ell1h | 120 | **2.3 ns** | 5.8 ns | ELL1H | pass |
| ng5_j1600 (both) | 625 | **4.1 ns** | 10 ns | DD | pass |
| ppta_j1741_ell1 | 111 | 5.8 ns | 12.7 ns | ELL1 | close |
| epta_j0613_t2_nrt1400 | 120 | 5.9 ns | 17.2 ns | T2 | close |
| **epta_j0030_isolated** | 10 | **~4.7 ns** | ~11 ns | none | **pass RMS** (Phase C) |
| **wsrt167** | 167 | **16.4 ns** | 110 ns | T2 | **fail** |
| epta_j0613_t2_ipta_all | 1369 | 36 ns | 720 ns | T2 | fail (clock-file extrapolation) |

**Binary models are fine.** T2, DD, DDH, ELL1H all sit at 2–4 ns. Do not chase binary
kernels for the wsrt167 ~16 ns floor.

### Longdouble pass — not necessary

Controlled experiment (native path, wsrt167):

```
NATIVE (longdouble clock) RMS: 35.73983864859414 ns
NATIVE (float64 clock)    RMS: 35.73983864859414 ns   ← bit-identical
PRODUCTION                RMS: 16.426632571201743 ns
```

The longdouble pass has been **reverted**.

### Native phase5 stack — quarantined

Production (`USE_NATIVE_BBAT_PHASE5 = False`):

- Spin: emission-time Taylor at geometry **`model_mjd`**
- TRACK −2: legacy ``−pnAdd`` wrap (equivalent to fixed pnNew on wsrt167)

Quarantined path (`USE_NATIVE_BBAT_PHASE5 = True` + formBats ``bbat``):

- **~36 ns RMS** on wsrt167 — **2× worse** than production

With **pytempo ``bbat``** + fixed pnNew + ``compute_tempo2_phase5``: **~17 ns RMS**
(validated; not wired to production). **Do not enable the quarantine flag.**

### Where the remaining gap lives

#### epta_j0030_isolated — Phase C TZR (done)

| Metric | Before | After Phase C |
|--------|--------|---------------|
| RMS Δ | **15.9 ns** | **~4.7 ns** |
| max \|Δ\| | ~38 ns | **~11 ns** (2×1999 TOAs) |

Roemer and site clock ruled out pre-fix. Remaining ~11 ns on two 1999 TOAs may be
early-epoch astrometry (separate from wsrt167).

#### wsrt167 — TRACK −2 / spin at bbat (Phase D)

- 324–382 MHz; `BINARY T2`; `TRACK -2`; ``-pn`` on all 167 TOAs
- Production RMS **16.4 ns**, max **110 ns** (TOA 85)
- Roemer matches libstempo to **~0.8 ns RMS** (harness)
- Phase D Step 1: pnNew convention fixed; ``phase5@pytempo bbat`` validates ``nphase``
- Phase D Step 2: wire oracle ``bbat`` into production — **not started**

### formBats diagnostic gap

JUG formBats ``bbat_mjd`` differs from pytempo/libstempo by **~65 s RMS** on wsrt167.
Production **does not use** formBats ``bbat`` for spin. Do not enable
``USE_NATIVE_BBAT_PHASE5`` until this gap is closed.

---

## Phase C — TZR (fix #1, done)

See [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) § "Phase C — TZR reference phase".

- Module: ``jug/residuals/tzr_geometry.py``
- Tests: ``tests/test_tempo2_tzr_parity.py``

---

## Phase D — wsrt167 TRACK −2 (fix #2)

**Why:** After Phase C, wsrt167 remains at **~16 ns RMS** / **~110 ns max**.

**Step 1 (done): pnNew / tim ``-pn`` convention**

| Issue | Resolution |
|-------|------------|
| ``track_minus2_frac_phase`` ``addPhase ~ 10¹⁰`` turns | ``pnAct = (pn[i]−pn[0]) + pnAdd`` |
| Identity | ``pn[i] − pn[0] == pnNew`` (exact on wsrt167) |
| Fixed pnNew + ``phase5@pytempo bbat`` | **~17 ns** RMS — not <5 ns yet |
| Production Taylor + legacy | **~16 ns** — still best until Step 2 |

**Step 2 (open): production spin at correct ``bbat``**

1. Port or substitute **oracle-correct ``bbat``** (keep delays at ``model_mjd``).
2. Wire TRACK −2 + all ``-pn`` to ``phase5`` + fixed ``track_minus2_frac_phase``.
3. Re-run ``tests/test_dev_oracle_wsrt167_parity.py`` toward <5 ns.

**Tests / harness**

```bash
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_track2_pnnew.py -q
```

---

## Recommended path to <5 ns (updated)

**Priority 1 — wsrt167 (Phase D Step 2)**

1. Close formBats / ``bbat`` gap vs pytempo (~65 s) or bypass with validated clock chain.
2. Wire ``phase5`` + fixed ``track_minus2_frac_phase`` for TRACK −2 in production.
3. Do **not** enable ``USE_NATIVE_BBAT_PHASE5`` wholesale.

**Priority 2 — J0030 polish**

- Two 1999 TOAs still ~11 ns after Phase C; investigate early-epoch astrometry if p99
  gate matters.

**Priority 3 — data coverage**

- `epta_j0613_t2_ipta_all`: clock-file extrapolation — update BIPM data, not algorithm.

**Defer**

- Native ``phase5`` / formBats production switch until ``bbat`` matches oracle
- tt2tb Earth-rotation frame refinement — secondary

---

## Code layout (post-review)

| Module | Role |
|--------|------|
| `jug/residuals/tempo2_clock.py` | IFTE + formBats — **diagnostics only** |
| `jug/residuals/tzr_geometry.py` | TZR apply modes (Phase C) |
| `jug/residuals/tempo2_spin.py` | ``phase5`` / TRACK −2; pnAct relative to obsn[0] (Phase D) |
| `jug/residuals/tempo2_native_quarantine.py` | `USE_NATIVE_BBAT_PHASE5 = False` |
| `jug/testing/tempo2_outlier_diff.py` | Per-TOA clock + Roemer diff harness |
| `jug/testing/tempo2_track2_oracle.py` | TRACK −2 pnNew oracle (Phase D) |
| `tools/run_tempo2_outlier_clock_roemer_diff.py` | CLI for harness |

---

## Verification

```bash
cd ref-packages/jug

# Strict parity gates
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_residual_parity.py -q

# Phase C — TZR
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_tzr_parity.py -q

# Phase D Step 1 — TRACK −2 pnNew
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_track2_pnnew.py -q

# Outlier clock / Roemer harness
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_outlier_clock_roemer_diff.py -m dev_oracle -q

# wsrt167 acceptance (still failing strict gate)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```

Strict gates on wsrt167 should fail until Phase D Step 2 closes ~16 ns → <5 ns.
