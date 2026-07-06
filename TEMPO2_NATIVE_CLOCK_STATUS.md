# Tempo2 native clock / spin parity — not done

**Status:** work in progress. **Not at parity** with tempo2/libstempo under the
project’s strict ns-level gate (5 / 25 / 10 ns) on **wsrt167** (~16 ns production).

Policy and architecture: [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md).
Broader parity tracker: [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md).

**Where we are (2026-07-06):**

| Fix | Fixture | Status |
|-----|---------|--------|
| **#1 Phase C — TZR** | `epta_j0030_isolated` | **Done** — 15.9 → **~4.7 ns RMS** |
| **#2 Phase D Step 1 — pnNew** | `wsrt167` | **Done** — relative ``-pn`` convention; tests added |
| **#2 Phase D Step 2 — wire ``phase5@bbat``** | `wsrt167` | **Ruled out** — ~17.5 ns vs production ~16.4 ns |
| **#2 Phase D Step 3 — ``-padd`` / ``jump_phase``** | `wsrt167` | **Ruled out** — JUG ``jump_phase`` exact vs pytempo |
| **#2 Phase D Steps 16–18** | `wsrt167` | **Done (investigation)** — ~16 ns = best JUG Taylor spin; 0 ns needs JAX chain |
| **Next** | MetaPulsar JAX | **JAX tempo2-native clock/delay chain** (see § below) |

---

## JAX tempo2-native clock/delay chain (planned — required for 0 ns)

End-to-end JAX matching libstempo/tempo2: ``calculate_bclt.C`` → ``formBats.C`` →
``formResiduals.C`` on shared in-flight epochs. **Not** the JUG production shortcut
``IFTE(tdb) model_mjd`` + ``(model−sat)×86400 − prebin``.

| Component | JUG production | tempo2 native | Investigation gap |
|-----------|---------------|---------------|-------------------|
| Delays / ``batCorr`` | IFTE ``model_mjd`` + ``prebinary`` | ``TT+TT_TB+tdis`` via ``formBats`` | **~286 ns** (Step 17) |
| Spin | Taylor@``dt_sec_ld`` + legacy TRACK −2 | ``phase2+phase3`` @ ``bbat`` + ``pnNew`` | **~16 ns** best JUG (Step 18) |
| Oracle | — | pytempo ``acceptance_residual_sec`` | **0 ns** |

**Scope:** iterative ``calculate_bclt`` Roemer epoch; longdouble until export; tempo2
``phase5`` + ``track_minus2_frac_phase``; derive ``torb`` from ``dt`` closure (never raw
pt ``torb_sec`` in JUG-composed spin — **172 ns** trap, Step 18).

**Status:** temp probes promoted to `jug/residuals/tempo2_native/` and `jug/testing/`.
`USE_JAX_TEMPO2_NATIVE_CHAIN = False` until Phase 4 gates pass.

**NumPy reference deprecation:** `chain_numpy.py` is dev-only, emits `DeprecationWarning`,
and will be env-gated (`JUG_DEV_NUMPY_TEMPO2_CHAIN=1`) or deleted when native gates are green.

Full spec:
[`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) § "JAX tempo2-native clock/delay chain".

**Do not implement in JUG production now:** ``phase5@bbat`` wiring (17.4 ns), ``tempo2_clock``
``bbat`` diagnostic as spin input (~65 s offset).

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
| Phase D Step 2 ``phase5@bbat`` wiring | **Ruled out** — ~17.5 ns vs production ~16.4 ns |
| Phase D Step 3 ``-padd`` / ``jump_phase`` | **Ruled out** — exact match to pytempo ``phase_offset_turns`` |
| Next parity work | formBats.C signed-term parity vs pytempo (Step 10); production spin stays ``model_mjd`` + ``dt_sec_ld`` |
| idx 85 (+110 ns max) | **Red herring** as isolated bug — pytempo exact; tail of spin error |

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

With **pytempo ``bbat``** + fixed pnNew + ``compute_tempo2_phase5``: **~17.5 ns RMS**
(validated; **worse than production**). **Do not enable the quarantine flag** or wire
``phase5@bbat`` to production for wsrt167.

**Oracle ``bbat`` from JUG geometry:** ``compute_tempo2_bbat_mjd(model_mjd, prebinary)``
= ``model_mjd − prebinary/86400`` — **~330 ns RMS** vs pytempo ``bbat`` on wsrt167 (Step 5).
formBats diagnostic ``bbat_mjd`` in ``tempo2_clock.py`` remains **~65 s** off because it bundles
``prebinary`` instead of tempo2's per-term ``formBats.C`` signs.

#### epta_j0030_isolated — Phase C TZR (done)

| Metric | Before | After Phase C |
|--------|--------|---------------|
| RMS Δ | **15.9 ns** | **~4.7 ns** |
| max \|Δ\| | ~38 ns | **~11 ns** (2×1999 TOAs) |

Roemer and site clock ruled out pre-fix. Remaining ~11 ns on two 1999 TOAs may be
early-epoch astrometry (separate from wsrt167).

#### wsrt167 — TRACK −2 / spin (Phase D)

- 324–382 MHz; `BINARY T2`; `TRACK -2`; ``-pn`` on all 167 TOAs
- Production RMS **16.4 ns**, max **110 ns** at idx 85 (debt-pin metric; tail of spin error)
- Roemer matches libstempo to **~0.8 ns RMS** (harness)
- Phase D Step 1: pnNew convention fixed
- Phase D Step 2: ``phase5@oracle bbat`` **ruled out** (~17.5 ns — worse than production)
- Phase D Step 4: Taylor vs tempo2 ``phase2+phase3`` **ruled out** (0.02 ns fractional)
- Phase D Step 5: per-TOA term diff — oracle ``bbat`` **~330 ns RMS** vs pytempo (r≈0.19);
  roemer/shapiro/sat closed; ``model_mjd − (sat+bc+prebin/86400)`` **~380 ns**
- Phase D Step 6: ``pt_torb ≈ prebin−total`` (0.17 ns); ``dt−deltaT(pt)`` ~207 ns (r≈0.18)
- Phase D Step 7: Taylor Horner is ld but **float64 ``model_mjd``/``total_delay`` inputs** cap
  ``dt_sec`` (~185 ns vs fresh ld); **Taylor@``deltaT(pt)`` → ~173 ns (worse than 16 ns)**
- Phase D Step 8: JAX two-part dt + compensated Taylor — **no gain** (Horner ruled out again)
- Phase D Step 9: production ``IFTE(tdb_ld)`` + ``dt_sec_ld`` best (**16.4 ns**); naive epoch
  counterfactuals worsen; ~330 ns ``bbat`` gap is not fixable by relabelling alone
- Phase D Step 10: formBats signed-term probe — **+65 s** JUG TT/batCorr baseline mismatch;
  inverted ``tdis ≈ −64 s`` vs ``dm+sw ≈ +1.3 s``; ``tempo2_clock.py`` bundled prebinary wrong
- Phase D Step 11: TT_TB inversion with ``dm+sw`` **confounded** (+65 s ≈ ``utc_to_tdb``); true
  ``correction_tt_tb`` ~14 s matches ``tt2tdb.C``
- Phase D Step 12: ``batCorrs = (model_mjd−sat)×86400 − prebinary`` (**286 ns**); naive formBats
  ``tt+tt_tb+dm+sw`` misses ``utc_to_tdb`` chain
- Phase D Step 13: model-epoch batCorr/bbat **temp prototype** closes libstempo (**286 ns**);
  bundled formBats still **~65 s** wrong; **~330 ns ``bbat``** vs pytempo unchanged
- **Open:** ``torb`` / ``formResiduals`` / in-flight ``model_mjd_ld`` (Step 14)
- **Red herring:** idx 85 pnNew / ``nphase`` ladder — pytempo exact at idx 85

### formBats diagnostic gap

JUG formBats ``bbat_mjd`` in ``tempo2_clock.py`` differs from pytempo by **~65 s RMS** on
wsrt167 (bundled ``tt+tt_tb−prebinary``). **Step 13 temp prototype** rebuilds from production
``model_mjd`` and closes libstempo ``batCorrs`` at **286 ns**:

```text
batCorr_sec = (model_mjd − sat)×86400 − prebinary_delay_sec
bat_mjd     = model_mjd − prebinary/86400
bbat_mjd    = bat_mjd − shklovskii/86400
```

**~330 ns ``bbat``** vs pytempo remains after Step 13 — separate from the 65 s diagnostic fix.
Step 14 closed ``torb`` at **0.17 ns**. Step 15 decomposed the ~330 ns gap into **~286 ns
``batCorr`` model-chain error** + **~304 ns Tier-1 float64 ``bat`` read** (not separate physics).
Step 16 ld replay: **286 ns does not collapse** — best ld path **214 ns** (~72 ns float64
arithmetic gain); **~214 ns persists** as JUG IFTE ``model_mjd`` vs tempo2 ``formBats`` epoch.
Step 17: formBats replay **0 ns**; ~286 ns = IFTE scatter (~260 ns) + anticorrelated
clock/delay slot cancellation — **not** Roemer magnitude or sign errors.

### pytempo / libstempo oracle tiers (tempo2 2026.4.1)

Both editable pytempo (``./ref-packages/pytempo``) and installed libstempo link to
``/opt/software/tempo2/install/lib/libtempo2.so.2``. Core acceptance path agrees at 0 ns.

| Tier | Safe for parity | Trap |
|------|-----------------|------|
| **1** | ``psr.residuals()``, ``acceptance_residual_sec``, ``bbat_mjd``, ``batCorrs``, ``pets`` | — |
| **2** | ``phase_offset_turns`` | ``residual_sec`` on TRACK −2 |
| **3** | informational | ``nphase`` vs ``pulse_number``; ``binarydelay`` stale on fresh construct; ``toas`` not ``bbat`` |

Naïve float64 recompositions ``sat + bat_corr → bat`` (~237–304 ns) or
``bbat − torb/86400 → pet`` (~275 ns) are export artifacts — tempo2 ``long double``
identities hold at ~10⁻¹² day. libstempo ``binarydelay`` reads **zeros** on fresh
construct — use ``torb_sec`` or ``prebinary − total`` (0.17 ns on wsrt167).

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
| Fixed pnNew + ``phase5@oracle bbat`` | **~17.5 ns** RMS — worse than production |
| Production Taylor + legacy | **~16.4 ns** — best JUG-composed path |

**Step 2 (ruled out, 2026-07-06): wire ``phase5@bbat`` to production**

Temp-only path ranking: production Taylor beats oracle ``phase5`` (~17.5 ns). Do **not**
enable ``USE_NATIVE_BBAT_PHASE5`` for parity gates.

**Next (open):**

1. **JAX tempo2-native pipeline** — implement ``calculate_bclt`` + ``formBats`` +
   ``formResiduals`` in JAX (see § above and ``TEMPO2_PARITY.md``).
2. Keep production spin at ``IFTE(tdb_ld)`` + ``dt_sec_ld`` + Taylor + legacy TRACK −2.

**Step 18 summary (2026-07-06):**

| Path | RMS ns |
|------|--------|
| pytempo acceptance | **0.00** |
| **Production Taylor + legacy TRACK −2** | **16.43** |
| ``phase5@pt bbat`` + ``torb(dt)`` + track2 | **17.43** |
| ``phase5@pt bbat`` + pt ``torb`` + track2 | **172.41** (trap) |

**Implement in JUG now?** **No.** ~16 ns is the best JUG-composed spin path; 0 ns needs
the JAX native chain. Never pair pytempo ``torb_sec`` in JUG ``phase5``.

Probe: ``/tmp/wsrt_taylor_formresiduals_step18_probe.py``

**Step 17 summary (2026-07-06):**

| Check | Result |
|-------|--------|
| formBats replay vs lib ``batCorrs`` | **0 ns** |
| ``(model−sat)×86400 − prebin`` vs lib | **286 ns** |
| IFTE per-TOA scatter | **260 ns** (``r≈−0.57`` with gap) |
| Roemer JUG vs ``−``pytempo | **0.8 ns** |
| ``tempo2_clock`` ``bbat`` vs pt | **~65 s** offset — do not wire |

**Implement in JUG now?** **No production changes.** Optional: export ``batcorr_model_sec``
diagnostic. Defer ``tempo2_clock`` ``bbat`` fix. JAX pipeline needs native formBats chain.

Probe: ``/tmp/wsrt_formbats_roemer_epoch_step17_probe.py``

**Step 16 summary (2026-07-06):**

| Check | RMS |
|-------|-----|
| ``batCorr`` f64 vs lib | **286 ns** |
| ``batCorr`` ld IFTE(``tdb_ld``) vs lib | **214 ns** (partial — no collapse) |
| ``model_f64`` vs IFTE(``tdb_ld``) | **0 ns** |
| Production spin | **16.43 ns** (unchanged) |

Probe: ``/tmp/wsrt_model_bbat_step16_probe.py``

**Step 15 summary (2026-07-06):**

| Link | RMS | Verdict |
|------|-----|---------|
| oracle ``model−prebin/86400`` vs ``bbat_mjd`` | **330 ns** | **OPEN** |
| ``batCorr`` model identity vs lib | **286 ns** | **primary lever** |
| lib ``bat`` vs ``bbat_mjd`` | **304 ns** | float64 export |
| IFTE ``(model−tdb)`` per-TOA scatter | **260 ns** | secondary |
| prebinary internal sum | **0 ns** | **closed** |
| ``deltaT(Tier-1 lib bbat, pt torb)`` spin | **252 ns** | worse than production **16.4 ns** |

Probe: ``/tmp/wsrt_model_bbat_step15_probe.py``

**Step 14 summary (2026-07-06):**

| Link | RMS | Verdict |
|------|-----|---------|
| ``prebinary − total`` vs ``torb_sec`` | **0.17 ns** | **CLOSED** |
| lib ``binarydelay`` vs ``torb_sec`` | **~705 s** | **stale** (zeros) — use ``torb_sec`` / ``prebin−total`` |
| oracle ``model−prebin/86400`` vs ``bbat_mjd`` | **330 ns** | **OPEN** |
| ``deltaT(pt_bbat, pt_torb)`` spin | **~173 ns** | worse than production **16.4 ns** |

Probe: ``/tmp/wsrt_torb_bbat_probe.py``

**Ruled out (Step 3):** WSRT ``-padd`` / ``jump_phase`` per ``-sys`` — not the ~10 ns mean spread.

**Red herring:** idx 85 as an isolated pnNew / ``nphase`` wrap bug — pytempo exact at idx 85;
neighbours share ``-padd`` with 2–23 ns errors; ``addPhase=+1`` matches tempo2.

**Tests / harness**

```bash
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_batcorr_from_model_probe.py -m dev_oracle -q
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_torb_bbat_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_model_bbat_step15_probe.py
```

---

## Recommended path to <5 ns (updated 2026-07-06)

**Priority 1 — JAX tempo2-native pipeline (new)**

1. Implement end-to-end JAX ``calculate_bclt`` → ``formBats`` → ``formResiduals`` matching
   libstempo epochs (Steps 17–18: cannot close 0 ns with JUG production patches alone).
2. Keep JUG production Taylor@``model_mjd`` unchanged until JAX path is validated.

**Priority 2 — wsrt167 production (unchanged)**

1. Do **not** wire ``phase5@bbat`` to production — ruled out (17.4 ns, Step 18).
2. Do **not** change ``-padd`` / ``jump_phase`` handling — ruled out (Step 3).
3. Do **not** pursue Horner / compensated float64 spin — ruled out (Step 8).
4. Do **not** open a separate idx 85 workstream — max \|Δ\| is a tail of the spin gap.
5. Do **not** use libstempo ``binarydelay`` as ``torb`` oracle — stale on fresh construct;
   use ``torb_sec`` or ``prebinary − total`` (Step 14).
6. Do **not** pair pytempo ``torb_sec`` in JUG-composed ``phase5`` — **172 ns** (Step 18).
7. Do **not** treat ``sat + bat_corr → bat`` or ``bbat − torb/86400 → pet`` float64 gaps
   (~237–304 ns) as pytempo/libstempo physics disagreements (§0 oracle cheat sheet).

**Priority 3 — J0030 polish**

- Two 1999 TOAs still ~11 ns after Phase C; investigate early-epoch astrometry if p99
  gate matters.

**Priority 4 — data coverage**

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
| `jug/residuals/tempo2_spin.py` | ``compute_tempo2_bbat_mjd``; ``phase5`` / TRACK −2 |
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

Strict gates on wsrt167 should fail until Taylor spin work closes ~16 ns → <5 ns.
