# Tempo2 parity — status, gaps, and work queue

Living route for JUG `compatibility="tempo2"` parity: measured debt, gap scorecard,
active work queue, and investigation log.

**Policy and architecture:** [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md)

**Status (2026-07-07):** Cases A/B/C green (~1–2 ns). **Hybrid host-frozen JAX path**
(wsrt167 pytempo gates **< 1 ns** on delay terms; production fitting default) is wired for
`design_matrix_method="autodiff"` when `term_diagnostics['tempo2_obs_state']` is present in
the residual cache. IPTA DR2 workloads **partially green**. **Fix #1 (TZR, Phase C):** done.
**Fix #2 (wsrt167, Phase D):** Steps 1–18 — production **Taylor emission spin @ model epoch**
+ legacy TRACK −2 wrapping remains best at **~15.5 ns** wsrt167 / **~31 ns** full EPTA RMS
(vs libstempo); JAX ``phase5@bbat`` + ``track_minus2_frac_phase`` quarantined (breaks
``-addsat`` at ~±1 s). Do **not** wire ``phase5@bbat`` or set
``JUG_TEMPO2_NATIVE_GRAPH_MODE=full`` in interactive sessions. **Primary report:**
[`TEMPO2_NATIVE_CLOCK_STATUS.md`](TEMPO2_NATIVE_CLOCK_STATUS.md).

### J0613 fast gates (2026-07-07)

Inner-loop tests (no full 1369-TOA libstempo in the default loop):

| Gate | File | Measured debt | Strict target |
|------|------|---------------|---------------|
| No TRACK / no ``-pn`` | ``tests/test_tempo2_j0613_fast_gates.py`` | nrt1400 ~4.4 ns | 1 ns |
| TRACK −2 ``-addsat`` mini | ``epta_j0613_addsat_min`` (11 TOAs) | ~0.3–0.7 µs on addsat TOAs | 1 ns |
| wsrt167 bulk spin | same + ``test_dev_oracle_wsrt167_parity.py`` | ~15.5 ns RMS | 5 ns |
| Full EPTA | ``test_tempo2_ipta_dr2_j0613_parity.py`` (xfail) | ~31 ns RMS | 5 ns |

**Production fix (2026-07-07):** Hybrid tempo2 residual routing in
``compute_residuals_simple``:

| Condition | Route | Notes |
|-----------|-------|-------|
| ``TRACK -2`` | ``compute_phase_residuals`` (Taylor + legacy TRACK −2) | wsrt167 ~15.5 ns RMS; do not use ``track_minus2_frac_phase`` on Taylor phase |
| ``TRACK`` absent (tempo2 default 0) | ``compute_phase_residuals`` (Taylor sequential) | Fixes stripped no-TRACK nrt1400 (~6 ns); native ``phase5@bbat`` trunc was ~1.4 ms wrong |
| Explicit non-``-2`` ``TRACK`` | ``compute_native_eval_residuals_jax`` | Autodiff / native delay chain staging |

Strict probe harness (``/tmp/jug_tempo2_strict/``, not in repo) measured on 2026-07-07:

| wsrt167 term | Raw vs pytempo | Residual after substitution |
|--------------|----------------|----------------------------|
| ``torb_sec`` | ~27 ns RMS | — |
| ``bbat_mjd`` | ~304 ns RMS (documented assembly debt) | substituting pytempo **worsens** residuals to ~5.6 µs |
| ``acceptance_residual`` baseline | — | **15.5 ns** RMS (unchanged) |

**Conclusion:** do **not** promote split longdouble ``formBats`` or ``phase5@bbat`` into production
for wsrt167; the ~15.5 ns floor is not delay-chain assembly. ``addsat`` mini fixture remains
~283 ns bulk / ~566 ns max on addsat TOAs (shifted-``sat`` + ``addsat_track2_turn_delta``).

**Environment-only failures (not code regressions):** libstempo parity tests that require
``DE440`` ephemeris fail in this container (`FileNotFoundError`). NG5 TDB broad parity remains
~5.3 µs vs libstempo in this environment (native JAX path); document separately from J0613 debt.

Component oracle gates: ``tests/test_tempo2_j0613_delay_terms.py`` (`dev_oracle`, pytempo).

---

## JAX tempo2-native clock/delay chain (planned — required for 0 ns)

**Goal:** end-to-end JAX code that reproduces libstempo/tempo2 ``updateBatsAll`` →
``calculate_bclt.C`` → ``formBats.C`` → ``formResiduals.C`` on the same per-TOA epochs,
not the JUG shortcut ``(model_mjd − sat)×86400 − prebinary``.

**Why a separate track (Steps 16–18):**

| Layer | JUG production today | tempo2 native | Gap |
|-------|---------------------|---------------|-----|
| Clock / delays | ``IFTE(tdb)`` emission ``model_mjd`` + bundled ``prebinary`` | ``TT+TT_TB`` + ``tdis`` slot via ``formBats`` | **~286 ns** ``batCorr`` (Step 17) |
| Spin | Taylor Horner on ``dt_sec_ld`` + legacy TRACK −2 | ``phase2+phase3`` at ``bbat`` + ``pnNew`` | **~16 ns** best JUG path (Step 18) |
| Oracle | pytempo ``acceptance_residual_sec`` | libstempo ``psr.residuals()`` | **0 ns** when full chain matches |

**Scope (in order):**

1. **``calculate_bclt`` iterative Roemer epoch** — ``sat + (TT+TT_TB+dt_SSB)/86400``;
   delays at tempo2 ``delt``, not JUG IFTE ``model_mjd`` alone.
2. **``formBats.C`` batCorr** — ``TT + TT_TB − tropo + roemer − shap − tdis`` (verified
   0 ns replay vs lib ``batCorrs`` in Step 17).
3. **``formResiduals.C`` spin** — ``phase2+phase3`` at true ``bbat``; ``torb`` from T2model;
   TRACK −2 ``pnNew`` via ``track_minus2_frac_phase``.
4. **Longdouble** end-to-end until final float64 export (Steps 7/16).

**Explicit non-goals for this track:**

- Do **not** patch production NumPy ``simple_calculator`` spin to ``phase5@bbat`` first
  (Step 18: **17.4 ns**, worse than Taylor **16.4 ns**).
- Do **not** use pytempo ``torb_sec`` in JUG-composed spin (Step 18: **172 ns** trap).
- Do **not** rely on ``(model−sat)−prebin`` as the JAX ``batCorr`` identity.

**Status (2026-07-06):** scaffold implemented under
``jug/residuals/tempo2_native/`` (JAX ``chain_jax.py`` + dev ``chain_numpy.py``).
``JUG_TEMPO2_NATIVE_GRAPH_MODE`` selects the traced graph (default ``staged_bclt``).

| Gate | Interim (dev_oracle) | Strict target | Notes |
|------|---------------------|---------------|-------|
| ``batCorr`` vs lib | **~286 ns** RMS | `< 1 ns` | IFTE model-epoch + JUG prebinary (production path) |
| ``bat_corr_days`` vs tempo2 | **~1.1 ns** RMS | `< 1 ns` | unified JAX strict formBats component sum (wsrt167) |
| ``bat_mjd`` / ``bbat_mjd`` vs tempo2 | **~304 ns** RMS | `< 1 ns` | **MJD assembly recipe** — see § below; not delay physics |
| ``torb`` closure vs pytempo | **~262 ns** RMS | `< 1 ns` | JUG ``dt`` + model-epoch ``bbat`` (production path) |
| BCLT ``roemer`` vs pytempo | **~18 ms** RMS | term ranking | fixed IFTE ``tdis`` in loop |
| Spin counterfactual | **~5.6 µs** RMS | `< 5 ns` | pending full BCLT ``formBats`` |
| Full residuals wsrt167 | skipped (flag off) | `< 5 ns` | flip after all gates green |

Tests: ``tests/test_tempo2_native_*.py``, ``tests/test_jax_tempo2_native_*.py`` (`dev_oracle`).
NumPy reference env-gated: ``JUG_DEV_NUMPY_TEMPO2_CHAIN=1``.

**Reference sources:** ``ref-packages/tempo2/{calculate_bclt,formBats,formResiduals,tt2tdb}.C``;
pytempo ``toa_diagnostics()`` for Tier-1 oracles.

### formBats ``bat_mjd`` / ``bbat_mjd`` assembly — unified JAX path

**Do not confuse delay parity with MJD epoch parity.** On wsrt167 the unified JAX
native chain (``jug/residuals/tempo2_native/formbats_jax.py``) shows:

| Quantity | Native vs tempo2 RMS | What it tests |
|----------|---------------------|---------------|
| ``bat_corr_days`` | **~1.1 ns** | Delay-component closure (physics gate) |
| ``bat_mjd`` | **~304 ns** | Assembled MJD epoch scalar |
| ``bbat_mjd`` | **~304 ns** | Same on wsrt167 (Shklovskii ≈ 0) |
| ``shklovskii_sec`` | **0 ns** | Not the wsrt167 blocker |

**Root cause:** tempo2 ``formBats.C`` does **not** build ``bat`` as a single
float64 sum ``sat + (all terms)/86400``. It uses ``long double`` and **splits the
UTC→TT term**:

```c
// ref-packages/tempo2/formBats.C (L67–83)
batCorr = getCorrectionTT(obsn)/SECDAY
        + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
bat  = sat + getCorrectionTT(obsn)/SECDAY
     + (correctionTT_TB - tropo + roemer - shap - tdis1 - tdis2)/SECDAY;
bbat = bat - shklovskii/SECDAY;
```

JUG's JAX helper instead sums all correction seconds in float64, divides once, and
assembles with ``assemble_mjd_from_day_sec`` (``two_sum`` in float64). That recipe is
**internally consistent** (``sat + bat_corr_days`` matches JUG ``bat_mjd`` to 0 ns)
but **does not reproduce tempo2's split long-double assembly** when ``sat`` is
O(10⁴) MJD and the net correction is O(10²) s.

**Verified on wsrt167 (2026-07-07):**

- tempo2 ``bat_corr_days`` matches the component-sum formula to **~0 ns**.
- tempo2 ``bat_mjd`` matches ``sat + bat_corr_days`` in **float64** to **~304 ns**.
- tempo2 ``bat_mjd`` matches the **split long-double recipe** to **0 ns**.
- JUG native ``bat_mjd`` matches the split long-double recipe with **JUG components**
  to **0 ns**, but differs from tempo2 ``bat_mjd`` by **~304 ns** because of the
  assembly recipe, not because roemer/clock/Shapiro components are wrong.

**Is this a residual-parity blocker?** Not automatically. The native chain defines
``torb`` as a closure (``dt_emission − (bbat − PEPOCH)·86400``) and feeds
``phase5(bbat, torb)``. A constant ~304 ns shift in ``bbat`` can cancel against
``torb`` in ``deltaT = (bbat−PEPOCH)·86400 + torb`` as long as the integer MJD day
of ``bbat`` is unchanged. So a failing ``bbat_mjd`` gate does **not** mean delay
physics is wrong; it means the exported epoch scalar is not yet tempo2-identical.

**What to gate on:**

- **Physics / delay closure:** ``bat_corr_days`` and per-component gates (``tt``,
  ``roemer``, ``tdis*``, ``shap``, …) — target **< 1 ns**.
- **MJD assembly parity:** ``bat_mjd`` / ``bbat_mjd`` — requires porting tempo2's
  split summation (or equivalent compensated float64) in ``formbats_jax.py``.
- **End-to-end:** ``acceptance_residual_sec`` vs libstempo/tempo2 — the only gate
  that ultimately matters for production parity.

**Tests:** ``test_native_strict_formbats_batcorr_wsrt167`` (delay gate, ~1 ns);
``test_native_bbat_strict_formbats_wsrt167`` (epoch gate, expected fail ~304 ns until
assembly is ported). See ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § "formBats MJD assembly".

---

## 0. Diagnostic workflow

### pytempo is expanded libstempo

[`ref-packages/pytempo`](../../pytempo) is **not** a second timing engine. It is an
**expanded libstempo**: same Cython → tempo2 wrapper (`updateBatsAll`, `formResiduals`,
`t2FitFunc_*`, …), forked from
[vhaasteren/libstempo `sandbox`](https://github.com/vhaasteren/libstempo/tree/sandbox)
(vendor SHA in [`pytempo/VENDOR_SHA`](../../pytempo/VENDOR_SHA)). pytempo adds:

- **Per-TOA `obsn[]` exposure** — `toa_diagnostics()`, `phase_diagnostics()`, and
  properties (`bbat`, `torb`, `nphase`, …) that libstempo never exported.
- **Sandbox RPC** — crash-isolated subprocess wrapper around the same `tempopulsar` API.
- **Updated `observation` struct** in `pytempo.pyx` aligned with the installed tempo2
  header (e.g. `/opt/software/tempo2/install/include/tempo2.h`).

Both packages must link to the **same** `libtempo2` (here: tempo2 **2026.4.1** at
`/opt/software/tempo2/install/lib`). On shared fixtures, `psr.residuals()`,
`roemer`, `shapiro_sun`, and `designmatrix()` match libstempo exactly when tempo2
matches.

### Acceptance oracle (pytest)

Raw pre-fit residuals vs **libstempo** via `jug/testing/tempo2_reference.py`. This is
the only oracle wired into jug pytest today. Tests that require libstempo are marked
`dev_oracle` (see `jug/testing/DEV_ORACLE.md`).

```bash
cd ref-packages/jug
PYTHONPATH=. pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```

### pytempo oracle cheat sheet

Use this when an agent or notebook calls `pytempo` `toa_diagnostics()` alongside
libstempo. False “inconsistencies” almost always come from comparing the **wrong**
dict key or residual product — not from pytempo reimplementing tempo2.

#### Reliability tiers

| Tier | Fields | Use for libstempo / JUG parity? |
|------|--------|----------------------------------|
| **1 — oracle** | `acceptance_residual_sec`, `pulse_number`, `bbat_mjd`; also `psr.residuals()` | **Yes** — validated 0 µs vs libstempo on EPTA J0613 / wsrt167 |
| **2 — conditional** | `phase_offset_turns`, `residual_sec` (only if `residual_sec_reliable`) | `-padd` exposure; raw `obsn[].residual` **unreliable on TRACK −2** |
| **3 — informational** | `prefit_residual_sec`, `nphase` / `spin_phase_turns`, `phase_turns` | **No** — different tempo2 internal products; not acceptance residuals or `-pn` |

#### Residual products (do not mix)

| Field / call | What it is | libstempo equivalent |
|--------------|------------|----------------------|
| `psr.residuals()` (default `removemean=True`) | tempo2 acceptance residual | **oracle** |
| `diag["acceptance_residual_sec"]` | second `formResiduals` pass with mean removal | same as `psr.residuals()` |
| `diag["residual_sec"]` | raw `obsn[].residual` after `formResiduals(removemean=False)` | **not** `psr.residuals()` on TRACK −2 |
| `diag["prefit_residual_sec"]` | `obsn[].prefitResidual` | not an acceptance residual |

`toa_diagnostics()` always calls `formResiduals` once with your `removemean` arg, copies
`residual_sec`, then **re-runs** `residuals(removemean=True)` for
`acceptance_residual_sec`. On **TRACK −2** with `removemean=False`, `residual_sec` can be
numerically garbage (wsrt167 RMS ~10¹⁴ ms vs acceptance; EPTA J0613 ~3 ms). Check
`residual_sec_reliable` before using `residual_sec` for cross-checks.

#### Phase / pulse fields (do not mix)

| Field | Meaning | Common mistake |
|-------|---------|------------------|
| `pulse_number` | tim `-pn` index (`obsn[].pulseN`) | compare to `nphase` |
| `nphase`, `spin_phase_turns` | spin-phase turn count (~10⁸–10¹¹ turns) | treating as pulse index |
| `phase_offset_turns` | **Python-computed** effective tim `-padd`/`-radd` (turns) | equating to `obsn_phase_offset_turns` (Parkes column; usually 0 on IPTA) |
| `phase_turns` | `obsn[].phase` after `formResiduals` | not `-pn` |

#### Python-side logic (not tempo2-native)

pytempo inherits libstempo’s small Python post-steps; only the diagnostic layer adds new
ones:

- `residuals(removemean='weighted'|'first')` — mean subtracted in Python after tempo2.
- `phaseresiduals()` — multiplies seconds by `F0` in Python.
- `designmatrix(fixunits=True, fixsigns=True)` — column scaling after tempo2 derivatives.
- `_tim_phase_offset_turns()` — sums tim `-padd` and `-radd×F0` from flags, not
  `obsn[].phaseOffset`.

`fit.py`, `like.py`, `toasim.py`, and `tim_file_analyzer.py` are separate scipy/numpy
paths — **not** the tempo2 oracle.

#### Agent checklist

1. Compare **`acceptance_residual_sec`** or **`psr.residuals()`**, never `residual_sec`
   on TRACK −2 workloads.
2. Read **`residual_sec_reliable`** before using raw `residual_sec`.
3. Never equate **`nphase`** with **`pulse_number`**.
4. Use **`phase_offset_turns`** for `-padd` parity, not `obsn_phase_offset_turns`.
5. Confirm both wrappers use the same tempo2 install (`pytempo.tempo2version()` /
   `libstempo.tempo2version()`).
6. Prefer **`pytempo.tempopulsar`** for diagnostics; **`pytempo.sandbox.tempopulsar`**
   returns the same values (RPC wrapper only).
7. Do **not** use libstempo **`binarydelay`** as a ``torb`` oracle on a fresh construct —
   it reads **zeros** until the full tempo2 pipeline has run; use **`torb_sec`** or
   JUG ``prebinary − total`` (0.17 ns on wsrt167).
8. Do **not** treat float64 recompositions as physics disagreements — see
   **§ formBats ``bat_mjd`` / ``bbat_mjd`` assembly** above. In short:
   ``sat + bat_corr → bat`` (~304 ns on wsrt167) fails in float64 because tempo2
   uses split ``long double`` assembly; ``bat_corr_days`` itself is the delay gate.
   ``bbat − torb/86400 → pet`` (~275 ns) is a similar export/assembly artifact.

#### libstempo property traps

| Property | Trap |
|----------|------|
| `binarydelay` | Often **zeros** on fresh construct — not ``obsn.torb`` |
| `toas` | Not documented as Tier-1 ``bbat`` — do not use for bbat parity |
| `stoas`, `batCorrs`, `pets` | Tier-1 when tempo2 pipeline has run — match pytempo exports at 0 ns |

Minimal example:

```python
from pytempo import tempopulsar

psr = tempopulsar(parfile="x.par", timfile="x.tim", dofit=False)
diag = psr.toa_diagnostics(removemean=False)

# Tier-1 oracle — compare to libstempo psr.residuals()
acc = diag["acceptance_residual_sec"]
pn = diag["pulse_number"]
bbat = diag["bbat_mjd"]

if not diag["residual_sec_reliable"]:
    # TRACK -2: do not compare diag["residual_sec"] to libstempo
    pass
```

See also [`pytempo/README.md`](../../pytempo/README.md) (per-TOA diagnostics section).

### Term-by-term debugging loop

1. Load fixture par/tim — start **`wsrt167`**, then `epta_j0613_t2_nrt1400`, then full EPTA.
2. Acceptance check: `tempo2_reference(par, tim)` vs
   `compute_residuals_simple(..., compatibility="tempo2")`.
3. Term decomposition: compare JUG `term_diagnostics` / top-level keys
   (`bbat_mjd`, `prebinary_delay_sec`, `roemer_sec`, `sw_delay_sec`, etc.) against
   libstempo properties or Phase A (`jug/testing/phase_a_comparison.py`).
4. Optional ad-hoc oracle: [`ref-packages/pytempo`](../../pytempo) `toa_diagnostics()`
   for per-TOA tempo2 `obsn[]` fields libstempo does not expose. **Not** a JUG
   dependency; **not** wired into jug test infrastructure. pytempo is expanded
   libstempo (same tempo2 backend) — see **§0 pytempo oracle cheat sheet** before
   comparing dict keys.

   **In-repo harness (preferred for outlier work):**
   `jug/testing/tempo2_outlier_diff.py` — per-TOA clock + Roemer diff vs libstempo;
   CLI: `tools/run_tempo2_outlier_clock_roemer_diff.py`; test:
   `tests/test_tempo2_outlier_clock_roemer_diff.py`.

   ```bash
   pip install -e ref-packages/pytempo   # requires $TEMPO2 runtime; ad-hoc only
   ```

   | `bbat_mjd` | ``model_mjd − prebinary/86400`` (oracle); formBats diag **~65 s off** |
   | `roemer_sec`, `torb_sec` | `term_diagnostics` / top-level keys |
   | `phase_offset_turns` | tim ``-padd`` via ``jump_phase``; not the same as ``addPhase`` from ``pnNew`` |
   | `nphase`, `phase_turns` | ``compute_tempo2_phase5`` at pytempo ``bbat`` (Phase D harness) |
   | `pulse_number` | tim ``-pn`` (absolute-looking; **use ``pn[i]−pn[0]``** for ``pnAct``) |
   | `acceptance_residual_sec` | libstempo ``psr.residuals()`` oracle (Tier 1) |

5. Rank largest term delta → fix JUG native physics → tighten pytest debt pin → re-run.

Use raw residuals (no mean subtraction) for tempo2 acceptance gates.

Phase A term oracle: `jug/testing/tempo2_diagnostics.py` (libstempo properties only).

---

## 1. Status dashboard

### Residual parity vs libstempo (JUG − libstempo, raw pre-fit)

| Workload | n | RMS Δ | max \|Δ\| | Gate | CI |
|----------|---|-------|-----------|------|-----|
| Cases A/B/C (TCB + NG5 TDB) | 10–625 | **~1–2 ns** | < 25 ns | 5 ns | green |
| PPTA native J0613 (`PPTA_dr1dr2`) | 410 | **1.43 ns** | 5.17 ns | 5 ns | ad hoc |
| PPTA alternate export par/tim | 410 | **15.96 ns** | 33.34 ns | TBD | ad hoc |
| EPTA full (`epta_j0613_t2_ipta_all`) | 1369 | **608 ns** | 4.5 µs | 5 ns | xfail + debt pin |
| EPTA NRT1400 excerpt | 120 | **62 ns** | — | TBD | debt pin |
| **epta_j0030_isolated** | 10 | **~4.7 ns** | ~11 ns | none | **pass RMS**; p99 ~11 ns on 2×1999 TOAs |
| **wsrt167** | 167 | **~16 ns** | **~110 ns** | T2 | **fail** — Taylor spin (Phase D); max at idx 85 |

### `-addsat` / tempo2 spin

| Item | Status |
|------|--------|
| `-addsat` regression (idx 247/256/561) | **Fixed** — each **< 1 µs** vs libstempo |
| Implicit `NE_SW = 4` cm⁻³ when par omits keyword | **Fixed (2026-07-05)** — `resolve_ne_sw_cm3()` |
| Roemer PM at POSEPOCH (not evolving `L_hat` only) | **Fixed (2026-07-05)** — `compatibility_providers.py` |
| IFTE + `formBats` clock (`tempo2_clock.py`) | **Diagnostic-only** — ~64 s formBats bat gap vs libstempo; production spin uses geometry `model_mjd` |
| Taylor spin at emission `model_mjd` (production) | **In use** — ~16 ns on wsrt167 |
| TZR post-wrap / pre-wrap / skip (`tzr_geometry.py`) | **Done (Phase C)** — J0030 15.9 → ~4.7 ns RMS |
| Native `phase5` at formBats `bbat` (`USE_NATIVE_BBAT_PHASE5`) | **Quarantined** — ~36 ns; formBats `bbat` ~65 s off pytempo |
| `track_minus2_frac_phase` pnAct | **Fixed (Phase D Step 1)** — ``pnAct = (pn[i]−pn[0]) + pnAdd``; matches legacy ``−pnAdd`` on wsrt167 |
| `compute_tempo2_phase5` at pytempo `bbat` | **Validated** — `nphase` exact; **~17.5 ns** RMS (not wired; worse than production) |
| `compute_tempo2_bbat_mjd` | **Done** — ``model − prebinary/86400`` matches libstempo/pytempo exactly |
| Longdouble clock/spin pass | **Reverted (2026-07-05)** — zero measurable benefit |
| Raw `phase5(bbat)−phase5(bbat−addsat)` wrap on legacy TRACK −2 | **Wrong** — ~67 µs at idx 247; do not use |
| `addsat_track2_turn_delta` int(F0) closure | **In use** — calibrated constants; not yet derived from `ff0` alone |
| `bbat_mjd` / `torb_sec` in JUG output | **Done** |
| BCLT iteration in `simple_calculator` | **Ruled out** — ~903 µs regression when wired |
| Fitter `TRACK -2` / `-addsat` wiring | **Open** — `optimized_fitter.py` |

Treat tempo2 mode as **experimental** outside curated par+tim tests. Do not use
`design_matrix_method="analytic"` on tempo2 sessions.

---

## 2. Gap scorecard

| Gap | Status | Summary |
|-----|--------|---------|
| **G1** NumPy `residual_delta(0) ≠ 0` | **Closed (2026-07-03)** | `get_longdouble()` for `HIGH_PRECISION_PARAMS` before zero perturbation |
| **G2** JAX autodiff at θ=0 | **Closed (2026-07-03)** | Unified `compute_total_delay_change` + `BinaryDelayPlan`; θ=0 peak ≲10⁻¹³ s |
| **G2 residual** NumPy vs JAX at θ≠0 | **Open** | ms-level mismatch on IPTA binary/astrometry perturbations; not CI-gated — **suspicious; see §G2 note** |
| **G4** Analytic design matrix | **Open** | Known broken; use autodiff |
| **G5** Fixture coverage | **Open** | Green on A/B/C; IPTA workloads partial (see §1) |
| **G6** Documented residual debt | **Open** | `ppta_j1741_ell1` ~5–8 ns; `DM_SERIES` warn-only |
| **G7** EPTA multi-backend | **Open (improved)** | ~608 ns bulk after integer-turn and `-addsat` fixes; wsrt167 subset ~263 ns |
| **G8** `DMASSPLANET` reflex correction | **Deferred** | Not parsed in JUG; unused in IPTA fixtures; easy patch when needed |
| **G9** Full `get_obsCoord` port | **Deferred** | Astropy/ERFA approximation already <0.01 cm on wsrt167; not blocking ns gates |

**Scorecard:** 2 closed (G1, G2 primary), 5 with open items, 2 deferred (G8, G9).

### G1 — closed

Historical ~27 ns peak from float64 round-trip on F0 in `residual_delta(0)`. Fixed by
reading `HIGH_PRECISION_PARAMS` via `get_longdouble()`. Touchpoint:
`jug/fitting/optimized_fitter.py`.

### G2 — closed at θ=0; residual open at θ≠0

Historical JAX binary dispatch used DD-only path (~2.1 s offset on J0613). Fixed with shared
binary dispatch. θ≠0 IPTA perturbations still show ms-level NumPy/JAX disagreement.

> **Note (2026-07-06):** The “G2 residual / ms-level at θ≠0” line is **suspicious and
> unlikely to be true** as stated. A nonzero perturbation δ is equivalent to evaluating at
> a shifted par vector; if NumPy and JAX agree at the reference point, they should agree at
> δ unless the two paths implement different physics (a bug), not a separate “θ≠0 regime.”
> This claim is not CI-gated and has no cited measurement; it may conflate the fixed ~2.1 s
> binary-dispatch bug or JUG-vs-libstempo ms-scale gaps with internal NumPy/JAX parity.
> `tests/test_jax_numpy_parity_deprecated.py` checks **nonzero** δ at picosecond tolerance on
> synthetic spin/DM/astrometry/binary setups. Treat “G2 residual” as **unverified scorecard
> debt** until reproducing evidence exists on a gated IPTA fixture—or remove it.

### G4 — analytic design matrix

Legacy analytic columns are a PINT-parity mistake on tempo2 setups. Use autodiff.

### G5 — fixture coverage

| Workload | In tempo2 CI? | Status |
|----------|---------------|--------|
| NG5 Cases B/C | Yes | Green |
| TCB Case A | Yes | Green |
| EPTA J0613 full TIM | Yes | ~608 ns RMS |
| EPTA J0613 nrt1400 | Yes | ~62 ns RMS |
| **wsrt167** (WSRT low-band) | dev_oracle | **~263 ns RMS** (debt pin) |
| PPTA native J0613 | No | ~1.4 ns ad hoc |
| PPTA alternate export | No | ~16 ns ad hoc |
| ELL1/T2 binaries | Partial | Autodiff green; `ppta_j1741_ell1` debt |

### G6 — documented residual debt

- **`ppta_j1741_ell1`:** RMS ~5–8 ns vs strict 5 ns gate; ELL1 convention mismatch.
- **`DM_SERIES`:** ignored by JUG (warn-only).

### G7 — EPTA multi-backend (~608 ns)

Integer-turn and per-TOA `-addsat` debt **closed** (2026-07-04). Remaining ~608 ns is
**bulk pulse-phase / spin bookkeeping** across the multi-backend mix — not Roemer, DM, or
binary delay kernels at ms scale.

Measured on `epta_j0613_t2_ipta_all` (`tests/test_tempo2_ipta_dr2_j0613_parity.py`):

| Quantity | Gate | EPTA (2026-07-04) |
|----------|------|-------------------|
| RMS Δ | < 5 ns | **≈ 608 ns** |
| p99 \|Δ\| | < 10 ns | **≈ 4 µs** |
| max \|Δ\| | < 25 ns | **≈ 4.5 µs** |
| `-addsat` TOAs (idx 247/256/561) | — | **< 1 µs** each |

Tests: `test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity` (xfail strict);
`test_epta_j0613_ipta_dr2_track_minus2_debt_reduced` (debt pin).

### G8 — `DMASSPLANET` / `DPHASEPLANET` (deferred)

**Priority:** not on the active queue. Recorded for completeness; does not block current
IPTA parity gates.

Tempo2 optionally adjusts `earth_ssb` for errors in planetary masses
(`ref-packages/tempo2/readEphemeris.C`):

```c
for (iplanet=0; iplanet < 9; iplanet++)
    if (psr[p].param[param_dmassplanet].paramSet[iplanet])
        for (icomp=0; icomp < 6; icomp++)
            psr[p].obsn[i].earth_ssb[icomp] -=
                psr[p].param[param_dmassplanet].val[iplanet] *
                psr[p].obsn[i].planet_ssb[iplanet][icomp];
```

| Item | Tempo2 | JUG today |
|------|--------|-----------|
| Par keywords | `DMASSPLANET1` … `DMASSPLANET9` (solar masses; `readParfile.C`) | Not parsed in `par_reader` |
| Physics | Subtract `dmass[i] × planet_ssb[i]` from `earth_ssb` pos/vel | No correction applied |
| `planet_ssb` source | Read from ephemeris when `DMASSPLANET` or `DPHASEPLANET` set | Already computed for Shapiro |
| `DPHASEPLANET` | Parsed; would use `planet_ssb_derv` | **Dead in Tempo2** — correction block commented out in `readEphemeris.C` |

**Impact:** none on gated fixtures (no `DMASSPLANET` in `tests/data_tempo2/`). Niche
ephemeris-systematics feature (e.g. fitting Jupiter mass offsets). MetaPulsar mock pulsars
set `DMASSPLANET*` to zero.

**Patch estimate (when needed):** easy (~½ day). Parse `DMASSPLANET1..9`; apply linear
correction after `earth_ssb` assembly in `tempo2_ephemeris.py` and `tempo2_geometry_jax.py`;
add dev_oracle test with non-zero `DMASSPLANET5` (Jupiter). Caveats: 9-slot planet index
map (Tempo2 Mercury–Neptune + Moon/Pluto vs JUG's 7 named planets); design-matrix columns
for fitting would be extra work beyond evaluation-only parity. Skip `DPHASEPLANET` unless
reviving dead Tempo2 code.

### G9 — Full `get_obsCoord` port (deferred)

**Priority:** not on the active queue. Recorded for completeness; does not block current
residual debt (~16 ns wsrt167 is in clock / formBats / spin, not site geometry).

JUG does **not** line-port `ref-packages/tempo2/get_obsCoord.C`. For ground telescopes it
approximates the modern IAU2000B path:

| Path | Implementation | Notes |
|------|----------------|-------|
| Host (`compute_tempo2_observatory_state`) | Astropy `EarthLocation.get_gcrs_posvel()` at `site_mjd` (TT) | Documented as *not* a line-by-line C port |
| JAX (`tempo2_site_jax.py`) | ERFA via `jax.pure_callback` (`c2i06a`, `pom00`, `c2tcio`) + Astropy polar motion / UT1 | Mirrors Astropy `get_gcrs_posvel` route, not Tempo2 `iau_c2t00b_` + `get_EOP` |

**Measured on wsrt167 (ground WSRT):** `observatory_earth` RMS **< 0.01 cm** vs pytempo
(`tests/test_tempo2_native_geometry_parity.py`). Ephemeris delay terms (`roemer_sec`,
`dt_ssb_sec`) already **< 1 ns** RMS vs pytempo on the same fixture.

**What Tempo2 `get_obsCoord.C` includes that JUG does not fully port:**

| Gap | Severity for IPTA | Notes |
|-----|-------------------|-------|
| Line-accurate `get_obsCoord_IAU2000B` + `get_EOP(eopc04_file)` | Low | Astropy IERS-B vs Tempo2 EOP file; mm-level possible at some epochs; not visible in current gates |
| Satellite `STL` / `STL_FBAT` | Niche | `TELX/TELY/TELZ` polynomials, tim `-telx/-tely/-telz`, `telDX/DY/DZ` tables; no STL fixtures in JUG tests |
| `STL_BAT` (barycentric satellite) | Niche | `observatory_earth = 0`, tropo off |
| Legacy nutation path (`t2cMethod` ≠ IAU2000B) | Very low | Nutations from ephemeris + precession + LMST; obsolete for modern data |
| Ecliptic rotation of site vectors | Low | JUG rotates combined `earth_ssb + observatory_earth` (equivalent rigid rotation) |
| Tropo zenith EOP | Medium for tropo only | `tropo_jax._host_zenith_gcrs_m` hardcodes `xp=yp=0` |
| Pure JAX ERFA (no `pure_callback`) | Architectural | JIT/autodiff unity, not correctness |

**Patch estimates (when needed):**

| Sub-feature | Effort | ROI for IPTA DR2 |
|-------------|--------|------------------|
| Ground IAU2000B line-port | Medium (~1–2 weeks) | Low — sub-cm already |
| Satellite / STL paths | Medium–hard (~1–2 weeks) | Only if spacecraft timing workloads appear |
| Legacy nutation path | Medium | Not recommended — obsolete |
| Tropo zenith with real EOP | Easy–medium (~1 day) | After clock/BCLT debt closes |
| Native JAX ERFA | Hard (~weeks) | Phase 4 JIT architecture only |

**Related context:** production tempo2 mode already uses host jplephem SPK for planetary
positions (`Tempo2DelayProvider`); the in-graph SPK path (`tempo2_spk_jax.py`) is for the
quarantined unified JIT chain (`JUG_TEMPO2_NATIVE_GRAPH_MODE=full`). Astropy ephemeris
(PINT path) is **not** a substitute for tempo2-native site motion — wrong clock chain,
Teph epoch coupling, and `get_obsCoord` conventions (see `TEMPO2_COMPATIBILITY.md` §2).

---

## 3. Active work queue

| Priority | Task | Oracle / fields | Status |
|----------|------|-----------------|--------|
| **1** | **Clock / ``model_mjd`` vs tempo2 ``updateBatsAll``** | pytempo ``pet_mjd``/``torb_sec``/``bbat_mjd``; Step 5–7 probes | **Open** — ~330 ns ``bbat``; float64 ``model_mjd`` inputs; **do not** swap Taylor→``deltaT(pt)`` |
| **2** | Close **wsrt167** to <5 ns gate | `test_dev_oracle_wsrt167_parity.py` | **Open** — blocked on priority 1 |
| **Ruled out** | **WSRT ``-padd`` / ``jump_phase``** per backend | pytempo ``phase_offset_turns`` vs JUG ``jump_phase`` | **Exact match** (Step 3) — not the ~10 ns inter-``-sys`` split |
| **Red herring** | **idx 85 as isolated pnNew / ``nphase`` ladder bug** | pytempo exact at idx 85; ``addPhase=+1`` | **Not a separate fix** — max \|Δ\| tail of spin error (§ Phase D) |
| **3** | Polish **epta_j0030** p99 (~11 ns on 2×1999 TOAs) | outlier harness; early-epoch astrometry | **Open** — RMS gate passes after Phase C TZR |
| **5** | Update BIPM clock files for `epta_j0613_t2_ipta_all` | clock-file coverage to MJD 56795 | **Open** — data, not algorithm |
| **6** | Validate on `epta_j0613_t2_nrt1400` (~6 ns) | libstempo + term diagnostics | **Open** |
| **7** | Alternate PPTA ~16 ns | Roemer/Shapiro + TZR at `TZRMJD` | **Open** |
| **8** | Fitter TRACK −2 / `-addsat` wiring | after subset gates pass | **Open** |
| **Done** | **Phase C — TZR** (fix #1) | `tests/test_tempo2_tzr_parity.py`; `tzr_geometry.py` | **Done** — J0030 15.9 → ~4.7 ns RMS |
| **Done** | **Phase D Step 1 — pnNew convention** | `tests/test_tempo2_track2_pnnew.py` | **Done** — relative ``-pn`` |
| **Ruled out** | **Phase D Step 2 — wire ``phase5@bbat`` to production** | `tempo2_track2_oracle.py` | **~17.5 ns** — worse than Taylor production |
| **Defer** | formBats ``bbat`` diagnostic fix | ~65 s off oracle | — |
| **Defer** | **G8 — `DMASSPLANET` reflex correction** | `readEphemeris.C`; no fixture coverage | Recorded §G8 — easy when needed |
| **Defer** | **G9 — full `get_obsCoord` port** | wsrt167 `< 0.01 cm` observatory_earth | Recorded §G9 — low ROI for ground IPTA |

---

## Phase C — TZR reference phase (fix #1, 2026-07-05)

**Goal:** close **epta_j0030_isolated** (~15.9 ns RMS dominated by 2×1999 TOAs far from
``TZRMJD``).

**Status:** **Done** in production tempo2 path.

### Root cause

JUG subtracted ``tzr_phase`` in **turn space before** phase wrapping. Tempo2
``REFPHS TZR`` inserts a synthetic TZR observation and subtracts its **wrapped residual**
after ``phas1`` / ``nlong``. The error grows with distance from ``TZRMJD`` (~4010 days for
the 1999 outlier TOAs vs ≤773 days on passing fixtures).

### Fix

``jug/residuals/tzr_geometry.py`` — ``resolve_tempo2_tzr_apply_mode()``:

| Mode | When | Behaviour |
|------|------|-----------|
| `pre_wrap` | TOAs within ~2000 days of ``TZRMJD`` | legacy pre-wrap ``tzr_phase`` (ng5 near-TZR) |
| `post_wrap` | explicit ``REFPHS TZR`` | subtract wrapped TZR residual after phase5 |
| `none` | TOAs far from ``TZRMJD`` | skip TZR shift (matches libstempo default for J0030) |

### Measured (J0030 vs libstempo)

| Metric | Before | After Phase C |
|--------|--------|---------------|
| RMS | 15.9 ns | **~4.7 ns** |
| max \|Δ\| | ~38 ns | **~11 ns** |
| Outlier TOAs 0–1 | ~32–38 ns | **~7–11 ns** |

Tests: ``tests/test_tempo2_tzr_parity.py``. ng5 / binary parity unchanged.

---

## Phase D — TRACK −2 pnNew (fix #2, 2026-07-05)

**Goal:** close **wsrt167** (~16 ns RMS, max ~110 ns) without re-enabling the quarantined
formBats / ``USE_NATIVE_BBAT_PHASE5`` stack.

**Status:** **Step 1 done** (pnNew convention + oracle tests). **Step 2 ruled out**
(2026-07-06). **Step 3 ruled out** (``-padd`` / ``jump_phase``). **Step 4 ruled out**
(Taylor vs ``phase2``). **Steps 5–7 done** (term diff / pet/torb / dt precision). **Next:**
align float64 ``model_mjd`` with tempo2 ``calculate_bclt`` epoch without breaking Taylor+TRACK−2.

### What we found (Step 1)

1. **tim ``-pn`` is not absolute.** On IPTA DR2 exports (wsrt167), ``-pn[i] − -pn[0]`` equals
   tempo2 ``pnNew`` (after ``pn0`` anchoring at obsn[0]). JUG was using raw ``-pn`` in
   ``pnAct``, producing ``addPhase ~ 10¹⁰`` turns when combined with ``phase5``.
2. **With relative ``-pn``, ``addPhase = −pnAdd``** (+1 turn on wsrt167 where every TOA has
   ``pnAdd = −1``). That matches the legacy production wrap (`add_phase = −pn_add`).
3. **`compute_tempo2_phase5` at pytempo ``bbat`` is validated** (`nphase` exact vs tempo2).

### Step 2 investigation (2026-07-06) — ruled out

Temp-only path ranking on wsrt167 vs libstempo:

| Path | RMS Δ | Notes |
|------|-------|-------|
| Production Taylor@``model_mjd`` + legacy wrap | **16.4 ns** | **Best JUG-composed path** |
| Oracle ``phase5@bbat`` + ``track_minus2_frac_phase`` | **17.5 ns** | Step 2 target — **worse** |
| ``USE_NATIVE_BBAT_PHASE5`` + formBats ``bbat`` | ~711 ms | formBats ``bbat`` ~65 s wrong |
| pytempo ``acceptance_residual_sec`` | **0 ns** | tempo2 internal path exact |

**Oracle ``bbat`` identity (from existing JUG delay terms):**

```
bbat = model_mjd − prebinary_delay_sec / 86400
```

Matches pytempo ``bbat_mjd`` to **~330 ns RMS** on wsrt167 (Step 5 corrected Step 2
claim). JUG formBats ``term_diagnostics['bbat_mjd']`` remains **~65 s wrong** — diagnostic
only; do not use for spin.

**``torb`` for ``phase5``:** use ``compute_tempo2_torb_sec(bbat, dt_sec, PEPOCH)`` only.
Raw ``-prebinary_delay_sec`` or pytempo ``torb_sec`` arrays mis-paired with ``dt_sec``
trigger discrete ``nphase`` / ``pnNew`` wrap errors (~172 ns).

**Inter-backend structure (WSRT ``-sys``):**

| ``-sys`` | n | mean Δ | RMS Δ |
|----------|---|--------|-------|
| WSRT.P1.328 | 29 | +10.4 ns | 25.2 ns |
| WSRT.P1.328.C | 58 | +2.8 ns | 12.2 ns |
| WSRT.P1.382 | 24 | −0.6 ns | 16.0 ns |
| WSRT.P1.382.C | 56 | −8.1 ns | 14.5 ns |

Per-``-sys`` mean removal → **~15 ns** floor. ~~``jump_phase`` / tim ``-padd`` correlates
with Δ at **r ≈ −0.34**~~ — **misleading** (Step 3): padd is correct; spin error varies
by frequency band. Clock/Roemer/sat ruled out (roemer **~0.8 ns**, sat **0 ns** on harness).

### Step 3 — ``-padd`` / ``jump_phase`` (2026-07-06) — ruled out

**tempo2 source** (`formResiduals.C` ~L2073–2095, ~L2169–2179):

1. ``phase5 = spin(bbat) + phaseJ + …``
2. ``phase5 += extra`` for tim ``-padd`` / ``-radd`` (positive add to phase)
3. ``phas1 = fortran_mod(phase5[first_valid])`` — single global anchor
4. ``phase5 -= phas1``; ``nlong``; TRACK −2 ``pnNew`` / ``addPhase``

**JUG** (`simple_calculator.py` ~L1757–1770, ``compute_phase_residuals``): Taylor spin →
``jump_phase += padd`` → ``phas1@tim[0]`` → legacy TRACK −2 wrap. **Order matches tempo2.**

**pytempo oracle:** ``max |JUG jump_phase − phase_offset_turns| = 0`` on wsrt167;
``obsn_phase_offset_turns`` all zero (Parkes ``phaseOffset`` unused on this tim).

**Per-``-sys`` mean Δ (production − libstempo, mean-subtracted):**

| ``-sys`` | n | ``-padd`` | mean Δ | RMS Δ |
|----------|---|-----------|--------|-------|
| WSRT.P1.328 | 29 | 0.599804 | +10.4 ns | 25.2 ns |
| WSRT.P1.328.C | 58 | 0.599804 | +2.8 ns | 12.2 ns |
| WSRT.P1.382 | 24 | 0.603482 | −0.6 ns | 16.0 ns |
| WSRT.P1.382.C | 56 | 0.603482 | −8.1 ns | 14.5 ns |

**Not a padd bug:**

- **328 vs 328.C** share ``padd=0.599804`` but means differ by **7.6 ns** — cannot be padd.
- Uncancelled ``padd[i]−padd[0]`` predicts **−11.3 ns** for 328 backends; actual mean is
  **+10.4 ns** (wrong sign).
- Counterfactuals (temp probe ``/tmp/wsrt_padd_jump_phase_debug.py``): remove
  ``padd−padd[0]`` → **~5.6 µs** RMS; per-``-sys`` ``phas1`` → **~19 µs** RMS.

**Path ranking (same padd / jump_phase, different spin + wrap):**

| Path | RMS Δ | idx 85 Δ |
|------|-------|----------|
| Production Taylor@``model_mjd`` + legacy wrap | **16.4 ns** | **+110.5 ns** |
| ``compute_tempo2_phase5``@oracle ``bbat`` + legacy / track2 oracle | 17.5 ns | +99.9 ns |
| pytempo ``acceptance_residual_sec`` | **0 ns** | exact |

**Conclusion:** tim ``-padd`` / JUG ``jump_phase`` is **closed**. The ~10 ns inter-backend
mean spread tracks **spin/wrap** error correlated with backend band, not padd. Do **not**
wire ``phase5@bbat`` to production — no RMS gain vs Taylor (Step 2). Step 4 shows Taylor
and tempo2 ``phase2+phase3`` agree to **~0.02 ns** fractionally; the ~16 ns floor is **not**
that formulation gap (see § Step 4).

### Step 4 — Taylor@``model_mjd`` vs tempo2 spin (2026-07-06, temp probes)

**Goal:** compare JUG production spin to ``formResiduals.C`` ``phase2+phase3`` at ``bbat``.

**tempo2** (`formResiduals.C` ~L507–536): ``deltaT = (bbat−PEPOCH)·86400 + torb``;
``phase2`` uses ``int(F0)`` / fractional-day split; ``phase3`` is the F1…Fn Taylor tail.

**JUG production:** ``dt_sec = (model_mjd−PEPOCH)·86400 − total_delay`` (longdouble,
**subtract MJDs first** — see comment ~L1843); Taylor Horner on ``dt_sec``; **not**
``phase2+phase3`` at ``bbat``.

**Probe:** ``/tmp/wsrt_taylor_spin_probe.py``, ``/tmp/TAYLOR_SPIN_INVESTIGATION.md``

| Check | Result |
|-------|--------|
| ``dt_jug`` vs ``deltaT(jug bbat, torb closure)`` | max **0.000 µs** |
| ``dt_jug`` vs ``deltaT(pytempo bbat, torb closure)`` | max **0.314 µs** |
| Absolute spin Taylor vs ``phase2+phase3`` | **~10¹⁰ turns** offset (expected) |
| Fractional phase after shared ``phas1@tim[0]`` | **0.02 ns** RMS — **not** the ~16 ns gap |
| Production Taylor + legacy wrap | **16.4 ns** RMS |
| ``compute_tempo2_phase5``@jug ``bbat`` + legacy wrap | **17.5 ns** (worse) |
| Rebuild Δt from pytempo ``bbat`` + ``torb`` closure, Taylor spin | **175 ns** — ``phas1`` drift (~3×10⁻⁵ turns) despite sub-µs Δt |
| ``torb`` pairing | **Must** use ``compute_tempo2_torb_sec(bbat, dt_jug, PEPOCH)``; raw pytempo ``torb_sec`` in ``phase5`` → **172 ns** |

**Conclusions (Step 4):**

1. **Do not replace** ``dt_sec`` with ``(bbat−PEPOCH)·86400+torb`` recomputation — even when
   algebraically equal to sub-µs, it shifts ``phas1`` and breaks parity vs mean-subtracted libstempo.
2. **Taylor Horner vs tempo2 ``phase2+phase3`` is ruled out** as the ~16 ns source (0.02 ns
   fractional difference after ``phas1``).
3. **``phase5@bbat`` wiring remains ruled out** (17.5 ns vs 16.4 ns production).
4. **Next:** see § Step 5 — ``model_mjd`` / prebinary vs tempo2 ``updateBatsAll`` /
   ``calculate_bclt`` + ``pet`` (not Taylor reformulation).

**Oracle note:** libstempo ``psr.residuals()`` returns **mean-subtracted** prefit residuals
(pytempo ``removemean=True``). Compare like-for-like when probing fractional phase.

### Step 5 — Per-TOA term diff (2026-07-06, temp probes)

**Goal:** correlate JUG − libstempo residual vector (~**16.4 ns** RMS) with per-TOA diffs in
``model_mjd``, ``prebinary_delay_sec``, and tempo2 epoch fields.

**Probes:** ``/tmp/wsrt_term_diff_probe.py``, ``/tmp/wsrt_bbat_decomp_probe.py``

**Oracle note:** raw pytempo ``residual_sec`` on TRACK −2 is **not** an acceptance oracle
(~10¹⁷ µs garbage). Use ``acceptance_residual_sec`` or libstempo ``psr.residuals()``.

| Term / diff | RMS vs pytempo | r(residual) | Verdict |
|-------------|----------------|-------------|---------|
| Roemer (``jug + pt_roemer``) | **0.8 ns** | −0.03 | **Closed** |
| Sun Shapiro | **0 ns** | +0.06 | **Closed** |
| ``sat_mjd`` (JUG vs pt) | **0 ns** | — | **Closed** |
| Oracle ``bbat`` − pt ``bbat`` | **~330 ns** | **+0.19** | **Strongest ns-scale epoch correlation** |
| ``model − (sat + bat_corr + prebin/86400)`` | **~380 ns** | **+0.21** | ``model_mjd`` not tempo2 ``bat + prebin`` |
| Prebin implied from ``model−sat−bc`` − jug ``prebinary`` | **~286 ns** | **+0.21** | prebinary pairing gap |
| ``dt_jug − deltaT(pt bbat, pt torb)`` | **~207 ns** | **+0.18** | spin argument mismatch |
| ``dt_jug − deltaT(oracle bbat, oracle torb)`` | **0 ns** | — | exact by construction |
| ``torb_oracle − pt torb`` | **~262 ns** | −0.11 | **dt-closure float64** — see Step 6; **not** T2model failure |
| ``prebin − total − pt torb`` | **0.17 ns** | — | **T2model closed** (Step 6 supersedes misread of row above) |
| formBats ``bbat`` − pt ``bbat`` | **~65 s** | +0.05 | diagnostic broken (known) |
| ``model_mjd − pt bat/bbat`` | **~10¹¹ ns** | ~0.10 | different epoch definition (TCB grid) |

**tempo2 reference** (`formBats.C`, `global.C` ``updateBatsAll`` → ``calculate_bclt`` +
``formBats``):

```
bat  = sat + getCorrectionTT/86400 + (correctionTT_TB − tropo + roemer − shapiro − tdis1 − tdis2)/86400
bbat = bat − shklovskii/86400
deltaT = (bbat − PEPOCH)·86400 + torb        // formResiduals.C ~507
pet    = bbat − torb/86400
```

**JUG production spin** uses geometry ``model_mjd`` (TCB IFTE map on TDB grid for TCB pars),
**not** formBats ``model_clock`` or ``term_diagnostics['bbat_mjd']``:

```
bbat_oracle = model_mjd − prebinary/86400     // ~330 ns vs pytempo bbat on wsrt167
dt_sec      = (model_mjd − PEPOCH)·86400 − total_delay   // longdouble, MJDs first
torb_oracle = dt_sec − (bbat_oracle − PEPOCH)·86400      // NOT total − prebinary
```

**Conclusions (Step 5):**

1. **Roemer, Shapiro, sat, prebinary component sum** are closed at sub-ns — not the ~16 ns
   source.
2. **Oracle ``bbat`` is not exact vs pytempo** (~330 ns RMS; max ~630 ns at idx 85). Step 2
   “0 s RMS” claim is **withdrawn**.
3. The residual correlates most strongly with **``model_mjd`` composition** vs tempo2
   ``sat + bat_corr + prebin/86400`` (~380 ns) and the induced **``dt`` vs ``deltaT(pt)``**
   gap (~207 ns). This is the spin floor after Steps 2–4, not padd/pnNew/Taylor formulation.
   *(Retrospective: the ~380 ns ``sat+bc+prebin`` decomposition partly overlaps the
   ~237–304 ns float64 ``sat + bat_corr → bat`` artifact class — the **~330 ns oracle
   bbat gap** remains the real physics/reference signal.)*
4. **formBats diagnostic** (~65 s off ``bat``) must not be wired to production; the production
   path uses a different epoch chain that is much closer but not exact.
5. **Next probe:** compare JUG ``model_mjd`` to tempo2 ``pet`` / BCLT iteration state
   (`calculate_bclt.C`); decompose whether the ~286 ns prebinary-implied gap is IFTE/TCB epoch
   map, tropo sign in ``bat_corr``, or missing ``tdis1``/``tdis2`` split vs ``dm+sw``.

**Outliers (|resid| > 50 ns):** idx **3** (+51 ns), idx **85** (+110 ns). At idx 85,
``bbat_oracle − pt`` ≈ ``model − (sat+bc+prebin/86400)`` ≈ **+629 ns** — same tail, not a
separate padd/pnNew bug. At idx **3**, ``bbat`` matches exactly but ``torb_oracle − pt_torb ≈ +54 ns``
→ ``dt − deltaT(pt) ≈ +60 ns`` (**float64 ``dt``/``torb_oracle`` closure**, not T2model
error — ``prebin−total`` vs ``pt_torb`` is 0.17 ns).

### Step 6 — ``model_mjd`` vs ``pet`` / ``torb`` / prebinary gap (2026-07-06)

**Goal:** compare JUG ``model_mjd`` to tempo2 ``pet = bbat − torb/86400``; decompose
``T2model(bbat)`` ``torb`` vs JUG ``dt_sec`` closure; test IFTE/TCB and ``tdis`` hypotheses
for the ~286 ns prebinary-implied gap.

**Probes:** ``/tmp/wsrt_model_pet_torb_probe.py``, ``/tmp/wsrt_model_pet_torb_probe.txt``

**tempo2 reference:**

- ``calculate_bclt.C``: iterative ``delt`` at ``sat + (TT+TT_TB+dt_SSB)/86400``; **not** ``model_mjd``.
- ``formResiduals.C`` ~506–511: ``torb = T2model(bbat)``; ``pet = bbat − torb/86400``;
  ``deltaT = (bbat−PEPOCH)·86400 + torb``.

| Check | RMS | r(resid) | Verdict |
|-------|-----|----------|---------|
| Tier-1 ``pet_mjd`` vs ``bbat − torb/86400`` (separate float64 exports) | **~275 ns** | — | **float64 artifact** — identity exact at ~10⁻¹² day |
| ``pet_closure − pt pet`` (JUG oracle bbat + dt-closure torb) | **~549 ns** | +0.14 | **not Tier-1** — JUG-composed, not tempo2 ``pet`` export |
| ``model_mjd − pt pet`` | ~10¹¹ ns | +0.10 | **different epoch** (TCB grid ≠ ``bbat−torb/86400``) |
| ``model − IFTE recomputed from tdb`` | **~319 ns** | +0.08 | float64 IFTE round-trip; not primary driver |
| ``prebin_implied − jug prebinary`` | **~286 ns** | **+0.21** | ``model ≠ sat+bc+prebin/86400`` |
| ``bbat_oracle − pt bbat`` | **~330 ns** | +0.19 | (Step 5) |
| ``pt_torb − (prebin−total)`` | **0.17 ns** | −0.08 | **T2model ≈ JUG binary sign bundle** |
| ``torb_f64 closure − pt torb`` | **~262 ns** | −0.11 | float64 ``dt_sec`` / ``torb`` closure |
| **stored ``dt_sec`` vs fresh longdouble ``dt``** | **~185 ns** | — | export precision loss |
| ``torb_ld fresh − (prebin−total)`` | **0 ns** | — | algebra exact in longdouble |
| ``dt − deltaT(pt bbat, pt torb)`` | **~207 ns** | **+0.18** | spin argument vs tempo2 |
| ``dt − deltaT(oracle bbat, torb_f64)`` | **0 ns** | — | self-consistent at float64 |
| ``tdis_implied − (dm+sw)`` | ~65 s | +0.05 | **unusable** — formBats ``bat_corr`` diag broken |

**121 TOAs with ``|bbat Δ| < 1 ns``:** ``torb_f64 − pt_torb`` RMS **~185 ns**, ``dt−ΔT_pt`` RMS **~185 ns**
— spin error persists **even when ``bbat`` matches** (idx 3: ``bbat Δ=0``, ``torb Δ≈+54 ns``).

**Conclusions (Step 6):**

1. **`model_mjd` is not tempo2 `pet`.** It is the TCB IFTE-mapped delay-grid epoch used for
   Roemer/DM/Shapiro evaluation (`compatibility_providers.py` TCB branch). Comparing
   ``model_mjd`` to ``pt pet`` directly is ~10¹¹ ns off-scale; the meaningful open gap is
   ``bbat_oracle − pt bbat`` (~330 ns). *(Retrospective: ``pet_closure − pt pet`` (~549 ns)
   is not evidence tempo2's ``pet = bbat − torb/86400`` fails — Tier-1 ``pet_mjd`` vs
   recomposed ``bbat_mjd − torb/86400`` from the **same** ``toa_diagnostics`` dict differs
   only at float64 export noise (~275 ns); ``pet_closure`` uses JUG oracle epochs.)*
2. **`pt_torb` (T2model@bbat) matches `prebin−total` to 0.17 ns** — tempo2 binary delay and
   JUG ``total−prebinary`` share the same sign bundle. The mismatch is **not** “wrong T2model.”
3. **Stored `dt_sec` (float64) drifts ~185 ns from fresh longdouble `(model−PEPOCH)*86400−total`.**
   Fresh longdouble gives ``torb = prebin−total`` exactly; exported ``dt_sec`` breaks that
   identity and drives ``dt − deltaT(pt)`` even when ``bbat`` agrees.
4. **Combined spin-argument gap `dt − deltaT(pt)` ~207 ns (r≈0.18)** = stored-``dt`` drift (~185 ns)
   plus ``bbat`` mismatch (~330 ns on subset). This is the ~16 ns RMS floor after Steps 2–4,
   not Taylor vs ``phase2`` (0.02 ns fractional).
5. **Prebinary-implied gap (~286 ns)** is ``(model−sat−bat_corr)*86400 − prebinary`` — JUG
   ``model_mjd`` is not decomposable into tempo2 ``sat + bat_corr + prebin/86400`` at float64.
   IFTE round-trip (~319 ns) is a separate, lower-correlation term. **``tdis1/tdis2` vs `dm+sw`**
   cannot be tested via reconstructed ``bat_corr`` until formBats diagnostic (~65 s off) is fixed.
6. **Next (temp):** trace where ``model_mjd`` / exported ``dt_sec`` lose longdouble precision;
   compare ``calculate_bclt.C`` ``delt`` epoch to JUG ``model_mjd`` per TOA.

### Step 7 — ``dt_sec`` precision + counterfactual ``deltaT(pt)`` (2026-07-06)

**Goal:** test whether float64 ``dt_sec`` export or swapping to tempo2 ``deltaT(pt)`` closes
the ~16 ns floor; trace ``simple_calculator.py`` longdouble path.

**Probe:** ``/tmp/wsrt_dt_spin_counterfactual_probe.py``, ``/tmp/wsrt_dt_spin_counterfactual_probe.txt``

**Code path** (``simple_calculator.py``):

```
spin_model_mjd_ld = asarray(model_mjd, longdouble)   # model_mjd already float64 from geometry
dt_sec            = (model−PEPOCH)*86400 − total_delay_sec   # total_delay also float64
compute_phase_residuals(dt_sec, ...)               # Taylor Horner in longdouble
dt_sec_ld / dt_sec export                          # ld array == float64 values (no extra bits)
```

| Check | Result | Verdict |
|-------|--------|---------|
| ``dt_sec_ld == dt_sec`` export | **0 ns** | float64 inputs cap precision |
| ``dt_ld`` vs fresh ld recompute from exported model/total | **~185 ns** RMS | float64 ``model_mjd`` bottleneck |
| ``dt_ld − deltaT(pt bbat, pt torb)`` | **~207 ns** RMS, r≈0.18 | correlates with residual |
| Taylor@production ``dt_ld`` | **16.4 ns** | **best JUG Taylor path** |
| Taylor@``deltaT(pt)`` (counterfactual) | **~173 ns** | **WORSE** — do not swap spin argument |
| Taylor@``dt`` from IFTE(tdb) round-trip | **~257 ns** | **WORSE** |
| Taylor@float64 export only | **17.1 ns** | +0.7 ns vs ld Horner (minor) |
| ``model − (sat+bc+prebin/86400)`` | **~380 ns** | (Step 5–6) drives ``bbat`` gap |
| ``pt_torb − (prebin−total)`` | **0.17 ns** | T2model closed |

**Conclusions (Step 7):**

1. **Production Taylor already runs in longdouble** — the ~185 ns ``dt`` drift is from
   **float64 ``model_mjd`` and ``total_delay_sec`` inputs**, not from the Horner loop or
   ``dt_sec`` export (export matches ``dt_sec_ld`` exactly).
2. **`dt` correlates with ``deltaT(pt)`` gap (~207 ns) but replacing ``dt→deltaT(pt)`` in
   Taylor+TRACK−2 worsens RMS to ~173 ns.** Emission-time ``dt`` and tempo2 ``deltaT`` are not
   interchangeable spin arguments despite Step 4 showing ~0.02 ns Taylor vs ``phase2+phase3``
   fractional diff **at fixed ``dt``**.
3. **IFTE round-trip alone is not the fix** — recomputing ``model_mjd`` from ``tdb`` via IFTE
   and rebuilding ``dt`` yields ~257 ns RMS.
4. **Fix direction:** improve ``model_mjd`` formation to match tempo2 ``calculate_bclt`` /
   ``formBats`` epoch chain **while keeping** emission-time Taylor+legacy TRACK−2 (not naive
   ``deltaT(pt)`` substitution, not ``phase5@bbat`` wiring).

### Step 8 — JAX float64 compensated spin prototype (2026-07-06)

**Goal:** test two-part ``dt`` + compensated Taylor Horner (JAX float64) without changing
TRACK −2.

**Probe:** ``/tmp/wsrt_jax_compensated_spin_probe.py``, ``/tmp/wsrt_jax_compensated_spin_probe.txt``

| Path | RMS vs libstempo |
|------|-------------------|
| Production ``dt_sec_ld`` + ld Horner | **16.43 ns** |
| JAX two-part ``dt`` + compensated Taylor | 15.36 ns* |
| JAX plain float64 ``dt`` + plain Taylor | 15.36 ns* |

\*All JAX spin variants identical before TRACK −2; ~1 ns “gain” vs production is from
float64 ``model_mjd`` replay artifact, not compensated arithmetic. **No production fix.**

**Conclusion:** Horner / two-part float64 micro-optimizations are **ruled out** again.

### Step 9 — epoch-chain parity review (2026-07-06)

**Goal:** localize ``model_mjd`` / ``bbat`` gap vs tempo2; test counterfactual epoch replacements.

**Probe:** ``/tmp/wsrt_epoch_chain_probe.py``, ``/tmp/wsrt_epoch_chain_probe.txt``

**Do not describe this as a tempo2 “definition difference”.** Tempo2 has one source path
(``calculate_bclt.C`` → ``formBats.C`` → ``formResiduals.C``). Gaps are JUG implementation /
probe-variable mismatches unless proven against ``ref-packages/tempo2/``.

| Check | RMS | Verdict |
|-------|-----|---------|
| Production ``dt_sec_ld`` replay | **16.43 ns** | baseline |
| ``IFTE(tdb_ld)`` → recompute ``dt_ld`` (true ld) | **16.43 ns** | in-flight chain self-consistent |
| ``model_mjd`` float64 replay | **186 ns** | export/replay artifact only |
| ``oracle_bbat − pt_bbat`` (float64) | **330 ns** | r≈0.19 |
| ``oracle_bbat − pt_bbat`` (longdouble) | **212 ns** | float64 decomposition noise |
| ``torb_ld − pt_torb`` | **0.2 ns** | **closed** in longdouble |
| ``prebin − total`` vs ``pt_torb`` | **0.2 ns** | T2 sign bundle closed |
| ``model − (sat+bc+prebin/86400)`` | **380 ns** | JUG ``model_mjd`` ≠ tempo2 bat identity |
| ``pt_bbat + prebin/86400`` counterfactual | **243 ns** | **worse** |
| ``sat + bat_corr + prebin/86400`` counterfactual | **315 ns** | **worse** |
| ``deltaT(pt bbat, pt torb)`` spin arg | **173 ns** | **worse** — do not swap |

**Conclusions (Step 9):**

1. Production path ``IFTE(tdb_ld)`` → ``dt_sec_ld`` → Taylor + TRACK −2 is **best at 16.4 ns**.
2. Naive epoch substitutions (``pt_bbat``, ``formBats model_clock``, ``pet``) **all worsen**
   residuals — not fix targets.
3. The ~330 ns ``bbat`` gap is real but **does not map 1:1** to the 16 ns floor when used as
   a naive spin-argument swap; fixing it requires matching tempo2’s **signed term chain**, not
   relabelling epochs.
4. **Next:** ``formBats.C`` signed-term reconstruction (Step 10), not more Taylor tuning.

### Step 10 — formBats.C signed-term reconstruction (2026-07-06)

**Goal:** brute-force tempo2 ``formBats.C`` sign bundle vs pytempo ``bat_corr``; identify
which JUG term(s) fail to compose into ``obsn.batCorr``.

**Reference:** ``ref-packages/tempo2/formBats.C`` lines 67–71:

```
batCorr = TT + (TT_TB - tropo + roemer - shapiro - tdis1 - tdis2) / 86400
```

**Probe:** ``/tmp/wsrt_formbats_sign_probe.py``, ``/tmp/wsrt_formbats_sign_probe.txt``

| Check | Result | Verdict |
|-------|--------|---------|
| pytempo ``sat + bat_corr − bat`` | **237 ns** | **float64 export artifact** — identity exact at ~10⁻¹² day in ``long double`` |
| export vs implied (``bat − sat``) | **237 ns** | same artifact class — **not** inconsistent tempo2 physics |
| Canonical signs + JUG ``tt``/``tt_tb`` + pt roemer/shap + dm+sw | **+65.0 s mean offset** vs bat_corr | **not a sign bug — baseline mismatch** |
| ``tdis`` inverted from formBats identity | mean **−64 s** vs ``dm+sw`` mean **+1.3 s** | JUG dm/sw ≠ tempo2 ``tdis1+tdis2`` in batCorr chain |
| JUG ``correction_tt`` mean | **~0 s** | ``utc_to_tdb`` mean **~65 s** — different quantity |
| ``tempo2_clock.py`` bundled ``TT+TT_TB−prebinary`` | same **~65 s** gap | bundled prebinary is **wrong abstraction** |
| ``oracle_bbat − pt_bbat`` | **330 ns** | unchanged — production spin path unaffected |
| ``deltaT(bbat_recon, torb_ld)`` counterfactual | **~855 µs** | formBats recon **much worse** than production |

**Conclusions (Step 10):**

1. The ~65 s ``tempo2_clock.py`` formBats diagnostic gap is **not** fixed by sign flips: JUG
   ``correction_tt`` + ``correction_tt_tb`` do not sit on the same baseline as pytempo
   ``bat_corr`` when combined with JUG delay exports.
2. Inverting formBats with JUG terms yields ``tdis_implied ≈ −64 s``, not ``dm+sw`` — the
   dispersion terms are not entering ``batCorr`` the way ``formBats.C`` expects.
3. **`compute_formbats_arrival()`` must not be used as an oracle** until rebuilt term-by-term
   with tempo2 signs (no bundled ``prebinary`` subtraction).
4. The **237 ns ``sat + bat_corr − bat`` gap is float64 export noise**, not evidence that
   pytempo ``bat_corr_days`` disagrees with tempo2 (§0, Step 14 retrospective).
5. **Next implementation target:** diff JUG ``compute_correction_tt_tb_sec`` against tempo2
   ``tt2tdb.C`` ``correctionTT_TB`` **as used inside formBats** — close the **+65 s** gap
   (Step 11).

### Step 11 — getCorrectionTT / correctionTT_TB vs libstempo batCorrs (2026-07-06)

**Goal:** localize the Step 10 **+65 s** batCorr offset to ``getCorrectionTT`` vs
``correctionTT_TB`` vs delay exports.

**Probe:** ``/tmp/wsrt_clock_chain_probe.py``, ``/tmp/wsrt_clock_chain_probe.txt``  
**Harness:** ``jug/testing/tempo2_clock_chain.py``, ``tests/test_tempo2_clock_chain_formbats.py``

| Check | Result | Verdict |
|-------|--------|---------|
| ``sat_mjd`` vs libstempo ``stoas`` | **0 ns** | **closed** |
| libstempo ``batCorrs`` vs pytempo ``bat_corr_days`` | **0 ns** | oracle consistent |
| JUG ``correction_tt`` (``getCorrectionTT``) | mean **~0 s** | matches tempo2 role |
| JUG ``correction_tt_tb`` | mean **+14.4 s** | |
| TT+TT_TB implied from ``batCorrs`` inversion | mean **+79.4 s** | tempo2-composed target |
| **TT_TB gap** (implied − JUG) | mean **+64.986 s** | **matches ``utc_to_tdb`` mean** |
| formBats canonical vs ``batCorrs`` offset | **+64.986 s** | same gap — not delay signs |
| Production ``dt_sec_ld`` spin | **16.43 ns** | unchanged baseline |

**Conclusions (Step 11):**

1. **Step 12 correction:** the apparent **+65 s ``correctionTT_TB`` gap** was a **tdis confound**
   — inverting ``batCorrs`` with JUG ``dm+sw`` as ``tdis1+tdis2`` falsely imputes ``utc_to_tdb``
   to TT_TB. JUG ``compute_correction_tt_tb_sec`` mean **~14.4 s** matches ``tt2tdb.C``.
2. ``getCorrectionTT`` (~0 s) and libstempo ``batCorrs`` == pytempo ``bat_corr`` (0 ns) are
   **closed**.
3. Naive ``formBats.C`` closure with ``dm+sw`` fails (+65 s); with **inverted ``tdis_implied``**
   closes at 0 ns — the dispersion slot in ``batCorrs`` is **not** JUG ``dm+sw`` export.
4. **Next (Step 12):** trace production ``model_mjd`` / ``tdb_mjd`` chain vs ``batCorrs``.

### Step 12 — ``batCorrs`` vs production ``model_mjd`` epoch chain (2026-07-06)

**Goal:** trace ``tt2tdb.C`` / ``calculate_bclt.C`` vs JUG JAX ``tdb_mjd`` +
``convert_tdb_epoch_to_tempo2_tcb``; explain ``batCorrs`` without relabelling ``utc_to_tdb``
as missing ``correctionTT_TB``.

**Reference:** ``ref-packages/tempo2/formBats.C`` (``batCorr`` per-term signs);
``calculate_bclt.C`` (``delt`` uses ``sat + (getCorrectionTT + correctionTT_TB + dt_SSB)/86400``);
JUG ``compatibility_providers._compute_tempo2_tcb_geometry_terms`` (IFTE ``model_mjd``).

**Probe:** ``/tmp/wsrt_batcorr_epoch_chain_probe.py``, ``/tmp/wsrt_batcorr_epoch_chain_probe.txt``  
**Harness:** ``jug/testing/tempo2_clock_chain.py`` (``compare_batcorr_epoch_chain``),
``tests/test_tempo2_batcorr_epoch_chain.py``

| Check | Result | Verdict |
|-------|--------|---------|
| ``batCorrs`` vs ``(model_mjd−sat)×86400 − prebinary`` | **286 ns** RMS | **oracle identity** |
| ``batCorrs`` vs ``utc_to_tdb + (model−tdb) − prebinary`` | **286 ns** RMS | equivalent split |
| ``model_mjd − tdb_mjd`` vs ``IFTE_KM1*(tdb−MJD0)+Teph0`` | **260 ns** RMS | IFTE epoch map |
| ``model_mjd − tdb_mjd`` vs ``tt2tdb tt_tb`` | **1.18 ms** RMS (r=1) | same mean (~14.4 s), per-TOA scatter |
| Naive ``formBats`` ``tt+tt_tb+delays−(dm+sw)`` | **~65 s** offset | **wrong tdis slot** |
| ``formBats`` with inverted ``tdis_implied`` | **0 ns** | closes algebraically |
| Production ``dt_sec_ld`` spin | **16.43 ns** | unchanged |

**Conclusions (Step 12):**

1. libstempo ``batCorrs`` is **not** naively ``getCorrectionTT + correctionTT_TB + physical
   delays`` with JUG ``dm+sw`` — it equals the **production JAX epoch chain**:
   ``(model_mjd − sat) − prebinary/86400`` (days).
2. Decomposition: ``batCorrs ≈ utc_to_tdb + (model_mjd−tdb_mjd) − prebinary`` where
   ``utc_to_tdb`` (~65 s) is site UTC→geocentric TDB (``compute_tdb_standalone``) and
   ``model−tdb`` (~14 s) is ``convert_tdb_epoch_to_tempo2_tcb`` IFTE TCB map (distinct from
   per-TOA ``tt2tdb`` ``obsTerm`` scatter).
3. ``tempo2_clock.py`` ``compute_formbats_arrival()`` (bundled ``prebinary``) and naive
   ``formBats.C`` term replay **cannot** match ``batCorrs`` until rebuilt from production
   ``model_mjd`` / ``tdb_mjd`` exports.
4. **Next (Step 13):** wire diagnostic ``batCorr`` / ``bbat`` from production epochs; re-test
   ``oracle_bbat`` (~330 ns) and spin counterfactuals without changing production ``dt_sec_ld``.

### Step 13 — model-epoch batCorr/bbat diagnostic rebuild (2026-07-06)

**Goal:** prototype JAX-pipeline diagnostic ``batCorr``/``bbat`` from production
``model_mjd`` exports — **temp-only** (``/tmp/step13_model_batcorr.py``), no
``tempo2_clock.py`` changes yet.

**Probe:** ``/tmp/wsrt_batcorr_from_model_probe.py``, ``/tmp/wsrt_batcorr_from_model_probe.txt``  
**Tests:** ``tests/test_tempo2_batcorr_from_model_probe.py``

| Check | Result | Verdict |
|-------|--------|---------|
| Bundled ``tempo2_clock.py`` bat vs libstempo | **64.99 s** RMS | **broken** |
| Model ``batCorr = (model−sat)×86400 − prebinary`` | **286 ns** | **closes libstempo** |
| Model ``bat = model − prebinary/86400`` | **380 ns** vs lib bat | **closes** |
| Model ``bbat`` vs oracle ``model−prebin/86400`` | **0 ns** | Shklovskii ≈ 0 |
| Model/oracle ``bbat`` vs pytempo ``bbat`` | **330 ns** | **unchanged** |
| Production ``dt_sec_ld`` spin | **16.43 ns** | unchanged |
| ``dt_ld`` vs ``deltaT(bbat_ld,torb_ld)`` replay | **~185 ns** | float64 export artifact |

**Conclusions (Step 13):**

1. **JAX pipeline target:** ``batCorr_sec = (model_mjd − sat)×86400 − prebinary_delay_sec``;
   ``bat_mjd = model_mjd − prebinary/86400``; ``bbat = bat − shk/86400``. Drop bundled
   ``tt+tt_tb−prebinary`` in future ``compute_formbats_arrival()``.
2. Model-epoch rebuild closes the **~65 s** bundled formBats diagnostic; **~330 ns ``bbat``**
   vs pytempo is a **separate** open gap (not fixed by batCorr source change).
3. Do **not** use float64-export ``deltaT(bbat,torb)`` for spin parity — ~185 ns replay cap
   (Steps 7/9); production ``dt_sec_ld`` path remains authoritative at **16.4 ns**.
4. **Next (Step 14):** ``torb`` / ``formResiduals.C`` / in-flight ``model_mjd_ld`` vs the
   **~330 ns ``bbat``** oracle gap.

### Step 14 — ``torb`` closed; ``bbat`` epoch mismatch (2026-07-06)

**Goal:** trace tempo2 ``formResiduals.C`` spin identity on wsrt167 (BINARY T2) and
determine whether the **~330 ns ``bbat``** gap is ``torb``, ``pet``, or emission-epoch
reference — **temp-only** (``/tmp/wsrt_torb_bbat_probe.py``), no source-tree changes.

**tempo2 reference** (``formBats.C`` / ``formResiduals.C``):

```text
batCorr = TT/86400 + (TT_TB − trop + roemer − shap − tdis1 − tdis2)/86400
bat     = sat + batCorr
bbat    = bat − shk/86400
deltaT  = (bbat − PEPOCH)×86400 + torb          // spin argument
pet     = bbat − torb/86400
torb    = T2model(psr, p, i, −1, 0) at obsn.bbat
```

**Probe:** ``/tmp/wsrt_torb_bbat_probe.py``, ``/tmp/wsrt_torb_bbat_probe.txt``

| Check | Result | Verdict |
|-------|--------|---------|
| ``prebinary − total`` vs pytempo ``torb_sec`` | **0.17 ns** | **CLOSED** |
| libstempo ``binarydelay`` vs ``torb_sec`` | **~705 s** | **stale** (zeros on fresh libstempo construct) — use ``torb_sec`` / ``prebinary − total`` |
| libstempo ``stoas`` / ``batCorrs`` vs pytempo | **0 ns** | Tier-1 exports agree |
| pytempo ``sat + bat_corr`` vs ``bat_mjd`` | **304 ns** | float64 composition artifact |
| pytempo ``pet`` vs ``bbat − torb/86400`` | **275 ns** | float64 composition artifact |
| libstempo ``pets`` vs pytempo ``pet_mjd`` | **0 ns** | Tier-1 closed |
| oracle ``model − prebin/86400`` vs pytempo ``bbat`` | **330 ns** | **OPEN (physics/reference)** |
| ``model_mjd`` vs ``pet_mjd`` | **~312 s** mean | different quantities |
| ``dt_ld`` vs ``(model−PE)×86400 − total`` | **~185 ns** | float64 ``model_mjd`` export (Steps 7/9) |
| ``deltaT(oracle, pt_torb)`` vs ``dt_ld`` | **~262 ns** | bbat reference drives gap |
| ``deltaT(pt_bbat, pt_torb)`` vs ``dt_ld`` | **~207 ns** | tempo2-native spin worse than production |
| Production ``dt_sec_ld`` spin | **16.43 ns** | unchanged — best path |
| ``corr(bbat_gap, residual)`` | **≈ 0.19** | moderate, not dominant |

**pytempo / libstempo oracle note (2026.4.1):** both wrap the same ``libtempo2.so.2``;
``residuals()`` / ``acceptance_residual_sec`` / ``bbat_mjd`` are Tier-1 oracles and agree
at 0 ns. Naïve recompositions ``sat + bat_corr → bat`` or ``bbat − torb/86400 → pet`` can
show **~275–304 ns** from float64 reads of separate ``long double`` fields — not a second
physics engine. libstempo ``binarydelay`` returns **zeros** on a fresh construct — use
``torb_sec`` or ``prebinary − total``.

**Conclusions (Step 14):**

1. **`torb` is closed** at sub-ns via ``prebinary_delay_sec − total_delay_sec = pytempo
   torb_sec`` (``T2model`` at tempo2 ``bbat``). The ~330 ns gap is **entirely in
   ``bat``/``bbat`` epoch**, not binary delay.
2. **``model_mjd − prebinary/86400`` is JUG's emission spin reference**, internally
   consistent with production ``dt_sec_ld = (model−PE)×86400 − total``. It is **not**
   tempo2 ``obsn.bbat`` (330 ns vs pytempo Tier-1 ``bbat_mjd``).
3. **Do not swap production spin** to tempo2-native ``deltaT(bbat, torb)`` with pytempo
   ``bbat`` — spin counterfactual **~173 ns** RMS, worse than **16.4 ns** Taylor@``model_mjd``.
4. **JAX pipeline split:** (a) diagnostic ``batCorr`` from ``model_mjd`` chain (Step 13,
   286 ns vs lib); (b) tempo2-native spin needs in-flight ``obsn.bbat`` from ``formBats``,
   not ``model − prebinary/86400``; (c) production Taylor path keeps ``model_mjd + total``.
5. **Next (Step 15):** trace why JUG ``model_mjd`` emission epoch differs from tempo2
   ``obsn.bbat`` by ~330 ns while ``batCorr`` closes at 286 ns — likely IFTE/TCB geometry
   vs ``formBats`` in-flight ``long double`` chain, not pytempo/libstempo disagreement.

### Oracle tier retrospective — Steps 4–14 (2026-07-06)

Re-read of Steps 4–14 after documenting pytempo/libstempo Tier-1 rules (§0). **Core
parity conclusions unchanged** (~16.4 ns production floor; ~330 ns ``bbat`` open; ``torb``
closed at 0.17 ns). Rows below are **corrected interpretations** only.

| Step | Original read | Retrospective verdict |
|------|---------------|---------------------|
| **4** | raw ``torb_sec`` in ``phase5`` → 172 ns | **Stands** — failure is wrong ``bbat``/``dt`` pairing for spin, not bad ``torb_sec`` export |
| **5** | ``torb_oracle − pt_torb`` ~262 ns ⇒ T2model ≠ ``prebin−total`` | **Withdrawn** — 262 ns is ``compute_tempo2_torb_sec(dt closure)`` float64 drift; ``prebin−total`` vs ``torb_sec`` is **0.17 ns** (Step 6) |
| **5** | idx 3 ``torb Δ≈+54 ns`` with exact ``bbat`` | **Reinterpreted** — same dt-closure float64 artifact, not isolated T2 bug |
| **6** | ``pet_closure − pt pet`` ~549 ns ⇒ tempo2 pet identity fails | **Withdrawn** — JUG-composed ``pet_closure`` is not Tier-1; ``pet = bbat − torb/86400`` holds at ~10⁻¹² day; separate float64 exports show ~275 ns |
| **10** | ``sat + bat_corr − bat`` ~237 ns ⇒ bat_corr inconsistent | **Reinterpreted** — float64 export artifact (same class as Step 14 ~304 ns); **+65 s** baseline mismatch conclusion **unchanged** |
| **11–12** | lib ``batCorrs`` == pytempo ``bat_corr`` at 0 ns | **Stands** — Tier-1 validated |
| **13** | model ``batCorr`` vs lib at 286 ns | **Stands** — compare Tier-1 ``batCorrs``, not recomposed ``sat+bc→bat`` |
| **14** | ``binarydelay`` ~705 s off ``torb`` | **Clarified** — ``binarydelay`` reads **zeros** on fresh libstempo construct, not an alternate physics field |
| **WSRT table** | ``torb`` vs ``−binarydelay`` 0.17 ns | **Withdrawn** — coincidental/wrong property; use ``prebin−total`` or ``torb_sec`` |
| **WSRT table** | ``bbat`` vs lib ``toas`` | **Withdrawn** — ``toas`` is not a Tier-1 ``bbat`` oracle |
| **WSRT table** | ``pets − torb/86400`` ~370 ns | **Reinterpreted** — float64 recomposition artifact; Tier-1 ``pets`` / ``pet_mjd`` agree at 0 ns |

**Unchanged open work:** JUG ``model_mjd`` emission spin reference vs tempo2 Tier-1
``bbat_mjd`` (~330 ns); production Taylor path remains best at 16.4 ns.

### Step 15 — ``model_mjd`` vs ``obsn.bbat`` decomposition (2026-07-06)

**Goal:** explain why JUG ``model_mjd − prebinary/86400`` differs from tempo2 Tier-1
``bbat_mjd`` by ~330 ns while Step 12 ``batCorrs`` identity closes at ~286 ns — **temp-only**
(``/tmp/wsrt_model_bbat_step15_probe.py``).

**Key algebra (Steps 12–14):**

```text
oracle_bbat = model_mjd − prebinary/86400 = sat + batCorr_model/86400   (exact)
tempo2 bat  = sat + batCorr                                               (formBats.C)
tempo2 bbat = bat − shk/86400
batCorrs    ≈ (model_mjd − sat)×86400 − prebinary   (286 ns vs libstempo)
batCorrs    ≈ utc_to_tdb + (model_mjd − tdb_mjd)×86400 − prebinary        (same 286 ns)
```

**Probe:** ``/tmp/wsrt_model_bbat_step15_probe.py``, ``/tmp/wsrt_model_bbat_step15_probe.txt``

| Check | Result | Verdict |
|-------|--------|---------|
| oracle ``model−prebin/86400`` vs ``bbat_mjd`` | **330 ns** | **OPEN** |
| oracle vs lib ``stoas+batCorrs`` | **380 ns** | ≈ √(286²+304²) quadrature |
| lib ``bat`` vs ``bbat_mjd`` | **304 ns** | Tier-1 float64 bat read (Shklovskii ≈ 0) |
| ``batCorr`` model identity vs lib | **286 ns** | **primary lever** |
| ``utc_to_tdb + (model−tdb) − prebin`` vs lib | **286 ns** | same — whole chain, not prebin alone |
| ``corr(batCorr error, bbat gap)`` | **−0.53** | moderate |
| ``(model−tdb)`` IFTE per-TOA scatter | **260 ns** | secondary; mean offset **131 µs** (constant) |
| ``model`` vs IFTE recomputed from ``tdb`` | **319 ns** | float64 ``model_mjd`` export |
| prebinary internal sum | **0 ns** | ``roemer_shapiro+dm+sw+tropo`` closed |
| ``corr(batCorr error, residual)`` | **−0.21** | tracks 16 ns floor weakly |
| Taylor@``deltaT(Tier-1 lib bbat, pt torb)`` | **252 ns** | **worse** than production **16.4 ns** |
| Production ``dt_sec_ld`` | **16.43 ns** | unchanged |

**Conclusions (Step 15):**

1. **The ~330 ns ``bbat`` gap is not a separate physics engine.** It decomposes into the
   Step 12 **~286 ns ``batCorr`` model-chain error** (JUG float64 ``model_mjd``/IFTE exports
   vs tempo2 in-flight ``long double`` ``batCorrs``) plus **~304 ns Tier-1 float64 ``bat``
   read** noise. ``oracle − lib bat`` (~380 ns) ≈ quadrature of the two.
   **Step 16 refinement:** only **~72 ns** of the 286 ns is float64 ``batCorr``/``utc_to_tdb``
   arithmetic; **~214 ns** persists on ld replay — see Step 16.
2. **`model_mjd` is the IFTE-mapped TCB geometry epoch** used for Roemer/DM/Shapiro
   (`compatibility_providers._compute_tempo2_tcb_geometry_terms`). It is **not** tempo2
   ``calculate_bclt.C``'s iterative ``delt`` epoch. The bat epoch is **`sat + batCorr`**, not
   ``model − prebin/86400`` as a separate tempo2 product — that subtraction is a JUG spin
   bookkeeping identity that happens to approximate ``sat + batCorr_model`` algebraically.
3. **Prebinary composition is closed** (0 ns internal sum). The 286 ns ``batCorr`` error is
   **not** a missing tropo/DM term in ``prebinary`` — it is the accumulated epoch-chain /
   float64 export mismatch in ``utc_to_tdb + IFTE(model−tdb) − prebinary``.
4. **Do not fix the 16 ns floor by substituting Tier-1 ``bbat`` into ``deltaT(bbat,torb)``
   alone** — counterfactual spin **~252 ns** RMS. Production Taylor@``model_mjd`` +
   ``dt_sec_ld`` remains best until **both** ``batCorr`` formation and spin reference are
   unified in longdouble.
5. **JAX pipeline target (updated):** (a) evaluate delays at ``tdb_mjd`` + IFTE map in
   ``longdouble``; (b) form ``batCorr`` matching tempo2 ``formBats.C`` in-flight (close 286 ns);
   (c) ``bbat = sat + batCorr − shk/86400``; (d) keep production ``dt = (model−PE)×86400 − total``
   for Taylor spin until (b) closes and spin path is revisited.
6. **Step 16 (done):** longdouble replay — **286 ns does not collapse**; see below.

### Phase D Step 16 — longdouble ``model_mjd`` / ``batCorr`` replay (2026-07-06)

**Goal:** Test whether the Step 12/15 **~286 ns ``batCorr``** error collapses when
``convert_tdb_epoch_to_tempo2_tcb`` and ``utc_to_tdb`` run from ``tdb_mjd_ld`` without
float64 export (cf. Step 7 ~185 ns ``dt`` replay cap).

**Probe:** ``/tmp/wsrt_model_bbat_step16_probe.py`` → ``/tmp/wsrt_model_bbat_step16_probe.txt``

| Check | Result | Verdict |
|-------|--------|---------|
| ``model_f64`` vs IFTE(``tdb_ld``) | **0 ns** | export lossless — production ld model preserved |
| ``model_f64`` vs IFTE(``tdb_f64``) | **319 ns** | round-trip from float64 ``tdb`` loses IFTE precision |
| ``utc_to_tdb`` f64 vs ld(``tdb_ld−sat``) | **180 ns** | float64 ``utc_to_tdb`` export truncation |
| ``dt_sec_ld`` vs ld IFTE(``tdb_ld``) replay | **0 ns** | spin already on full ld model |
| ``model_f64`` vs implied from ``dt_ld+total`` | **186 ns** | ``total_delay_sec`` float64 in ``dt`` (Step 7 cap) |
| ``batCorr`` f64 identity vs lib | **286 ns** | Step 15 baseline |
| ``batCorr`` ld IFTE(``tdb_ld``) vs lib | **214 ns** | **partial** — Δ≈72 ns, **not** collapse |
| ``batCorr`` ld IFTE(``tdb_f64``) vs lib | **270 ns** | tdb export hurts ld replay too |
| ``corr(batCorr error, residual)`` | **−0.21 / −0.23** | weak — batCorr gap ≠ 16 ns floor driver |
| Taylor@production ``dt_sec_ld`` | **16.43 ns** | unchanged |
| Taylor@dt from IFTE(``tdb_ld``) replay | **16.43 ns** | identical — no spin gain |
| Taylor@dt from lib-implied model | **271 ns** | tempo2 ``batCorr`` epoch ≠ JUG emission model |

**Conclusions (Step 16):**

1. **The ~286 ns ``batCorr`` error does not collapse on longdouble replay.** Best ld path
   (**214 ns**) improves only **~72 ns** vs float64 identity test. **~214 ns persists** — this
   is **physics/epoch mismatch**, not ``model_mjd`` float64 export alone.
2. **Production spin already uses longdouble ``model_mjd`` internally** before float64 export
   (``spin_model_mjd_ld`` at ``simple_calculator.py`` L1846–1847). ``model_f64`` export vs
   IFTE(``tdb_ld``) is **0 ns** — the 16 ns floor is **not** fixable by keeping ``model_mjd``
   in ld for export alone.
3. **Step 15 hypothesis refined:** the ~286 ns splits roughly **~72 ns** float64 ``batCorr``
   arithmetic / ``utc_to_tdb`` export + **~214 ns** JUG IFTE emission ``model_mjd`` vs tempo2
   in-flight ``batCorrs`` epoch (``formBats.C`` / ``calculate_bclt.C`` Roemer ``delt``).
4. **``total_delay_sec`` float64** still caps ``dt`` round-trip at **~186 ns** (Step 7); orthogonal
   to the **~214 ns** batCorr physics gap.
5. **Do not swap spin to lib-implied model** ``sat + (batCorr+prebin)/86400`` — counterfactual
   **271 ns** RMS. Production Taylor@``model_mjd`` + ``dt_sec_ld`` remains best.
6. **Step 17 (done):** formBats / calculate_bclt epoch trace — see below.

### Phase D Step 17 — formBats / calculate_bclt Roemer epoch vs JUG ``model_mjd`` (2026-07-06)

**Goal:** Trace the Step 16 **~214–286 ns ``batCorr``** residual to tempo2
``formBats.C`` / ``calculate_bclt.C`` Roemer iteration epoch vs JUG IFTE ``model_mjd``.

**Sources:** ``ref-packages/tempo2/formBats.C`` (L67–71 ``batCorr``),
``calculate_bclt.C`` (L131–132 iterative ``delt`` at ``sat+(TT+TT_TB+dt_SSB)/86400``),
JUG ``compatibility_providers._compute_tempo2_tcb_geometry_terms``.

**Probe:** ``/tmp/wsrt_formbats_roemer_epoch_step17_probe.py`` →
``/tmp/wsrt_formbats_roemer_epoch_step17_probe.txt``

| Check | Result | Verdict |
|-------|--------|---------|
| formBats replay vs lib ``batCorrs`` | **0 ns** | tempo2 source algebra **closed** |
| Roemer JUG vs ``−``pytempo | **0.8 ns** | sign flip only (Step 5) — magnitude closed |
| ``(model−sat)×86400 − prebin`` vs lib | **286 ns** | Step 12/15 baseline |
| IFTE ``(model−tdb)`` per-TOA scatter | **260 ns** | tracks gap (``r≈−0.57``) |
| ``corr(clock_excess, delay_excess)`` | **≈ −1** | 65 s mean offsets cancel per TOA |
| ``corr(batCorr gap, 16 ns residual)`` | **−0.21** | weak — gap ≠ spin floor driver |
| ``model−model_clock`` mean | **+65.0 s** | ``model_clock = sat+(TT+TT_TB)/86400`` |
| ``model−tdb`` mean (IFTE linear) | **+14.4 s** | IFTE TCB map offset |
| JUG ``tempo2_clock`` ``bbat_mjd`` vs pt | **~65 s** | omits ``utc_to_tdb`` — diagnostic only |
| oracle ``model−prebin/86400`` vs pt ``bbat`` | **330 ns** | unchanged |

**Conclusions (Step 17):**

1. **tempo2 ``formBats.C`` is self-consistent** — replay with inverted ``tdis`` closes
   lib ``batCorrs`` at **0 ns**. The ~286 ns gap is **not** a sign error or Roemer-formula bug.
2. **JUG vs tempo2 use different ``batCorr`` decompositions** that cancel ~65 s mean offsets:
   - JUG: ``(IFTE(tdb) − sat)×86400 − prebinary`` (single emission-epoch clock slot)
   - tempo2: ``TT + TT_TB − tropo + roemer − shap − tdis`` (``formBats.C`` split)
   Clock and delay slots are anticorrelated (**``r ≈ −1``**); the **~286 ns** residual is
   **imperfect cancellation**, dominated by IFTE per-TOA scatter (**~260 ns**).
3. **Roemer evaluation epoch:** ``calculate_bclt.C`` iter-0 uses ``model_clock =
   sat+(TT+TT_TB)/86400``; JUG geometry uses ``model_mjd = IFTE(tdb)`` (~65 s + ~14 s
   offset). Roemer **magnitude** still matches pytempo at **0.8 ns** — epoch difference
   enters via the clock/delay **packaging**, not a separate Roemer export bug.
4. **Do not wire ``tempo2_clock.py`` ``bbat_mjd`` to production** — ~65 s baseline error
   (missing ``utc_to_tdb`` in ``model_clock`` path).
5. **Do not expect a ``batCorr``-only JUG fix to close the ~16 ns spin floor** — gap
   correlates weakly with residual (**``r≈−0.21``**). Production Taylor@``model_mjd`` +
   ``dt_sec_ld`` remains best (Step 18 confirmed **16.4 ns** is the JUG spin floor).
6. **Step 18 (done):** Taylor / ``formResiduals.C`` — see § Phase D Step 18; **0 ns** needs
   the JAX tempo2-native chain (§ above), not spin-only JUG patches.

**JUG implementation guidance (Step 17):**

| Action | Implement now? | Rationale |
|--------|----------------|-----------|
| Wire ``tempo2_clock`` ``bbat``/``phase5`` to production | **No** | ~65 s / ~252–271 ns counterfactuals worse than **16.4 ns** |
| Export ``batcorr_model_sec`` diagnostic from Step 13 identity | **Optional** | Helps probes; zero production impact |
| Fix ``tempo2_clock.py`` bundled ``bbat`` diagnostic | **Defer** | Diagnostic-only; use ``model_mjd`` oracle instead |
| Re-evaluate delays at ``calculate_bclt`` iterative epoch | **Investigate first** | Large change; may affect batCorr **and** spin — needs isolated probe |
| End-to-end JAX ``formBats`` + ``calculate_bclt`` chain | **Yes (JAX track)** | Required for libstempo parity; cannot shortcut via ``(model−sat)−prebin`` |

### Phase D Step 18 — Taylor / ``formResiduals.C`` spin bookkeeping (2026-07-06)

**Goal:** explain why **~16 ns** persists when TRACK −2 wrap is closed (Step 1) and whether
switching to tempo2 ``phase2+phase3`` + ``pnNew`` closes the gap.

**Reference:** ``ref-packages/tempo2/formResiduals.C`` (L507–536 spin; L2255–2291 TRACK −2).

**Probe:** ``/tmp/wsrt_taylor_formresiduals_step18_probe.py`` →
``/tmp/wsrt_taylor_formresiduals_step18_probe.txt``

| Path | RMS ns | Verdict |
|------|--------|---------|
| pytempo ``acceptance_residual_sec`` | **0.00** | Tier-1 oracle |
| **Production Taylor + legacy TRACK −2** | **16.43** | **best JUG path** |
| Taylor + legacy wrap (manual replay) | **17.37** | ~0.9 ns vs canonical ``compute_phase_residuals`` |
| ``phase5@pt bbat`` + ``torb(dt)`` + track2 | **17.43** | tempo2 spin **worse** than Taylor |
| ``phase5@jug bbat`` + track2 | **17.53** | Step 2 oracle confirmed |
| ``phase5@pt bbat`` + **pt ``torb_sec``** + track2 | **172.41** | **trap** — never pair pt torb in JUG spin |
| frac spin Taylor vs phase5@jug @ shared phas1 | **18.5 ns** | Horner vs ``phase2+phase3`` at jug ``bbat`` |
| pytempo ``torb`` vs ``torb(dt closure)`` | **207 ns** | spin-argument export trap |

**Per ``-sys`` production residual (mean / RMS ns):**

| ``-sys`` | n | mean | RMS |
|----------|---|------|-----|
| WSRT.P1.328 | 29 | +10.4 | 25.2 |
| WSRT.P1.328.C | 58 | +2.8 | 12.2 |
| WSRT.P1.382 | 24 | −0.5 | 16.0 |
| WSRT.P1.382.C | 56 | −8.1 | 14.5 |

**Conclusions (Step 18):**

1. **TRACK −2 bookkeeping is closed** — legacy ``add_phase = −pnAdd`` matches
   ``track_minus2_frac_phase`` when spin uses the same ``phase5`` (Step 1 / ``test_tempo2_track2_pnnew.py``).
2. **~16 ns is not a wrap/pnNew bug** — it is the best residual achievable with
   **JUG-composed** delays + **Taylor@``model_mjd`` ``dt``**. No spin-only JUG patch
   beats **16.4 ns**.
3. **Do not switch production to ``phase5@bbat``** — even with pytempo ``bbat`` and
   ``torb(dt)`` closure, tempo2 spin is **~17.4 ns** (worse). Taylor Horner on emission
   ``dt`` is the correct production compromise until the JAX native chain lands.
4. **Do not use pytempo ``torb_sec`` in JUG-composed ``phase5``** — **172 ns** when paired
   with JUG ``dt``/``bbat`` exports (Step 6 float64 trap class). Always derive
   ``torb = dt − (bbat−PE)×86400`` from the active spin argument.
5. **0 ns requires the JAX tempo2-native clock/delay chain** (§ above) — matching
   ``bbat``/``torb``/``phase5`` from the same in-flight tempo2 path, not patching spin alone.
6. **Do not implement spin changes in JUG production now.** Optional: document the
   ``torb(dt closure)`` rule in ``tempo2_spin.py`` docstrings when JAX work starts.

### idx 85 max \|Δ\| (+110 ns) — red herring as a separate bug

The debt pin reports **max \|Δ\| ≈ 110 ns at TOA idx 85**. Treat this as a **symptom
metric**, not a standalone fix target.

| Field | Value |
|-------|-------|
| Index | 85 / 167 |
| ``-sys`` | WSRT.P1.328 @ 328 MHz |
| Production Δ | **+110.5 ns** (largest in the set) |
| tim ``-padd`` | 0.599804 (same as neighbours idx 83–87) |
| pytempo ``acceptance_residual_sec`` | **exact** vs libstempo at idx 85 |
| Neighbours (same ``-sys`` / ``-padd``) | idx 83 **+2 ns**, 84 **+23 ns**, 86 **−4 ns**, 87 **+22 ns** |

**Ruled out at idx 85 (temp probes ``/tmp/wsrt167_idx85_probe.py``, Step 3):**

- **padd** — removing padd at idx 85 alone blows up to **~1.2 ms** \|Δ\|; padd handling is correct.
- **Isolated pnNew / ``addPhase`` bug** — ``addPhase = +1`` matches tempo2 at idx 85; pnNew identity holds.
- **``nphase`` ladder mismatch (idx 84→85)** — **red herring**. JUG ``nphase`` and pytempo
  ``nphase`` differ by **~10¹⁰ turns on all 167 TOAs** (different absolute spin scales between
  Taylor@``model_mjd`` and tempo2 ``phase2``). Comparing Δ``nphase`` or ``pulse_number`` ladders
  across backends is not diagnostic for a single-TOA wrap bug.
- **Clock / Roemer / sat** — ruled out on harness.

**What idx 85 actually is:** the **tail** of the same Taylor fractional-phase error that
produces **~16 ns RMS** (~3.6×10⁻⁵ turns at F0). Fixing Taylor vs tempo2 spin should pull
both RMS and max \|Δ\| together. ``phase5@bbat`` oracle still leaves idx 85 at **~+100 ns** —
so the outlier is **not** closed by Step 2 wiring either; it tracks the spin reference gap.

**Do not wire ``phase5@bbat`` to production** for wsrt167. **Next:** close Taylor@``model_mjd``
vs full tempo2 ``formResiduals`` spin path (pytempo 0 ns acceptance).

### Code / tests

| Artifact | Role |
|----------|------|
| `jug/residuals/tempo2_spin.py` | ``compute_tempo2_bbat_mjd``; ``track_minus2_frac_phase`` |
| `jug/testing/tempo2_track2_oracle.py` | pytempo-backed TRACK −2 harness |
| `tests/test_tempo2_track2_pnnew.py` | pnNew identity; documents ~17.5 ns oracle floor |

```bash
cd ref-packages/jug
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_track2_pnnew.py -q
```

### WSRT167 (2026-07-05)

167-TOA WSRT low-band subset (328/382 MHz) from `epta_j0613_t2_ipta_all`.
Fixture: `tests/data_tempo2/wsrt167/`. Debt pin:
`tests/test_dev_oracle_wsrt167_parity.py`.

| Measurement | Before `32dd71a` | After `32dd71a` |
|-------------|------------------|-----------------|
| RMS vs libstempo | **~1056 ns** | **~263 ns** |
| max \|Δ\| | **~3233 ns** | **~550 ns** |
| Mean Δ | ~0 ns | \|mean\| < 50 ns |

**Root cause closed in `32dd71a`:** tempo2 uses implicit **`NE_SW = 4` cm⁻³**
(`NE_SW_DEFAULT` in `tempo2/initialise.C`) even when the par omits `NE_SW`. JUG had
default 0 → missing solar-wind `tdis2` in prebinary → wrong `bbat` → spin scatter.
Also fixed: Roemer PM at POSEPOCH. Later IFTE + `formBats` (`tempo2_clock.py`)
reduced wsrt167 to **~16 ns RMS** (still above the 5 ns gate); see
`TEMPO2_NATIVE_CLOCK_STATUS.md`.

Decomposition vs libstempo on the same 167 TOAs (raw pre-fit, unweighted mean):

| Check | RMS / correlation | Notes |
|-------|-------------------|-------|
| Residual RMS (JUG − libstempo) | **263 ns** | max ≈ 550 ns; mean ≈ 0 |
| Intra-`-sys` scatter | **≈ 263 ns** | Not a 328/382 MHz group-mean split (group means differ ≲ 40 ns) |
| `prebinary − total` vs `torb_sec` | **0.17 ns** | **T2model / binary delay closed** |
| libstempo `binarydelay` vs `torb_sec` | **~705 s** | **stale property** (zeros on fresh construct) — **do not use** |
| `sun_shapiro` | **0 ns** | Sun Shapiro matches |
| `roemer` property | sign flip only | libstempo `roemer` = −JUG `roemer_sec`; combined Roemer+Shapiro path OK |
| JUG oracle `bbat` vs pytempo `bbat_mjd` | **330 ns** | corr(residual) ≈ 0.19 — emission spin reference gap |
| Recomposed `pets − torb/86400` vs Tier-1 `pets` | **~275 ns** | **float64 artifact** — use Tier-1 `pets` / `pet_mjd` directly |
| Oracle `bbat` from lib `pets` + torb closure | **222 ns** | libstempo-only JUG-composed path — not Tier-1 |
| `tempo2_spin=True` + legacy TRACK −2 | catastrophic | Do not enable — switches to `pnNew` wrap |
| `compute_tempo2_phase5` + legacy TRACK −2 | **≈ 264 ns** | Same as Taylor when `torb` sign = `−jug_torb` |
| Weighted mean subtraction | **882 ns** | tempo2 uses unweighted (`phase_mean_mode=unweighted`) ✓ |

**Leading hypothesis (updated Step 18):** the ~16 ns floor is the **best JUG-composed
spin path** (Taylor@``model_mjd`` ``dt_sec_ld`` + legacy TRACK −2). Wrap/pnNew/padd are
closed. Switching to tempo2 ``phase5`` (**17.4 ns**) or pt ``torb`` exports (**172 ns**)
does **not** help. The **~286 ns ``batCorr`` gap** (Step 17) and **~330 ns ``bbat`` gap**
are separate clock/delay issues. **0 ns** requires the **JAX tempo2-native pipeline**
(``calculate_bclt`` → ``formBats`` → ``formResiduals``), not production JUG patches.

**Next native fix targets:** implement JAX tempo2-native clock/delay chain (planned);
keep production Taylor@``model_mjd`` unchanged until that lands.

**Historical ad-hoc notes (pre-fix, same TOAs in full 1369-TOA mix):**

- Mean Δ **+1.287 ns** uniform shift when same wsrt167 TOAs embedded in full mix
- **`-padd` / `jump_phase`:** required; without it RMS → ~6 ns
- **328 vs 382 bands:** ~0.4 ns group split; TRACK −2 / `-pn` interaction

---

## 4. J0613 three parity budgets

Closing one budget does not automatically close the others.

### 1. EPTA full multi-backend (~608 ns) — dominant

Integer-turn and `-addsat` debt closed. Remaining bulk is pulse-phase / spin bookkeeping
across multi-backend mix. Decompose via wsrt167 (~263 ns today) + mean-anchor shifts.

**Focus:** close wsrt167 Taylor spin gap (priority 1); full-mix mean-subtraction anchor;
nrt1400 (~62 ns) before touching full mix. ``-padd`` / ``jump_phase`` closed (Step 3).

### 2. Alternate PPTA export (~16 ns)

Native PPTA already ~1.4 ns. ~16 ns appears only with **full alternate-export par** +
tim (TDB, EPTA-reference T2/astrometry on PPTA TOAs) — not from `-pn` format alone.

**Focus:** TDB Roemer+Shapiro; TZR at `TZRMJD` / `TZRSITE pks`.

### 3. Native PPTA (~1.4 ns) — essentially done

No further work required for parity on native `PPTA_dr1dr2` par/tim.

### Ablation summary (alternate export)

| Perturbation | RMS Δ |
|--------------|-------|
| Native par + native tim | **1.43 ns** |
| Full alternate-export par + alternate tim | **15.96 ns** |

Single-field par edits do not reproduce the ~16 ns gap.

---

## 5. Phase and TRACK −2 semantics

### Comparison table

| Aspect | PINT | tempo2 | JUG (both modes) |
|--------|------|--------|------------------|
| Default reference | Index 0 or TZRMJD | tim index 0 (`obsn[0]`) | Sequential by `dt` (no TRACK −2) |
| `TRACK -2` | `model.phase(abs) − pn` | `pnNew` vs `-pn`; `phas1` at tim index 0 | Production: legacy ``−pnAdd``; Phase D: ``pnAct = (pn[i]−pn[0])+pnAdd`` |
| Mean removal | Weighted | Unweighted | Mode-dependent |
| Delay path split? | — | — | **Yes** (providers differ) |
| Phase path split? | — | — | **No** (shared `compute_phase_residuals`) |

### tempo2 `formResiduals.C` order (TRACK −2)

1. Compute spin phase at `bbat` → `phase5`
2. Add tim **`-padd`** to `phase5`
3. `phas1 = fortran_mod(phase5[tim_index_0], 1.0)`; subtract from all TOAs
4. `nphase = fortran_nlong(phase5[i])`; fractional residual = `phase5 − nphase`
5. Optional `addPhase` from `pnNew` vs `-pn` flags

### JUG shipping path (TRACK −2, tempo2 mode)

**Production:** emission-time Taylor spin at geometry **`model_mjd`** → add
`jump_phase` (includes tim **`-padd`** / **`-radd`**) → `phas1@tim[0]` → `fortran_nlong`
per TOA → legacy `add_phase = −pn_add` (equivalent to fixed ``pnNew`` when
``pn[i]−pn[0] == pnNew``).

**Phase D (not production):** ``compute_tempo2_phase5`` at oracle ``bbat`` +
fixed ``track_minus2_frac_phase`` → **~17.5 ns** RMS on wsrt167 (validated in
``tests/test_tempo2_track2_pnnew.py``) — **worse than production**.

**Quarantined / do not enable:** ``USE_NATIVE_BBAT_PHASE5`` — oracle harness only;
formBats ``bbat_mjd`` diagnostic remains **~65 s** off tempo2.

Key touchpoints: `jug/residuals/tempo2_spin.py`,
`jug/residuals/tempo2_native_quarantine.py`, `jug/residuals/simple_calculator.py`.

### Subset pitfall

tim ``-pn`` values are written as **pulse-number offsets relative to full-tim obsn[0]**:
``pn[i] − pn[0]`` equals tempo2 ``pnNew`` (after ``pn0`` anchoring). Raw ``-pn`` in
``pnAct`` produces ``addPhase ~ 10¹⁰`` turns. Isolated sub-tims without regenerating
``-pn`` also change obsn[0] semantics — prefer full-tim oracle pull + mask.

---

## 6. J0613 EPTA investigation summary

Condensed from the former `EPTA_J0613_EFF_PARITY.md` investigation record.

### Fixture

| Field | Value |
|-------|-------|
| ID | `epta_j0613_t2_ipta_all` |
| Par/tim | `tests/data_tempo2/epta_j0613_t2_ipta_all/` |
| TOAs | 1369 (flat tim; `TRACK -2`; `-pn` on all TOAs via `add_pulseNumber`) |
| Sibling | `epta_j0613_t2_nrt1400` (120 TOAs, NRT.BON.1400 excerpt) |
| Unit test | `wsrt167` — 167 WSRT TOAs, freq < 1000 MHz |
| PSRJ | J0613-0200; `BINARY T2`; `F0` ≈ 326.6 Hz |

First tim index (`obsn[0]`): JBO.DFB.1400, MJD ≈ 54847, `-pn 0`. Earliest emission time
is **not** index 0 (WSRT ≈ MJD 52958).

### Historical parity scale

| Configuration | RMS (JUG − libstempo) |
|---------------|----------------------|
| No TRACK −2, no `-pn` | ~2.9 ms |
| TRACK −2 + `-pn`, wrong anchor (pre-fix) | ~46.8 ms |
| TRACK −2 fix (2026-07-03) | ~57 µs → ~2 µs with scalar `-addsat` |
| Per-TOA `-addsat` fix (2026-07-04) | **~608 ns** bulk; `-addsat` TOAs < 1 µs |
| wsrt167 NE_SW + bbat spin (2026-07-05) | **~1056 ns → ~263 ns** on wsrt167 subset |

### What was fixed

1. **TRACK −2 anchor (2026-07-03):** JUG uses `phas1@tim index 0` + per-TOA
   `fortran_nlong`, not `argsort(dt)[0]` + `base_pn + (-pn)`.
2. **±326-turn EFF TOAs (2026-07-03):** eight TOAs on EFF.EBPP.1360/2639 with integer
   errors of ±(int)F0 turns — closed by TRACK −2 fix.
3. **Per-TOA `-addsat` (2026-07-04):** `addsat_track2_turn_delta` closes idx 247/256/561
   (~68 µs → < 1 µs).
4. **Implicit NE_SW + bbat spin (2026-07-05):** `resolve_ne_sw_cm3()` (default 4 cm⁻³),
   Roemer PM at POSEPOCH, Taylor spin at `bbat`; wsrt167 **~1056 ns → ~263 ns**.

### What was ruled out

- Roemer/DM/binary kernels as ms-scale drivers on full mix (post TRACK −2 fix).
- Clock dir JUG vs `$TEMPO2/clock` swap — no RMS change.
- BIPM2011/2019/2024 CLK sweep — no change.
- Full `pnNew`/`bbat` rewrite without correct spin decomposition — catastrophic (~10¹¹ ms).
- BCLT iteration wired into `simple_calculator` — ~903 µs regression vs libstempo.

### EFF backend notes

| `-sys` | n | Isolated RMS (TRACK −2) |
|--------|---|-------------------------|
| EFF.EBPP.1410 | 241 | **~69 ns** |
| EFF.EBPP.1360 | 42 | was ±326-turn outliers; now sub-µs on bad TOAs |
| EFF.EBPP.2639 | 64 | same |
| JBO / NRT / WSRT (isolated) | — | sub-µs to ~1 µs (pre wsrt167 fix); wsrt167 now ~263 ns |

Non-EFF backends match libstempo in isolation but show ~0.73 ms offsets in the **full mix**
when EFF 1360/2639 present — cross-backend `phas1`/anchor coupling (historical).

---

## 7. Other open items

- **G4:** deprecate or repair analytic tempo2 columns; autodiff is supported path.
- **G6:** narrow `ppta_j1741_ell1`; implement or reject `DM_SERIES`.
- **G2 residual:** θ≠0 NumPy/JAX on IPTA workloads — **suspicious / likely false**; see §G2 note.
- **pytest debt pins:** wsrt167 (~263 ns, dev_oracle); native PPTA (~1.4 ns) and alternate
  PPTA (~16 ns) still ad hoc only.

---

## 8. Investigation log

| Date | Summary |
|------|---------|
| 2026-06-01 | Phases A–E: native TDB geometry, TZR, Cases B/C green, design-matrix/fit parity |
| 2026-07-03 | G1/G2 primary closed; J0613 fixtures get TRACK −2 + `-pn`; TRACK −2 anchor fix (~46.8 ms → ~2 µs) |
| 2026-07-04 | Per-TOA `-addsat` int(F0) coupling; EPTA RMS **~608 ns**; PPTA native ~1.4 ns, alternate ~16 ns measured |
| 2026-07-04 | `tempo2_spin=True` attempt wrong for legacy stack; production restored via `addsat_track2_turn_delta` |
| 2026-07-05 | Doc consolidation: three parity markdown files merged into this doc + [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md) |
| 2026-07-05 | **wsrt167 promoted:** fixture under `tests/data_tempo2/wsrt167/`; dev-oracle debt pin; implicit **NE_SW=4**, Roemer PM at POSEPOCH, spin at **bbat** — **~1056 ns → ~263 ns RMS** (`32dd71a`) |
| 2026-07-05 | **Parity review:** longdouble reverted; native phase5 quarantined; outlier harness added |
| 2026-07-05 | **Phase C (fix #1):** TZR apply modes in ``tzr_geometry.py`` — J0030 **15.9 → ~4.7 ns RMS** |
| 2026-07-05 | **Phase D Step 1 (fix #2):** tim ``-pn`` relative to obsn[0]; ``tempo2_track2_oracle.py`` + ``test_tempo2_track2_pnnew.py`` |
| 2026-07-06 | **Phase D Step 3 ruled out:** ``-padd``/``jump_phase`` exact vs pytempo; per-``-sys`` split is spin reference not padd |
| 2026-07-06 | **Phase D Step 4:** Taylor vs tempo2 ``phase2+phase3`` ruled out (0.02 ns fractional); keep ``dt_sec`` model_mjd path; probe ``/tmp/wsrt_taylor_spin_probe.py`` |
| 2026-07-06 | **Phase D Step 5:** per-TOA term diff — oracle ``bbat`` ~330 ns vs pytempo (r≈0.19); roemer/shapiro closed; probes ``/tmp/wsrt_term_diff_probe.py`` |
| 2026-07-06 | **Phase D Step 6:** ``pt_torb≈prebin−total``; ``dt−deltaT(pt)`` ~207 ns; probe ``/tmp/wsrt_model_pet_torb_probe.py`` |
| 2026-07-06 | **Phase D Step 7:** float64 ``model_mjd`` caps ``dt_sec``; Taylor@``deltaT(pt)`` ~173 ns (worse); probe ``/tmp/wsrt_dt_spin_counterfactual_probe.py`` |
| 2026-07-06 | **Phase D Step 8:** JAX two-part dt + compensated Taylor — no gain; probe ``/tmp/wsrt_jax_compensated_spin_probe.py`` |
| 2026-07-06 | **Phase D Step 9:** epoch-chain review — production ``dt_sec_ld`` best (16.4 ns); naive epoch swaps worsen; probe ``/tmp/wsrt_epoch_chain_probe.py`` |
| 2026-07-06 | **Phase D Step 10:** formBats sign probe — +65 s TT/batCorr baseline mismatch; tdis inverted ≈ −64 s vs dm+sw; probe ``/tmp/wsrt_formbats_sign_probe.py`` |
| 2026-07-06 | **Phase D Step 11:** TT_TB gap +64.986 s ≈ ``utc_to_tdb`` (tdis confound); probe ``/tmp/wsrt_clock_chain_probe.py`` |
| 2026-07-06 | **Phase D Step 12:** ``batCorrs = (model−sat)×86400 − prebinary`` (286 ns); probe ``/tmp/wsrt_batcorr_epoch_chain_probe.py`` |
| 2026-07-06 | **Phase D Step 13:** model-epoch batCorr/bbat temp rebuild closes libstempo; ~330 ns bbat open; probe ``/tmp/wsrt_batcorr_from_model_probe.py`` |
| 2026-07-06 | **Phase D Step 14:** ``torb`` closed (0.17 ns); ~330 ns gap is ``model_mjd`` vs ``obsn.bbat``; probe ``/tmp/wsrt_torb_bbat_probe.py`` |
| 2026-07-06 | **Oracle tier retrospective:** Steps 5/6/10/WSRT-table rows corrected — float64 traps, ``binarydelay`` stale |
| 2026-07-06 | **Phase D Step 15:** ~330 ns ``bbat`` = ~286 ns ``batCorr`` + ~304 ns float64 bat read; probe ``/tmp/wsrt_model_bbat_step15_probe.py`` |
| 2026-07-06 | **Phase D Step 16:** ld ``batCorr`` replay 286→214 ns (no collapse); ``model_f64`` vs IFTE(``tdb_ld``) 0 ns; probe ``/tmp/wsrt_model_bbat_step16_probe.py`` |
| 2026-07-06 | **Phase D Step 17:** formBats replay 0 ns; ~286 ns = IFTE scatter + slot cancellation; probe ``/tmp/wsrt_formbats_roemer_epoch_step17_probe.py`` |
| 2026-07-06 | **Phase D Step 18:** ~16 ns = best JUG Taylor spin; phase5@bbat 17.4 ns; pt torb trap 172 ns; probe ``/tmp/wsrt_taylor_formresiduals_step18_probe.py`` |

**When updating parity status:** add a fixture + pytest gate **first**, then update this log.
Ad hoc runs document status between gates but do not substitute for CI.

---

## 9. Commands and tests

```bash
# JUG tempo2 parity (acceptance oracle = libstempo)
cd ref-packages/jug

# wsrt167 debt pin (dev oracle — requires libstempo + $TEMPO2)
PYTHONPATH=. pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q

# Outlier clock / Roemer harness
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_outlier_clock_roemer_diff.py -m dev_oracle -q

# TRACK −2 pnNew harness (Phase D Step 1)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_track2_pnnew.py -q

# TZR parity (Phase C)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_tzr_parity.py -q

# All tempo2 oracle tests
JUG_TEST_TEMPO2=1 PYTHONPATH=. pytest tests/test_tempo2_*.py -m dev_oracle -q

# JUG-only CI path (no libstempo)
PYTHONPATH=. pytest -m 'not dev_oracle' -q

# J0613 EPTA debt pins
PYTHONPATH=. pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -q

# Phase A term ranking (libstempo oracle)
python tools/run_phase_a_diagnostics.py --output /tmp/phase_a_report.json

# Phase D Step 5–6 temp probes (pytempo + libstempo; not in CI)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_term_diff_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_model_pet_torb_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_dt_spin_counterfactual_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_jax_compensated_spin_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_epoch_chain_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_formbats_sign_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_clock_chain_probe.py
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_batcorr_epoch_chain_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_batcorr_from_model_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_batcorr_from_model_probe.py -m dev_oracle -q
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_torb_bbat_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_model_bbat_step15_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_model_bbat_step16_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_formbats_roemer_epoch_step17_probe.py
PYTHONPATH=.:tests:/tmp TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python /tmp/wsrt_taylor_formresiduals_step18_probe.py

# Optional ad-hoc pytempo (NOT in jug CI)
cd ref-packages/pytempo && pip install -e . && python -m pytest tests -q
```

Key modules:

| Module | Role |
|--------|------|
| `jug/residuals/simple_calculator.py` | JUG residuals under test |
| `jug/residuals/tempo2_spin.py` | ``compute_tempo2_bbat_mjd``; ``phase5`` / TRACK −2; pnAct relative to obsn[0] |
| `jug/residuals/diagnostic_conventions.py` | `resolve_ne_sw_cm3()`, conventions |
| `jug/residuals/compatibility_providers.py` | Tempo2 TDB delay provider |
| `jug/residuals/tzr_geometry.py` | TZR apply modes (Phase C) |
| `jug/testing/tempo2_track2_oracle.py` | TRACK −2 pnNew / ``phase5@bbat`` oracle (Phase D) |
| `jug/testing/tempo2_outlier_diff.py` | Per-TOA clock + Roemer diff vs libstempo (outlier work) |
| `jug/testing/tempo2_reference.py` | libstempo acceptance oracle |
| `jug/testing/tempo2_diagnostics.py` | Phase A term oracle (libstempo properties) |
| `jug/testing/phase_a_comparison.py` | Phase A ranking |
| `jug/testing/DEV_ORACLE.md` | Delete-when-standalone checklist |
| `ref-packages/pytempo` | Optional ad-hoc per-TOA diagnostic oracle (external) |

Tempo2 reference source (debug only): `ref-packages/tempo2/` — `formBats.C`,
`calculate_bclt.C`, `formResiduals.C`, `initialise.C` (`NE_SW_DEFAULT`).
