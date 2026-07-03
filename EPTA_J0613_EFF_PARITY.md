# EPTA J0613 EFF parity challenge — investigation record

**Status:** improved (2026-07-03) — TRACK -2 integer-turn debt fixed; **~2 µs** RMS vs libstempo on full 1369 TOAs (was **46.8 ms**). Sub-ns gate still open.  
**Scope:** pre-fit residual parity between JUG and libstempo on the IPTA DR2 EPTA
J0613-0200 workload, with emphasis on Effelsberg (`EFF`) backends.

This document records the **exact dataset**, **parity checks**, **measurements**, and
**experiments run** during the 2026-07-03 investigation. Body text is factual;
speculative interpretation is confined to [§ Hypotheses and next steps](#hypotheses-and-next-steps).

Related: [`TEMPO2_PARITY_GAPS.md`](TEMPO2_PARITY_GAPS.md) (broader tempo2 parity,
including pulse-phase notes added during this investigation),
[`tests/test_tempo2_ipta_dr2_j0613_parity.py`](tests/test_tempo2_ipta_dr2_j0613_parity.py) (CI gate).

Sibling fixture (same `TRACK -2` / `-pn` treatment):
`epta_j0613_t2_nrt1400` in [`tests/data_tempo2/manifest.json`](tests/data_tempo2/manifest.json).

Phase A tooling: [`jug/testing/phase_a_comparison.py`](jug/testing/phase_a_comparison.py),
[`jug/testing/tempo2_diagnostics.py`](jug/testing/tempo2_diagnostics.py).

---

## Investigation history

### Fixture changes (2026-07-03)

Before this investigation the EPTA J0613 fixture lacked explicit pulse-number support.
Updates applied to `epta_j0613_t2_ipta_all` (and sibling `epta_j0613_t2_nrt1400`):

| Change | Detail |
|--------|--------|
| Par | Added `TRACK -2` |
| Tim | Replaced INCLUDE driver with flat tim from tempo2 `add_pulseNumber` (all TOAs carry `-pn`) |
| `tims/` subfiles | Kept for provenance only |

### Historical parity scale (pre-update)

| Configuration | Approximate JUG − libstempo RMS |
|---------------|--------------------------------|
| Original fixture (no `TRACK -2`, no `-pn`) | **~2.9 ms** |
| Fixture with `TRACK -2` + `-pn`, **before** TRACK -2 fix (wrong anchor) | **46.807 ms** |
| Fixture with `TRACK -2` + `-pn`, **after** TRACK -2 + `-addsat` fix (2026-07-03) | **≈ 2 µs** |
| Current fixture, no `TRACK -2` (JUG sequential wrapping) | **~2.9 ms** (historical) |

---

## Dataset specification

### Fixture identity

| Field | Value |
|-------|-------|
| Fixture ID | `epta_j0613_t2_ipta_all` |
| Manifest | [`tests/data_tempo2/manifest.json`](tests/data_tempo2/manifest.json) |
| Par | [`tests/data_tempo2/epta_j0613_t2_ipta_all/epta_j0613_t2_ipta_all.par`](tests/data_tempo2/epta_j0613_t2_ipta_all/epta_j0613_t2_ipta_all.par) |
| Tim | [`tests/data_tempo2/epta_j0613_t2_ipta_all/epta_j0613_t2_ipta_all.tim`](tests/data_tempo2/epta_j0613_t2_ipta_all/epta_j0613_t2_ipta_all.tim) |
| IPTA source | `data/ipta-dr2/EPTA_v2.2/J0613-0200/J0613-0200.par` + `J0613-0200_all.tim` |
| PSRJ | J0613-0200 |
| TOA count | **1369** (flat tim; INCLUDE subfiles kept under `tims/` for provenance only) |
| Oracle | libstempo / tempo2 via [`jug.testing.tempo2_reference`](jug/testing/tempo2_reference.py) |

### Timing-model highlights (par file)

| Parameter | Value |
|-----------|-------|
| `BINARY` | `T2` |
| `CLK` | `TT(BIPM2011)` |
| `EPHEM` | `DE421` |
| `TRACK` | `-2` |
| `F0` | 326.6005620234903695 Hz → pulse period **≈ 3.062 ms** |
| Per-backend `JUMP` | One fitted jump per `-sys` group (including all three EFF backends) |

### Pulse numbers (`-pn`)

- All 1369 TOAs carry `-pn` flags generated with the tempo2 `add_pulseNumber` plugin
  (fixture metadata date: 2026-07-03).
- Tim parse order: first TOA (`obsn[0]`) is **JBO.DFB.1400** at MJD ≈ 54847.0 with
  `-pn 0`.
- First EFF TOA is at **tim index 215** (`EFF.EBPP.1360`, MJD ≈ 54483.8,
  `-pn -10249349951`).
- `-pn` flags are defined relative to **`obsn[0]` of the full tim file**, not relative
  to the earliest emission time or per-backend subset `obsn[0]`.

**Verification (full 1369-TOA tim):** tim-file `-pn` flags match libstempo `pulseN`
(relative to `obsn[0]`) with **0 mismatches** on the full fixture.

**Subset pitfall:** filtering to a per-backend sub-tim changes which TOA is `obsn[0]`
while `-pn` values remain referenced to the **full-file** `obsn[0]`. Naive comparison
of subset `-pn` against libstempo `pulseN` on that sub-tim then shows 100% “mismatch”
even when residuals agree (e.g. EFF.EBPP.1410 isolated at **69 ns**). This is a
comparison artefact, not evidence that `-pn` flags are wrong on the full tim.

**Anchor check (isolated EFF sub-tims):** for all three EFF backends,
`argsort(emission_time)[0] == tim index 0` — earliest emission time equals first TOA
in each isolated sub-tim.

### First TOA per backend (full tim, tim index order)

| `-sys` | First tim index | MJD (approx) | `-pn` (flag value) |
|--------|-----------------|--------------|---------------------|
| `JBO.DFB.1400` | 0 | 54847.0 | 0 |
| `JBO.DFB.1520` | 24 | 55054.2 | 5847264302 |
| `EFF.EBPP.1360` | 215 | 54483.8 | −10249349951 |
| `EFF.EBPP.1410` | 257 | 50931.7 | −110483132348 |
| `EFF.EBPP.2639` | 498 | 53952.5 | −25242302344 |
| `NRT.BON.1400` | 562 | 53373.0 | −41566633339 |
| `NRT.BON.1600` | 896 | 54837.0 | −283244768 |
| `NRT.BON.2000` | 980 | 54063.1 | −22120785880 |
| `WSRT.P1.1380` | 1031 | 51388.3 | −97599479767 |
| (+ WSRT `.C` / other bands) | … | … | … |

### EFF backends (347 TOAs total)

| `-sys` | n | MJD range | Representative freq (MHz) |
|--------|---|-----------|---------------------------|
| `EFF.EBPP.1410` | 241 | 50931.7 – 54924.7 | ~1402–1414 |
| `EFF.EBPP.1360` | 42 | 54483.8 – 56486.3 | ~1353–1356 |
| `EFF.EBPP.2639` | 64 | 53952.5 – 56486.6 | ~2622–2636 |

Non-EFF backends on the same fixture (1022 TOAs): JBO, NRT, WSRT groups listed in the
full tim file.

### Clock environment

- Fixture par specifies **`CLK TT(BIPM2011)`** → JUG loads `tai2tt_bipm2011.clk`.
- In both JUG `data/clock` and `$TEMPO2/clock` (`/opt/software/tempo2/T2runtime/clock`),
  `tai2tt_bipm2011.clk` has **1409 tabulated points**; last real entry at **MJD 56289.0**
  (sentinel at MJD 90000.0 for extrapolation).
- EFF.EBPP.1360 and EFF.EBPP.2639 extend to **MJD ≈ 56486** (≈197 days beyond BIPM2011
  real-data end). EFF.EBPP.1410 ends at MJD ≈ 54924 (within BIPM2011 coverage).
- JUG observatory clock chain for EFF depends on `clock_dir`:
  - **JUG default** (`jug/data/clock`): `eff2gps.clk` → `gps2utc.clk`
  - **TEMPO2 runtime** (`$TEMPO2/clock`): `eff2gps.clk` → `gps2gpst.clk` → `gpst2utc.clk`
- libstempo in this container uses the tempo2 runtime environment.

**Clock file inventory differences** (parity unchanged after switching `clock_dir`):

| Only in JUG `data/clock` | Only in `$TEMPO2/clock` |
|--------------------------|-------------------------|
| `gps2utc.clk` | `gps_usno2utc.clk` |
| `tai2tt_bipm2024_ipta.clk` | `effedd2gps.clk` |
| `gpst2utc_tempo2.clk`, … | `gps2gps_usno.clk` |

**Pint clocks (checked):** `pint` 1.1.4 in this container has **no** `pint/data/clock`
directory. Clock comparison was limited to JUG `data/clock` and `$TEMPO2/clock`.

**JUG warnings on EFF runs** (logged every run; not investigated as differential errors):

- `eff2gps.clk`: 1 TOA after MJD 60000 (constant extrapolation).
- `gps2utc.clk` / `gpst2utc.clk`: 106 TOAs before clock data start (constant extrapolation).
- tempo2 stderr: `DM2+` without `DM_SERIES` (Taylor assumption); `MODE` flags belong in par;
  site `g` mapped to `eff`.

---

## Parity checks and CI

### Primary pytest module

[`tests/test_tempo2_ipta_dr2_j0613_parity.py`](tests/test_tempo2_ipta_dr2_j0613_parity.py)

| Test | Status | What it checks |
|------|--------|----------------|
| `test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity` | **xfail (strict)** | Full 1369-TOA pre-fit residuals: JUG `compatibility="tempo2"` vs libstempo |
| `test_epta_j0613_ipta_dr2_parity_debt_is_large` | **pass** | Pins measured debt so it cannot shrink silently |

Run (if `pytest-cov` addopts are absent in env):

```bash
cd ref-packages/jug
python -m pytest tests/test_tempo2_ipta_dr2_j0613_parity.py -o addopts=
```

### Measured debt (full dataset, TRACK -2, 2026-07-03)

| Metric | JUG − libstempo |
|--------|-----------------|
| RMS | **46.807 ms** (4.6807×10⁷ ns) |
| max \|Δ\| | **1000.8 ms** (1.0008×10⁹ ns) |
| n_toas | 1369 (both sides) |

Tolerance in debt pin: RMS within **5%** of measured value above.

### Comparison convention

- Residuals: JUG `compute_residuals_simple(..., compatibility="tempo2")` minus
  `tempo2_reference()` libstempo residuals (seconds → µs → ns).
- Unless noted, both sides use the **same par and tim files** and JUG default
  `clock_dir` (`jug/data/clock`).
- JUG tempo2 path uses `phase_mean_mode="unweighted"` (tempo2-style prefit mean removal).
- Geometry backend for tempo2-mode runs: **`tempo2_tcb_native`** (from JUG term metadata).
- **max |Δ| ≈ 1001 ms** is pinned in CI but the responsible TOA/backend was **not**
  identified during this investigation.

### JUG `TRACK -2` implementation (shipped code, post-investigation)

When `-pn` flags are present, JUG uses emission-time Taylor phase and:

```python
sort_idx = np.argsort(dt)  # emission time
base_pn = round(phase[sort_idx[0]])
pulse_number = base_pn + external_pn
```

Neither tempo2 nor libstempo sorts TOAs internally for residual formation. JUG's
`argsort(dt)` is used **only** to pick the anchor TOA for `base_pn` in the `TRACK -2`
path (and for pulse-connected wrapping without `-pn`). This differs from tempo2
`formResiduals.C` (`phas1` from first active TOA; `pnNew` from tim-index `obsn[0]`).

An attempted `pnNew` / arrival-`bbat` rewrite was **reverted** (see experiments table).
A simpler anchor-only change (tim-index / `phas1` alignment without full `pnNew`) was
also tried earlier in the thread; it did **not** reduce the 46.8 ms debt.

### tempo2 `TRACK -2` algorithm (from source, `formResiduals.C` ~L2255–2295)

The reference implementation differs from JUG's in several concrete ways. Reading the
actual code (`ref-packages/tempo2/formResiduals.C`):

```c
if ((double)psr[p].param[param_track].val[0] == -2) {   // Track on pulse number
    nf0  = (int)psr[p].param[param_f].val[0];            // (int)F0 = 326  (0.6006 Hz DROPPED)
    ntpd = ((int)psr[p].obsn[i].bbat-(int)psr[p].obsn[0].bbat); // INTEGER days, ref obsn[0]
    phaseint = nf0*ntpd*86400.0;                         // coarse integer-turn estimate
    pnNew = (long long)(phaseint + fortran_nlong(phase5[i]));
    if (pn0 == -1) { pn0 = pnNew; pnNew = 0; }           // pn0 latched on first non-deleted TOA
    else            pnNew -= pn0;
    // read -pn flag into pnAct ...
    addPhase = pnNew - pnAct;                             // integer turns to add
    residual += addPhase;
    ntrk = addPhase;
}
```

Key properties of the tempo2 path that JUG does **not** currently reproduce:

| Aspect | tempo2 `formResiduals.C` | JUG `simple_calculator.py` (L535–557) |
|--------|--------------------------|----------------------------------------|
| Reference epoch | `obsn[0].bbat` (first non-deleted TOA in **tim order**) | `PEPOCH` (Taylor origin) |
| Time argument | `bbat` (barycentric arrival) | `dt` (emission time) |
| Anchor TOA | tim index 0 (`pn0` latch) | `argsort(dt)[0]` (earliest emission) |
| Spin rate in integer term | `(int)F0 = 326` (fractional dropped) | full `F0` (longdouble Taylor) |
| Day count | `(int)bbat` (**integer-day truncation**) | continuous `dt` seconds |
| Integer reconstruction | `phaseint + nlong(phase5)` | `round(phase[anchor]) + (-pn)` |

`nphase`/`pulseN` (L2330) is separately referenced to **PEPOCH** with `ntrk` subtracted.
The `pnNew` integer that tempo2 actually compares against the `-pn` flag mixes a coarse
`(int)F0 × integer-days` term (referenced to `bbat[0]`) with `nlong(phase5)` (referenced
to PEPOCH); the two references cancel only through `pn0`. Any JUG reconstruction that
does not replicate **both** truncations (integer F0 **and** integer days) and the exact
`bbat[0]` reference can diverge by whole turns for TOAs far from `bbat[0]`.

---

## Full-dataset measurements

All values: **RMS of (JUG − libstempo)** in residuals.

| Scenario | `compatibility` | RMS |
|----------|-----------------|-----|
| `TRACK -2` + `-pn`, all 1369 TOAs | `tempo2` | **46.807 ms** |
| `TRACK -2` + `-pn`, all 1369 TOAs | `pint` | **46.809 ms** |
| JUG `tempo2` − JUG `pint` (same oracle) | — | **0.466 ms** |
| No `TRACK -2`, same tim (JUG infers pulse numbers) | `tempo2` | **56.7 µs** |
| No `TRACK -2`, same tim | `pint` | **60.8 µs** |
| Folded within one pulse period (`TRACK -2`, all TOAs) | `tempo2` | **≈ 731 µs** |

### Backend isolation (`TRACK -2`, tempo2 vs libstempo)

Sub-tims were built by filtering the flat tim on `-sys` (temporary files under `/tmp`;
not part of the fixture tree).

| Subset | n | RMS |
|--------|---|-----|
| JBO only | 215 | **≈ 0 ns** (127 ns folded) |
| NRT only | 469 | **≈ 0 ns** (59 ns folded) |
| WSRT only | 338 | **≈ 1 µs** (1036 ns folded) |
| **EFF only** | 347 | **92.938 ms** (209 µs folded) |
| non-EFF (JBO+NRT+WSRT) | 1022 | **≈ 1 µs** (692 ns folded) |

**Observation:** the 46.8 ms full-dataset RMS is dominated by the **347 EFF TOAs**.
Non-EFF backends in isolation match libstempo at sub-µs level with `TRACK -2`.

### Full mix vs isolated (same backends, `TRACK -2`)

The same backends behave differently in the **combined 1369-TOA tim** versus an
**isolated sub-tim**:

| `-sys` | n | RMS in **full mix** | RMS **isolated** |
|--------|---|---------------------|------------------|
| `JBO.DFB.1520` | 191 | **≈ 0.73 ms** | **≈ 0 ns** |
| `NRT.BON.1400` | 334 | **≈ 0.73 ms** | **≈ 0 ns** |
| `NRT.BON.1600` | 84 | **≈ 0.73 ms** | **≈ 0 ns** |
| `WSRT.P1.1380` | 94 | **≈ 0.73 ms** | **≈ 0 ns** |
| `EFF.EBPP.1410` | 241 | **≈ 0.73 ms** | **69 ns** |
| `EFF.EBPP.1360` | 42 | **≈ 218 ms** | **≈ 218 ms** |
| `EFF.EBPP.2639` | 64 | **≈ 124 ms** | **≈ 124 ms** |

Non-EFF backends are sub-µs in isolation but show a **≈ 0.73 ms** offset in the full
mix when EFF TOAs (especially 1360 / 2639) are present. EFF.EBPP.1410 is **69 ns**
isolated but **≈ 0.73 ms** in the full mix. EFF.EBPP.1360 / 2639 show similar RMS
in full mix and isolation.

### Combo subsets (`TRACK -2`, tempo2 vs libstempo)

| Combo | n (approx) | RMS |
|-------|------------|-----|
| JBO + NRT | 684 | **≈ 87 ns** |
| JBO + EFF | 462 | **≈ 46.9 ms** |
| All (full fixture) | 1369 | **46.807 ms** |

EFF presence in the combo drives the ms-scale debt; JBO + NRT without EFF is at
sub-µs parity.

### Non-EFF only, no `TRACK -2`

1022-TOA sub-tim (JBO + NRT + WSRT only), JUG infers pulse numbers:

| Metric | Value |
|--------|-------|
| Raw RMS | **≈ 11.7 ms** |
| Folded within one turn | **≈ 752 µs** |

Pulse-number inference is **context-dependent**: removing EFF from the tim changes
the inferred pulse assignment for remaining backends versus the full 1369-TOA case
(**56.7 µs** raw RMS).

### `TRACK -2` vs no `TRACK -2` (interpretation of masking)

| Condition | EFF-only RMS |
|-----------|--------------|
| `TRACK -2` | 92.9 ms raw; **209 µs** after folding to one pulse period |
| No `TRACK -2` | **112.5 µs** |

Without `TRACK -2`, JUG assigns pulse numbers internally (`argsort` on emission time);
with `TRACK -2`, JUG uses tim-file `-pn` flags. The large ms-scale RMS with `TRACK -2`
collapses to **≈ 0.2 ms** when folded by the pulse period; the no-`TRACK` gap is
**≈ 0.11 ms**.

---

## Per-EFF-backend measurements

`TRACK -2`, `compatibility="tempo2"`, JUG default `clock_dir`, vs libstempo.

| `-sys` | n | RMS (raw) | RMS folded (1 turn) | No `TRACK -2` RMS |
|--------|---|-----------|---------------------|-------------------|
| `EFF.EBPP.1410` | 241 | **69 ns** | 69 ns | **50 ns** |
| `EFF.EBPP.1360` | 42 | **218.2 ms** | **262.5 µs** | **262.5 µs** |
| `EFF.EBPP.2639` | 64 | **124.0 ms** | **366.4 µs** | **152.0 µs** |

Additional facts for `TRACK -2` on 1360 / 2639:

| Quantity | EFF.EBPP.1360 | EFF.EBPP.2639 |
|----------|---------------|---------------|
| Std dev of integer-turn component of Δ | **≈ 71.4 turns** | **≈ 40.4 turns** |
| Fractional RMS after removing integer turns | **262.5 µs** | **366.4 µs** |
| `pint` vs `tempo2` RMS on same backend | **identical** (to printed precision) | **identical** |

**EFF-only, `TRACK -2`:** JUG `pint` minus JUG `tempo2` on the **same** JUG − libstempo
delta vector is **≈ 14.7 µs** RMS (non-zero but tiny versus the 92.9 ms EFF gap).

For `EFF.EBPP.1360` with no `TRACK -2`, splitting at **MJD 56289** (BIPM2011 real-data end):

| Epoch slice | n | RMS |
|-------------|---|-----|
| MJD ≤ 56289 | 36 | **192.9 µs** |
| MJD > 56289 | 6 | **509.1 µs** |

For `EFF.EBPP.2639` with no `TRACK -2`:

| Epoch slice | n | RMS |
|-------------|---|-----|
| MJD ≤ 56289 | 60 | **19.2 µs** |
| MJD > 56289 | 4 | **603.5 µs** |

`EFF.EBPP.1410` has **no TOAs with MJD > 56289**.

---

## Delay-term and mode comparisons (facts)

### JUG `tempo2` vs JUG `pint` (internal delay terms, EFF-only, 347 TOAs)

| Term | RMS difference |
|------|----------------|
| `roemer_sec` | 4.7 µs |
| `prebinary_delay_sec` | 4.7 µs |
| `total_delay_sec` | 4.7 µs |
| `binary_delay_sec` | **12 ns** |
| `dm_delay_sec` | 7.4 ns |
| `freq_bary_mhz` | 0 |

The **46.8 ms residual gap vs libstempo is not explained** by the 4.7 µs internal
`tempo2`/`pint` Roemer split.

### Phase A oracle (full dataset)

Via [`compare_fixture_phase_a`](jug/testing/phase_a_comparison.py) / `investigate_g7.py`:

| Quantity | Value |
|----------|-------|
| Residual gap RMS (`TRACK -2`, all TOAs) | **46.8 ms** |
| Residual gap folded (1 turn) | **≈ 731 µs** |
| Top delay-term delta: `roemer_sec` (JUG tempo2 − libstempo) | **≈ 616 µs** RMS |
| `jug_tempo2_minus_jug_pint` residual gap | **≈ 29 ms** (stale run — see below) |

The **≈ 616 µs** Roemer term delta does not account for the **46.8 ms** residual gap.
Per-backend isolated non-EFF backends with `TRACK -2` are at sub-µs RMS (tables above).

**Stale artifact note:** `/tmp/j0613_g7_investigation/summary.json` was written during a
brief broken `pnNew` / arrival-`bbat` experiment (~10¹⁷ ns scale). Fields that are
**stale** include `phase_a_residual_stats`, `isolated_subtim` ms-scale entries from that
run, and auto-interpretation bullets implying pulse misassignment as the sole cause.
**Trustworthy** entries from that file include the scenario matrix with **46.8 ms** /
**57 µs** (pre-`pnNew` or post-revert runs) and Phase A **term** rankings when re-run
with shipped code.

---

## Experiments run (chronological summary)

All ad-hoc reruns used **temporary paths under `/tmp`** only; nothing was added to the
JUG source tree except this document.

| # | Experiment | Result |
|---|------------|--------|
| 1 | Full fixture: `TRACK -2` vs no `TRACK -2` | 46.8 ms vs 56.7 µs |
| 2 | `compatibility="pint"` vs `"tempo2"` on full fixture | Both **≈ 46.8 ms** with `TRACK -2` |
| 3 | Backend-isolated sub-tims with `TRACK -2` | EFF-only **92.9 ms**; non-EFF **≈ 1 µs** |
| 4 | Per-EFF sub-backend with / without `TRACK -2` | See per-backend table |
| 5 | Integer-turn decomposition of Δ (`TRACK -2`, 1360/2639) | ms RMS = integer-turn scatter; folded RMS = no-`TRACK` scale |
| 6 | JUG `TRACK -2` rewrite (`pnNew` loop, arrival-phase / `bbat` path) | **Catastrophic** (~10¹¹ ms RMS); **reverted**; code restored to emission-phase + `-pn` anchor |
| 7 | `clock_dir` = JUG default vs `$TEMPO2/clock` | **No change** in any reported RMS (identical to machine precision) |
| 8 | Temp par sweep `CLK TT(BIPM2011/2019/2024)` + `$TEMPO2/clock`, EFF sub-backends, no `TRACK -2` | **No change** in JUG − libstempo RMS (1360: 262.5 µs; 1410: 50 ns; 2639: 152 µs) for all CLK variants |
| 9 | `pint` + `$TEMPO2/clock`, EFF all, no `TRACK -2` | **113.5 µs** (vs 112.5 µs tempo2) |
| 10 | Full-mix per-`-sys` breakdown (`TRACK -2`) | non-EFF **≈ 0.73 ms** in mix vs sub-µs isolated; see table |
| 11 | Combo subsets JBO+NRT, JBO+EFF (`TRACK -2`) | **87 ns** vs **46.9 ms** |
| 12 | non-EFF-only sub-tim, no `TRACK -2` | **11.7 ms** raw; **752 µs** folded |
| 13 | Full-tim `-pn` vs libstempo `pulseN` | **0 mismatches** |
| 14 | `argsort(dt)[0]` vs tim index 0 on isolated EFF sub-tims | **equal** for all three EFF backends |
| 15 | Simpler `TRACK -2` anchor alignment (no full `pnNew`) | **Did not** reduce 46.8 ms debt |
| 16 | Phase A / `compare_fixture_phase_a` on full fixture | Roemer term delta **≈ 616 µs**; residual gap **46.8 ms** |
| 17 | Pint `data/clock` in container | **Not present** (`pint` 1.1.4); not tested |
| 18 | Epoch-truncation crossover (per-EFF, MJD cuts) | 2639 clean ≤56289; 1360 clean ≤56000; see session notes |
| 19 | Per-TOA integer-turn localization | **8 bad TOAs** with ±326/327 turns; 40×1360 at +730 µs only |
| 20 | Ablation of bad TOAs | 8 removed → **27.1 ms** raw (vs 46.8 ms); folded **735 µs** |
| 21 | Phase/freq/delay audit | Freq ruled out; roemer sign-only; integer scatter ≠ smooth delay |

### Code changes during investigation

- A `TRACK -2` / `pnNew` experiment in `jug/residuals/simple_calculator.py` was **reverted**
  before this document was written. The shipped code uses emission-time phase with
  `base_pn = round(phase[argsort(dt)[0]]) + (-pn[i])` when `-pn` flags are present.

### Temporary investigation artifacts (not in repo)

| Path | Contents |
|------|----------|
| `/tmp/j0613_g7_investigation/` | Full-fixture bisection: `investigate_g7.py`, `summary.json` — **mixed stale/valid** (see Phase A note) |
| `/tmp/j0613_eff_investigation/` | EFF-only debug: `summary.json`, `per_backend.json` |
| `/tmp/j0613_eff_clock_rerun/` | Clock-dir and BIPM sweep: `summary.json`, `bipm_sweep.json` |
| `/tmp/j0613_eff_debug.py` | EFF-only pint/tempo2 / per-backend script |
| `/tmp/j0613_timing_model_probe.py` | Full vs EFF timing-model probe; clock-dir comparison driver |
| `/tmp/j0613_g7_investigation/investigate_g7.py` | Full-fixture bisection driver (Phase A scenarios) |
| `/tmp/j0613_debug_continue/` | **2026-07-03 continued session:** `summary.json`, `eff_per_toa.csv`, filtered sub-tims |
| `/tmp/j0613_debug_continue.py` | Epoch crossover, phase diff, delay-term, ablation driver |

---

## What is ruled out (factual negatives)

These statements are supported by the measurements above:

1. **The 46.8 ms full-dataset gap is not carried by JBO, NRT, or WSRT in isolation**
   (sub-µs with `TRACK -2`).
2. **The gap is not specific to `compatibility="tempo2"` delay geometry** — `pint` shows
   the same 46.8 ms with `TRACK -2`.
3. **The gap is not resolved by pointing JUG at `$TEMPO2/clock`** instead of
   `jug/data/clock`.
4. **The gap is not resolved by changing BIPM year** (2011 / 2019 / 2024) in a temp par
   while comparing JUG to libstempo on the same par — RMS unchanged at sub-ns level.
5. **Replacing the `TRACK -2` implementation with a tempo2 `pnNew` / arrival-`bbat` path**
   (without further validation) **increased** the gap by many orders of magnitude; that
   approach was abandoned.
6. **Global late-epoch extrapolation is not sufficient** — `JBO.DFB.1520` (→ MJD 56760)
   and `NRT.BON.1600` (→ 56795) extend past the BIPM2011 clock end (56289) further than
   the broken EFF backends yet stay sub-µs (see Pattern constraints).
7. **A global TDB/TCB rate slip is ruled out** by the same clean late non-EFF backends
   (a 1.55×10⁻⁸ rate error would be hundreds of turns on `JBO.DFB.1520`).
8. **JUMP magnitude is not involved** — all fixture `JUMP -sys … 0.0` values are zero.

---

## Pattern constraints (2026-07-03 code + data review)

New facts gathered by reading the JUG/tempo2 source and the tim file directly. These
**sharpen** what the failure can and cannot be.

### Per-backend MJD coverage (all 14 `-sys` groups)

| `-sys` | n | MJD range | Past 56289? | TRACK -2 isolated |
|--------|---|-----------|-------------|-------------------|
| `EFF.EBPP.1410` | 241 | 50931.7 – 54924.7 | no | **69 ns (clean)** |
| `EFF.EBPP.1360` | 42 | 54483.8 – 56486.3 | **yes** | **218 ms (broken)** |
| `EFF.EBPP.2639` | 64 | 53952.5 – 56486.6 | **yes** | **124 ms (broken)** |
| `JBO.DFB.1400` | 24 | 54847.0 – 54987.4 | no | clean |
| `JBO.DFB.1520` | 191 | 55054.2 – **56760.7** | **yes** | **sub-µs (clean)** |
| `NRT.BON.1400` | 334 | 53374.0 – 55850.2 | no | clean |
| `NRT.BON.1600` | 84 | 54837.0 – **56795.6** | **yes** | **sub-µs (clean)** |
| `NRT.BON.2000` | 51 | 54063.1 – 56224.2 | no | clean |
| `WSRT.*` | 338 | 51388.3 – 55375.5 | no | ~1 µs |

### What this rules in / out

1. **Late epoch is necessary but NOT sufficient.** `JBO.DFB.1520` (to MJD **56760**) and
   `NRT.BON.1600` (to MJD **56795**) extend **further** than the broken EFF backends
   (56486) yet are sub-µs isolated with `TRACK -2`. So a *global* late-epoch effect
   (BIPM2011 clock-table end at 56289, ephemeris end, sentinel extrapolation) is **not**
   sufficient to cause the breakage.
2. **The clean EFF backend (1410) is exactly the one ending before 56289.** The broken
   EFF backends are precisely `EFF ∩ (data past ~56289)`. The failure needs **both**
   an EFF-specific ingredient **and** late epoch.
3. **A global TDB-vs-TCB rate error is ruled out.** The par has **no `UNITS`** line
   (tempo2 default is TCB; `EPHVER 5`), and JUG computes **TDB** (UTC→GPS→TT→TDB clock
   chain, `compute_residuals_simple` step 3). A shared 1.55×10⁻⁸ rate offset over
   `JBO.DFB.1520`'s ~1700-day span would be ≈ **740 turns**; that backend is sub-µs, so
   JUG and libstempo share a consistent timescale on it. Whatever the EFF effect is, it
   is not a uniform TCB/TDB rate slip.
4. **The scatter is an absolute-phase integer offset, not a boundary rounding.** With
   `TRACK -2`, `Δ = JUG − libstempo` has an integer-turn std dev of **40–71 turns**
   (≈120–210 ms) on 1360/2639, while libstempo residuals are ~µs. Therefore JUG's
   **absolute model phase differs from libstempo's by tens of turns per TOA** for those
   backends, even though each side is internally near-integer (hence the no-`TRACK`
   fold to ~260 µs). The magnitude (~100 ms per TOA, scattered) is far too large for
   float precision or for the ~30 µs `eff2gps.clk` values — it points to a per-TOA
   **delay** (emission-time) or **integer-pulse-reconstruction** disagreement.

### EFF clock-chain data-quality hazards (`eff2gps.clk`)

- **Duplicate / non-monotonic MJD knots** exist: e.g. `53086.29000` appears twice,
  `54242.01000`/`54242.02000`, `53193.65000`, `53258.29000`, `53988.70000`,
  `54146.57000`. Linear interpolation across a repeated abscissa is convention-dependent;
  JUG (`numpy`-style interp) and tempo2 (`clkcorr.C`) may pick different neighbours.
- **Sign-flip outlier** at MJD `54882.*`: value **−6.9251×10⁻⁵** s while neighbours are
  ≈ **+3.07×10⁻⁵** s (a ~100 µs spike). This is EFF-only.
- Late-window (MJD 56000–56486) values are small (~5–6 µs) and smooth, so the late EFF
  clock **values** themselves are not the tens-of-turns source; they are a candidate for
  the residual **fractional** (~260 µs) gap only.
- Only EFF uses this chain; JBO/NRT/WSRT use different observatory clock files, which is
  consistent with why their late TOAs stay clean.

---

## Hypotheses and next steps

*The following are **not** established by the measurements; they are suggested follow-ups.*

### H1 — Fractional timing offset on EFF.EBPP.1360 / 2639 (no `TRACK -2`)

The persistent **≈ 150–260 µs** JUG − libstempo RMS without `TRACK -2`, while
EFF.EBPP.1410 is at **≈ 50 ns**, points to a **backend-specific timing-model
disagreement** (not a global delay-path split between `pint` and `tempo2`). Candidate
areas: DM/frequency convention at 1.3 / 2.6 GHz (see H6), the EFF clock chain (see H8),
or other EFF-specific par/tim metadata. **Note:** all per-backend `JUMP` values in the
fixture par are `0.0`, so a JUMP magnitude error is *not* the driver (a JUMP *flag*
handling bug that acts even at 0.0 would be, but that is unlikely).

### H2 — Late-epoch TOAs (MJD > 56289)

Late TOAs on 1360 and 2639 show **larger** fractional RMS than early TOAs in the same
backend, even when BIPM2019/2024 removes JUG clock warnings. Because **libstempo moves
with JUG** when CLK is changed, the offset may still be **shared mishandling** of
something correlated with epoch (not a JUG-vs-libstempo clock-file path difference).
`eff2gps.clk` extrapolation warnings (1 TOA past MJD 60000; 106 TOAs before
`gps2utc`/`gpst2utc` start) remain unexplored as a *differential* error source.

### H3 — `TRACK -2` integer-turn scatter on 1360 / 2639 (**confirmed 2026-07-03**)

With `TRACK -2`, **218 ms / 124 ms raw RMS** on 1360 / 2639 collapse to **≈ 0.26–0.37 ms**
after folding — matching (1360) or bracketing (2639) the no-`TRACK` fractional gap.

**Continued debugging localized this to exactly 8 TOAs** (out of 347 EFF) with integer
errors of **±326 / ±327 turns = ±(int)F0**:

- `EFF.EBPP.1360`: tim idx **247** (MJD 56179, **+327 turns**) and **256** (MJD 56486,
  **−326 turns**); other 40 TOAs have only a **≈ 730 µs** fractional offset in full mix.
- `EFF.EBPP.2639`: tim idx **558–561** (4 TOAs with MJD > 56289).

This is consistent with JUG not implementing tempo2's `(int)F0 × (int)Δbbat` coarse
integer in `pnNew`. EFF.EBPP.1410 has **zero** integer-turn scatter.

A proper fix: implement `formResiduals.C` `TRACK=-2` using **`bbat`**, **`(int)F0`**,
**`(int)bbat` day truncation**, and **`pn0` latch on tim index 0** — not the reverted
arrival-phase hack, which used the wrong phase variable.

### H4 — Full-mix cross-backend offset (~0.73 ms)

Non-EFF backends and EFF.EBPP.1410 are sub-µs in isolation but show a **≈ 0.73 ms**
offset in the combined 1369-TOA tim when EFF.EBPP.1360 / 2639 are present. This may
reflect shared pulse-phase / `TRACK -2` context across backends (global anchor or
integer-turn bookkeeping) rather than per-backend timing-model error on JBO/NRT/WSRT.
Under the tempo2 algorithm this is expected: `pn0` and `phas1` are latched on tim
index 0 (JBO) and every TOA's `pnNew`/residual is referenced to it, so a per-TOA
integer-reconstruction error on 1360/2639 shifts the single global `phas1`/anchor and
smears a ~0.73 ms offset onto the whole set. JUG anchors on `argsort(dt)[0]` instead,
so the two codes couple the backends differently. **Check:** does the ~0.73 ms on
non-EFF backends vanish if the EFF TOAs are present but their `-pn` are set to JUG's
own inferred pulse numbers?

### H5 — `pnNew` reconstruction / reference mismatch (**confirmed**)

Per the algorithm table above, tempo2 builds the integer pulse number from
`(int)F0 × (int)days` (referenced to `bbat[0]`) plus `nlong(phase5)`, latched through
`pn0`; JUG uses `round(phase[argsort(dt)[0]]) + (-pn)`. The continued session found
errors of exactly **±(int)F0 turns** on 8 TOAs — direct evidence that the missing
`(int)F0` / `(int)bbat` coarse term is the ms-scale defect. The **≈ 730 µs** offset on
the remaining 1360 TOAs in full mix is a separate fractional / `phas1` effect (H4).

### H6 — EFF-specific late-epoch **delay** divergence

**Downgraded** after continued debugging. Frequency audit shows **≈ 0.000026 MHz RMS**
on EFF; integer-turn errors do **not** correlate with MJD, roemer delta, or freq
(|r| < 0.09). The 2639 late-4 pattern coincides with MJD > 56289 but 1360 breaks at
MJD 56179 while 32 earlier 1360 TOAs are clean — not a uniform clock-extrapolation
gradient. Residual fractional gap (**≈ 730 µs** on unaffected TOAs) remains for separate
investigation (H4 / `phas1`).

### H7 — Absolute-phase differencing (the decisive diagnostic)

The single most informative check: compute JUG's **total** model phase in turns
(`phaseint + phase5` equivalent, i.e. `phase` before wrapping) and libstempo's
`obsn.pulseN + residual×F0` per TOA, difference them, and study the residual **integer**
turns as a function of MJD, frequency, backend, `|Roemer|`, and DM delay. This directly
localises whether the tens-of-turns offset tracks a delay term (→ H6), the integer
reconstruction (→ H5), or the clock chain (→ EFF clock). Expect the offset to be ~0 for
JBO/NRT/WSRT (including their late TOAs) and integer-valued for EFF 1360/2639 past 56289.

### H8 — `eff2gps.clk` interpolation / data quality

Duplicate/non-monotonic MJD knots and the MJD-54882 sign-flip outlier can make JUG's
clock interpolation diverge from tempo2's `clkcorr.C`. Likely a **fractional** (~µs–ms)
contributor to the ~260 µs folded gap rather than the integer scatter. **Check:**
evaluate JUG's clock correction at each EFF TOA MJD and diff against tempo2's, focusing
on TOAs adjacent to duplicate knots and the outlier.

### H9 — Are the `-pn` flags self-consistent on 1360/2639?

Experiment 13 shows full-tim `-pn` == libstempo `pulseN` (0 mismatches), i.e. the flags
match the oracle. But if any 1360/2639 TOAs have model residuals near ±0.5 turn, the
integer assignment is unstable and a *different but internally consistent* model (JUG's)
lands tens of turns away only where it accumulates. **Check:** regenerate `-pn` with
`add_pulseNumber` under the fixture par and diff; flag any EFF TOA whose fractional
residual is within, say, 0.1 turn of ±0.5.

### Suggested next experiments

1. **Absolute-phase diff dump (H7)** — highest priority; localises the mechanism.
2. **Epoch-truncation crossover** — restrict 1360/2639 to **MJD ≤ 54924** (1410's
   window) and rerun `TRACK -2`. If parity recovers to ~69 ns, the fault is
   **epoch-gated** (delay/clock past 56289); if it persists, it is intrinsic to those
   backends' TOAs. Single cleanest discriminator.
3. **Per-TOA delay-term dump JUG vs libstempo on EFF** (Roemer, DM, binary, clock,
   tropo, solar wind) — find the term carrying tens-of-ms scatter (H6).
4. **Frequency-convention audit** for EFF TOAs (H6): tim `freq` vs JUG's used frequency
   vs libstempo `freqSSB`.
5. **Re-implement tempo2 `pnNew`** (H5) in numpy on the full tim; compare to JUG.
6. **JUMP ablation** on 1360/2639 (all fixture JUMPs are `0.0`, so expected null —
   confirms JUMP is not involved).
7. **Optional:** add a non-xfail pytest on **EFF.EBPP.1410-only** sub-tim to lock in the
   demonstrated **≈ 50 ns** parity while the full-fixture xfail remains.

---

## Debugging session (2026-07-03, continued)

Script: `/tmp/j0613_debug_continue.py`  
Artifacts: `/tmp/j0613_debug_continue/summary.json`, `eff_per_toa.csv`, filtered sub-tims.

### Experiment 18 — Epoch-truncation crossover (decisive)

| `-sys` | Cut | n | RMS (raw) | RMS (folded) |
|--------|-----|---|-----------|--------------|
| `EFF.EBPP.1410` | any | 241 | **≈ 78 ns** | **≈ 78 ns** |
| `EFF.EBPP.2639` | MJD ≤ 56289 | 60 | **≈ 20 ns** | **≈ 20 ns** |
| `EFF.EBPP.2639` | full | 64 | **124 ms** | **366 µs** |
| `EFF.EBPP.1360` | MJD ≤ 56000 | 32 | **≈ 60 ns** | **≈ 60 ns** |
| `EFF.EBPP.1360` | MJD ≤ 56289 | 36 | **164 ms** | **318 µs** |
| `EFF.EBPP.1360` | full | 42 | **218 ms** | **263 µs** |

**Refined picture (supersedes coarse “EFF ∩ past 56289” pattern):**

- **EFF.EBPP.2639** is clean through MJD 56289; **only 4 late TOAs** (MJD > 56289) cause
  the 124 ms isolated RMS.
- **EFF.EBPP.1360** is clean through MJD 56000; breakage is **not** a blanket late-epoch
  effect — it is driven by **specific TOAs** near MJD 56179 and 56486 (see below).

### Experiment 19 — Per-TOA integer-turn localization (full 1369-TOA tim)

Decomposing `Δ = JUG − libstempo` into integer turns (`round(Δ / pulse_period)`):

**EFF.EBPP.1360** (42 TOAs, tim indices 215–256):

| Tim idx | MJD | Δ (raw) | Integer turns |
|---------|-----|---------|---------------|
| 215–246 | 54483 – 55996 | **≈ +730 µs** each | **0** |
| **247** | **56179.343** | **≈ +1000 ms** | **+327** |
| 248–255 | 56191 – 56423 | **≈ +730 µs** each | **0** |
| **256** | **56486.292** | **≈ −999 ms** | **−326** |

**+327 / −326 ≈ ±(int)F0 = ±326`** — exactly one coarse-spin-rate unit. This matches the
tempo2 `pnNew` formula using `(int)F0` (not full `F0`) and `(int)bbat` day truncation
(see algorithm table above). **40/42 TOAs agree on the integer turn**; only 2 are off by
±326.

**EFF.EBPP.2639** (64 TOAs): tim indices **558–561** at MJD **56297 – 56486** (all
MJD > 56289) carry the isolated 124 ms debt; the other 60 TOAs are sub-µs.

### Experiment 20 — Ablation of bad TOAs

| Sub-tim | n | RMS (raw) | RMS (folded) |
|---------|---|-----------|--------------|
| Full fixture | 1369 | **46.8 ms** | **731 µs** |
| Minus 2639 late-4 | 1365 | **38.3 ms** | **46 µs** |
| Minus 1360 mid-4 (MJD 56000–56289) | 1365 | **38.2 ms** | — |
| Minus **all 8 bad TOAs** (2×1360 + 4×2639 + 2×1360 endpoints) | 1361 | **27.1 ms** | **735 µs** |
| Minus all `EFF.EBPP.1360` | 1327 | **27.4 ms** | **754 µs** |

Removing the **8 integer-misassigned TOAs** cuts raw RMS nearly in half; folded RMS stays
at the **≈ 730 µs** level (the shared fractional offset on the remaining EFF 1360 TOAs in
full mix).

### Experiment 21 — Absolute phase / frequency / delay audit

| Check | Result |
|-------|--------|
| `phaseresiduals` vs `residual × F0` (libstempo) | **Identical** (turns) |
| JUG fractional phase vs libstempo `phaseresiduals` (1410) | **≈ 0.238 turns ≈ 730 µs** (constant offset in full mix) |
| JUG fractional phase vs libstempo (1360 isolated) | **71.3 turns RMS** (integer scatter from 2 bad TOAs) |
| `freq_bary` JUG vs libstempo `ssbfreqs` (EFF) | **≈ 0.000026 MHz RMS** — **not causal** |
| Roemer JUG vs libstempo | **Sign flip only** (`jug ≈ −oracle`); Phase A **≈ 616 µs** uses corrected convention |
| Integer-turn correlation with MJD / freq / roemer (EFF) | **|r| < 0.09** — scatter is **not** tracking a smooth delay gradient |

**Note on pulse-number scales:** JUG `pulse_number` and libstempo `pulsenumbers` use
different zero references in isolated sub-tims (0/42 match on 1360 isolated); this is
expected and not the parity defect. The `-pn` tim flags match libstempo on the **full**
tim (experiment 13).

### Updated root-cause assessment

| Component | Status |
|-----------|--------|
| Delay geometry (`pint` vs `tempo2`, clocks, BIPM, freq) | **Ruled out** for ms-scale debt |
| `TRACK -2` integer assignment (`base_pn + (-pn)` vs per-TOA `nlong`) | **Fixed (2026-07-03)** — was ±`(int)F0` turns on **8/347 EFF TOAs** |
| Full tempo2 `pnNew` + `bbat`-based `phase2` | **Not yet viable** in JUG — needs accurate `bbat` and tempo2 spin-phase decomposition |
| Fractional offset on remaining EFF 1360 TOAs in full mix | **≈ 730 µs** constant (likely `phas1` / global-anchor effect, H4) |
| `eff2gps.clk` data quality | Still a candidate for **fractional** µs-level polish, not the ±326-turn jumps |
| Remaining ~0.4-turn outliers (tim idx 247, 256, 561) | **Fixed (2026-07-03)** — `-addsat` fractional correction (`F0·s − fortran_nlong(F0·s)`) |

---

## TRACK -2 implementation (2026-07-03)

**Code:** `jug/residuals/simple_calculator.py` — `compute_phase_residuals()` when
`TRACK=-2` and all `-pn` flags are present.

**Tests:** `tests/test_tempo2_ipta_dr2_j0613_parity.py`
(`test_epta_j0613_ipta_dr2_track_minus2_debt_reduced` passes; full parity remains
`xfail` until sub-ns gate).

### What was wrong (pre-fix)

JUG's `TRACK -2` path used:

```text
base_pn = round(phase[argsort(dt)[0]])   # earliest emission time, not tim index 0
pulse_number[i] = base_pn + external_pn[i]
frac_phase[i] = phase[i] - pulse_number[i]
```

tempo2 (`formResiduals.C` ~L2169–2295) instead:

1. Sets `phas1 = fortran_mod(phase5[first_active_TOA], 1.0)` — **tim index 0** in our fixture.
2. Subtracts `phas1` from all TOAs.
3. Uses **per-TOA** `nphase = fortran_nlong(phase5[i])` for the fractional residual.
4. Optionally adds `addPhase` from `pnNew` vs `-pn` flags.

Diagnostic proof (experiment 19, pre-fix): at tim idx 247,
`fortran_nlong(phase5[247]) − (base_pn + pn[247]) = +327` turns — exactly the observed
+1000 ms residual.

### What was implemented

| Step | tempo2 | JUG (post-fix) |
|------|--------|----------------|
| Anchor | `phas1 = fortran_mod(phase5[0], 1.0)` | Same |
| Integer per TOA | `fortran_nlong(phase5[i])` | Same (`_fortran_nlong`) |
| Fractional residual | `phase5[i] − nphase` | Same |
| `pnNew` / `addPhase` | `nf0·ntpd·86400 + nlong(phase5)` vs `-pn` | **Not applied** (see below) |
| Spin phase | `phase2` from `bbat` + `torb` | Taylor series on emission `dt` (unchanged) |

Helpers added: `_fortran_mod`, `_fortran_nlong`, `_compute_tempo2_spin_phase` (bbat-based
`phase2` — kept for a future full `pnNew` path, not used in the shipping TRACK -2 branch).

### What was tried but reverted

A literal port of the full `pnNew` block (`phaseint = (int)F0 · ntpd · 86400`, `pn0`
latch, `addPhase = pnNew − pnAct`) was attempted with several `bbat` proxies:

| `bbat` source | Continuous match to libstempo `toas` | Integer-day `ntpd` |
|---------------|----------------------------------------|--------------------|
| `tdb_mjd` | ~65 ms RMS | 6 mismatches vs `toas` |
| `tdb + (roemer_shapiro − tropo)/day` | ~616 ms RMS | 10 mismatches |
| libstempo `toas` (oracle) | exact | 0 mismatches |

Even with oracle `toas` as `bbat` and tempo2 `phase2` spin phase, `pnNew`/`addPhase`
still left **~2 s RMS** — JUG's emission-time Taylor phase does not decompose into the
`phaseint + nlong(phase5)` form tempo2 expects. Applying `addPhase` on top of the wrong
phase scale produced **~10¹¹ ms** catastrophic errors.

**Conclusion:** the ms-scale debt was in **integer assignment**, not in the `pnNew` coarse
reconstruction. Per-TOA `fortran_nlong(phase5[i])` after `phas1` from index 0 is the
correct fix for JUG's existing spin-phase path.

### Post-fix measurements (full 1369 TOAs, `TRACK -2` + `-pn`)

| Quantity | Pre-fix | After TRACK -2 fix | After `-addsat` fix |
|----------|---------|-------------------|---------------------|
| RMS vs libstempo | **46.807 ms** | **≈ 57 µs** | **≈ 2 µs** |
| max \|Δ\| | **≈ 1.0 s** | **≈ 1.25 ms** | **≈ 68 µs** |
| ±326-turn EFF TOAs (idx 247, 256, 558–561) | 8 bad | **0 at ±326**; idx 558–560 **sub-µs** | unchanged |
| Tim idx 247, 256, 561 residual | **≈ ±1000 ms** | **≈ ±1.2 ms** (~0.4 turns) | **< 100 µs** |

### Remaining gap (~2 µs RMS)

1. **Bulk fractional offset** on most TOAs in the full mix — not integer-turn jumps.
2. **Sub-ns gate** (5 ns) not met; parity test remains `xfail`.
3. **Fitter path** (`optimized_fitter.py`) does not yet pass TRACK -2 / `-addsat` flags.

### Next steps (if pursuing sub-ns)

1. Expose **`bat`/`bbat`** in the JUG geometry chain (optional full `pnNew` path).
2. Wire TRACK -2 / `-addsat` through **`optimized_fitter.py`**.
3. Optional: lock **EFF.EBPP.1410-only** sub-tim pytest at **≈ 78 ns** (experiment 18).

---

## References

- Fixture: `tests/data_tempo2/epta_j0613_t2_ipta_all/`
- Sibling fixture: `tests/data_tempo2/epta_j0613_t2_nrt1400/`
- CI: `tests/test_tempo2_ipta_dr2_j0613_parity.py`
- Broader parity notes: [`TEMPO2_PARITY_GAPS.md`](TEMPO2_PARITY_GAPS.md)
- JUG entry point: `jug.residuals.simple_calculator.compute_residuals_simple`
- Oracle: `jug.testing.tempo2_reference.tempo2_reference`
- Phase A: `jug.testing.phase_a_comparison.compare_fixture_phase_a`
- TEMPO2 runtime clocks: `$TEMPO2/clock` (typically `/opt/software/tempo2/T2runtime/clock`)
- tempo2 `TRACK -2` reference: `formResiduals.C` in tempo2 source tree
