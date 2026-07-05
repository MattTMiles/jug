# Tempo2 compatibility mode — policy and architecture

This document defines **what** `compatibility="tempo2"` means in JUG: locked product
decisions, oracle policy, fixture matrix, acceptance metrics, and delivered architecture.

**Status (2026-07-05):** Cases A/B/C (TCB + NG5 TDB) are **green** on raw residual gates
(~1–2 ns vs libstempo). IPTA DR2 workloads are **partially green** — see measured debt in
[`TEMPO2_PARITY.md`](TEMPO2_PARITY.md).

**Measured gaps, active work queue, investigation log:** [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md)

---

## Runtime dependencies vs test oracles

**JUG must not depend on libstempo, tempo2, or pytempo.** The shipped package
(`jug-timing` in `pyproject.toml`) has no runtime dependency on any of them.
`compatibility="tempo2"` is implemented **natively inside JUG** (jplephem, native
delay kernels, native phase bookkeeping).

| Package | Runtime JUG | Test / debug only |
|---------|-------------|-------------------|
| **libstempo** + tempo2 | **Must not** | pytest acceptance oracle today (`jug/testing/tempo2_reference.py`, vendored `jug/testing/sandbox_tempo2.py`) |
| **pytempo** | **Must not** | Planned per-TOA diagnostic oracle (`ref-packages/pytempo`, external repo) |

**Known mistake to correct:** parity work currently routes through libstempo inside
`jug/testing/` (including a vendored libstempo sandbox). That is **test harness
coupling**, not an architectural choice. JUG must not grow a hard dependency on
libstempo; the oracle layer should remain optional, isolated under tests/tools, and
replaceable (e.g. golden vectors, external harness, or pytempo in a separate process).

Parity is **defined** by matching tempo2/libstempo on identical par+tim inputs, but
**implemented** without calling them at runtime. See §2 and §3.

---

## 1. Scope

### What JUG parity is

Given a `.par` file and a `.tim` file, pre-fit residuals (and, where gated, design-matrix
columns) from `JUG(compatibility="tempo2")` must match libstempo/tempo2 on **the same
inputs**. Nothing else participates in that definition.

### What JUG parity is not

**MetaPulsar has nothing to do with JUG parity.** Notebook or export paths named in parity
docs are **dataset provenance only** — a way to identify a par+tim pair. Downstream
orchestration does not affect JUG↔libstempo residual parity.

Green pytest on Cases A/B/C does **not** imply end-to-end parity for every par/tim pair,
θ≠0 NumPy/JAX agreement on real IPTA workloads, or readiness for unconstrained production
use outside curated tests.

---

## 2. Locked decisions (do not reopen without explicit review)

| Question | Decision |
|----------|----------|
| What does `compatibility="tempo2"` mean? | Match tempo2 **residuals and phase conventions end-to-end**, not isolated delay-term tweaks or post-hoc centering tricks. |
| Parity metric for tempo2 mode | **Raw pre-fit residuals** vs libstempo — same gate as `tests/test_tempo2_residual_parity.py` (RMS, p99, max, WRMS on uncentered δ). **Do not** subtract a weighted (or any other) mean for tempo2 acceptance. |
| Phase / mean subtraction | tempo2 uses an **unweighted** phase offset; pint mode uses **weighted**. JUG(tempo2) applies tempo2 phase semantics internally; parity compares residuals **as returned**. |
| Implementation strategy | **Native only.** Reimplement tempo2-equivalent physics inside JUG. **Do not** wrap tempo2, libstempo, or tempo2 plugins at runtime or as a fallback. |
| Runtime dependencies | **No libstempo, tempo2, or pytempo.** Not in `pyproject.toml` dependencies; not importable from `jug/` production modules. Current libstempo use under `jug/testing/` is test-only coupling to remove over time. |
| Test oracle — acceptance | **libstempo** via `jug/testing/tempo2_reference.py` for scalar residual gates (pytest only, optional extra). Oracle use does not permit wrapping tempo2 in the JUG(tempo2) code path. |
| Test oracle — diagnostics | **pytempo** (`ref-packages/pytempo`, separate repo) is the **intended** per-TOA diagnostic oracle. Not a JUG dependency. See §3 — several pytempo fields are not yet reliable on IPTA workloads. |
| Shared PINT-family stack in tempo2 mode | On TDB, tempo2 mode must **not** rely on the pint-mode delay pipeline for terms tempo2 implements differently. |
| Ephemeris / Roemer / Shapiro | tempo2-equivalent native table integration and delay geometry. Matching the `EPHEM` keyword alone is insufficient. |
| Omitted par keywords on TDB | Follow **tempo2 implicit defaults** (IF99, DILATEFREQ, etc.) when par omits them — not PINT defaults. |
| TZR / absolute phase | **Mode-specific**: tempo2 native TZR geometry and clocks in tempo2 mode; pint path in pint mode. |
| Demo / notebook display | **Raw δ** for tempo2 compatibility panels. **Weighted-mean-centered δ** only for pint-family-vs-pint-family comparisons. |
| Canonical TDB fixtures | Cases A (TCB), B (equatorial NG5), and C (ecliptic cross-engine) must stay green. |
| PINT vs tempo2 cross-engine floor | Closing PINT↔tempo2 gaps inside PINT is **out of scope**. |
| Design matrix in tempo2 mode | Use `design_matrix_method="autodiff"`. **Do not** use `"analytic"` on tempo2 sessions (known broken). |

---

## 3. Oracle policy

Parity work may use **external oracles** for pytest and debugging. **None** of them are
JUG runtime dependencies, and none may be called from the `compatibility="tempo2"` code
path (`jug/residuals/`, `jug/delays/`, fitters, GUI, etc.).

| Layer | Tool | JUG dependency? | Role |
|-------|------|-----------------|------|
| **Runtime JUG** | `compute_residuals_simple(..., compatibility="tempo2")` | — | Native port under test |
| **Acceptance (scalar gates)** | `jug.testing.tempo2_reference` (libstempo sandbox) | **No** (test-only today) | Raw pre-fit residual RMS / p99 / max for pytest debt pins |
| **Diagnostics (intended)** | [`pytempo`](../../pytempo) → `toa_diagnostics()` / `phase_diagnostics()` | **No** (external repo) | Per-TOA tempo2 term dumps — **partially working**; see pytempo bug list in parity investigations |
| **Legacy (thin)** | `jug.testing.tempo2_diagnostics` (libstempo properties) | **No** (test-only) | Superseded by pytempo when its diagnostic fields are fixed |

### pytempo package

[`ref-packages/pytempo`](../../pytempo) is a standalone tempo2 wrapper built for this
parity project (vendored from libstempo `sandbox`). It exposes per-TOA `obsn[]` fields
that libstempo properties do not surface cleanly. **68 tests pass** on the bundled J1909
fixture; `nphase` and `phase_offset_turns` are available on the current build.

```python
from pytempo.sandbox import tempopulsar

psr = tempopulsar(parfile=par, timfile=tim, dofit=False)
diag = psr.toa_diagnostics(removemean=False)   # deterministic delay/phase terms
```

Install: `pip install -e ref-packages/pytempo` (requires `$TEMPO2` runtime). Full API:
[`pytempo/README.md`](../../pytempo/README.md).

Key `toa_diagnostics()` fields for parity work:

| Field | Units | Use |
|-------|-------|-----|
| `bbat_mjd`, `bat_mjd`, `pet_mjd` | MJD | Epoch comparison vs JUG |
| `roemer_sec`, `sun_shapiro_sec`, `torb_sec` | seconds | Delay terms |
| `freq_ssb_hz` | Hz | Barycentric frequency |
| `phase_turns`, `nphase` | turns | TRACK −2 / wrapping |
| `phase_offset_turns` | turns | Intended for `-padd`; **currently broken on IPTA** (reads unused `phaseOffset`) |
| `residual_sec`, `prefit_residual_sec` | seconds | **Not** drop-in acceptance oracles on TRACK −2 workloads — use `psr.residuals()` for scalar checks |
| `pulse_number` | integer | Raw `obsn[].pulseN` |

### Comparison conventions

1. **Residual acceptance** — compare raw pre-fit residuals as returned; no post-hoc
   weighted centering for tempo2 gates.
2. **Deterministic term comparison** — call pytempo with `removemean=False` when ranking
   per-TOA delay/phase deltas; mean-subtraction artifacts dominate full-mix vs isolated
   subset comparisons otherwise.
3. **TZRMJD anchoring** — for absolute-phase / TZR workloads, par must carry `TZRMJD`
   (and `TZRSITE`). Compare TZR-sensitive terms with JUG `subtract_tzr=True`. Separates
   delay physics from pulse-wrapping ambiguity on alternate PPTA export par/tim (~16 ns
   budget).
4. **Subset tim pitfall** — `-pn` flags are relative to **full-tim** `obsn[0]`. Running
   pytempo on an isolated sub-tim changes index-0 semantics; prefer full-tim oracle pull
   + mask when comparing phase fields on filtered subsets.

Diagnostic workflow (step-by-step): [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) §0.

---

## 4. Fixture matrix

Do not conflate these cases. Full paths and sizes: [`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md).

| Case | Description | CI status |
|------|-------------|-----------|
| **A. TCB regression** | `tests/data_tempo2/*` with `UNITS=TCB`, IF99, DILATEFREQ, equatorial astrometry | Green (~1–2 ns) |
| **B. NG5 equatorial TDB** | NG5 J1600 after `T2CMETHOD` removal only | Green (~1.3 ns WRMS) |
| **C. NG5 ecliptic cross-engine** | Layer-B harmonized par (LAMBDA/BETA, `ECL IERS2003`, DD, TZRMJD) | Green (~1.3 ns WRMS) |
| **IPTA DR2 J0613** | `epta_j0613_t2_ipta_all` (1369 TOAs), `epta_j0613_t2_nrt1400` (120 TOAs), ad hoc PPTA pairs | Partial — see parity doc |

Case C par keywords (reference):

```text
PSRJ 1600-3053
LAMBDA / BETA / PMLAMBDA / PMBETA
ECL IERS2003
BINARY DD
TZRMJD / TZRFRQ / TZRSITE gbt
UNITS TDB
CLK TT(BIPM2011)
EPHEM DE405
```

---

## 5. Acceptance metrics

`compute_residuals_simple(..., compatibility="tempo2")` raw pre-fit residuals must match
libstempo at:

| Metric | Gate |
|--------|------|
| RMS δ | < 5 ns |
| p99 \|Δ\| | < 10 ns |
| max \|Δ\| | < 25 ns |
| WRMS δ | < 5 ns |

**No weighted-mean centering** in comparisons. Tests: `tests/test_tempo2_residual_parity.py`.

Run tempo2-gated tests:

```bash
cd ref-packages/jug
JUG_TEST_TEMPO2=1 pytest tests/test_tempo2_*.py -q -o addopts=''
```

Requires libstempo + `$TEMPO2` runtime (see [`README.md`](README.md)).

---

## 6. Architecture (delivered)

Phases A–E (2026-06) delivered native tempo2 TDB geometry, mode-specific TZR, CI fixtures,
and design-matrix/fit parity on Cases B/C.

| Layer | Module | Role |
|-------|--------|------|
| Residual engine | `jug/residuals/simple_calculator.py` | `compute_residuals_simple`, `compute_phase_residuals` |
| Runtime conventions | `jug/residuals/engine_conventions.py` | `EngineConventionProfile` — physics defaults from par + tempo2 implicit rules |
| Diagnostic conventions | `jug/residuals/diagnostic_conventions.py` | Comparison knobs only (`residual_metric`, `term_set`, …) |
| Pint geometry | `PintDelayProvider` | Astropy JPL + PINT-family Roemer/Shapiro |
| Tempo2 TDB geometry | `Tempo2DelayProvider` → `_compute_tempo2_tdb_geometry_terms` | jplephem SPK + tempo2 delay kernels |
| Tempo2 TCB geometry | `Tempo2DelayProvider` → `_compute_tempo2_tcb_geometry_terms` | IFTE + epoch map (Case A) |
| TZR dispatch | `jug/residuals/tzr_geometry.py` | `compute_tzr_astrometry_tempo2` / `_pint`; `resolve_tzrmjd_epochs` |
| Ephemeris | `jug/delays/tempo2_ephemeris.py` | jplephem DE405 SPK state vectors |
| Tempo2 helpers | `jug/delays/tempo2_geometry.py` | Ecliptic / Roemer-Shapiro helpers |
| Phase / TRACK −2 | `compute_phase_residuals()` in `simple_calculator.py` | Shared between pint and tempo2 modes; `mean_mode` differs |
| Tempo2 spin scaffolding | `jug/residuals/tempo2_spin.py` | `bbat_mjd`, `addsat_track2_turn_delta`; `tempo2_spin=True` not production |
| libstempo acceptance oracle | `jug/testing/tempo2_reference.py` | Scalar residual gates |
| Phase A (legacy oracle) | `jug/testing/tempo2_diagnostics.py`, `phase_a_comparison.py` | libstempo properties — target: pytempo |
| **pytempo diagnostic oracle** | `ref-packages/pytempo` | Per-TOA term dumps — **primary for new debugging** |

**Mode split:** pint and tempo2 diverge in barycentric geometry, engine conventions, TZR
handling, binary param normalization, and mean subtraction (weighted vs unweighted). They
**share** `compute_phase_residuals()` for the phase path.

---

## 7. Non-goals

- Closing the PINT vs tempo2 cross-engine floor **inside PINT**.
- Making `compatibility="pint"` match tempo2.
- Wrapping tempo2 or libstempo inside the JUG(tempo2) runtime path.
- Adding libstempo, tempo2, or pytempo as JUG runtime or `pyproject.toml` dependencies.
- Using weighted-mean-centered residuals for tempo2 acceptance.

---

## 8. FAQ

**Q: Which oracle for debugging?**  
A: **pytempo** `toa_diagnostics()` for per-TOA term dumps. See [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) §0.

**Q: Which oracle for CI green?**  
A: **libstempo** raw residuals via `tempo2_reference()` — unchanged acceptance gates.

**Q: Should tempo2 parity tests subtract a weighted mean?**  
A: **No.** Compare raw residuals as returned.

**Q: Can we call libstempo or tempo2 inside `compatibility="tempo2"`?**  
A: **No** at runtime. External oracles for pytest/debug only.

**Q: Is libstempo a JUG dependency?**  
A: **No.** It is used in the test harness today (`jug/testing/`). That coupling is a
mistake to unwind — JUG must remain installable and runnable without libstempo.

**Q: Is pytempo a JUG dependency?**  
A: **No.** It lives in `ref-packages/pytempo` as an optional external diagnostic tool.

**Q: Why do TCB tests pass but IPTA full-TIM does not?**  
A: Case A activates IFTE, TCB epoch mapping, and unweighted phase mean. IPTA multi-backend
debt is phase-bookkeeping and per-group offsets — see parity doc, not missing TCB machinery.

**Q: Why does weighted-centering make JUG(tempo2) look like JUG(pint)?**  
A: On some TDB models delay shapes correlate; centering removes the ~61 ns phase-offset
signal from mean-subtraction convention differences and hides the libstempo gap.

**Q: Is `design_matrix_method="analytic"` OK in tempo2 mode?**  
A: **No.** Use `"autodiff"`.

---

## 9. Related documents

| Document | Role |
|----------|------|
| [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md) | Status dashboard, gaps, pytempo workflow, work queue, investigation log |
| [`TEST_DATA_MANIFESTO.md`](TEST_DATA_MANIFESTO.md) | Fixture provenance and sizes |
| [`README.md`](README.md) | Install, compatibility modes, pytest entry points |
| [`pytempo/README.md`](../../pytempo/README.md) | pytempo install and diagnostic API |
