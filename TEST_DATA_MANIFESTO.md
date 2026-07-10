# JUG test data manifesto

**Status:** active policy (2026-07-09)

This document inventories what timing test data JUG currently ships, where it came
from, and the principles we should use when adding or trimming fixtures. It is **not**
a license statement — all listed collaboration excerpts are treated as public research
data today — but we have not yet decided which subsets belong in the repository
long-term.

---

## Principles

JUG is a **timing engine**. Most correctness properties — delay kernels, binary
dispatch, design-matrix columns, autodiff Jacobians — can be validated on:

- synthetic `GeneralFitSetup` objects (no par/tim files at all);
- a **handful of TOAs** (single-digit to low hundreds);
- **single-PTA** par/tim pairs, not multi-PTA composite hosts.

Full multi-thousand-TOA datasets are useful for integration stress tests and
historical parity with PINT/libstempo, but they are **unnecessary for most unit and
regression tests** and slow CI unnecessarily.

**Working policy (2026-07-08):**

1. Prefer **synthetic** setups for new engine tests (autodiff, dispatch, traceability).
2. For tempo2 par/tim parity, prefer **`tests/data_tempo2_sim/`** — libstempo-generated,
   ideal-noiseless fixtures with **5–12 TOAs** each.
3. Keep **`tests/data_tempo2/`** real excerpts as **transitional anchors** for TIM-format
   edge cases (multi-`-sys`, large NG5/TDB debt probes, historical IPTA layouts).
4. For other real par/tim tests, prefer **single-PTA MPTA DR2** fixtures trimmed to at
   most a few hundred TOAs unless a test explicitly needs more.
5. Revisit whether large bundled MPTA files (~3k–7k TOAs) should be replaced by trimmed
   variants in-repo, with full files optional via env vars.

**Tempo2/IPTA iteration rule (2026-07-08):** do not use the full tempo2 pytest sweep
or all 65 IPTA DR2 pulsars as the default development loop. `pytest tests/ -k
"tempo2"` currently selects hundreds of oracle-heavy tests, and a full IPTA DR2 TDB
campaign can run for hours because individual pulsars contain hundreds to thousands
of TOAs. Use focused fixtures first (`tests/data_tempo2_sim/`, wsrt167, J0613 fast
gates), then run one real IPTA pulsar or a small named batch when a specific parity
question requires it. Full sweeps are release/explicit-request jobs.

Current real-data mini gates: `wsrt167_mini` and `epta_j0613_nrt1400_mini` are
20-TOA excerpts for inner-loop tempo2 checks; `epta_j0613_addsat_min` remains the
11-TOA TRACK −2 / `-addsat` regression guard. Full `wsrt167` (167 TOAs) is the
primary JAX native-chain oracle (formBats, bbat, torb, autodiff columns); full
`epta_j0613_t2_ipta_all` (1369 TOAs) is the production-scale host residual gate.

---

## Collections at a glance

| Directory | Role | Typical size | Primary author / date |
|-----------|------|--------------|------------------------|
| `tests/data_golden/` | Fast CI, golden JSON, synthetic/hand-trimmed | 21–2800 TOA lines | Matt Miles, Feb 2026 |
| **`tests/data_tempo2_sim/`** | **Default tempo2 parity: libstempo-simulated par/tim** | **6–10 TOAs** | **Generated, Jul 2026** |
| `tests/data_tempo2/` | Tempo2/libstempo parity (real EPTA/PPTA/IPTA excerpts + NG5) | 10–1369 TOAs | Rutger, May–Jul 2026 |
| `tests/data_mpta/` | Bundled **MPTA DR2** par/tim | ~1.3k–7.2k TOA lines | Rutger, May 2026 |
| `data/pulsars/` | Demo pars, NG15yr J1909 PINT partim | up to ~35k TOA lines | Matt / various |

Loaders:

- `tests/test_paths.py` — MPTA fixtures + env overrides + legacy Matt paths
- `tests/tempo2_fixtures.py` — **`data_tempo2_sim/manifest.json`** and `data_tempo2/manifest.json`
- `tools/generate_tempo2_sim_fixtures.py` — maintainer-only generator for simulated fixtures
- `tests/data_mpta/manifest.json`, `tests/data_tempo2/manifest.json`, `tests/data_tempo2_sim/manifest.json` — provenance records

---

## 1. `tests/data_golden/`

**Added by:** Matt Miles (`74105bd`, 2026-02-27 — testing brought from experimental branch).

**Purpose:** Fast regression, PINT three-way parity, golden reference values. Not
collaboration release bundles.

| Asset | Origin | ~TOAs | Notes |
|-------|--------|-------|-------|
| `J1909_mini.par/tim` | Hand-trimmed J1909 | **21** | Primary fast CI; `J1909_mini_golden.json` |
| `J1909_proper.par/tim` | J1909 variant | ~101 | Fit/correctness |
| `J1909_noisefree.par`, `J1909_parity*.par/tim` | Derived variants | up to ~2800 lines | PINT/JUG parity |
| `J0437_tdb.par`, `J0437_tdb_noisefree.par` | J0437 par only | — | Pairs with `data_mpta/j0437` tim |
| `J0125-2327_parity.npz/json` | Precomputed reference | — | J0125 parity artifacts |
| `ng15yr_pint_golden.json` | NG 15yr PINT WRMS table | **JSON only** | No par/tim (Matt, `c98f03f`) |

**Binary on mini:** ELL1 (J1909-3744).

---

## 2. `tests/data_tempo2_sim/` — simulated tempo2 parity (default)

**Added:** 2026-07-08 — libstempo `fakepulsar` generator at
`tools/generate_tempo2_sim_fixtures.py`.

**Purpose:** Fast, comprehensive tempo2 parity on **ideal noiseless TOAs** with
orthogonal option coverage. These fixtures are **hand-authored template pars**, not
collaboration release excerpts.

**Manifest:** `tests/data_tempo2_sim/manifest.json`  
**Loader:** `tests/tempo2_fixtures.py` (`list_tempo2_sim_fixtures()`)  
**Tests:** `tests/test_tempo2_simulated_fixtures.py`

Regenerate committed artifacts (maintainer only):

```bash
cd ref-packages/jug
PYTHONPATH=.:tests python tools/generate_tempo2_sim_fixtures.py
PYTHONPATH=.:tests python tools/generate_tempo2_sim_fixtures.py --check
```

| Fixture ID | Binary | TOAs | Option tags (summary) | Design-matrix params |
|------------|--------|------|------------------------|----------------------|
| `sim_isolated_tcb` | isolated | 6 | TCB, DILATEFREQ=Y | F0, F1, DM |
| `sim_t2_tcb` | T2 | 8 | TCB | F0, PB, A1, EPS1, EPS2 |
| `sim_ell1_tcb` | ELL1 | 8 | TCB | F0, PB, A1, EPS1, EPS2 |
| `sim_ell1h_tcb` | ELL1H | 8 | TCB | F0, PB, A1, EPS1, EPS2 |
| `sim_dd_tcb` | DD | 8 | TCB | F0, PB, A1 |
| `sim_ddh_tcb` | DDH | 8 | TCB | F0, PB, A1 |
| `sim_bt_tcb` | BT | 8 | TCB | F0, PB, A1 |
| `sim_ddk_tcb` | DDK | 8 | TCB | F0, PB, A1 |
| `sim_dd_tdb` | DD | 8 | TDB | F0 (strict gate; TDB spin-epoch debt closed) |
| `sim_dd_ecliptic_tcb` | DD | 8 | ecliptic coords | F0, PB (strict gate; ecliptic frame debt closed) |
| `sim_t2_track2_pn` | T2 | 8 | TRACK=-2, TIM=-pn | — |
| `sim_t2_track2_addsat` | T2 | 10 | TRACK=-2, TIM=-pn, TIM=-addsat | — |
| `sim_t2_multisys` | T2 | 8 | TIM=multi-sys | — |
| `sim_fd_tcb` | T2 | 8 | FD | F0, FD1 |
| `sim_dilatefreq_no` | isolated | 6 | DILATEFREQ=N | F0, DM |

Most simulated fixtures gate at **5 ns RMS** vs libstempo. Known debt classes use
fixture-specific relaxed gates documented in `tests/test_tempo2_simulated_fixtures.py`.

---

## 3. `tests/data_mpta/` — MPTA DR2 (legacy real-data source)

**Added by:** Rutger van Haasteren (`074413b`, 2026-05-29 — “Bundle MPTA DR2 test
fixtures and remove hard-coded local paths”).

**Manifest:** `tests/data_mpta/manifest.json`

**Provenance:** Australian **MPTA DR2** (not IPTA DR2). Original paths recorded as
`data-check/MPTA_DR2/...`.

**Access:** `tests/test_paths.py` (`get_j0613_paths()`, etc.).

| Fixture ID | Pulsar | Binary | ~TIM lines | Main tests |
|------------|--------|--------|------------|------------|
| **`j0613_ell1h`** | **J0613-0200** | **ELL1H** | **~3069** | **Primary binary/FD/H3 target; trim candidate** |
| `j2241_fb` | J2241-5236 | ELL1 + FB0..FB10 | ~3407 | FB orbital frequency |
| `j1713_binary` | J1713+0747 | ELL1 | ~1275 | Binary delay / prebinary cache |
| `j1909_t2` | J1909-3744 | T2 | ~7201 | Default path tests |
| `j1022_ell1h` | J1022+1001 | ELL1H | ~2947 | H3/STIG |
| `j0125_ell1h` | J0125-2327 | ELL1H | ~3172 | `test_ell1h_j0125.py` |
| `j0437` | J0437-4715 | ELL1 | ~3519 | Three-way parity |

Pars are TDB-converted (`*_tdb.par`) except J0437.

**Trimmed fixture now bundled:** `j0613_ell1h_trim300` at
`tests/data_mpta/j0613_ell1h/J0613-0200_trim300.tim` (~300 TOAs, currently 307).

---

## 4. `tests/data_tempo2/` — tempo2 / libstempo parity (transitional real excerpts)

**Added by:** Rutger — `6dcb732` (2026-05-29) EPTA/PPTA excerpts; `dc17cfb` (2026-06-02)
NG5 Case B/C.

**Manifest:** `tests/data_tempo2/manifest.json`  
**Loader:** `tests/tempo2_fixtures.py`

| Fixture ID | Provenance | Binary | Bundled TOAs |
|------------|------------|--------|--------------|
| `epta_j0030_isolated` | EPTA DR2 (J0030) | isolated | **10** | Phase C TZR: **~4.7 ns RMS** |
| `wsrt167` | IPTA DR2 J0613 WSRT low-band excerpt | T2 | **167** | Phase D / wsrt167 fixes: **~1.4 ns RMS** after tropo-in-dt + longdouble wrap |
| `epta_j1909_t2` | EPTA DR2 (J1909) | T2 | **49** |
| `epta_j1918_ddh` | EPTA DR2 (J1918) | DDH | **12** |
| `ppta_j1741_ell1` | PPTA DR3 UWL | ELL1 | **111** |
| `ppta_j1902_ell1h` | PPTA DR3 | ELL1H | **120** |
| `ng5_j1600_tdb_equatorial` | NANOGrav dfg+12 Case B | DD | **625** |
| `ng5_j1600_tdb_ecliptic_cross_engine` | NANOGrav dfg+12 Case C | DD | **625** |
| **`epta_j0613_t2_nrt1400`** | **IPTA DR2 EPTA J0613 (NRT.BON.1400 excerpt)** | **T2** | **120** |
| **`epta_j0613_t2_ipta_all`** | **IPTA DR2 EPTA J0613 (full INCLUDE collection)** | **T2** | **1369** |

**IPTA DR2 EPTA J0613 (added 2026-07-03):** multi-backend TIM layouts used for TRACK −2
and `-addsat` regression gates. Prefer the simulated `sim_t2_track2_*` fixtures for
fast CI; keep these excerpts for real TIM-format probes.

Manifest `source_*` fields point at external collaboration paths; the repo holds
**reduced TIM copies**, not full collaboration releases.

**Used by:** `@pytest.mark.tempo2` — residual, design matrix, fit, TZR parity (Phase C),
TRACK −2 pnNew parity (Phase D). Parity status:
[`PARITY_ROADMAP.md`](PARITY_ROADMAP.md).

---

## 5. `data/pulsars/` — demos and large optional assets

| Path | Notes |
|------|-------|
| `J1909-3744_demo.par`, `J1909-3744_noise_demo.par` | Example / noise demos |
| `NG_15yr_partim/J1909-3744_PINT_*.par/tim` | NG 15yr PINT partim (~35k TIM lines) |

Clock and observatory files under `data/clock/`, `data/observatory/` are runtime
infrastructure (hashed in `data/manifest.json`).

---

## Timeline (who added what)

| Date | Author | Change |
|------|--------|--------|
| 2026-02 | Matt Miles | `data_golden`, early J1909/J0437 tests, NG15yr golden JSON |
| 2026-05-29 | Rutger van Haasteren | `data_tempo2` (EPTA/PPTA excerpts), tempo2 parity infra |
| 2026-05-29 | Rutger | `data_mpta` (full MPTA DR2 bundles) |
| 2026-06-02 | Rutger | NG5 Case B/C in `data_tempo2` |
| 2026-07-03 | Parity investigation | IPTA DR2 EPTA J0613 fixtures + fast gates |
| 2026-07-08 | Simulated tempo2 suite | `data_tempo2_sim` + generator + structural/parity tests |

---

## What we do *not* ship

- **Full IPTA DR2 multi-PTA composite** sessions — not bundled in JUG.
- **Full** EPTA/PPTA release TOA sets — only excerpts in `data_tempo2`.

---

## Recommended test-data tiers (for new work)

| Tier | Data | Use for |
|------|------|---------|
| **A — Synthetic in-memory** | `GeneralFitSetup` in pytest | Dispatch, traceability, zero-delta |
| **B — Simulated tempo2 par/tim** | **`tests/data_tempo2_sim/`** (6–10 TOAs) | **Default tempo2 parity CI** |
| **C — Mini real tempo2** | `data_tempo2` excerpts (10–167 TOAs) | TIM-format anchors, TRACK/NG5 debt |
| **D — Trimmed MPTA** | `j0613_ell1h_trim300` (~300 TOAs) | Legacy binary autodiff / whitening |
| **E — Full MPTA** | Current `data_mpta` files | Legacy fit tests |
| **F — Local / env var** | Full IPTA/MPTA via `JUG_TEST_*` | Manual integration only |

---

## Open decisions (revisit later)

1. Which **`data_tempo2` real excerpts** should remain after simulated coverage is green?
2. Should **`data_mpta` full TIM files** stay in-repo, or become env-var-only?
3. Keep **`j0613_ell1h_trim300`** as the MPTA autodiff anchor?
4. Document redistribution expectations per collaboration in `CONTRIBUTING.md`.

---

## Related docs

- Tempo2 parity theory and policy: [`PARITY_THEORY.md`](PARITY_THEORY.md)
- Tempo2 parity status and roadmap: [`PARITY_ROADMAP.md`](PARITY_ROADMAP.md)
- MPTA loader: `tests/test_paths.py`
- Tempo2 fixtures: `tests/tempo2_fixtures.py`
