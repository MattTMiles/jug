# JUG test data manifesto

**Status:** draft for later review (2026-07-03)

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

**Working policy (2026-07-03, pending final review):**

1. Prefer **synthetic** setups for new engine tests (autodiff, dispatch, traceability).
2. For real par/tim tests, prefer **single-PTA MPTA DR2** fixtures, **trimmed to at
   most a few hundred TOAs** unless a test explicitly needs more.
3. **J0613-0200** (`j0613_ell1h`, ELL1H binary) is the primary real-data anchor for
   binary autodiff and MetaPulsar-style whitening — not IPTA DR2 composite sessions.
4. Keep tempo2/libstempo parity fixtures (`data_tempo2`) as-is for now; they are already
   small excerpts (10–625 TOAs for EPTA/PPTA rows).
5. Revisit whether large bundled MPTA files (~3k–7k TOAs) should be replaced by trimmed
   variants in-repo, with full files optional via env vars.

---

## Collections at a glance

| Directory | Role | Typical size | Primary author / date |
|-----------|------|--------------|------------------------|
| `tests/data_golden/` | Fast CI, golden JSON, synthetic/hand-trimmed | 21–2800 TOA lines | Matt Miles, Feb 2026 |
| `tests/data_mpta/` | Bundled **MPTA DR2** par/tim | ~1.3k–7.2k TOA lines | Rutger, May 2026 |
| `tests/data_tempo2/` | Tempo2/libstempo parity (EPTA/PPTA excerpts + NG5) | 10–625 TOAs | Rutger, May–Jun 2026 |
| `data/pulsars/` | Demo pars, NG15yr J1909 PINT partim | up to ~35k TOA lines | Matt / various |

Loaders:

- `tests/test_paths.py` — MPTA fixtures + env overrides + legacy Matt paths
- `tests/tempo2_fixtures.py` — `data_tempo2/manifest.json`
- `tests/data_mpta/manifest.json`, `tests/data_tempo2/manifest.json` — provenance records

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

## 2. `tests/data_mpta/` — MPTA DR2 (preferred real-data source)

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

## 3. `tests/data_tempo2/` — tempo2 / libstempo parity

**Added by:** Rutger — `6dcb732` (2026-05-29) EPTA/PPTA excerpts; `dc17cfb` (2026-06-02)
NG5 Case B/C.

**Manifest:** `tests/data_tempo2/manifest.json`  
**Loader:** `tests/tempo2_fixtures.py`

| Fixture ID | Provenance | Binary | Bundled TOAs |
|------------|------------|--------|--------------|
| `epta_j0030_isolated` | EPTA DR2 (J0030) | isolated | **10** |
| `epta_j1909_t2` | EPTA DR2 (J1909) | T2 | **49** |
| `epta_j1918_ddh` | EPTA DR2 (J1918) | DDH | **12** |
| `ppta_j1741_ell1` | PPTA DR3 UWL | ELL1 | **111** |
| `ppta_j1902_ell1h` | PPTA DR3 | ELL1H | **120** |
| `ng5_j1600_tdb_equatorial` | NANOGrav dfg+12 Case B | DD | **625** |
| `ng5_j1600_tdb_ecliptic_cross_engine` | NANOGrav dfg+12 Case C | DD | **625** |

Manifest `source_*` fields point at EPTA_DR2 / PPTA_DR3 / MetaPulsar notebook paths;
the repo holds **reduced TIM copies**, not full collaboration releases.

**Used by:** `@pytest.mark.tempo2` — residual, design matrix, fit, TZR parity.

---

## 4. `data/pulsars/` — demos and large optional assets

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

---

## What we do *not* ship

- **Full IPTA DR2 multi-PTA composite** sessions (EPTA+PPTA+ng9 on one host) — used in
  MetaPulsar notebooks but not bundled in JUG.
- **Full** EPTA/PPTA release TOA sets — only tiny excerpts in `data_tempo2`.

---

## Recommended test-data tiers (for new work)

| Tier | Data | Use for |
|------|------|---------|
| **A — Synthetic** | `GeneralFitSetup` in pytest | Dispatch, traceability, zero-delta, family coverage |
| **B — Mini real** | `J1909_mini` (21 TOAs), tempo2 excerpts (10–120 TOAs) | Fast CI smoke |
| **C — Trimmed MPTA** | **`j0613_ell1h` trimmed to ~300 TOAs** (proposed) | Binary autodiff, design matrix, MetaPulsar JUG backend |
| **D — Full MPTA** | Current `data_mpta` files | Legacy fit/parity tests; consider demoting over time |
| **E — Notebooks / local** | IPTA DR2, full MPTA via env vars | Manual integration only |

---

## Open decisions (revisit later)

1. Should **`data_mpta` full TIM files** stay in-repo, or should trimmed variants become
   the default with full files via `JUG_TEST_*` env vars?
2. Should **`data_tempo2` EPTA/PPTA excerpts** remain, or be regenerated from a single
   canonical trimmed set?
3. Keep **`tests/data_mpta/j0613_ell1h/J0613-0200_trim300.tim`** (par unchanged,
   tim evenly subsampled) as the standard autodiff/whitening anchor?
4. Document redistribution expectations per collaboration in `CONTRIBUTING.md` when we
   finalize the policy.

---

## Related docs

- Tempo2 parity gaps: `TEMPO2_PARITY_GAPS.md`
- Tempo2 project brief: `TEMPO2_COMPATIBILITY_PROJECT.md`
- MPTA loader: `tests/test_paths.py`
- Tempo2 fixtures: `tests/tempo2_fixtures.py`
