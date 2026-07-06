# Development oracle harness (delete when JUG is standalone)

JUG is **standalone**: no runtime dependency on libstempo, tempo2, or pytempo.
During parity development we keep optional **external oracles** here and in
marked pytest modules.

## Current parity work (2026-07-05)

| Phase | Fix | Tests | Status |
|-------|-----|-------|--------|
| **C** | TZR apply modes (`tzr_geometry.py`) | `tests/test_tempo2_tzr_parity.py` | **Done** — J0030 ~4.7 ns RMS |
| **D Step 1** | tim ``-pn`` relative to obsn[0] | `tests/test_tempo2_track2_pnnew.py` | **Done** — pnNew oracle |
| **D Step 2** | ``phase5@bbat`` production wiring | — | **Ruled out** — ~17.5 ns vs ~16.4 ns production |
| **D Step 3** | ``-padd`` / ``jump_phase`` | pytempo ``phase_offset_turns`` | **Ruled out** — exact match |
| **D Step 4** | Taylor vs tempo2 ``phase2+phase3`` | ``/tmp/wsrt_taylor_spin_probe.py`` | **Ruled out** — 0.02 ns fractional |
| **D Step 5** | Per-TOA term diff | ``/tmp/wsrt_term_diff_probe.py`` | **Done** — ~330 ns ``bbat`` gap |
| **D Step 6** | ``model_mjd`` vs ``pet``/``torb`` | ``/tmp/wsrt_model_pet_torb_probe.py`` | **Done** |
| **D Step 7** | ``dt_sec`` precision + ``deltaT(pt)`` counterfactual | ``/tmp/wsrt_dt_spin_counterfactual_probe.py`` | **Done** — float64 inputs; swap worsens |
| **Next** | ``model_mjd`` vs tempo2 ``calculate_bclt`` epoch | read-only trace | **Open** |

Docs: [`TEMPO2_PARITY.md`](../TEMPO2_PARITY.md), [`TEMPO2_NATIVE_CLOCK_STATUS.md`](../TEMPO2_NATIVE_CLOCK_STATUS.md).

## Current parity work (2026-07-06)

| Phase | Fix | Tests | Status |
|-------|-----|-------|--------|
| **Native chain** | JAX `tempo2_native/` module | `tests/test_tempo2_native_*.py` | **In progress** — strict formBats path wired |
| **Granular closure** | pytempo delay-chain diagnostics | `tests/test_tempo2_native_formbats_closure.py` | **Added** — component ranking |
| **Oracle** | pytempo Tier-1 via `tempo2_pytempo_oracle.py` | dev_oracle gates | **Added** |

libstempo remains the **acceptance residual** oracle; pytempo provides **per-TOA term** gates
(`bbat_mjd`, `torb_sec`, `roemer_sec`, `pulse_number`). See `TEMPO2_PARITY.md` §0 cheat sheet.

## Delete checklist

When JUG no longer needs tempo2 cross-checks:

1. Remove this file and oracle modules:
   - `jug/testing/sandbox_tempo2.py`
   - `jug/testing/tempo2_reference.py`
   - `jug/testing/tempo2_diagnostics.py` (or strip oracle backends, keep JUG-only helpers)
   - `jug/testing/tempo2_track2_oracle.py`
   - `jug/testing/tempo2_pytempo_oracle.py`
   - `jug/testing/tempo2_native_probes.py`
2. Remove pytest modules matching:
   - `tests/test_dev_oracle_*.py`
   - `tests/test_tempo2_*.py`
3. Drop `dev_oracle` / `tempo2` markers from `pyproject.toml` if unused.
4. Replace acceptance with native golden vectors or an external harness repo.

## Grep targets

```bash
rg 'dev_oracle|sandbox_tempo2|tempo2_reference|tempo2_track2_oracle|importorskip\("libstempo"\)' jug tests
pytest -m dev_oracle   # run only oracle-backed dev tests
pytest -m 'not dev_oracle'  # JUG-only CI path
```

## Useful oracle commands

```bash
cd ref-packages/jug

# Phase C — TZR
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_tzr_parity.py -q

# Phase D — TRACK −2 pnNew
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_track2_pnnew.py -q

# wsrt167 acceptance (strict gate; still failing)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```
