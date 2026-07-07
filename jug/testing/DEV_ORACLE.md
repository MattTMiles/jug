# Development oracle harness (delete when JUG is standalone)

JUG is **standalone**: no runtime dependency on libstempo, tempo2, or pytempo.
During parity development we keep optional **external oracles** here and in
marked pytest modules.

## Hybrid Tempo2 native parity (2026-07-07)

Production tempo2-native fitting uses a **host-frozen** path by default:
`term_diagnostics['tempo2_obs_state']` geometry + one-time JAX clock + a slim
differentiable JAX tail (BCLT → formBats → spin). The slow unified in-graph
model (`compute_tempo2_toa_model_jax`) is **opt-in only** via
`USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH` or `JUG_TEMPO2_NATIVE_FULL_INGRAPH=1`.

**Do not** use legacy top-level `jug["ssb_obs_pos_ls"]`, `jug["obs_sun_pos_ls"]`,
or `jug["obs_planet_pos_ls"]` for native parity probes — use
`term_diagnostics['tempo2_obs_state']` via `host_frozen_vectors_from_tempo2_obs_state`.

### wsrt167 hybrid gates (< 1 ns RMS vs pytempo)

| Term | Frozen staging | NumPy reference | Full in-graph | Notes |
|------|----------------|-----------------|---------------|-------|
| `correction_tt_sec` | yes | yes | opt-in | JAX `clock_jax` once; not host NumPy |
| `correction_tt_tb_sec` | yes | yes | opt-in | Host diagnostics |
| `roemer_sec` | yes | yes | opt-in | |
| `tdis1_sec` / `tdis2_sec` | yes | yes | opt-in | |
| `dt_ssb_sec` | yes | yes | opt-in | |
| `bat_corr_days` | yes | yes | opt-in | Sub-1 ns after TCB implicit tropo + planet Shapiro fixes |
| `bbat_mjd` | **deferred** | **deferred** | opt-in | ~304 ns long-double assembly debt — see `TEMPO2_PARITY.md` |
| DILATEFREQ / DMX / ecliptic fixtures | **deferred** | **deferred** | opt-in | After wsrt167 green |

Fast hybrid probes (~seconds compile; no full ephemeris/clock JIT):

```bash
cd /workspaces/metapulsar/ref-packages/jug
JAX_ENABLE_X64=1 PYTHONPATH=.:tests python3 -m pytest \
  tests/test_tempo2_native_numpy_reference_parity.py \
  tests/test_tempo2_native_staging_host_frozen.py \
  tests/test_tempo2_native_residual_delta_jax.py -q
```

Slow full-in-graph oracle (minutes compile; manual only):

```bash
cd /workspaces/metapulsar/ref-packages/jug
JUG_TEMPO2_NATIVE_FULL_INGRAPH=1 JAX_ENABLE_X64=1 PYTHONPATH=.:tests \
  python3 -m pytest tests/test_tempo2_native_jax_no_host_roundtrip.py -q
```

NumPy reference path is dev-only: set `JUG_DEV_NUMPY_TEMPO2_CHAIN=1` for
`chain_numpy` tests.

### MetaPulsar / notebook integration

`export_jax_timing_state` (MetaPulsar `JugEngine`) needs a `TimingSession` cache that
includes `term_diagnostics['tempo2_obs_state']`. After JUG upgrades, call
`compute_residuals(force_recompute=True)` on tempo2 sessions before binding NTM autodiff.
IPTA example: `examples/notebooks-dev/nlt_ipta_dr2_compare_jug.ipynb` uses
`nlt_ipta_dr2_compare_jug_lib.prime_jug_tempo2_native_sessions()`.

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

## wsrt167 native delay-chain parity (2026-07-06)

Gate policy: **1 ns RMS** on every formBats slot vs pytempo ``toa_diagnostics``.

| Term | RMS (ns) | Gate | Notes |
|------|----------|------|-------|
| `tt` | ~270 | 1 | ``getCorrectionTT`` / Astropy TT path |
| `tt_tb` | ~44 000 | 1 | Fixed SPK vel (km/day→km/s) + ``formbats_correction_tt`` |
| `roemer` (host export) | ~0.8 | 1 | Delay provider at ``model_mjd`` |
| `roemer` (native BCLT) | ~4800 | 1 | IFTE fixed geometry vs pytempo BCLT epoch |
| `tdis1` | ~42 | 1 | DM Taylor only; DMX/shapelets open |
| `tdis2` | ~0.1 | 1 | Closed |
| `tropo` | ~26 | 1 | Open |
| `shap` | ~24 | 1 | Open |

Temp probes (outside repo):

```bash
PYTHONPATH=/workspaces/metapulsar/ref-packages/jug:/workspaces/metapulsar/ref-packages/jug/tests \
  python /tmp/jug_geometry_vector_probe.py
PYTHONPATH=/workspaces/metapulsar/ref-packages/jug:/workspaces/metapulsar/ref-packages/jug/tests \
  python /tmp/jug_clock_tt_probe.py
PYTHONPATH=/workspaces/metapulsar/ref-packages/jug:/workspaces/metapulsar/ref-packages/jug/tests \
  python /tmp/jug_wsrt167_parity_probe.py
# writes /tmp/jug_wsrt167_parity_probe.txt
```

Fast native gate path (skips ``@pytest.mark.slow`` modules — ~45 s vs ~2.5 min full):

```bash
cd ref-packages/jug
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_native_*.py -m 'dev_oracle and not slow' --no-cov -q
```

Slow modules (tagged ``slow``): ``residual_delta_jax``, ``geometry_parity``,
``clock_jax``, ``jax_no_host_roundtrip``, ``numpy_reference_parity``,
``spin_phase5``, ``roemer_probe``. Primary delay gates stay on the fast path:
``formbats_closure``, ``bclt_terms``, ``bbat_parity``, ``batcorr_parity``,
``torb_closure``.

Full suite (sprint sign-off only):

```bash
cd ref-packages/jug
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_tempo2_native_*.py -m dev_oracle --no-cov -q
```

Exit criteria: all ``test_tempo2_native_*.py`` gates at **1 ns**; production path uses
``compute_tempo2_toa_model_jax`` with no mid-chain ``device_get``.


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
