# Development oracle harness (delete when JUG is standalone)

JUG is **standalone**: no runtime dependency on libstempo, tempo2, or pytempo.
During parity development we keep optional **external oracles** here and in
marked pytest modules.

## Hybrid Tempo2 native parity (2026-07-07)

Production tempo2-native fitting uses a **host-frozen** path by default
(``JUG_TEMPO2_NATIVE_GRAPH_MODE=staged_bclt``):
`term_diagnostics['tempo2_obs_state']` geometry + one-time JAX clock + a slim
differentiable JAX tail (BCLT → formBats → spin). The slow unified in-graph
model (`compute_tempo2_toa_model_jax`) is **opt-in only** via
``JUG_TEMPO2_NATIVE_GRAPH_MODE=full``.

JAX BCLT uses a **fixed-length** `lax.scan` (default 12 iterations,
`JUG_TEMPO2_BCLT_FIXED_ITER` override) so reverse-mode AD (NUTS/HMC) works.
Host NumPy BCLT keeps dynamic convergence; convergence flags are diagnostics only
on the JAX path.

**Do not** use legacy top-level `jug["ssb_obs_pos_ls"]`, `jug["obs_sun_pos_ls"]`,
or `jug["obs_planet_pos_ls"]` for native parity probes — use
`term_diagnostics['tempo2_obs_state']` via `host_frozen_vectors_from_tempo2_obs_state`.

### wsrt167 hybrid gates (2026-07-08)

| Term | Frozen staging | NumPy reference | Full in-graph | Notes |
|------|----------------|-----------------|---------------|-------|
| `correction_tt_sec` | yes | yes | opt-in | JAX `clock_jax` once; not host NumPy |
| `correction_tt_tb_sec` | yes | yes | opt-in | Host diagnostics |
| `roemer_sec` | yes | yes | opt-in | |
| `tdis1_sec` / `tdis2_sec` | yes | yes | opt-in | |
| `dt_ssb_sec` | yes | yes | opt-in | |
| `bat_corr_days` | yes | yes | opt-in | Sub-1 ns after TCB implicit tropo + planet Shapiro fixes |
| `bbat_mjd` | **two-part daysec in JAX tail** | opt-in | Host Taylor spin ~1.4 ns wsrt167; native ``phase5@bbat`` for fit Jacobian |
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
JUG_TEMPO2_NATIVE_GRAPH_MODE=full JAX_ENABLE_X64=1 PYTHONPATH=.:tests \
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

## Current parity work (2026-07-08)

This file is the oracle-harness guide. The authoritative theory/policy document is
[`PARITY_THEORY.md`](../PARITY_THEORY.md); the authoritative status and roadmap is
[`PARITY_ROADMAP.md`](../PARITY_ROADMAP.md).

| Workstream | Tests / oracle | Status |
|------------|----------------|--------|
| **Autodiff oracle breadth** | libstempo design-matrix columns beyond F0; IPTA TRACK −2; `full` mode | **Open** — production path wired; wsrt167 F0 green |
| **NG5 TDB** | pytempo `deltaT`; F0 design-matrix column | **Open** — ~5.3 µs spin-epoch / TDB-TCB map |
| **EPTA J0613 full** | `tests/test_tempo2_ipta_dr2_j0613_parity.py` | **Done** — **1.22 ns** RMS vs live libstempo (2026-07-08 re-baseline) |
| **wsrt167 host parity** | `tests/test_dev_oracle_wsrt167_parity.py` | **Done** — ~1.4 ns RMS after tropo-in-dt + longdouble wrap |
| **TRACK −2 pnNew** | `tests/test_tempo2_track2_pnnew.py` | **Done** |
| **TZR** | `tests/test_tempo2_tzr_parity.py` | **Done** — J0030 ~4.7 ns RMS |
| **`-addsat` SAT resync** | `epta_j0613_addsat_min` / dt-chain diag | **Done** — was ~±1 s; residual scatter **1.43 ns** mini / **2.33 ns** max addsat (2026-07-08) |
| **Stale dev-oracle assertions** | `pytest -m dev_oracle` | **Open** — audit/retire assertion debt |

## Native delay-chain notes

Gate policy remains **1 ns RMS** for formBats component comparisons vs pytempo
`toa_diagnostics()`, but not every epoch scalar is a physics gate:

| Quantity | Current interpretation |
|----------|------------------------|
| `correction_tt_sec`, `correction_tt_tb_sec` | Clock chain is closed to <1 ns on wsrt167-class probes |
| `bat_corr_days` | Primary delay-physics gate (~1.1 ns on wsrt167 strict formBats path) |
| `bat_mjd`, `bbat_mjd` | MJD assembly recipe mismatch can show ~304 ns even when delay physics is ~1 ns |
| `torb_sec` | Use pytempo `torb_sec` or `prebinary - total`; libstempo `binarydelay` is stale on fresh construct |

See `PARITY_ROADMAP.md` § formBats MJD assembly before
treating epoch-scalar gaps as delay-physics bugs.

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

Exit criteria for dev-oracle promotion: all required native component gates are within
their stated envelopes, stale assertion-only failures have been retired, and the
canonical tangent is validated against libstempo perturbation oracles.


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

# wsrt167 acceptance (strict gate)
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```
