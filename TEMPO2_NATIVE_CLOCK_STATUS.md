# Tempo2 native clock / spin parity — not done

**Status:** work in progress. **Not at parity** with tempo2/libstempo under the
project’s strict ns-level gate.

Policy and architecture: [`TEMPO2_COMPATIBILITY.md`](TEMPO2_COMPATIBILITY.md).
Broader parity tracker: [`TEMPO2_PARITY.md`](TEMPO2_PARITY.md).

## What landed (partial)

Three tempo2-native pieces were started for `compatibility="tempo2"`:

1. **IFTE `correctionTT_TB`** — `jug/utils/ifteph.py` + `TIMEEPH_short.te405`
2. **`formBats` clock split** — `jug/residuals/tempo2_clock.py` (`sat`, TT, TT_TB,
   `bat`, `bbat`, `model_mjd`)
3. **Native spin / TRACK −2 (WIP, disabled)** — `phase5` +
   `track_minus2_frac_phase` in `jug/residuals/tempo2_spin.py`

Production residuals today use **formBats-corrected `model_mjd`** with
**emission-time Taylor spin** and legacy TRACK −2 wrapping. That is **not** the
full tempo2 `formResiduals.C` path at native `bbat`.

## Measured gap (2026-07-05)

| Fixture | Before IFTE/formBats | Current | Strict gate |
|---------|---------------------|---------|-------------|
| wsrt167 (TCB, TRACK −2) | ~263 ns RMS | **~16 ns RMS** | **< 5 ns RMS** |
| epta_j0030_isolated | ~255 ns RMS | **~16 ns RMS** | **< 5 ns RMS** |

Improvement is real (~250 ns → ~16 ns) but **does not satisfy** the hard parity
gate. Max delta on wsrt167 is ~110 ns; p99 ~38 ns.

## Parity gates

During bring-up, pytest thresholds were **temporarily relaxed** (20 ns RMS /
120 ns max) so partial progress could be exercised. Those gates have been
**restored** to the strict values in `tests/test_tempo2_residual_parity.py`
(5 / 25 / 10 ns). Expect failures until parity work is complete.

## Disabled path (kept on purpose)

`USE_NATIVE_BBAT_PHASE5 = False` in `simple_calculator.py` keeps the
`phase5 + track_minus2_frac_phase` path **compiled and callable** but **off** in
production. Enabling it without further fixes produces ~710 ms residuals on
wsrt167 (broken pulse-number coupling), not ns-level error.

Do **not** delete that path; it is the intended next development surface.

## What remains for true ns-level parity

### A. Longdouble time pipeline

Keep `sat`, TT, TT_TB, `bat`, `bbat`, `model_mjd`, and spin `dt` in
`np.longdouble` through clock + phase evaluation; stop repeated float64
round-trips in `tempo2_clock.py` and phase spin. Isolate float64 only at
Astropy/JAX boundaries.

### B. Working native `phase5` at `bbat`

- Single source of truth: `tempo2_clock.compute_formbats_arrival`
- One canonical `torb` sign convention (tempo2 `obsn[i].torb`)
- Exact `formResiduals.C` order: `phas1`, `nphase`, `pnNew`, `pnadd`, `addPhase`,
  `-addsat`
- Per-TOA oracle tests on intermediate terms, not only final RMS

### C. tt2tb geometry

Evaluate Earth–site dot product with the same ephemeris epoch tempo2 uses
(possibly reuse delay-provider SSB velocity instead of a separate Astropy pass).

## Verification

```bash
cd ref-packages/jug
PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_tempo2_residual_parity.py -q

PYTHONPATH=.:tests TEMPO2=/opt/software/tempo2/T2runtime \
  /opt/venvs/pta/bin/python -m pytest tests/test_dev_oracle_wsrt167_parity.py -m dev_oracle -q
```

Strict gates should fail until A/B/C above are closed.
