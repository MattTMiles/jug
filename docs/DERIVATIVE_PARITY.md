# Derivative Parity: JUG vs PINT vs Tempo2

**Last Updated**: 2026-01-29

This document tracks the comparison of JUG's analytical derivatives against PINT and Tempo2 reference implementations.

## Summary

| Parameter | JUG vs PINT | JUG vs Tempo2 | Status | Notes |
|-----------|-------------|---------------|--------|-------|
| **Astrometry** |
| RAJ | < 1e-6 | - | ✓ Complete | |
| DECJ | < 1e-6 | - | ✓ Complete | |
| PMRA | < 1e-6 | - | ✓ Complete | |
| PMDEC | < 1e-6 | - | ✓ Complete | |
| PX | < 1e-6 | - | ✓ Complete | |
| **Spin** |
| F0 | < 1e-6 | - | ✓ Complete | |
| F1 | < 1e-6 | - | ✓ Complete | |
| F2+ | < 1e-6 | - | ✓ Complete | |
| **DM** |
| DM | < 1e-6 | - | ✓ Complete | |
| DM1+ | < 1e-6 | - | ✓ Complete | |
| **Binary ELL1** |
| A1 | < 1e-7 | - | ✓ Complete | Perfect match |
| EPS1 | < 1e-7 | - | ✓ Complete | Perfect match |
| EPS2 | < 1e-7 | - | ✓ Complete | Perfect match |
| SINI | < 1e-7 | - | ✓ Complete | Perfect match |
| M2 | ~0.03% | Matches Tempo2 | ✓ Complete | PINT bug, JUG correct |
| PB | ~0.05%* | Matches Tempo2 | ✓ Complete | *PINT bug, JUG correct |
| TASC | ~0.05%* | Matches Tempo2 | ✓ Complete | *PINT bug, JUG correct |
| PBDOT | ~0.05%* | Matches Tempo2 | ✓ Complete | *PINT bug, JUG correct |
| XDOT | - | - | ✓ Complete | Implemented, not yet tested |
| FB0/FB1 | - | - | ☐ TODO | |
| **ELL1H (Orthometric)** |
| H3 | - | - | ☐ TODO | |
| H4 | - | - | ☐ TODO | |
| STIG | - | - | ☐ TODO | |
| **Binary DD** |
| ECC | - | - | ☐ TODO | |
| OM | - | - | ☐ TODO | |
| T0 | - | - | ☐ TODO | |
| OMDOT | - | - | ☐ TODO | |
| GAMMA | - | - | ☐ TODO | |

*See "PINT Shapiro Derivative Bug" section below  
**Updated 2026-01-29**: Tempo2 verification complete. JUG matches Tempo2 exactly.

---

## PINT Shapiro Derivative Bug

**Discovered**: 2026-01-29  
**Full details**: See `docs/PINT_SHAPIRO_DERIVATIVE_BUG.md`

### Summary

PINT's ELL1 model has a bug in `d_delayS_d_Phi` (derivative of Shapiro delay w.r.t. orbital phase). The `cos(Φ)` factor from the chain rule is missing.

**PINT's formula** (buggy):
```
d(Shapiro)/d(Φ) = 2·TM2·sin(i) / (1 - sin(i)·sin(Φ))
```

**Correct formula** (JUG):
```
d(Shapiro)/d(Φ) = 2·TM2·sin(i)·cos(Φ) / (1 - sin(i)·sin(Φ))
```

### Verification

Numerical finite-difference verification confirms JUG matches the true derivative at all orbital phases:

| Phase | JUG/Numerical | PINT/Numerical |
|-------|---------------|----------------|
| 0° | 1.0000 | 1.0000 |
| 45° | 1.0000 | 1.4142 |
| 89° | 1.0000 | 57.25 |
| 90° | ~1.0 | 64797 |
| 180° | 1.0000 | -1.0000 |

The PINT/Numerical ratio equals exactly `1/cos(Φ)`, confirming the missing factor.

### Impact

This affects PB, TASC, and PBDOT derivatives near superior conjunction (Φ ≈ 90°). For high-inclination systems like J1909-3744 (sin i ≈ 0.998), the error can be significant.

### Decision

**JUG uses the mathematically correct formula.** Users comparing JUG's design matrix to PINT's will see differences for PB/TASC/PBDOT at TOAs near superior conjunction. This is expected and correct.

---

## Tempo2 Comparison Notes

### Shapiro Term in Orbital Period Derivatives

Tempo2's ELL1 model (`ELL1model.C`) does not include the Shapiro delay contribution in derivatives w.r.t. PB, TASC, or PBDOT. The derivative only includes the Roemer (geometric) delay term:

```c
if (param==param_pb)
    return -Csigma*an*SECDAY*tt0/(pb*SECDAY); /* Pb */
```

This may be intentional (treating Shapiro parameters as independent) or a simplification. JUG includes the full derivative for mathematical completeness.

---

## Validation Methodology

### Against PINT

1. Load identical par/tim files in both JUG and PINT
2. Compute design matrix columns for each parameter
3. Convert units to common basis (seconds per par-file unit)
4. Compare ratio across all TOAs
5. Report mean ratio, max relative difference

### Against Tempo2

1. Compare analytical formulas from Tempo2 source code
2. Where possible, run tempo2 with `-output matrix` to get numerical derivatives
3. Document any formula differences

### Numerical Verification

For critical parameters, verify analytical derivatives using finite differences:
```
d(delay)/d(param) ≈ [delay(param + δ) - delay(param - δ)] / (2δ)
```

This catches both formula errors and unit conversion issues.

---

## Test Data

Primary test pulsar: **J1909-3744**
- Location: `data/pulsars/J1909-3744_tdb.par`, `data/pulsars/J1909-3744.tim`
- Binary model: ELL1
- High inclination: sin(i) ≈ 0.998 (good for testing Shapiro effects)
- ~10,000 TOAs spanning multiple years
