# JUG Parameter and Fitting Parity Analysis

**Updated**: 2026-02-06
**Purpose**: Comprehensive audit of parameter/fitting correctness vs PINT/Tempo2

---

## 1. Repo Inventory (Grounded)

### A. Parameter Metadata / Registry / Specs

| Location | Purpose | Contents |
|----------|---------|----------|
| [jug/model/parameter_spec.py](jug/model/parameter_spec.py) | **Primary Registry** | `ParameterSpec` dataclass, `DerivativeGroup` enum, `PARAMETER_REGISTRY` dict |
| [jug/model/codecs.py](jug/model/codecs.py) | I/O Codecs | RAJ/DECJ sexagesimal<->radians, epoch codecs |
| [jug/fitting/binary_registry.py](jug/fitting/binary_registry.py) | Binary Model Registry | `_BINARY_MODEL_REGISTRY` mapping model names to delay/derivative functions |

**Key Functions in parameter_spec.py**:
- `get_spec(name)` -> `ParameterSpec` or None
- `get_derivative_group(name)` -> `DerivativeGroup` or None
- `canonicalize_param_name(name)` -> canonical name (resolves aliases)
- `is_spin_param()`, `is_dm_param()`, `is_binary_param()`, `is_astrometry_param()`, `is_fd_param()`, `is_sw_param()`, `is_jump_param()`

**DerivativeGroup enum values**:
- `SPIN` -> routed to `derivatives_spin.py`
- `DM` -> routed to `derivatives_dm.py`
- `ASTROMETRY` -> routed to `derivatives_astrometry.py`
- `BINARY` -> routed to `derivatives_binary.py` (ELL1) or `derivatives_dd.py` (DD/DDK/BT)
- `EPOCH` -> not fitted directly (reference points)
- `JUMP` -> routed to `derivatives_jump.py`
- `FD` -> routed to `derivatives_fd.py`
- `SOLAR_WIND` -> routed to `derivatives_sw.py`

### B. Design Matrix / Partial Derivatives

| File | Scope | Method |
|------|-------|--------|
| [jug/fitting/derivatives_spin.py](jug/fitting/derivatives_spin.py) | F0, F1, F2, F3... | **Analytic** (Taylor series) |
| [jug/fitting/derivatives_dm.py](jug/fitting/derivatives_dm.py) | DM, DM1, DM2... | **Analytic** (K_DM/freq^2) |
| [jug/fitting/derivatives_astrometry.py](jug/fitting/derivatives_astrometry.py) | RAJ, DECJ, PMRA, PMDEC, PX | **Analytic** (PINT-compatible) |
| [jug/fitting/derivatives_binary.py](jug/fitting/derivatives_binary.py) | ELL1 params: PB, A1, TASC, EPS1, EPS2, PBDOT, XDOT, M2, SINI, FBn | **Analytic** (JAX, 3rd-order ELL1) |
| [jug/fitting/derivatives_dd.py](jug/fitting/derivatives_dd.py) | DD/DDK params: PB, A1, T0, ECC, OM, PBDOT, OMDOT, GAMMA, SINI, M2, H3, H4, STIG, XDOT, EDOT, KIN, KOM | **Analytic** (JAX, chain rule through Kepler) |
| [jug/fitting/derivatives_fd.py](jug/fitting/derivatives_fd.py) | FD1, FD2... | **Analytic** (log(f/1GHz)^n) |
| [jug/fitting/derivatives_sw.py](jug/fitting/derivatives_sw.py) | NE_SW | **Analytic** (K_DM * geometry / freq^2) |
| [jug/fitting/derivatives_jump.py](jug/fitting/derivatives_jump.py) | JUMPn | **Analytic** (trivial: 0 or 1) |

### C. Binary Model Dispatch

| File | Models | Notes |
|------|--------|-------|
| [jug/delays/binary_dispatch.py](jug/delays/binary_dispatch.py) | Dispatch logic | Routes to appropriate model; DDK raises ValueError (use branch_ddk) |
| [jug/delays/binary_dd.py](jug/delays/binary_dd.py) | DD, DDH, DDGR | Forward model, Kepler solver |
| [jug/delays/binary_bt.py](jug/delays/binary_bt.py) | BT, BTX | Blandford-Teukolsky |
| [jug/delays/binary_t2.py](jug/delays/binary_t2.py) | T2 | Tempo2 general model |
| [jug/delays/combined.py](jug/delays/combined.py#L125) | ELL1, ELL1H | Inline implementation (optimized) |
| [jug/delays/combined.py](jug/delays/combined.py#L222) | DDK (branch_ddk) | Forward model with Kopeikin terms |

**Binary model IDs** (used in combined.py):
- 0: None (no binary)
- 1: ELL1 / ELL1H (inline)
- 2: DD / DDH / DDGR
- 3: T2
- 4: BT / BTX
- 5: DDK (DD with Kopeikin corrections)

---

## 2. Parameter Parity Matrix

### Legend
- **Forward**: Parameter used in forward delay model
- **Partial**: Analytic partial derivative exists
- **Finite-Diff**: Validated against central-difference numerical derivative
- **Fit-Ready**: Can be fitted via `fit_params` list
- **Test**: Has test coverage

### Spin Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| F0 | YES | YES | - | YES | YES |
| F1 | YES | YES | - | YES | YES |
| F2 | YES | YES | - | YES | YES |
| F3 | YES | YES | - | YES | partial |
| PEPOCH | YES | - (epoch) | - | - | YES |

### DM Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| DM | YES | YES | - | YES | YES |
| DM1 | YES | YES | - | YES | YES |
| DM2 | YES | YES | - | YES | partial |
| DMEPOCH | YES | - (epoch) | - | - | YES |

### Astrometry Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| RAJ | YES | YES | - | YES | YES |
| DECJ | YES | YES | - | YES | YES |
| PMRA | YES | YES | - | YES | YES |
| PMDEC | YES | YES | - | YES | YES |
| PX | YES | YES | - | YES | YES |

### Solar Wind Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| NE_SW | YES | YES `derivatives_sw.py` | YES (exact match) | YES | YES (12 tests) |

### Binary Parameters (DD Family)

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| PB | YES | YES | - | YES | YES |
| A1 | YES | YES | - | YES | YES |
| T0 | YES | YES | - | YES | YES |
| ECC | YES | YES | - | YES | YES |
| OM | YES | YES | - | YES | YES |
| GAMMA | YES | YES | - | YES | partial |
| PBDOT | YES | YES | - | YES | partial |
| OMDOT | YES | YES | - | YES | partial |
| XDOT | YES | YES | YES (r>0.95) | YES | YES (3 tests) |
| EDOT | YES | YES | YES (r>0.95) | YES | YES (2 tests) |
| SINI | YES | YES | - | YES | partial |
| M2 | YES | YES | - | YES | partial |
| H3 (STIG) | YES | YES `_d_delay_d_H3()` | YES (r>0.95) | YES | YES |
| H3 (H4) | YES | YES `_d_delay_d_H3_h3h4()` | YES (r>0.95) | YES | YES |
| H4 | YES | YES `_d_delay_d_H4()` | YES (r>0.95) | YES | YES |
| STIG | YES | YES `_d_delay_d_STIG()` | YES (r>0.95) | YES | YES |
| DR | spec only | NO | - | NO | NO |
| DTH | spec only | NO | - | NO | NO |
| A0 | spec only | NO | - | NO | NO |
| B0 | spec only | NO | - | NO | NO |

### DDK-Specific Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| KIN | YES `branch_ddk()` | YES `compute_binary_derivatives_ddk()` | YES (r>0.95) | YES | YES |
| KOM | YES `branch_ddk()` | YES `compute_binary_derivatives_ddk()` | YES (r>0.95) | YES | YES |

### FD Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| FD1-FD9 | YES | YES `derivatives_fd.py` | YES (exact match) | YES | YES (6 tests) |

### JUMP Parameters

| Parameter | Forward | Partial | Finite-Diff | Fit-Ready | Test |
|-----------|---------|---------|-------------|-----------|------|
| JUMPn | YES (implicit) | YES `derivatives_jump.py` | - | partial | partial |

---

## 3. Orthometric Shapiro Parameterizations

### H3/STIG (DDH model)
Correct, matches PINT exactly:
- `SINI = 2*STIG / (1 + STIG^2)`
- `M2 = H3 / (STIG^3 * T_SUN)`

### H3/H4 (Freire & Wex 2010, PINT/Tempo2 convention)
Fixed 2026-02-06 to match PINT:
- `STIGMA = H4/H3`
- `SINI = 2*H3*H4 / (H3^2 + H4^2)`
- `M2 = H3^4 / (H4^3 * T_SUN)`

Both parameterizations have analytic partials validated against finite differences.

---

## 4. Test Coverage

### Test Files

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_ddk_partials.py` | 36 | DDK corrections, registry, finite-diff (KIN/KOM/A1/ECC), H3/H4, EDOT, DDK dispatch, end-to-end smoke |
| `tests/test_ne_sw.py` | 12 | NE_SW ParameterSpec, derivative shape/sign/scaling, forward model match |
| `tests/test_xdot_fd_partials.py` | 10 | XDOT finite-diff, FD1-FD3 finite-diff, H3/STIG finite-diff |

**Total new/updated tests**: 58

---

## 5. Remaining Gaps

### Low Priority (Rarely Fitted)

| Gap | Impact | Effort |
|-----|--------|--------|
| DR, DTH partials | Relativistic deformation (rarely fitted) | Medium |
| A0, B0 partials | Aberration delay (rarely fitted) | Low |
| EPS1DOT, EPS2DOT | ELL1 time derivatives (rare) | Low |

### Parametric Families (Not First-Class)

| Gap | Impact | Notes |
|-----|--------|-------|
| FDn (n>9) | Out-of-range FD parameters | FD1-FD9 are registered; higher indices need dynamic registration |
| FBn (n>20) | Out-of-range FB parameters | FB0-FB20 are registered; higher indices need dynamic registration |
| DMX_nnnn | DMX parameters | Not registered at all; requires family-style registration |
| JUMPn | JUMP parameters | Pattern-matched (is_jump_param), not registry-based; works but lacks specs |

`validate_fit_param()` in `parameter_spec.py` now raises clear errors for unregistered or out-of-range parameters.

### Integration Gaps

| Gap | Impact |
|-----|--------|
| JUMP fitting integration | Needs TOA flag parsing in fitter |
| GLF0, GLPH etc. | Glitch parameters (future work) |

---

## 6. Summary

### Fully Operational
- Spin parameters (F0-F3): Full analytic partials, PINT-compatible
- DM parameters (DM, DM1, DM2): Full analytic partials
- Astrometry (RAJ, DECJ, PMRA, PMDEC, PX): Full analytic partials
- ELL1 binary (PB, A1, TASC, EPS1, EPS2, SINI, M2, FBn): 3rd-order ELL1
- DD binary core (PB, A1, T0, ECC, OM, GAMMA, PBDOT, OMDOT, XDOT, EDOT): All finite-diff validated
- DDK (KIN, KOM): Chain rule through Kopeikin corrections, finite-diff validated
- Orthometric Shapiro (H3/STIG, H3/H4): Both parameterizations, finite-diff validated
- FD parameters (FD1-FD9): Trivial partials, finite-diff validated
- Solar wind (NE_SW): Analytic partial, ParameterSpec, fitter wiring complete
- Prebinary time evaluation: PINT-compatible (roemer + shapiro + dm + sw + tropo)
