# JUG Parameter and Fitting Parity Analysis

**Generated**: 2026-02-04
**Purpose**: Comprehensive audit of parameter/fitting correctness vs PINT/Tempo2

---

## 1. Repo Inventory (Grounded)

### A. Parameter Metadata / Registry / Specs

| Location | Purpose | Contents |
|----------|---------|----------|
| [jug/model/parameter_spec.py](jug/model/parameter_spec.py) | **Primary Registry** | `ParameterSpec` dataclass, `DerivativeGroup` enum, `PARAMETER_REGISTRY` dict |
| [jug/model/codecs.py](jug/model/codecs.py) | I/O Codecs | RAJ/DECJ sexagesimal↔radians, epoch codecs |
| [jug/fitting/binary_registry.py](jug/fitting/binary_registry.py) | Binary Model Registry | `_BINARY_MODEL_REGISTRY` mapping model names to delay/derivative functions |

**Key Functions in parameter_spec.py**:
- `get_spec(name)` → `ParameterSpec` or None
- `get_derivative_group(name)` → `DerivativeGroup` or None
- `canonicalize_param_name(name)` → canonical name (resolves aliases)
- `is_spin_param()`, `is_dm_param()`, `is_binary_param()`, `is_astrometry_param()`, `is_fd_param()`, `is_jump_param()`

**DerivativeGroup enum values**:
- `SPIN` → routed to `derivatives_spin.py`
- `DM` → routed to `derivatives_dm.py`
- `ASTROMETRY` → routed to `derivatives_astrometry.py`
- `BINARY` → routed to `derivatives_binary.py` (ELL1) or `derivatives_dd.py` (DD/BT)
- `EPOCH` → not fitted directly (reference points)
- `JUMP` → routed to `derivatives_jump.py`
- `FD` → routed to `derivatives_fd.py`

### B. Design Matrix / Partial Derivatives

| File | Scope | Method |
|------|-------|--------|
| [jug/fitting/derivatives_spin.py](jug/fitting/derivatives_spin.py) | F0, F1, F2, F3... | **Analytic** (Taylor series) |
| [jug/fitting/derivatives_dm.py](jug/fitting/derivatives_dm.py) | DM, DM1, DM2... | **Analytic** (K_DM/freq²) |
| [jug/fitting/derivatives_astrometry.py](jug/fitting/derivatives_astrometry.py) | RAJ, DECJ, PMRA, PMDEC, PX | **Analytic** (PINT-compatible) |
| [jug/fitting/derivatives_binary.py](jug/fitting/derivatives_binary.py) | ELL1 params: PB, A1, TASC, EPS1, EPS2, PBDOT, XDOT, M2, SINI, FBn | **Analytic** (JAX, 3rd-order ELL1) |
| [jug/fitting/derivatives_dd.py](jug/fitting/derivatives_dd.py) | DD params: PB, A1, T0, ECC, OM, PBDOT, OMDOT, GAMMA, SINI, M2, H3, STIG | **Analytic** (JAX, chain rule through Kepler) |
| [jug/fitting/derivatives_fd.py](jug/fitting/derivatives_fd.py) | FD1, FD2... | **Analytic** (log(f/1GHz)^n) |
| [jug/fitting/derivatives_jump.py](jug/fitting/derivatives_jump.py) | JUMPn | **Analytic** (trivial: 0 or 1) |

**Design matrix assembly** in [jug/fitting/optimized_fitter.py](jug/fitting/optimized_fitter.py#L1507):
```python
# Build design matrix - BATCHED derivative computation
spin_derivs = compute_spin_derivatives(params, toas_mjd, spin_params_list)
dm_derivs = compute_dm_derivatives(params, toas_mjd, freq_mhz, dm_params_list)
binary_derivs = compute_binary_derivatives(params, toas_prebinary_mjd, binary_params_list)
astrometry_derivs = compute_astrometry_derivatives(params, toas_mjd, ssb_obs_pos_ls, astrometry_params_list)
fd_derivs = compute_fd_derivatives(params, freq_mhz, fd_params_list)
```

### C. Delay Stage Computation Order

**Location**: [jug/residuals/simple_calculator.py](jug/residuals/simple_calculator.py) and [jug/delays/combined.py](jug/delays/combined.py)

**Delay evaluation order** (matching PINT):
1. **Roemer delay** (geometric light travel time to SSB)
2. **Shapiro delay** (gravitational delay from Sun + planets)
3. **Tropospheric delay** (if CORRECT_TROPOSPHERE)
4. **DM delay** (dispersion, Taylor series in time)
5. **Solar Wind delay** (NE_SW term)
6. **Binary delay** (evaluated at prebinary time)
7. **FD delay** (frequency-dependent)

**Prebinary time** (PINT-compatible):
```python
# From combined.py line ~95-105
t_prebinary = tdbld - (roemer_shapiro + dm + sw + tropo) / SECS_PER_DAY
```

This matches PINT's `delay_before_binary = roemer + shapiro + DM + SW + tropo`.

### D. Binary Model Dispatch and Implementation

| File | Models | Notes |
|------|--------|-------|
| [jug/delays/binary_dispatch.py](jug/delays/binary_dispatch.py) | Dispatch logic | Routes to appropriate model |
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

**Current DDK Status**:
- Forward model in `combined.py:branch_ddk()` (~lines 222-335)
- Kopeikin (1995) parallax + K96 proper motion corrections: fully implemented
- KIN/KOM analytic partial derivatives: implemented, finite-difference validated
- Fitting: fully wired via binary_registry DDK entry

---

## 2. Parameter Parity Matrix

### Legend
- **Forward**: Parameter used in forward delay model
- **∂/∂p Analytic**: Analytic partial derivative exists
- **∂/∂p Numeric**: Numeric derivative (finite difference)
- **Fit-Ready**: Can be fitted via `fit_params` list
- **Test**: Has test coverage
- ✅ = Yes, ❌ = No, ⚠️ = Partial

### Spin Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| F0 | ✅ | ✅ `derivatives_spin.py:d_phase_d_F()` | ❌ | ✅ | ✅ | PINT-compatible Taylor series |
| F1 | ✅ | ✅ same | ❌ | ✅ | ✅ | |
| F2 | ✅ | ✅ same | ❌ | ✅ | ✅ | Uses longdouble internally |
| F3 | ✅ | ✅ same | ❌ | ✅ | ⚠️ | Sparse data for F3+ |
| PEPOCH | ✅ | ❌ (epoch) | ❌ | ❌ | ✅ | Reference epoch, not fitted |

### DM Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| DM | ✅ | ✅ `derivatives_dm.py:d_delay_d_DM()` | ❌ | ✅ | ✅ | K_DM/freq² |
| DM1 | ✅ | ✅ `d_delay_d_DM1()` | ❌ | ✅ | ✅ | Linear time derivative |
| DM2 | ✅ | ✅ `d_delay_d_DM2()` | ❌ | ✅ | ⚠️ | Quadratic time derivative |
| DMEPOCH | ✅ | ❌ (epoch) | ❌ | ❌ | ✅ | Reference epoch |

### Astrometry Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| RAJ | ✅ | ✅ `derivatives_astrometry.py:d_delay_d_RAJ()` | ❌ | ✅ | ✅ | PINT-compatible |
| DECJ | ✅ | ✅ `d_delay_d_DECJ()` | ❌ | ✅ | ✅ | PINT-compatible |
| PMRA | ✅ | ✅ `d_delay_d_PMRA()` | ❌ | ✅ | ✅ | mas/yr |
| PMDEC | ✅ | ✅ `d_delay_d_PMDEC()` | ❌ | ✅ | ✅ | mas/yr |
| PX | ✅ | ✅ `d_delay_d_PX()` | ❌ | ✅ | ✅ | Parallax in mas |
| POSEPOCH | ✅ | ❌ (epoch) | ❌ | ❌ | ✅ | Reference epoch |

### Binary Parameters (ELL1 Family)

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| PB | ✅ | ✅ `derivatives_binary.py:d_delay_d_PB_ell1()` | ❌ | ✅ | ✅ | |
| A1 | ✅ | ✅ `d_delay_d_A1_ell1()` | ❌ | ✅ | ✅ | |
| TASC | ✅ | ✅ `d_delay_d_TASC_ell1()` | ❌ | ✅ | ✅ | |
| EPS1 | ✅ | ✅ `d_delay_d_EPS1()` | ❌ | ✅ | ✅ | 3rd-order ELL1 |
| EPS2 | ✅ | ✅ `d_delay_d_EPS2()` | ❌ | ✅ | ✅ | 3rd-order ELL1 |
| PBDOT | ✅ | ✅ | ❌ | ✅ | ✅ | |
| XDOT | ✅ | ✅ | ❌ | ✅ | ⚠️ | |
| SINI | ✅ | ✅ `d_delay_d_SINI_ell1()` | ❌ | ✅ | ⚠️ | Shapiro delay |
| M2 | ✅ | ✅ `d_delay_d_M2_ell1()` | ❌ | ✅ | ⚠️ | Shapiro delay |
| FB0-FB20 | ✅ | ✅ `d_Phi_d_FBi()` | ❌ | ✅ | ⚠️ J2241 | Orbital frequency Taylor |
| H3 | ✅ | ✅ | ❌ | ✅ | ⚠️ | ELL1H orthometric |
| H4 | ✅ | ✅ `_d_delay_d_H4()` | ❌ | ✅ | ✅ | H3/H4 orthometric |
| STIG | ✅ | ✅ | ❌ | ✅ | ⚠️ | ELL1H orthometric |

### Binary Parameters (DD Family)

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| PB | ✅ | ✅ `derivatives_dd.py:_d_delay_d_PB()` | ❌ | ✅ | ✅ | Chain rule through M |
| A1 | ✅ | ✅ `_d_delay_d_A1()` | ❌ | ✅ | ✅ | |
| T0 | ✅ | ✅ `_d_delay_d_T0()` | ❌ | ✅ | ✅ | |
| ECC | ✅ | ✅ `_d_delay_d_ECC()` | ❌ | ✅ | ✅ | |
| OM | ✅ | ✅ `_d_delay_d_OM()` | ❌ | ✅ | ✅ | In degrees |
| GAMMA | ✅ | ✅ `_d_delay_d_GAMMA()` | ❌ | ✅ | ⚠️ | Einstein delay |
| PBDOT | ✅ | ✅ `_d_delay_d_PBDOT()` | ❌ | ✅ | ⚠️ | |
| OMDOT | ✅ | ✅ `_d_delay_d_OMDOT()` | ❌ | ✅ | ⚠️ | Periastron advance |
| XDOT | ✅ | ✅ | ❌ | ✅ | ⚠️ | A1DOT alias |
| EDOT | ✅ | ✅ (chain rule through ECC) | ❌ | ✅ | ✅ | Finite-diff validated |
| SINI | ✅ | ✅ `_d_delay_d_SINI()` | ❌ | ✅ | ⚠️ | |
| M2 | ✅ | ✅ `_d_delay_d_M2()` | ❌ | ✅ | ⚠️ | |
| H3 | ✅ | ✅ `_d_delay_d_H3()` | ❌ | ✅ | ⚠️ | DDH orthometric |
| H4 | ✅ | ✅ `_d_delay_d_H4()` | ❌ | ✅ | ✅ | H3/H4 orthometric |
| STIG | ✅ | ✅ `_d_delay_d_STIG()` | ❌ | ✅ | ⚠️ | DDH orthometric |
| DR | ✅ (spec) | ❌ | ❌ | ❌ | ❌ | **NOT IMPLEMENTED** |
| DTH | ✅ (spec) | ❌ | ❌ | ❌ | ❌ | **NOT IMPLEMENTED** |
| A0 | ✅ (spec) | ❌ | ❌ | ❌ | ❌ | **NOT IMPLEMENTED** (aberration) |
| B0 | ✅ (spec) | ❌ | ❌ | ❌ | ❌ | **NOT IMPLEMENTED** (aberration) |

### DDK-Specific Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| KIN | ✅ `combined.py:branch_ddk()` | ✅ `derivatives_dd.py:compute_binary_derivatives_ddk()` | ❌ | ✅ | ✅ | Chain rule through A1_eff/OM_eff/SINI_eff; finite-diff validated |
| KOM | ✅ `combined.py:branch_ddk()` | ✅ `derivatives_dd.py:compute_binary_derivatives_ddk()` | ❌ | ✅ | ✅ | Chain rule through A1_eff/OM_eff; finite-diff validated |

### FD Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| FD1-FD9 | ✅ | ✅ `derivatives_fd.py:compute_fd_derivatives()` | ❌ | ✅ | ⚠️ | log(f/1GHz)^n |

### JUMP Parameters

| Parameter | Forward | ∂/∂p Analytic | ∂/∂p Numeric | Fit-Ready | Test | Notes |
|-----------|---------|---------------|--------------|-----------|------|-------|
| JUMPn | ✅ (implicit) | ✅ `derivatives_jump.py` | ❌ | ⚠️ | ⚠️ | Mask-based, requires TOA flags |

---

## 3. DDK Implementation Plan

### 3.1 Current DDK Status

**Forward Model**: EXISTS in `jug/delays/combined.py:branch_ddk()` (lines 222-335)
- K96 proper motion corrections (Kopeikin 1996): δ_KIN, δ_a1, δ_omega
- Kopeikin 1995 annual orbital parallax corrections: δ_a1_px, δ_omega_px
- Uses observer position in light-seconds (`obs_pos_ls`)
- Computes effective A1 and OM, then calls DD delay

**Status**: Fully implemented. Forward model, analytic partials (KIN/KOM), fitting all operational.

### 3.2 What DDK Requires

**Required Parameters** (already in ParameterSpec):
- `KIN` - orbital inclination (degrees)
- `KOM` - position angle of ascending node (degrees)
- `PX` - parallax (mas) - used for distance calculation
- `PMRA` - proper motion in RA (mas/yr)
- `PMDEC` - proper motion in DEC (mas/yr)

**Plus all DD parameters**: PB, A1, ECC, OM, T0, GAMMA, PBDOT, OMDOT, SINI, M2

### 3.3 Forward Model Implementation (COMPLETE)

The DDK forward model in `combined.py:branch_ddk()` implements:

1. **K96 Proper Motion Corrections** (Kopeikin 1996):
   ```python
   # δ_KIN from proper motion (Eq 10)
   delta_kin_pm = (-pmra * sin(KOM) + pmdec * cos(KOM)) * (t - T0)
   
   # δ_a1 from proper motion (Eq 8)
   delta_a1_pm = a1 * delta_kin_pm / tan(KIN)
   
   # δ_omega from proper motion (Eq 9)
   delta_omega_pm = (1/sin(KIN)) * (pmra * cos(KOM) + pmdec * sin(KOM)) * (t - T0)
   ```

2. **Kopeikin 1995 Annual Orbital Parallax**:
   ```python
   # Observer projection terms
   delta_I0 = -x * sin_ra + y * cos_ra
   delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
   
   # Distance from parallax
   d_ls = 1000 * PC_TO_LIGHT_SEC / px_mas
   
   # δ_a1 from parallax (Eq 17)
   delta_a1_px = (a1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
   
   # δ_omega from parallax (Eq 19)
   delta_omega_px = -(1 / sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
   ```

3. **Apply corrections**:
   ```python
   a1_eff = a1 + delta_a1_pm + delta_a1_px
   om_eff = om + delta_omega_pm + delta_omega_px
   sini_eff = sin(KIN_eff) if SINI not provided
   
   return dd_binary_delay(t, pb, a1_eff, ecc, om_eff, ...)
   ```

### 3.4 Missing: Partial Derivatives for DDK

**To fit KIN and KOM**, we need analytic partials:

```python
# d(delay)/d(KIN) - chain rule through corrections
d_delay_d_KIN = (
    d_delay_d_a1 * d_a1_eff_d_KIN + 
    d_delay_d_om * d_om_eff_d_KIN +
    d_delay_d_sini * d_sini_eff_d_KIN
)

# d(delay)/d(KOM) - chain rule through corrections
d_delay_d_KOM = (
    d_delay_d_a1 * d_a1_eff_d_KOM + 
    d_delay_d_om * d_om_eff_d_KOM
)
```

### 3.5 Step-by-Step Implementation Plan

**Phase 1: Enable DDK Forward Model (No Fitting)**

1. ~~Modify `resolve_binary_model()`~~ — DONE: Override mechanism removed, DDK fully implemented.

2. **Pass observer position to combined.py**:
   - `simple_calculator.py` already computes `ssb_obs_pos_km`
   - Convert to light-seconds and pass to `combined_delays()`

3. **Pass K96 parameters**:
   - Extract PMRA, PMDEC from params (convert mas/yr → rad/s)
   - Pass to `combined_delays()` as `pmra_rad_per_sec`, `pmdec_rad_per_sec`

**Phase 2: Implement DDK Partials**

File: `jug/fitting/derivatives_ddk.py` (NEW)

```python
def compute_binary_derivatives_ddk(params, toas_bary_mjd, fit_params):
    """Compute DDK parameter derivatives.
    
    For DD parameters (PB, A1, ECC, OM, T0, etc.):
        Use modified DD derivatives with effective A1/OM
    
    For DDK-specific parameters (KIN, KOM):
        Compute chain rule derivatives through Kopeikin corrections
    """
    # Get base DD derivatives with effective parameters
    dd_derivs = compute_binary_derivatives_dd(params_eff, toas_bary_mjd, dd_params)
    
    # Compute KIN derivative
    if 'KIN' in fit_params:
        d_KIN = _d_delay_d_KIN(toas, params, dd_derivs)
        
    # Compute KOM derivative
    if 'KOM' in fit_params:
        d_KOM = _d_delay_d_KOM(toas, params, dd_derivs)
```

**Phase 3: Register DDK in Binary Registry**

```python
# In binary_registry.py
register_binary_model(
    'DDK',
    compute_ddk_binary_delay,  # wrapper that enables model_id=5
    compute_binary_derivatives_ddk
)
```

**Phase 4: Testing**

1. **Forward model test**: Compare JUG DDK vs PINT DDK for J0437-4715
2. **Partial derivative test**: Numeric gradient check vs analytic
3. **Parity test**: Fit KIN/KOM and compare to PINT

### 3.6 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `jug/fitting/derivatives_ddk.py` | CREATE | DDK-specific partials (KIN, KOM) |
| `jug/fitting/binary_registry.py` | MODIFY | Register DDK with new derivative function |
| `jug/utils/binary_model_overrides.py` | DELETED | Override mechanism removed (DDK fully implemented) |
| `jug/residuals/simple_calculator.py` | MODIFY | Pass obs_pos_ls for DDK |
| `tests/test_ddk_forward.py` | CREATE | Forward model vs PINT |
| `tests/test_ddk_derivatives.py` | CREATE | Numeric vs analytic partials |
| `tests/data_golden/J0437_mini.par` | CREATE | Test DDK par file |

---

## 4. Tests to Add

### 4.1 Unit Tests for New Helpers

**File**: `tests/test_derivatives_ddk.py`
```python
def test_d_delay_d_KIN_analytic_vs_numeric():
    """Verify KIN partial matches numeric gradient."""
    
def test_d_delay_d_KOM_analytic_vs_numeric():
    """Verify KOM partial matches numeric gradient."""
    
def test_ddk_corrections_zero_without_parallax():
    """Kopeikin corrections should be zero if PX=0."""
    
def test_ddk_corrections_grow_with_time():
    """K96 corrections should grow linearly with time from T0."""
```

### 4.2 Regression Tests for Evaluation Order

**File**: `tests/test_prebinary_evaluation.py` (extends existing)
```python
def test_prebinary_time_components():
    """Verify prebinary_delay_sec = roemer + shapiro + dm + sw + tropo."""
    
def test_prebinary_excludes_fd():
    """Verify FD delay is NOT included in prebinary time."""
    
def test_binary_evaluated_at_prebinary():
    """Verify binary delay function receives t - prebinary, not t - roemer."""
```

### 4.3 PINT Parity Tests

**File**: `tests/test_pint_parity_dd.py`
```python
def test_dd_delay_vs_pint():
    """Compare DD binary delay to PINT for test pulsar."""
    
def test_dd_derivatives_vs_pint():
    """Compare DD partial derivatives to PINT."""
```

**File**: `tests/test_pint_parity_ddk.py`
```python
def test_ddk_delay_vs_pint():
    """Compare DDK binary delay to PINT for J0437-4715."""
    
def test_ddk_kopeikin_corrections_vs_pint():
    """Verify Kopeikin correction terms match PINT."""
```

### 4.4 DDK Dataset Strategy

**Option A: Bundled Mini Dataset**
- Create `tests/data_golden/J0437_mini.par` with DDK binary model
- Create `tests/data_golden/J0437_mini.tim` with ~50 TOAs
- Tests run without external dependencies

**Option B: External Data Test (Auto-Skip)**
```python
@pytest.mark.skipif(not J0437_DATA_EXISTS, reason="J0437 data not available")
def test_ddk_full_fit_vs_pint():
    """Full DDK fit comparison to PINT (requires external data)."""
```

### 4.5 Test File Summary

| File | Category | What It Tests |
|------|----------|---------------|
| `tests/test_derivatives_ddk.py` | unit | KIN/KOM analytic partials |
| `tests/test_prebinary_evaluation.py` | regression | Evaluation order invariants |
| `tests/test_pint_parity_dd.py` | parity | DD vs PINT |
| `tests/test_pint_parity_ddk.py` | parity | DDK vs PINT |
| `tests/test_ddk_forward.py` | critical | DDK forward model correctness |

---

## 5. Gaps and Missing Parameters

### High Priority (Blocks Science)

| Gap | Impact | Effort | File to Modify |
|-----|--------|--------|----------------|
| **DDK partials** | Can't fit J0437-4715 | Medium | NEW: derivatives_ddk.py |
| **EDOT partial** | Can't fit eccentricity derivative | Low | derivatives_dd.py |

### Medium Priority (Less Common)

| Gap | Impact | Effort |
|-----|--------|--------|
| DR, DTH partials | Relativistic deformation (rarely fitted) | Medium |
| A0, B0 partials | Aberration delay (rarely fitted) | Low |
| EPS1DOT, EPS2DOT | ELL1 time derivatives (rare) | Low |

### Low Priority (Future Work)

| Gap | Impact |
|-----|--------|
| JUMP fitting integration | Needs TOA flag parsing |
| GLF0, GLPH etc. | Glitch parameters |
| FBn derivatives for n>20 | Extreme pulsars |

---

## 6. Summary

### What Works Well ✅
- Spin parameters (F0-F3): Full analytic partials, PINT-compatible
- DM parameters (DM, DM1, DM2): Full analytic partials
- Astrometry (RAJ, DECJ, PMRA, PMDEC, PX): Full analytic partials with PINT-style damping
- ELL1 binary (PB, A1, TASC, EPS1, EPS2, SINI, M2, FBn): 3rd-order ELL1 corrections
- DD binary core (PB, A1, T0, ECC, OM, GAMMA, PBDOT, OMDOT): Chain rule through Kepler
- FD parameters (FD1-FD9): Trivial partials
- Prebinary time evaluation: PINT-compatible (roemer + shapiro + dm + sw + tropo)

### What Needs Work 🚧
- **DDK**: Forward model exists but blocked; partials not implemented
- **EDOT partial**: Missing in derivatives_dd.py
- **DR, DTH, A0, B0**: In ParameterSpec but no implementation
- **JUMP fitting**: Code exists but integration needs testing

### Recommended Next Steps

1. **Enable DDK forward model** (Phase 1) - ~2 hours
2. **Implement DDK partials** (Phase 2) - ~4 hours  
3. **Add EDOT partial** - ~30 minutes
4. **Create J0437 mini dataset** - ~1 hour
5. **Add parity tests** - ~3 hours

Total estimated effort: ~10-12 hours for full DDK support
