# Milestone 2 - Session 5: Binary Model Expansion Plan

**Date**: 2025-11-30
**Status**: Planning Complete
**Session Focus**: Multi-pulsar testing strategy and binary model expansion

---

## Session Summary

After completing the core Gauss-Newton fitting implementation (Session 4), we identified a critical gap: JUG currently only supports **ELL1** binary model, but real pulsar datasets use diverse binary models (BT, DD, DDH, DDGR, T2).

This session establishes a testing strategy for validating JUG on multiple pulsars and a plan to implement the missing binary models, particularly the **T2 (Tempo2 general)** model.

---

## Problem Statement

### Current Limitation
- JUG only implements ELL1 binary delay (low-eccentricity approximation)
- Many pulsars use BT/DD (Keplerian + 1PN) or T2 (general Tempo2 model)
- Cannot validate fitting on diverse pulsar population

### Impact
- **Cannot test** on non-binary pulsars (e.g., J0437-4715)
- **Cannot test** on high-eccentricity binaries (e.g., J1614-2230 with BT/DD)
- **Risk** of hidden bugs that only appear with other binary models
- **Incomplete** Milestone 2 validation

---

## Multi-Pulsar Testing Strategy

### Test Pulsar Selection

We need at least 3 diverse pulsars to validate robustness:

| Pulsar | Type | Binary Model | Purpose |
|--------|------|--------------|---------|
| **J1909-3744** | Binary MSP | ELL1 | Already validated (M1) |
| **J0437-4715** | Non-binary MSP | None | Test spin-only timing |
| **J1614-2230** | Binary, massive NS | BT/DD | Test Keplerian model |

### Why These Pulsars?

1. **J1909-3744** (Already working):
   - Binary MSP with very low eccentricity
   - Uses ELL1 model (our current implementation)
   - Excellent data quality (10,408 TOAs from MPTA)
   - Strong Shapiro delay signal (SINI ~ 0.99)

2. **J0437-4715** (NEW - Non-binary):
   - One of the closest MSPs (157 pc)
   - No binary companion → tests spin timing without orbital effects
   - Very stable → excellent for validating F0, F1, F2 derivatives
   - Large proper motion → tests astrometry (future)

3. **J1614-2230** (NEW - Keplerian binary):
   - Massive neutron star (1.97 ± 0.04 M☉)
   - High orbital eccentricity (e ~ 0.0002) → needs BT/DD model
   - Short orbital period (8.7 days)
   - Tests Keplerian + relativistic parameters

---

## Binary Models to Implement

### 1. BT (Blandford-Teukolsky) / DD (Damour-Deruelle)

**Status**: Not implemented
**Priority**: HIGH
**Estimated time**: 2 hours

**Parameters**:
- PB: Orbital period (days)
- A1: Projected semi-major axis (light-seconds)
- ECC: Orbital eccentricity
- OM: Longitude of periastron (degrees)
- T0: Time of periastron passage (MJD)
- GAMMA: Einstein delay parameter (s)
- PBDOT: Orbital period derivative
- OMDOT: Periastron advance (deg/yr) [DD only]
- SINI: Sine of inclination angle [DD only]

**Delay Components**:
1. **Roemer delay**: Light travel time in elliptical orbit
   - Solve Kepler's equation: `E - e*sin(E) = 2π(t-T0)/PB`
   - Compute true anomaly and projected position
   - Delay = `A1 * [sin(ω+ν) + e*sin(ω)]`

2. **Einstein delay**: Time dilation in eccentric orbit
   - Delay = `GAMMA * sin(E)`

3. **Shapiro delay**: Light bending by companion
   - Delay = `-2*RANGE * log(1 - SINI*sin(ω+ν))`
   - RANGE = T☉ * M2 (companion mass in time units)

**Implementation Plan**:
1. Create `jug/delays/binary_bt.py` with Kepler solver
2. Add to `combined_delays()` as alternative to ELL1
3. Implement binary model detection in `simple_calculator.py`
4. Add analytical derivatives in `design_matrix.py`

**References**:
- Blandford & Teukolsky (1976), ApJ 205, 580
- Damour & Deruelle (1985), Ann. Inst. H. Poincaré (Physique Théorique) 43, 107
- PINT: `pint/models/binary_bt.py`, `pint/models/binary_dd.py`
- Tempo2: `T2model_BTmodel.C`

---

### 2. T2 (Tempo2 General Binary Model)

**Status**: Not implemented
**Priority**: CRITICAL
**Estimated time**: 3-4 hours

**Purpose**: 
The T2 model is Tempo2's "universal" binary model that can emulate any other binary model by setting appropriate parameters. Many real .par files specify `BINARY T2`, so JUG must support it.

**Parameters** (subset - T2 has many):
- PB, A1, ECC, OM, T0 (Keplerian)
- SINI, M2 (Shapiro delay)
- KIN, KOM (inclination, ascending node)
- GAMMA, PBDOT, OMDOT (relativistic)
- XDOT, EDOT (time derivatives)

**Key Features**:
1. **Generality**: Can represent BT, DD, ELL1, and others
2. **Flexibility**: Supports arbitrary parameter subsets
3. **Compatibility**: Many published .par files use T2

**Implementation Challenges**:
1. Many optional parameters (need graceful defaults)
2. Multiple parameterizations (e.g., M2/SINI vs. H3/STIG)
3. Coordinate transformations (KIN/KOM → sky plane)

**Implementation Plan**:
1. Create `jug/delays/binary_t2.py` with full Kepler solver
2. Support all T2 parameter combinations
3. Add auto-detection of parameter sets (e.g., detect M2+SINI vs H3+STIG)
4. Validate against Tempo2 output on real .par files
5. Add to design matrix for fitting support

**References**:
- Tempo2: `T2model.C`, `T2model.h`
- Edwards+ (2006), MNRAS 372, 1549 (Tempo2 paper)
- PINT: `pint/models/binary_t2.py` (if exists)

---

## Implementation Task List

### Task 2.9: Multi-Pulsar Testing and Binary Model Expansion

**Assigned to**: Claude
**Priority**: HIGH
**Estimated time**: 3-4 hours
**Status**: Not started

#### Subtasks:

1. **Download test pulsars** (30 min):
   - [ ] Get J0437-4715.par and .tim from public archives
   - [ ] Get J1614-2230.par and .tim from public archives
   - [ ] Verify data format compatibility with JUG

2. **Implement BT/DD binary model** (2 hours):
   - [ ] Create `jug/delays/binary_bt.py`
   - [ ] Implement Kepler equation solver (Newton-Raphson)
   - [ ] Implement Roemer delay for elliptical orbits
   - [ ] Implement Einstein delay (GAMMA)
   - [ ] Implement Shapiro delay (RANGE, SINI)
   - [ ] Add JAX JIT compilation
   - [ ] Test against Tempo2/PINT output

3. **Implement T2 binary model** (2-3 hours):
   - [ ] Create `jug/delays/binary_t2.py`
   - [ ] Support full Keplerian parameters
   - [ ] Support M2/SINI and H3/STIG parameterizations
   - [ ] Implement coordinate transformations (KIN/KOM)
   - [ ] Add JAX JIT compilation
   - [ ] Test against Tempo2 T2 output

4. **Binary model auto-detection** (30 min):
   - [ ] Modify `simple_calculator.py` to detect BINARY parameter
   - [ ] Route to correct binary delay function (ELL1/BT/DD/T2)
   - [ ] Raise informative error for unsupported models

5. **Validation testing** (1 hour):
   - [ ] Run JUG on J0437-4715 (no binary)
   - [ ] Run JUG on J1614-2230 (BT/DD model)
   - [ ] Compare residuals with PINT/Tempo2
   - [ ] Document RMS differences

6. **Design matrix expansion** (1 hour):
   - [ ] Add analytical derivatives for BT/DD parameters
   - [ ] Add analytical derivatives for T2 parameters
   - [ ] Test fitting convergence on BT binary

7. **Documentation** (30 min):
   - [ ] Update `CLAUDE.md` with BT/DD/T2 implementation details
   - [ ] Create `MULTI_PULSAR_VALIDATION.md` with test results
   - [ ] Update `JUG_PROGRESS_TRACKER.md`

---

## Technical Implementation Details

### Kepler Equation Solver

Both BT and T2 require solving Kepler's equation:
```
E - e*sin(E) = M    (M = mean anomaly)
```

**Algorithm**: Newton-Raphson iteration
```python
@jax.jit
def solve_kepler(mean_anomaly, eccentricity, tol=1e-12, max_iter=20):
    """Solve Kepler's equation E - e*sin(E) = M.
    
    Uses Newton-Raphson with starting guess E0 = M.
    """
    E = mean_anomaly
    for i in range(max_iter):
        f = E - eccentricity * jnp.sin(E) - mean_anomaly
        fp = 1.0 - eccentricity * jnp.cos(E)
        E_new = E - f / fp
        
        if jnp.abs(E_new - E) < tol:
            break
        E = E_new
    
    return E
```

**Vectorization**: Use `jax.vmap()` for batch processing all TOAs.

### Binary Delay Integration

Modify `combined_delays()` in `jug/delays/combined.py`:

```python
# Current (ELL1 only):
binary_sec = jnp.where(
    has_binary,
    compute_ell1_binary(...),
    0.0
)

# Proposed (multi-model):
if binary_model == 'ELL1':
    binary_sec = compute_ell1_binary(...)
elif binary_model == 'BT' or binary_model == 'DD':
    binary_sec = compute_bt_binary(...)
elif binary_model == 'T2':
    binary_sec = compute_t2_binary(...)
else:
    raise ValueError(f"Unsupported binary model: {binary_model}")
```

**Note**: Need to pass `binary_model` parameter through the call chain.

---

## Success Criteria

Task 2.9 is complete when:

- [ ] JUG computes residuals for J0437-4715 (non-binary)
- [ ] JUG computes residuals for J1614-2230 (BT/DD binary)
- [ ] Residual RMS matches PINT within 1%
- [ ] Residual differences < 0.1 μs standard deviation
- [ ] BT/DD and T2 binary delays implemented
- [ ] Binary model auto-detection works from .par file
- [ ] Design matrix includes BT/DD/T2 parameter derivatives
- [ ] Documentation updated with new models

---

## Open Questions

1. **Which pulsar archives to use?**
   - IPTA Data Release?
   - Individual observatory archives (Parkes, Arecibo)?
   - PINT example data?

2. **How to handle T2 parameter subsets?**
   - Default unused parameters to zero?
   - Require explicit declaration in .par file?

3. **Priority order for binary models?**
   - Suggested: T2 first (most general), then BT/DD

4. **Should we support deprecated models?**
   - MSS (low-eccentricity, superseded by ELL1)
   - DDS (DD with SHAPMAX parameter)
   - Probably not needed if T2 handles them

---

## Timeline

**Total estimated time**: 6-8 hours

**Suggested breakdown**:
- Session A (2-3 hours): Implement BT/DD model, test on J1614-2230
- Session B (3-4 hours): Implement T2 model, test on multiple pulsars
- Session C (1 hour): Add design matrix derivatives, documentation

**Target completion**: Before Milestone 3 (white noise)

---

## References

### Papers
- Blandford & Teukolsky (1976) - BT model
- Damour & Deruelle (1985) - DD model
- Edwards+ (2006) - Tempo2 (includes T2 model)

### Software
- PINT: `pint/models/binary_*.py`
- Tempo2: `T2model*.C` files
- libstempo: Python wrapper for Tempo2

### Data Sources
- IPTA Data Release: https://www.ipta4gw.org/
- NANOGrav 15-year: https://nanograv.org/
- EPTA: https://www.epta.eu.org/

---

**Next Steps**: Start with BT/DD implementation, test on J1614-2230, then tackle T2 model.

