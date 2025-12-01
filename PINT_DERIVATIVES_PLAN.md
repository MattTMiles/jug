# Implementation Plan: PINT Analytical Derivatives in JUG

**Date**: 2025-12-01  
**Approach**: Copy PINT's hand-coded analytical derivatives
**Estimated Time**: 6-12 hours
**Status**: STARTING

---

## Strategy

Copy PINT's proven analytical derivative formulas into JUG, using float64 instead of float128.

### Why This Works

1. **PINT's derivatives are mathematical formulas** - not dependent on float128
2. **Our residuals already work** - 0.4 μs with float64
3. **Analytical derivatives avoid numerical issues** - no finite differences
4. **Proven approach** - PINT has used this for years

---

## Implementation Steps

### Phase 1: Spin Parameters (2 hours) ✓ START HERE

**Parameters**: F0, F1, F2, F3, F4, ...

**PINT Source**: 
- `src/pint/models/spindown.py` - Spindown class
- Method: `d_phase_d_F0()`, `d_phase_d_F1()`, etc.

**Derivatives** (simple polynomials):
```python
d(phase)/d(F0) = dt
d(phase)/d(F1) = 0.5 * dt^2
d(phase)/d(F2) = (1/6) * dt^3
d(phase)/d(F3) = (1/24) * dt^4
```

**JUG Implementation**:
- Create: `jug/fitting/derivatives_spin.py`
- Functions for each F derivative
- Take params dict, return column of design matrix

---

### Phase 2: DM Parameters (1-2 hours)

**Parameters**: DM, DM1, DM2, DM3, ...

**PINT Source**:
- `src/pint/models/dispersion_model.py`
- Method: `d_delay_d_DM()`, etc.

**Derivatives**:
```python
d(delay)/d(DM) = K_DM / freq^2
d(delay)/d(DM1) = K_DM / freq^2 * (t - DMEPOCH)
d(delay)/d(DM2) = K_DM / freq^2 * 0.5 * (t - DMEPOCH)^2
```

**Note**: DM affects delay, not phase directly. Need chain rule:
```python
d(phase)/d(DM) = d(phase)/d(delay) * d(delay)/d(DM)
                = -F0 * d(delay)/d(DM)
```

**JUG Implementation**:
- Create: `jug/fitting/derivatives_dm.py`
- Functions for each DM derivative

---

### Phase 3: Binary Parameters (3-4 hours) - COMPLEX

**Parameters**: 
- ELL1: PB, A1, TASC, EPS1, EPS2, EPS1DOT, EPS2DOT
- DD/BT: PB, A1, ECC, OM, T0, OMDOT, PBDOT, GAMMA, SINI, M2

**PINT Source**:
- `src/pint/models/binary_ell1.py` - ELL1 derivatives
- `src/pint/models/binary_dd.py` - DD derivatives
- `src/pint/models/binary_bt.py` - BT derivatives

**Challenge**: Chain rule through Kepler equation and orbital geometry

**Example** (simplified):
```python
d(binary_delay)/d(PB):
  1. Compute d(mean_anomaly)/d(PB)
  2. Compute d(eccentric_anomaly)/d(mean_anomaly) via implicit diff
  3. Compute d(delay)/d(eccentric_anomaly)
  4. Chain rule: combine all three
```

**JUG Implementation**:
- Create: `jug/fitting/derivatives_binary.py`
- Separate functions for each binary model
- Start with ELL1 (simpler), then DD, then BT

---

### Phase 4: Astrometry Parameters (2-3 hours)

**Parameters**: RAJ, DECJ, PMRA, PMDEC, PX

**PINT Source**:
- `src/pint/models/astrometry.py`
- Methods: `d_delay_d_RAJ()`, etc.

**Derivatives**: Geometric - position changes → Roemer delay changes

**Example**:
```python
d(roemer_delay)/d(RAJ):
  Direction vector changes with RA
  Affects dot product with position vector
  Uses chain rule through coordinate transforms
```

**JUG Implementation**:
- Create: `jug/fitting/derivatives_astrometry.py`
- Vector calculus for geometric derivatives

---

### Phase 5: Integration Layer (1-2 hours)

**Create**: `jug/fitting/design_matrix_analytical.py`

**Purpose**: Assemble all derivative columns into design matrix

```python
def compute_design_matrix_analytical(
    params: Dict,
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    fit_params: List[str]
) -> np.ndarray:
    """
    Compute design matrix using PINT-style analytical derivatives.
    
    Returns
    -------
    M : np.ndarray, shape (n_toas, n_params)
        Design matrix where M[i,j] = d(residual_i)/d(param_j)
    """
    n_toas = len(toas_mjd)
    n_params = len(fit_params)
    M = np.zeros((n_toas, n_params))
    
    for j, param in enumerate(fit_params):
        if param.startswith('F'):
            M[:, j] = compute_spin_derivative(param, ...)
        elif param.startswith('DM'):
            M[:, j] = compute_dm_derivative(param, ...)
        elif param in ['PB', 'A1', 'ECC', ...]:
            M[:, j] = compute_binary_derivative(param, ...)
        elif param in ['RAJ', 'DECJ', ...]:
            M[:, j] = compute_astrometry_derivative(param, ...)
    
    return M
```

---

### Phase 6: Testing (1-2 hours)

**Test 1**: Validate derivatives against PINT
- Compute design matrix with JUG
- Compute design matrix with PINT
- Compare column by column
- Should match to machine precision

**Test 2**: Fit J1909-3744
- Use analytical design matrix
- Use WLS solver
- Check convergence
- Compare fitted params to PINT

**Test 3**: Fit multiple pulsars
- Binary pulsar
- Isolated pulsar
- High proper motion pulsar

---

## File Structure

```
jug/fitting/
├── derivatives_spin.py         # F0, F1, F2, F3 derivatives
├── derivatives_dm.py            # DM, DM1, DM2 derivatives
├── derivatives_binary.py        # Binary parameter derivatives
├── derivatives_astrometry.py    # Position/proper motion derivatives
├── design_matrix_analytical.py  # Integration layer
└── wls_fitter.py               # Already exists
```

---

## Timeline (Conservative)

| Phase | Time | Cumulative |
|-------|------|------------|
| 1. Spin parameters | 2 hrs | 2 hrs |
| 2. DM parameters | 2 hrs | 4 hrs |
| 3. Binary parameters | 4 hrs | 8 hrs |
| 4. Astrometry parameters | 3 hrs | 11 hrs |
| 5. Integration layer | 1 hr | 12 hrs |
| 6. Testing & debugging | 2 hrs | 14 hrs |

**Total**: 12-14 hours (split across multiple sessions)

---

## Session Plan

**Session 13 (now)**: 
- Set up structure
- Implement Phase 1 (spin derivatives)
- Test spin derivatives

**Session 14**:
- Implement Phase 2 (DM derivatives)
- Start Phase 3 (binary - ELL1 only)

**Session 15**:
- Complete Phase 3 (binary - DD/BT)
- Implement Phase 4 (astrometry)

**Session 16**:
- Phase 5 (integration)
- Phase 6 (testing)
- Complete Milestone 2

---

## Resources

**PINT Source Code**:
- GitHub: https://github.com/nanograv/PINT
- Relevant files:
  - `src/pint/models/spindown.py`
  - `src/pint/models/dispersion_model.py`
  - `src/pint/models/binary_*.py`
  - `src/pint/models/astrometry.py`

**JUG Existing Code**:
- `jug/delays/combined.py` - JAX delay calculations
- `jug/residuals/simple_calculator.py` - Residual computation
- `jug/fitting/wls_fitter.py` - WLS solver

---

## Notes

1. **Float64 is fine** - PINT's formulas don't require float128, just use careful arithmetic
2. **Start simple** - Get spin working first, then add complexity
3. **Test incrementally** - Validate each derivative before moving on
4. **Copy carefully** - PINT's formulas are proven, don't modify
5. **Document sources** - Note which PINT file/line each formula comes from

---

## Success Criteria

✅ All derivatives implemented
✅ Design matrix matches PINT column-by-column
✅ Fitting converges on J1909-3744
✅ Fitted parameters match PINT within uncertainties
✅ RMS achieves ~0.4 μs (reference value)

---

**Ready to start Phase 1: Spin derivatives**

---

## Progress Update

### Phase 1: Spin Parameters ✅ COMPLETE (Session 13)

**Time**: 1 hour
**Status**: Working and tested

**Files Created**:
- `jug/fitting/derivatives_spin.py` (250 lines)

**What Works**:
- `taylor_horner()` - PINT's Horner scheme for Taylor series
- `d_phase_d_F()` - General derivative for any F parameter
- `compute_spin_derivatives()` - Integration function for design matrix
- Individual functions: `d_phase_d_F0()`, `d_phase_d_F1()`, `d_phase_d_F2()`, `d_phase_d_F3()`

**Test Results**:
```
taylor_horner(2.0, [10, 3, 4, 12]) = 40.0  ✓
d/dF0 = [1, 2, 3]  ✓
d/dF1 = [0.5, 2.0, 4.5]  ✓ 
d/dF2 = [0.167, 1.333, 4.5]  ✓
```

**Next**: Phase 2 (DM derivatives) - 2 hours

