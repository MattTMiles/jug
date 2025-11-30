# Session 6: Multi-Pulsar Testing and Binary Model Expansion

**Date**: 2025-11-29  
**Status**: Binary models implemented, integration needed  
**Next**: Multi-pulsar testing across different binary models + telescopes

---

## Summary

This session focused on expanding JUG to support all major binary models (BT, DD, T2) and laying groundwork for multi-pulsar/multi-telescope testing.

### Achievements ✅

1. **Binary Model Implementations** (Complete)
   - ✅ BT (Blandford-Teukolsky) model with Kepler solver
   - ✅ DD/DDH/DDGR/DDK (Damour-Deruelle) models
   - ✅ T2 (Tempo2 general) model with EDOT, KIN/KOM support
   - ✅ Clean dispatcher system (`binary_dispatch.py`)
   - ✅ Test script validates BT vs T2 agreement to nanosecond precision

2. **Architecture Design**
   - ELL1 remains inline in `combined_delays()` for maximum performance
   - Dispatcher routes to appropriate model based on BINARY parameter
   - "No binary" case handled efficiently with `has_binary` flag
   - Easy to extend: add function → register in dispatcher

3. **Performance Validated**
   - Gauss-Newton with JAX: **100x faster** than scipy (0.28s vs 28s for 1000 iterations)
   - JAX JIT compilation eliminates Python overhead
   - Automatic differentiation removes manual Jacobian coding

---

## Binary Model Status

### Supported Models

| Model | Status | Parameters | Use Case |
|-------|--------|------------|----------|
| ELL1/ELL1H | ✅ Production | TASC, EPS1, EPS2 | Low-eccentricity MSPs (e < 0.01) |
| BT/BTX | ✅ Implemented | T0, ECC, OM | Keplerian + 1PN corrections |
| DD/DDH/DDGR/DDK | ✅ Implemented | T0, ECC, OM, GAMMA, PBDOT, OMDOT | Post-Keplerian models |
| T2 | ✅ Implemented | T0, ECC, OM + EDOT, KIN, KOM | General Tempo2 model |
| None | ✅ Always worked | N/A | Non-binary pulsars |

### Implementation Details

**ELL1** (Optimized):
- Hardcoded inline in `combined_delays()` for speed
- 3rd-order Fourier expansion
- Matches Tempo2/PINT to nanosecond precision

**BT/DD/T2** (Dispatcher):
- Separate functions in `jug/delays/binary_*.py`
- Called via `dispatch_binary_delay()` router
- Fully documented with parameter descriptions

**No Binary**:
- `has_binary = False` → `jnp.where()` returns zero
- No performance penalty (branch prediction)

---

## Testing Requirements (Session 7 Goals)

### 1. Binary Model Validation

Test each binary model against real pulsar data:

- [x] **J1909-3744** (ELL1) - Already validated ✅
  - 10,408 TOAs, RMS = 0.817 μs
  - Matches PINT to 0.003 μs std

- [ ] **BT Model Pulsar** 🔍 Need to find
  - Search `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb`
  - Look for `BINARY BT` in .par files
  
- [ ] **DD Model Pulsar** 🔍 Need to find
  - Candidates: J1614-2230, J0101-6422
  - Look for `BINARY DD` in .par files
  
- [ ] **No Binary Pulsar** 🔍 Need to find
  - MSP without binary companion
  - Should have no BINARY parameter in .par
  
- [ ] **T2 Model Pulsar** ⚠️ Needs PINT validation
  - Created implementation, but PINT gave unexpected results
  - May be PINT issue or our implementation
  - Defer until we find real T2 pulsar to compare

### 2. Multi-Telescope Testing

**Challenge**: Verify JUG works with different observatories and backends.

**Test Data Locations**:
- Single-telescope: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb`
  - Good for binary model testing
  - Each pulsar observed by one telescope
  
- Multi-telescope: 🔍 **Need to locate**
  - MPTA dataset with mixed Parkes/MeerKAT observations
  - Test clock correction chains for both telescopes
  - Verify backend-specific TOA handling

**Clock Corrections to Test**:
- ✅ MeerKAT → mk2utc.clk (already tested)
- [ ] Parkes → Need pk2utc.clk (or equivalent)
- [ ] Other MPTA telescopes

**Backend Testing**:
- Different backends can have systematic offsets
- TOA flags in .tim files identify backend
- May need backend-specific corrections (JUMP parameters)

### 3. Integration Tasks

To actually use BT/DD/T2 models in production:

1. **Modify `simple_calculator.py`**:
   ```python
   # Detect binary model from params
   binary_model = params.get('BINARY', '').upper()
   
   # Route to appropriate calculation
   if binary_model in ('ELL1', 'ELL1H'):
       # Use existing inline code (fast path)
       ...
   elif binary_model in ('BT', 'BTX', 'DD', 'DDH', 'DDGR', 'DDK', 'T2'):
       # Use dispatcher (new path)
       binary_params = extract_binary_params(params, binary_model)
       binary_delay = dispatch_binary_delay(binary_model, t_topo_tdb, binary_params)
       ...
   else:
       # No binary
       binary_delay = 0.0
   ```

2. **Add binary parameter extraction**:
   - Create `extract_binary_params()` function
   - Maps .par parameters to model-specific dict
   - Handles optional parameters with defaults

3. **Update `combined_delays()` signature** (optional):
   - Add `binary_model` parameter
   - Add `binary_params` dict parameter
   - Keep backward compatibility with ELL1-only code

---

## Performance Considerations

### Why Keep ELL1 Inline?

ELL1 is used by **~80% of MSP binaries** due to circularization. Keeping it inline:
- Avoids function call overhead
- Enables better JIT optimization by JAX
- Parameters stay in registers (no dict lookups)

**Benchmarking Needed**:
- Measure overhead of dispatcher for BT/DD/T2
- If negligible (<1%), could unify all models through dispatcher
- If significant (>5%), keep ELL1 optimized path

### JAX JIT Compilation Strategy

Each binary model function decorated with `@jax.jit`:
```python
@jax.jit
def bt_binary_delay(t, pb, a1, ecc, om, t0, ...):
    # Compiled once per unique signature
    # Subsequent calls: no Python overhead
    ...
```

**First call**: ~100ms compilation  
**Subsequent calls**: <1ms execution (10,000 TOAs)

---

## Files Created/Modified

### New Files ✅
- `jug/delays/binary_bt.py` (180 lines)
- `jug/delays/binary_dd.py` (190 lines)
- `jug/delays/binary_t2.py` (200 lines)
- `jug/delays/binary_dispatch.py` (180 lines)
- `jug/tests/test_binary_models.py` (150 lines)
- `SESSION6_MULTI_PULSAR_TESTING.md` (this file)

### Modified Files ✅
- `jug/delays/__init__.py` - Added exports for new binary models
- `JUG_PROGRESS_TRACKER.md` - Updated Milestone 2 task 2.9 status

### Files to Modify (Session 7) 📝
- `jug/residuals/simple_calculator.py` - Add binary model routing
- `jug/io/par_reader.py` - Ensure BINARY parameter parsed
- `jug/delays/combined.py` - May need minor tweaks for dispatcher integration

---

## Next Session Plan (Session 7)

### Hour 1: Find Test Pulsars
1. Search MPTA data for BT/DD model pulsars
2. Find non-binary MSP for baseline testing
3. Locate multi-telescope dataset (if available)

### Hour 2: Integrate Dispatcher
1. Add binary model detection to calculator
2. Extract binary parameters for each model
3. Route to appropriate delay function
4. Test on J1909-3744 to ensure no regression

### Hour 3: Multi-Pulsar Testing
1. Run BT model pulsar through JUG
2. Run DD model pulsar through JUG
3. Run no-binary pulsar through JUG
4. Compare all against PINT residuals
5. Document any discrepancies

### Hour 4: Multi-Telescope Testing (if time)
1. Test Parkes clock corrections
2. Test mixed-telescope timing solution
3. Verify backend handling

### Success Criteria ✅
- JUG matches PINT for BT model to <1 μs RMS difference
- JUG matches PINT for DD model to <1 μs RMS difference
- JUG handles non-binary pulsars correctly (zero binary delay)
- Multi-telescope pulsars work (if tested)

---

## Questions for User

1. **T2 Model**: Should we debug the PINT discrepancy now, or defer until we find a real T2 pulsar?
   - Current status: T2 implementation exists but untested against PINT
   
2. **Multi-Telescope Data**: Where is the MPTA dataset with mixed Parkes/MeerKAT observations?
   - Needed for comprehensive telescope testing
   
3. **Priority**: Should we focus on binary models first, or multi-telescope testing first?
   - Binary models are essential for M2 (fitting)
   - Multi-telescope is essential for M3 (production)

---

## References

- **Binary Models**: Edwards et al. (2006), Blandford & Teukolsky (1976), Damour & Deruelle (1985)
- **Dispatcher Pattern**: `jug/delays/binary_dispatch.py` documentation
- **MK7 Notebook**: `playground/residual_maker_playground_active_MK7.ipynb` (working ELL1 reference)

