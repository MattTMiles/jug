# DD Binary Model Integration Plan

## Current Status

The JUG codebase currently has:
- ✅ ELL1 binary model fully working (inline in `combined_delays()`)
- ✅ DD, BT, T2 binary model implementations exist in separate files
- ✅ Binary dispatch system created (`binary_dispatch.py`)
- ❌ No integration between simple_calculator and DD/BT/T2 models

## Problem

`simple_calculator.py` and `combined.py` are hardcoded for ELL1 model:
- Uses TASC, EPS1, EPS2 (ELL1 parameters)
- Doesn't detect BINARY parameter from .par file
- Can't handle T0, ECC, OM (DD/BT/T2 parameters)

## Solution Architecture

### Keep ELL1 Fast Path (No Changes)
For ELL1/ELL1H pulsars, keep the current inline code path:
- `combined_delays()` computes ELL1 inline for maximum JIT optimization
- No performance regression for J1909-3744

### Add Multi-Model Support
For DD/BT/T2 pulsars, create a new code path:

1. **Detection** in `simple_calculator.py`:
   ```python
   binary_model = params.get('BINARY', 'ELL1').upper()
   is_ell1 = binary_model in ('ELL1', 'ELL1H')
   ```

2. **Parameter Extraction**:
   - ELL1: TASC, EPS1, EPS2, PBDOT, XDOT, GAMMA, M2/SINI
   - DD/BT/T2: T0, ECC, OM, PBDOT, XDOT, OMDOT, EDOT, GAMMA, M2/SINI

3. **Dual Compute Paths**:
   ```python
   if is_ell1:
       # Current code path (fast, inline ELL1)
       total_delay = compute_total_delay_jax_ell1(...)
   else:
       # New code path (DD/BT/T2 via dispatcher)
       total_delay = compute_total_delay_jax_keplerian(binary_model, ...)
   ```

## Implementation Steps

### Step 1: Create Keplerian Delay Function
New file: `jug/delays/combined_keplerian.py`
- Copy structure from `combined.py`
- Replace inline ELL1 code with call to `dispatch_binary_delay()`
- Keep DM, solar wind, FD delays identical

### Step 2: Update simple_calculator.py
- Detect binary model from params
- Extract appropriate parameters based on model
- Route to correct compute function

### Step 3: Test
- Verify J1909-3744 (ELL1) still works (no regression)
- Test J0437-4715 (DDK) matches PINT
- Test other DD pulsars

## Files to Modify

1. `jug/delays/combined_keplerian.py` (NEW)
   - JIT-compiled delay function for DD/BT/T2 models
   
2. `jug/residuals/simple_calculator.py` (MODIFY)
   - Add binary model detection
   - Add parameter extraction for Keplerian models
   - Route to appropriate delay function

3. `jug/delays/__init__.py` (UPDATE)
   - Export new functions

## Testing Plan

1. **Baseline**: Run J1909-3744 (ELL1) before and after changes
2. **DD Model**: Run J0437-4715 (DDK) and compare to PINT
3. **Other Models**: Test BT, DDH, DDGR pulsars from MPTA

## Performance Considerations

- ELL1 keeps inline optimization (no change)
- DD/BT/T2 will be slightly slower (extra function call) but still JIT-compiled
- Trade-off: ~5% slower for DD models, but gain full model support

## Questions for User

1. Should I proceed with this dual-path architecture?
2. Is ~5% performance hit for non-ELL1 models acceptable?
3. Any concerns about breaking existing ELL1 code?
