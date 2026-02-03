# DD/DDK Model Testing Status

**Date**: 2026-02-03 (Updated by Copilot CLI Debugging Session)
**Author**: Claude (Copilot CLI Agent)  
**Status**: ⚠️ DD-family models have ~2.8 μs WRMS discrepancy vs PINT/Tempo2

## Summary

The DD-family binary model implementations (DD, DDK, DDH) in JUG were tested against PINT and Tempo2. While the ELL1 model agrees with Tempo2 to within ~83 ns, the DD models show a systematic discrepancy of **~2.8 μs WRMS** for J1713+0747 (DD).

**Results (2026-02-03 debugging session):**
| Pulsar | Binary | PINT WRMS | Tempo2 WRMS | JUG WRMS | JUG vs Tempo2 |
|--------|--------|-----------|-------------|----------|---------------|
| J0030+0451 | None | 2.98 μs | 2.99 μs | 2.98 μs | ✓ AGREE |
| J0613-0200 | ELL1H | 2.99 μs | 0.97 μs | 1.05 μs | ✓ AGREE |
| J1713+0747 | DD | 0.17 μs | 0.17 μs | 2.77 μs | ✗ DIFFER by 2.6 μs |

## Root Cause Analysis (Narrowed Down)

### The DD Algorithm is CORRECT ✅

**Critical Finding**: When calling `dd_binary_delay()` directly with identical parameters and time, JUG matches PINT to **< 1 ns**.

**Proof (from debugging session):**
```
Direct dd_binary_delay vs PINT binary: -0.043 ns  ← Algorithm is correct
JUG total_delay implies binary:        2.048 μs different from PINT
```

### The Issue is Inside `combined_delays` Function

The bug is in how `combined_delays()` returns the binary delay, NOT in the `dd_binary_delay()` function itself.

**Evidence:**
1. Calling `dd_binary_delay()` directly gives the correct value
2. Extracting the binary delay from JUG's `total_delay_sec` shows a ~2 μs discrepancy
3. The `combined_delays()` function appears to return incorrect values for DD models
4. ELL1 model works correctly through the same `combined_delays()` pathway

### Investigation Status

**Ruled Out:**
- ❌ The `dd_binary_delay()` algorithm - verified correct to <1 ns
- ❌ FD delay order (FD before/after binary) - tested, does not fix issue
- ❌ JAX JIT compilation issues - problem persists with JIT disabled
- ❌ Module caching (.pyc files) - cleared and retested
- ❌ Roemer+Shapiro delays - match PINT to <1 ns
- ❌ DM delay computation - matches PINT exactly

**Likely Cause:**
The issue appears to be in how `jax.lax.switch` handles the `branch_dd` closure in `combined.py` line 320-326. The branch functions capture parameters from the outer scope, and there may be a subtle bug in how these closures interact with JAX tracing.

**Key Code Location:**
```python
# jug/delays/combined.py, lines 187-192
def branch_dd(t):
    return dd_binary_delay(
        t, pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot,
        sini, m2, h3, h4, stig
    )
```

The parameters (pb, a1, ecc, om, t0, etc.) are captured from the outer `combined_delays` function scope.

### Recommendations for Future Work

1. **Add explicit parameter passing** instead of relying on closures for `branch_dd`:
   ```python
   # Instead of closure, pass parameters explicitly through the switch
   binary_params = (pb, a1, ecc, om, t0, gamma, pbdot, omdot, xdot, edot, sini, m2, h3, h4, stig)
   # ... find a way to pass these through jax.lax.switch
   ```

2. **Test with a minimal reproduction case** that isolates the `jax.lax.switch` behavior with closures

3. **Consider using `jax.lax.cond` instead of `switch`** for the DD branch specifically

4. **Add a fallback Python implementation** for debugging purposes that doesn't use JAX tracing

## FD Verbose Output Bug (Minor)

The verbose output in `simple_calculator.py` line 667 computes FD incorrectly:
```python
tzr_fd_delay = np.polyval(list(fd_coeffs_jax)[::-1], log_freq)  # WRONG
```

Should be:
```python
tzr_fd_delay = np.polyval(list(fd_coeffs_jax)[::-1] + [0], log_freq)  # CORRECT
```

This only affects the verbose debug output, not the actual residual computation.

## Test Commands

```bash
# Quick test
cd /home/mattm/soft/JUG
mamba activate discotech
python -c "
from jug.residuals.simple_calculator import compute_residuals_simple
from pathlib import Path
import numpy as np
base = Path('/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb')
result = compute_residuals_simple(base/'J1713+0747_tdb.par', base/'J1713+0747.tim')
weights = 1.0 / result['errors_us']**2
wrms = np.sqrt(np.sum(weights * result['residuals_us']**2) / np.sum(weights))
print(f'DD WRMS: {wrms:.3f} μs (expected: ~0.17 μs)')
"
```

## Recommended Fix

### Root Cause: DM Delay Calculation

The ~3.3 μs DM delay difference is the primary source of the discrepancy. To fix:

1. **Match PINT's K_DM constant exactly**:
   - Check PINT's value in `pint/utils.py` or dispersion module
   - Update `jug/utils/constants.py` to match

2. **Match PINT's barycentric frequency calculation**:
   - PINT computes barycentric freq = topo_freq * (1 - v_obs·L_hat/c)
   - Verify JUG's `compute_barycentric_freq()` matches

3. **Match PINT's DM derivative handling**:
   - Ensure DM1, DM2 are applied identically at TZR epoch

### Alternative: Use PINT's Delays for TZR

As a quick fix, compute TZR delays using PINT and pass them to JUG:
```python
# In simple_calculator.py:
if use_pint_tzr:
    pint_tzr_delay = model.delay(tzr_toa)[0].to('s').value
    tzr_delay = pint_tzr_delay
```

## DD-Family Implementation Components

The DD model is implemented across three files:

### 1. `jug/delays/binary_dd.py`
- `dd_binary_delay()` - Core DD binary delay calculation
- Kepler equation solver
- Roemer, Einstein, and Shapiro delay components

### 2. `jug/delays/combined.py`
- Branch 2 (DD) calls `dd_binary_delay()`
- Branch 5 (DDK) applies Kopeikin corrections then calls `dd_binary_delay()`
- `t_topo_tdb` calculation at line 94

### 3. `jug/delays/barycentric.py`
- `compute_roemer_delay()` - Simple geometric delay (needs proper motion)
- `compute_shapiro_delay()` - Solar system Shapiro delay
- `compute_pulsar_direction()` - Already supports proper motion, but not used in Roemer calc

## Test Data Locations

- **J0437-4715 (DDK)**:
  - Par: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715_tdb.par`
  - Tim: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715.tim`

- **Other DDK pulsars in MPTA**:
  - J1933-6211
  - J2222-0137

## Environment

```bash
mamba activate discotech
# PINT version: 1.1.4
# Tempo2 version: (system installation)
```
