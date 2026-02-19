# B1855+09 NaN Covariance Bug - Fix Summary

## Date
2026-02-17

## Problem
When fitting B1855+09 (313 TOAs, 123 DMX parameters, ecliptic coordinates LAMBDA/BETA, DE436 ephemeris), the fitter produced **NaN uncertainties** for all parameters. The covariance matrix was entirely NaN, causing the fit to spuriously report convergence without actually updating parameters.

## Root Causes

### 1. Silent NaN from `jnp.linalg.inv()`
**Location**: `jug/fitting/optimized_fitter.py` lines ~2031, ~2085

**Issue**: When computing the covariance matrix as `inv(M^T M)`, JAX's `inv()` can silently return NaN for ill-conditioned matrices without raising an exception. The `pinv()` fallback only triggered on exceptions, so NaN propagated uncaught.

**Fix**: Added explicit NaN checks after `inv()` calls. If NaN detected, fall through to `pinv()` (pseudoinverse).

```python
try:
    cov_all = np.asarray(jnp.linalg.inv(MtM_j))
    # Check for NaN - inv() can silently return NaN for ill-conditioned matrices
    if np.any(np.isnan(cov_all)):
        cov_all = np.asarray(jnp.linalg.pinv(MtM_j))
except Exception:
    cov_all = np.asarray(jnp.linalg.pinv(MtM_j))
```

### 2. NaN in `wls_solve_svd` covariance
**Location**: `jug/fitting/wls_fitter.py` lines ~94-102

**Issue**: When singular values are thresholded to `inf`, the computation `1/Sdiag²` can produce NaN in the covariance matrix.

**Fix**: Added NaN check after covariance computation, with fallback to pseudoinverse approach.

```python
Sigma_ = (VT.T / (Sdiag**2)) @ VT
Sigma = (Sigma_ / Adiag).T / Adiag

# Check for NaN in covariance (can occur with ill-conditioned matrices)
if jnp.any(jnp.isnan(Sigma)):
    # Recompute using pseudoinverse: pinv(M2.T @ M2)
    M2TM2 = M2.T @ M2
    Sigma_ = jnp.linalg.pinv(M2TM2)
    Sigma = (Sigma_ / Adiag).T / Adiag
```

### 3. Silent NaN in parameter updates
**Location**: `jug/fitting/optimized_fitter.py` lines ~2026, ~2083

**Issue**: `jnp.linalg.solve()` can return NaN without exception for singular matrices, causing "step rejected" errors when the real issue is a failed solve.

**Fix**: Added NaN checks after `solve()` calls, falling back to `lstsq()`.

```python
try:
    delta_normalized = np.asarray(jnp.linalg.solve(MtM_j, Mtr_j))
    # Check for NaN - solve() can silently return NaN for singular matrices
    if np.any(np.isnan(delta_normalized)):
        delta_normalized = np.asarray(jnp.linalg.lstsq(MtM_j, Mtr_j, rcond=None)[0])
except Exception:
    delta_normalized = np.asarray(jnp.linalg.lstsq(MtM_j, Mtr_j, rcond=None)[0])
```

## Secondary Issues Fixed

### 4. Mis-parsed Tempo2 noise keywords
**Location**: `jug/io/par_reader.py` line 94

**Issue**: `DMEFAC` and `DMJUMP` lines were not in `_NOISE_KEYWORDS`, so multi-token lines like `DMEFAC -f 430_ASP 1.283` were incorrectly parsed as `params['DMEFAC'] = '-f'` (only first token captured, rest lost).

**Fix**: Added `'DMEFAC'` and `'DMJUMP'` to `_NOISE_KEYWORDS` set. Lines are now captured in `_noise_lines` for future implementation.

**Note**: Actually *applying* DMEFAC/DMJUMP to fits is not yet implemented. Users now see a clear warning when these are detected.

### 5. Missing Tempo2-native red noise detection
**Location**: `jug/engine/noise_mode.py` lines 78-80

**Issue**: `RNAMP`/`RNIDX` (Tempo2-native red noise format) was not detected. Only TempoNest (`TNRedAmp`/`TNRedGam`) and enterprise (`RN_log10_A`/`RN_gamma`) formats were recognized.

**Fix**: Added detection for `RNAMP` and `RNIDX` in `_has_red_noise()`.

**Note**: Applying Tempo2-native red noise is not yet implemented. Users see a warning when detected.

### 6. User warnings for unsupported features
**Location**: `jug/fitting/optimized_fitter.py` lines 1249-1256, 1296-1298

**Fix**: Added console warnings when unsupported-but-parsed keywords are detected:
- `[WARNING] N DMEFAC line(s) found but not yet supported — DM uncertainties will not be scaled`
- `[WARNING] N DMJUMP line(s) found but not yet supported — DM offsets will not be applied`
- `[WARNING] RNAMP/RNIDX (Tempo2-native red noise) detected but not yet supported — red noise will not be applied`

This prevents silent data quality issues.

## Test Results

### Before Fix
```
F0: 1.8649408123545334e+02 ± nan ✗ NaN!
F1: -6.2048467769059996e-16 ± nan ✗ NaN!
Covariance matrix: all NaN
```

### After Fix
```
F0: 1.8649408123538808e+02 ± 6.2918981854492730e-14 ✓
F1: -6.2016343103511880e-16 ± 1.2127942863717042e-21 ✓
Covariance matrix: all finite, no NaN
```

### Test Suite
- **373 tests passed** (same as before fix)
- **1 test failed** (pre-existing, unrelated to changes)
- **0 new failures** introduced by the fix

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `jug/fitting/optimized_fitter.py` | 2026, 2031, 2083, 2085, 1249-1256, 1296-1298 | NaN guards, warnings |
| `jug/fitting/wls_fitter.py` | 95-102 | NaN guard in SVD covariance |
| `jug/io/par_reader.py` | 94 | Parse DMEFAC/DMJUMP |
| `jug/engine/noise_mode.py` | 78-80 | Detect RNAMP/RNIDX |
| `test_b1855_fit.py` | (new file) | Regression test |

## Design Principles

1. **Never inject fake data** — The fix is purely numerical robustness (using `pinv()` for ill-conditioned matrices), not injecting default noise parameters.

2. **Fail gracefully** — Instead of silent NaN propagation, detect and use robust alternatives (`pinv`, `lstsq`).

3. **Warn users** — When unsupported keywords are detected, print clear warnings so users understand data limitations.

4. **Minimal changes** — Only fix the NaN bug and parsing. Don't implement full DMEFAC/DMJUMP/RNAMP application (that's a separate feature).

## Future Work

The following Tempo2-native features are now correctly parsed and warned about, but not yet implemented:

- **DMEFAC**: Scaling DM uncertainties per backend (analogous to T2EFAC for TOA errors)
- **DMJUMP**: Adding DM offsets per frontend (analogous to JUMP for phase)
- **RNAMP/RNIDX**: Applying Tempo2-native red noise (amplitude + spectral index format)

These require additional design work and should be implemented as separate features with proper testing.

## Verification

Run the regression test:
```bash
cd /home/mattm/soft/JUG
python test_b1855_fit.py
```

Expected output: `✓ TEST PASSED - B1855+09 fit produces finite uncertainties`
