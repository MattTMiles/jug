# JUG Residual Calculator Speed Optimization Guide

## Summary

This document describes how to create `residual_maker_playground_active_MK2.ipynb` from the optimizations developed in `residual_maker_playground_active_MK1_speedups.ipynb`.

**Performance Results:**
| Version | Time | Speedup vs PINT |
|---------|------|-----------------|
| PINT | ~820 ms | 1x |
| JUG Final | ~1.1 ms | **749x** |

**Accuracy:** 2.5 ns RMS agreement with PINT

---

## Key Components to Include in MK2

The MK2 notebook should include these cells **in this order**:

### Cell 1: Configuration
```python
# === CONFIGURATION ===
PULSAR_NAME = "J1909-3744"
DATA_DIR = "/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb"
N_BENCHMARK_ITERATIONS = 10
```

### Cell 2: Imports and JAX Setup
```python
# === IMPORTS AND SETUP ===
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import jax
import jax.numpy as jnp
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris
import matplotlib.pyplot as plt

# PINT (used only for loading data and TZR reference)
from pint.models import get_model as pint_get_model
from pint.toa import get_TOAs as pint_get_TOAs
from pint.residuals import Residuals

jax.config.update('jax_enable_x64', True)
print(f"JAX {jax.__version__}, Float64: {jax.config.jax_enable_x64}")
print(f"JAX devices: {jax.devices()}")
```

### Cell 3: Constants
```python
# === CONSTANTS ===
SECS_PER_DAY = 86400.0
C_KM_S = 299792.458
T_SUN_SEC = 4.925490947e-6
AU_KM = 149597870.7
K_DM_SEC = 1.0 / 2.41e-4

# Planetary GM/c^3 (seconds)
T_PLANET = {
    'jupiter': 4.702819050227708e-09,
    'saturn':  1.408128810019423e-09,
    'uranus':  2.150589551363761e-10,
    'neptune': 2.537311999186760e-10,
    'venus':   1.205680558494223e-11,
}

# Observatory coordinates (ITRF, km)
OBSERVATORIES = {
    'meerkat': np.array([5109360.133, 2006852.586, -3238948.127]) / 1000,
    'parkes': np.array([-4554231.533, 2816759.046, -3454036.323]) / 1000,
    'gbt': np.array([882589.289, -4924872.368, 3943729.418]) / 1000,
}

HIGH_PRECISION_PARAMS = {'F0', 'F1', 'F2', 'F3', 'PEPOCH', 'TZRMJD', 'POSEPOCH', 'DMEPOCH'}
```

### Cell 4: Parameter Parsing Functions
Copy from MK1 lines 68-126 (Cell 4 `#VSC-1e11217d`):
- `parse_par_file(path)` - parses .par file with high precision
- `get_longdouble(params, key, default)` - returns np.longdouble
- `parse_ra(ra_str)` - parses RA to radians
- `parse_dec(dec_str)` - parses DEC to radians

### Cell 5: Astrometric Functions
Copy from MK1 lines 129-203 (Cell 5 `#VSC-341ba89e`):
- `compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km)` - SSB positions via Astropy/DE440
- `compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, t_mjd)` - unit vector with proper motion
- `compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)` - Roemer + parallax
- `compute_shapiro_delay(obs_body_pos_km, L_hat, T_body)` - Shapiro delay
- `compute_barycentric_freq(freq_topo_mhz, ssb_obs_vel_km_s, L_hat)` - Doppler correction

### Cell 6: Combined JAX Delay Function (CRITICAL)
Copy from MK1 lines 1101-1213 (Cell 16 `#VSC-41e55a8e`):
```python
@jax.jit
def combined_delays(tdbld, freq_bary, obs_sun_pos, L_hat,
                    dm_coeffs, dm_factorials, dm_epoch,
                    ne_sw, fd_coeffs, has_fd,
                    roemer_shapiro, has_binary,
                    pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap):
    """Combined delay calculation - single JAX kernel."""
    # ... (full implementation from MK1)
```

This function combines:
- DM delay (time-varying)
- Solar wind delay
- FD delay
- ELL1 binary delay (third-order eccentricity + aberration correction)

### Cell 7: Total Delay JAX Function (CRITICAL)
Copy from MK1 lines 1591-1626 (Cell 20 `#VSC-7a7f9dce`):
```python
@jax.jit
def compute_total_delay_jax(tdbld, freq_bary, obs_sun, L_hat,
                            dm_coeffs, dm_factorials, dm_epoch,
                            ne_sw, fd_coeffs, has_fd,
                            roemer_shapiro, has_binary,
                            pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap):
    """Compute total delay in a single JAX kernel."""
    combined_sec = combined_delays(
        tdbld, freq_bary, obs_sun, L_hat,
        dm_coeffs, dm_factorials, dm_epoch,
        ne_sw, fd_coeffs, has_fd,
        roemer_shapiro, has_binary,
        pb, a1, tasc, eps1, eps2, pbdot, xdot, gamma, r_shap, s_shap
    )
    return roemer_shapiro + combined_sec
```

### Cell 8: JUGResidualCalculatorFinal Class (THE FASTEST VERSION)
Copy from MK1 lines 1964-2175 (Cell 24 `#VSC-0a6b72d4`):
```python
@dataclass
class JUGResidualCalculatorFinal:
    """Final optimized JUG calculator with pre-computed tdbld_sec."""
    # ... (full implementation)
```

**Key optimizations in this class:**
1. Pre-computes `tdbld_sec_ld = tdbld * SECS_PER_DAY` as longdouble (saves N multiplications)
2. Pre-computes `inv_F0_1e6 = 1e6 / F0` (combines division and multiplication)
3. Pre-computes `F1_half = 0.5 * F1` and `F2_sixth = F2 / 6.0`
4. Uses Horner's method for phase polynomial
5. Uses `np.asarray()` instead of `np.array()` for views
6. Calls `block_until_ready()` for proper JAX synchronization
7. **TZR reference uses PINT once during init** (this gives 2.5 ns accuracy)

### Cell 9: Load Data
```python
# === LOAD DATA ===
par_file = Path(DATA_DIR) / f"{PULSAR_NAME}_tdb.par"
tim_file = Path(DATA_DIR) / f"{PULSAR_NAME}.tim"

pint_model = pint_get_model(str(par_file))
pint_toas = pint_get_TOAs(str(tim_file), model=pint_model)
print(f"Loaded {pint_toas.ntoas} TOAs for {pint_model.PSR.value}")

par_params = parse_par_file(par_file)
obs_itrf_km = OBSERVATORIES.get('meerkat')
```

### Cell 10: Initialize Calculator
```python
# === INITIALIZE JUG CALCULATOR ===
print("Initializing JUG calculator (precomputing astrometry)...")
t_start = time.perf_counter()

jug_calc = JUGResidualCalculatorFinal(
    par_params=par_params,
    pint_model=pint_model,
    pint_toas=pint_toas,
    obs_itrf_km=obs_itrf_km
)

print(f"JUG initialization time: {time.perf_counter() - t_start:.3f} s")
```

### Cell 11: Benchmark and Validate
```python
# === BENCHMARK ===
# Get PINT residuals for comparison
pint_residuals_obj = Residuals(pint_toas, pint_model)
pint_residuals_us = pint_residuals_obj.calc_phase_resids() / pint_model.F0.value * 1e6

# Time JUG
N_ITER = 200
jug_times = []
for _ in range(N_ITER):
    t = time.time()
    jug_residuals = jug_calc.compute_residuals()
    jug_times.append(time.time() - t)

jug_mean = np.mean(jug_times) * 1000
print(f"JUG time: {jug_mean:.3f} ms")

# Time PINT
pint_times = []
for _ in range(10):
    t = time.time()
    _ = Residuals(pint_toas, pint_model).calc_phase_resids()
    pint_times.append(time.time() - t)

pint_mean = np.mean(pint_times) * 1000
print(f"PINT time: {pint_mean:.2f} ms")
print(f"Speedup: {pint_mean/jug_mean:.0f}x")

# Accuracy check
diff_ns = (jug_residuals - pint_residuals_us) * 1000
print(f"RMS difference: {np.std(diff_ns):.3f} ns")
```

---

## Why PINT is Still Used (TZR Reference)

The fastest calculator (`JUGResidualCalculatorFinal`) still uses PINT **once during initialization** for the TZR (timing zero reference) calculation:

```python
pint_tzr_toa = self.pint_model.get_TZR_toa(self.pint_toas)
TZRMJD_TDB = np.longdouble(pint_tzr_toa.table['tdbld'][0])
pint_tzr_delay = float(self.pint_model.delay(pint_tzr_toa).to('s').value[0])
```

**Why?** PINT applies clock corrections (UTC→TDB) with ~28 µs higher precision than Astropy's direct conversion. Without this, residuals have ~180 ns RMS disagreement instead of 2.5 ns.

**Trade-off documented in MK1:**
- Use PINT for TZR: ~1.1 ms, 2.5 ns accuracy ← **RECOMMENDED**
- Fully PINT-free: ~0.83 ms, 180 ns accuracy

---

## Files to Copy From MK1

| MK1 Cell | Lines | Content |
|----------|-------|---------|
| Cell 1 | 2-10 | Configuration |
| Cell 2 | 13-38 | Imports |
| Cell 3 | 41-65 | Constants |
| Cell 4 | 68-126 | Parameter parsing |
| Cell 5 | 129-203 | Astrometric functions |
| Cell 16 | 1101-1213 | `combined_delays()` JAX function |
| Cell 20 | 1591-1626 | `compute_total_delay_jax()` |
| Cell 24 | 1964-2175 | `JUGResidualCalculatorFinal` class |

---

## Quick Copy Instructions

1. Create new notebook `/home/mattm/soft/JUG/residual_maker_playground_active_MK2.ipynb`
2. Copy cells in order: 1, 2, 3, 4, 5, 16, 20, 24 from MK1
3. Add data loading cell
4. Add initialization cell
5. Add benchmark cell

The result will be a clean ~11 cell notebook with the fastest JUG calculator.
