# TDB Calculation Precision Fix - Handoff Document

## Overview

JUG needs a standalone TDB (Barycentric Dynamical Time) calculation that matches PINT exactly. We have achieved **99.97% exact matches** (10,405/10,408 TOAs), but **3 outliers remain with ~628 ns differences**.

The root cause is a **numerical precision cliff** in Astropy's TT→TDB conversion where tiny (~2.7 ns) UTC differences get amplified **232x** to ~628 ns in TDB.

## Current Status

| Metric | Value |
|--------|-------|
| Total TOAs | 10,408 |
| Exact matches (< 0.001 ns) | 10,405 (99.97%) |
| Outliers (~628 ns) | 3 |
| Outlier indices | 10191, 10263, 10321 |
| Root cause | jd1/jd2 normalization difference → precision cliff |

## Root Cause Analysis

### The Problem

1. **MJD Parsing Difference**: PINT and our code parse the raw MJD string slightly differently, leading to different `jd1/jd2` splits:
   - Our code: `jd1 = 2460804.5, jd2 = +0.082...`
   - PINT: `jd1 = 2460805.0, jd2 = -0.417...`

2. **Same UTC, Different Representation**: Both represent the same UTC instant, but with different internal jd1/jd2 normalization.

3. **Clock Correction Application**: When we add clock corrections as a `TimeDelta`, tiny numerical differences (~2.7 ns) appear in the corrected `jd2` values.

4. **Precision Cliff**: At certain `jd2` values near specific thresholds, Astropy's TT→TDB conversion has a numerical precision discontinuity. A 2.7 ns UTC difference becomes 628 ns in TDB (232x amplification).

### Evidence

```python
# For outlier TOA 10191:
Our corrected jd2:  -0.41766675593965730
PINT corrected jd2: -0.41766675593962599

UTC jd2 diff: -2.705 ns
TDB diff: -628.643 ns
Amplification factor: 232.4x
```

## The Solution

To match PINT exactly, we must use **PINT's exact MJD parsing method** which uses the `pulsar_mjd` format internally. The key is ensuring identical `jd1/jd2` values before any conversions.

### PINT's MJD Parsing Flow

PINT uses this sequence (from `src/pint/pulsar_mjd.py`):

1. Parse MJD string to `(integer, fraction)` 
2. Convert to calendar date using ERFA: `erfa.jd2cal(DJM0 + int, frac)`
3. Convert fractional day to `(hour, minute, second)`
4. Convert back to JD using `erfa.dtf2d("UTC", y, mo, d, h, m, s)`
5. This produces properly normalized `jd1/jd2` that handle leap seconds

### Key Function to Replicate

From PINT's `pulsar_mjd.py`:

```python
def mjds_to_jds_pulsar(mjd1, mjd2):
    """Convert MJD to JD using pulsar timing convention."""
    # Normalize the MJD split
    v1, v2 = day_frac(mjd1, mjd2)
    
    # Convert to calendar date
    y, mo, d, frac = erfa.jd2cal(erfa.DJM0 + v1, v2)
    
    # Convert fractional day to h:m:s (using 86400s days)
    frac = frac * 24
    h = int(np.floor(frac))
    frac = frac - h
    frac = frac * 60
    m = int(np.floor(frac))
    frac = frac - m
    frac = frac * 60
    s = frac
    
    # Convert back to JD with proper UTC leap second handling
    jd1, jd2 = erfa.dtf2d("UTC", y, mo, d, h, m, s)
    
    return jd1, jd2
```

## Implementation Tasks

### Task 1: Match PINT's jd1/jd2 Normalization

The issue is that after we create a Time object and add clock corrections, our `jd1/jd2` values differ slightly from PINT's.

**Approach A - Renormalize after clock correction:**
```python
def renormalize_time_like_pint(time_obj):
    """
    Renormalize a Time object's jd1/jd2 to match PINT's convention.
    
    PINT normalizes such that jd2 is in range [-0.5, 0.5).
    """
    # Get the total JD
    jd_total = time_obj.jd1 + time_obj.jd2
    
    # Renormalize: jd1 = nearest integer, jd2 = remainder
    jd1_new = np.round(jd_total)
    jd2_new = jd_total - jd1_new
    
    # Create new Time object with renormalized values
    return Time(jd1_new, jd2_new, format='jd', scale=time_obj.scale, 
                location=time_obj.location)
```

**Approach B - Use PINT's exact pulsar_mjd functions:**
```python
import erfa

def day_frac(val1, val2):
    """Normalize val1 + val2 to (integer, fraction) with frac in [-0.5, 0.5)."""
    sum_val = val1 + val2
    d = np.round(sum_val)
    f = sum_val - d
    return d, f

def mjd_to_jd_pint_style(mjd_int, mjd_frac):
    """
    Convert MJD to JD using PINT's pulsar_mjd convention.
    This ensures jd1/jd2 values match PINT exactly.
    """
    # Step 1: Normalize the MJD split
    v1, v2 = day_frac(float(mjd_int), mjd_frac)
    
    # Step 2: Convert to calendar date using ERFA
    y, mo, d, frac = erfa.jd2cal(erfa.DJM0 + v1, v2)
    
    # Step 3: Convert fractional day to h:m:s
    frac = frac * 24
    h = int(np.floor(frac))
    frac = frac - h
    frac = frac * 60
    m = int(np.floor(frac))
    frac = frac - m
    frac = frac * 60
    s = frac
    
    # Step 4: Convert to JD with proper leap second handling
    jd1, jd2 = erfa.dtf2d("UTC", y, mo, d, h, m, s)
    
    return jd1, jd2
```

### Task 2: Modify Clock Correction Application

The clock correction must be added in a way that preserves PINT's normalization:

```python
def apply_clock_correction_pint_style(mjd_int, mjd_frac, clock_corr_seconds, location):
    """
    Apply clock correction matching PINT's method exactly.
    
    Parameters
    ----------
    mjd_int : int
        Integer part of MJD
    mjd_frac : float
        Fractional part of MJD
    clock_corr_seconds : float
        Total clock correction in seconds (BIPM_small + MK + GPS)
    location : EarthLocation
        Observatory location
    
    Returns
    -------
    Time
        Clock-corrected UTC time with PINT-compatible jd1/jd2
    """
    # Add clock correction to fractional MJD (in days)
    clock_corr_days = clock_corr_seconds / 86400.0
    mjd_frac_corrected = mjd_frac + clock_corr_days
    
    # Handle overflow (frac might exceed 1.0)
    extra_days = int(np.floor(mjd_frac_corrected))
    mjd_int_corrected = mjd_int + extra_days
    mjd_frac_corrected = mjd_frac_corrected - extra_days
    
    # Convert to JD using PINT's method
    jd1, jd2 = mjd_to_jd_pint_style(mjd_int_corrected, mjd_frac_corrected)
    
    # Create Time object
    return Time(jd1, jd2, format='jd', scale='utc', location=location)
```

### Task 3: Test the Fix

```python
def test_tdb_calculation():
    """Verify all 10,408 TOAs match PINT within 0.001 ns."""
    
    # Load PINT reference values
    # ... (load pint_toas_with_mk)
    
    # Calculate our TDB values with the fix
    our_tdb = []
    for t in parsed_toas:
        # Get clock corrections
        bipm_small = interpolate_clock(bipm_clock, t.mjd) - 32.184
        mk_corr = interpolate_clock(mk_clock, t.mjd)
        gps_corr = interpolate_clock(gps_clock, t.mjd)
        total_corr = bipm_small + mk_corr + gps_corr
        
        # Apply using PINT-style method
        corrected_time = apply_clock_correction_pint_style(
            t.mjd_int, t.mjd_frac, total_corr, mk_location
        )
        
        our_tdb.append(corrected_time.tdb.mjd)
    
    # Compare
    diff_ns = (np.array(our_tdb) - pint_tdb_with_mk) * 86400e9
    
    exact_matches = np.sum(np.abs(diff_ns) < 0.001)
    assert exact_matches == len(our_tdb), f"Only {exact_matches}/{len(our_tdb)} match"
    print("✅ All TOAs match PINT within 0.001 ns!")
```

## Files to Reference

### Working Notebook
`/home/mattm/soft/JUG/TDB_calculation_standalone.ipynb`
- Contains all experiments and comparisons
- Has working clock correction functions
- Shows the precision cliff investigation

### Key Clock Files
- BIPM: `/home/mattm/soft/JUG/data/clock/tai2tt_bipm2024.clk` (gives TT-TAI)
- MeerKAT: `/home/mattm/soft/JUG/data/clock/mk2utc.clk` (updated, MJD 58484-60994)
- GPS: From PINT cache or TEMPO2 directory

### Test Data
- Par file: `/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744_tdb.par`
- Tim file: `/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim`

### PINT Source Reference
- `PINT/src/pint/pulsar_mjd.py` - MJD parsing logic
- `PINT/src/pint/observatory/topo_obs.py` - Clock correction application

## Integration into JUG Main Code

Once the fix is validated:

### Step 1: Create TDB Module

Create `/home/mattm/soft/JUG/src/jug/tdb.py`:

```python
"""
Standalone TDB calculation for JUG.
Matches PINT exactly without requiring PINT as a dependency.
"""

import numpy as np
import erfa
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation
import astropy.units as u
from pathlib import Path
from bisect import bisect_left


# === Clock File Handling ===

def parse_clock_file(path):
    """Parse tempo2-style clock correction file."""
    mjds, offsets = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mjds.append(float(parts[0]))
                    offsets.append(float(parts[1]))
                except ValueError:
                    continue
    return {'mjd': np.array(mjds), 'offset': np.array(offsets)}


def interpolate_clock(clock_data, mjd):
    """Linear interpolation of clock correction at given MJD."""
    mjds, offsets = clock_data['mjd'], clock_data['offset']
    if len(mjds) == 0:
        return 0.0
    if mjd <= mjds[0]:
        return offsets[0]
    if mjd >= mjds[-1]:
        return offsets[-1]
    idx = bisect_left(mjds, mjd)
    mjd0, mjd1 = mjds[idx-1], mjds[idx]
    off0, off1 = offsets[idx-1], offsets[idx]
    frac = (mjd - mjd0) / (mjd1 - mjd0)
    return off0 + frac * (off1 - off0)


# === PINT-Compatible Time Handling ===

def day_frac(val1, val2):
    """Normalize to (integer, fraction) with frac in [-0.5, 0.5)."""
    sum_val = val1 + val2
    d = np.round(sum_val)
    f = sum_val - d
    return float(d), float(f)


def mjd_to_jd_pint_style(mjd_int, mjd_frac):
    """Convert MJD to JD using PINT's pulsar_mjd convention."""
    v1, v2 = day_frac(float(mjd_int), mjd_frac)
    y, mo, d, frac = erfa.jd2cal(erfa.DJM0 + v1, v2)
    
    frac = frac * 24
    h = int(np.floor(frac))
    frac = frac - h
    frac = frac * 60
    m = int(np.floor(frac))
    frac = frac - m
    frac = frac * 60
    s = frac
    
    jd1, jd2 = erfa.dtf2d("UTC", y, mo, d, h, m, s)
    return jd1, jd2


def compute_tdb(mjd_int, mjd_frac, clock_corr_seconds, location):
    """
    Compute TDB MJD from UTC MJD with clock corrections.
    Matches PINT exactly.
    
    Parameters
    ----------
    mjd_int : int
        Integer part of UTC MJD
    mjd_frac : float
        Fractional part of UTC MJD
    clock_corr_seconds : float
        Total clock correction in seconds
    location : EarthLocation
        Observatory location
    
    Returns
    -------
    float
        TDB MJD value
    """
    # Add clock correction
    clock_corr_days = clock_corr_seconds / 86400.0
    mjd_frac_corr = mjd_frac + clock_corr_days
    
    # Handle overflow
    extra_days = int(np.floor(mjd_frac_corr))
    mjd_int_corr = mjd_int + extra_days
    mjd_frac_corr = mjd_frac_corr - extra_days
    
    # Convert using PINT method
    jd1, jd2 = mjd_to_jd_pint_style(mjd_int_corr, mjd_frac_corr)
    
    # Create Time and convert to TDB
    t = Time(jd1, jd2, format='jd', scale='utc', location=location)
    return t.tdb.mjd
```

### Step 2: Update JUG's TOA Processing

Integrate `compute_tdb()` into the main TOA processing pipeline, replacing any PINT dependency for TDB calculation.

### Step 3: Add Tests

Create unit tests that compare against PINT for validation:

```python
def test_tdb_matches_pint():
    """Regression test: TDB must match PINT within 0.001 ns."""
    # Load test data
    # Compare all TOAs
    # Assert exact matches
```

## Verification Checklist

- [ ] All 10,408 TOAs match PINT within 0.001 ns
- [ ] The 3 previous outliers (10191, 10263, 10321) now match
- [ ] Clock corrections are applied correctly (BIPM_small + MK + GPS)
- [ ] Observatory location is used for TDB conversion
- [ ] Works for different observatories (not just MeerKAT)

## Contact / Questions

The working implementation is in the notebook. Key variables in kernel:
- `pint_toas_with_mk` - PINT's TOAs with updated MK clock
- `parsed_toas` - Our parsed TOAs
- `mk_clock_updated` - Updated MK clock data
- `bipm_clock`, `gps_clock` - Other clock corrections

The precision cliff is documented in cells 145-152 of the notebook.
