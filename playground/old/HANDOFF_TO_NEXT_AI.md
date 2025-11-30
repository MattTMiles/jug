# Handoff Document for Next AI Agent

**Project:** JUG (JAX-based pulsar timing pipeline)
**Date:** November 28, 2025
**Current Status:** Working version achieves <5 ns, but not fully independent
**Next Goal:** Achieve <5 ns RMS with fully independent SSB computation

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [What Has Been Accomplished](#what-has-been-accomplished)
3. [Current Status](#current-status)
4. [Outstanding Issues](#outstanding-issues)
5. [Technical Background](#technical-background)
6. [File Locations](#file-locations)
7. [What Needs To Be Fixed](#what-needs-to-be-fixed)
8. [How To Verify Success](#how-to-verify-success)
9. [Important Code Sections](#important-code-sections)
10. [Resources and References](#resources-and-references)

---

## Project Overview

### The Goal

Build a **fully independent** pulsar timing pipeline (JUG) that achieves **sub-5 nanosecond RMS agreement** with PINT, without depending on PINT for any computed values.

### Why This Matters

- **Independence:** JUG should be PINT/Tempo2-free for validation and cross-checking
- **Precision:** Pulsar timing requires ~100 ns precision; <5 ns RMS is research-grade
- **Validation:** Independent implementation verifies timing model physics

### The Pipeline

JUG computes pulsar timing residuals through these stages:
1. Parse `.par` (timing parameters) and `.tim` (TOA data) files
2. Apply clock corrections (observatory → UTC → TAI → TT)
3. Compute barycentric corrections (SSB positions, Roemer delay)
4. Apply Solar System Shapiro delays (Sun + 5 planets)
5. Apply binary delays (ELL1 orbital model)
6. Apply DM, solar wind, and FD delays
7. Compute phase residuals

**Key constraint:** All computations must be independent of PINT.

---

## What Has Been Accomplished

### ✅ Successfully Implemented Components

All of these achieve **sub-nanosecond to few-nanosecond agreement** with PINT:

1. **DM Delay:** 0.1 ns RMS ✅ (Perfect)
2. **Solar Wind Delay:** ~0 ns RMS ✅ (Perfect)
3. **FD Delay:** ~0 ns RMS ✅ (Perfect)
4. **Binary Model (ELL1):** 3.8 ns RMS ✅ (Excellent)
5. **Roemer Delay:** 0.8 ns RMS ✅ (Excellent)
6. **Shapiro Delay Formula:** 0.000 ns RMS when using correct positions ✅ (Perfect)
7. **Planetary Shapiro:** Correctly computes Jupiter, Saturn, Venus, Uranus, Neptune

### ✅ Working Version Created

**File:** `residual_maker_playground_MK4_maybe_working.ipynb`

**Achievement:** **<5 ns RMS vs PINT** by using PINT's pre-computed SSB positions

**This proves:**
- All JUG formulas are correct
- The pipeline logic is correct
- The ONLY issue is SSB position computation

### ✅ Code Quality

- Uses JAX for JIT compilation
- Uses `np.longdouble` for phase precision
- Properly handles TCB↔TDB conversions
- Reads tempo2-style clock files
- Implements full timing model (spin, DM, binary, astrometry)

---

## Current Status

### Two Versions Exist

#### 1. JUG Independent (MK3/MK4 Cell #13)
**Method:** Computes SSB positions using Astropy's `get_body_barycentric_posvel()` with DE440

**Result:** ~12 ns RMS vs PINT

**Problem:** SSB positions differ from PINT by ~50 meters → causes 16 ns SS Shapiro error

**Status:** ❌ Does not achieve <5 ns target

#### 2. JUG Working (MK4 Cell #14)
**Method:** Uses PINT's pre-computed SSB positions from `pint_toas.table['ssb_obs_pos']`

**Result:** ~3-4 ns RMS vs PINT ✅

**Problem:** ❌ NOT independent (relies on PINT for SSB computation)

**Status:** ✅ Achieves <5 ns but violates independence requirement

---

## Outstanding Issues

### PRIMARY ISSUE: 50m SSB Position Error

**The Problem:**

JUG's independently computed SSB (Solar System Barycenter) observatory positions differ from PINT's by approximately **50 meters**.

**Impact:**

This 50m error propagates through planetary Shapiro delay calculations:
- Causes **16 ns RMS error** in Solar System Shapiro delays
- Results in **~12 ns total RMS** residual difference vs PINT
- Prevents achieving the <5 ns target independently

**What We've Tried:**

1. ✅ Used jplephem with DE440s ephemeris → Still 50m error
2. ✅ Switched to Astropy's `get_body_barycentric_posvel()` with DE440 → Still 50m error
3. ✅ Verified ITRF→GCRS transformation uses Astropy → Still 50m error

**Conclusion:**

The 50m error persists across multiple approaches. PINT uses a different method or applies corrections that we haven't identified.

### SECONDARY ISSUE: Binary Model Small Differences

**The Problem:**

ELL1 binary delay shows **3.8 ns RMS difference** vs PINT.

**Impact:**

Minor - this is already "excellent" agreement. Not a blocker for <5 ns target once SSB is fixed.

**Possible causes:**
- Numerical precision in orbital equations
- Subtle formula ordering differences
- Different treatment of higher-order terms

**Priority:** LOW (can address after SSB is fixed)

---

## Technical Background

### The 50m SSB Mystery Explained

**What SSB positions are:**

SSB = Solar System Barycenter (center of mass of the solar system)

Observatory position relative to SSB = Earth position + Observatory offset

**How JUG computes it:**

```python
# Get Earth barycentric position
with solar_system_ephemeris.set('de440'):
    earth_pv = get_body_barycentric_posvel('earth', times)
    ssb_geo_pos = earth_pv[0].xyz.to(u.km).value.T

# Get observatory in GCRS
obs_gcrs = obs_itrf.get_gcrs(obstime=times)
geo_obs_pos = [obs_gcrs.cartesian.x, y, z]

# Combine
ssb_obs_pos = ssb_geo_pos + geo_obs_pos
```

**Result:** Positions differ from PINT's by ~50m

**Why it matters for Shapiro:**

Shapiro delay formula: `delay = -2 * (GM/c³) * ln((r - r·L̂) / AU)`

Where `r` = observatory→planet vector

If SSB position is wrong by 50m, then ALL planet vectors are wrong by 50m.

For Jupiter at ~6 AU:
- Fractional error: 50m / 9×10¹¹m ≈ 5×10⁻¹¹
- Accumulates across 5 planets
- Results in **16 ns RMS** total SS Shapiro error

### What PINT Does (that we need to match)

PINT's SSB computation is in:
- `pint/solar_system_ephemerides.py`
- `pint/observatory/topo_obs.py`
- `pint/toa.py` (`compute_posvels()` method)

**Key PINT features we might be missing:**

1. **Ephemeris access method:** PINT might access ephemeris differently than Astropy's standard API
2. **Observatory position:** Might apply corrections we don't know about
3. **Earth orientation:** Might use different EOP (Earth Orientation Parameters)
4. **Velocity computation:** Might compute velocities differently
5. **Coordinate frames:** Might use intermediate frames we're skipping

---

## File Locations

### Main Notebooks

| File | Purpose | Status |
|------|---------|--------|
| `residual_maker_playground_MK3.ipynb` | Independent version (Astropy/DE440) | ~12 ns RMS |
| `residual_maker_playground_MK4_maybe_working.ipynb` | Working + Independent + Comparisons | Working: <5 ns ✅ |

### Documentation

| File | Contents |
|------|----------|
| `HANDOFF_PLANETARY_SHAPIRO_FIX.md` | Previous handoff (outdated claims) |
| `FINAL_STATUS_REPORT.md` | Investigation report from this session |
| `INVESTIGATION_REPORT_12ns_TO_BELOW_5ns.md` | Technical analysis of 12 ns discrepancy |
| `MK4_README.md` | How to use MK4 notebook |
| `CLAUDE.md` | Project overview and architecture |
| **`HANDOFF_TO_NEXT_AI.md`** | **This document** |

### Data Files

| Path | Purpose |
|------|---------|
| `data/ephemeris/de440s.bsp` | JPL DE440s ephemeris (small version, 32 MB) |
| `data/ephemeris/DE440.1950.2050` | Alternative ephemeris (9 MB) |
| `data/clock/*.clk` | Observatory clock corrections (tempo2 format) |
| `data/observatory/observatories.dat` | Observatory ITRF coordinates |

### Test Data

| Path | Purpose |
|------|---------|
| `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par` | Timing model parameters |
| `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim` | TOA data |
| `temp_pre_general2.out` | Tempo2 residuals for comparison |

---

## What Needs To Be Fixed

### THE PRIMARY GOAL

**Fix the SSB computation to match PINT within <1 meter**

This will:
- Eliminate the 16 ns SS Shapiro error
- Reduce total RMS from 12 ns to ~4 ns
- Achieve the <5 ns target independently ✅

### Approach 1: Reverse-Engineer PINT's Exact Method (RECOMMENDED)

**Steps:**

1. **Study PINT source code:**
   ```bash
   # Clone PINT
   git clone https://github.com/nanograv/PINT.git
   cd PINT

   # Key files to examine:
   # - pint/solar_system_ephemerides.py (ephemeris access)
   # - pint/observatory/topo_obs.py (observatory positions)
   # - pint/toa.py (compute_posvels method)
   ```

2. **Trace PINT's SSB computation:**
   - Insert print statements in PINT's code
   - Run PINT on J1909-3744 data
   - Record every intermediate value:
     - Earth position from ephemeris
     - Observatory ITRF coordinates
     - GCRS transformation parameters
     - Final SSB position

3. **Compare with JUG step-by-step:**
   - Create notebook cell that replicates PINT's exact sequence
   - Identify where values diverge
   - Fix JUG to match

4. **Verify:**
   - SSB positions match PINT to <1 m
   - SS Shapiro matches to <1 ns
   - Total RMS <5 ns

### Approach 2: Contact PINT Developers (ALTERNATIVE)

If reverse-engineering is too time-consuming:

1. Email PINT developers (nanograv-software@nanograv.org)
2. Describe the 50m SSB position discrepancy
3. Ask about:
   - Exact ephemeris access method they use
   - Any corrections applied to SSB positions
   - Earth orientation parameters they use
   - Known differences from standard Astropy methods

### Approach 3: Test Hypotheses Systematically

Possible causes to test:

#### Hypothesis 1: Earth Orientation Parameters (EOP)
```python
# Test: Use PINT's EOP data
from astropy.utils import iers
iers.conf.auto_download = True
# Compare with/without IERS data
```

#### Hypothesis 2: Ephemeris Segment Access
```python
# Test: Access ephemeris via jplephem vs Astropy
# PINT might use jplephem internally
from jplephem.spk import SPK
kernel = SPK.open('data/ephemeris/de440s.bsp')
# Compare segment access methods
```

#### Hypothesis 3: Observatory Velocity Computation
```python
# Test: Analytical vs numerical derivatives
# PINT might compute velocities analytically
```

#### Hypothesis 4: Coordinate Frame Chain
```python
# Test: Different transformation paths
# ITRF → GCRS (JUG) vs
# ITRF → something else → GCRS (PINT?)
```

---

## How To Verify Success

### Success Criteria

When SSB is fixed correctly, you should see:

1. ✅ **SSB positions match PINT:** `np.linalg.norm(jug_ssb - pint_ssb) < 1.0` meters
2. ✅ **SS Shapiro matches:** RMS difference < 1 ns
3. ✅ **Total residuals match:** RMS difference < 5 ns
4. ✅ **Independence maintained:** No use of PINT's computed values

### How To Test

```python
# In notebook:
# 1. Compute JUG's SSB positions independently
ssb_obs_pos_jug = compute_ssb_obs_pos_vel(tdbld, obs_itrf_km, ephem_kernel)

# 2. Get PINT's SSB positions
pint_ssb_pos = pint_toas.table['ssb_obs_pos'].value

# 3. Compare
position_diff = np.linalg.norm(ssb_obs_pos_jug - pint_ssb_pos, axis=1)
print(f"SSB position difference: {position_diff.mean():.1f} m")

# Success if: position_diff.mean() < 1.0 meters
```

### Verification Checklist

- [ ] SSB position difference < 1 m (currently ~50 m)
- [ ] SS Shapiro RMS < 1 ns (currently 16 ns)
- [ ] Planetary Shapiro shows ~0 ns for each planet
- [ ] Total residual RMS < 5 ns (currently 12 ns)
- [ ] No use of `pint_toas.table['ssb_obs_pos']` or similar
- [ ] Code passes with PINT installed but not used for computations

---

## Important Code Sections

### Key Function: `compute_ssb_obs_pos_vel()`

**Location:** MK3/MK4 Cell #5 (ID: `ca6b26f5`)

**Current implementation:**
```python
def compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, ephem_kernel):
    times = Time(tdb_mjd, format='mjd', scale='tdb')

    # Get Earth position - THIS IS WHERE THE 50m ERROR OCCURS
    with solar_system_ephemeris.set('de440'):
        earth_pv = get_body_barycentric_posvel('earth', times)
        ssb_geo_pos = earth_pv[0].xyz.to(u.km).value.T
        ssb_geo_vel = earth_pv[1].xyz.to(u.km/u.s).value.T

    # Observatory GCRS position - THIS MIGHT ALSO BE WRONG
    obs_gcrs = obs_itrf.get_gcrs(obstime=times)
    geo_obs_pos = [obs_gcrs.cartesian.x/y/z]

    # Combine - ERROR ACCUMULATES HERE
    ssb_obs_pos = ssb_geo_pos + geo_obs_pos

    return ssb_obs_pos, ssb_obs_vel
```

**What needs to change:**

The implementation needs to match PINT's method exactly. Study PINT's code to find what's different.

### Key Function: Planetary Shapiro

**Location:** MK3/MK4 Cell #13 (ID: `b4778961`)

**Current implementation:**
```python
if planet_shapiro_enabled:
    times = Time(tdbld, format='mjd', scale='tdb')
    with solar_system_ephemeris.set('de440'):
        for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
            # Get planet position - USES SAME BROKEN SSB METHOD
            planet_pv = get_body_barycentric_posvel(planet, times)
            ssb_planet_km = planet_pv[0].xyz.to(u.km).value.T
            obs_planet_pos_km = ssb_planet_km - ssb_obs_pos_km

            # Shapiro formula is CORRECT
            r = sqrt(sum(obs_planet_pos_km²))
            rcostheta = obs_planet_pos_km · L_hat
            delay = -2 * T_PLANET[planet] * ln((r - rcostheta) / AU)
```

**Note:** The formula is correct! Only the input positions are wrong.

### Working Version Reference

**Location:** MK4 Cell #14 (ID: `working_version`)

**Key lines:**
```python
# This achieves <5 ns by using PINT's SSB positions
ssb_obs_pos_km_work = pint_toas.table['ssb_obs_pos'].value  # NOT INDEPENDENT
obs_sun_pos_km_work = pint_toas.table['obs_sun_pos'].value   # NOT INDEPENDENT

# Uses PINT's planet positions too
planet_pos_km = pint_toas.table[f'obs_{planet}_pos'].quantity.to(u.km).value
```

**Goal:** Replace these with independent computations that produce identical values.

---

## Resources and References

### PINT Documentation

- GitHub: https://github.com/nanograv/PINT
- Docs: https://nanograv-pint.readthedocs.io/
- Key papers:
  - Luo et al. 2021 (ApJ 911, 45): PINT description
  - Hobbs et al. 2006: Tempo2 (for comparison)

### Ephemeris References

- JPL Horizons: https://ssd.jpl.nasa.gov/horizons/
- DE440/DE441 documentation: https://ssd.jpl.nasa.gov/doc/de440.html
- Astropy ephemeris: https://docs.astropy.org/en/stable/coordinates/solarsystem.html

### Pulsar Timing Theory

- Lorimer & Kramer 2004: "Handbook of Pulsar Astronomy"
- Edwards, Hobbs & Manchester 2006: Tempo2 paper
- Damour & Deruelle 1986: Binary timing models

### Coordinate Systems

- IERS Conventions: https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html
- Astropy coordinates: https://docs.astropy.org/en/stable/coordinates/

---

## Debugging Strategy

### Step-by-Step Debug Plan

1. **Add detailed logging to JUG's SSB computation:**
   ```python
   print(f"Earth pos from ephemeris: {ssb_geo_pos[0]}")
   print(f"Observatory ITRF: {obs_itrf_km}")
   print(f"Observatory GCRS: {geo_obs_pos[0]}")
   print(f"Final SSB pos: {ssb_obs_pos[0]}")
   ```

2. **Add same logging to PINT:**
   - Modify PINT source to print intermediate values
   - Or use Python debugger to inspect values

3. **Compare line by line:**
   - Find first divergence point
   - Investigate why values differ
   - Fix JUG to match

4. **Test incrementally:**
   - Test each fix in isolation
   - Verify SSB difference decreases
   - Don't move forward until each step matches

### Quick Test

Create this diagnostic cell:
```python
# Compare SSB computation methods
print("="*80)
print("SSB POSITION DIAGNOSTIC")
print("="*80)

# Method 1: JUG independent
jug_ssb = compute_ssb_obs_pos_vel(tdbld[:1], obs_itrf_km, ephem_kernel)[0][0]

# Method 2: PINT stored
pint_ssb = pint_toas.table['ssb_obs_pos'].value[0]

# Compare
diff = np.linalg.norm(jug_ssb - pint_ssb) * 1000  # Convert to meters
print(f"\nSSB position difference: {diff:.1f} meters")
print(f"JUG:  {jug_ssb}")
print(f"PINT: {pint_ssb}")
print(f"Diff: {jug_ssb - pint_ssb} km")

if diff < 1.0:
    print("\n✅ SUCCESS: SSB positions match!")
else:
    print(f"\n❌ PROBLEM: {diff:.1f}m difference (target: <1m)")
```

Run this after each attempted fix to track progress.

---

## Common Pitfalls to Avoid

### ❌ DON'T: Use PINT's values as inputs

```python
# WRONG - This defeats the purpose
ssb_pos = pint_toas.table['ssb_obs_pos'].value
```

Even if it achieves <5 ns, it's not independent.

### ❌ DON'T: Assume Astropy = PINT

PINT might use Astropy internally but with different settings, parameters, or methods.

### ❌ DON'T: Ignore small differences

A "small" 1 meter SSB error → ~0.3 ns Shapiro error for Jupiter.
You need <1 meter precision.

### ❌ DON'T: Fix symptoms instead of root cause

Don't try to "correct" the Shapiro delays with empirical factors. Fix the SSB positions.

### ✅ DO: Test against multiple pulsars

J1909-3744 is the current test case, but verify fixes work for other pulsars too.

### ✅ DO: Maintain precision

Use `np.longdouble` for accumulated phase, `float64` for delays, proper unit handling.

### ✅ DO: Document what you find

Update this handoff document with discoveries about PINT's methods.

---

## Expected Timeline

**If PINT's method is well-documented:** 4-8 hours
- 2 hours: Study PINT source code
- 2 hours: Implement matching method
- 2 hours: Debug and verify
- 2 hours: Test on multiple pulsars

**If PINT's method requires deep investigation:** 1-3 days
- Study PINT internals thoroughly
- Test multiple hypotheses
- Potentially contact PINT developers

**If PINT uses non-standard methods:** Could take longer
- May need to understand PINT's full architecture
- May need PINT developer assistance

---

## Questions for Next Session

When you start working on this, first answer:

1. **Is PINT using standard Astropy methods?**
   - Check PINT's imports and function calls
   - See if they patch/modify Astropy behavior

2. **What ephemeris does PINT actually load?**
   - Check `pint_toas.ephem` or similar
   - Verify it's DE440 (not DE440s, DE421, etc.)

3. **Does PINT apply any corrections JUG doesn't?**
   - Gravitational deflection?
   - Atmospheric refraction?
   - Aberration corrections?

4. **What coordinate frames does PINT use?**
   - Direct ITRF→GCRS?
   - Or ITRF→CIRS→GCRS?
   - Or something else?

5. **Are velocities computed or extracted?**
   - PINT might get velocities directly from ephemeris
   - JUG computes numerically - is this the issue?

---

## Success Looks Like

When you're done, the user should be able to:

1. Run MK4 notebook
2. See JUG Independent version achieve <5 ns RMS
3. See component breakdown showing:
   - SS Shapiro: <1 ns RMS ✅
   - Total residuals: <5 ns RMS ✅
4. Verify no PINT values are used as inputs
5. Cross-check with Tempo2 (similar agreement)

**Final deliverable:** Update MK4 so Cell #13 (JUG Independent) achieves <5 ns RMS.

---

## Good Luck!

The groundwork is done. All timing formulas are correct. The ONLY issue is the 50m SSB position error.

Fix that, and you'll have a fully independent, research-grade pulsar timing pipeline.

**Key insight to remember:** The "working" version (Cell #14) proves the formulas are right. You just need to compute the inputs correctly.

---

## Contact Information

If you need to discuss this with the user:
- User is familiar with pulsar timing
- Has access to PINT, Tempo2, and all necessary data files
- Wants full independence while maintaining <5 ns precision
- Is willing to trade some dev time for a proper solution

**Most important:** Don't be afraid to ask PINT developers for help. They're generally responsive and helpful to people implementing independent timing packages.
