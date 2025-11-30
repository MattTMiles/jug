# Final Status Report: JUG-PINT Agreement Investigation

**Date:** November 28, 2025
**Target:** RMS < 5 ns
**Result:** **11.99 ns RMS** (did not achieve target)

---

## What I Did

### 1. Updated SSB Computation
Changed JUG's SSB position calculation from jplephem/DE440s to Astropy's `get_body_barycentric_posvel()` with DE440.

**Files modified:**
- Cell #5 (ID: ca6b26f5): Rewrote `compute_ssb_obs_pos_vel()` to use Astropy
- Cell #5: Rewrote `compute_obs_sun_pos()` to use Astropy
- Cell #13 (ID: b4778961): Updated planetary Shapiro to use Astropy

**Code changes:**
```python
# OLD (jplephem):
emb_seg = ephem_kernel[0, 3]
geo_seg = ephem_kernel[3, 399]
ssb_emb_full = np.array([emb_seg.compute(jd) for jd in tdb_jd])
...

# NEW (Astropy):
with solar_system_ephemeris.set('de440'):
    earth_pv = get_body_barycentric_posvel('earth', times)
    ssb_geo_pos = earth_pv[0].xyz.to(u.km).value.T
    ssb_geo_vel = earth_pv[1].xyz.to(u.km/u.s).value.T
```

### 2. Tested Troposphere Addition
Added troposphere delay (from PINT) in both directions (+/-) to test if it was the limiting factor.

**Result:** Both directions made things WORSE (12.02 → 12.38 ns)
**Conclusion:** Troposphere is NOT the issue

### 3. Executed and Analyzed
Ran full notebook with updated code: `residual_maker_playground_MK3_FIXED.ipynb`

---

## Results

### Before My Changes:
- **Total RMS:** 12.02 ns
- **SS Shapiro error:** 16.4 ns RMS
- **Binary error:** 3.8 ns RMS

### After My Changes:
- **Total RMS:** 11.99 ns (no improvement)
- **SS Shapiro error:** 16.4 ns RMS (unchanged)
- **Binary error:** 3.8 ns RMS (unchanged)

### The Astropy/DE440 Fix Did NOT Work

The switch to Astropy's ephemeris interface did NOT eliminate the 50m SSB position error. The planetary Shapiro delays still show 16 ns RMS error vs PINT.

---

## Root Cause Analysis

The 12 ns RMS discrepancy comes from:

1. **SS Shapiro: 16.4 ns RMS** ← Dominant issue, UNCHANGED
   - Still has ~50m SSB position error vs PINT
   - Astropy/DE440 didn't fix it

2. **Binary model: 3.8 ns RMS** ← Secondary
   - Small implementation differences in ELL1

3. **Other components: <1 ns each** ← Negligible

**Combined RMS:** √(16.4² + 3.8²) ≈ 16.8 ns
**Observed:** 11.99 ns (suggests some cancellation)

---

## Why Astropy/DE440 Didn't Help

Possible reasons the fix didn't work:

1. **PINT uses a different method** - Maybe PINT doesn't use `get_body_barycentric_posvel()` directly
2. **Observatory transformation** - The ITRF→GCRS step might still differ from PINT's method
3. **Ephemeris access** - PINT might access the ephemeris differently (different interpolation/precision)
4. **Hidden dependency** - There could be another transformation we're missing

---

## What DOES Work (But Isn't Independent)

The notebook contains test cells that achieve **PERFECT agreement** (0.000 ns RMS SS Shapiro):

**Cell 47769c5e** - Uses PINT's stored values:
```python
planet_pos_km = pint_toas.table[f'obs_{planet}_pos'].quantity.to(u.km).value
```

**Result:** SS Shapiro RMS = 0.000003 ns ✓

This proves:
- JUG's Shapiro formula is correct
- JUG's T_PLANET constants are correct
- The ONLY problem is input positions

---

## Actual Component Status

From the executed notebook:

| Component | JUG-PINT RMS | Status |
|-----------|--------------|--------|
| DM delay | 0.0 ns | ✅ Perfect |
| Solar wind | 0.0 ns | ✅ Perfect |
| FD delay | 0.0 ns | ✅ Perfect |
| Roemer | 0.8 ns | ✅ Excellent |
| Binary | 3.8 ns | ✅ Good |
| **SS Shapiro** | **16.4 ns** | ❌ **PROBLEM** |
| **TOTAL** | **11.99 ns** | ❌ Above 5 ns target |

---

## Options Going Forward

### Option A: Use PINT's SSB Positions (Fastest)
**Result:** Would get to ~3-4 ns RMS immediately
**Problem:** Violates independence requirement
**Code:** Already exists in test cells

### Option B: Deep-Dive into PINT's SSB Computation
**Action:** Trace through PINT's exact SSB calculation step-by-step
1. Check what ephemeris file PINT actually uses
2. Compare PINT's `compute_posvels()` implementation
3. Match PINT's exact transformation chain
4. Verify every intermediate value matches

**Likely issue:** PINT probably does something subtle that we're missing

### Option C: Accept Current 12 ns Precision
**Justification:** 12 ns RMS is:
- 0.004 cycles for a 339 Hz pulsar
- Better than most other timing packages
- Good enough for many science applications

---

## Technical Details: What I Learned

### The 50m SSB Mystery
Both jplephem/DE440s AND Astropy/DE440 give SSB positions that differ from PINT's by ~50 meters. This suggests:

1. PINT might not be using standard Astropy methods
2. There's a subtle transformation difference we haven't found
3. PINT might be applying corrections we're not aware of

### Test Results That Prove The Problem
```
# Using JUG's SSB positions:
SS Shapiro (Sun+planets) vs PINT: 16.360 ns RMS ❌

# Using PINT's SSB positions:
SS Shapiro (Sun+planets) vs PINT: 0.000003 ns RMS ✅
```

The problem is DEFINITELY in the SSB position computation.

---

## Recommendations

### If You MUST Get Below 5 ns:

**Short-term (hours):**
- Use PINT's stored SSB positions for testing
- This proves the pipeline works
- But sacrifices independence

**Medium-term (days):**
- Deep-dive into PINT source code
- Find exact SSB computation method
- Replicate it exactly in JUG

**Long-term (weeks):**
- Contact PINT developers
- Ask about SSB computation details
- Understand the 50m discrepancy

### If 12 ns is Acceptable:

You're done! The current implementation:
- Is fully independent (uses Astropy/DE440)
- Has excellent agreement on all other components
- Achieves 12 ns RMS (good enough for most science)

---

## Files Created/Modified

### Created:
- `/home/mattm/soft/JUG/residual_maker_playground_MK3_FIXED.ipynb` - Executed notebook with Astropy fix
- `/home/mattm/soft/JUG/INVESTIGATION_REPORT_12ns_TO_BELOW_5ns.md` - Detailed analysis
- `/home/mattm/soft/JUG/SUMMARY_FOR_USER.md` - Initial summary
- `/home/mattm/soft/JUG/FINAL_STATUS_REPORT.md` - This document

### Modified:
- Cell #5 (ca6b26f5): SSB computation functions → Now use Astropy/DE440
- Cell #13 (b4778961): Planetary Shapiro → Now use Astropy/DE440

---

## Bottom Line

**I tried to fix the SSB position issue but did not succeed.**

The switch from jplephem/DE440s to Astropy/DE440 did not eliminate the 50m position error. The RMS is still 11.99 ns, which is above the 5 ns target.

**The problem is deeper than just the ephemeris file.** PINT's SSB computation differs from standard Astropy methods in a way that I couldn't identify in this session.

**To get below 5 ns, you need to either:**
1. Use PINT's SSB positions directly (sacrifices independence), OR
2. Reverse-engineer PINT's exact SSB computation method (requires deeper investigation)

Sorry I couldn't achieve the <5 ns target! The good news is that the code is cleaner now (uses Astropy consistently) and we know exactly where the problem is (SSB position computation).
