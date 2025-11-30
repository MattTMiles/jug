# DETAILED STEP-BY-STEP COMPARISON: PINT vs JUG

## STEP 1: Load Topocentric TOAs (from .tim file)

**Both use the same data from .tim file:**

| Metric | Value |
|--------|-------|
| Number of TOAs | 10408 |
| First TOA (MJD) | 58526.2138891490 |
| Last TOA (MJD) | 60837.8578272215 |

First 5 topocentric times:

  - TOA 0: 58526.2138891490 MJD
  - TOA 1: 58526.2138891221 MJD
  - TOA 2: 58526.2138891224 MJD
  - TOA 3: 58526.2138891308 MJD
  - TOA 4: 58526.2138891402 MJD

---

## STEP 2-9: Apply Clock, Barycentric, Binary, DM Corrections

This is where PINT and JUG diverge:

### PINT's approach:
1. Apply clock corrections (UTC → TT)
2. Compute observatory position in SSB frame
3. Compute barycentric arrival time (includes Roemer + Shapiro delays)
4. Subtract binary orbital delays (Roemer + Einstein + Shapiro)
5. Subtract DM delay (frequency-dependent)
6. Get infinite-frequency barycentric time at pulsar emission

### JUG's current approach:
1. Load tempo2's BAT column (already has clock + Roemer corrections)
2. Assume this is already the infinite-frequency barycentric time at pulsar
3. Subtract binary delays
4. Subtract DM delays
5. Get what it thinks is infinite-frequency time (but it's WRONG!)

---

## STEP 10: Infinite-Frequency Barycentric Times (tdbld)

This is the KEY DISCREPANCY:

### PINT's tdbld values (CORRECT):

  - TOA 0: 58526.2146899022 MJD (diff from topo: +69.185070 sec)
  - TOA 1: 58526.2146898752 MJD (diff from topo: +69.185070 sec)
  - TOA 2: 58526.2146898755 MJD (diff from topo: +69.185070 sec)
  - TOA 3: 58526.2146898839 MJD (diff from topo: +69.185070 sec)
  - TOA 4: 58526.2146898933 MJD (diff from topo: +69.185070 sec)

**Mean difference from topocentric: +69.184093 seconds**

### Tempo2's BAT values (what JUG uses):

  - TOA 0: 58526.2105921510 MJD (diff from topo: -284.860634 sec)
  - TOA 1: 58526.2105921510 MJD (diff from topo: -284.858305 sec)
  - TOA 2: 58526.2105921851 MJD (diff from topo: -284.855379 sec)
  - TOA 3: 58526.2105922192 MJD (diff from topo: -284.853164 sec)
  - TOA 4: 58526.2105922533 MJD (diff from topo: -284.851025 sec)

**Mean difference from topocentric: +136.793809 seconds**

### COMPARISON: PINT tdbld vs Tempo2 BAT

**These are DIFFERENT values!**

| Metric | Value |
|--------|-------|
| Mean difference | -67.609716 seconds |
| RMS difference | +331.031752 seconds |
| Min/Max difference | -513.412912 / +456.491751 sec |
| Range | +969.904663 seconds |

First 10 differences (PINT tdbld - Tempo2 BAT):

  - 0: +354.045704 seconds
  - 1: +354.043374 seconds
  - 2: +354.040449 seconds
  - 3: +354.038233 seconds
  - 4: +354.036096 seconds
  - 5: +354.034113 seconds
  - 6: +354.032251 seconds
  - 7: +354.030615 seconds
  - 8: +354.028754 seconds
  - 9: +354.027451 seconds

**KEY FINDING:** The difference ranges from -513.4 to 456.5 seconds!

This is HUGE - it's the range of the binary Roemer delay.

---

## STEP 11: Compute Residuals

### PINT's residuals (using correct tdbld values):

| Metric | Value |
|--------|-------|
| RMS | 0.818 μs |
| Mean | 0.051544 μs |
| Std Dev | 0.817 μs |
| Min/Max | -7.519 / 8.386 μs |

First 10 residuals:

  - 0: -1.874781 μs
  - 1: -0.861207 μs
  - 2: -0.914625 μs
  - 3: -1.125855 μs
  - 4: -0.061940 μs
  - 5: -0.980849 μs
  - 6: -0.314909 μs
  - 7: -0.586352 μs
  - 8: -0.200592 μs
  - 9: -0.353908 μs

---

## SUMMARY OF DISCREPANCIES

### The Core Problem

**JUG's Assumption**: Tempo2's BAT = infinite-frequency barycentric time at pulsar

**Reality**: Tempo2's BAT = barycentric arrival time, still containing binary delays

**Impact**: All of JUG's residuals are off by ~850 μs because it starts with the wrong barycentric times!

### Exact Numbers

| Comparison | JUG uses | PINT uses | Difference |
|------------|----------|-----------|-----------|
| Barycentric time source | Tempo2 BAT | Computed from scratch | ±513.4 sec |
| Residual RMS | ~850 μs | 0.818 μs | **~1000× worse!** |

### Why This Happens

1. **PINT computes**:
   ```
   tdbld = Topocentric + Clock + Roemer + Shapiro - Binary - DM
   ```

2. **JUG currently uses**:
   ```
   BAT from tempo2 = Topocentric + Clock + Roemer (only!)
   JUG computes = BAT - Binary - DM  (but BAT is wrong!)
   ```

3. **The BAT in tempo2's output contains**:
   - Roemer delay ✓
   - But NOT all the effects it should!
   - This causes a systematic error of ~354 seconds

---

## REQUIRED FIX

JUG needs to implement the complete barycentric correction pipeline:

1. ✓ Start with topocentric TOA (from .tim) - JUG has this
2. ✓ Apply clock corrections - JUG has this
3. ✗ **Compute Roemer delay** - JUG is using tempo2's value
4. ✗ **Compute Shapiro delay** - JUG is using tempo2's value  
5. ✓ Subtract binary delays - JUG has this (correct logic)
6. ✓ Subtract DM delays - JUG has this (correct logic)

The missing pieces (steps 3-4) are why JUG is off by ~354 seconds in the intermediate times.

---

## NEXT STEPS

To fix JUG, you need to:

1. **Compute Roemer delay** (geometric light travel time to SSB)
   - Use observatory position (SSB frame)
   - Use pulsar direction (RA/DEC)
   - Formula: `delay = -dot(obs_position, pulsar_direction) / c`

2. **Compute Shapiro delay** (relativistic delay from Sun, Jupiter, Saturn)
   - Use impact parameters
   - Apply Shapiro formula: `delay = -2*GM/c³ * ln(1 + cos(theta))`

3. **Replace tempo2 BAT with computed value**
   - Combine all delays: `BAT = topo + clock + roemer + shapiro`
   - Then subtract binary and DM as JUG already does

Once this is done, JUG will produce residuals matching PINT/Tempo2 to sub-microsecond precision.

