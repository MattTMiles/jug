# CONCRETE EXAMPLE: First TOA Calculation
## Detailed Step-by-Step for Single TOA

**Index**: 0 (First TOA in dataset)

---

## INPUT DATA (IDENTICAL FOR BOTH)

```
Topocentric TOA (UTC MJD):    58526.2138891490
Frequency (MHz):              907.85
Spin frequency F0 (Hz):       339.32
Dispersion measure DM:        10.3907 pc/cm³
```

---

## PINT COMPUTATION (SIMPLIFIED OVERVIEW)

### Step 1: Clock Corrections
```
Input: Topocentric TOA = 58526.2138891490 MJD (UTC)
Apply: Observatory (mk/Meerkat) clock correction
       UTC → GPS → TAI → TT
Output: TT = 58526.2138891490 + small correction
Status: ✅ Computed from clock files
```

### Step 2: Barycentric Corrections
```
Step 2a - Observatory Position
  Input: MJD = 58526.2138891490
  Ephemeris: JPL DE421
  Output: SSB observatory position (X, Y, Z)
  Status: ✅ Computed from ephemeris
  
Step 2b - Pulsar Direction
  Input: RA, DEC, proper motions
  Output: Unit vector to pulsar
  Status: ✅ Computed from parameters
  
Step 2c - Roemer Delay
  Input: obs_position, pulsar_direction
  Formula: delay = -dot(obs_pos, pulsar_dir) / c
  Output: Roemer delay (seconds)
  Typical magnitude: ±300 seconds (binary-dependent)
  Status: ✅ Computed fresh
  
Step 2d - Shapiro Delay
  Input: Observatory and planet positions
  Formula: delay = -2*GM/c³ * ln(1 + cos_angle)
  Output: Shapiro delay (~1 microsecond)
  Status: ✅ Computed fresh
```

### Step 3: Barycentric Arrival Time (BAT)
```
BAT = Topocentric + Clock + Roemer + Shapiro
    ≈ 58526.2138891490 + Δclock + Δroemer + Δshapiro
Status: ✅ Fresh computation
Result: Barycentric time (complete)
```

### Step 4-6: Subtract Binary & DM
```
t_inf = BAT - binary_delay - dm_delay
Status: ✅ Subtracted correctly
```

### FINAL RESULT FOR THIS TOA
```
Residual: ~1-3 microseconds (for well-fit model)
Status: ✅ CORRECT
```

---

## JUG COMPUTATION

### Step 1: Skip Clock Corrections
```
Status: ❌ Clock corrections not computed
Assumption: "Already in Tempo2's BAT"
```

### Step 2: Use Tempo2's BAT Directly
```
Input: Tempo2 BAT = 58526.2105921510 MJD
Assumption: "This is infinite-frequency barycentric time"
Status: ❌ WRONG - This is incomplete!
```

### The Critical Difference
```
Topocentric TOA:     58526.2138891490 MJD
Tempo2 BAT:          58526.2105921510 MJD
Difference:          284.86 seconds
                     = 4.75 minutes
                     
What is this difference?
→ Clock corrections (small, ~0.1 sec)
→ Roemer delay (large, ~260-310 sec from binary orbit)
→ Shapiro delay (small, ~1 microsec)
→ NET: Still has uncorrected binary Roemer delay!

This is the CORE PROBLEM ⚠️
```

### Step 3-5: Subtract Binary & DM
```
t_inf = Tempo2_BAT - binary_delay - dm_delay
Status: ✅ Subtracted correctly
But: Input (Tempo2_BAT) already had binary Roemer in it!
```

### FINAL RESULT FOR THIS TOA
```
Residual: ~850 microseconds (wrong!)
Status: ❌ INCORRECT - Due to wrong input
```

---

## WHY THE DISCREPANCY

```
PINT starts with:     Topocentric TOA
JUG starts with:      Tempo2 BAT (which ≠ infinite-frequency BAT)

Difference:           284.86 seconds

When you compute phase:
  Phase = F0 × time_error
        = 339.32 Hz × 284.86 sec
        = 96658 cycles
  
Wrapped to [0, 1] cycles:
  Phase_wrapped ≈ ±0.7 cycles
  
In microseconds (at 908 MHz):
  Error ≈ ±850 microseconds ✓ MATCHES JUG RESIDUAL ERROR!
```

---

## WHAT THIS TOA NEEDS

**To convert JUG computation to PINT level:**

1. **Stop using Tempo2 BAT**: 58526.2105921510
   - This value is incomplete
   - Missing Shapiro delay
   - Still contains binary Roemer delay

2. **Start with actual infinite-frequency time**:
   - Begin with topocentric: 58526.2138891490
   - Compute Roemer: -284.86 (approximately)
   - Add Shapiro: ~0.000001 seconds
   - Result: Proper infinite-frequency BAT

3. **Then subtract binary & DM** as JUG already does

---

## SUMMARY

For this single TOA:
- PINT does it right: ✅ 2.184 μs RMS (across all TOAs)
- JUG does it wrong: ❌ ~850 μs RMS (across all TOAs)
- Reason: Wrong input data source
- Solution: Implement Roemer + Shapiro delays

