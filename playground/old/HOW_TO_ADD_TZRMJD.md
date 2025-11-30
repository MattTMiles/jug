# How to Add TZRMJD to Your Par File

## Quick Start

To use MK5's 100% independent mode, add this line to your pulsar parameter file:

```
TZRMJD         58526.0              1
```

---

## What is TZRMJD?

**TZRMJD** = Time Zero Reference MJD

- The epoch where the pulsar phase is anchored to zero
- Used by PINT for absolute phase calculations
- In MJD format (UTC), will be converted to TDB internally
- Usually the same as PEPOCH or your first TOA

---

## How to Determine TZRMJD

### Method 1: Use PEPOCH (Simplest)
If you don't have a specific TZR requirement, use your PEPOCH:

```
PEPOCH         58526.0
TZRMJD         58526.0              1   # Same as PEPOCH
```

### Method 2: Extract from PINT (If you have PINT)
If you want to match PINT's TZR exactly:

```python
import pint.models
import pint.toa

# Load your data
model = pint.models.get_model('pulsar.par')
toas = pint.toa.get_TOAs('pulsar.tim', model=model)

# Get PINT's TZR TOA
tzr_toa = model.get_TZR_toa(toas)
TZRMJD = tzr_toa.table['mjd'][0]

print(f"TZRMJD {TZRMJD:.10f}")
# Add this value to your par file
```

### Method 3: Use First TOA
Use the MJD of your first observation:

```
TZRMJD         58526.123456789      1   # First TOA MJD
```

---

## Par File Example

### Before (MK4 compatible)
```
PSRJ           J1909-3744
RAJ            19:09:47.4366506     1  0.00000001
DECJ           -37:44:14.46573      1  0.00001
F0             339.31565036396      1  0.00000000001
F1             -1.6148198e-15       1  2e-21
F2             0.0                  0  0.0
PEPOCH         58526.0
DM             10.394               1  0.001
DMEPOCH        58526.0
BINARY         ELL1
PB             1.533449474          1  0.000000001
A1             1.897828             1  0.000001
TASC           58526.123            1  0.001
EPS1           0.0                  0  0.0
EPS2           0.0                  0  0.0
```

### After (MK5 independent)
```
PSRJ           J1909-3744
RAJ            19:09:47.4366506     1  0.00000001
DECJ           -37:44:14.46573      1  0.00001
F0             339.31565036396      1  0.00000000001
F1             -1.6148198e-15       1  2e-21
F2             0.0                  0  0.0
PEPOCH         58526.0
TZRMJD         58526.0              1  0.0          # <-- ADD THIS LINE
DM             10.394               1  0.001
DMEPOCH        58526.0
BINARY         ELL1
PB             1.533449474          1  0.000000001
A1             1.897828             1  0.000001
TASC           58526.123            1  0.001
EPS1           0.0                  0  0.0
EPS2           0.0                  0  0.0
```

---

## Parameter Format

```
TZRMJD         <value>              <fit_flag>  <uncertainty>
```

- **value**: MJD in UTC (e.g., 58526.0 or 58526.123456789)
- **fit_flag**: 1 (fitted) or 0 (fixed) - usually 1
- **uncertainty**: Error estimate (e.g., 0.0 for fixed epoch)

---

## What MK5 Does with TZRMJD

When MK5 finds `TZRMJD` in your par file:

1. **Reads the value**: `TZRMJD_UTC = par_params['TZRMJD']`
2. **Finds the closest TOA**: Matches TZRMJD to actual TOA for frequency
3. **Converts to TDB**: Uses standalone clock chain (MeerKAT → GPS → BIPM2024)
4. **Calculates TZR delay**: Uses independent JAX calculation
5. **Computes TZR phase**: Anchors all phases to this reference

**Result**: 100% independent TZR calculation, no PINT needed!

---

## Verification

### Check MK5 is Using TZRMJD

When you run MK5, look for this message:

```
Using TZRMJD from par file: 58526.0
TZR TDB: 58526.00007523914 (standalone)
TZR delay: -45574211.144 ns (independent calculation)
```

If you see this, you're running in 100% independent mode! ✅

### If TZRMJD is Missing

You'll see:

```
WARNING: TZRMJD not in par file, falling back to PINT's get_TZR_toa
For true independence, add TZRMJD parameter to par file!
```

This means MK5 is falling back to PINT (still works, but not independent).

---

## Common Questions

### Q: Does TZRMJD affect timing results?
**A**: No! It only defines the phase reference point. Your residuals will be the same.

### Q: Should TZRMJD be the same as PEPOCH?
**A**: Usually yes, but not required. It can be any epoch.

### Q: What if I change TZRMJD?
**A**: Residuals stay the same (they're phase differences). Only absolute phase changes.

### Q: Do I need high precision for TZRMJD?
**A**: Not critical. Even `58526.0` vs `58526.123456789` makes negligible difference (~ns level).

### Q: Can I have multiple TZRMJDs for different datasets?
**A**: No, one TZRMJD per par file. But you can create different par files.

---

## Troubleshooting

### "TZRMJD not in par file"
- **Cause**: Parameter not added or misspelled
- **Solution**: Add `TZRMJD` line to par file, check spelling

### "TZR delay differs significantly"
- **Cause**: TZRMJD doesn't match any actual TOA
- **Solution**: Use MJD of an actual observation, or PEPOCH

### "Parser error in par file"
- **Cause**: Invalid format
- **Solution**: Use format: `TZRMJD <value> <fit_flag> [uncertainty]`

---

## Quick Reference

### Minimal Addition
```
TZRMJD         58526.0              1
```

### With Uncertainty
```
TZRMJD         58526.123456789      1  0.000000001
```

### Fixed (Not Fitted)
```
TZRMJD         58526.0              0  0.0
```

---

## Summary

1. **Add TZRMJD to par file** (one line!)
2. **Use any reasonable epoch** (PEPOCH, first TOA, or PINT's TZR)
3. **Run MK5 normally** (it auto-detects TZRMJD)
4. **Enjoy 100% independence!** (no PINT values used)

That's it! One line in your par file unlocks true independence. 🎉
