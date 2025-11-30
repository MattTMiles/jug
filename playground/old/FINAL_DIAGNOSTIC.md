# Final Diagnostic Summary

## Current Status

After extensive debugging, JUG produces residuals with:
- **RMS**: ~850 μs
- **Tempo2 RMS**: 0.817 μs
- **Correlation**: ~0 (essentially random)
- **Offset**: ~850 μs ≈ 0.288 cycles

## What We've Verified

✓ **Binary delay calculation is CORRECT**
  - Matches manual calculation within 0.5 μs
  - Includes PBDOT correction
  - Uses correct TASC, PB, A1, EPS1, EPS2
  - Shapiro delay computed from M2/SINI

✓ **DM delay calculation is CORRECT**
  - Matches manual calculation within 45 nanoseconds
  - Uses correct DM, DM1 polynomial
  - Frequency-dependent as expected

✓ **Tempo2's BAT contains binary delays**
  - Verified that BAT - binary - DM gives sensible infinite-frequency times
  - BAT is NOT emission time, NOT infinite-frequency time

✓ **Function updates are working**
  - Debug prints confirm the correct code is running
  - Manual calculations match JUG's computed values

## What We've Tried

1. ✗ Subtracting phase_offset_cycles → Made it worse
2. ✗ NOT subtracting phase_offset_cycles → Still wrong
3. ✗ Assuming BAT has no binary delays → Still wrong
4. ✗ TCB/TDB conversion → Not needed (UNITS=TDB in .par file)
5. ✗ Different reference epochs → No improvement

## Remaining Hypotheses

### Hypothesis 1: Tempo2 uses different delay model
- Maybe tempo2's binary delay formula differs from JUG's ELL1 implementation
- **Test**: Compare JUG's binary delay with tempo2's internal calculation

### Hypothesis 2: Phase wrapping issue
- The 850 μs ≈ 0.288 cycles suggests phase wrapping at wrong point
- **Test**: Check if tempo2 uses different wrapping convention

### Hypothesis 3: Missing parameter
- Maybe there's a PHASE or other offset parameter not in the .par file
- **Test**: Check tempo2's actual parameter values vs .par file

### Hypothesis 4: Reference frame issue
- Maybe TZR calculation uses wrong time scale or reference
- **Test**: Verify TZR epoch matches tempo2's internal TZR

### Hypothesis 5: Tempo2's "general2" output is not what we think
- Maybe the columns in general2 output mean something different
- **Test**: Run tempo2 with explicit column specifications

## Recommended Next Steps

1. **Compare tempo2's binary delay directly**
   - Run: `tempo2 -output delays -f file.par file.tim`
   - Compare with JUG's binary_delays_sec array

2. **Check tempo2's actual F0/F1 values**
   - Run tempo2 with `-printp` to see actual used parameters
   - Compare with .par file values

3. **Verify TZR calculation**
   - Check if tempo2's TZR handling matches JUG's
   - Maybe TZR should be at topocentric time, not BAT?

4. **Test with simpler pulsar**
   - Try a non-binary pulsar to eliminate binary delay as variable
   - Or try with BINARY=0 in .par file temporarily

## Key Numbers (First TOA)

```
Topocentric MJD: 58526.21388914872
Tempo2 BAT:      58526.21059215097
Frequency:       907.852369 MHz

JUG's delays:
  Binary:        0.384888193 s
  DM:            0.052304127 s
  Total:         0.437192 s

JUG's t_inf:     58526.210587090871 MJD

JUG's residual:  ~785 μs (without phase_offset subtraction)
                 ~1041 μs (with phase_offset subtraction)
Tempo2 residual: -2.016 μs

Difference:      ~787 μs ≈ 0.267 cycles
```

## Critical Question

**Why is there a systematic ~0.27 cycle offset between JUG and tempo2?**

This is not a small numerical error - it's a fundamental disagreement about the absolute phase. Either:
- JUG's delay calculations are systematically wrong by ~0.44 seconds
- The reference epoch/phase is wrong
- Tempo2 is using different physics/conventions

The fact that the RMS is huge and correlation is zero suggests we're not even close to the right answer, not just off by a constant.
