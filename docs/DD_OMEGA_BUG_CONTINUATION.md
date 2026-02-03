# DD Binary Model Omega Bug - Continuation Document

## Summary

JUG's DD binary model produces incorrect pre-fit residuals compared to PINT/Tempo2. The root cause has been identified: **the omega (longitude of periastron) calculation uses time instead of true anomaly**.

## The Bug

### JUG's Current (Buggy) Implementation
```python
# Lines 419-422 in jug/delays/binary_dd.py (BEFORE my attempted fix)
dt_years = dt_days / 365.25
omega_current_deg = omega_deg + omdot_deg_yr * dt_years
omega_rad = jnp.deg2rad(omega_current_deg)
```
This computes omega as a **linear function of time**: `ω = OM + OMDOT × (t - T0)`

### Correct Implementation (PINT/Tempo2)
```python
# PINT DD_model.py line 96:
omega = OM + k * nu  # where nu is accumulated true anomaly

# Tempo2 DDmodel.C lines 131-136:
ae = 2.0*M_PI*orbits + atan2(sae,cae) - phase;
omega = omz/rad2deg + k*ae;
```

Where:
- `k = OMDOT * PB / (360 * 365.25)` (dimensionless, with OMDOT in deg/yr, PB in days)
- `ae` (accumulated true anomaly) = `2π × orbits + ν_wrapped - M`
- `ν_wrapped` = true anomaly in [0, 2π)
- `M` = mean anomaly

## What I Attempted

I modified `jug/delays/binary_dd.py` to compute omega using the accumulated true anomaly (see current state of file around lines 444-474). However, **the fix made things worse** - differences went from ~8 μs to ~2000 μs.

## Key Finding

When I tested the omega calculation in isolation, **both methods give nearly identical results** (difference of only 7.7e-12 radians). This means:

1. For J1713+0747 (OMDOT = 0.0001 deg/yr, very small eccentricity), the omega correction is negligible
2. The ~8 μs orbital-phase-dependent error must have a **different root cause**
3. My code change introduced a regression somewhere else

## Files Modified

- `/home/mattm/soft/JUG/jug/delays/binary_dd.py` - Contains my attempted fix (currently broken)

## Reference Implementations

1. **PINT**: `/home/mattm/soft/PINT/src/pint/models/stand_alone_psr_binaries/DD_model.py`
   - `omega()` method at line 86-96
   - `k()` method at line 83
   - `nu()` in `binary_generic.py` lines 536-547

2. **Tempo2**: `/home/mattm/not_in_use_reference/tempo2/DDmodel.C`
   - Lines 128-136 for true anomaly and omega calculation
   - Key formula: `omega = omz/rad2deg + k*ae`

## Test Case

- Par file: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747_tdb.par`
- Tim file: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1713+0747.tim`
- Binary model: DD
- Key parameters: PB=67.8 days, A1=32.3 lt-s, ECC=7.5e-5, OMDOT=0.0001 deg/yr

## Debug Scripts Created

- `debug_dd_compare.py` - Compares JUG vs PINT binary delays
- `debug_dd_orbit.py` - Compares orbital element calculations
- `debug_dd_pint.py` - Direct JUG vs PINT comparison
- `debug_dd_delay.py` - Step-by-step DD delay calculation

## Instructions for Next Claude

### Step 1: Revert My Changes
First, revert the omega calculation back to the original (linear in time) version:

```python
# In jug/delays/binary_dd.py, replace the omega section (lines ~444-474) with:

# Apply periastron advance: omega(t) = OM + OMDOT * (t - T0) / 1 year
dt_years = dt_days / 365.25
omega_current_deg = omega_deg + omdot_deg_yr * dt_years
omega_rad = jnp.deg2rad(omega_current_deg)

# Trigonometric functions
sinE = jnp.sin(E)
cosE = jnp.cos(E)
sinOm = jnp.sin(omega_rad)
cosOm = jnp.cos(omega_rad)
```

### Step 2: Verify Baseline
Run `python debug_dd_compare.py` and confirm the differences are back to ~8 μs (not ~2000 μs).

### Step 3: Find the Real Bug
The ~8 μs orbital-phase-dependent error is NOT from the omega calculation (as proven by my test showing omega values match to 7.7e-12 rad).

Investigate these possible causes:

1. **Inverse delay transformation** (D&D eq [52]) - Check if JUG's formula matches PINT/Tempo2 exactly:
   ```python
   # JUG (lines 513-519):
   correction_factor = (
       1.0
       - nhat * Drep
       + (nhat * Drep)**2
       + 0.5 * nhat**2 * Dre * Drepp
       - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat**2 * Dre * Drep
   )
   ```
   Compare with Tempo2 line 157-158.

2. **nhat calculation** - Check if the mean motion derivative is computed correctly

3. **Shapiro delay** - Compare JUG's delayS with PINT/Tempo2

4. **Time used for binary model** - PINT uses a specific time (tdbld minus geometric delays), ensure JUG uses equivalent

### Step 4: Create Detailed Comparison
Write a script that computes and prints every intermediate value (E, nu, omega, alpha, beta, Dre, Drep, etc.) for both JUG and PINT at a single TOA, then identify where they first diverge.

### Key Tempo2 Formulas (DDmodel.C)

```c
// Line 116: orbits calculation
orbits = tt0/pb - 0.5*(pbdot+xpbdot)*(tt0/pb)*(tt0/pb);

// Lines 121-125: Kepler solver
u=phase+ecc*sin(phase)*(1.0+ecc*cos(phase));
do {
    du=(phase-(u-ecc*sin(u)))/(1.0-ecc*cos(u));
    u=u+du;
} while (fabs(du)>1.0e-12);

// Lines 139-144: Roemer delay components
alpha=x*sw;
beta=x*sqrt(1-pow(eth,2))*cw;
bg=beta+gamma;
dre=alpha*(cu-er) + bg*su;
drep=-alpha*su + bg*cu;
drepp=-alpha*cu - bg*su;

// Lines 157-158: Inverse delay (d2bar)
d2bar=dre*(1-anhat*drep+(pow(anhat,2))*(pow(drep,2) + 0.5*dre*drepp -
            0.5*ecc*su*dre*drep/onemecu)) + ds + da;
```

## Expected Outcome

JUG should achieve <100 ns RMS agreement with PINT/Tempo2 for DD model pulsars (matching the ~1 μs agreement already achieved for ELL1 model).

## Git Status

I reverted `jug/delays/binary_dd.py` to its original state. However, there are other modified files that may be relevant:

```
 M jug/delays/binary_t2.py
 M jug/delays/combined.py
 M jug/fitting/derivatives_binary.py
 M jug/fitting/derivatives_dd.py
 M jug/fitting/optimized_fitter.py
 M jug/model/parameter_spec.py
 M jug/residuals/simple_calculator.py
```

**IMPORTANT**: The 17 ms differences persist even after reverting binary_dd.py. This suggests the bug may be in one of these other modified files, OR the baseline was already broken.

To get a clean baseline:
```bash
git stash  # Save all changes
python debug_dd_compare.py  # Test with pristine code
git stash pop  # Restore changes
```

## Additional Notes

The debug script `debug_dd_compare.py` compares JUG vs PINT binary delays at TDB times from the tim file. Current output shows:
- Mean difference: 170 μs
- Max difference: 17 ms
- The differences are orbital-phase-dependent

This is much worse than expected. The ELL1 model achieves ~1 μs agreement, so DD should be achievable too.
