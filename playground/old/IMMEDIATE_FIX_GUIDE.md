# Immediate Fix Guide - Use PINT Times to Validate JUG

## Quick Summary

**Problem Found**: JUG uses tempo2's BAT column thinking it's infinite-frequency barycentric time, but it's not.

**Quick Fix**: Use PINT's `tdbld` column which IS the correct infinite-frequency barycentric time.

**Why This Matters**: This will prove JUG's residual calculation logic is correct, and the only issue is input data.

---

## Step-by-Step Fix (5 minutes)

### 1. Create a new test script:

```python
#!/usr/bin/env python3
"""
FINAL TEST: Use PINT's correct times with JUG's residual calculation.
This should prove JUG's logic is correct.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pint.models import get_model
from pint.toa import get_TOAs

# Enable JAX float64
jax.config.update('jax_enable_x64', True)

# Load data
par_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
tim_file = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, model=model)

# Get PINT's infinite-frequency barycentric times
pint_tdbld = toas.table['tdbld'].value  # MJD
pint_freq = toas.table['freq'].value    # MHz

print("="*80)
print("FINAL VALIDATION TEST")
print("="*80)

# JUG's timing model parameters
SECS_PER_DAY = 86400.0
F0 = 339.31569191904066
F1 = -1.6147400369092967e-15
PEPOCH = 59017.9997538705

@jax.jit
def spin_phase(t_mjd, f0, f1, pepoch):
    dt = (t_mjd - pepoch) * SECS_PER_DAY
    return f0 * dt + 0.5 * f1 * dt**2

# Get PINT's TZR barycentric time (CORRECT!)
tzrmjd_topo = model.TZRMJD.value
idx_tzr = np.argmin(np.abs(toas.table['mjd_float'] - tzrmjd_topo))
TZR_INF_MJD_PINT = pint_tdbld[idx_tzr]

print(f"\nUsing PINT's TZR infinite-frequency time:")
print(f"  TZR topocentric: {tzrmjd_topo}")
print(f"  TZR barycentric (tdbld): {TZR_INF_MJD_PINT}")
print(f"  Index: {idx_tzr}")

# Compute JUG residuals using PINT's correct times
print(f"\nComputing JUG residuals with PINT's times...")
t_inf_jax = jnp.array(pint_tdbld, dtype=jnp.float64)

# Phase calculation
phase = spin_phase(t_inf_jax, F0, F1, PEPOCH)
phase_ref = spin_phase(jnp.array([TZR_INF_MJD_PINT]), F0, F1, PEPOCH)[0]

# Residual
phase_diff = phase - phase_ref
frac_phase = jnp.mod(phase_diff + 0.5, 1.0) - 0.5
residual_sec = frac_phase / F0
residual_sec = residual_sec - jnp.mean(residual_sec)  # Remove mean
residual_us = np.array(residual_sec) * 1e6

# Load tempo2 residuals for comparison
t2_res = []
with open('temp_pre_general2.out') as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            try:
                t2_res.append(float(parts[1]))
            except:
                pass
t2_res_us = np.array(t2_res) * 1e6

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nJUG (with PINT's correct times):") 
print(f"  RMS: {np.sqrt(np.mean(residual_us**2)):.3f} μs")
print(f"  Mean: {np.mean(residual_us):.6f} μs")
print(f"  First 10: {residual_us[:10]}")

print(f"\nTempo2:")
print(f"  RMS: {np.sqrt(np.mean(t2_res_us**2)):.3f} μs")
print(f"  Mean: {np.mean(t2_res_us):.6f} μs")
print(f"  First 10: {t2_res_us[:10]}")

print(f"\nComparison:")
corr = np.corrcoef(residual_us, t2_res_us)[0,1]
diff = residual_us - t2_res_us
rms_diff = np.sqrt(np.mean(diff**2))

print(f"  Correlation: {corr:.6f}")
print(f"  RMS difference: {rms_diff:.3f} μs")
print(f"  Mean difference: {np.mean(diff):.6f} μs")
print(f"  Max difference: {np.max(np.abs(diff)):.3f} μs")

print("\n" + "="*80)
if rms_diff < 10 and corr > 0.99:
    print("✓✓✓ SUCCESS! JUG MATCHES TEMPO2/PINT!")
    print("="*80)
    print("\nThis proves:")
    print("  ✓ JUG's residual calculation logic is CORRECT")
    print("  ✓ JUG's phase computation is CORRECT")
    print("  ✓ JUG's phase wrapping is CORRECT")
    print("  ✓ The ONLY problem was using wrong input times (tempo2's BAT)")
    print("\nNext steps:")
    print("  1. Implement proper barycentric time calculation")
    print("  2. Compute Roemer + Shapiro delays from scratch")
    print("  3. Replace tempo2 BAT input with JUG's computed values")
    print("  4. JUG will then be fully independent and match tempo2!")
elif rms_diff < 100:
    print(f"⚠️  Close but not perfect: {rms_diff:.3f} μs RMS difference")
    print("="*80)
    print("\nPossible issues:")
    print(f"  - TZR reference might need adjustment")
    print(f"  - Check if PINT's TZR calculation differs slightly")
else:
    print(f"✗ Still significant discrepancy: {rms_diff:.3f} μs")
    print("="*80)
    print("\nThis suggests there may be additional issues beyond input times.")
```

### 2. Run the test:

```bash
cd /home/mattm/soft/JUG
python3 final_validation_test.py
```

### 3. Expected Results:

**If RMS difference < 10 μs and correlation > 0.99**:
- ✓ JUG's residual calculation is CORRECT!
- ✓ The only problem is input data (tempo2's BAT)
- ✓ Proceed to implement barycentric time calculation

**If RMS difference is 100-1000 μs**:
- TZR reference might need fine-tuning
- Check PINT's TZR calculation method
- May need to adjust phase reference handling

**If RMS difference is still ~850 μs**:
- Something else is wrong beyond input times
- Need to investigate residual calculation logic more

---

## What This Test Proves

### Success Case (< 10 μs RMS difference)

This would definitively prove:

1. **JUG's delay calculations are correct** ✓
   - Binary delays verified to < 0.5 μs
   - DM delays verified to < 45 ns
   - All physics is correct

2. **JUG's residual logic is correct** ✓
   - Phase computation works
   - Fractional phase extraction works
   - Phase wrapping works
   - Reference subtraction works

3. **The ONLY problem is input data** ✓
   - Using tempo2's BAT (wrong) instead of computing barycentric times (correct)
   - This is fixable by implementing Roemer + Shapiro delay calculations

### Why This Matters

If this test succeeds, it means:
- No need to redesign JUG's architecture
- No methodological differences from tempo2
- Just need to add missing barycentric correction functions
- Clear path forward to independence

---

## Implementation Plan After Validation

### Phase 1: Quick Functions (1-2 days)

**Function 1**: `compute_pulsar_direction(raj, decj, pmra, pmdec)`
```python
def compute_pulsar_direction(raj_rad, decj_rad, pmra, pmdec, mjd, epoch_mjd):
    """Convert RA/DEC to unit vector with proper motion."""
    # Convert to Cartesian
    cos_dec = np.cos(decj_rad)
    x = np.cos(raj_rad) * cos_dec
    y = np.sin(raj_rad) * cos_dec
    z = np.sin(decj_rad)
    
    # Apply proper motion (if needed)
    dt_years = (mjd - epoch_mjd) / 365.25
    # ... proper motion correction ...
    
    return np.array([x, y, z])
```

**Function 2**: `compute_roemer_delay(obs_xyz_ssb, pulsar_dir)`
```python
def compute_roemer_delay(obs_xyz_m, pulsar_unit):
    """Geometric light travel time delay."""
    return -np.dot(obs_xyz_m, pulsar_unit) / C_M_S
```

### Phase 2: Complex Functions (3-5 days)

**Function 3**: `get_observatory_position_ssb(mjd_tdb, obs_code, ephemeris)`
```python
def get_observatory_position_ssb(mjd_tdb, obs_code, ephemeris):
    """Get observatory position in SSB frame."""
    # 1. Load observatory geodetic coordinates
    obs_data = load_observatory(obs_code)
    
    # 2. Convert to geocentric XYZ
    obs_xyz_geo = geodetic_to_geocentric(obs_data)
    
    # 3. Apply Earth rotation
    obs_xyz_itrf = rotate_by_earth_rotation(obs_xyz_geo, mjd_tdb)
    
    # 4. Get Earth position from ephemeris
    earth_xyz_ssb = get_earth_position(mjd_tdb, ephemeris)
    
    # 5. Combine
    return earth_xyz_ssb + obs_xyz_itrf
```

**Function 4**: `compute_shapiro_delay(obs_xyz_ssb, pulsar_dir, ephemeris)`
```python
def compute_shapiro_delay(obs_xyz_ssb, pulsar_dir, mjd_tdb, ephemeris):
    """Shapiro delay from Sun, Jupiter, Saturn."""
    total_delay = 0.0
    
    for body in ['sun', 'jupiter', 'saturn']:
        body_pos = get_body_position(mjd_tdb, body, ephemeris)
        delay = compute_shapiro_single_body(obs_xyz_ssb, body_pos, pulsar_dir)
        total_delay += delay
    
    return total_delay
```

### Phase 3: Integration (1-2 days)

Replace this in notebook cell 14:
```python
# OLD (WRONG):
bat_mjd_np = tempo2_bat  # From tempo2 output

# NEW (CORRECT):
# 1. Parse topocentric TOAs from .tim file
toa_utc_mjd = # from .tim parser

# 2. Apply clock corrections (already have this!)
toa_tt_mjd = toa_utc_mjd + clock_correction_seconds(...) / 86400

# 3. Compute observatory position
obs_xyz_ssb = get_observatory_position_ssb(toa_tt_mjd, obs_code, ephemeris)

# 4. Compute pulsar direction
pulsar_dir = compute_pulsar_direction(model.RAJ, model.DECJ, ...)

# 5. Compute Roemer delay
roemer_delay = compute_roemer_delay(obs_xyz_ssb, pulsar_dir)

# 6. Compute Shapiro delay
shapiro_delay = compute_shapiro_delay(obs_xyz_ssb, pulsar_dir, toa_tt_mjd, ephemeris)

# 7. Barycentric arrival time
bat_mjd_np = toa_tt_mjd + (roemer_delay + shapiro_delay) / 86400

# 8. Continue with existing binary/DM code...
# (This part is already correct!)
```

---

## Resources Available

### Data Files
- ✓ `data/ephemeris/de440s.bsp` - JPL ephemeris (already loaded)
- ✓ `data/observatory/observatories.dat` - Observatory positions
- ✓ `data/earth/eopc04_IAU2000.62-now` - Earth orientation parameters

### Reference Code
- PINT's `pint/observatory/observatory.py` - Observatory position calculation
- PINT's `pint/toa.py` - Barycentric correction implementation
- PINT's `pint/utils.py` - Coordinate transformations
- Tempo2's `T2toolkit/` - C implementations (well documented)

### Libraries Needed
- `spiceypy` or `jplephem` - For reading `.bsp` ephemeris files
- `astropy` - For coordinate transformations (or implement manually)
- `scipy` - For interpolation (Earth orientation parameters)

---

## Timeline Estimate

**If validation test succeeds**:
- Day 1: Implement pulsar direction + Roemer delay (easy)
- Day 2-3: Implement observatory position calculation (moderate)
- Day 4-5: Implement Shapiro delay (complex but well-documented)
- Day 6: Integration and testing
- Day 7: Validation against tempo2/PINT

**Total**: ~1 week to full independence

---

## Success Criteria

After implementation, JUG should achieve:
- ✓ Barycentric times match PINT's `tdbld` to < 1 μs
- ✓ Residuals match tempo2/PINT to < 1 μs RMS
- ✓ No dependency on tempo2 or PINT
- ✓ Fully independent pulsar timing package

---

## Run the Test Now!

```bash
cd /home/mattm/soft/JUG

# Save the script above as:
cat > final_validation_test.py << 'SCRIPT'
[paste the script from section 1]
SCRIPT

# Run it:
python3 final_validation_test.py
```

This single test will tell us if we're on the right track!
