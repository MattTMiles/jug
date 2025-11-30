# JUG-PINT SS Shapiro Delay Investigation - SOLVED!

## Status: MYSTERY SOLVED! (Nov 28, 2025)

### The Answer: PLANET_SHAPIRO = True

**PINT includes planetary Shapiro delays when `PLANET_SHAPIRO=True` in the par file!**

JUG was only computing the Sun's contribution, while PINT was computing:
- Sun: ~-6000 ns (dominant)
- Jupiter: -3.0 ns mean, 13.7 ns RMS (main "missing" contribution!)
- Saturn: -0.7 ns mean, 2.9 ns RMS
- Uranus: -1.4 ns mean
- Neptune: -1.5 ns mean
- Venus: +0.006 ns mean

**Total planetary contribution: -6.67 ns mean, 16.36 ns RMS**

This EXACTLY explains the 16 ns RMS difference we were seeing!

### Verification

When comparing JUG's Sun-only calculation with PINT's Sun-only:
```
JUG - PINT(Sun-only): mean=0.000 ns, RMS=0.000 ns  ✅ PERFECT MATCH!
```

When comparing Sun + planets with PINT's stored result:
```
Difference: mean=0.000000 ns, RMS=0.000000 ns  ✅ PERFECT MATCH!
```

## What to Investigate Next

### 1. Check PINT's actual SS Shapiro source code execution

The formula from PINT's `solar_system_shapiro.py` (lines 59-81):
```python
r = (np.sqrt(np.sum(obj_pos**2, axis=1))) * obj_pos.unit
rcostheta = np.sum(obj_pos * psr_dir, axis=1)
return -2.0 * T_obj * np.log((r - rcostheta) / const.au).value
```

**BUT** - check if there's something happening BEFORE the formula:
- Does PINT modify `obs_sun_pos` before using it?
- Does PINT use a different `psr_dir` than `ssb_to_psb_xyz_ICRS`?
- Is there unit handling that changes values?

### 2. Debug inside PINT's `solar_system_shapiro_delay` method

Add print statements or step through:
```python
# In cell, temporarily modify PINT's calculation
ss_comp = pint_model.components['SolarSystemShapiro']
# Check what ss_comp.ss_obj_shapiro_delay actually receives
```

### 3. Check if PINT normalizes the pulsar direction

The pulsar direction from `ssb_to_psb_xyz_ICRS` might not be a unit vector. PINT might normalize it internally:
```python
psr_dir_norm = psr_dir / np.linalg.norm(psr_dir, axis=0)
```

### 4. Check aberration effects

PINT might include annual aberration in the SS Shapiro calculation that JUG doesn't. This could cause an annual pattern + drift.

### 5. Verify PINT's Tsun value

```python
from pint import Tsun
print(f"PINT Tsun = {Tsun.value:.15e}")  # Should be 4.925490948309e-06
```

Make sure JUG uses exactly this value.

## Files and Locations

- **Notebook**: `/home/mattm/soft/JUG/residual_maker_playground_MK3.ipynb`
- **Main computation cell**: Cell 13 (defines `jug_shapiro_sec` using `compute_shapiro_delay`)
- **PINT SS Shapiro source**: `pint/models/solar_system_shapiro.py`
- **Key investigation cells**: Near end of notebook (cells 63-70)

## Key Variables in Notebook

- `jug_shapiro_sec`: JUG's SS Shapiro delay (array, seconds)
- `pint_ss_shapiro_delay`: PINT's SS Shapiro delay (from `ss_shapiro_comp.solar_system_shapiro_delay(pint_toas)`)
- `obs_sun_pos_km`: JUG's Sun position relative to observer (N×3 array, km)
- `L_hat`: JUG's pulsar direction unit vector (N×3 array, dimensionless)
- `pint_psr_dir`: PINT's pulsar direction from `ssb_to_psb_xyz_ICRS`

## Code to Continue Investigation

```python
# 1. Check if psr_dir needs normalization
pint_psr_dir = astro.ssb_to_psb_xyz_ICRS(epoch=mjd_float)
pint_psr_dir_magnitude = np.linalg.norm(pint_psr_dir, axis=0 if pint_psr_dir.shape[0]==3 else 1)
print(f"PINT psr_dir magnitude: mean={np.mean(pint_psr_dir_magnitude)}, std={np.std(pint_psr_dir_magnitude)}")
# If not 1.0, then PINT might normalize internally

# 2. Step through PINT's actual calculation
import pint.models.solar_system_shapiro as ss_module
import inspect
print(inspect.getsource(ss_module.SolarSystemShapiro.ss_obj_shapiro_delay))

# 3. Check if there's aberration correction
# Look for velocity terms in the pulsar direction calculation
```

## Summary of Confirmed Matches

| Component | JUG-PINT Diff | Status |
|-----------|--------------|--------|
| Roemer delay | 0.8 ns RMS | ✅ Fixed |
| DM delay | 0.2 ns RMS | ✅ Good |
| Solar wind | ~0 ns | ✅ Good |
| FD delay | ~0 ns | ✅ Good |
| Binary delay | 3.8 ns RMS | ✅ Good |
| SS Shapiro | **16.4 ns RMS, 7.8 ns/yr drift** | ❌ Problem |

## The Fix (Once Found)

Once you understand why PINT's SS Shapiro differs, update JUG's `compute_shapiro_delay` function in the main computation cell to match PINT's exact approach. The function is currently:

```python
def compute_shapiro_delay(obs_sun_pos_km, L_hat):
    """Compute Solar System Shapiro delay."""
    r = np.sqrt(np.sum(obs_sun_pos_km**2, axis=1))
    rcostheta = np.sum(obs_sun_pos_km * L_hat, axis=1)
    return -2.0 * T_SUN_SEC * np.log((r - rcostheta) / AU_KM)
```

Likely changes needed:
1. Use PINT's exact `Tsun` value
2. Possibly normalize `L_hat` or handle it differently  
3. Possibly include aberration correction
4. Check if there's a parallax correction in SS Shapiro
