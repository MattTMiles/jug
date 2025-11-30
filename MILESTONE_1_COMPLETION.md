# JUG Milestone 1 - COMPLETION REPORT

## Status: ✅ COMPLETED

### Final Results
- **JUG RMS**: 0.817 μs
- **PINT RMS**: 0.818 μs
- **Difference**: 0.003 μs std (well below 0.1 μs target)

### Problem Identified
The 3.4 μs standard deviation error was caused by **missing binary Shapiro delay** computation. JUG was looking for H3/STIG (orthometric) Shapiro parameters but the J1909-3744 par file uses M2/SINI (mass/inclination) parameterization instead.

### Root Cause
In `/home/mattm/soft/JUG/jug/residuals/simple_calculator.py`, the code was:
```python
H3 = float(params.get('H3', 0.0))
STIG = float(params.get('STIG', 0.0))
r_shap_jax = jnp.array(H3)
s_shap_jax = jnp.array(STIG)
```

Since the par file had M2/SINI instead of H3/STIG, both parameters defaulted to 0.0, completely disabling the binary Shapiro delay (which is ~2 μs for J1909-3744 at certain orbital phases).

### Solution
Added support for both parameterizations by converting M2/SINI to r/s:
```python
# Handle both H3/STIG (orthometric) and M2/SINI (mass/inclination) parameterizations
if 'H3' in params and 'STIG' in params:
    # Orthometric parameters (H3, STIG)
    H3 = float(params['H3'])
    STIG = float(params['STIG'])
    r_shap_jax = jnp.array(H3)
    s_shap_jax = jnp.array(STIG)
elif 'M2' in params and 'SINI' in params:
    # Convert M2/SINI to r/s
    # r = TSUN * M2 (where TSUN = G*Msun/c^3 = 4.925490947e-6 s)
    # s = SINI
    M2 = float(params['M2'])  # solar masses
    SINI = float(params['SINI'])
    r_shap_jax = jnp.array(T_SUN_SEC * M2)
    s_shap_jax = jnp.array(SINI)
else:
    # No Shapiro delay parameters
    r_shap_jax = jnp.array(0.0)
    s_shap_jax = jnp.array(0.0)
```

### File Modified
- `/home/mattm/soft/JUG/jug/residuals/simple_calculator.py` (lines 203-221)

### Verification
Test command:
```bash
cd /home/mattm/soft/JUG
python -m jug.scripts.compute_residuals \
  /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par \
  /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim
```

Results comparison (JUG - PINT):
- Mean difference: 0.000 μs
- Std difference: 0.003 μs
- RMS difference: 0.003 μs
- Max difference: 0.013 μs

### Technical Details
- **Binary model**: ELL1 (low eccentricity approximation)
- **Orbital period**: 1.533 days
- **Companion mass**: M2 = 0.204 solar masses
- **Inclination**: SINI = 0.998
- **Shapiro range parameter**: r = TSUN × M2 = 1.004e-6 s
- **Shapiro shape parameter**: s = SINI = 0.998

The binary Shapiro delay varies from 0 to ~2 μs over the orbit, which explains why some TOAs had 11 μs errors (accumulated over multiple cycles) when this component was missing.

### Milestone 1 Status
✅ **COMPLETE** - JUG now matches PINT accuracy to <0.01 μs for J1909-3744, a challenging millisecond pulsar binary system.

Date: 2025-11-29
