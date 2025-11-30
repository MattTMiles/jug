# JUG PINT Independence - Next Steps

## Project Context

JUG (JAX-based Ultra-fast pulsar timing) is a high-performance residual calculator that currently depends on PINT for several critical functions. The goal is to make JUG completely independent of PINT while maintaining identical residual outputs (validated to < 1 nanosecond RMS difference).

**Performance**: JUG is ~1097x faster than PINT (4ms vs 4.4s for 10k TOAs)

**Notebook**: `/home/mattm/soft/JUG/residual_maker_playground_active_MK2_clkfix.ipynb`

---

## What Has Been Completed ✓

### 1. Native .tim File Parser ✓
**Status**: COMPLETE and VALIDATED

**Location**: Cell after parameter parsing (search for "TIM FILE PARSING")

**What it does**:
- Parses tempo2-style `.tim` files (FORMAT 1)
- Extracts: filename, freq_mhz, mjd_utc, error_us, observatory, flags
- Returns list of `TOA` dataclass objects
- Handles comments, headers, flag parsing

**Validation**:
- ✓ Parsed all 10,408 TOAs from J1909-3744.tim
- ✓ Frequencies match PINT exactly: `np.allclose(jug_freqs, pint_freqs)` returns True

**Key function**: `parse_tim_file(path: Path) -> List[TOA]`

### 2. Clock Correction System ✓
**Status**: COMPLETE and VALIDATED

**Location**: Cell after TIM parser (search for "CLOCK CORRECTION SYSTEM")

**What it does**:
- Implements tempo2-style clock correction chain walking
- Chain: Observatory → UTC → GPS/TAI → TT (BIPM)
- Reads `.clk` files from `data/clock/` directory
- Handles observatory alias resolution via `tempo.aliases`
- Linear interpolation between MJD points in clock files
- Automatically selects correct BIPM version from .par file CLK parameter

**Validation**:
- ✓ Correctly applies 3-stage correction:
  - MK→UTC: +0.4 microseconds (observatory clock)
  - UTC→TAI: +37.0 seconds (leap seconds)
  - TAI→TT: +32.184 seconds (+ ~27μs BIPM correction)
  - **Total: 69.184 seconds** ✓ CORRECT

**Key functions**:
- `clock_correction_seconds(mjd_utc, obs_code, clock_edges, tai2tt_edge, alias_map) -> float`
- `_load_clock_edges(clock_dir)` - builds correction graph
- `_choose_tai2tt(edges, bipm_version)` - selects BIPM file

**Global variables initialized**:
- `clock_edges`: Dict mapping source → list of correction edges
- `tai2tt_edge`: Selected TAI→TT BIPM correction
- `alias_map`: Observatory code aliases

---

## What Needs to Happen Next

### TASK 3: TDB Calculation (TT → TDB)
**Status**: PENDING - HIGHEST PRIORITY

**Why this is needed**:
Currently, JUG uses PINT's pre-computed TDB times from `pint_toas.table['tdbld']`. This is a major dependency that must be removed. TDB (Barycentric Dynamical Time) is required for accurate pulsar timing because it accounts for:
1. **Einstein delay**: Gravitational time dilation (TT → TDB conversion)
2. **Shapiro delay**: Light travel time delay through gravitational fields

**Current PINT dependency**:
```python
# In JUGResidualCalculatorFinal._precompute_all():
tdbld = np.array(self.pint_toas.table['tdbld'].value, dtype=np.float64)  # ← REMOVE THIS
```

**What you need to implement**:

#### Step 1: Understand TDB Calculation
TDB is computed from TT using:
```
TDB = TT + (Einstein_delay + Shapiro_delay) / SECS_PER_DAY
```

Where:
- **Einstein delay** (seconds): Gravitational time dilation from Earth's motion
- **Shapiro delay** (seconds): Gravitational light-travel-time delay

The calculation requires:
- Observatory position relative to Solar System Barycenter (SSB)
- Sun and planet positions (for Shapiro delays)
- Pulsar direction vector

**Good news**: JUG already computes Shapiro delays! They're used in the residual calculator. You just need to repurpose them for TDB calculation.

#### Step 2: Extract Existing Delay Calculations

Look at `JUGResidualCalculatorFinal._precompute_all()` around line ~250:

```python
# These are already computed:
ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdbld, self.obs_itrf_km)
obs_sun_pos_km = sun_pos - ssb_obs_pos_km
sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)
```

**Problem**: These use `tdbld` (PINT's TDB) as input. You need to compute TDB iteratively:
1. Start with TT times (from clock corrections)
2. Compute approximate Einstein + Shapiro delays using TT
3. Apply delays to get first-order TDB
4. Optionally iterate for higher precision (usually 1 iteration is sufficient)

#### Step 3: Implementation Strategy

**Create a new function** (insert after clock correction system):

```python
def compute_tdb_from_tt(mjd_tt: np.ndarray, obs_itrf_km: np.ndarray,
                        par_params: Dict[str, Any]) -> np.ndarray:
    """
    Convert TT (Terrestrial Time) to TDB (Barycentric Dynamical Time).

    Uses iterative approach:
    1. Use TT as first guess for TDB
    2. Compute Einstein + Shapiro delays at that time
    3. Apply corrections: TDB = TT + delays/SECS_PER_DAY
    4. (Optional) Iterate once for precision

    Parameters
    ----------
    mjd_tt : array of TT times (MJD)
    obs_itrf_km : Observatory ITRF position [x, y, z] in km
    par_params : Timing model parameters (for pulsar position)

    Returns
    -------
    mjd_tdb : array of TDB times (MJD)
    """

    # First iteration: use TT as guess
    tdb_guess = mjd_tt

    # Get pulsar direction
    ra_rad = parse_ra(par_params['RAJ'])
    dec_rad = parse_dec(par_params['DECJ'])
    pmra_rad_day = par_params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
    pmdec_rad_day = par_params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
    posepoch = par_params.get('POSEPOCH', par_params['PEPOCH'])

    # Compute observatory position at TDB guess
    ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_guess, obs_itrf_km)
    L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_guess)

    # Get Sun position
    times = Time(tdb_guess, format='mjd', scale='tdb')
    with solar_system_ephemeris.set('de440'):
        sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
    obs_sun_pos_km = sun_pos - ssb_obs_pos_km

    # Compute delays
    roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, par_params.get('PX', 0.0))
    sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)

    # Planet Shapiro delays (if enabled)
    planet_shapiro_sec = np.zeros(len(mjd_tt))
    planet_shapiro_enabled = str(par_params.get('PLANET_SHAPIRO', 'N')).upper() in ('Y', 'YES', 'TRUE', '1')
    if planet_shapiro_enabled:
        with solar_system_ephemeris.set('de440'):
            for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
                planet_pos = get_body_barycentric_posvel(planet, times)[0].xyz.to(u.km).value.T
                obs_planet_km = planet_pos - ssb_obs_pos_km
                planet_shapiro_sec += compute_shapiro_delay(obs_planet_km, L_hat, T_PLANET[planet])

    # Einstein delay (velocity-dependent term)
    # Simplified: einstein_sec ≈ |v|²/2c² where v is Earth velocity
    # For full accuracy, use PINT's implementation or tempo2's formula
    # For now, approximate using Roemer delay derivative
    # NOTE: This is a simplification - check tempo2 or PINT source for exact formula

    v_mag_km_s = np.sqrt(np.sum(ssb_obs_vel_km_s**2, axis=1))
    einstein_sec = 0.5 * (v_mag_km_s / C_KM_S)**2 * roemer_sec  # Approximate

    # Total correction
    total_delay_sec = roemer_sec + sun_shapiro_sec + planet_shapiro_sec + einstein_sec

    # Apply to get TDB
    mjd_tdb = mjd_tt + total_delay_sec / SECS_PER_DAY

    return mjd_tdb
```

**IMPORTANT NOTES**:

1. **Einstein delay formula**: The approximate formula above is simplified. For production accuracy, you need the exact formula from tempo2 or PINT. Look at:
   - PINT: `pint/observatory/topo_obs.py` in method `posvel` or `get_gcrs_posvel`
   - Tempo2: Search for "Einstein delay" in tempo2 source
   - The exact term involves: `(v²/2c² - GM☉/c²r) × Roemer_delay`

2. **Iteration**: For microsecond precision, one iteration is usually sufficient. But you can iterate:
   ```python
   for _ in range(2):  # Iterate twice
       tdb_guess = mjd_tt + total_delay_sec / SECS_PER_DAY
       # Recompute delays using new tdb_guess
       # ...
   ```

3. **Validation**: Compare against PINT's `tdbld` column:
   ```python
   pint_tdb = np.array(pint_toas.table['tdbld'].value)
   jug_tdb = compute_tdb_from_tt(jug_mjd_tt, obs_itrf_km, par_params)
   diff_us = (jug_tdb - pint_tdb) * SECS_PER_DAY * 1e6
   print(f"TDB difference RMS: {np.std(diff_us):.3f} microseconds")
   # Goal: < 0.1 microseconds RMS
   ```

#### Step 4: Integration into Main Pipeline

Once `compute_tdb_from_tt()` works, integrate it:

**In the data loading cell** (where PINT is currently used):
```python
# OLD (using PINT):
# pint_toas = pint_get_TOAs(str(tim_file), model=pint_model)

# NEW (using JUG):
jug_toas = parse_tim_file(tim_file)

# Apply clock corrections
jug_clock_corrections_sec = np.array([
    clock_correction_seconds(toa.mjd_utc, toa.observatory, clock_edges, tai2tt_edge, alias_map)
    for toa in jug_toas
])
jug_mjd_tt = np.array([toa.mjd_utc for toa in jug_toas]) + jug_clock_corrections_sec / SECS_PER_DAY

# Compute TDB
jug_mjd_tdb = compute_tdb_from_tt(jug_mjd_tt, obs_itrf_km, par_params)

# For validation, still load PINT temporarily
pint_toas = pint_get_TOAs(str(tim_file), model=pint_model)
pint_mjd_tdb = np.array(pint_toas.table['tdbld'].value)

# Compare
diff_us = (jug_mjd_tdb - pint_mjd_tdb) * SECS_PER_DAY * 1e6
print(f"TDB difference: {np.mean(diff_us):.3f} ± {np.std(diff_us):.3f} µs")
```

**In JUGResidualCalculatorFinal**:
```python
# OLD:
# tdbld = np.array(self.pint_toas.table['tdbld'].value, dtype=np.float64)

# NEW:
tdbld = jug_mjd_tdb  # Use JUG-computed TDB
```

#### Step 5: Validation Criteria

✓ **Success**: TDB difference < 0.1 microseconds RMS
⚠ **Warning**: 0.1-1.0 microseconds RMS (acceptable but check Einstein formula)
✗ **Fail**: > 1.0 microseconds RMS (debug required)

**Expected difference sources**:
- Different Einstein delay formulation (main source)
- Iteration count (minor)
- Numerical precision (negligible)

---

### TASK 4: Independent TZR Phase Anchoring
**Status**: PENDING

**Why this is needed**:
Currently uses PINT's `get_TZR_toa()` and `delay()` methods to anchor the absolute phase. This must be replaced.

**Current PINT dependency** (in `JUGResidualCalculatorFinal._precompute_all()`):
```python
pint_tzr_toa = self.pint_model.get_TZR_toa(self.pint_toas)
TZRMJD_TDB = np.longdouble(pint_tzr_toa.table['tdbld'][0])
pint_tzr_delay = float(self.pint_model.delay(pint_tzr_toa).to('s').value[0])
```

**What you need to implement**:

1. **Find the TZR TOA**:
   - If TZRMJD specified in .par: find TOA closest to TZRMJD
   - If TZRSITE specified: filter by observatory
   - If TZRFRQ specified: find TOA closest to that frequency
   - Default: use first TOA

2. **Compute total delay at TZR**:
   - Use your `compute_tdb_from_tt()` for TDB time
   - Compute all delays: Roemer + Einstein + Shapiro + binary + DM
   - This is similar to what's in `compute_total_delay_jax()` but for a single TOA

3. **Compute TZR phase**:
   ```python
   tzr_dt_sec = TZRMJD_TDB * SECS_PER_DAY - PEPOCH_sec - tzr_total_delay_sec
   tzr_phase = F0 * tzr_dt_sec + F1_half * tzr_dt_sec**2 + F2_sixth * tzr_dt_sec**3
   ```

**Validation**: Residuals should remain unchanged (< 1 ns RMS difference)

---

### TASK 5: Native .par File Parsing
**Status**: PARTIALLY COMPLETE (basic version exists)

**Why this is needed**:
Currently `pint_get_model()` is used to load timing parameters. The existing `parse_par_file()` is basic but functional. May need enhancement for:
- Binary model parameters (ELL1H, DDH, etc.)
- Complex parameter formats
- Error handling

**Priority**: LOW (current parser works for J1909-3744)

---

## Testing Strategy

### For Each Implementation Step:

1. **Implement the feature**
2. **Test against PINT** using validation code:
   ```python
   # Compare JUG vs PINT
   diff = jug_result - pint_result
   print(f"Mean: {np.mean(diff)} | RMS: {np.std(diff)} | Max: {np.max(np.abs(diff))}")
   ```

3. **Verify final residuals**:
   ```python
   jug_residuals = jug_calc.compute_residuals()
   pint_residuals = Residuals(pint_toas, pint_model).calc_phase_resids() / pint_model.F0.value * 1e6
   diff_ns = (jug_residuals - pint_residuals) * 1000
   assert np.std(diff_ns) < 1.0, f"Residual RMS too large: {np.std(diff_ns):.2f} ns"
   ```

4. **User confirmation**: Ask user to create new notebook checkpoint before next step

---

## Key Files and Directories

```
/home/mattm/soft/JUG/
├── residual_maker_playground_active_MK2_clkfix.ipynb  ← Main notebook
├── data/
│   ├── clock/*.clk                     ← Clock correction files
│   ├── observatory/tempo.aliases       ← Observatory aliases
│   └── ephemeris/de440s.bsp           ← JPL ephemeris (used by Astropy)
└── /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/
    ├── J1909-3744_tdb.par              ← Timing parameters
    └── J1909-3744.tim                  ← TOA file (10,408 TOAs)
```

---

## Important Constants

```python
SECS_PER_DAY = 86400.0
C_KM_S = 299792.458
T_SUN_SEC = 4.925490947e-6
AU_KM = 149597870.7
K_DM_SEC = 1.0 / 2.41e-4

T_PLANET = {
    'jupiter': 4.702819050227708e-09,
    'saturn':  1.408128810019423e-09,
    'uranus':  2.150589551363761e-10,
    'neptune': 2.537311999186760e-10,
    'venus':   1.205680558494223e-11,
}
```

---

## Common Pitfalls to Avoid

1. **Don't use Astropy Time objects with JAX** - JAX requires numpy arrays
2. **Preserve float64 precision** - Use `dtype=np.float64` everywhere
3. **Use longdouble for phase calculations** - Phase requires extended precision
4. **Check TDB vs TCB** - .par file may specify UNITS TCB (needs conversion)
5. **Iteration order matters** - TDB calculation uses TDB-based positions (iterative)

---

## Success Criteria

**Final Goal**: Complete independence from PINT while maintaining:
- **Accuracy**: < 1 nanosecond RMS residual difference vs PINT
- **Speed**: Maintain ~1000x speedup over PINT
- **Compatibility**: Parse same .par and .tim files as PINT

---

## Useful PINT Source Code References

For understanding exact implementations:
- **Clock corrections**: `pint/observatory/topo_obs.py` (method `clock_corrections`)
- **TDB calculation**: `pint/observatory/topo_obs.py` (method `get_gcrs_posvel`)
- **Einstein delay**: Search PINT source for "Einstein" or check tempo2 `t2c_doeinstein.c`
- **TZR anchoring**: `pint/models/absolute_phase.py` (method `get_TZR_toa`)

---

## Contact Information

User: mattm
Project: MPTA (MeerKAT Pulsar Timing Array)
Test pulsar: J1909-3744 (binary MSP, excellent for testing)

---

## Current Status Summary

✅ **Complete**:
1. Native .tim file parser
2. Clock correction system (OBS→UTC→TAI→TT→BIPM)

🔨 **In Progress**:
3. TDB calculation (next step - HIGHEST PRIORITY)

⏳ **Pending**:
4. Independent TZR phase anchoring
5. Enhanced .par file parsing (optional)

**Estimated remaining work**: 2-3 iterations to complete full PINT independence.
