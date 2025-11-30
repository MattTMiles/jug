# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JUG (JAX-based pulsar timing) is a PINT/Tempo2-free pipeline for pulsar timing analysis. The project implements a complete timing pipeline using JAX for fast, JIT-compiled residual computation while parsing `.par` and `.tim` files manually.

**Key Design Philosophy**: This is a minimal, from-scratch implementation that does NOT depend on PINT or Tempo2 libraries. All timing model components are implemented directly using JAX and numpy.

## Core Architecture

### Timing Pipeline Flow

1. **File Parsing** → Parse `.par` (timing parameters) and `.tim` (time-of-arrival data) files
2. **Clock Corrections** → Apply observatory clock corrections via tempo2-style clock chain
3. **Barycentric Corrections** → Transform topocentric times to Solar System Barycenter (SSB)
4. **Binary Delays** → Apply binary companion orbital delays (ELL1/BT models)
5. **Emission Times** → Compute pulsar emission times after binary correction
6. **Dispersion Correction** → Remove cold-plasma DM delay to get infinite-frequency TOA
7. **Phase Residuals** → Compute spin phase residuals using JAX

### Key Components

**Delay Computations** (in order of application):
- **Clock corrections**: Observatory clock → UTC → TAI → TT chain
- **Geometric (Roemer) delay**: Light travel time from observatory to SSB
- **Einstein delay**: TT → TDB/TCB gravitational time dilation
- **Shapiro delay**: Solar system gravitational time delay
- **Binary delays**: Roemer + Einstein + Shapiro from companion star
- **DM delay**: Cold-plasma dispersion delay (K_DM * DM / freq²)
- **FD delays**: Frequency-dependent profile evolution

**Models Implemented**:
- Spin: F0, F1, F2 (spin frequency and derivatives)
- DM: Polynomial DM model (DM, DM1, DM2, ...)
- Binary: ELL1/ELL1H and BT/DD/DDH/DDGR/T2 orbital models
- Astrometry: RA/DEC, proper motion (PMRA/PMDEC), parallax (PX)

### JAX Integration

All timing residual calculations use JAX with `@jax.jit` decorators for performance:
- `spin_phase()`: Compute rotational phase from F0/F1/F2
- `dm_delay_sec()`: Cold-plasma dispersion delay
- `residuals_seconds()`: Full residual calculation

JAX is configured for float64 precision (`jax.config.update('jax_enable_x64', True)`) since pulsar timing requires microsecond precision.

### Data Files Required

The `data/` directory contains essential reference files:

- `data/clock/*.clk` - Observatory clock correction files (tempo2 format)
  - Chain format: `{src}2{dst}.clk` (e.g., `mk2utc.clk`, `utc2tai.clk`)
  - TAI→TT files: `tai2tt_bipm{year}.clk`

- `data/ephemeris/de440s.bsp` - JPL ephemeris kernel (SPK format)
  - Segment [0,3]: Earth-Moon barycenter relative to SSB
  - Segment [3,399]: Geocenter relative to EMB

- `data/observatory/observatories.dat` - Observatory positions (X,Y,Z in meters)
- `data/observatory/tempo.aliases` - Observatory code aliases (tempo format)
- `data/earth/eopc04_IAU2000.62-now` - IERS Earth orientation parameters

### Time Scales & Conventions

**Time Scale Handling**:
- Input TOAs: UTC (MJD)
- After clock correction: UTC → TT
- Barycentric times: TDB or TCB (controlled by UNITS parameter)
- TCB↔TDB conversion: Uses IAU L_B = 1.550519768e-8

**UNITS Parameter**: If `.par` file has `UNITS TCB`, spin parameters (F0/F1/F2) are converted to TDB equivalents by dividing by (1 + L_B). A temporary `temp_model_tdb.par` is written for processing.

**TZR (Time Zero Reference)**: Phase reference anchoring uses TZRMJD/TZRSITE/TZRFRQ parameters. The phase at the TZR TOA is rounded to nearest integer to set `phase_offset_cycles`.

### Observatory Clock Correction System

The clock correction system implements tempo2-style chaining:

1. Resolve observatory code to base name via `tempo.aliases`
2. Walk clock chain: `observatory → UTC → GPS/TAI → TT`
3. Apply linear interpolation between clock file entries
4. Select most recent BIPM file for TAI→TT conversion

Helper functions:
- `resolve_obs()`: Map observatory code to canonical name
- `clock_correction_seconds()`: Compute total correction for given MJD and observatory
- `_clock_sources_for_obs()`: Find all alias candidates for observatory

### Binary Models

**Current Implementation Status** (as of 2025-11-30):

**ELL1/ELL1H** (low-eccentricity) - ✅ IMPLEMENTED:
- Parameters: PB, A1, TASC, EPS1, EPS2, GAMMA, PBDOT, XDOT
- Orthometric parameters (H3/H4) converted to R/S if needed
- Third-order Fourier expansion for high accuracy
- Implemented in: `jug/delays/combined.py` (JAX JIT-compiled)
- **Critical fix** (2025-11-29): Added M2/SINI → r/s Shapiro delay conversion
  - Before: Only recognized H3/STIG parameters
  - After: Also handles M2/SINI (used by J1909-3744)
  - Impact: Reduced residual error from 3.4 μs to 0.003 μs

**BT/DD (Keplerian/relativistic)** - ✅ IMPLEMENTED (not yet integrated):
- **Status**: Implemented in `jug/delays/binary_bt.py` (2025-11-30, Session 6)
- **BT (Blandford-Teukolsky)**: Keplerian + Einstein + Shapiro delays
  - Parameters: PB, A1, ECC, OM, T0, GAMMA, PBDOT, M2, SINI
  - Kepler equation solver using Newton-Raphson (JAX JIT-compiled)
  - Roemer delay from elliptical orbit geometry
  - Einstein delay: GAMMA * sin(E)
  - Shapiro delay: -2*r*log(1 - SINI*sin(ω+ν)) where r = T_☉ * M2
  
- **DD (Damour-Deruelle)**: BT + periastron advance + A1 derivative
  - Additional parameters: OMDOT (deg/yr), XDOT (light-sec/sec)
  - Same function as BT, controlled by OMDOT/XDOT values
  - Variants: DD, DDH, DDK, DDGR all use this implementation
  
- **Testing**: Validated on synthetic data, produces physically reasonable delays
- **Integration**: Pending (requires routing logic in simple_calculator.py)

**T2 (Tempo2 General)** - ✅ IMPLEMENTED (not yet integrated):
- **Status**: Implemented in `jug/delays/binary_t2.py` (2025-11-30, Session 6)
- Universal binary model that can emulate any other binary model
- Extends BT/DD with additional parameters:
  - EDOT: Eccentricity derivative (1/sec)
  - KIN, KOM: 3D orbital geometry (degrees)
- Supports arbitrary parameter subsets with graceful defaults
- Many published .par files use BINARY T2
- **Testing**: Validated against BT model (match to nanosecond precision)
- **Integration**: Pending
  - Essential for broad compatibility

**Implementation Priority**:
1. T2 (most general, highest priority)
2. BT/DD (for Keplerian binaries like J1614-2230)
3. Keep ELL1 for low-eccentricity MSPs

All models return Roemer + Einstein + Shapiro delays in seconds.

## Development Workflow

### Working with the Notebook

The primary development interface is the Jupyter notebook `residual_maker_playground.ipynb`.

**Cell Organization**:
1. Imports and JAX configuration
2. File parsers (`.par` and `.tim` format)
3. Data loading and TCB→TDB conversion
4. Observatory/ephemeris setup
5. Clock correction system
6. Timing model dataclass (`SpinDMModel`)
7. Sky position and barycentric delay calculations
8. Shapiro and Einstein delays
9. Binary models
10. Barycentric/emission time computation
11. FD delay setup
12. JAX residual functions
13. Residual computation and comparison

### Tempo2 Comparison Mode

The code supports reading tempo2 output for validation:
- `temp_pre_components_next.out`: BAT, Roemer, Shapiro components from tempo2 `-output general2`
- `temp_pre_general2.out`: Pre-fit residuals from tempo2
- Set `use_tempo2 = True` to use tempo2 BAT instead of computing from scratch

## Common Tasks

### Computing Residuals

```python
# Requires: loaded .par/.tim files, computed t_em_mjd (emission times)
res_sec = residuals_seconds(t_inf_jax, freq_mhz_jax, model)
res_us = np.array(res_sec) * 1e6  # Convert to microseconds
```

### Adding New Delay Terms

1. Implement delay function (preferably with `@jax.jit`)
2. Add delay to appropriate stage in pipeline (see "Timing Pipeline Flow")
3. Update `SpinDMModel` dataclass if new parameters needed
4. Register with JAX pytree if dataclass modified

### Updating Clock Files

Clock files are in tempo2 format. To update:
1. Download from http://www.atnf.csiro.au/research/pulsar/timing/newclkcorr.html
2. Place in `data/clock/` directory
3. Follow naming convention: `{src}2{dst}.clk`

## Implementation Notes

### Known Issues & Fixes (Production Package)

#### Binary Shapiro Delay Parameterization (Fixed: 2025-11-29)

**Issue**: JUG originally only supported H3/STIG (orthometric) parameters for binary Shapiro delay, but many par files use M2/SINI (mass/inclination) parameterization.

**Impact**: For pulsars with M2/SINI parameters (like J1909-3744), Shapiro delay was incorrectly set to zero, causing residual errors of ~3.4 μs std.

**Fix**: Added automatic conversion in `jug/residuals/simple_calculator.py`:
```python
# Handle both H3/STIG and M2/SINI parameterizations
if 'M2' in params and 'SINI' in params:
    # Convert M2/SINI to r/s
    # r = TSUN * M2 (where TSUN = G*Msun/c^3 = 4.925490947e-6 s)
    # s = SINI
    r_shap = T_SUN_SEC * M2  # range parameter (seconds)
    s_shap = SINI             # shape parameter (dimensionless)
```

**Result**: Reduced residual error from 3.4 μs std → 0.003 μs std vs PINT (1000x improvement).

**Validation**: Tested on J1909-3744 (tight binary MSP, 10,408 TOAs) - now matches PINT exactly.

### Current Limitations (Production Package)

- No JUMP, PHASE, EFAC/EQUAD noise parameters yet (Milestone 3)
- Binary models: ELL1 implemented, BT/DD in progress
- Only single-pulsar mode (no simultaneous multi-pulsar fitting)
- GUI not yet implemented (Milestone 5)

### JAX Pytree Registration

`SpinDMModel` is registered as a JAX pytree to enable JIT compilation of functions that take the model as input. When adding fields to the dataclass, update the pytree registration accordingly.

### Precision Requirements

Pulsar timing requires ~100 nanosecond precision. Always use:
- `dtype=np.float64` for all time arrays
- JAX float64 mode (`jax_enable_x64=True`)
- MJD values propagated without premature rounding

### Constants

Key constants defined in the notebook:
- `C_M_S = 299792458.0` (speed of light, m/s)
- `SECS_PER_DAY = 86400.0`
- `K_DM_SEC = 4.148808e3` (DM constant, MHz² pc⁻¹ cm³ s)
- `L_B = 1.550519768e-8` (IAU TCB-TDB scaling)
- `AU_M = 149597870700.0` (astronomical unit, meters)

## Testing & Validation

Compare JUG residuals against tempo2:
1. Run tempo2 with `-output general2` flag
2. Compare output in `temp_pre_general2.out`
3. Difference plots show JUG - tempo2 residuals

Typical agreement: RMS difference should be << 1 microsecond if implementations match.
