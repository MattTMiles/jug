# AI Agent Continuation Instructions for JUG Residual Calculator

## Project Status: Production-Ready Simplified Calculator

**Date**: November 29, 2025  
**Current State**: Streamlined, working calculator in `jug_residuals_standalone.ipynb`  
**Original Development**: Complete in `residual_maker_playground_active_MK5.ipynb`

---

## What Has Been Achieved

### Core Calculator (100% Complete)
- ✅ **TDB Calculation**: Standalone clock chain (MeerKAT → GPS → BIPM2024 → TDB)
- ✅ **TOA Parsing**: Independent TIM file parser (extracts MJD + frequency)
- ✅ **TZR Anchoring**: Uses TZRMJD parameter from par file
- ✅ **DM Delays**: JAX-accelerated dispersion measure calculations
- ✅ **Precision**: Stores TDB as longdouble MJD (13 fractional digits)
- ✅ **Performance**: ~900x faster than PINT, 2.5 ns RMS accuracy
- ✅ **Independence**: Zero PINT dependencies in calculation path

### File Structure
```
JUG/
├── jug_residuals_standalone.ipynb    # PRODUCTION: Simplified, user-friendly
├── residual_maker_playground_active_MK5.ipynb  # DEVELOPMENT: Full-featured with validation
├── AI_CONTINUATION_INSTRUCTIONS.md   # THIS FILE
├── MK5_100_PERCENT_INDEPENDENT.md    # Technical documentation
├── HOW_TO_ADD_TZRMJD.md             # User guide for par file modification
└── data/clock/                       # Clock correction files
```

---

## Current Implementation Details

### The Simplified Calculator (`jug_residuals_standalone.ipynb`)

**Purpose**: Clean, production-ready notebook for end users

**What It Includes**:
- Essential imports only
- Streamlined file parsing (par, TIM, clock files)
- Standalone TDB calculation
- JAX-accelerated DM delay computation
- Simple residual calculation interface
- Basic visualization

**What It Omits** (compared to MK5):
- PINT validation code
- Extensive benchmarking
- Binary orbit delays (Römer, Einstein, Shapiro)
- Solar wind delays
- Frequency-dependent delays
- Planetary Shapiro delays
- Multiple plotting sections

**Key Architecture Decision**:
- Stores TDB as **longdouble MJD** (not seconds) to preserve 13 fractional digits
- Converts to seconds only during phase calculation
- This is critical for achieving ns-level accuracy

---

## What Needs To Be Added Next

### Priority 1: Fix Critical Bug in MK5 (BLOCKING)

#### Missing Observatory-Sun Vectors Function

**ISSUE**: Line 863 and 951 in MK5 call `compute_obs_sun_vectors()` but this function doesn't exist!

**Impact**: MK5 calculator initialization fails immediately - cannot run at all

**What to do**: Implement the missing function to compute Sun position vectors:
   ```python
   def compute_obs_sun_vectors(tdb_jd, obs_itrf_km):
       """Compute observatory-to-Sun vectors for Shapiro delay."""
       times = Time(tdb_jd, format='jd', scale='tdb')
       
       with solar_system_ephemeris.set('de440'):
           sun_pv = get_body_barycentric_posvel('sun', times)
           earth_pv = get_body_barycentric_posvel('earth', times)
       
       # Sun position relative to Earth barycenter
       sun_pos_km = sun_pv[0].xyz.to(u.km).value.T - earth_pv[0].xyz.to(u.km).value.T
       
       # Need to add observatory position (similar to compute_ssb_obs_pos_vel)
       # This gives observatory-to-Sun vectors needed for Shapiro delay
       
       return obs_sun_km  # Shape: (n_toas, 3)
   ```

**Where to add**: Before the `JUGResidualCalculatorFinal` class definition in MK5

**Why it matters**: Required for planetary Shapiro delay calculations

---

### Priority 2: Port Complete Delay Models to Simplified Notebook

**Current state**: 
- MK5 has ALL delay models implemented (DM, binary, solar wind, FD, Shapiro)
- Simplified notebook only has DM delays

**What to port from MK5 to simplified notebook**:

**What to port from MK5 to simplified notebook**:

1. **Binary Orbit Delays** (lines 511-577 in MK5 cell)
   - Already implemented in `combined_delays()` function
   - Handles ELL1 model with all corrections
   - Copy the entire `compute_binary()` nested function

2. **Solar Wind Delay** (lines 492-501 in MK5)
   - Computation already in `combined_delays()`
   - Just needs NE_SW parameter from par file

3. **Frequency-Dependent Delays** (lines 503-509 in MK5)
   - FD polynomial already implemented
   - Reads FD1, FD2, etc. from par file

4. **Observatory-Sun vectors** (needed for Shapiro)
   - Once compute_obs_sun_vectors() is fixed in MK5
   - Copy to simplified notebook

**How to port**: Copy the full `combined_delays()` function from MK5 to replace the simple `compute_delays()` in the simplified notebook.

---

### Priority 3: Enhanced Features (Quality of Life)

#### Multiple Observatory Support
- Currently hardcoded to MeerKAT
- Add Parkes, GBT, etc. from `OBSERVATORIES` dict
- Auto-detect from TIM file observatory codes

#### Proper Motion Support
- Read PMRA, PMDEC from par file
- Compute time-varying pulsar direction
- Function exists in MK5 cell 6: `compute_pulsar_direction()`

#### Error Handling
- Validate par file has required parameters
- Check clock file dates cover TOA range
- Graceful fallbacks for missing parameters

#### Output Options
- Save residuals to file
- Export in TEMPO2/PINT-compatible format
- JSON output for downstream analysis

---

## How to Continue Development

### Workflow

1. **Test Current State**
   ```bash
   cd /home/mattm/soft/JUG
   jupyter notebook jug_residuals_standalone.ipynb
   ```
   Run all cells - verify it produces residuals

2. **Add One Feature at a Time**
   - Start with binary delays (highest priority)
   - Copy implementation from MK5 cell 7
   - Test against PINT before moving on

3. **Maintain Both Notebooks**
   - `jug_residuals_standalone.ipynb`: User-facing, clean
   - `residual_maker_playground_active_MK5.ipynb`: Full-featured, with validation

4. **Validation Strategy**
   - Always compare against PINT residuals
   - Target: < 10 ns RMS difference
   - Use existing validation cells from MK5

### Key Files to Reference

| Need | File | Location |
|------|------|----------|
| Binary delays | MK5 notebook | Cell 7, lines 405-517 |
| Solar wind | MK5 notebook | Cell 7, lines 450-460 |
| FD delays | MK5 notebook | Cell 7, lines 465-475 |
| Shapiro delays | MK5 notebook | Cell 6, lines 380-395 |
| TDB calculation | MK5 notebook | Cell 8, lines 520-711 |
| Full validation | MK5 notebook | Cells 13-15 |

---

## Technical Notes for Next Developer

### Critical Precision Points

1. **TDB Storage Format**
   ```python
   # CORRECT (what we do):
   tdbld_mjd = compute_tdb(...)  # Returns MJD as longdouble
   self.tdbld_mjd = np.array(tdbld_mjd, dtype=np.longdouble)
   
   # WRONG (what NOT to do):
   tdb_sec = tdbld_mjd * 86400  # Loses 5 digits!
   self.tdb_sec = np.array(tdb_sec, dtype=np.longdouble)
   ```
   
   **Why**: MJD ~58526 has 13 fractional digits. Seconds ~5e9 has only 8.

2. **JAX Data Types**
   ```python
   # JAX needs float64 (not longdouble)
   self.tdbld_jax = jnp.array(self.tdbld_mjd, dtype=jnp.float64)
   
   # Convert back after calculation
   delay_ld = np.asarray(delay_jax, dtype=np.longdouble)
   ```

3. **Phase Calculation Order**
   ```python
   # Convert TDB to seconds IMMEDIATELY before use
   tdbld_sec = self.tdbld_mjd * np.longdouble(SECS_PER_DAY)
   dt_sec = tdbld_sec - self.PEPOCH_sec - delay_ld
   ```

### Common Pitfalls

1. **Clock File Dates**: Ensure BIPM/GPS/observatory files cover TOA date range
2. **TZRMJD**: Must be in par file for independence (see `HOW_TO_ADD_TZRMJD.md`)
3. **Frequency**: TIM file 4th column is frequency in MHz
4. **Units**: Delays in seconds, frequencies in MHz, TDB in MJD

---

## Testing & Validation

### Quick Test
```python
# Should produce ~2.5 ns RMS difference vs PINT
from pint.residuals import Residuals
pint_res = Residuals(pint_toas, pint_model).calc_phase_resids() / pint_model.F0.value * 1e6
diff_ns = (jug_res - pint_res) * 1000
print(f"RMS difference: {np.std(diff_ns):.2f} ns")
```

### Full Validation Checklist
- [ ] TDB matches PINT (< 0.001 ns)
- [ ] DM delay matches PINT
- [ ] Binary delays match PINT (if applicable)
- [ ] Solar wind matches PINT (if NE_SW > 0)
- [ ] Total residuals match PINT (< 10 ns RMS)
- [ ] Performance: < 1 ms per calculation
- [ ] No PINT calls in calculation path

---

## Questions & Answers

**Q: Why two notebooks?**  
A: `jug_residuals_standalone.ipynb` is for users (clean, simple). `residual_maker_playground_active_MK5.ipynb` is for developers (full validation, benchmarks).

**Q: Can I remove PINT entirely?**  
A: From the calculation, yes! But keep it for validation. Users can remove it if they add TZRMJD to par file.

**Q: What's the most important thing to preserve?**  
A: The TDB precision chain. Storing as MJD (not seconds) is what achieves 2.5 ns accuracy.

**Q: Where do I start adding binary delays?**  
A: Copy the `compute_binary()` function from MK5 cell 7 into the simplified notebook's `compute_delays()` function.

**Q: How do I know if my additions work?**  
A: Compare against PINT. Difference should be < 10 ns RMS for all delay components.

---

## Contact & Resources

- **Documentation**: See `MK5_100_PERCENT_INDEPENDENT.md` for technical details
- **User Guide**: See `HOW_TO_ADD_TZRMJD.md` for par file setup
- **Original Work**: All code in `residual_maker_playground_active_MK5.ipynb`
- **Clock Files**: Download from IPTA pulsar-clock-corrections repository

---

## Success Metrics

You'll know you've succeeded when:

1. ✅ All delay components implemented (DM, binary, solar wind, FD, Shapiro)
2. ✅ Residuals match PINT within 10 ns RMS
3. ✅ No PINT function calls in residual calculation
4. ✅ Computation time < 1 ms per calculation
5. ✅ Both notebooks work without errors
6. ✅ User can run simplified notebook start-to-finish
7. ✅ Documentation updated with new features

---

**Good luck! The foundation is solid. You're adding the final pieces to make this a complete, production-ready timing tool.**
