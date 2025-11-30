# JUG Implementation Handoff - Session 3

## Current Status: Milestone 1 - 95% Complete, Debugging Final 3.4 μs Error

### What We're Doing
Completing Milestone 1 of the JUG (JAX-based pulsar timing) package by achieving PINT-level residual accuracy. We've extracted notebook code into a clean Python package and are now in final precision debugging.

### Critical Discovery Just Made
**TDB computation is PERFECT** (0.0005 μs std difference vs PINT) - the 3.4 μs std error is entirely in the **delay calculation**, not time conversion.

### The Problem (Must Fix NOW)
Testing on J1909-3744 pulsar (10,408 TOAs from MPTA):
- **JUG RMS**: 3.557 μs
- **PINT RMS**: 0.817 μs (our target)
- **Residual Difference Statistics**:
  - Mean: 0.075 μs (✅ excellent - no systematic offset)
  - **Std: 3.435 μs** (❌ TOO HIGH - needs to be <0.1 μs)
  - TDB difference: 0.0005 μs std (✅ perfect)
  
**Interpretation**: The problem is NOT a TZR phase offset (which would show as constant mean difference). It's **TOA-dependent delay calculation differences** causing varying errors across the dataset.

### Recent Progress (This Session)
1. ✅ Fixed major binary delay bugs (reduced RMS from 60 μs → 3.6 μs):
   - Einstein delay: Changed to `gamma * sin_Phi`
   - Shapiro delay: Changed to `-2*r_shap*log(1-s_shap*sin_Phi)`
   - Roemer delay: Added higher-order correction terms
2. ✅ Verified TDB calculation is perfect (matches PINT to 0.0005 μs)
3. 🔍 Currently isolating which delay component has the 3.4 μs variation

### Key Files & Their Status

#### `/home/mattm/soft/JUG/jug/delays/combined.py` (Recently Fixed)
- **Purpose**: JAX kernel for total delay (DM + solar wind + FD + binary)
- **Lines 185-205**: Binary delay calculation (RECENTLY FIXED)
- **Status**: Fixed major bugs, but still has ~3.4 μs std error vs PINT

#### `/home/mattm/soft/JUG/jug/residuals/simple_calculator.py`
- **Purpose**: End-to-end residual calculator
- **Status**: Complete with TZR computation, BIPM2024 clocks
- **Key**: Uses `TZRMJD`/`TZRFRQ` from par file, weighted mean subtraction

#### `/home/mattm/soft/JUG/jug/scripts/compute_residuals.py`
- **Purpose**: CLI entry point
- **Status**: Complete, enabled JAX x64 at import level

#### `/home/mattm/soft/JUG/jug/delays/barycentric.py`
- **Purpose**: Astrometric delays (Roemer, Shapiro)
- **Status**: ✅ Complete and verified correct

### Critical Configuration
```python
jax.config.update("jax_enable_x64", True)  # MUST be enabled at import!
```

### Test Command
```bash
cd /home/mattm/soft/JUG
python -m jug.scripts.compute_residuals \
  /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par \
  /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim
```

### Next Steps (PRIORITY)

1. **Compare Individual Delay Components** (START HERE):
   ```python
   # Get PINT's delay breakdown
   model.delay_funcs  # List all delay components
   # Compare: DM delay, Roemer, Shapiro, binary Roemer, Einstein, Shapiro_binary
   ```

2. **Likely Culprits** (check in this order):
   - Binary orbital delays (J1909-3744 is in tight binary, ELL1 model)
   - DM polynomial terms (we have DM1, DM2)
   - Frequency-dependent delays
   - Higher-order binary terms we might be missing

3. **Reference Implementation**:
   - Original notebook: `/home/mattm/soft/JUG/examples/` (look for compute_tzr_delay_standalone)
   - Check notebook's delay breakdown matches our code

4. **Debugging Strategy**:
   - Extract delays for ~10 TOAs from both JUG and PINT
   - Compare component-by-component to find which varies by 3.4 μs
   - Check formula differences in that component
   - Fix and retest

### User Requirements
- **USER INSISTS**: Must fix NOW, cannot leave for future milestones
- **Target**: Match PINT accuracy (<1 μs RMS, ideally <0.1 μs like notebook's 2.6 ns)
- **Test Data**: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744*`

### Technical Details
- **Observatory**: MeerKAT, ITRF coords: [5109360.133, 2006852.586, -3238948.127] km
- **Clock Files**: `data/clock/tai2tt_bipm2024.clk` (BIPM2024, not 2023!)
- **Ephemeris**: DE440 via Astropy
- **Binary Model**: ELL1 with 3rd-order Fourier expansions
- **Python**: 3.10+ with `np.longdouble` for high precision
- **JAX**: 0.4+ with x64 REQUIRED

### What NOT to Do
- ❌ Don't suggest "leave for future milestone" - user rejected this
- ❌ Don't disable JAX x64 - causes 10-100x worse errors
- ❌ Don't guess at formulas - compare with notebook/PINT line-by-line
- ❌ Don't focus on TDB conversion - it's already perfect

### Validation
When you think you've fixed it:
```bash
# Should see RMS < 1 μs, std of differences < 0.1 μs
python -m jug.scripts.compute_residuals [paths as above]
```

### Key Insight
The mean difference (0.075 μs) being near zero means our TZR phase and systematic delays are correct. The **standard deviation** (3.4 μs) being high means some delay term has **TOA-dependent errors** - likely orbital phase-dependent in the binary model, or frequency-dependent in DM/FD terms.

### Conda Environment
```bash
conda activate discotech  # Has PINT, JAX, etc.
cd /home/mattm/soft/JUG
```

### Questions to Answer
1. Which specific delay component differs by 3.4 μs std between JUG and PINT?
2. Is it in the binary model (orbital phase variations)?
3. Is it in DM derivatives (DM1, DM2 terms)?
4. Is it in frequency-dependent delays?

Start by comparing JUG's and PINT's delay breakdowns for the same TOAs. The smoking gun will be whichever component has 3.4 μs std difference.

**GO FIX IT!** 🚀
