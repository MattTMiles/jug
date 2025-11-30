# JUG Implementation Plan: Complete Overview

## What You Now Understand

You have **complete visibility into PINT's pipeline**:

1. ✅ How .tim files are loaded
2. ✅ How times are converted (UTC → TT → TDB)
3. ✅ How ephemeris data is used
4. ✅ How the timing model is evaluated
5. ✅ How residuals are extracted
6. ✅ Why residuals match PINT perfectly (0.9945 correlation)

## Three Documents Explain Everything

### 1. **PINT_PIPELINE_FLOWCHART.md** (18KB)
   - Complete technical documentation
   - Every step from .par/.tim to residuals
   - Mathematical formulas for each component
   - Parameter tables and explanations
   - Data flow diagrams
   - **Read this for deep understanding**

### 2. **IMPLEMENTATION_GUIDE.md** (20KB)
   - Step-by-step code snippets
   - Python pseudocode for each phase
   - Data structures (TimingModel, TOA)
   - Functions to implement
   - Testing strategy
   - **Read this to start coding**

### 3. **FLOWCHART_SUMMARY.txt** (7KB)
   - Quick reference guide
   - What each component does
   - Key implementation points
   - Success metrics
   - **Read this for quick reference**

## Two Visual Flowcharts

### **PINT_PIPELINE_DIAGRAM.png**
   - Full pipeline from inputs to output
   - Shows all 7 major steps
   - Color-coded phases
   - Component list on the right

### **MODEL_PHASE_COMPONENTS.png**
   - Details of `model.phase()` evaluation
   - What each component contributes
   - Order of operations

## Files for Your Reference

```
/home/mattm/soft/JUG/
├── PINT_PIPELINE_FLOWCHART.md          ← Start here for theory
├── IMPLEMENTATION_GUIDE.md              ← Start here for coding
├── FLOWCHART_SUMMARY.txt                ← Quick reference
├── PINT_PIPELINE_DIAGRAM.png            ← Visual overview
├── MODEL_PHASE_COMPONENTS.png           ← Component details
├── pint_tempo2_residuals_CORRECTED.png  ← Proof (0.9945 correlation)
└── README_JUG_IMPLEMENTATION.md         ← This file
```

## The Implementation Roadmap

### Stage 1: Minimal Working Version (1-2 days)
```
.par ──→ Parse F0, F1, DM, PEPOCH, RA, DEC
.tim ──→ Parse MJD, freq, obs
         ↓
      TDB ≈ MJD_utc (simplified, no correction)
         ↓
      phase = F0*(t-PEPOCH) - K_DM*DM/freq² - Roemer
         ↓
      residual = phase.frac / F0
         ↓
      Compare with PINT
```
**Goal**: Get within 50% of PINT's RMS

### Stage 2: Proper Time Conversions (1-2 days)
```
Add: Clock corrections (UTC → TT)
Add: Simple TDB (TT - constant_offset)
Add: Roemer delay calculation

Goal: Match PINT's residuals to 90% correlation
```

### Stage 3: Full Ephemeris (1-2 days)
```
Add: JPL ephemeris loading
Add: Observatory position in SSB frame
Add: Einstein delay
Add: Shapiro delay (Sun)

Goal: 99% correlation with PINT
```

### Stage 4: Refinement & Optimization (1 day)
```
Add: Binary support (if needed)
Add: DM variations (DM1, DM2)
Add: JAX integration
Add: Error handling

Goal: Production-ready code
```

## Key Implementation Facts

### What's Complex
1. **Clock corrections** - Need to parse tempo2 .clk files
2. **JPL ephemeris** - Need to load DE440 kernel
3. **Einstein/Shapiro delays** - Need relativistic calculations
4. **Binary models** - Need to solve Kepler equation (if binary)

### What's Simple
1. **Parsing .par/.tim** - Just text parsing
2. **DM delay** - One formula: `K_DM * DM / freq²`
3. **Spindown phase** - Three polynomial terms
4. **Residual extraction** - One division: `phase.frac / F0`

### What's Critical
1. **Use TDB times**, not UTC
2. **Handle all delay components** in correct order
3. **Test incrementally** at each stage
4. **Compare with PINT** to validate

## Testing Strategy

**Test 1**: Does spindown-only match PINT spindown?
```python
# Compare when only using F0*(t-PEPOCH)
# Should see variation from ~-1 to +1 μs
```

**Test 2**: Does adding DM help?
```python
# Compare when adding DM delay
# Should reduce RMS from ~2 to ~1.5 μs
```

**Test 3**: Does adding Roemer match?
```python
# Compare when adding astrometric delays
# Should get close to PINT RMS (~0.8 μs)
```

**Test 4**: Full comparison
```python
# Compare all residuals
# correlation should be > 0.99
# RMS should match to 1%
```

## Why This Approach Works

You're not inventing new physics - you're **implementing PINT's proven approach**.

✓ PINT has been tested by the pulsar community for 10+ years
✓ It produces correct residuals (verified against real pulsars)
✓ You understand the complete pipeline now
✓ You can validate at each step
✓ You'll know immediately if something's wrong

## Success Looks Like

When done:
```
$ python jug_pipeline.py J1909-3744.par J1909-3744.tim
✓ Loaded 10408 TOAs
✓ Computed residuals
✓ RMS: 0.8175 μs
✓ Correlation with PINT: 0.9945
✓ All checks passed!
```

## Quick Start Checklist

- [ ] Read PINT_PIPELINE_FLOWCHART.md (30 min)
- [ ] Read IMPLEMENTATION_GUIDE.md (30 min)
- [ ] Write .par/.tim parsers (2 hours)
- [ ] Implement spindown phase (1 hour)
- [ ] Test against PINT spindown (1 hour)
- [ ] Implement DM delay (30 min)
- [ ] Implement clock corrections (3 hours)
- [ ] Implement TDB computation (2 hours)
- [ ] Implement Roemer delay (2 hours)
- [ ] Full test & validation (2 hours)

**Total**: ~15 hours for a working JUG!

## Resources You Have

1. **PINT source code** - `/home/mattm/soft/PINT/src/pint/`
2. **Test data** - J1909-3744.par and .tim files
3. **Reference output** - PINT's residuals (0.8175 μs RMS)
4. **Visual guides** - Flowcharts and diagrams
5. **Detailed documentation** - All three docs above

## Next Steps

1. **Start with parsing** - Get .par and .tim loading working
2. **Build incrementally** - Add one component at a time
3. **Test continuously** - Compare against PINT at each stage
4. **Refactor with JAX** - Once it works, optimize with JAX

## Questions to Answer During Implementation

As you code, these questions should get answered:

- [ ] What columns does the TOA table need?
- [ ] How are clock corrections actually applied?
- [ ] Where does the 30 μs DM delay at 1400 MHz come from?
- [ ] Why does Roemer delay vary by 500 seconds?
- [ ] How does parallax affect the delay?
- [ ] What's the relationship between phase and time residuals?

If you can answer all these, JUG will work!

## Final Thoughts

**You now have everything needed to implement JUG.**

The complete pipeline is documented. Every step has been explained. You have reference implementations (PINT). You have test data. You have expected outputs.

This is the point where understanding ends and implementation begins.

**Good luck!** You've got this. 🚀

---

**Documents Created**: 
- Flowchart (detailed): PINT_PIPELINE_FLOWCHART.md
- Implementation guide: IMPLEMENTATION_GUIDE.md  
- Quick reference: FLOWCHART_SUMMARY.txt
- Visual diagrams: PINT_PIPELINE_DIAGRAM.png, MODEL_PHASE_COMPONENTS.png

**Key Insight**: PINT works, PINT is proven, replicate PINT. Simple as that.
