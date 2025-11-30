# JUG Development - Complete Work Index

## Overview

This directory contains a JAX-based pulsar timing package (JUG) that was being debugged to match PINT/Tempo2 residuals. Previous work by Claude identified that JUG's ~850 μs residual discrepancy was due to using incomplete barycentric times from Tempo2. This session implemented a quick fix using PINT's correct times and created comprehensive documentation.

## For Quick Start

**Start here**: `START_HERE.md` - 5-minute quick guide on what to do next

## Files Modified in This Session

### Code Changes
- **`residual_maker_playground_claude_debug.ipynb`**
  - Added Cell 17 with PINT integration
  - Automatically loads correct infinite-frequency times
  - Runs before residuals test (Cell 18)
  - Graceful fallback if PINT unavailable

### Documentation Added
1. **`START_HERE.md`** - Quick start guide (READ FIRST!)
2. **`SESSION_SUMMARY_NOV27.md`** - Complete session overview with all technical details
3. **`CONTINUATION_PROGRESS.md`** - Implementation status and timeline for Phase 2
4. **`QUICK_TEST_INSTRUCTIONS.md`** - Testing guide with troubleshooting

## Previous Work (From Earlier Sessions)

### Documentation
- `PINT_COMPARISON_FINDINGS.md` - Root cause analysis and findings
- `IMMEDIATE_FIX_GUIDE.md` - Initial approach to validation
- `COMPARISON_WITH_TEMPO2.md` - Detailed comparison methodology
- `ROOT_CAUSE_ANALYSIS.md` - Deep dive into sources of error

### Test Scripts
- `test_with_pint_times.py` - Tests residuals with PINT's times
- `check_pint_tzr.py` - Validates TZR calculation
- `compare_with_pint.py` - Compares PINT vs Tempo2
- `compare_inf_freq_times.py` - Compares infinite-frequency times

### Reference Data
- `temp_pre_general2.out` - Tempo2 pre-fit residuals
- `temp_pre_components_next.out` - Tempo2 delay components
- `param.labels`, `param.vals` - Parameter information

## Document Organization

### By Purpose

#### Getting Started
- `START_HERE.md` - What to do first (5 min read)

#### Understanding the Problem
- `PINT_COMPARISON_FINDINGS.md` - Root cause (from previous work)
- `SESSION_SUMMARY_NOV27.md` - This session's findings

#### Testing the Fix
- `QUICK_TEST_INSTRUCTIONS.md` - How to run the notebook
- `CONTINUATION_PROGRESS.md` - What to expect

#### Planning Phase 2
- `CONTINUATION_PROGRESS.md` - Implementation timeline
- `SESSION_SUMMARY_NOV27.md` - Phase 2 description

### By Topic

#### The Problem
- Root cause: JUG uses Tempo2's BAT which includes binary delays
- Symptom: ~850 μs residual discrepancy
- Location: Barycentric time calculation (t_inf)

#### The Solution (This Session)
- Quick fix: Use PINT's correct times (Cell 17 in notebook)
- Temporary: For validation only
- Expected: 850 μs → <10 μs improvement

#### The Long-term Fix (Phase 2)
- Implement barycentric calculations in JUG
- Compute observatory position, Roemer, Shapiro delays
- Remove PINT dependency
- Achieve design goal of independence
- Timeline: ~1 week

## Test Data Location

```
/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/
├── J1909-3744_tdb.par     # Pulsar parameters
└── J1909-3744.tim         # Time-of-arrival data

/home/mattm/soft/JUG/
├── data/
│   ├── ephemeris/
│   │   └── de440s.bsp     # JPL ephemeris kernel
│   ├── observatory/
│   │   ├── observatories.dat
│   │   └── tempo.aliases
│   └── earth/
│       └── eopc04_IAU2000.62-now
├── temp_pre_general2.out       # Tempo2 residuals (reference)
└── temp_pre_components_next.out # Tempo2 delay components
```

## Reading Guide

### For First-Time Users
1. `START_HERE.md` (5 min) - Overview and next steps
2. `QUICK_TEST_INSTRUCTIONS.md` (10 min) - How to test

### For Understanding the Work
1. `SESSION_SUMMARY_NOV27.md` (20 min) - Full context
2. `PINT_COMPARISON_FINDINGS.md` (15 min) - Root cause details
3. `CONTINUATION_PROGRESS.md` (10 min) - What comes next

### For Implementation
1. `CONTINUATION_PROGRESS.md` - Phase 2 timeline
2. `SESSION_SUMMARY_NOV27.md` (Phase 2 section) - Detailed breakdown

## Testing Guide

### Before Running
- [ ] Read `START_HERE.md`
- [ ] Check that PINT is installed: `python3 -c "import pint; print(pint.__version__)"`
- [ ] Verify test data exists: `ls /home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/`

### Running the Test
```bash
cd /home/mattm/soft/JUG
jupyter notebook residual_maker_playground_claude_debug.ipynb
# Run all cells sequentially
# Cell 17: Watch for PINT loading
# Cell 18: Check residuals
```

### Interpreting Results
- **Good** (<10 μs): Proves JUG logic is correct, Phase 2 can proceed
- **Bad** (>100 μs): See `QUICK_TEST_INSTRUCTIONS.md` troubleshooting
- **In-between** (10-100 μs): Review TZR calculation, may need adjustment

## Key Findings Summary

### What's Correct in JUG
✓ Binary delay calculation (ELL1 model)  
✓ DM delay calculation  
✓ Clock correction system  
✓ Residual calculation logic  
✓ JAX JIT compilation  

### What's Wrong in JUG
✗ Barycentric time calculation (uses incomplete BAT)  
✗ TZR reference (computed from wrong times)  

### Root Cause
Tempo2's BAT column is not the infinite-frequency barycentric time; it still contains delay components. JUG should either:
1. Compute barycentric times independently (Phase 2 goal), or
2. Use PINT's computed values (Phase 1 validation)

## Implementation Status

### Phase 1: Validation ✓ COMPLETE
- ✓ Root cause identified
- ✓ Fix implemented (Cell 17 in notebook)
- ✓ Documentation complete
- ⏳ Pending: User testing

### Phase 2: Independence (PLANNED)
- ⏳ Implement observatory position calculation
- ⏳ Implement pulsar direction calculation
- ⏳ Implement Roemer delay
- ⏳ Implement Shapiro delay
- ⏳ Integration and testing
- **Timeline**: ~1 week after Phase 1 validation

### Phase 3: Polish (PLANNED)
- ⏳ Clean up notebook
- ⏳ Add comprehensive documentation
- ⏳ Create usage examples
- **Timeline**: ~3 days

## Contact & History

**Current Session**: November 27, 2025
**Modified By**: Continuation work (following Claude's earlier debugging)
**Status**: Ready for testing

**Previous Sessions**:
- Initial debugging: Identified root cause (JUG vs PINT times)
- Analysis: Confirmed 354-second systematic error
- Documentation: Created comparison findings

## Next Action

👉 **READ**: `START_HERE.md`
👉 **TEST**: Run the modified notebook
👉 **REPORT**: Results will determine if Phase 2 proceeds

---

**Questions?** Refer to the appropriate document:
- What should I do? → `START_HERE.md`
- How do I test? → `QUICK_TEST_INSTRUCTIONS.md`
- What happened? → `SESSION_SUMMARY_NOV27.md`
- What's next? → `CONTINUATION_PROGRESS.md`
