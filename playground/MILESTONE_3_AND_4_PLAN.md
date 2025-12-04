# Milestone 3 & 4: Multi-Binary and Multi-Telescope Testing Plan

**Created**: 2025-11-29
**Status**: Planning complete, ready for implementation after Milestone 2

---

## Overview

After completing Milestone 2 (gradient-based fitting), we need to validate JUG works across:
1. **Different binary models** (Milestone 3)
2. **Different telescopes and backends** (Milestone 4)

---

## Milestone 3: Multi-Binary Model Support (v0.3.0)

### Motivation
Currently, JUG has been tested primarily on J1909-3744 (ELL1 binary). We need to ensure it works for:
- Non-binary pulsars
- DD, DDK, BT binary models
- **T2 binary model** (tempo2's general catch-all model - critical!)

### Test Dataset
- **Location**: `/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb`
- **Contents**: 88 MeerKAT pulsars with diverse binary models
- **Note**: All single telescope (MeerKAT), so good for binary diversity but not multi-telescope

### Tasks
1. Test non-binary pulsars (isolated MSPs)
2. Test DD binary model
3. Test DDK binary model
4. **Test T2 binary model** (most important - it's tempo2's universal model)
5. Test BT binary model (already implemented)
6. Verify binary parameter fitting works for all models

### Success Criteria
- ✅ Residuals match PINT for all binary model types
- ✅ Binary parameters can be fitted successfully
- ✅ Non-binary pulsars work correctly

---

## Milestone 4: Multi-Telescope & Multi-Backend Support (v0.4.0)

### Motivation
JUG currently tested only on MeerKAT data. Need to verify:
- Different observatory clock corrections
- Different receiver systems and backends
- Multi-telescope datasets (same pulsar, different sites)

### Test Targets
1. **Parkes** - Australian telescope
2. **WSRT** - European (Netherlands)
3. **ATNF** - Australian telescope array
4. **NANOGrav** - Multiple US telescopes with different backends:
   - GUPPI (Green Bank)
   - ASP (Arecibo - legacy)
   - PUPPI (Arecibo - newer)

### Tasks
1. Test Parkes data - verify clock corrections
2. Test WSRT data - European site
3. Test ATNF data
4. Test NANOGrav data - multiple backends
5. Test multi-telescope datasets - same pulsar, multiple sites
6. Verify all observatory clock files work correctly

### Test Data
- **Location**: TBD (need to find multi-telescope datasets)
- **Requirements**: Pulsars observed from multiple sites

### Success Criteria
- ✅ Works with Parkes, MeerKAT, WSRT, NANOGrav data
- ✅ Clock corrections validated for all sites
- ✅ Multi-telescope datasets combine correctly

---

## Key Implementation Notes

### T2 Binary Model (Critical)
The T2 binary model is tempo2's general binary model that can represent ANY binary system. It's essentially a parameterization that allows arbitrary orbital shapes. We MUST support this because:
- Many tempo2 .par files use it
- It's unlikely to go away
- It's the fallback when other models don't fit well

### Clock Corrections
JUG implements tempo2-style clock chains:
```
Observatory → UTC → GPS/TAI → TT → TDB/TCB
```
We need to verify this works for all major observatories.

### Multi-Telescope Data
When combining data from multiple telescopes:
- Each site has its own clock correction
- Each backend may have different systematics
- TOA flags must properly identify telescope/backend

---

## Timeline

**After Milestone 2 completes**:
1. Milestone 3 (Multi-Binary): 1-2 weeks
2. Milestone 4 (Multi-Telescope): 2-3 weeks

**Priority**: Get Milestone 2 (fitting) working first, then tackle these validation milestones.

---

## Documentation Updated

- ✅ `JUG_PROGRESS_TRACKER.md` - Renumbered milestones 3-12
- ✅ Added detailed tasks for Milestone 3 (binary models)
- ✅ Added detailed tasks for Milestone 4 (multi-telescope)
- ✅ This planning document created

