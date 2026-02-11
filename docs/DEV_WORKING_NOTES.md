# JUG Development Working Notes

## Current Goal / Active Task
Phases 0–4 complete. GUI integration and remaining phases pending.

## Repo State Summary
- **Branch:** `main`  
- **Latest commit:** `08cf5c5` — "Jitting some stuff" (plus uncommitted roadmap additions)

## Test Status
- **Full suite:** 426 passed, 2 skipped, 28 warnings (61s)
- **New tests:** 113 across 6 test files

## Completed Checklist
- [x] Phase 0.1: `jug/engine/validation.py` — TOA data integrity checks
- [x] Phase 0.2: `jug/engine/flag_mapping.py` — flag aliasing + user mapping
- [x] Phase 0.3: `jug/engine/diagnostics.py` — noise/backend diagnostics
- [x] Phase 1.1: ECORR integration (already done in existing code)
- [x] Phase 1.2: `jug/noise/red_noise.py` — Fourier-basis red/DM noise
- [x] Phase 2.1: `jug/model/dmx.py` — DMX support
- [x] Phase 3.1: `jug/engine/selection.py` — interactive selection + averaging
- [x] Phase 4.1: `jug/engine/session_workflow.py` — undo/redo + JSON snapshots

## Remaining
- [ ] Phase 5: GUI integration (noise toggle, DMX viewer, selection)
- [ ] Phase 6: Extended binary models / astrometry refinements
- [ ] Phase 7: Advanced diagnostics (residual colormaps, H-test)

## How to Resume
1. Activate environment: `mamba activate discotech`
2. Run tests: `cd /home/mattm/soft/JUG && python -m pytest jug/tests/ -o 'addopts=' -q`
3. Continue with Phase 5 (GUI integration)
