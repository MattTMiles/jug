# JUG Refactor Worklog — Repo Inventory & Dead Code Signals

**Generated**: 2026-02-13
**Methodology**: Static import analysis (ripgrep), symbol reference counting, pytest runtime signals.
**Constraint**: Read-only audit — nothing deleted or refactored.

---

## 1. Quick Inventory

### Top-Level Directories

| Directory | Purpose | Status |
|-----------|---------|--------|
| `jug/` | Main package — timing engine, fitter, GUI, noise models | Active |
| `tests/` | Top-level integration/regression tests (152 tests) | Active |
| `tools/` | J0125 parity diagnostics, parity harness (7 scripts) | Active but niche |
| `data/` | Pulsar par/tim files, clock files, ephemeris data | Active |
| `clock_files/` | Observatory clock correction files | Active |
| `docs/` | Documentation, progress tracker, guides | Active |
| `notebooks/` | Jupyter notebooks (white noise pint vs jug, etc.) | Active |
| `examples/` | Single notebook (`full_walkthrough.ipynb`) | Stale (1 file) |
| `playground/` | 74 .py scripts, 213 .md files, 91 .png — **138 MB total** | Legacy exploration; mostly dead |
| `archival_code/` | 131 .py files — old fitting/residual code, explicitly archived | Dead (has ARCHIVE_INFO.md) |
| `src/` | Empty wrapper (contains `jug/` symlink or alias) | Vestigial |

### jug/ Subpackages

| Package | Files | Purpose |
|---------|-------|---------|
| `jug/delays/` | 7 | Delay models: barycentric, binary (BT/DD/T2), troposphere, combined |
| `jug/engine/` | 9 | Session management, API facade, noise config, stats, diagnostics, selection, validation |
| `jug/fitting/` | 13 | Optimized fitter, WLS solver, design matrix, derivative modules (spin/DM/astrometry/binary/DD/FD/JUMP/SW), binary registry |
| `jug/gui/` | 12 | PySide6 GUI: main window, theme, widgets (noise panel, averaging, workflow, colorbar), workers (compute/fit/session/warmup) |
| `jug/io/` | 3 | Par reader, tim reader, clock file loader |
| `jug/model/` | 5 | ParameterSpec registry, codecs, DMX support, component graph (spin/dispersion) |
| `jug/models/` | 1 | **Empty package** (`__init__.py` only) |
| `jug/noise/` | 4 | White noise (EFAC/EQUAD), ECORR, red/DM noise, MAP estimator |
| `jug/residuals/` | 2 | `simple_calculator.py` (main), `fast_evaluator.py` (cached re-eval) |
| `jug/scripts/` | 8 | CLI tools: compute residuals, fit params, GUI launcher, benchmarks, comparisons, server |
| `jug/server/` | 2 | FastAPI REST server (`app.py`) for remote access |
| `jug/testing/` | 1 | `fingerprint.py` — residual fingerprinting for parity tests |
| `jug/utils/` | 6 | JAX setup, caching (geom/JAX), astropy config, device selection, constants |
| `jug/tests/` | 44 | Unit/integration tests (466 collected by pytest) |

### Entry Points

| Entry Point | Definition | Module |
|-------------|-----------|--------|
| `jug-gui` | `pyproject.toml` `[project.scripts]` | `jug.gui.main:main` |
| `jug-fit` | `pyproject.toml` `[project.scripts]` | `jug.scripts.fit_parameters:main` |
| `jug-compute-residuals` | `pyproject.toml` `[project.scripts]` | `jug.scripts.compute_residuals:main` |
| `jug-benchmark` | `pyproject.toml` `[project.scripts]` | `jug.scripts.benchmark_stages:main` |
| `jugd` | `pyproject.toml` `[project.scripts]` | `jug.scripts.jugd:main` |
| Library API | `jug.engine.open_session()` | `jug.engine.api` / `jug.engine.__init__` |
| Test runner | `tests/run_all.py` | Script-style runner (not pytest) |
| pytest | `pyproject.toml` `testpaths = ["jug/tests"]` | Standard pytest discovery |

---

## 2. Static Dead-Code Signals

### 2A. Python Files Never Imported

Files under `jug/` (non-test, non-script) that have **zero imports** from any `.py` in the repo:

| File | Size (lines) | Confidence | Why It's Flagged |
|------|-------------|------------|------------------|
| `jug/gui/widgets/averaging_widget.py` | ~100 | **High** | No file imports `AveragingOverlay`. Only reference is *inside* `averaging_widget.py` itself and `selection.py` imports from `engine.selection`. Appears to be an unused GUI widget. |
| `jug/gui/widgets/workflow_panel.py` | ~100 | **Med** | Only imported by `session_workflow.py` (which is itself only imported by `workflow_panel.py` and tests). Circular dead cluster. GUI `main_window.py` never uses it. |
| `jug/models/__init__.py` | 0 | **High** | Empty `__init__.py`. Package is never imported. `jug/model/` (singular) is the real model package. |
| `jug/gui/models/__init__.py` | 0 | **High** | Empty `__init__.py`. Never imported. No files in the package. |

### 2B. Production Modules with Very Narrow Usage

| File | Size | Importers | Confidence | Notes |
|------|------|-----------|------------|-------|
| `jug/fitting/design_matrix.py` | 214 | **0** (within jug/) | **High — dead** | Old design matrix module. Superseded by `optimized_fitter.py`. Only referenced in `archival_code/`. |
| `jug/residuals/fast_evaluator.py` | ~200 | 1 (`session.py:277`) | **Low** | Used for cached re-evaluation. Single caller but on a hot path. |
| `jug/server/app.py` | ~300 | 1 (`jugd.py`) | **Med** | FastAPI server — only used via `jugd` CLI. May be premature/unused if nobody runs the server. |
| `jug/gui/widgets/noise_panel.py` | 152 | 1 (`main_window.py`) | **Low** | `NoiseDiagnosticsDialog` imported dynamically. Active code. |
| `jug/engine/session_workflow.py` | ~200 | 2 (workflow_panel, test) | **Med** | Only consumed by `workflow_panel.py` (itself unused in main_window) and a test. |
| `jug/engine/selection.py` | ~300 | 2 (session_workflow, averaging_widget) | **Med** | Both consumers are themselves marginal. But concept (TOA selection) is used in GUI via different code path. |
| `jug/engine/validation.py` | ~200 | 2 (noise_panel, test) | **Low** | Active — used in diagnostics dialog. |
| `jug/delays/binary_bt.py` | ~150 | 1 (combined.py) | **Low** | BT model — less common but valid binary model. |
| `jug/testing/fingerprint.py` | 122 | 2 (parity_harness, test) | **Low** | Testing utility — used by parity infrastructure. |

### 2C. Unreferenced Public Symbols (Best-Effort)

82 public functions/classes defined in `jug/` (non-test) are never referenced outside their own file. Key clusters:

**Entire file is dead:**
- `jug/fitting/design_matrix.py`: `compute_derivative()`, `get_default_fit_params()` — 0 external refs.

**GUI theme helpers (17 functions):**
- `set_synthwave_variant`, `get_synthwave_accent_primary/secondary`, `get_synthwave_btn_gradient`, `get_synthwave_rms_color`, `get_synthwave_data_colors`, `set_light_variant`, `get_light_data_colors`, `get_current_theme`, `BorderRadius` class, `get_dynamic_accent_secondary`, `get_stats_card_style`, `get_stat_label_style`, `get_stat_value_style`, `get_plot_axis_style`, `get_error_bar_color_for_data`
- Confidence: **Med** — theme functions may be used via `from jug.gui.theme import *` or accessed in QSS strings. Need dynamic analysis.

**Individual binary/fitting derivative helpers (18 functions):**
- `d_phase_d_F0/F1/F2/F3`, `compute_earth_position_angles`, `compute_pulsar_unit_vector`, `d_Phi_d_FBi`, `d_delayR_da1`, various `d_delayR_*` and `d_shapiro_*` functions, `compute_dd_roemer_delay`, `compute_dd_einstein_delay`, `compute_dd_shapiro_delay`, `d_delay_d_DM/DM1/DM2`, `get_fd_derivative_column`
- Confidence: **Med** — these are low-level building blocks. They may be called from within the same file by the top-level `compute_*_derivatives()` dispatcher that IS referenced externally. Need per-function audit.

**Fitter helpers (4 functions):**
- `wls_iteration_jax`, `wls_iteration_numerical`, `fit_wls_numerical`, `fit_wls_jax` in `wls_fitter.py`
- Confidence: **Med** — `wls_solve_svd` from the same file IS used. These 4 may be legacy fitter paths.

**GUI worker signal classes (4):**
- `SessionWorkerSignals`, `WarmupSignals`, `WorkerSignals`, `ComputeWorkerSignals`
- Confidence: **Low — not dead** — these are Qt signal objects used internally by their respective worker classes.

**Utils (5 functions):**
- `device.py`: `get_device_preference`, `estimate_computation_cost`, `should_use_gpu`, `list_available_devices`
- `geom_cache.py`: `get_version_info`
- `jax_setup.py`: `get_jax_info`
- Confidence: **Med** — helper/info functions that may be called from CLI or interactively but not from production code paths.

### 2D. Duplicate / Parallel Implementations

| Old Path | New Path | Evidence | Confidence |
|----------|----------|----------|------------|
| `jug/fitting/design_matrix.py` | `jug/fitting/optimized_fitter.py` (inline) | `design_matrix.py` has 0 imports from `jug/`. `optimized_fitter.py` builds design matrix internally. | **High** — old file is dead |
| `jug/fitting/wls_fitter.py` (full fitters) | `jug/fitting/optimized_fitter.py` | `wls_solve_svd` is still used, but `wls_iteration_jax`, `wls_iteration_numerical`, `fit_wls_numerical`, `fit_wls_jax` have 0 external refs. | **Med** — 4 functions dead, 1 alive |
| `jug/engine/selection.py` + `session_workflow.py` | GUI TOA selection in `main_window.py` | The engine selection module is only used by `workflow_panel.py` (unused widget) and `averaging_widget.py` (unused widget). The GUI has its own selection logic. | **Med** — orphaned code cluster |
| `archival_code/fitting/` (12 files) | `jug/fitting/` | Explicitly archived. Has `ARCHIVE_INFO.md`. | **High** — known dead |
| `archival_code/residuals/core.py` | `jug/residuals/simple_calculator.py` | Explicitly archived. | **High** — known dead |

### 2E. "Old Path vs New Path" Forks

**Fitter paths in `optimized_fitter.py`:**
The fitter has two main entry points that share significant code:
1. `fit_parameters_optimized()` — file-based path (reads par/tim from disk)
2. `fit_parameters_optimized_cached()` — session-based path (uses cached data from GUI)

Both call `_run_general_fit_iterations()` but build the setup differently:
- `_build_general_fit_setup_from_files()` (line 1206) — for file path
- `_build_general_fit_setup_from_cache()` (line 2342) — for session path

**Risk**: These two setup functions duplicate noise-wiring logic (EFAC/EQUAD/ECORR/red/DM) and could diverge. They are ~400 lines each with parallel structure.

**`warmup_worker.py`:**
Referenced in `main_window.py` but only in commented-out code (lines 1547-1548). The worker class exists but is never instantiated.

---

## 3. Runtime Signals

### pytest results (`jug/tests/`, 2026-02-13)

```
Command: python -m pytest jug/tests/ --override-ini="addopts=-v" -q --tb=line
         --ignore=jug/tests/test_data_manifest.py
Result:  21 failed, 429 passed, 28 warnings in 70.20s
```

**Failure categories:**

| Category | Count | Tests | Likely Cause |
|----------|-------|-------|-------------|
| Golden regression | 9 | `test_golden_regression.py` (all 9 tests) | Golden files stale after noise integration changes. Need regeneration. |
| ECORR tests | 8 | `test_ecorr.py` (8 of 14) | Likely API change in `build_ecorr_whitener` during recent refactor. |
| DMX test | 1 | `test_dmx.py::test_frequency_filtering` | Edge case in frequency filtering logic. |
| Parity tests | 2 | `test_parity_smoke.py`, `test_regression_parity.py` | Golden/parity data stale after noise changes. |
| Data manifest | 1 | `test_data_manifest.py` (skipped in run) | Checksum mismatch — data files changed. |

**Warnings (28):**
- 4× `PytestReturnNotNoneWarning` in `test_binary_registry.py` — tests return `True` instead of using `assert`.
- 4× `PytestReturnNotNoneWarning` in `test_stats.py`.
- Remaining: deprecation/config warnings.

### Top-level `tests/` directory (152 tests)
Not run in this audit (separate from `jug/tests/`; configured `testpaths` only points to `jug/tests/`). These are integration tests requiring real pulsar data and are typically run via `tests/run_all.py`.

---

## 4. Ranked Dead-Code Candidates

### Tier 1: Likely Dead — Safe to Remove (High Confidence)

| # | Item | Type | Evidence |
|---|------|------|----------|
| 1 | `jug/fitting/design_matrix.py` | File (214 lines) | 0 imports from any non-archival `.py`. Superseded by `optimized_fitter.py`. |
| 2 | `jug/models/__init__.py` | Empty package | 0 imports. `jug/model/` (singular) is the real package. |
| 3 | `jug/gui/models/__init__.py` | Empty package | 0 imports. No files in the package. |
| 4 | `jug/gui/widgets/averaging_widget.py` | File (~100 lines) | 0 imports from `main_window.py` or any active GUI code. |
| 5 | `playground/` directory | 74 .py, 213 .md, 91 .png = 138 MB | Legacy exploration scripts. None imported by `jug/`. Could archive or `.gitignore`. |
| 6 | `archival_code/` directory | 131 .py files = 1.1 MB | Already marked as archived with `ARCHIVE_INFO.md`. |

### Tier 2: Probably Dead — Needs Confirmation (Medium Confidence)

| # | Item | Type | Evidence | Risk |
|---|------|------|----------|------|
| 7 | `jug/gui/widgets/workflow_panel.py` | File (~100 lines) | Not imported by `main_window.py`. Only by `session_workflow.py` (also marginal). | May be planned feature |
| 8 | `jug/engine/session_workflow.py` | File (~200 lines) | Only imported by `workflow_panel.py` (unused) and its own test. | Same cluster as #7 |
| 9 | `jug/engine/selection.py` | File (~300 lines) | Only imported by `session_workflow.py` and `averaging_widget.py` — both unused. | TOA selection concept is active in GUI via different code. |
| 10 | `jug/fitting/wls_fitter.py` (partial) | 4 functions | `wls_iteration_jax`, `wls_iteration_numerical`, `fit_wls_numerical`, `fit_wls_jax` — 0 external refs. `wls_solve_svd` IS used. | Legacy fitter wrappers. |
| 11 | `jug/server/app.py` | File (~300 lines) | Only used via `jugd` CLI entry point. No evidence anyone runs the server. | Could be feature-planned for Tauri GUI. |
| 12 | `jug/gui/workers/warmup_worker.py` | File (~50 lines) | Referenced only in commented-out code in `main_window.py`. | May be planned feature. |
| 13 | `jug/utils/device.py` (partial) | 4 functions | `get_device_preference`, `estimate_computation_cost`, `should_use_gpu`, `list_available_devices` — 0 refs in production paths. | May be used interactively or from CLI `--gpu` flag. |
| 14 | 17 theme helper functions in `theme.py` | Functions | 0 refs outside `theme.py`. May be consumed via stylesheets or `*` imports. | Need dynamic analysis. |

### Tier 3: Risky Deletions — Dynamic or Lazy Imports

| # | Item | Why Risky |
|---|------|-----------|
| A | `jug/gui/widgets/noise_panel.py` | Imported dynamically: `main_window.py` does `from jug.gui.widgets.noise_panel import NoiseDiagnosticsDialog` inside a method. |
| B | `jug/gui/widgets/noise_control_panel.py` | Imported dynamically in `main_window.py` methods. |
| C | `jug/noise/map_estimator.py` | Imported dynamically inside `main_window.py._on_estimate_noise()`. |
| D | `jug/noise/red_noise.py` | Imported dynamically inside `main_window.py` and `optimized_fitter.py` methods. |
| E | `jug/noise/ecorr.py` | Imported dynamically inside `optimized_fitter.py` method. |
| F | `jug/engine/api.py` | Imported by `jug/engine/__init__.py` — serves as the public API facade. |
| G | `jug/gui/widgets/colorbar.py` | Imported dynamically in `main_window.py`. |
| H | All `derivatives_*.py` modules | Some imported dynamically in `optimized_fitter.py` and `binary_registry.py`. |
| I | `jug/scripts/*.py` | Entry points in `pyproject.toml`. Not imported, but invoked as CLI commands. |
| J | `jug/gui/main.py` | Entry point `jug-gui`. Not imported — run directly. |

---

## 5. Recommended Next Sweep

### Phase 1: Safe Cleanup (no behavioral risk)
1. **Delete `jug/fitting/design_matrix.py`** — 0 imports, fully superseded.
2. **Delete `jug/models/`** — empty package, `jug/model/` is the real one.
3. **Delete `jug/gui/models/`** — empty package, no contents.
4. **Delete `jug/gui/widgets/averaging_widget.py`** — 0 imports from active code.
5. **Archive or .gitignore `playground/`** — 138 MB of legacy exploration. Consider moving to a `playground` branch or adding to `.gitignore`.

### Phase 2: Dead Cluster Removal (confirm with owner)
6. **Remove workflow cluster**: `workflow_panel.py` + `session_workflow.py` + `selection.py` — unless these are planned for a future feature.
7. **Remove legacy fitter functions** from `wls_fitter.py` — keep only `wls_solve_svd`.
8. **Remove or comment `warmup_worker.py`** — only referenced in commented-out code.

### Phase 3: Consolidation (larger refactor)
9. ~~**Unify the two `_build_general_fit_setup_*` paths**~~ ✅ DONE — extracted `_build_setup_common()`, net -217 lines
10. ~~**Regenerate golden test data**~~ ✅ DONE in Phase 0
11. **Audit `device.py` utility functions** — determine if the GPU selection helpers are used by the `--gpu` CLI flag or are dead.
12. **Audit theme.py** — determine which of the 17 unreferenced helper functions are consumed via stylesheet strings vs truly dead.
13. ~~**Consider whether `jug/server/` is a live feature**~~ ✅ KEPT — used by `jugd` CLI entry point.

### Phase 4: Hygiene
14. ~~Fix PytestReturnNotNoneWarning warnings~~ ✅ DONE — 28→3 warnings (3 are PINT's, not ours)
15. ~~Fix `test_dmx.py::test_frequency_filtering` failure~~ ✅ DONE in Phase 0
16. ~~Fix 8 failing ECORR tests~~ ✅ DONE in Phase 0

---

## Completion Summary (2026-02-13)

| Phase | Status | Key result |
|-------|--------|-----------|
| 0 — Baseline fixes | ✅ Complete | 21 failures → 0; 450 passed |
| 1 — Tier-1 safe deletions | ✅ Complete | 5 files/dirs removed |
| 2 — Tier-2 dead code | ✅ Complete | 7 files removed, 4 functions removed; 426 tests |
| 3 — Simplification | ✅ Complete | `optimized_fitter.py` -217 lines (unified fitter setup) |
| 4 — Hygiene | ✅ Complete | 28→3 pytest warnings |

**Final test status**: 426 passed, 0 failed, 3 warnings (all external/PINT)

**Remaining items** (lower priority, not blockers):
- Audit `device.py` GPU helper usage
- Audit `theme.py` unreferenced helpers
- See `docs/CODE_CLEANUP_LOG.md` for full deletion evidence

---

## Appendix: Commands Used

```bash
# File inventory
find jug -name '*.py' -not -path '*/__pycache__/*' | sort

# Entry points
grep -A20 '[project.scripts]' pyproject.toml

# Import analysis (Python script scanning all .py for import patterns)
# See methodology in section 2A

# Unreferenced symbol scan (Python script using regex on all definitions)
# See methodology in section 2C

# Test execution
python -m pytest jug/tests/ --override-ini="addopts=-v" -q --tb=line \
    --ignore=jug/tests/test_data_manifest.py
# Result: 21 failed, 429 passed, 28 warnings in 70.20s

# Size analysis
find jug -name '*.py' -not -path '*/__pycache__/*' -not -path '*/tests/*' \
    -not -path '*/scripts/*' -exec wc -l {} + | sort -rn | head -20
du -sh playground/ archival_code/
```
