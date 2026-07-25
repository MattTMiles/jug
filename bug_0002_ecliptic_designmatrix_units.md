# BUG 002: Ecliptic astrometry autodiff design-matrix columns get RAJ/DECJ unit scales

**Status:** confirmed independently and amended (2026-07-23); BUG 001 is fixed at the tested revision  
**Severity:** high for any ecliptic (`ELONG`/`ELAT` or `LAMBDA`/`BETA`) user of JUG autodiff design matrices / `JaxTimingState.design_matrix`  
**Component:**  
- `jug/utils/units.py` (`native_derivative_to_fit_column`, `native_to_fit_value`, `fit_to_native_value`)  
- `jug/model/parameter_spec.py` (`LAMBDA` aliased to `RAJ`, `BETA` aliased to `DECJ`)  
- `jug/fitting/jax_residual_delta.py` (`compute_autodiff_designmatrix_from_setup`)  
- `jug/fitting/jax_timing_state.py` (`export_jax_timing_state` design-matrix assembly)  
**JUG revision tested:** `b95b658af6914c3990833997564da3d5f6a277be` (`b95b658`), package version `0.1.0`, branch `tempo2-dev`  
**Reporter context:** MetaPulsar / nltiming IPTA DR2 Kepler Model D sampling on NANOGrav 9-yr **J0023+0923** (ecliptic ELL1). Downstream symptom: NUTS step size collapsed to ~1e−7–1e−9 with accept≈0.85 when sampling all timing axes with `design_matrix_method="autodiff"`.

---

## Summary

For a real ecliptic pulsar, JUG’s **autodiff design-matrix columns** for ecliptic longitude/latitude disagree with:

1. the host **PINT** design matrix (`MetaPulsar.Mmat`, timing package `pint`), and  
2. JUG’s own **`jacfwd(residual_delta_jax)`** at the same reference,

by **exact constant scale factors**:

| Parameter (host / mapped backend) | Scale \|PINT\| / \|JUG autodiff design\| | Constant |
|---|---|---|
| `ELONG` → backend `LAMBDA` | **3.8197186342** | `12/π` = `HOURANGLE_PER_RAD` (RAJ fit↔native) |
| `ELAT` → backend `BETA` | **57.2957795131** | `180/π` = `RAD_TO_DEG` (DECJ fit↔native) |

After removing the per-column mean, cosine similarity is **1.000000** for both columns — pure scale error, not shape/physics error.

All other fitpars on this pulsar (PX, PMELONG/PMELAT, F0/F1, DM/DM1/DM2, PB/A1/TASC/EPS1/EPS2, JUMP, FD1, Offset) agree between PINT and JUG autodiff design to ~1e−8…1e−15 after mean subtraction.

Critically, **the residual Jacobian is correct**. Only the **exported design matrix** is wrong. That splits log-likelihood geometry from prior/WLS/cheat-prior geometry whenever consumers use `design_matrix` / `linearized_design_matrix()` for widths while differentiating residuals for the likelihood.

An independent reproduction on JUG's bundled real
`ng5_j1600_tdb_ecliptic_cross_engine` fixture gives the same exact factors
when the public exported columns are compared directly with the raw Jacobian
inside the same simplified-autodiff calculation. This isolates the fault to
the final unit conversion; neither a PINT comparison nor the native/full
Tempo2 residual graph is required to establish the bug.

---

## Why this is a bug (contract)

JUG’s public design-matrix contract is: columns are in **API fit units**, with

\[
M \approx -\frac{\partial r}{\partial \theta_{\mathrm{fit}}}
\]

and RAJ/DECJ specially converted between native radians and fit hourangle/degrees via `jug.utils.units`.

Registry facts that make the current behavior inconsistent:

1. **`ELONG` / `ELAT` are first-class parameters** with fit unit **`deg`** (`parameter_spec._resolve_fit_unit`: `"ELONG": "deg"`, `"ELAT": "deg"`).  
2. The registry comment on RAJ explicitly says *“ELONG is a separate param (ecliptic fitting)”*.  
3. But **`LAMBDA` is registered as an alias of `RAJ`**, and **`BETA` as an alias of `DECJ`**:

```278:289:ref-packages/jug/jug/model/parameter_spec.py
        aliases=("LAMBDA",),  # ELONG is a separate param (ecliptic fitting)
    ...
        aliases=("BETA",),  # ELAT is a separate param (ecliptic fitting)
```

4. Unit helpers canonicalize first, then apply equatorial-only rules:

```71:120:ref-packages/jug/jug/utils/units.py
def native_to_fit_value(param_name: str, native_value: float) -> float:
    param = _canonical_param(param_name)  # LAMBDA → RAJ, BETA → DECJ
    ...
    if param == "RAJ":
        return native * HOURANGLE_PER_RAD
    if param == "DECJ":
        return native * RAD_TO_DEG
    return native

def native_derivative_to_fit_column(param_name: str, col_native) -> float:
    param = _canonical_param(param_name)
    ...
    if param == "RAJ":
        scale = 1.0 / HOURANGLE_PER_RAD
    elif param == "DECJ":
        scale = 1.0 / RAD_TO_DEG
    return col_native * scale
```

So for ecliptic names:

| Name passed in | After `canonicalize_param_name` | `fit_unit` / deriv scale applied | Physically correct for ecliptic? |
|---|---|---|---|
| `ELONG` | `ELONG` | identity (deg↔deg) | yes |
| `ELAT` | `ELAT` | identity | yes |
| `LAMBDA` | **`RAJ`** | **hourangle** (`12/π`) | **no** — LAMBDA is degrees |
| `BETA` | **`DECJ`** | **deg via `180/π`** | wrong *application* when column is already degree-basis (see mechanism) |

Autodiff design export always runs that helper on the **backend** names:

```649:675:ref-packages/jug/jug/fitting/jax_residual_delta.py
def compute_autodiff_designmatrix_from_setup(...):
    ...
    for col, param in enumerate(fit_params):
        public_native_col = -jac_native[:, col]
        cols.append(native_derivative_to_fit_column(param, public_native_col))
```

And `export_jax_timing_state` builds the stored state design from **mapped** backend names while residual deltas use the same mapping:

```187:246:ref-packages/jug/jug/fitting/jax_timing_state.py
jug_fit_params = [mapping.get(name, name) for name in fit_params]
...
design_matrix = -_compute_designmatrix_from_setup(setup, jug_fit_params)
for col, name in enumerate(fit_params):
    design_matrix[:, col] = _fit_unit_column_to_native_delta(name, design_matrix[:, col])
```

On J0023 the live mapping was:

```text
ELONG → LAMBDA
ELAT  → BETA
PMELONG → PMLAMBDA
PMELAT  → PMBETA
DM_ng9 → DM
...
```

Leaf `JugEngine._native_scale` for host names `ELONG`/`ELAT` is all ones (because `native_to_fit_value("ELONG")` is identity). So `linearized_design_matrix = state.design_matrix / 1` **preserves** the wrongly scaled LAMBDA/BETA columns that autodiff export already wrote.

**Contract violation:** for ecliptic longitude, fit unit is degrees (same as `ELONG`), not RAJ hourangle. Applying `1/HOURANGLE_PER_RAD` to a LAMBDA column is incorrect. The measured ratios match that mistake exactly.

---

## Observed numerical evidence (J0023+0923)

**Setup**

- Data: IPTA DR2 path, PTA `ng9` only  
- Host: MetaPulsar `combination_strategy="shared"` (strips DMX; unrelated to this scale bug)  
- `n_TOA = 4373`, `n_fit = 18`  
- Fitpars include `ELONG`, `ELAT` (PINT names)  
- Engines: discovery/all-JUG, `design_matrix_method="autodiff"`, `tempo2_native="fixed_state_stripped"`  
- Environment: singularity `cuda13`, `jug 0.1.0` at `b95b658`

**Mean-subtracted column norms**

```text
ELONG: |Mmat|=3.733466e+02  |autodiff design|=9.774192e+01  |-jacfwd(residual)|=3.733466e+02
       |Mmat|/|autodiff| = 3.8197186193   (= 12/π)
       |-jac|/|autodiff| = 3.8197186342
       |Mmat|/|-jac|     = 0.9999999961

ELAT:  |Mmat|=4.750897e+01  |autodiff design|=8.291879e-01  |-jacfwd(residual)|=4.750897e+01
       |Mmat|/|autodiff| = 57.2957804969  (= 180/π)
       |-jac|/|autodiff| = 57.2957795131
       |Mmat|/|-jac|     = 1.0000000172
```

**Leaf engine state (direct inspection)**

```text
jug_fitpars: includes ELONG, ELAT (host names)
param_mapping: ELONG→LAMBDA, ELAT→BETA, ...
native_scale: all 1.0 for jug_fitpars
state.design_matrix norms: ELONG≈9.79e1, ELAT≈8.31e-1
  (== linearized_design_matrix; no further rescaling)
```

**Unit helper smoke check (same env)**

```text
native_to_fit_value("ELONG", 1) = 1.0
native_to_fit_value("ELAT", 1)  = 1.0
native_to_fit_value("LAMBDA", 1) = 3.8197...   # wrongly RAJ
native_to_fit_value("BETA", 1)   = 57.2957...  # wrongly DECJ
native_derivative_to_fit_column("LAMBDA", [1]) = 0.2618... = π/12
native_derivative_to_fit_column("BETA", [1])   = 0.01745... = π/180
```

**Interpretation of the two factors**

- Autodiff jac for these ecliptic axes is already on a **degree** basis consistent with PINT/`residual_delta_jax`.  
- `native_derivative_to_fit_column("LAMBDA")` still multiplies by `π/12` (RAJ hourangle), shrinking the column by `12/π`.  
- `native_derivative_to_fit_column("BETA")` multiplies by `π/180` (DECJ), shrinking by `180/π`.  
- Residual path does **not** apply that conversion to host `ELONG`/`ELAT` deltas (`native_scale=1`), so `-∂r/∂θ` matches PINT.

### Independent bundled-fixture reproduction

On the bundled real `ng5_j1600_tdb_ecliptic_cross_engine` fixture (a DD
binary), compare:

1. the mean-subtracted raw `-jacfwd` columns returned by the same
   `delay_model="simplified"` calculation, before
   `native_derivative_to_fit_column`, and
2. `compute_autodiff_designmatrix_from_setup` after that conversion.

```text
LAMBDA:
  |raw -jacfwd| = 1.521549279808038e+02
  |exported|    = 3.983406699599811e+01
  ratio         = 3.819718634205487 = 12/pi
  cosine        = 1.0

BETA:
  |raw -jacfwd| = 2.667679698956164e+01
  |exported|    = 4.655979413539620e-01
  ratio         = 57.295779513082316 = 180/pi
  cosine        = 1.0000000000000002
```

The corresponding current helper results are:

```text
ELONG  -> canonical ELONG, fit unit deg,       value scale 1,            derivative scale 1
ELAT   -> canonical ELAT,  fit unit deg,       value scale 1,            derivative scale 1
LAMBDA -> canonical RAJ,   fit unit hourangle, value scale 12/pi,        derivative scale pi/12
BETA   -> canonical DECJ,  fit unit deg,       value scale 180/pi,       derivative scale pi/180
```

---

## Mechanism (call chain)

```text
MetaPulsar fitpar "ELONG" (deg)
    │
    ├─ param_mapping ──► backend name "LAMBDA"
    │
    ├─ residual_delta_jax path
    │     delta_ELONG / native_scale(ELONG=1)  →  residual_fn(jug_fit_params includes LAMBDA)
    │     jacfwd(residual) matches PINT Mmat          ✅
    │
    └─ design_matrix path
          compute_autodiff_designmatrix_from_setup(..., fit_params=["LAMBDA", ...])
              jac_native = jacfwd(simplified residual_delta)(0)
              col = native_derivative_to_fit_column("LAMBDA", -jac_native[:,i])
                    └─ canonicalize LAMBDA → RAJ
                    └─ scale *= 1/HOURANGLE_PER_RAD     ❌
          export_jax_timing_state stores that column on JaxTimingState.design_matrix
          JugEngine.linearized_design_matrix: divide by native_scale(ELONG)=1
          → still wrong by 12/π
```

Same chain for `ELAT` → `BETA` → canonicalize `DECJ` → `1/RAD_TO_DEG`.

Secondary metadata bug: `fit_unit("LAMBDA")` / `get_fit_unit("LAMBDA")` resolve through the RAJ alias and report **`hourangle`**, while ecliptic longitude is degrees. `validate_column_units` / column-unit strings for LAMBDA are therefore also wrong.

---

## What is *not* broken (same pulsar / same run)

- PINT `Mmat` vs `engine.design_matrix()` (LinearModel copy of host Mmat): **identical**.  
- PINT `Mmat` vs `-jacfwd(residual_delta_jax)` after mean subtract: **identical** for ELONG/ELAT.  
- Autodiff vs PINT for non-ecliptic-position params after mean subtract: **OK**.  
- Proper-motion ecliptic columns `PMELONG`/`PMELAT` on this pulsar: **OK** after mean subtract (mas/yr path; PMLAMBDA→PMRA alias did not reproduce the same failure here).  
- This is **not** BUG 001 (DD analytic vs autodiff Kepler). J0023 is ELL1; binary columns match.

---

## Impact

Consumers that trust `JaxTimingState.design_matrix` / `compute_autodiff_designmatrix_from_setup` / nltiming `design_matrix_method="autodiff"` for:

- WLS / cheat-prior σ widths,  
- chart / coordinate prior boxes,  
- any metric built from design columns rather than `jacfwd(residual_delta)`,

get ELONG/ELAT prior scales wrong by `12/π` and `180/π`.  

In nltiming Model D (`TimingInference.sample_all()` + `joint_model`), ELONG/ELAT are sampled: likelihood uses correct residual gradients, priors use wrong autodiff widths → pathological ξ curvature / NUTS dual averaging to tiny step sizes with still-high acceptance. Model B often delta-flats ELONG/ELAT, so it can hide the bug.

Equatorial pulsars (`RAJ`/`DECJ` only) are unaffected by this specific alias mistake.

---

## Suggested fix (for JUG maintainers)

Minimum correct behavior:

1. **Do not treat `LAMBDA`/`BETA` as RAJ/DECJ for unit conversion.**  
   Options:
   - Remove `LAMBDA`/`BETA` from RAJ/DECJ aliases and give them (or route them through) ELONG/ELAT specs with fit unit `deg`; or  
   - Special-case ecliptic names in `native_to_fit_value` / `fit_to_native_value` / `native_derivative_to_fit_column` / `fd_step_in_fit_units` **before** or **instead of** RAJ/DECJ alias handling.

2. **For the current JUG implementation, use identity conversion for all four
   ecliptic position names.** `par_reader` stores `LAMBDA`/`BETA` in degrees,
   the `ELONG`/`ELAT` specs have `internal_unit="deg"`, and the ecliptic
   residual calculation differentiates with respect to those degree-valued
   parameters. Therefore:

   - `native_to_fit_value`, `fit_to_native_value`, and
     `native_derivative_to_fit_column` should all be identity for
     `LAMBDA`, `BETA`, `ELONG`, and `ELAT`;
   - `fd_step_in_fit_units` should likewise use a degree-valued step;
   - do **not** apply `π/180` here unless JUG first changes these parameters
     to radians throughout storage and differentiation. Applying it only at
     the export boundary would preserve the present BETA error and replace
     the LAMBDA error with a different degree/radian mismatch.

3. Align `fit_unit("LAMBDA")` with degrees (same as ELONG), not hourangle.

4. Ensure `export_jax_timing_state`’s pair  
   `(native_derivative_to_fit_column on jug/backend names)` +  
   `(_fit_unit_column_to_native_delta on host/canonical names)`  
   is consistent under `ELONG↔LAMBDA` mapping (today host-side `_fit_unit_column_to_native_delta("ELONG")` is a no-op while backend-side already applied RAJ scaling).

---

## Recommended regression tests

CI currently covers equatorial astrometry units (`tests/test_astrometry_designmatrix_units.py` uses `ASTROMETRY_EXPORT_PARAMS = ("RAJ","DECJ",...)`) and has ecliptic residual parity tests, but **does not gate autodiff design columns for LAMBDA/BETA/ELONG/ELAT against residual jacfwd or against ELONG degree units**.

Add gates roughly like:

1. **Unit helper invariants**

```python
assert native_to_fit_value("LAMBDA", 1.0) == native_to_fit_value("ELONG", 1.0) == 1.0
assert native_to_fit_value("BETA", 1.0) == native_to_fit_value("ELAT", 1.0) == 1.0
assert fit_unit("LAMBDA") in {"deg", "degrees"}
assert fit_unit("LAMBDA") != "hourangle"
assert abs(native_derivative_to_fit_column("LAMBDA", 1.0)
           - native_derivative_to_fit_column("ELONG", 1.0)) < 1e-15
assert abs(native_derivative_to_fit_column("BETA", 1.0)
           - native_derivative_to_fit_column("ELAT", 1.0)) < 1e-15
```

2. **Bundled real ecliptic pulsar.** Prefer
   `ng5_j1600_tdb_ecliptic_cross_engine`, which is already shipped with JUG
   and is also a DD binary:

```python
M_auto = compute_autodiff_designmatrix_from_setup(setup, ["LAMBDA", "BETA"])
J = -jacfwd(the_same_simplified_residual_core)(0)
# mean-subtract both
for name in ("LAMBDA", "BETA"):
    assert relative_rms(M_auto[name], J[name]) < 1e-8
    assert abs(median_abs_ratio(M_auto[name], J[name]) - 1.0) < 1e-8
```

3. **End-to-end mapped-name gate.** Exercise
   `export_jax_timing_state` with host names `ELONG`/`ELAT` mapped to backend
   names `LAMBDA`/`BETA`. Assert that `state.design_matrix`,
   `linearized_design_matrix()`, and `-jacfwd(residual_delta_jax)` have unit
   ratios after the applicable mean projection. Helper-only tests will not
   protect the mapping/export interaction that caused the downstream
   failure.

4. **Cross-check vs PINT** on an NG ecliptic par/tim (optional external gate):
   the ratios must be 1, not `12/π` / `180/π`.

5. **Negative control:** equatorial RAJ/DECJ columns must still match existing
   `test_astrometry_designmatrix_units.py` (do not break RAJ hourangle).

6. **BUG 001 DD regression:** reuse the bundled J1600 DD fixture, or add the
   original J1640 case, to compare analytic and simplified-autodiff
   PB/A1/T0/ECC/OM columns. The current binary regression only covers an
   ELL1 case (`PB`, `EPS1`, `EPS2`), so the original DD failure is fixed but
   not directly protected by that test.

---

## Minimal reproduction sketch

```python
# Inside the project singularity env, jug at b95b658
# Build any ecliptic session where fit params include ELONG/ELAT or LAMBDA/BETA,
# design_matrix_method="autodiff".

import numpy as np
import jax
import jax.numpy as jnp
from jug.utils.units import HOURANGLE_PER_RAD, RAD_TO_DEG
from jug.utils.units import native_derivative_to_fit_column

# A) unit helper (no pulsar needed)
assert abs(native_derivative_to_fit_column("LAMBDA", 1.0) - (1.0/HOURANGLE_PER_RAD)) < 1e-12
# ↑ this assertion documents CURRENT (buggy) behavior; after fix it should match ELONG/deg policy

# B) on JaxTimingState / JugEngine for J0023:
# compare mean-subtracted columns:
#   host PINT Mmat["ELONG"] / state.design_matrix["ELONG"] == HOURANGLE_PER_RAD
#   host PINT Mmat["ELAT"]  / state.design_matrix["ELAT"]  == RAD_TO_DEG
#   (-jacfwd(residual_delta_jax)["ELONG"]) / state.design_matrix["ELONG"] == HOURANGLE_PER_RAD
```

Concrete host used here:

- Par/tim: IPTA DR2 NANOGrav 9-yr J0023+0923  
- MetaPulsar shared combination, clock dir `$TEMPO2/clock`  
- Mapping observed: `ELONG→LAMBDA`, `ELAT→BETA`

---

## Relation to BUG 001

| | BUG 001 | BUG 002 (this) |
|---|---|---|
| Pulsar | J1640+2224 DD | J0023+0923 ELL1 ecliptic |
| Failure | analytic Kepler ≠ autodiff Kepler | autodiff ecliptic scale ≠ residual/PINT |
| Shape | bad relative structure | perfect shape, wrong constant scale |
| Constants | N/A | exactly `12/π`, `180/π` |
| Residual jacfwd | (analytic path issue) | **correct** vs PINT |
| Status at `b95b658` | **fixed** | **confirmed, still present** |

The bugs are orthogonal. BUG 001 has been fixed at the tested revision by
aligning the public analytic and autodiff design matrices on the simplified
timing tangent. A real NANOGrav 9-year J1640 DD check of PB/A1/T0/ECC/OM gives
relative differences of approximately `2.3e-5` to `5.8e-5`, within JUG's
`1e-4` delay-column tolerance, with cosine similarities above
`0.999999998`.

This specifically establishes the current **public design-matrix contract**.
The public autodiff design matrix now deliberately uses
`delay_model="simplified"`; it is not a claim that the analytic matrix equals
every native/full Tempo2 graph tangent.

---

## Downstream note (nltiming / MetaPulsar; not JUG-owned)

nltiming with `design_matrix_method="autodiff"` installs `engine.linearized_design_matrix()` into `ctx.design_matrix` for priors/charts, while `local_timing_block()` / `W_z` comes from `jacfwd(residual_delta_jax)`. On ecliptic Model D runs those two disagree on ELONG/ELAT by the factors above. Fix belongs in JUG unit/alias handling so residual and design export obey one fit-unit contract.
