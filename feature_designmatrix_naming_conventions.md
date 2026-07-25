# Feature: one explicit JUG design-matrix contract

**Status:** revised proposal (not implemented)  
**Package:** `jug` (`ref-packages/jug`)  
**Related:** `nltiming` redesign in `ref-packages/nltiming/bug_designmatrix.md`  
**Backward compatibility:** not required. JUG, nltiming, MetaPulsar, and the
validation callers are controlled together. Prefer a clean break over aliases.

---

## 0. Executive decision

JUG will expose one meaning for every public object named `design_matrix`:

```text
M_raw[:, p] = derivative of the unprojected timing-model prediction
              with respect to parameter p
```

More concretely, let \(d(\theta)\) be the absolute modeled timing correction
(or the equivalent time-domain correction obtained from the phase model for
spin/phase parameters), before residual mean removal or Offset fitting. Then:

```text
M_raw = ∂d / ∂θ_fit
```

The public parameter coordinates and column units are always the existing
**PINT–Vela fit-unit convention**, for both computational compatibility modes.
For example, RAJ uses hourangle and DECJ uses degrees at the public boundary.

The computational physics is selected independently:

| `compatibility` | Computation differentiated by `method="autodiff"` |
|---|---|
| `"pint"` | JUG's existing PINT-style JAX timing path |
| `"tempo2"` | The JUG tempo2 JAX path selected by the session's `tempo2_native` value |

`fixed_state_stripped` remains the normal/default tempo2 mode and its demonstrated
accuracy is sufficient reason not to build a new full PINT graph. It is not,
however, hard-coded by design-matrix construction. If the user selected
`fixed_state_bclt`, `staged_bclt`, `full`, or another supported mode, autodiff
uses that exact selected graph.

Residual centering, row ordering, weighted-mean selection, and the implicit
fitter Offset are operations applied **after** construction of `M_raw`. They
must not be baked irreversibly into the only public design matrix. TZR anchoring
is not a fixed projection; its PINT-compatible special status is specified in
§5.

Analytic and autodiff are independent construction methods:

- `analytic` uses JUG's analytic derivative blocks, including their intentional
  local/simplified formulas;
- `autodiff` differentiates the selected existing JAX computational path;
- analytic autodiff of the simplified formulas may exist as a private test
  oracle, but it is not the public autodiff implementation;
- no acceptance criterion requires analytic and autodiff matrices to be
  numerically identical.

---

## 1. Current problem

JUG's public design-matrix code already describes its columns as the fitter
timing basis:

```text
M ≈ -∂r_unprojected/∂θ_fit
```

This agrees with \(M_\mathrm{raw}=\partial d/\partial\theta_\mathrm{fit}\) when
an increase in modeled timing correction decreases the corresponding raw
residual.

But `export_jax_timing_state` negates the matrix again:

```python
# Current code in jug/fitting/jax_timing_state.py
design_matrix = -np.asarray(
    _compute_designmatrix_from_setup(setup, jug_fit_params), dtype=np.float64
)
```

The field called `JaxTimingState.design_matrix` therefore contains a residual
Jacobian for most timing parameters rather than a design matrix. Downstream,
nltiming combines those columns with exact-linear JUMP/DMX/Offset columns taken
from a fitter matrix, producing a mixed-sign object.

There are several related problems:

1. `design_matrix` is used for both a fitter matrix and a residual Jacobian.
2. The public autodiff builder explicitly selects
   `delay_model="simplified"` instead of differentiating the JAX path selected
   by the session compatibility mode.
3. `JaxTimingState` accepts `design_matrix_method`, allowing the state to export
   one nonlinear graph and an unrelated tangent construction.
4. Export-time mean removal conflates a raw timing-model derivative with a
   particular residual projection.
5. Tempo2 computations are returned in PINT–Vela units, but documentation
   sometimes describes that as PINT computational semantics rather than a
   deliberate public-unit convention.
6. The two current autodiff builders have identical simplified-path bodies,
   obscuring the intended distinction between public computational autodiff and
   the private analytic-formula oracle.

This proposal fixes all six issues together.

---

## 2. Mathematical contract

### 2.1 Raw timing-model design matrix

For public fit coordinates \(\theta_\mathrm{fit}\), define:

\[
M_\mathrm{raw}
  := \frac{\partial d(\theta)}{\partial\theta_\mathrm{fit}}.
\]

Here \(d\) means the backend-selected absolute timing-model prediction expressed
as a time correction:

- propagation and binary parameters use their modeled delay;
- spin and phase parameters use the equivalent time-domain correction obtained
  from the phase model;
- JUMP and DMX use their modeled time/phase correction with the same public sign
  convention;
- TZR parameters define the absolute-phase reference, but—following PINT—they
  are not ordinary design-matrix coordinates; see §5.

For an unprojected residual convention \(r_\mathrm{raw}=t_\mathrm{obs}-d\), the
local identity is:

\[
\Delta r_\mathrm{raw} \approx -M_\mathrm{raw}\,\delta\theta_\mathrm{fit}.
\]

Code that needs the residual derivative must call it
`residual_jacobian`; it must never place that derivative in a field named
`design_matrix`.

### 2.2 Residual projection is separate

Let \(P\) describe the residual operation selected by a session:

- weighted mean removal on the PINT path;
- unweighted mean removal on the tempo2 path;
- optional row reordering;
- any other explicitly documented linear residual post-processing.

`ResidualProjection` contains exactly:

```python
@dataclass(frozen=True)
class ResidualProjection:
    mean_mode: Literal["none", "weighted", "unweighted"]
    mean_weights: np.ndarray | None  # frozen normalized export-time weights
    row_indices: np.ndarray | None   # output rows drawn from session TOA order
```

“Weighted” here means weighted **mean removal**. `ResidualProjection` does not
whiten rows, multiply by TOA uncertainties, apply EFAC/EQUAD, subtract TZR
phase, or add an Offset column. The weights are frozen at state export; later
noise-parameter changes do not mutate the timing state.

Let \(A_\mathrm{TZR}\) be the Jacobian of the row-constant absolute-phase
reference term that PINT deliberately omits from `model.designmatrix`. Then the
exact exported residual Jacobian is:

\[
\Delta r_\mathrm{export}
  \approx P((-M_\mathrm{raw}+A_\mathrm{TZR})\,\delta\theta),
\qquad
J_\mathrm{export}=P(-M_\mathrm{raw}+A_\mathrm{TZR}).
\]

When `subtract_tzr=False`, \(A_\mathrm{TZR}=0\).

For weighted or unweighted mean removal, \(P A_\mathrm{TZR}=0\), so the familiar
identity \(J_\mathrm{export}=-P M_\mathrm{raw}\) holds. With
`mean_mode="none"`, it need not hold; this is a documented consequence of
following PINT's option-(b) design matrix. Code needing the exact tangent uses
`state.residual_jacobian_native()`, obtained from the selected residual graph.

Current production state export always resolves to `"weighted"` (PINT) or
`"unweighted"` (tempo2); it does not currently produce `"none"`. The enum value
is reserved for a future explicit uncentered/PHOFF-compatible export. Its
algebra and unit tests are retained now so adding that path cannot silently
reinstate the false unconditional \(M=-J\) identity.

The projected matrix, when needed, is:

\[
M_\mathrm{projected}=P M_\mathrm{raw}.
\]

It is derived data, not a second sign convention. APIs must name the projection:

```python
state.projected_design_matrix()
state.linearized_model_residual_delta_fit(delta)
state.linearized_residual_delta_native(delta)
```

The model helper evaluates:

```python
state.residual_projection.apply(-state.design_matrix @ delta)
```

The exact residual helper evaluates `state.residual_jacobian_native() @ delta`
and includes the TZR-reference response. The two are equal after ordinary mean
removal; tests cover the uncentered exception explicitly.

### 2.3 Public units and internal storage

All public JUG design matrices use PINT–Vela fit units, regardless of
`compatibility`:

```text
compatibility="pint"    -> PINT-style computation, PINT–Vela public units
compatibility="tempo2"  -> tempo2-style computation, PINT–Vela public units
```

Internal JAX storage may continue to use radians or other native numeric units.
That conversion is a JUG implementation detail and should be centralized in
`jug.utils.units`/parameter metadata.

This proposal does **not** introduce a general consumer-coordinate transform in
JUG. MetaPulsar/nltiming may transform JUG's PINT–Vela columns into sampling
charts or global parameter coordinates, but that is downstream functionality
and belongs outside this feature.

The only conversion in scope here is:

```text
JUG internal/native parameter delta <-> JUG public PINT–Vela fit delta
```

Do not duplicate hard-coded RAJ/DECJ scaling in multiple layers. Prefer a
single parameter conversion API used for both values and derivative columns.

---

## 3. Compatibility contract

### 3.1 Two computational paths, one public unit convention

`compatibility` selects the computational formulas, conventions, and numerical
values. It does not select public design-matrix units.

#### PINT-compatible path

- Residuals/TOAs use JUG's existing PINT-style implementation.
- Public autodiff differentiates that existing JAX implementation.
- No new full PINT graph is required by this feature.
- Analytic derivatives may use their existing simplified/local formulas.
- PINT is the numerical oracle for supported quantities, subject to documented
  differences between PINT's analytic derivative and differentiation of JUG's
  computational graph.

#### tempo2-compatible path

- Residuals/TOAs use the tempo2-compatible JAX computation.
- Public autodiff uses exactly `setup.tempo2_native`, inherited from the session.
- If the caller did not select a mode, normal session default resolution applies
  (`fixed_state_stripped` today).
- Design-matrix construction must not override the selected mode, build a
  second tempo2 graph, or silently fall back to a different mode.
- Tempo2/libstempo is the numerical oracle for supported quantities.
- Output columns are converted to PINT–Vela fit units at the JUG boundary.

### 3.2 Contradictions and known differences

The following are not silently normalized away:

1. If a backend's analytical design-matrix convention differs from the
   derivative of its computational path, record the discrepancy by parameter
   and method.
2. If PINT and tempo2 compute different numerical derivatives for the same par
   file, each compatibility mode follows its own computational oracle.
3. Public units remain PINT–Vela even when tempo2 uses different native
   coordinates. Tests must apply the documented tempo2-to-public scale before
   comparing numerical columns.
4. If a parameter is implemented analytically but not in the selected JAX graph,
   `method="autodiff"` raises an informative unsupported-parameter error. It
   must not fall back to the analytic formula.
5. If a parameter is implemented in the JAX graph but lacks an analytic
   derivative, `method="analytic"` raises. It must not silently invoke autodiff.

This feature must add a small structured capability manifest covering graph
mode, supported autodiff parameters, supported analytic parameters, and the
oracle/tolerance used for each certified family. It is the source for
unsupported-parameter errors and parameterized routing tests, rather than
duplicated prose-only lists.

---

## 4. Autodiff and analytic construction

### 4.1 Public autodiff means the selected JAX computation

`compute_autodiff_designmatrix_from_setup` must stop forcing
`delay_model="simplified"`.

Conceptually:

```python
def compute_autodiff_designmatrix_from_setup(setup, fit_params):
    timing_delta_fn = make_timing_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        # Uses setup.compatibility and setup.tempo2_native.
    )
    jac_native = jacfwd(timing_delta_fn)(zeros)
    return convert_native_derivative_columns_to_fit(
        fit_params, jac_native  # implemented via native_derivative_to_fit_column
    )
```

The implementation should reuse the same cached JAX bundle as nonlinear timing
evaluation. In particular:

- PINT sessions differentiate the current PINT-style JAX computation;
- tempo2 sessions differentiate the graph named by `setup.tempo2_native`.

Autodiff adds no new tempo2 payload or lifecycle rule. It uses the same
`native_chain_static`/`term_diagnostics` payload already required by that
session's nonlinear residual graph. If normal state export has to populate that
payload, it does so once using the existing session behavior; matrix
construction neither forces a different mode nor triggers a second recompute.
If the configured graph cannot be built from a valid prepared session, fail
with the existing graph-setup error rather than silently changing modes.

The one-shot API must expose the same choice:

```python
compute_designmatrix(
    ...,
    compatibility="tempo2",
    tempo2_native=None,  # normal default resolution, or an explicit mode
)
```

Because it has no pre-existing prepared session, the one-shot call prepares the
selected graph and required payload once as part of its normal computation.
This is not a fallback or hidden mode change. Session/state APIs reuse their
already prepared graph instead.

The graph should expose the unprojected timing/phase correction before mean
removal. If the current graph only returns a centered residual delta, refactor
the final stage into:

```python
raw_timing_delta = compute_raw_timing_delta(...)
raw_residual_delta = -raw_timing_delta
export_residual_delta = residual_projection.apply(raw_residual_delta)
```

Autodiff design-matrix construction differentiates `raw_timing_delta`.
Nonlinear likelihood evaluation returns `export_residual_delta`.

This is a factoring change, not a request for a new physical model.

### 4.2 Analytic construction remains independent

`assemble_analytic_designmatrix` remains the analytic implementation:

```python
M_analytic_raw = assemble_analytic_designmatrix(
    setup,
    fit_params,
    output_units="fit",
)
```

It should return raw, uncentered columns.

The public contract does not say:

```text
M_analytic == M_autodiff
```

They may agree closely for some parameters, but they are intentionally
independent algorithms. Tests of the analytic implementation may differentiate
the simplified analytic formulas using a separately and explicitly named
private helper, for example:

```python
compute_simplified_analytic_autodiff_oracle(...)
```

That helper must not be called by the public autodiff route.

### 4.3 `design_matrix_method` does not belong in the frozen state

Remove `design_matrix_method` from `export_jax_timing_state`.

A frozen JAX state represents one selected computational graph. Its
`design_matrix` is therefore the autodiff matrix of that graph, not a
method-selected unrelated snapshot.

Session-level APIs may still accept:

```python
compute_designmatrix(..., design_matrix_method="analytic")
compute_designmatrix(..., design_matrix_method="autodiff")
```

But:

- `analytic` dispatches to analytic assembly;
- `autodiff` dispatches to the selected session JAX graph;
- both return raw fitter/timing-prediction columns in PINT–Vela fit units;
- neither changes residual mean handling or adds an Offset column.

---

## 5. PINT-compatible TZR and Offset policy

### 5.1 Choose PINT's option: TZRMJD is model state, not a fit coordinate

JUG follows PINT's existing split:

- `TZRMJD`, `TZRSITE`, and `TZRFRQ` are real timing-model state defining a
  special reference TOA and the absolute-phase anchor;
- PINT's `AbsPhase` component registers no phase derivative for `TZRMJD`;
- consequently `TZRMJD` is not a PINT `fittable_param` and has no column in
  `model.designmatrix`;
- JUG's raw public design matrix likewise does not support `TZRMJD`,
  `TZRSITE`, or `TZRFRQ` as fit columns.

This is option (b): the raw design matrix differentiates the TOA timing/phase
prediction used by PINT's fitter matrix, excluding the parameter-dependent TZR
reference subtraction. TZR anchoring remains part of absolute residual
calculation, but is not represented as a fixed `ResidualProjection` and is not
silently assigned a zero column. A request to fit a TZR parameter raises the
same class of informative “not fittable/no derivative” error as PINT.

This is an intentional timing-package inconsistency that consumers must be able
to see:

```text
TZRMJD is genuine model state, but not a design-matrix coordinate.
```

Because changing the TZR reference adds a parameter-dependent constant to phase
columns, projected residual tests alone cannot validate this choice: mean
removal would hide it. The acceptance suite therefore compares the **raw F0
column** with `PINT model.designmatrix(..., incoffset=False)` and verifies that
the pre-TZR timing derivative—not a TZR-referenced residual derivative—was
used.

### 5.2 The invisible Offset follows PINT numerically

When no explicit `PhaseOffset` component is present, PINT prepends the synthetic
`Offset` coordinate to `model.designmatrix(..., incoffset=True)`. In the vendored
PINT implementation its time-domain column is:

```python
M[:, 0] = +1.0 / F0.value
label = "Offset"
```

JUG must match that numerical column, sign, unit, label, and **leading-column
position** exactly. The current JUG autodiff helper's `-1.0` column is not the
contract and must be removed.

PINT reports the Offset unit as `u.s / u.s`, even though the physical
interpretation of the numerical column can invite different bookkeeping.
JUG and its tests copy the unit returned by PINT; they must not “correct” the
oracle from dimensional intuition.

Therefore:

- raw timing-parameter matrices exclude implicit Offset by default;
- `include_offset_column=True` prepends, rather than appends, the PINT Offset;
- adding Offset does not modify any timing-parameter column;
- Offset is never obtained by perturbing `TZRMJD`;
- if an explicit PINT-compatible `PHOFF` model component is present, no implicit
  Offset is added and `PHOFF` is an ordinary differentiable timing parameter;
- projection/centering may make the implicit constant direction degenerate,
  and rank handling belongs to the fitter.

Recommended metadata:

```python
column_kind = "timing_parameter" | "implicit_offset"
```

This is the one intentional exception to “every design-matrix column is the
derivative with respect to an ordinary par-file fit parameter.”

---

## 6. Target API

### 6.1 `DesignMatrixResult`

Extend or clarify the result object:

```python
@dataclass(frozen=True)
class DesignMatrixResult:
    matrix: np.ndarray
    labels: tuple[str, ...]
    column_units: tuple[str, ...]
    column_kinds: tuple[str, ...]
    unit_convention: Literal["pint-vela"]
    compatibility: Literal["pint", "tempo2"]
    method: Literal["analytic", "autodiff"]
    tempo2_native: str | None
```

`DesignMatrixResult` always contains the raw, unprojected matrix. Projected
matrices are returned only by explicitly named state helpers and do not create
a second `DesignMatrixResult` variant.

### 6.2 `JaxTimingState`

The frozen state should contain:

```python
@dataclass(frozen=True)
class JaxTimingState:
    fit_params: tuple[str, ...]
    param_mapping: tuple[tuple[str, str], ...]
    ref_params: dict[str, object]
    ref_theta_native: np.ndarray
    reference_residuals_sec: np.ndarray
    subtract_tzr: bool
    design_matrix: np.ndarray
    column_units: tuple[str, ...]
    compatibility: str
    tempo2_native: str | None
    residual_projection: ResidualProjection
    setup: Any
    _raw_timing_delta_jax_fn: Any
    _residual_delta_jax_fn: Any
```

As today, `@dataclass(frozen=True)` is only shallow immutability: dictionaries
and NumPy arrays stored in the state remain mutable objects. This proposal does
not strengthen that behavior into deep immutability.

Contract:

```text
state.design_matrix:
    raw autodiff timing-model matrix at the export reference;
    PINT–Vela public fit units;
    session TOA row order;
    no mean removal, row reordering, or implicit Offset.

state.raw_timing_delta_jax(delta_native):
    raw timing-model delta in internal/native parameter coordinates.

state.residual_delta_jax(delta_native):
    projected nonlinear residual delta used by likelihood consumers.
```

Because the state graph accepts native deltas while the public matrix uses fit
units, linearized helpers must perform the one authoritative JUG
fit-to-native/native-to-fit conversion rather than relying on callers to know
the distinction.

Suggested helpers:

```python
def projected_design_matrix(self) -> np.ndarray:
    return self.residual_projection.apply_matrix(self.design_matrix)

def linearized_model_residual_delta_fit(self, delta_fit):
    return -(self.projected_design_matrix() @ delta_fit)

def residual_jacobian_native(self):
    return jacfwd(self.residual_delta_jax)(zeros_native)

def linearized_residual_delta_native(self, delta_native):
    return self.residual_jacobian_native() @ delta_native
```

If native-delta helpers remain public, include `_native` in their names.

#### Migration of current state fields

| Current field | Revised destination |
|---|---|
| `fit_params` | keep |
| `param_mapping` | keep; required by nltiming aliases |
| `ref_params` | keep |
| `ref_theta` | rename `ref_theta_native` and document units |
| `reference_residuals_sec` | keep |
| `subtract_tzr` | keep as residual-evaluation configuration, not projection |
| `compatibility` | keep |
| `phase_mean_mode` | move into `residual_projection.mean_mode` |
| `isort` | move into `residual_projection.row_indices`; update consumers |
| `design_matrix` | keep name, but change sign, projection, and RAJ/DECJ numbers as specified |
| `column_units` | keep; now unambiguously describes public fit-unit columns |
| `setup` | keep for the current nltiming/diagnostic consumers |
| `_residual_delta_jax_fn` | keep and add `_raw_timing_delta_jax_fn` |

The `design_matrix` unit migration is intentional and breaking: it currently
stores native-delta columns after `_fit_unit_column_to_native_delta`; after this
feature it stores PINT–Vela fit-unit columns. RAJ/DECJ numerical values therefore
change even independently of the sign and projection fixes.

### 6.3 Naming rules

- `design_matrix` means raw timing/fitter matrix \(M\), never residual Jacobian.
- `projected_design_matrix` means \(P M\), with the same sign and units.
- `residual_jacobian` means \(\partial r/\partial\theta\).
- `linearized_model_residual_delta` applies \(-PM\) and follows the fitter
  matrix.
- `linearized_residual_delta` uses the exact residual Jacobian, including the
  uncentered TZR-reference term.
- “analytic” and “autodiff” describe construction, not signs or projections.
- “native” in a name means JUG internal parameter units, not tempo2
  compatibility.

---

## 7. Concrete implementation changes

### 7.1 `jug/fitting/jax_residual_delta.py`

1. Factor the selected JAX computation into an unprojected timing-delta core and
   a residual-projection wrapper.
2. Make the public autodiff builder use the session-selected graph; remove its
   forced `delay_model="simplified"`.
3. Delete the duplicated simplified body from
   `compute_autodiff_designmatrix_from_setup`; retain one separately named
   private simplified-formula oracle only for analytic tests.
4. Share the session cache between nonlinear evaluation and autodiff. A PINT
   state should not build a second graph solely for its public matrix; neither
   should a tempo2 state.
5. Read `setup.tempo2_native` without overriding it. Normal session default
   resolution remains authoritative.
6. Convert derivative columns from internal/native parameter units to
   PINT–Vela fit units exactly once at the public boundary.
7. Do not demean columns. Prepend the PINT-compatible Offset only when
   explicitly requested by a fitter-facing wrapper.
8. Reject TZR parameters as non-fittable design-matrix coordinates.
9. Add the structured method/graph/parameter capability manifest required by
   §3.2.

### 7.2 `jug/fitting/jax_timing_state.py`

1. Remove `design_matrix_method` from `export_jax_timing_state`.
2. Build `state.design_matrix` from the selected graph's raw autodiff matrix.
3. Remove the extra export negation.
4. Stop modifying the raw matrix by weighted/unweighted column-mean removal.
5. Store the residual projection separately and apply it only in residual and
   projected-matrix helpers.
6. Replace hard-coded duplicate unit scaling with the authoritative unit
   conversion helper.
7. Explicitly change `state.design_matrix` from native-delta column units to
   PINT–Vela fit-unit columns; do not preserve the old RAJ/DECJ numbers.
8. Split the helpers explicitly: the fitter-model helper uses the projected
   matrix with a minus; the exact residual helper uses the selected graph's
   residual Jacobian and retains any uncentered TZR-reference response.
9. Update docstrings to distinguish public fit coordinates from state-native
   deltas.
10. Keep or migrate every current state field according to §6.2; update
    nltiming consumers of `isort`/`phase_mean_mode`.
11. Remove `design_matrix_method` from the state-export setup plumbing, not just
    from the function signature. `GeneralFitSetup.design_matrix_method` remains
    only for session-level `compute_designmatrix` dispatch.

### 7.3 `jug/fitting/optimized_fitter.py`

1. Keep session-level `design_matrix_method`.
2. Ensure both methods return raw, unprojected matrices.
3. Add/pass through `tempo2_native` on the one-shot `compute_designmatrix` API
   and resolve it exactly as session construction does.
4. Update `compute_designmatrix` documentation:

   ```text
   compatibility selects computational physics.
   unit_convention is always "pint-vela".
   method="analytic" and method="autodiff" are independent algorithms.
   ```

5. Do not claim the tempo2 analytic matrix is tempo2-native merely because the
   session uses `compatibility="tempo2"`. Document the analytic implementation
   actually used.
6. Move mean subtraction, Offset augmentation, and rank handling into explicit
   fitter preparation stages.

### 7.4 `jug/utils/units.py` and parameter metadata

1. Retain `unit_convention="pint-vela"` package-wide.
2. Centralize parameter-value and derivative-column conversions.
3. Make conversions alias-aware through canonical parameter metadata.
4. Validate labels and column units together.
5. Avoid a second conversion implementation in `JaxTimingState`.

### 7.5 Documentation cleanup

Search for and correct:

```bash
rg -n \
  "design_matrix.*(residual|Jacobian)|plain matmul|delay_model=\"simplified\"|pint-vela" \
  jug tests README.md PARITY_*.md
```

No public docstring may:

- call `design_matrix` a residual Jacobian;
- imply autodiff means autodiff of the analytic approximation;
- imply tempo2 compatibility changes the public unit convention;
- imply raw matrix construction includes residual centering;
- describe `TZRMJD` as not being a timing-model parameter.

---

## 8. Tests

Tests should be organized around raw timing derivatives. Avoid correlation-only
acceptance tests; compare arrays with explicit absolute/relative tolerances,
labels, ordering, units, and column kinds.

### Test 1 — Analytic raw delay derivative

For each supported parameter:

```python
M = compute_designmatrix(..., method="analytic")
assert M.unit_convention == "pint-vela"
assert M.column_kinds == ("timing_parameter", ...)
```

Compare to the appropriate analytic/backend oracle where one exists. Do not
compare to the public autodiff builder as the definition of correctness.

### Test 2 — PINT-path autodiff differentiates the selected JAX path

```python
state = export_jax_timing_state(pint_session, fit_params=fit_params)
J_timing_native = jacfwd(state.raw_timing_delta_jax)(zeros_native)
M_expected = column_stack([
    native_derivative_to_fit_column(p, J_timing_native[:, i])
    for i, p in enumerate(fit_params)
])
assert_allclose(state.design_matrix, M_expected, ...)
```

Also perform small central finite differences of the same raw JAX timing
computation. The test must not call the simplified analytic autodiff oracle.

For a raw TZR-sensitive column, especially F0, also compare against
`PINT model.designmatrix(..., incoffset=False)` before any mean projection.
This pins the option-(b) TZR definition and would fail if autodiff instead used
the derivative of a TZR-referenced residual.

### Test 3 — tempo2 autodiff preserves the configured graph

```python
for configured_mode in supported_test_modes:
    session = open_session(..., tempo2_native=configured_mode)
    graph_builds_before = graph_build_counter(session)
    state = export_jax_timing_state(session, fit_params=fit_params)
    assert state.tempo2_native == configured_mode
    assert state.setup.tempo2_native == configured_mode
    assert selected_graph_cache_key(state) == configured_mode
    assert graph_build_counter(session) - graph_builds_before <= 1

    builds_after_export = graph_build_counter(session)
    state.residual_delta_jax(zeros_native)
    assert graph_build_counter(session) == builds_after_export
```

The test must demonstrate that matrix construction did not replace the selected
mode, build a second graph, or perform a matrix-specific recompute. Use a real
build/cache counter rather than inferring this only from the cache key. At least
the normal `fixed_state_stripped` mode
gets full raw-column comparison against tempo2/libstempo for the named
certification parameters in the capability manifest (initially F0, RAJ, DECJ,
and DM), after documented PINT–Vela unit conversion. Tolerances live in that
manifest rather than as vague global correlation thresholds. Other configured
modes receive routing/cache tests and their own oracle comparisons where
already certified.

Also parameterize the one-shot path:

```python
for configured_mode in supported_test_modes:
    result = compute_designmatrix(
        par,
        tim,
        fit_params,
        compatibility="tempo2",
        tempo2_native=configured_mode,
        design_matrix_method="autodiff",
    )
    assert result.tempo2_native == configured_mode
    assert one_shot_graph_build_count() == 1
```

This closes the separate pass-through/preparation contract of §4.1 and §7.3;
session-only tests are insufficient.

### Test 4 — Analytic and autodiff are independent

Replace tests named or documented as:

```text
autodiff builder still equals analytic
```

with two independent tests:

- analytic agrees with its analytic/backend oracle;
- autodiff agrees with finite differences of the selected JAX computation.

A diagnostic comparison between the two is allowed, but it must not be a
required equality unless a parameter-specific compatibility fact explicitly
requires it.

Add a routing test that fails if public autodiff calls the simplified analytic
autodiff helper.

### Test 5 — Projection happens after raw construction

For both compatibility modes:

```python
M_raw = state.design_matrix
M_projected = state.projected_design_matrix()
expected = projection.apply_matrix(M_raw)
assert_allclose(M_projected, expected)

delta = small_fit_delta
assert_allclose(
    state.linearized_model_residual_delta_fit(delta),
    -(M_projected @ delta),
)
```

Assert that `M_raw` itself is unchanged by weighted versus unweighted mean
selection.

### Test 6 — Nonlinear small-step residual identity

For graph-supported parameters:

```python
delta_native = small_native_delta
r = state.residual_delta_jax(delta_native)
r_linear = state.residual_jacobian_native() @ delta_native
assert_allclose(r, r_linear, parameter_specific_tolerance)
```

For weighted/unweighted mean modes, additionally assert that the exact tangent
equals the appropriately unit-converted \(-P M_\mathrm{raw}\). For
`mean_mode="none"`, assert that their difference is the row-constant TZR
reference response rather than incorrectly requiring equality. This tests the
exported residual behavior without defining the raw design matrix through a
TZR-referenced or demeaned residual.

### Test 7 — PINT TZR/Offset behavior

Required assertions:

1. `TZRMJD`, `TZRSITE`, and `TZRFRQ` requests are rejected as non-fittable
   design-matrix coordinates.
2. The raw F0 column matches `PINT model.designmatrix(..., incoffset=False)`.
3. `include_offset_column=False` is the raw matrix default.
4. `include_offset_column=True` prepends a column labeled `Offset`, with
   `column_kind="implicit_offset"`.
5. Numerically assert the Offset column is `+1/F0` with the same unit as the
   vendored PINT oracle (`u.s/u.s`, quirks included); do not merely test that it
   is constant or substitute a physically “cleaner” unit.
6. Assert the remaining timing columns are byte-for-byte/order-identical to the
   no-Offset result.
7. With explicit `PHOFF`, assert no implicit Offset is present and the PHOFF
   column matches PINT.
8. Projection/rank tests may show that the implicit constant Offset is
   degenerate after mean removal; that is expected fitter behavior.

### Test 8 — Sign regression and downstream handoff

At the JUG boundary:

```python
assert design_matrix is not residual_jacobian
if state.residual_projection.mean_mode != "none":
    assert_allclose(
        residual_jacobian_fit,
        -state.projected_design_matrix(),
    )
else:
    assert_row_constant(
        residual_jacobian_fit + state.projected_design_matrix()
    )
```

Update nltiming integration tests so JUG-owned and exact-linear columns use one
fitter sign. No JUMP/DMX column may receive a parameter-class-specific sign
exception.

### Test 9 — State migration and unit ownership

Construct a state with parameter aliases, RAJ/DECJ, nontrivial weights, and row
sorting. Assert:

- every field in the §6.2 migration table is retained or available at its
  documented destination;
- `state.design_matrix` remains in session TOA order;
- only `ResidualProjection.row_indices` changes output order;
- public RAJ/DECJ columns equal `native_derivative_to_fit_column(...)`;
- `_fit_unit_column_to_native_delta` no longer exists;
- `jug.utils.units`/parameter metadata is the sole owner of the astrometry
  derivative scale constants.

---

## 9. Downstream impact

After this JUG feature:

```text
state.design_matrix
    = raw autodiff timing-model matrix
    = PINT–Vela fit units
    = unprojected

state.projected_design_matrix()
    = residual_projection(state.design_matrix)

residual_jacobian_fit
    = -state.projected_design_matrix()          when mean removal is active
    = -state.design_matrix + A_TZR              when mean_mode == "none"
```

nltiming must:

- stop treating `state.design_matrix` as a residual Jacobian;
- stop mixing JUG residual-Jacobian columns with fitter-sign JUMP/DMX columns;
- use the projected matrix for fitter bases and the exact residual Jacobian when
  linearizing an uncentered exported residual;
- retain any sampling-chart or global-coordinate transformations in nltiming,
  not JUG;
- read row ordering through `ResidualProjection.row_indices` rather than the
  removed top-level `state.isort`;
- update `bug_designmatrix.md` to reflect that the JUG state stores a raw matrix
  and projection is explicit;
- remove the companion document's unconditional `M = -J for every fit
  parameter` claim: it holds as `J = -P M` after ordinary mean removal, while
  an uncentered residual has `J = -M + A_TZR`.

MetaPulsar must:

- continue to regard JUG public columns as PINT–Vela fit-unit columns;
- not infer computational compatibility from `unit_convention`;
- compare tempo2 computations to tempo2 after the documented public-unit
  conversion.

---

## 10. Out of scope

- Building a new full PINT JAX graph.
- Changing the user's selected `tempo2_native` mode or making one mode mandatory
  for autodiff.
- General nltiming sampling-coordinate/chart transformations.
- Preserving aliases for the old residual-Jacobian storage.
- Making analytic and autodiff matrices numerically identical.
- Adding TZR parameters to the set of PINT-compatible fit coordinates.

An ecliptic or alias unit bug is nominally separate, but it is not permitted to
invalidate this feature's public-unit acceptance tests. If a required test
exposes such a bug, fix it in the same integration sequence or mark the affected
parameter unsupported with a precise error.

---

## 11. Acceptance checklist

- [ ] Every public `design_matrix` is a raw timing/fitter matrix, never a
      residual Jacobian.
- [ ] Public matrices use PINT–Vela fit units in both compatibility modes.
- [ ] `compatibility` selects computational physics, not public units.
- [ ] PINT autodiff differentiates the existing PINT-style JAX path.
- [ ] Tempo2 autodiff differentiates exactly the session-selected
      `tempo2_native` graph; design-matrix construction never overrides it.
- [ ] One-shot `compute_designmatrix` accepts/resolves `tempo2_native` using the
      same rules and prepares that selected graph exactly once.
- [ ] Autodiff reuses the selected residual graph/cache and causes no second
      graph build or matrix-specific recompute.
- [ ] Public autodiff never calls the simplified analytic autodiff oracle.
- [ ] Analytic and autodiff have independent correctness tests.
- [ ] `export_jax_timing_state` no longer accepts `design_matrix_method`.
- [ ] State-export setup plumbing does not use
      `GeneralFitSetup.design_matrix_method`.
- [ ] The frozen state's matrix comes from its selected JAX graph.
- [ ] Raw design matrices are not demeaned or mean-weighted.
- [ ] Raw matrices are in session TOA order.
- [ ] Residual projection is explicit and separately testable.
- [ ] `ResidualProjection` contains only frozen mean-removal and row-order data;
      it performs no whitening or TZR subtraction.
- [ ] Current exports produce weighted/unweighted projections only; `none` is
      explicitly reserved for a future uncentered/PHOFF-compatible path.
- [ ] \(A_\mathrm{TZR}=0\) when `subtract_tzr=False`.
- [ ] The fitter-model linearization uses the projected matrix with an explicit
      minus; the exact residual linearization uses `residual_jacobian`.
- [ ] Mean-removed residual Jacobians equal `-projected_design_matrix`; the
      uncentered exception is tested as the row-constant TZR response.
- [ ] Internal/native-to-PINT–Vela conversion is centralized.
- [ ] `state.design_matrix` uses PINT–Vela fit-unit columns, not its former
      native-delta RAJ/DECJ columns.
- [ ] TZR parameters remain model state but are rejected as PINT-compatible
      design-matrix coordinates.
- [ ] Raw F0 matches the unprojected PINT design matrix before mean removal.
- [ ] Implicit Offset is a leading `+1/F0` PINT-compatible column, added only on
      request and never implemented as a TZRMJD perturbation.
- [ ] Offset unit metadata matches PINT's returned `u.s/u.s` exactly.
- [ ] Explicit PHOFF suppresses implicit Offset and matches PINT.
- [ ] Every current `JaxTimingState` field has the documented migration
      destination and downstream consumers are updated.
- [ ] The method/graph/parameter capability manifest ships with the feature.
- [ ] PINT-path numerical tests use PINT or direct selected-graph finite
      differences as appropriate.
- [ ] Tempo2-path numerical tests use tempo2/libstempo after explicit
      PINT–Vela unit conversion.
- [ ] nltiming integration uses a single fitter sign for all parameter classes.
