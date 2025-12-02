# JUG Copilot Instructions

JUG is a JAX-based pulsar timing package that implements a complete timing pipeline **without** PINT or Tempo2 dependencies. All timing model components are implemented from scratch using JAX for JIT compilation.

## Architecture Overview

**Pipeline Flow**: `.par/.tim` files → Clock corrections → Barycentric corrections → Binary delays → Emission times → DM correction → Phase residuals

Key modules:
- `jug/io/` - Parsers for `.par` (timing model) and `.tim` (TOA) files
- `jug/delays/` - JAX-JIT delay calculations (`combined.py` is the 100x speedup kernel)
- `jug/residuals/simple_calculator.py` - Main entry point for residual computation
- `jug/fitting/` - WLS fitting with analytical PINT-compatible derivatives

## Critical Requirements

**Float64 Precision**: Always enable JAX float64 mode at module import:
```python
import jax
jax.config.update("jax_enable_x64", True)
```
Import `jug` before other jug submodules to ensure this is set.

**High-Precision Parameters**: F0, F1, PEPOCH, TZRMJD require `np.longdouble`. Use `get_longdouble()` from `jug/io/par_reader.py`:
```python
from jug.io.par_reader import parse_par_file, get_longdouble
params = parse_par_file("pulsar.par")
f0 = get_longdouble(params, 'F0')
```

## Binary Models

- **ELL1/ELL1H**: Low-eccentricity MSPs (implemented in `jug/delays/combined.py`)
- **BT/DD**: Keplerian binaries (implemented in `jug/delays/binary_bt.py`, `binary_dd.py`)
- **T2**: Universal model (implemented in `jug/delays/binary_t2.py`)

Handle both parameterizations for Shapiro delay:
- H3/STIG (orthometric) 
- M2/SINI (mass/inclination) - convert using `r = T_SUN_SEC * M2`, `s = SINI`

## Fitting Implementation

**Sign Convention** (PINT-compatible): Derivatives are POSITIVE mathematically, then negated in `compute_spin_derivatives()`:
```python
# In derivatives_spin.py
def d_phase_d_F(...):
    return derivative  # POSITIVE

def compute_spin_derivatives(...):
    derivatives[param] = -deriv / f0  # Negate + convert cycles→seconds
```

**WLS Fitting**: Use `wls_solve_svd()` with `negate_dpars=False`:
```python
from jug.fitting.wls_fitter import wls_solve_svd
delta_params, cov, _ = wls_solve_svd(residuals, errors, design_matrix, negate_dpars=False)
```

## Data Files

Required data in `data/` directory:
- `clock/*.clk` - Tempo2-format clock files (chain: observatory→UTC→TAI→TT)
- `ephemeris/de440s.bsp` - JPL ephemeris kernel
- `observatory/observatories.dat` - Observatory ITRF coordinates
- `pulsars/` - Example .par/.tim files for testing

## Testing

```bash
pytest jug/tests -v                    # Run all tests
pytest jug/tests/test_binary_models.py # Test specific module
```

Validation against Tempo2: Compare with `-output general2` output. RMS difference should be << 1 μs.

## Common Patterns

**Computing residuals**:
```python
from jug.residuals.simple_calculator import compute_residuals_simple
result = compute_residuals_simple("pulsar.par", "pulsar.tim", clock_dir="data/clock")
print(f"RMS: {result['rms_us']:.3f} μs")
```

**Adding new delays**: Implement with `@jax.jit`, add to pipeline in `simple_calculator.py`, register new parameters in `SpinDMModel` pytree if needed.

## Current Status

- **Milestone 2 Complete**: Core timing + gradient-based fitting validated against PINT/Tempo2
- **Next**: Milestone 3 (EFAC/EQUAD noise models), Milestone 5 (PyQt6 GUI)

See `JUG_PROGRESS_TRACKER.md` for detailed status and `CLAUDE.md` for implementation history.
