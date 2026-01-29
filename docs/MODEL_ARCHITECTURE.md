# JUG Model Architecture: Adding New Parameters

This guide explains how to safely add new parameter support to JUG without breaking existing functionality.

## Architecture Overview

JUG uses a **ParameterSpec + Component Graph** architecture:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ParameterSpec  │────▶│   Components     │────▶│  Derivatives    │
│  (metadata)     │     │  (routing)       │     │  (computation)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   - name/aliases          - filter params          - actual math
   - derivative_group      - route to derivs        - returns columns
   - internal_unit         - combine results        - bit-identical
   - codec for I/O
```

## Step-by-Step: Adding a New Parameter

### Example: Adding PMRA (Proper Motion in RA)

#### Step 1: Add ParameterSpec Entry

Edit `jug/model/parameter_spec.py`:

```python
# In PARAMETER_REGISTRY initialization
'PMRA': ParameterSpec(
    name='PMRA',
    group='astrometry',
    dtype='float',
    internal_unit='rad/yr',    # Radians per year
    par_unit_str='mas/yr',     # Milliarcsec per year for display
    aliases=['PMRAC'],
    component_name='Astrometry',
    derivative_group=DerivativeGroup.ASTROMETRY,
    default_fit=True,
    gui_visible=True,
    requires=None,
    par_codec_name='pmra',     # For unit conversion
),
```

#### Step 2: Add Codec (if needed)

Edit `jug/model/codecs.py`:

```python
class PMRACodec(Codec):
    """Proper motion in RA: mas/yr <-> rad/yr"""
    
    def decode(self, value: str) -> float:
        """Parse mas/yr to rad/yr."""
        mas_per_yr = float(value)
        return mas_per_yr * (np.pi / (180 * 3600 * 1000))  # mas -> rad
    
    def encode(self, value: float) -> str:
        """Format rad/yr as mas/yr."""
        mas_per_yr = value / (np.pi / (180 * 3600 * 1000))
        return f"{mas_per_yr:.10f}"

# Add to registry
CODECS['pmra'] = PMRACodec()
```

#### Step 3: Implement Derivative Function

Create `jug/fitting/derivatives_astrometry.py`:

```python
def compute_astrometry_derivatives(
    params: Dict,
    toas_mjd: np.ndarray,
    fit_params: List[str],
    ssb_obs_pos: np.ndarray,  # From geometry cache
    ssb_obs_vel: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute astrometry parameter derivatives.
    
    Returns
    -------
    dict
        {param_name: derivative_column} for each param in fit_params
    """
    result = {}
    posepoch = params.get('POSEPOCH', params.get('PEPOCH'))
    dt_years = (toas_mjd - posepoch) / 365.25
    
    if 'PMRA' in fit_params:
        # d(delay)/d(PMRA) = -dot(pos_hat_RA, obs_pos) * dt / c
        # ... actual math here ...
        result['PMRA'] = d_delay_d_pmra
    
    return result
```

#### Step 4: Create or Update Component

Create `jug/model/components/astrometry.py`:

```python
from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives
from jug.model.parameter_spec import is_astrometry_param

class AstrometryComponent:
    """Component for astrometry parameters."""
    
    _PROVIDED_PARAMS = ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']
    
    def provides_params(self) -> List[str]:
        return self._PROVIDED_PARAMS.copy()
    
    def filter_fit_params(self, fit_params: List[str]) -> List[str]:
        return [p for p in fit_params if is_astrometry_param(p)]
    
    def compute_derivatives(
        self,
        params: Dict,
        toas_mjd: np.ndarray,
        fit_params: List[str],
        ssb_obs_pos: np.ndarray = None,
        ssb_obs_vel: np.ndarray = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        astro_params = self.filter_fit_params(fit_params)
        if not astro_params:
            return {}
        
        return compute_astrometry_derivatives(
            params=params,
            toas_mjd=toas_mjd,
            fit_params=astro_params,
            ssb_obs_pos=ssb_obs_pos,
            ssb_obs_vel=ssb_obs_vel,
        )
```

#### Step 5: Register Component

Edit `jug/model/components/__init__.py`:

```python
from .astrometry import AstrometryComponent

COMPONENT_REGISTRY = {
    'SpinComponent': SpinComponent(),
    'DispersionComponent': DispersionComponent(),
    'AstrometryComponent': AstrometryComponent(),  # NEW
}
```

#### Step 6: Add Tests

Create `jug/tests/test_astrometry_derivatives.py`:

```python
def test_pmra_derivative_shape():
    """Test PMRA derivative has correct shape."""
    # ... setup ...
    result = compute_astrometry_derivatives(
        params=params,
        toas_mjd=toas_mjd,
        fit_params=['PMRA'],
        ssb_obs_pos=ssb_obs_pos,
        ssb_obs_vel=ssb_obs_vel,
    )
    assert 'PMRA' in result
    assert result['PMRA'].shape == (n_toas,)
    assert result['PMRA'].dtype == np.float64

def test_pmra_component_equivalence():
    """Test component wrapper matches direct call."""
    direct = compute_astrometry_derivatives(...)
    component = AstrometryComponent()
    wrapped = component.compute_derivatives(...)
    
    assert np.array_equal(direct['PMRA'], wrapped['PMRA'])
```

## Key Principles

### 1. Bit-for-Bit Equivalence
Any routing refactor must produce identical numerical outputs:
```python
# Test pattern
assert np.array_equal(old_result, new_result)  # NO tolerances
```

### 2. Spec is Metadata Only
`ParameterSpec` contains no computation. It's just metadata for:
- Routing (which component handles this param)
- I/O (how to parse/format values)
- UI (display name, visibility)

### 3. Components are Thin Wrappers
Components call existing derivative functions exactly as before:
```python
# Good: delegate to existing function
return _compute_spin_derivatives(params, toas_mjd, fit_params)

# Bad: re-implement math in component
dt = (toas_mjd - pepoch) * SECONDS_PER_DAY
deriv = 2 * np.pi * dt  # Don't do this!
```

### 4. Engine is Source of Truth
GUI never implements computation. All paths go through engine:
```
GUI -> session.fit_parameters() -> optimized_fitter -> components -> derivatives
CLI -> session.fit_parameters() -> optimized_fitter -> components -> derivatives
API -> session.fit_parameters() -> optimized_fitter -> components -> derivatives
```

## Checklist for New Parameters

- [ ] Add `ParameterSpec` entry with all fields
- [ ] Add codec if non-standard units
- [ ] Implement derivative function (or add to existing module)
- [ ] Create/update component wrapper
- [ ] Register component in `COMPONENT_REGISTRY`
- [ ] Add unit tests for derivative shape/dtype
- [ ] Add bit-for-bit equivalence tests
- [ ] Update golden regression tests if needed
- [ ] Verify GUI/CLI/API all produce identical results

## File Organization

```
jug/
├── model/
│   ├── parameter_spec.py     # ParameterSpec registry
│   ├── codecs.py             # I/O codecs
│   └── components/
│       ├── base.py           # TimingComponent protocol
│       ├── spin.py           # SpinComponent
│       ├── dispersion.py     # DispersionComponent
│       └── astrometry.py     # AstrometryComponent (future)
├── fitting/
│   ├── derivatives_spin.py   # Spin derivative computation
│   ├── derivatives_dm.py     # DM derivative computation
│   └── derivatives_astrometry.py  # Astrometry derivatives (future)
└── tests/
    ├── test_parameter_spec.py
    ├── test_component_graph.py
    └── test_fitter_routing.py
```
