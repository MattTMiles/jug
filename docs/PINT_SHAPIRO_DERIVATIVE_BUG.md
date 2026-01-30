# PINT Bug: Missing cos(Φ) in Shapiro Delay Derivative

**Date discovered:** 2026-01-29  
**Affects:** PINT's ELL1 binary model `d_delayS_d_Phi` computation  
**Impact:** PB, TASC, PBDOT derivatives incorrect near superior conjunction

## Summary

PINT's ELL1 binary model has a bug in the derivative of Shapiro delay with respect to orbital phase. The `cos(Φ)` factor is missing from the chain rule derivative.

## Mathematical Derivation

The ELL1 Shapiro delay is:

$$\Delta t_S = -2 \cdot T_{M2} \cdot \ln(1 - \sin i \cdot \sin \Phi)$$

where $T_{M2} = T_\odot \cdot M_2$ is the companion mass in time units.

Taking the derivative with respect to orbital phase $\Phi$:

$$\frac{\partial \Delta t_S}{\partial \Phi} = -2 \cdot T_{M2} \cdot \frac{1}{1 - \sin i \cdot \sin \Phi} \cdot \frac{\partial}{\partial \Phi}(1 - \sin i \cdot \sin \Phi)$$

$$= -2 \cdot T_{M2} \cdot \frac{1}{1 - \sin i \cdot \sin \Phi} \cdot (-\sin i \cdot \cos \Phi)$$

$$= \frac{2 \cdot T_{M2} \cdot \sin i \cdot \cos \Phi}{1 - \sin i \cdot \sin \Phi}$$

## The Bug

**PINT's implementation** (in `pint/models/stand_alone_psr_binaries/ELL1_model.py`):

```python
d_delayS_d_Phi = -2 * TM2 * 1.0 / (1 - self.SINI * np.sin(Phi)) * (-self.SINI)
              = 2 * TM2 * SINI / (1 - SINI * sin(Phi))  # Missing cos(Phi)!
```

**Correct formula** (used by JUG):

```python
d_delayS_d_Phi = 2 * TM2 * SINI * np.cos(Phi) / (1 - SINI * np.sin(Phi))
```

## Numerical Verification

The full derivative d(Shapiro)/d(PB) involves the chain rule:
```
d(Shapiro)/d(PB) = d(Shapiro)/d(Φ) × d(Φ)/d(PB)
```

Using finite differences with δPB = 10⁻¹⁰ days to perturb PB and measure the change in Shapiro delay:

| Phase | Numerical | JUG | PINT | JUG/Num | PINT/Num |
|-------|-----------|-----|------|---------|----------|
| 0° | -2.53e-2 | -2.53e-2 | -2.53e-2 | 1.00 | 1.00 |
| 45° | -6.07e-2 | -6.07e-2 | -8.58e-2 | 1.00 | **1.41** |
| 80° | -2.55e-1 | -2.55e-1 | -1.47e+0 | 1.00 | **5.76** |
| 89° | -2.01e-1 | -2.01e-1 | -1.15e+1 | 1.00 | **57.2** |
| 90° | -1.91e-4 | -1.83e-4 | -1.24e+1 | ~1.0 | **64797** |
| 91° | +2.01e-1 | +2.01e-1 | -1.15e+1 | 1.00 | **-57.3** |
| 135° | +6.07e-2 | +6.07e-2 | -8.58e-2 | 1.00 | **-1.41** |
| 180° | +2.53e-2 | +2.53e-2 | -2.53e-2 | 1.00 | **-1.00** |

**Key observations:**
1. JUG matches numerical at ALL orbital phases
2. PINT/Numerical ratio equals exactly 1/cos(Φ)
3. At Φ=45°: PINT error = 1/cos(45°) = √2 ≈ 1.414
4. At Φ=90°: cos(Φ)→0, so PINT error → ∞

## Impact on Parameter Derivatives

This bug affects all parameters whose derivatives include the chain rule through orbital phase:

- **PB** (orbital period)
- **TASC** (time of ascending node)
- **PBDOT** (orbital period derivative)

The error is proportional to (1/cos(Φ)) and becomes significant:
- Near superior conjunction (Φ ≈ π/2): Error → ∞
- At inferior conjunction (Φ ≈ 3π/2): Error → ∞
- At quadrature (Φ ≈ 0, π): Error is zero

## Quantitative Impact on Fitting

For pulsar J1909-3744 (sin i ≈ 0.998, M₂ ≈ 0.21 M☉):

| Orbital Phase | |PINT/JUG - 1| |
|---------------|------------------|
| 0° (inf. conj.) | < 10⁻⁷ |
| 45° | ~41% |
| 80° | ~475% |
| 89° | ~5700% |
| 90° (sup. conj.) | → ∞ |

## Visual Comparison

![Shapiro derivative comparison](figures/shapiro_derivative_bug.png)

The figure shows:
1. **Top left:** Shapiro delay vs orbital phase (peaks at superior conjunction)
2. **Top right:** Derivative comparison - JUG matches numerical, PINT diverges
3. **Bottom left:** Zoomed view near Φ = 90° showing PINT's incorrect spike
4. **Bottom right:** Ratio PINT/JUG = 1/cos(Φ) confirms the missing factor

## Recommendation

The fix in PINT would be to change line in `ELL1_model.py`:

```python
# Current (buggy):
d_delayS_d_Phi = -2 * TM2 * 1.0 / (1 - self.SINI * np.sin(Phi)) * (-self.SINI)

# Fixed:
d_delayS_d_Phi = -2 * TM2 * 1.0 / (1 - self.SINI * np.sin(Phi)) * (-self.SINI * np.cos(Phi))
```

## Tempo2 Comparison

Tempo2's ELL1 model (`ELL1model.C`) takes a different approach - it **completely omits** the Shapiro term from the PB derivative:

```c
if (param==param_pb)
    return -Csigma*an*SECDAY*tt0/(pb*SECDAY); /* Pb */
```

Where `Csigma = x*cos(phase)` is purely the Roemer delay contribution. The Shapiro delay contribution via d(Shapiro)/d(Phi) × d(Phi)/d(PB) is not included.

This omission may be intentional (treating Shapiro parameters as independent) or may be a simplification for the typical case where Shapiro effects are small. However, for high-inclination systems like J1909-3744 (sin i ≈ 0.998), the Shapiro contribution can be significant near superior conjunction.

**Summary of implementations:**
| Code | d(Shapiro)/d(Phi) in PB derivative |
|------|-----------------------------------|
| JUG | Correct: 2·TM2·sin(i)·cos(Φ)/(1-sin(i)·sin(Φ)) |
| PINT | Buggy: 2·TM2·sin(i)/(1-sin(i)·sin(Φ)) (missing cos) |
| Tempo2 | Omitted entirely |

## References

- Lange et al. (2001), MNRAS 326, 274 - ELL1 binary model
- Edwards, Hobbs & Manchester (2006), MNRAS 372, 1549 - TEMPO2 timing model

## JUG Implementation

JUG's correct implementation is in `jug/fitting/derivatives_binary.py`:

```python
@jax.jit
def d_shapiro_d_Phi(phi, sini, m2):
    """Derivative of Shapiro delay w.r.t. Phi.
    
    d(Δt_S)/d(Phi) = 2 × TM2 × SINI × cos(Φ) / (1 - SINI × sin(Φ))
    """
    TM2 = T_SUN * m2
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    
    denominator = 1 - sini * sin_phi
    denominator = jnp.maximum(denominator, 1e-10)
    
    return 2 * TM2 * sini * cos_phi / denominator
```
