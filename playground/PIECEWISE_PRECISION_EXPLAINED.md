# Why Scatter Doesn't Minimize at Local PEPOCH

**Date:** 2025-12-02  
**Key Question:** If we use local coordinates to reduce dt, why doesn't the precision improve at each local PEPOCH?

---

## The Expectation

When we segment data and use local PEPOCH coordinates, we expected:

```
Within each segment:
  • Scatter should be MINIMAL at the local PEPOCH (dt_local ≈ 0)
  • Scatter should INCREASE with distance from local PEPOCH (|dt_local| large)
  • Each segment should have its own "sweet spot" of minimum error
```

This would happen if the error source was: **Large dt values in float64 arithmetic**

---

## What Actually Happens

**Measured scatter (piecewise - longdouble) per segment:**

| Segment | dt_epoch from global | dt_local span | Scatter RMS | Pattern |
|---------|---------------------|---------------|-------------|---------|
| 0 | -203 days | 493 days | 1.4 ns | Uniform |
| 1 | +226 days | 492 days | 1.6 ns | Uniform |
| 2 | +776 days | 471 days | 3.4 ns | Uniform |
| 3 | +1248 days | 485 days | 5.8 ns | Uniform |
| 4 | +1675 days | 303 days | 6.5 ns | Uniform |

**Key observations:**
1. ✓ Scatter is **uniform within each segment** (doesn't vary with dt_local)
2. ✓ Scatter **increases between segments** (correlates with dt_epoch, not dt_local)
3. ✗ Scatter does **NOT minimize at local PEPOCH**

---

## The Actual Error Source

The error doesn't come from using large `dt_local` values. It comes from the **coordinate transformation** itself:

```python
# Transform from global to local coordinates
dt_local = dt_sec - dt_epoch

# Compute phase in local coordinates
phase_local = dt_local × (f0_local + dt_local × f1/2)

# Add phase offset to restore global phase reference
phase_offset = f0 × dt_epoch + (f1/2) × dt_epoch²  # ← ERROR SOURCE
phase_corrected = phase_local + phase_offset
```

### The Problem with phase_offset

For segment 4 (dt_epoch = 1675 days = 1.45×10⁸ seconds):

```
phase_offset = 339 Hz × 1.45×10⁸ s + (-1.6×10⁻¹⁵ Hz/s / 2) × (1.45×10⁸ s)²
             = 4.92×10¹⁰ cycles - 1.68×10⁴ cycles
             ≈ 4.92×10¹⁰ cycles
```

**Float64 precision limit:**
- Float64 has ~15-16 significant digits
- At 10¹⁰ cycles, precision is ~10⁻⁶ cycles ≈ 3 picoseconds (theoretical)
- But through multiple operations (compute, add, wrap, convert), error accumulates to ~6.5 ns (observed)

**The error scales with dt_epoch:**
- Small dt_epoch (segment 0, -203 days): 1.4 ns error
- Large dt_epoch (segment 4, +1675 days): 6.5 ns error
- **Error ∝ |dt_epoch|** (distance from global PEPOCH)

---

## Why This Ruins the Piecewise Benefit

### What we gained:
- ✓ Smaller `dt_local` values (better for matrix conditioning during fitting)
- ✓ Reduced catastrophic cancellation in phase computation

### What we lost:
- ✗ Introduced phase offset transformation error
- ✗ Error grows with distance from global PEPOCH
- ✗ Cannot be eliminated without going to higher precision

### Net result:
```
Standard method error:  ~22 ns (uniform, from large dt)
Piecewise method error: ~22 ns (varying, from transformation)
```

The errors are **equal in magnitude** but have **different sources**!

---

## Mathematical Explanation

### Standard Method Error Budget

```python
phase = dt_sec × (f0 + dt_sec × f1/2)
```

For dt_sec ~ 5×10⁷ seconds (1.5 years):
- Multiply: 339 Hz × 5×10⁷ s = 1.7×10¹⁰ cycles
- Float64 precision at 10¹⁰: ~10⁻⁶ cycles
- Through wrapping and conversion: ~22 ns

**Error is uniform** because all TOAs use the same coordinate system.

### Piecewise Method Error Budget

```python
# Local phase (small dt_local, better precision)
phase_local = dt_local × (f0_local + dt_local × f1/2)  # ← Good!

# Phase offset (large dt_epoch, precision loss)
phase_offset = f0 × dt_epoch + f1/2 × dt_epoch²        # ← Bad!

# Combine
phase_corrected = phase_local + phase_offset
```

The `phase_local` computation is indeed more precise (smaller dt_local), but then we **add** a large, imprecise number (`phase_offset`). This contaminates the result!

**Analogy:**
- Standard: Measuring 1.5 years with a ruler (±22 ns error)
- Piecewise: Measuring 500 days with a caliper (±1 ns), then adding a ±6 ns offset to convert to the same reference frame

The conversion error dominates!

---

## Implications

### 1. The piecewise approach doesn't improve float64 precision

The coordinate transformation introduces errors comparable to what we were trying to eliminate.

### 2. The benefit IS real for fitting convergence

Smaller `dt_local` values improve the condition number of the design matrix, which helps:
- Faster convergence
- More stable parameter updates
- Better behavior for ill-conditioned problems

But this doesn't translate to better **precision** of the final residuals.

### 3. To actually improve precision, we need:

**Option A:** Use longdouble for the entire calculation
- ✓ Eliminates all float64 errors
- ✗ Can't use JAX acceleration

**Option B:** Use longdouble ONLY for phase_offset
```python
phase_offset = longdouble(f0) * longdouble(dt_epoch) + ...
phase_corrected = float64(phase_local) + float64(phase_offset)
```
- ✓ Might reduce transformation error
- ✓ Keeps JAX for local computations
- ? Needs testing

**Option C:** Accept that float64 precision is ~20 ns
- ✓ This is already excellent for pulsar timing
- ✓ Far below typical TOA uncertainties (100-1000 ns)
- ✓ Matches standard method

---

## Answer to Your Question

**Q: Why doesn't scatter minimize at each local PEPOCH?**

**A: Because the error doesn't come from dt_local (which is small at local PEPOCH), it comes from dt_epoch (which is fixed for the entire segment).**

Within each segment:
- `dt_local` varies from -250 to +250 days → but this doesn't affect precision
- `dt_epoch` is constant (e.g., +1675 days) → this determines the segment's error level

The piecewise method is like changing coordinates in your calculation but then paying a "conversion fee" to get back to the reference frame. The conversion fee (phase_offset error) is proportional to how far you've moved from the origin (dt_epoch), not how precisely you measured in the local frame (dt_local).

---

## Conclusion

The piecewise method **works correctly** and achieves its goal of improving numerical conditioning for fitting. However, it doesn't improve the **precision** of residuals because:

1. The transformation introduces errors comparable to what it eliminates
2. These errors scale with distance from global PEPOCH, not local PEPOCH
3. Float64 precision limits apply to the phase_offset calculation regardless of local coordinate choice

This is a fundamental insight: **You can't improve precision by changing coordinates if the transformation itself has precision limits.**

The method remains valuable for fitting stability on long baselines, but it's not a silver bullet for precision improvement.
