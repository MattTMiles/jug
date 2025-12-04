# Alternative Precision Enhancement Strategies

**Date:** 2025-12-02  
**Context:** Piecewise PEPOCH showed that coordinate transformations introduce ~6 ns errors. Can we do better?

---

## The Core Problem

The piecewise method fails because:
```python
phase_offset = float64(longdouble(f0) × longdouble(dt_epoch) + ...)
                       ↑ Precise calculation ↑
              ↓ Convert back to float64 ↓
              ↓ Loses ~6 ns precision ↓
```

**Key insight:** We compute precisely but then **throw away** precision when converting back to float64.

**Question:** What if we keep MORE information in longdouble throughout the pipeline?

---

## Strategy 1: Spline Representation of Phase Evolution

### Concept

Instead of using a quadratic phase model globally:
```python
phase(t) = F0 × dt + (F1/2) × dt²
```

Use a **cubic spline** with knots at segment boundaries that maintains physical continuity.

### Mathematical Framework

**At each knot t_k, enforce:**
1. **Position continuity:** phase(t_k⁻) = phase(t_k⁺)
2. **Velocity continuity:** dphase/dt(t_k⁻) = dphase/dt(t_k⁺) = F0_k
3. **Acceleration continuity:** d²phase/dt²(t_k⁻) = d²phase/dt²(t_k⁺) = F1_k

**Within each segment [t_k, t_{k+1}]:**
```python
phase(t) = a_k + b_k×τ + c_k×τ² + d_k×τ³
where τ = t - t_k  (local time in segment)
```

**Constraints:**
- `b_k = F0_k` (frequency at knot)
- `2×c_k = F1_k` (frequency derivative at knot)
- `a_k, d_k` determined by continuity

### Implementation

```python
class SplinePhaseModel:
    def __init__(self, knots_mjd, pepoch_mjd, f0_global, f1_global):
        """
        knots_mjd: Array of knot positions in MJD
        Each knot has a local F0, F1 (fitted or interpolated)
        """
        self.knots = knots_mjd
        self.pepoch = pepoch_mjd
        
        # Store phase and derivatives at each knot in LONGDOUBLE
        self.phase_at_knots = np.zeros(len(knots), dtype=np.longdouble)
        self.f0_at_knots = np.zeros(len(knots), dtype=np.longdouble)
        self.f1_at_knots = np.zeros(len(knots), dtype=np.longdouble)
        
        # Initialize from global model
        self._initialize_from_global(f0_global, f1_global)
    
    def _initialize_from_global(self, f0, f1):
        """Compute phase at each knot using LONGDOUBLE"""
        for i, t_knot in enumerate(self.knots):
            dt = np.longdouble((t_knot - self.pepoch) * SECS_PER_DAY)
            f0_ld = np.longdouble(f0)
            f1_ld = np.longdouble(f1)
            
            # Store phase at knot in longdouble
            self.phase_at_knots[i] = f0_ld * dt + (f1_ld / 2.0) * dt**2
            self.f0_at_knots[i] = f0_ld + f1_ld * dt
            self.f1_at_knots[i] = f1_ld
    
    def compute_phase(self, t_mjd):
        """
        Compute phase for TOAs using spline interpolation.
        Key: Only convert to float64 at the VERY END.
        """
        # Find which segment each TOA belongs to
        segment_idx = np.searchsorted(self.knots[1:], t_mjd)
        
        phases = np.zeros(len(t_mjd))
        
        for i in range(len(self.knots) - 1):
            mask = (segment_idx == i)
            if not np.any(mask):
                continue
            
            t_segment = t_mjd[mask]
            
            # Local time in LONGDOUBLE
            tau_ld = np.longdouble((t_segment - self.knots[i]) * SECS_PER_DAY)
            
            # Get knot values (already in longdouble)
            phase_k = self.phase_at_knots[i]
            f0_k = self.f0_at_knots[i]
            f1_k = self.f1_at_knots[i]
            
            # Cubic spline coefficients
            # For now, use simple quadratic (can extend to cubic)
            phase_local_ld = phase_k + f0_k * tau_ld + (f1_k / 2.0) * tau_ld**2
            
            # Convert to float64 ONLY at the end
            phases[mask] = np.array(phase_local_ld, dtype=np.float64)
        
        return phases
```

### Advantages

1. **Maintains longdouble precision at knots** - The "anchor points" are stored precisely
2. **Short segments** - Interpolation only over ~500 days, not 6 years
3. **Physical continuity** - Enforces that frequency and its derivative are continuous
4. **Flexible** - Can add more knots where precision is critical

### Challenges

1. **More parameters** - Need to fit F0, F1 at each knot
2. **Constraint equations** - Must enforce continuity during fitting
3. **Complexity** - More sophisticated than simple piecewise

---

## Strategy 2: Residual-Based Longdouble Correction

### Concept

Use the **difference** between float64 and longdouble as a correction term.

```python
# Step 1: Compute in float64 (JAX-compatible)
phase_f64 = compute_phase_float64(dt, f0, f1)

# Step 2: For critical points, compute longdouble correction
correction_ld = compute_phase_longdouble(dt, f0, f1) - longdouble(phase_f64)

# Step 3: Apply correction (still in longdouble)
phase_corrected = longdouble(phase_f64) + correction_ld

# Step 4: Convert to float64 at the end
phase_final = float64(phase_corrected)
```

### Key Insight

The **correction term** is small (~20 ns / 3 ms period ≈ 10⁻⁸ cycles), so converting it to float64 loses less precision than converting the full phase.

### Implementation

```python
def compute_phase_hybrid(dt_sec, f0, f1, use_correction=True):
    """
    Hybrid precision phase computation.
    
    Returns:
        phase: float64 array with longdouble correction applied
        correction_magnitude: How much correction was needed (for diagnostics)
    """
    # Float64 computation (fast, JAX-compatible)
    phase_f64 = dt_sec * (f0 + dt_sec * (f1 / 2.0))
    
    if not use_correction:
        return phase_f64, np.zeros_like(phase_f64)
    
    # Longdouble computation (slow, precise)
    dt_ld = np.array(dt_sec, dtype=np.longdouble)
    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)
    phase_ld = dt_ld * (f0_ld + dt_ld * (f1_ld / 2.0))
    
    # Compute correction in longdouble
    correction_ld = phase_ld - np.longdouble(phase_f64)
    
    # Apply correction (this is the key step!)
    phase_corrected_ld = np.longdouble(phase_f64) + correction_ld
    
    # Convert back to float64
    phase_corrected_f64 = np.array(phase_corrected_ld, dtype=np.float64)
    
    return phase_corrected_f64, np.array(correction_ld, dtype=np.float64)
```

### Advantages

1. **Minimal code changes** - Drop-in replacement for phase computation
2. **JAX compatibility** - Can still use JAX for most operations
3. **Quantifiable** - Can measure how much correction is needed

### Challenges

1. **Still loses precision** - Converting correction to float64 may not help much
2. **Computational cost** - Need to do longdouble computation anyway
3. **Unclear benefit** - If correction is ~10⁻⁸ cycles, float64 can handle it

---

## Strategy 3: Store Phase Residuals in Longdouble

### Concept

The **phase residual** (after subtracting the model) is much smaller than the total phase:
- Total phase: ~10¹⁰ cycles (loses precision in float64)
- Phase residual: ~10⁻⁴ cycles (fits easily in float64)

**What if we keep the "large part" in longdouble and only convert the "small part"?**

### Implementation

```python
class LongdoublePhaseTracker:
    """
    Maintains cumulative phase in longdouble, exposes residuals in float64.
    """
    def __init__(self, f0, f1, pepoch_mjd):
        self.f0_ld = np.longdouble(f0)
        self.f1_ld = np.longdouble(f1)
        self.pepoch_ld = np.longdouble(pepoch_mjd)
        
        # Store model phase at each TOA in longdouble
        self.model_phase_ld = None
    
    def set_toas(self, tdb_mjd, dt_sec):
        """Pre-compute model phase in longdouble"""
        n_toas = len(dt_sec)
        self.model_phase_ld = np.zeros(n_toas, dtype=np.longdouble)
        
        dt_ld = np.array(dt_sec, dtype=np.longdouble)
        
        for i in range(n_toas):
            self.model_phase_ld[i] = (self.f0_ld * dt_ld[i] + 
                                      (self.f1_ld / 2.0) * dt_ld[i]**2)
    
    def compute_residuals(self, observed_phase_f64):
        """
        Compute residuals: observed - model
        
        Key: Model is in longdouble, so subtraction is precise!
        """
        # Convert observed to longdouble for subtraction
        observed_ld = np.array(observed_phase_f64, dtype=np.longdouble)
        
        # Subtract in longdouble
        residual_ld = observed_ld - self.model_phase_ld
        
        # Wrap to [-0.5, 0.5] in longdouble
        residual_wrapped_ld = residual_ld - np.round(residual_ld)
        
        # NOW convert to float64 (residual is small, so precision is preserved!)
        residual_f64 = np.array(residual_wrapped_ld, dtype=np.float64)
        
        return residual_f64
    
    def update_parameters(self, delta_f0, delta_f1):
        """Update parameters in longdouble"""
        self.f0_ld += np.longdouble(delta_f0)
        self.f1_ld += np.longdouble(delta_f1)
        
        # Recompute model phases
        # (in practice, would update incrementally)
```

### Advantages

1. **Residuals are small** - Converting 10⁻⁴ cycles to float64 preserves ~10⁻¹² relative precision
2. **Model precise** - Large phase values stay in longdouble
3. **Fitting in float64** - Can still use JAX for linear algebra (operates on residuals)

### Challenges

1. **Where do observed phases come from?** - Still need to compute them somehow
2. **Parameter updates** - Need to carefully manage precision during fitting
3. **Complexity** - Two parallel representations (longdouble model, float64 residuals)

---

## Strategy 4: Multi-Precision Arithmetic Library

### Concept

Use a library that supports arbitrary precision arithmetic, not just longdouble.

**Options:**
- `mpmath` - Python arbitrary precision math
- `gmpy2` - Fast multiple precision library
- `boost::multiprecision` - C++ library (via pybind11)

### Example with mpmath

```python
from mpmath import mp

# Set precision to 50 decimal digits
mp.dps = 50

def compute_phase_mpmath(dt_sec, f0, f1):
    """Compute phase with arbitrary precision"""
    phases = []
    for dt in dt_sec:
        dt_mp = mp.mpf(dt)
        f0_mp = mp.mpf(f0)
        f1_mp = mp.mpf(f1)
        
        phase_mp = f0_mp * dt_mp + (f1_mp / 2) * dt_mp**2
        phases.append(float(phase_mp))  # Convert to float64 at the end
    
    return np.array(phases)
```

### Advantages

1. **Unlimited precision** - Can go beyond longdouble's 80 bits
2. **Flexible** - Set precision as needed for specific calculations
3. **Well-tested** - Established libraries with good numerics

### Challenges

1. **Very slow** - 10-100× slower than longdouble
2. **Not JAX-compatible** - Pure Python
3. **Overkill?** - Do we really need more than longdouble?

---

## Strategy 5: Hybrid Piecewise with Longdouble Phase Accumulator

### Concept

Combine piecewise segments with a longdouble accumulator that tracks phase precisely across segment boundaries.

### Implementation Sketch

```python
class HybridPiecewiseFitter:
    def __init__(self, segments):
        self.segments = segments
        
        # Longdouble accumulator for phase at each segment boundary
        self.boundary_phases_ld = np.zeros(len(segments) + 1, dtype=np.longdouble)
    
    def initialize_boundaries(self, f0, f1, pepoch):
        """Compute phase at each boundary in longdouble"""
        f0_ld = np.longdouble(f0)
        f1_ld = np.longdouble(f1)
        
        for i, seg in enumerate(self.segments):
            dt_boundary = np.longdouble((seg['tmin_mjd'] - pepoch) * SECS_PER_DAY)
            self.boundary_phases_ld[i] = (f0_ld * dt_boundary + 
                                         (f1_ld / 2.0) * dt_boundary**2)
    
    def compute_residuals_segment(self, seg_idx, dt_local_f64):
        """
        Compute residuals within segment.
        Key: Reference to boundary phase (longdouble), local computation (float64)
        """
        # Get phase at segment start (longdouble, precise!)
        phase_boundary_ld = self.boundary_phases_ld[seg_idx]
        
        # Compute local phase (float64, small values)
        seg = self.segments[seg_idx]
        f0_local = seg['f0']  # float64
        f1_local = seg['f1']  # float64
        
        phase_local_f64 = dt_local_f64 * (f0_local + dt_local_f64 * (f1_local / 2.0))
        
        # Add boundary phase (promote to longdouble)
        phase_total_ld = phase_boundary_ld + np.longdouble(phase_local_f64)
        
        # Wrap in longdouble
        phase_wrapped_ld = phase_total_ld - np.round(phase_total_ld)
        
        # Convert final residual to float64
        residual_f64 = np.array(phase_wrapped_ld, dtype=np.float64)
        
        return residual_f64
```

### Advantages

1. **Anchored precisely** - Segment boundaries stored in longdouble
2. **Fast locally** - Within-segment computation in float64
3. **Continuous** - Enforces phase continuity at boundaries

### Key Difference from Previous Piecewise

Instead of computing `phase_offset` in longdouble and adding it (which loses precision on conversion), we:
1. **Store** the boundary phase directly in longdouble (no conversion)
2. **Add** the small local phase increment in float64
3. **Wrap** in longdouble (wrapping operation is precise)
4. **Convert** only the final wrapped phase to float64

This might preserve more precision because we're not converting large phase_offset values.

---

## Recommendation: What to Try Next

### Priority 1: Hybrid Piecewise with Longdouble Boundaries (Strategy 5)

**Why:** Most likely to work with reasonable complexity
- Uses piecewise segments (good for conditioning)
- Stores critical values (boundaries) in longdouble
- Converts to float64 only after wrapping (when values are small)

**Test it:** Implement and compare with current piecewise

### Priority 2: Residual-Based Correction (Strategy 2)

**Why:** Simplest to implement as a test
- Minimal code changes
- Easy to measure if it helps
- Can be applied to any method

**Test it:** Add as an option to current fitter

### Priority 3: Spline Representation (Strategy 1)

**Why:** Most physically motivated
- Natural extension of piecewise idea
- Enforces smoothness
- Potentially publishable if it works

**Test it:** After Priority 1 & 2 if they show promise

---

## Next Steps

1. **Implement Strategy 5** (Hybrid with longdouble boundaries)
2. **Compare precision** with current piecewise and standard methods
3. **Measure scatter** within segments - does it still increase with dt_epoch?
4. **If successful:** Extend to full fitting loop
5. **If unsuccessful:** Document why and move on

The key question: **Can we defer the float64 conversion until AFTER the wrapping operation?**

This might be the missing piece that lets us actually benefit from higher precision.
