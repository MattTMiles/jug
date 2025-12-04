# Piecewise PEPOCH Fitting: Proposal for Long-Baseline Precision

## Executive Summary

Current JUG fitting suffers from **float64 precision limitations** on long-baseline datasets (>10 years). The issue manifests as:
- Quadratic error growth with distance from PEPOCH: Δresidual ∝ ΔF1 × (t - PEPOCH)²
- Ill-conditioned design matrices when dt is large (~10⁸ seconds)
- Residual differences between float64 and longdouble methods correlate 99.7% with ΔF1

**Proposed solution:** Piecewise fitting with multiple local PEPOCHs that maintains a single global F0/F1 model.

---

## Problem Statement

### Numerical Precision Breakdown

For a 30-year pulsar timing dataset:
- Maximum dt from PEPOCH: ±15 years ≈ ±4.7×10⁸ seconds
- Design matrix element for F1: M[i,1] = -dt²/(2F0) ≈ -3×10¹⁶
- Float64 condition number: >10²³ (effectively singular)
- **Result:** Float64 precision insufficient for WLS solve

### Current Status (6-year J1909-3744 dataset)

**Measured:**
- ΔF1 (longdouble - JUG) = 7.0×10⁻²³ Hz/s
- Residual difference RMS: 0.01 μs
- Growth rate: 3.5×10⁻⁶ μs/day from PEPOCH
- Minimum scatter: At PEPOCH (dt=0)
- **R² = 0.997**: 99.7% of variance explained by ΔF1 × dt²

**Extrapolated to 30 years:**
- Residual difference would grow to ~0.05 μs = 50 ns
- Approaching the 100 ns acceptability threshold

### Why Current Methods Fail

1. **JUG (float64):** Design matrix ill-conditioned for large dt
2. **Hybrid chunked phase:** Improves phase calculation but still uses float64 for derivatives/WLS
3. **Longdouble everywhere:** Too slow, defeats JAX performance benefits

---

## Proposed Solution: Piecewise PEPOCH Fitting

### Concept

Divide the dataset into temporal segments, each with a local PEPOCH, then tie them together into a **unified global model**.

### Mathematical Framework

#### Segmentation
```
Segment i: [t_start_i, t_end_i]
Local PEPOCH_i = (t_start_i + t_end_i) / 2
```

#### Phase Model (per segment)
```
phase_i(t) = F0_i × dt_i + (F1_i/2) × dt_i²
where dt_i = t - PEPOCH_i
```

#### Global Continuity Constraint

The local parameters must represent a **single underlying spin model**:

```
F0_i = F0_global + F1_global × (PEPOCH_i - PEPOCH_global)
F1_i = F1_global
```

This ensures:
- Same global spin-down rate everywhere
- Smooth phase evolution across segment boundaries
- No spurious discontinuities

#### Fitting Strategy

**Option A: Constrained optimization**
```
Fit: {F0_global, F1_global}
Subject to: F0_i = F0_global + F1_global × (PEPOCH_i - PEPOCH_global) for all i
Method: Compute residuals segment-by-segment, combine into global χ²
```

**Option B: Two-stage fitting**
```
Stage 1: Fit {F0_i, F1_i} independently per segment
Stage 2: Fit {F0_global, F1_global} to the {F0_i, PEPOCH_i} points with weights
Check: Verify F1_i ≈ F1_global across all segments
```

**Recommended: Option A** - mathematically cleaner, no information loss

---

## Implementation Sketch

### Core Algorithm

```python
def fit_piecewise_pepoch(toas, segment_duration=500):
    """
    Fit F0/F1 with piecewise local PEPOCHs.
    
    Parameters
    ----------
    toas : TOA data
    segment_duration : float
        Duration of each segment in days
        
    Returns
    -------
    F0_global, F1_global : float
        Global spin parameters
    """
    # 1. Segment the data
    segments = create_time_segments(toas, segment_duration)
    
    # 2. Define local PEPOCHs
    for seg in segments:
        seg.local_pepoch = (seg.tmin + seg.tmax) / 2
    
    # 3. Fit global parameters
    def chi2(params):
        F0_global, F1_global = params
        total_chi2 = 0
        
        for seg in segments:
            # Local parameters from global constraint
            dt_pepoch = seg.local_pepoch - PEPOCH_global
            F0_local = F0_global + F1_global * dt_pepoch
            F1_local = F1_global
            
            # Compute residuals with local PEPOCH
            dt_local = seg.times - seg.local_pepoch  # SMALL!
            phase = F0_local * dt_local + (F1_local/2) * dt_local**2
            residuals = (phase - round(phase)) / F0_local
            
            # Accumulate chi2
            total_chi2 += sum((residuals / seg.errors)**2)
        
        return total_chi2
    
    # Minimize
    result = scipy.optimize.minimize(chi2, [F0_initial, F1_initial])
    return result.x
```

### Key Properties

**Precision improvement:**
- dt_local always small: ±250 days = ±2.16×10⁷ seconds
- Design matrix elements: ~10¹⁴ (vs 10¹⁶ for full baseline)
- Condition number: ~10¹⁸ (vs 10²³) → float64 sufficient!

**Computational cost:**
- Segments computed in parallel (same as current caching)
- No additional overhead vs current method
- Can still use JAX for phase calculation within segments

---

## Critical Questions for Validation

### 1. Signal Preservation

**Question:** Does piecewise fitting risk absorbing real astrophysical signals?

**Concern:** Gravitational wave (GW) signals manifest as correlated timing residuals across pulsars. If segment boundaries create artificial flexibility, we might fit out the GW signal.

**Analysis needed:**
- Compare piecewise vs global fitting on simulated GW injection
- Verify that F1_global remains consistent across segments
- Check for spurious segment-boundary effects in residuals

**Safety check:** 
```
If F1_i varies significantly across segments beyond measurement uncertainty,
this indicates either:
  a) Real intrinsic spin variations (glitches, timing noise)
  b) Systematic errors (DM variations, unmodeled effects)
  c) Overfitting artifacts

We should REQUIRE: |F1_i - F1_global| < 3σ_F1 for all i
```

### 2. Chromatic vs Achromatic Signals

**Question:** How do we ensure piecewise fitting doesn't absorb achromatic signals (GWs, clock errors) while properly handling chromatic ones (DM variations)?

**Key distinction:**
- **Chromatic (frequency-dependent):** DM, scattering, solar wind → legitimately different per segment
- **Achromatic (frequency-independent):** GWs, clock errors, spin parameters → must be GLOBAL

**Safeguard:** Only fit achromatic parameters (F0, F1) with piecewise method. Keep DM, binary parameters global (or use separate piecewise strategy with physical constraints).

### 3. Degrees of Freedom

**Question:** Does piecewise fitting add degrees of freedom that reduce sensitivity?

**Analysis:**
- Global fitting: 2 parameters (F0, F1)
- Piecewise with N segments: Still 2 parameters! (F0_global, F1_global)
- The local PEPOCHs are not free parameters - they're fixed geometric choices

**Conclusion:** No additional DoF, just a coordinate transformation that improves numerical conditioning.

### 4. Comparison with Established Methods

**PINT/Tempo2 precedents:**
- **JUMP parameters:** Fit phase offsets between observing sessions
- **PhaseJump:** Similar concept but additive offset, not reference epoch shift
- **Difference:** Our method maintains phase continuity, theirs explicitly break it

**Question:** Is our method more or less conservative than JUMPs?

**Analysis:** 
- JUMPs add 1 parameter per segment → more flexible, but justified for instrumental changes
- Our method adds 0 parameters → more conservative, purely numerical technique
- **Advantage:** Our method can't overfit since it has the same model complexity

---

## Validation Strategy

### Phase 1: Synthetic Data Tests

1. **Generate perfect data** with known F0, F1 across 30 years
2. Add realistic noise and frequency changes
3. **Compare:**
   - Global fitting (float64) → expect precision loss
   - Piecewise fitting (float64) → expect perfect recovery
   - Longdouble fitting → ground truth

**Success criteria:** Piecewise recovers F0, F1 to within longdouble precision

### Phase 2: GW Signal Injection

1. Simulate GW signal using established PTAs waveforms
2. Inject into real pulsar TOAs
3. **Compare detection sensitivity:**
   - Global fitting: Signal strength / Noise
   - Piecewise fitting: Signal strength / Noise
   
**Success criteria:** No degradation in GW sensitivity (within 5%)

### Phase 3: Real Data Cross-Validation

1. Apply to IPTA/NANOGrav datasets
2. Compare with published results
3. Check for systematic biases in:
   - Spin parameters
   - Binary parameters (if applicable)
   - DM measurements

**Success criteria:** Agreement within 1σ with published values

### Phase 4: Residual Analysis

1. Compute residuals with piecewise-fitted parameters
2. Look for:
   - Discontinuities at segment boundaries
   - Increased scatter in any segment
   - Correlation between segments (shouldn't exist if fitting is correct)

**Success criteria:** Residuals statistically indistinguishable from global fit (after removing precision differences)

---

## Alternative Approaches to Consider

### A. Hierarchical Bayesian Model

Instead of hard constraints, use Bayesian priors:
```
F0_i ~ Normal(F0_global + F1_global * dt_i, σ_F0)
F1_i ~ Normal(F1_global, σ_F1)
```

**Pros:** Naturally handles model uncertainty, can detect real spin variations
**Cons:** More complex, computationally expensive

### B. Spline-Based Spin Model

Fit F(t) as a smooth spline instead of polynomial:
```
F(t) = F0_ref + ∫[t_ref to t] F1(τ) dτ
where F1(t) is a spline
```

**Pros:** Can capture real timing noise, glitches
**Cons:** Adds many parameters, may absorb GW signal

### C. Hybrid: Local PEPOCHs for Derivatives Only

Keep global F0, F1 but compute derivatives using local reference points:
```
d(phase)/d(F1) at TOA_i = (t_i - PEPOCH_local_i)² / (2*F0)
where PEPOCH_local_i is the nearest segment center
```

**Pros:** Minimal modification to existing code
**Cons:** Not as clean mathematically, may have subtle biases

---

## Recommended Next Steps

1. **Mathematical review (1-2 hours):**
   - Verify that piecewise parameterization is equivalent to global model
   - Prove that no signal loss occurs for achromatic signals
   - Derive uncertainty propagation formulas

2. **Prototype implementation (1 day):**
   - Implement piecewise fitting in JUG
   - Test on synthetic data with known parameters
   - Validate precision improvement

3. **GW signal testing (2-3 days):**
   - Simulate GW signals with various amplitudes/frequencies
   - Compare detection statistics: global vs piecewise
   - Ensure no sensitivity loss

4. **Real data validation (1 week):**
   - Apply to J1909-3744 full dataset
   - Compare with PINT/Tempo2 results
   - Check residual quality and parameter consistency

5. **Publication/documentation:**
   - Document the method in JUG
   - Write technical note explaining numerical benefits
   - Propose to IPTA software working group

---

## Open Questions for Expert Review

### For Numerical Analysts:
1. Is the condition number improvement sufficient to justify this approach?
2. Are there better coordinate transformations than piecewise PEPOCHs?
3. What is the optimal segment duration (500 days? 1 year? Longer?)

### For Pulsar Astronomers:
1. Does this method introduce any physical inconsistencies?
2. How does it compare to existing techniques (JUMPs, PhaseJumps)?
3. Are there any astrophysical signals this might accidentally absorb?

### For Gravitational Wave Researchers:
1. Does piecewise fitting affect GW sensitivity? (critical!)
2. Should we apply this to all PTA analyses going forward?
3. Could this explain any existing tension between different PTA results?

---

## Conclusion

The piecewise PEPOCH fitting method offers:
- ✅ **Numerical improvement:** Eliminates float64 precision loss on long baselines
- ✅ **Same model complexity:** No additional parameters, same DoF
- ✅ **Computational efficiency:** Compatible with JAX, parallelizable
- ❓ **Signal preservation:** Requires validation (critical!)
- ❓ **Established precedent:** Novel approach, needs community review

**Recommendation:** This deserves careful mathematical analysis and GW signal testing before production use. The numerical benefits are clear, but we must ensure no unintended consequences for science goals.

---

## References for Further Reading

1. **Condition number in timing:** Hobbs et al. (2006) - Tempo2 paper
2. **JUMP parameters:** PINT documentation, `PhaseJump` model
3. **Numerical precision in timing:** Vallisneri (2020) - libstempo precision tests
4. **PTA sensitivity:** NANOGrav 15-year data set methodology papers

---

**Author's Note:** This proposal requires input from experts in numerical analysis, pulsar timing, and gravitational wave data analysis. The mathematical soundness looks good, but validation against real science cases (especially GW detection) is essential before adoption.
