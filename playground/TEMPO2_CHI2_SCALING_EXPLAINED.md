# Why Tempo2 Scales Uncertainties by √(χ²ᵣ) - And Why You're Right to Be Skeptical

**Your Question:** If Tempo2 scales uncertainties by √(χ²ᵣ), wouldn't that make χ²ᵣ = 1 always, making it useless for model selection?

**Short Answer:** You're absolutely right about the math! But Tempo2 only scales the **reported uncertainties**, NOT the TOA errors used in the fit. So χ²ᵣ stays intact for model comparison.

---

## The Apparent Paradox

If you scale TOA errors by √(χ²ᵣ):
```
χ²_new = Σ[(data - model)² / (σ × √χ²ᵣ)²]
       = χ²_old / χ²ᵣ
       = (χ²ᵣ × DOF) / χ²ᵣ
       = DOF

Therefore: χ²ᵣ_new = DOF / DOF = 1
```

This would make χ²ᵣ always equal 1 after rescaling → **useless for model selection!**

---

## What Tempo2 Actually Does

### The Workflow (Critical Distinction):

1. **Fit with original TOA errors**
   - Use σ_TOA from `.tim` file (unchanged)
   - Compute χ² = Σ[(data - model)²/σ²_TOA]
   - Get χ²ᵣ = 32.2

2. **Compute formal parameter uncertainties**
   - From covariance: Cov = (M^T W M)^(-1)
   - σ_formal(F0) = √(Cov[0,0]) = 1.02 × 10⁻¹⁴ Hz

3. **Inflate ONLY the parameter uncertainties**
   - σ_reported = σ_formal × √(χ²ᵣ)
   - σ_Tempo2(F0) = 1.02 × 10⁻¹⁴ × 5.68 = 5.77 × 10⁻¹⁴ Hz
   - This goes in the output `.par` file

4. **χ² remains unchanged**
   - Still computed with original σ_TOA
   - χ²ᵣ = 32.2 preserved for model comparison!

### Key Insight:
The scaling is applied to **parameter uncertainties** (what you report in papers), NOT to **TOA errors** (what you use for fitting). So χ²ᵣ stays useful.

---

## Proof: Tempo2 Output

From `J1909-3744_tdb_refit_F0_F1.par`:
```
F0   339.31569191904082325  1  5.77e-14
CHI2R  32.2232  10405
```

Notice:
- F0 uncertainty is inflated (5.77 × 10⁻¹⁴ Hz)
- But **CHI2R is still 32.2, not 1.0**!
- This proves the TOA errors weren't rescaled

---

## Model Selection Still Works!

### Example: Should we add F1?

**Model 1 (F0 only):**
- χ² = 1,086,695,296,445
- DOF = 10,407
- χ²ᵣ = 104,419,650
- RMS = 726 μs

**Model 2 (F0 + F1):**
- χ² = 335,656
- DOF = 10,406
- χ²ᵣ = 32.26
- RMS = 0.40 μs

**Comparison:**
- Δχ² = 1,086,694,960,790
- Δ_DOF = 1
- p-value < 10⁻¹⁰⁰ (insanely significant!)

**Conclusion:** Adding F1 reduces χ² by a factor of 3 million! Clearly justified.

### The Point:
- Both models have χ²ᵣ >> 1 (absolute fit quality is poor)
- But the **change** in χ² tells you F1 is essential
- The uncertainty scaling is irrelevant for this comparison
- χ²ᵣ is still perfectly useful for model selection

---

## Why Does Tempo2 Do This?

### Philosophy: Conservative Reporting

When χ²ᵣ >> 1, it means:
1. TOA errors are underestimated
2. Model is incomplete
3. Systematic effects present

Formal uncertainties from covariance assume the model is correct. But if χ²ᵣ = 32, something's wrong! 

Tempo2's solution: Inflate parameter uncertainties to account for poor fit quality.

### Precedent:
This is standard practice in metrology (ISO GUM):
- "Type B uncertainty evaluation"
- When χ²ᵣ >> 1, multiply uncertainties by √(χ²ᵣ)
- Protects against overconfident claims

### The Downside:
Creates confusion about what the errors mean!

---

## The Modern View (PINT/JUG)

### Why NOT to inflate uncertainties:

1. **χ²ᵣ >> 1 is almost always true** for pulsar timing
   - TOA errors rarely include all systematics
   - Red noise is ubiquitous

2. **Proper solution: Model the noise**
   - Add EFAC/EQUAD (white noise scaling)
   - Add red noise (RNAMP, RNIDX)
   - Use Bayesian methods (enterprise, tempo2 -nobs)
   - With proper noise modeling, χ²ᵣ ≈ 1

3. **The √(χ²ᵣ) hack has problems:**
   - Doesn't account for parameter correlations
   - Treats all parameters equally
   - Can over- or under-correct
   - Arbitrary and ad-hoc

4. **Better to be explicit:**
   - Report formal uncertainties
   - Report χ²ᵣ separately
   - Let users decide how to interpret

### JUG/PINT Approach:
Give you the **formal uncertainties** and let YOU decide whether to scale based on fit quality. More transparent!

---

## Practical Recommendations

### For Model Selection:
✅ Use χ²ᵣ directly - it's not affected by uncertainty scaling  
✅ Compare Δχ² between nested models  
✅ Use F-test or likelihood ratio test  
✅ Ignore the inflated Tempo2 uncertainties for this purpose  

### For Reporting Parameters:
Choose based on your goal:

**Formal uncertainties (JUG/PINT):**
- Tool comparisons
- When χ²ᵣ ≈ 1 (good fit with proper noise modeling)
- Internal consistency checks
- You want actual statistical uncertainty

**Inflated uncertainties (Tempo2):**
- Publication (be conservative)
- When χ²ᵣ >> 1 and you haven't modeled noise
- Comparing to literature values (may use Tempo2 convention)
- You want to account for poor fit quality

### Best Practice:
**Model the noise properly!** Then:
- χ²ᵣ ≈ 1 (good fit)
- Formal uncertainties are appropriate
- No need for ad-hoc inflation
- Bayesian posteriors give robust uncertainties

---

## Key Takeaways

1. ✅ **Your intuition was correct:** Scaling TOA errors by √(χ²ᵣ) would make χ²ᵣ = 1
2. ✅ **But Tempo2 doesn't do that:** Only scales parameter uncertainties in output
3. ✅ **χ²ᵣ is preserved:** Still useful for model selection
4. ✅ **Two separate concepts:** 
   - Fit quality (χ²ᵣ) - unchanged
   - Parameter uncertainty reporting - inflated
5. ✅ **JUG/PINT are clearer:** Report formal errors, let you decide

### The Bottom Line:
Tempo2's uncertainty scaling is a **reporting convention**, not part of the fitting algorithm. The χ²ᵣ you see is based on the original TOA errors and remains useful for model comparison. 

JUG and PINT follow the modern approach: report formal uncertainties and let the user interpret based on fit quality. This is more transparent and flexible.

---

## Files Created

1. `explain_tempo2_chi2_scaling.py` - Interactive demonstration
2. `TEMPO2_UNCERTAINTY_EXPLANATION.md` - Original detailed analysis
3. This document - Focused answer to your specific question

**The example shows:**
- F0-only: χ²ᵣ = 104,419,650 (terrible!)
- F0+F1: χ²ᵣ = 32.26 (much better but still high)
- Δχ² test clearly favors F0+F1
- Model selection works perfectly despite the poor absolute fit quality
