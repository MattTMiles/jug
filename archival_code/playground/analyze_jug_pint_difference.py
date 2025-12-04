"""Analyze the remaining difference between JUG and PINT residuals."""

import numpy as np
import matplotlib.pyplot as plt

# Import JUG
from jug.residuals.simple_calculator import compute_residuals_simple

# Import PINT
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Test pulsar
PAR_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235_tdb.par"
TIM_FILE = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1012-4235.tim"

print("="*80)
print("DETAILED JUG vs PINT RESIDUAL COMPARISON")
print("="*80)
print(f"\nPulsar: J1012-4235 (DD binary)")
print(f"Target: < 50 ns RMS difference")

# Compute JUG residuals
print("\n" + "="*80)
print("Computing JUG residuals...")
print("="*80)
jug_result = compute_residuals_simple(PAR_FILE, TIM_FILE, clock_dir="data/clock")
jug_res_us = jug_result['residuals_us']
print(f"JUG Weighted RMS: {jug_result['weighted_rms_us']:.6f} μs")
print(f"JUG Unweighted RMS: {jug_result['unweighted_rms_us']:.6f} μs")

# Compute PINT residuals
print("\n" + "="*80)
print("Computing PINT residuals...")
print("="*80)
model_pint = get_model(PAR_FILE)
toas_pint = get_TOAs(TIM_FILE, planets=True)
res_pint = Residuals(toas_pint, model_pint)
pint_res_us = res_pint.time_resids.to_value('us')
print(f"PINT RMS: {np.std(pint_res_us):.6f} μs")
print(f"PINT Mean: {np.mean(pint_res_us):.6f} μs")

# Compute differences
print("\n" + "="*80)
print("DIFFERENCE ANALYSIS")
print("="*80)

diff_us = jug_res_us - pint_res_us
diff_rms = np.std(diff_us)
diff_mean = np.mean(diff_us)

print(f"\nJUG - PINT Difference:")
print(f"  Mean: {diff_mean:.6f} μs = {diff_mean*1000:.3f} ns")
print(f"  RMS: {diff_rms:.6f} μs = {diff_rms*1000:.3f} ns")
print(f"  Min: {np.min(diff_us):.6f} μs = {np.min(diff_us)*1000:.3f} ns")
print(f"  Max: {np.max(diff_us):.6f} μs = {np.max(diff_us)*1000:.3f} ns")

# Check if there's a constant offset
print(f"\nOffset analysis:")
print(f"  Median difference: {np.median(diff_us):.6f} μs = {np.median(diff_us)*1000:.3f} ns")
diff_no_offset = diff_us - diff_mean
diff_no_offset_rms = np.std(diff_no_offset)
print(f"  RMS after removing mean: {diff_no_offset_rms:.6f} μs = {diff_no_offset_rms*1000:.3f} ns")

# Statistical summary
print(f"\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"\nTarget: < 50 ns RMS")
print(f"Actual: {diff_rms*1000:.3f} ns RMS")
print(f"Factor: {(diff_rms*1000)/50:.1f}× too large")

if diff_rms*1000 < 50:
    print("\n✅ SUCCESS! Within 50 ns target!")
else:
    print("\n❌ Still outside target - need further investigation")

# Look for patterns in the differences
print(f"\n" + "="*80)
print("PATTERN ANALYSIS")
print("="*80)

# Check if difference correlates with time
toa_mjds = toas_pint.get_mjds().value
time_range = np.max(toa_mjds) - np.min(toa_mjds)
print(f"\nTime span: {time_range:.1f} days")

# Correlation with MJD
corr_time = np.corrcoef(toa_mjds, diff_us)[0, 1]
print(f"Correlation with MJD: {corr_time:.6f}")

# Check percentiles
print(f"\nDifference percentiles (μs):")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(diff_us, p)
    print(f"  {p:2d}%: {val:8.6f} μs = {val*1000:8.3f} ns")

# Histogram of differences
print(f"\n" + "="*80)
print("DIFFERENCE HISTOGRAM")
print("="*80)

bins = np.linspace(-2, 2, 41)  # 0.1 μs bins
hist, edges = np.histogram(diff_us, bins=bins)
print(f"\nDistribution of (JUG - PINT) in μs:")
for i in range(len(hist)):
    if hist[i] > 0:
        bar = '#' * int(50 * hist[i] / np.max(hist))
        print(f"  [{edges[i]:6.3f}, {edges[i+1]:6.3f}): {hist[i]:4d} {bar}")

# Save detailed results for further analysis
print(f"\n" + "="*80)
print("SAVING DETAILED DATA")
print("="*80)
np.savez('jug_pint_comparison.npz',
         jug_res_us=jug_res_us,
         pint_res_us=pint_res_us,
         diff_us=diff_us,
         toa_mjds=toa_mjds)
print("Saved to: jug_pint_comparison.npz")

print("\n" + "="*80)
