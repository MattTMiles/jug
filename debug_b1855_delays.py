"""Compare JUG vs PINT delay components for B1855+09."""

from pathlib import Path
import numpy as np

# Get JUG residuals
from jug.residuals.simple_calculator import compute_residuals_simple

par = Path('data/pulsars/NG_data/PINT_testing_data/B1855+09_NANOGrav_12yv3.wb.gls.par')
tim = Path('data/pulsars/NG_data/PINT_testing_data/B1855+09_NANOGrav_12yv3.wb.tim')

print("=" * 70)
print("JUG vs PINT Delay Component Comparison")
print("=" * 70)

print("\n1. Computing JUG residuals...")
jug_result = compute_residuals_simple(par, tim, verbose=False)
jug_resids = jug_result['residuals_us']
jug_wrms = jug_result['weighted_rms_us']
print(f"   JUG prefit wRMS: {jug_wrms:.6f} μs")
print(f"   JUG residual range: [{jug_resids.min():.2f}, {jug_resids.max():.2f}] μs")
print(f"   JUG mean: {jug_result['mean_us']:.2f} μs")
print(f"   JUG unweighted RMS: {jug_result['unweighted_rms_us']:.2f} μs")

print("\n2. Computing PINT residuals...")
try:
    import pint.toa as toa
    import pint.models as models
    from pint.residuals import Residuals
    
    # Load with PINT
    toas_pint = toa.get_TOAs(str(tim))
    model_pint = models.get_model(str(par))
    
    # Compute PINT residuals
    resids_pint_obj = Residuals(toas_pint, model_pint)
    resids_pint = resids_pint_obj.time_resids.to_value('us')
    wrms_pint = resids_pint_obj.rms_weighted().to_value('us')
    
    print(f"   PINT prefit wRMS: {wrms_pint:.6f} μs")
    print(f"   PINT residual range: [{resids_pint.min():.2f}, {resids_pint.max():.2f}] μs")
    print(f"   PINT mean: {resids_pint.mean():.2f} μs")
    
    # Compare residuals
    diff = jug_resids - resids_pint
    print(f"\n3. Residual difference (JUG - PINT):")
    print(f"   Mean: {diff.mean():.6f} μs")
    print(f"   Std: {diff.std():.6f} μs")
    print(f"   Range: [{diff.min():.2f}, {diff.max():.2f}] μs")
    
    # Check if frequency-dependent
    print(f"\n4. Frequency dependence of difference:")
    freqs = toas_pint.table['freq'].value
    
    # Group by frequency
    freq_groups = {}
    for i, f in enumerate(freqs):
        key = f"{f:.0f}"
        if key not in freq_groups:
            freq_groups[key] = []
        freq_groups[key].append(i)
    
    for freq_key in sorted(freq_groups.keys(), key=lambda x: float(x)):
        idx = freq_groups[freq_key]
        print(f"   {freq_key} MHz: mean diff = {diff[idx].mean():.2f} μs, "
              f"std = {diff[idx].std():.2f} μs, N = {len(idx)}")
    
    # Try to extract delay components from PINT
    print(f"\n5. Extracting JUG delay components...")
    print(f"   Roemer+Shapiro range: [{jug_result['roemer_shapiro_sec'].min():.6f}, {jug_result['roemer_shapiro_sec'].max():.6f}] s")
    print(f"   DM delay range: [{jug_result['dm_delay_sec'].min():.6f}, {jug_result['dm_delay_sec'].max():.6f}] s")
    print(f"   Solar wind delay range: [{jug_result['sw_delay_sec'].min():.6f}, {jug_result['sw_delay_sec'].max():.6f}] s")
    print(f"   Total delay range: [{jug_result['total_delay_sec'].min():.6f}, {jug_result['total_delay_sec'].max():.6f}] s")
    
    # DM delay statistics by frequency band
    print(f"\n6. JUG DM delay by frequency band:")
    for freq_key in ['430', '1400']:
        freq_val = float(freq_key)
        if freq_key == '430':
            mask = freqs < 500
        else:
            mask = freqs > 1000
        if np.any(mask):
            dm_delays_us = jug_result['dm_delay_sec'][mask] * 1e6
            print(f"   {freq_key} MHz band: mean = {dm_delays_us.mean():.2f} μs, "
                  f"range = [{dm_delays_us.min():.2f}, {dm_delays_us.max():.2f}] μs")
    
    print(f"\n7. Checking frequency-dependent pattern...")
    # Fit difference vs 1/freq^2 (DM signature)
    inv_freq_sq = 1.0 / (freqs**2)
    from scipy.stats import linregress
    slope, intercept, r, p, stderr = linregress(inv_freq_sq, diff)
    print(f"   Linear fit: diff = {slope:.2e} / freq² + {intercept:.2f}")
    print(f"   R² = {r**2:.4f}")
    print(f"   This implies a DM offset of ~{slope * 2.41e-4:.6f} pc/cm³")
    
    try:
        # Get delay components from PINT too
        print(f"\n8. Attempting to extract PINT delay components...")
        # Get delay function directly
        d = model_pint.delay(toas_pint)
        total_pint = d.to_value('s')
        print(f"   PINT total delay range: [{total_pint.min():.6f}, {total_pint.max():.6f}] s")
        
        # Compare total delays
        delay_diff = jug_result['total_delay_sec'] - total_pint
        print(f"\n9. Total delay difference (JUG - PINT):")
        print(f"   Mean: {delay_diff.mean():.9f} s = {delay_diff.mean()*1e6:.3f} μs")
        print(f"   Std: {delay_diff.std():.9f} s = {delay_diff.std()*1e6:.3f} μs")
        print(f"   Range: [{delay_diff.min()*1e6:.2f}, {delay_diff.max()*1e6:.2f}] μs")
        
    except Exception as e:
        print(f"   Could not extract detailed components: {e}")
    
except ImportError:
    print("   PINT not available - skipping PINT comparison")
except Exception as e:
    print(f"   Error loading PINT: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
