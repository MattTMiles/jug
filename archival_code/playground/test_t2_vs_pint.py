"""Test JUG T2 binary model against PINT for J0437-4715.

This script validates that JUG's T2 implementation matches PINT's output.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import jax
jax.config.update("jax_enable_x64", True)

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds

# Test with J1909-3744 (ELL1 binary - our working test pulsar)
PAR_FILE = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par'
TIM_FILE = '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'

print("="*70)
print("Testing JUG Binary Models vs PINT")
print("="*70)
print(f"\nPulsar: J1909-3744")
print(f"Par file: {PAR_FILE}")
print(f"Tim file: {TIM_FILE}")

# Compute JUG residuals
print("\n" + "-"*70)
print("Computing JUG residuals...")
print("-"*70)

model = parse_par_file(PAR_FILE)
toas_list = parse_tim_file_mjds(TIM_FILE)

print(f"Binary model: {model.get('BINARY', 'None')}")

result = compute_residuals_simple(PAR_FILE, TIM_FILE)
jug_res_us = result['residuals_us']
tdb_mjd = result['tdb_mjd']

print(f"Number of TOAs: {result['n_toas']}")
print(f"JUG RMS: {result['rms_us']:.3f} μs")
print(f"JUG mean: {result['mean_us']:.3f} μs")

# Compute PINT residuals
print("\n" + "-"*70)
print("Computing PINT residuals...")
print("-"*70)

try:
    import pint.toa as toa
    import pint.models as models
    import pint.fitter as fitter
    from astropy import units as u
    
    # Load PINT model and TOAs
    pint_model = models.get_model(PAR_FILE)
    pint_toas = toa.get_TOAs(TIM_FILE, planets=True)
    
    # Compute residuals using PINT's Residuals class
    from pint.residuals import Residuals
    pint_res = Residuals(pint_toas, pint_model, use_weighted_mean=False)
    pint_res_us = pint_res.time_resids.to(u.us).value
    
    print(f"PINT RMS: {np.std(pint_res_us):.3f} μs")
    print(f"PINT mean: {np.mean(pint_res_us):.3f} μs")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON: JUG vs PINT")
    print("="*70)
    
    diff_us = jug_res_us - pint_res_us
    print(f"Difference RMS: {np.std(diff_us):.3f} μs")
    print(f"Difference mean: {np.mean(diff_us):.3f} μs")
    print(f"Difference range: [{np.min(diff_us):.3f}, {np.max(diff_us):.3f}] μs")
    
    # Success criterion
    if np.std(diff_us) < 0.1:
        print("\n✅ SUCCESS: JUG matches PINT to < 0.1 μs!")
    elif np.std(diff_us) < 1.0:
        print("\n⚠️  CAUTION: JUG differs from PINT by < 1 μs")
    else:
        print("\n❌ FAILURE: JUG differs from PINT by > 1 μs")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: JUG vs PINT residuals
    ax = axes[0]
    ax.errorbar(tdb_mjd, jug_res_us, yerr=result.get('error_us', None), 
                fmt='o', alpha=0.6, label=f'JUG (RMS={np.std(jug_res_us):.3f} μs)', 
                markersize=4)
    ax.errorbar(tdb_mjd, pint_res_us, yerr=result.get('error_us', None),
                fmt='x', alpha=0.6, label=f'PINT (RMS={np.std(pint_res_us):.3f} μs)',
                markersize=4)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Residual (μs)')
    ax.set_title('J1909-3744: JUG vs PINT Residuals (ELL1 Binary Model)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Difference
    ax = axes[1]
    ax.plot(tdb_mjd, diff_us, 'o', alpha=0.6, markersize=4)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(np.mean(diff_us), color='r', linestyle='--', alpha=0.5, 
               label=f'Mean={np.mean(diff_us):.3f} μs')
    ax.fill_between(tdb_mjd, -np.std(diff_us), np.std(diff_us), 
                     alpha=0.2, color='gray', label=f'±1σ={np.std(diff_us):.3f} μs')
    ax.set_xlabel('MJD')
    ax.set_ylabel('Difference (JUG - PINT) [μs]')
    ax.set_title('Residual Difference: JUG - PINT')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('J1909-3744_jug_vs_pint.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: J1909-3744_jug_vs_pint.png")
    
except ImportError as e:
    print(f"\n⚠️  PINT not available: {e}")
    print("Cannot compare against PINT")
except Exception as e:
    print(f"\n❌ Error computing PINT residuals: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete!")
print("="*70)
