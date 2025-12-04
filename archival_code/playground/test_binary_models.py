"""Test all binary models against PINT.

This script validates JUG's binary model implementations by comparing
residuals against PINT for real pulsars with different binary models.

Usage:
    python test_binary_models.py                    # Test all models
    python test_binary_models.py --model DD         # Test specific model
    python test_binary_models.py --pulsar J1012    # Test specific pulsar
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from jug.residuals.simple_calculator import compute_residuals_simple
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals

# Target: < 50 ns RMS difference
TARGET_NS = 50.0

# Test cases: (pulsar_name, binary_model, par_file, tim_file)
TEST_CASES = {
    # DD variants
    'J1012-4235': ('DD', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235_tdb.par', 
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1012-4235.tim'),
    'J0101-6422': ('DD', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0101-6422_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0101-6422.tim'),
    'J1017-7156': ('DDH', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1017-7156_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1017-7156.tim'),
    'J1022+1001': ('DDH', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1022+1001.tim'),
    'J0437-4715': ('DDK', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0437-4715.tim'),
    'J0955-6150': ('DDGR', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0955-6150_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0955-6150.tim'),
    
    # ELL1 variants (already working, for comparison)
    'J1909-3744': ('ELL1', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J1909-3744.tim'),
    'J0125-2327': ('ELL1H', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0125-2327_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0125-2327.tim'),
    
    # Non-binary (for completeness)
    'J0030+0451': ('NONE', '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0030+0451_tdb.par',
                   '/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb/J0030+0451.tim'),
}


def test_binary_model(pulsar_name, binary_model, par_file, tim_file, clock_dir="data/clock", plot=True):
    """Test a single pulsar binary model against PINT.
    
    Returns
    -------
    dict
        Results containing RMS difference, pass/fail status, etc.
    """
    print("="*80)
    print(f"{pulsar_name} ({binary_model} BINARY) VALIDATION TEST")
    print("="*80)
    
    # Check files exist
    if not Path(par_file).exists():
        print(f"⚠️  SKIPPED: {par_file} not found")
        return None
    if not Path(tim_file).exists():
        print(f"⚠️  SKIPPED: {tim_file} not found")
        return None
    
    try:
        # Compute JUG residuals
        print("\nComputing JUG residuals...")
        jug_result = compute_residuals_simple(par_file, tim_file, clock_dir=clock_dir)
        jug_res_us = jug_result['residuals_us']
        print(f"  JUG: {len(jug_res_us)} TOAs")
        print(f"  JUG RMS: {np.std(jug_res_us):.6f} μs")
        
        # Compute PINT residuals (use BIPM2024 and DE440 to match JUG)
        print("\nComputing PINT residuals...")
        model_pint = get_model(par_file)
        toas_pint = get_TOAs(tim_file, planets=True, ephem="DE440", include_bipm=True, bipm_version="BIPM2024")
        res_pint = Residuals(toas_pint, model_pint)
        pint_res_us = res_pint.time_resids.to_value('us')
        print(f"  PINT: {len(pint_res_us)} TOAs")
        print(f"  PINT RMS: {np.std(pint_res_us):.6f} μs")
        
        # Compute difference
        diff_us = jug_res_us - pint_res_us
        diff_ns = diff_us * 1000.0
        
        print("\n" + "="*80)
        print("COMPARISON (JUG - PINT)")
        print("="*80)
        print(f"\nDifference statistics:")
        print(f"  Mean: {np.mean(diff_us):.6f} μs = {np.mean(diff_ns):.3f} ns")
        print(f"  RMS:  {np.std(diff_us):.6f} μs = {np.std(diff_ns):.3f} ns")
        print(f"  Max:  {np.max(np.abs(diff_us)):.6f} μs = {np.max(np.abs(diff_ns)):.3f} ns")
        
        # Check success
        passed = np.std(diff_ns) < TARGET_NS
        print("\n" + "="*80)
        if passed:
            print(f"✅ SUCCESS! RMS difference ({np.std(diff_ns):.1f} ns) < {TARGET_NS} ns target")
            print(f"   {binary_model} binary model is working correctly!")
        else:
            print(f"❌ FAILED! RMS difference ({np.std(diff_ns):.1f} ns) > {TARGET_NS} ns target")
            print(f"   Factor over target: {np.std(diff_ns) / TARGET_NS:.1f}×")
        
        # Create comparison plots
        if plot:
            print("\nCreating comparison plots...")
            mjds = toas_pint.get_mjds().value
            dt_years = (mjds - mjds[0]) / 365.25
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            # Plot 1: JUG and PINT residuals
            ax = axes[0]
            ax.plot(dt_years, jug_res_us, 'b.', markersize=2, alpha=0.5, label='JUG')
            ax.plot(dt_years, pint_res_us, 'r.', markersize=2, alpha=0.5, label='PINT')
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Residuals (μs)')
            ax.set_title(f'{pulsar_name} ({binary_model}): JUG vs PINT Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Difference (μs)
            ax = axes[1]
            ax.plot(dt_years, diff_us, 'g.', markersize=2, alpha=0.6)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Difference (μs)')
            ax.set_title(f'Residual Difference (JUG - PINT): RMS = {np.std(diff_us):.6f} μs')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Difference (ns)
            ax = axes[2]
            ax.plot(dt_years, diff_ns, 'g.', markersize=2, alpha=0.6)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.axhline(TARGET_NS, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'±{TARGET_NS} ns target')
            ax.axhline(-TARGET_NS, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylabel('Difference (ns)')
            ax.set_xlabel('Time (years since start)')
            ax.set_title(f'Residual Difference (JUG - PINT): RMS = {np.std(diff_ns):.3f} ns')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Histogram of differences (ns)
            ax = axes[3]
            ax.hist(diff_ns, bins=50, color='green', alpha=0.7, edgecolor='black')
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.axvline(TARGET_NS, color='r', linestyle='--', linewidth=1, alpha=0.5, label=f'±{TARGET_NS} ns')
            ax.axvline(-TARGET_NS, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Difference (ns)')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution: Mean = {np.mean(diff_ns):.3f} ns, RMS = {np.std(diff_ns):.3f} ns')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_file = f'{pulsar_name.lower().replace("+", "p")}_{binary_model.lower()}_comparison.png'
            plt.savefig(plot_file, dpi=150)
            print(f"Plot saved to: {plot_file}")
            plt.close()
        
        return {
            'pulsar': pulsar_name,
            'binary_model': binary_model,
            'n_toas': len(jug_res_us),
            'jug_rms_us': np.std(jug_res_us),
            'pint_rms_us': np.std(pint_res_us),
            'diff_mean_ns': np.mean(diff_ns),
            'diff_rms_ns': np.std(diff_ns),
            'diff_max_ns': np.max(np.abs(diff_ns)),
            'passed': passed
        }
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Test JUG binary models against PINT')
    parser.add_argument('--model', type=str, help='Test specific binary model (DD, DDH, DDK, etc.)')
    parser.add_argument('--pulsar', type=str, help='Test specific pulsar (J1012, J0437, etc.)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()
    
    # Filter test cases
    test_cases = TEST_CASES.copy()
    if args.model:
        test_cases = {k: v for k, v in test_cases.items() if v[0] == args.model.upper()}
    if args.pulsar:
        test_cases = {k: v for k, v in test_cases.items() if args.pulsar.upper() in k}
    
    if not test_cases:
        print(f"No test cases found matching filters")
        return
    
    print(f"\nTesting {len(test_cases)} pulsars...\n")
    
    # Run tests
    results = []
    for pulsar_name, (binary_model, par_file, tim_file) in test_cases.items():
        result = test_binary_model(pulsar_name, binary_model, par_file, tim_file, plot=not args.no_plot)
        if result:
            results.append(result)
        print()
    
    # Summary table
    if results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"\n{'Pulsar':<15} {'Model':<8} {'N_TOAs':<8} {'JUG RMS':<12} {'PINT RMS':<12} {'Diff RMS':<12} {'Status':<8}")
        print(f"{'':=<15} {'':=<8} {'':=<8} {'':=<12} {'':=<12} {'':=<12} {'':=<8}")
        
        for r in results:
            status = '✅ PASS' if r['passed'] else '❌ FAIL'
            print(f"{r['pulsar']:<15} {r['binary_model']:<8} {r['n_toas']:<8} "
                  f"{r['jug_rms_us']:>10.3f} μs {r['pint_rms_us']:>10.3f} μs "
                  f"{r['diff_rms_ns']:>10.1f} ns {status:<8}")
        
        # Overall statistics
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        print(f"\n{'':=<80}")
        print(f"Overall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        
        # Group by model
        models = {}
        for r in results:
            model = r['binary_model']
            if model not in models:
                models[model] = []
            models[model].append(r)
        
        print(f"\nBy model:")
        for model, model_results in sorted(models.items()):
            model_passed = sum(1 for r in model_results if r['passed'])
            model_total = len(model_results)
            avg_rms = np.mean([r['diff_rms_ns'] for r in model_results])
            status = '✅' if model_passed == model_total else '⚠️'
            print(f"  {status} {model:<8}: {model_passed}/{model_total} passed, avg RMS = {avg_rms:.1f} ns")


if __name__ == '__main__':
    main()
