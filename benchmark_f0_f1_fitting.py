#!/usr/bin/env python3
"""
Benchmark F0+F1 fitting performance: Tempo2 vs PINT vs JUG (optimized)
Includes prefit/postfit residual plots and weighted RMS metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess
import tempfile
import os

def benchmark_tempo2(par_file, tim_file):
    """Benchmark Tempo2 F0+F1 fitting."""
    print("\n" + "="*60)
    print("TEMPO2 BENCHMARK")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temp directory
        import shutil
        tmp_par = os.path.join(tmpdir, 'pulsar.par')
        tmp_tim = os.path.join(tmpdir, 'pulsar.tim')
        shutil.copy(par_file, tmp_par)
        shutil.copy(tim_file, tmp_tim)
        
        # Get prefit residuals
        start = time.time()
        result_pre = subprocess.run(
            ['tempo2', '-f', tmp_par, tmp_tim, '-nofit', '-output', 'general2'],
            cwd=tmpdir, capture_output=True, text=True
        )
        
        # Read prefit residuals
        gen2_file = os.path.join(tmpdir, 'general2.tmp')
        if os.path.exists(gen2_file):
            data = np.loadtxt(gen2_file, usecols=(2, 3))  # residual_us, error_us
            prefit_res = data[:, 0]
            errors = data[:, 1]
            prefit_wrms = np.sqrt(np.average(prefit_res**2, weights=1.0/errors**2))
        else:
            prefit_res = None
            prefit_wrms = None
        
        # Run fitting
        result_fit = subprocess.run(
            ['tempo2', '-f', tmp_par, tmp_tim, '-fit', 'F0', '-fit', 'F1', '-output', 'general2'],
            cwd=tmpdir, capture_output=True, text=True
        )
        elapsed = time.time() - start
        
        # Read postfit residuals
        if os.path.exists(gen2_file):
            data = np.loadtxt(gen2_file, usecols=(2, 3))
            postfit_res = data[:, 0]
            errors = data[:, 1]
            postfit_wrms = np.sqrt(np.average(postfit_res**2, weights=1.0/errors**2))
        else:
            postfit_res = None
            postfit_wrms = None
        
        # Extract fitted parameters
        new_par = os.path.join(tmpdir, 'new.par')
        f0_fit = f1_fit = None
        if os.path.exists(new_par):
            with open(new_par, 'r') as f:
                for line in f:
                    if line.startswith('F0'):
                        f0_fit = float(line.split()[1])
                    elif line.startswith('F1'):
                        f1_fit = float(line.split()[1])
        
        print(f"Time: {elapsed:.3f}s")
        print(f"Prefit WRMS: {prefit_wrms:.6f} μs" if prefit_wrms else "Prefit WRMS: N/A")
        print(f"Postfit WRMS: {postfit_wrms:.6f} μs" if postfit_wrms else "Postfit WRMS: N/A")
        print(f"F0: {f0_fit:.20f} Hz" if f0_fit else "F0: N/A")
        print(f"F1: {f1_fit:.15e} Hz/s" if f1_fit else "F1: N/A")
        
        return {
            'time': elapsed,
            'prefit_res': prefit_res,
            'postfit_res': postfit_res,
            'prefit_wrms': prefit_wrms,
            'postfit_wrms': postfit_wrms,
            'f0': f0_fit,
            'f1': f1_fit,
            'errors': errors
        }

def benchmark_pint(par_file, tim_file):
    """Benchmark PINT F0+F1 fitting."""
    print("\n" + "="*60)
    print("PINT BENCHMARK")
    print("="*60)
    
    import pint.models as pm
    import pint.toa as pt
    import pint.fitter as pf
    
    start = time.time()
    model = pm.get_model(par_file)
    toas = pt.get_TOAs(tim_file, model=model)
    
    # Prefit residuals
    prefit_res = pint.residuals.Residuals(toas, model)
    prefit_res_us = prefit_res.time_resids.to_value('us')
    errors_us = toas.get_errors().to_value('us')
    prefit_wrms = np.sqrt(np.average(prefit_res_us**2, weights=1.0/errors_us**2))
    
    # Fit
    fitter = pf.WLSFitter(toas, model)
    fitter.fit_toas()
    
    # Postfit residuals
    postfit_res_us = fitter.resids.time_resids.to_value('us')
    postfit_wrms = np.sqrt(np.average(postfit_res_us**2, weights=1.0/errors_us**2))
    
    elapsed = time.time() - start
    
    f0_fit = fitter.model.F0.quantity.value
    f1_fit = fitter.model.F1.quantity.value
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Prefit WRMS: {prefit_wrms:.6f} μs")
    print(f"Postfit WRMS: {postfit_wrms:.6f} μs")
    print(f"F0: {f0_fit:.20f} Hz")
    print(f"F1: {f1_fit:.15e} Hz/s")
    print(f"Iterations: {fitter.iteration}")
    
    return {
        'time': elapsed,
        'prefit_res': prefit_res_us,
        'postfit_res': postfit_res_us,
        'prefit_wrms': prefit_wrms,
        'postfit_wrms': postfit_wrms,
        'f0': f0_fit,
        'f1': f1_fit,
        'errors': errors_us,
        'iterations': fitter.iteration
    }

def benchmark_jug(par_file, tim_file):
    """Benchmark JUG F0+F1 fitting with Level 2 JAX optimization."""
    print("\n" + "="*60)
    print("JUG BENCHMARK (Level 2 JAX Optimized)")
    print("="*60)
    
    # Import JUG components
    from jug.io.par_reader import read_par_file
    from jug.io.tim_reader import read_tim_file
    from jug.residuals.simple_calculator import compute_residuals
    from jug.fitting.optimized_jax_fitter import fit_parameters_jax
    
    start = time.time()
    
    # Load data
    params = read_par_file(par_file)
    tim_data = read_tim_file(tim_file)
    
    # Prefit residuals
    prefit_result = compute_residuals(params, tim_data)
    prefit_res_us = prefit_result['residuals_us']
    errors_us = prefit_result['errors_us']
    prefit_wrms = np.sqrt(np.average(prefit_res_us**2, weights=1.0/errors_us**2))
    
    # Fit
    fit_params = ['F0', 'F1']
    result = fit_parameters_jax(params, tim_data, fit_params, max_iter=20, tol=1e-9)
    
    # Postfit residuals
    postfit_result = compute_residuals(result['params'], tim_data)
    postfit_res_us = postfit_result['residuals_us']
    postfit_wrms = np.sqrt(np.average(postfit_res_us**2, weights=1.0/errors_us**2))
    
    elapsed = time.time() - start
    
    f0_fit = result['params']['F0']
    f1_fit = result['params']['F1']
    
    print(f"Time: {elapsed:.3f}s")
    print(f"Prefit WRMS: {prefit_wrms:.6f} μs")
    print(f"Postfit WRMS: {postfit_wrms:.6f} μs")
    print(f"F0: {f0_fit:.20f} Hz")
    print(f"F1: {f1_fit:.15e} Hz/s")
    print(f"Iterations: {result['iterations']}")
    print(f"Converged: {result['converged']}")
    
    return {
        'time': elapsed,
        'prefit_res': prefit_res_us,
        'postfit_res': postfit_res_us,
        'prefit_wrms': prefit_wrms,
        'postfit_wrms': postfit_wrms,
        'f0': f0_fit,
        'f1': f1_fit,
        'errors': errors_us,
        'iterations': result['iterations'],
        'converged': result['converged']
    }

def plot_comparison(tempo2_res, pint_res, jug_res):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    methods = [
        ('Tempo2', tempo2_res),
        ('PINT', pint_res),
        ('JUG (Level 2 JAX)', jug_res)
    ]
    
    for idx, (name, res) in enumerate(methods):
        # Prefit
        ax_pre = axes[0, idx]
        if res['prefit_res'] is not None:
            ax_pre.scatter(range(len(res['prefit_res'])), res['prefit_res'], 
                          s=1, alpha=0.5, c='blue')
            ax_pre.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax_pre.set_xlabel('TOA Index')
            ax_pre.set_ylabel('Residual (μs)')
            ax_pre.set_title(f'{name} Prefit\nWRMS={res["prefit_wrms"]:.3f} μs')
            ax_pre.grid(True, alpha=0.3)
        
        # Postfit
        ax_post = axes[1, idx]
        if res['postfit_res'] is not None:
            ax_post.scatter(range(len(res['postfit_res'])), res['postfit_res'], 
                           s=1, alpha=0.5, c='red')
            ax_post.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax_post.set_xlabel('TOA Index')
            ax_post.set_ylabel('Residual (μs)')
            ax_post.set_title(f'{name} Postfit\nWRMS={res["postfit_wrms"]:.3f} μs')
            ax_post.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_f0_f1_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: benchmark_f0_f1_results.png")
    plt.close()

def main():
    par_file = 'data/J1909-3744_tdb_wrong.par'
    tim_file = 'data/J1909-3744.tim'
    
    print("="*60)
    print("F0+F1 FITTING BENCHMARK")
    print("="*60)
    print(f"Par file: {par_file}")
    print(f"Tim file: {tim_file}")
    
    # Run benchmarks
    tempo2_res = benchmark_tempo2(par_file, tim_file)
    pint_res = benchmark_pint(par_file, tim_file)
    jug_res = benchmark_jug(par_file, tim_file)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<25} {'Time (s)':<12} {'Postfit WRMS (μs)':<20} {'Iterations':<12}")
    print("-"*60)
    print(f"{'Tempo2':<25} {tempo2_res['time']:<12.3f} {tempo2_res['postfit_wrms']:<20.6f} {'N/A':<12}")
    print(f"{'PINT':<25} {pint_res['time']:<12.3f} {pint_res['postfit_wrms']:<20.6f} {pint_res['iterations']:<12}")
    print(f"{'JUG (Level 2 JAX)':<25} {jug_res['time']:<12.3f} {jug_res['postfit_wrms']:<20.6f} {jug_res['iterations']:<12}")
    
    print("\nSpeedup vs PINT:")
    print(f"  JUG: {pint_res['time']/jug_res['time']:.2f}x faster")
    
    print("\nSpeedup vs Tempo2:")
    print(f"  JUG: {tempo2_res['time']/jug_res['time']:.2f}x faster")
    
    # Plot comparison
    plot_comparison(tempo2_res, pint_res, jug_res)
    
    # Save numerical results
    with open('benchmark_f0_f1_results.txt', 'w') as f:
        f.write("F0+F1 FITTING BENCHMARK RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Par file: {par_file}\n")
        f.write(f"Tim file: {tim_file}\n\n")
        
        f.write("TEMPO2:\n")
        f.write(f"  Time: {tempo2_res['time']:.3f}s\n")
        f.write(f"  Prefit WRMS: {tempo2_res['prefit_wrms']:.6f} μs\n")
        f.write(f"  Postfit WRMS: {tempo2_res['postfit_wrms']:.6f} μs\n")
        f.write(f"  F0: {tempo2_res['f0']:.20f} Hz\n")
        f.write(f"  F1: {tempo2_res['f1']:.15e} Hz/s\n\n")
        
        f.write("PINT:\n")
        f.write(f"  Time: {pint_res['time']:.3f}s\n")
        f.write(f"  Prefit WRMS: {pint_res['prefit_wrms']:.6f} μs\n")
        f.write(f"  Postfit WRMS: {pint_res['postfit_wrms']:.6f} μs\n")
        f.write(f"  F0: {pint_res['f0']:.20f} Hz\n")
        f.write(f"  F1: {pint_res['f1']:.15e} Hz/s\n")
        f.write(f"  Iterations: {pint_res['iterations']}\n\n")
        
        f.write("JUG (Level 2 JAX):\n")
        f.write(f"  Time: {jug_res['time']:.3f}s\n")
        f.write(f"  Prefit WRMS: {jug_res['prefit_wrms']:.6f} μs\n")
        f.write(f"  Postfit WRMS: {jug_res['postfit_wrms']:.6f} μs\n")
        f.write(f"  F0: {jug_res['f0']:.20f} Hz\n")
        f.write(f"  F1: {jug_res['f1']:.15e} Hz/s\n")
        f.write(f"  Iterations: {jug_res['iterations']}\n")
        f.write(f"  Converged: {jug_res['converged']}\n\n")
        
        f.write("SPEEDUP:\n")
        f.write(f"  JUG vs PINT: {pint_res['time']/jug_res['time']:.2f}x\n")
        f.write(f"  JUG vs Tempo2: {tempo2_res['time']/jug_res['time']:.2f}x\n")
    
    print("\nResults saved to: benchmark_f0_f1_results.txt")

if __name__ == '__main__':
    main()
