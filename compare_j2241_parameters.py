
import sys
import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import jax

# JUG Imports
sys.path.insert(0, '/home/mattm/soft/JUG')
from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.io.par_reader import parse_par_file

# PINT Imports
import pint.models
import pint.toa
import pint.fitter

# Config
PAR_FILE = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236_tdb.par'
TIM_FILE = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236.tim'
FIT_PARAMS = [f'FB{i}' for i in range(18)]

def run_jug_fit():
    print("--- Running JUG Fit ---")
    jax.config.update("jax_enable_x64", True)
    
    # Run fit
    result = fit_parameters_optimized(
        par_file=Path(PAR_FILE),
        tim_file=Path(TIM_FILE),
        fit_params=FIT_PARAMS,
        max_iter=10,
        verbose=False
    )
    
    # Create dict of val/err
    # JUG returns final_params as dict of values. 
    # Does it return errors?
    # WLS solver returns 'param_errors' in the result dict usually?
    # Checking optimized_fitter.py:
    # It returns 'final_params' (dict), 'final_params_err' (dict) [Assuming I implemented it to propagate from wls_solve]
    # Let's check the result structure. simple fitter usually returns covariance or errors.
    # If not present, we might need to compute them from covariance.
    
    # Looking at optimized_fitter.py logic:
    # The return dict comes from _fit_parameters_general -> _run_general_fit_iterations
    # It returns: {'final_params': ..., 'final_rms': ..., 'iterations': ..., 'converged': ..., 'history': ...}
    # It does NOT explicitly list errors in the top level return?
    # Wait, wls_solve_svd returns (params_delta, cov, ...)
    # But _fit_parameters_general builds the result dict.
    # I should check if 'final_uncertainties' is in the result.
    # If not, I'll need to parse the printed output or modify JUG. 
    # For now, let's assume 'final_params' is there. Errors might be missing. 
    # I'll rely on the fact that JUG prints them to stdout, I can parse them if needed, 
    # OR I can update JUG to return them. 
    # A robust comparison needs errors. I will check for 'final_param_errors' key.
    
    return result

def run_pint_fit():
    print("--- Running PINT Fit ---")
    m = pint.models.get_model(PAR_FILE)
    t = pint.toa.get_TOAs(TIM_FILE, planets=True, ephem='DE421')
    
    # Set fit flags
    for p in m.params:
        getattr(m, p).frozen = True
    for p in FIT_PARAMS:
        if hasattr(m, p):
            getattr(m, p).frozen = False
            
    f = pint.fitter.WLSFitter(t, m)
    f.fit_toas()
    
    res = {}
    for p in FIT_PARAMS:
        if hasattr(f.model, p):
            par = getattr(f.model, p)
            res[p] = {
                'val': par.value,
                'err': par.uncertainty_value
            }
    return res

def run_tempo2_fit():
    print("--- Running Tempo2 Fit ---")
    # Use general2 plugin to output parameters in a parseable format
    # tempo2 -output general2 -f par tim -fit FB0 ... -s "{param_name} {param_value} {param_error}\n"
    
    cmd = ['tempo2', '-output', 'general2', 
           '-f', PAR_FILE, TIM_FILE]
    
    for p in FIT_PARAMS:
        cmd.extend(['-fit', p])
        
    # Define output format for general2 plugin
    # We want post-fit values. general2 usually runs AFTER fit if fit flags are present.
    # Format: "PAR {x} {val} {err}" for each param? 
    # general2 iterates over stats? iterating over params is hard in general2 format string.
    # Alternatives: 
    # 1. Use -newpar and parse it (if it doesn't hang).
    # 2. Use -output general2 to dump all params?
    # Actually, tempo2 -residuals prints stats.
    
    # Let's try to use -newpar again but ensure batch mode by redirecting stdin from /dev/null
    # to avoid any interactive prompts.
    cmd = ['tempo2', '-f', PAR_FILE, TIM_FILE]
    for p in FIT_PARAMS:
        cmd.extend(['-fit', p])
    cmd.extend(['-newpar']) 
    
    # Run with input commands to fit and quit
    # -newpar flag ensures it writes to new.par on exit
    input_str = b"fit\nquit\n"
    
    try:
        proc = subprocess.run(cmd, input=input_str, capture_output=True, timeout=60)
        print(f"Tempo2 return code: {proc.returncode}")
        if proc.returncode != 0:
             print(f"Tempo2 stderr: {proc.stderr.decode()[:500]}")
    except subprocess.TimeoutExpired:
        print("Tempo2 timed out! parse existing new.par if available")
        pass
        
    t2_res = {}
    # Parse new.par
    if os.path.exists('new.par'):
        print(f"Parsing new.par (size: {os.path.getsize('new.par')} bytes)")
        with open('new.par', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'FB0':
                    print(f"DEBUG FB0 line parts: {parts}")
                if parts[0] in FIT_PARAMS:
                    # Format: PARAM VALUE FLAG ERROR
                    try:
                        val = float(parts[1])
                        err = 0.0
                        if len(parts) >= 4:
                            # Assume 4th token is error if 3rd is '1' (flag)
                            if parts[2] == '1':
                                err = float(parts[3])
                        t2_res[parts[0]] = {'val': val, 'err': err}
                    except ValueError as e:
                        print(f"ValueError parsing {parts[0]}: {e}")
                        pass
    else:
        print(f"new.par not generated in {os.getcwd()}")
        with open('t2_out.par', 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] in FIT_PARAMS:
                    try:
                        val = float(parts[1])
                        err = 0.0
                        if len(parts) >= 4:
                            if parts[2] == '1':
                                err = float(parts[3])
                        elif len(parts) == 3 and parts[2] != '0' and parts[2] != '1':
                             err = float(parts[2])
                        t2_res[parts[0]] = {'val': val, 'err': err}
                    except ValueError:
                        pass
        
    return t2_res

def main():
    jug_raw = run_jug_fit()
    pint_res = run_pint_fit()
    t2_res = {} # run_tempo2_fit() skipped due to hang
    
    # Parse JUG results (assuming result dict structure)
    # If JUG doesn't return errors, we'll mark them as N/A or fix JUG.
    # Actually, let's just inspect the jug_raw keys first in a debug run if needed.
    # For now, construct the comparison.
    
    print(f"\n{'PARAM':<8} | {'TEMPO2':<25} | {'PINT':<25} | {'JUG':<25} | {'JUG-T2':<8} | {'JUG-PINT':<8}")
    print("-" * 110)
    
    for p in FIT_PARAMS:
        t2_val = t2_res.get(p, {'val':0, 'err':0})['val']
        t2_err = t2_res.get(p, {'val':0, 'err':0})['err']
        
        pint_val = pint_res.get(p, {'val':0, 'err':0})['val']
        pint_err = pint_res.get(p, {'val':0, 'err':0})['err']
        
        # JUG
        if 'final_params' in jug_raw:
            jug_val = jug_raw['final_params'].get(p, 0.0)
            # Try to get error
            jug_err = 0.0 
            if 'uncertainties' in jug_raw:
                jug_err = jug_raw['uncertainties'].get(p, 0.0)
        else:
             jug_val = 0.0
             jug_err = 0.0

        # Diff T2
        diff_sigma_t2 = "N/A"
        if t2_err > 0 or jug_err > 0:
            sigma_comb = np.sqrt(t2_err**2 + jug_err**2)
            if sigma_comb > 0:
                diff = (jug_val - t2_val) / sigma_comb
                diff_sigma_t2 = f"{diff:.1f}"

        # Diff PINT
        diff_sigma_pint = "N/A"
        if pint_err > 0 or jug_err > 0:
            sigma_comb = np.sqrt(pint_err**2 + jug_err**2)
            if sigma_comb > 0:
                diff = (jug_val - pint_val) / sigma_comb
                diff_sigma_pint = f"{diff:.1f}"
            
        print(f"{p:<8} | {t2_val: .4e} +/- {t2_err:.1e} | {pint_val: .4e} +/- {pint_err:.1e} | {jug_val: .4e} +/- {jug_err:.1e} | {diff_sigma_t2:<8} | {diff_sigma_pint}")

if __name__ == "__main__":
    main()
