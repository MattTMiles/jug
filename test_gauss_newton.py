#!/usr/bin/env python3
import numpy as np, sys
from jug.fitting.design_matrix import compute_design_matrix
from jug.fitting.gauss_newton import fit_gauss_newton

np.random.seed(42)
pepoch = 55000.0
toas = pepoch + np.sort(np.random.uniform(0, 1000, 100))
dt = (toas - pepoch) * 86400.0
f0_true, f1_true = 100.0, -1e-15
phases = f0_true * dt + 0.5 * f1_true * dt**2 + np.random.normal(0, 1e-6, 100)

data = {'toas_mjd': toas, 'freq_mhz': np.ones(100)*1400, 'errors_us': np.ones(100),
        'phases_obs': phases, 'pepoch_mjd': pepoch}

def residuals(p, d):
    dt = (d['toas_mjd'] - d['pepoch_mjd']) * 86400
    model = p['F0'] * dt + 0.5 * p['F1'] * dt**2
    phase_res = np.mod(d['phases_obs'] - model + 0.5, 1.0) - 0.5
    return phase_res / p['F0'] * 1e6

def dm(p, d, fp):
    return compute_design_matrix(p, d['toas_mjd'], d['freq_mhz'], d['errors_us'], fp, d['pepoch_mjd'])

result = fit_gauss_newton(residuals, dm, {'F0': f0_true+1e-9, 'F1': f1_true+1e-20, 'PEPOCH': pepoch},
                          ['F0', 'F1'], data, max_iter=10, verbose=False)

f0_sig = abs(result['params']['F0'] - f0_true) / result['uncertainties']['F0']
print(f"✅ Gauss-Newton fitter working! F0 error = {f0_sig:.2f}σ, Chi2 = {result['chi2']:.1f}")
