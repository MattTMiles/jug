
import numpy as np
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')
from jug.residuals.simple_calculator import compute_residuals_simple

# 1. Run JUG
par = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236_tdb.par'
tim = '/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb/J2241-5236.tim'
print('Running JUG...')
res_jug = compute_residuals_simple(par, tim, verbose=False)
jug_mjds = res_jug['tdb_mjd']
jug_res = res_jug['residuals_us'] * 1e-6  # Convert us to sec

# 2. Load Tempo2 residuals
print('Loading Tempo2 residuals...')
t2_data = np.loadtxt('t2_prefit_j2241_clean.txt')
t2_mjds = t2_data[:, 0]
t2_res = t2_data[:, 1]
t2_err = t2_data[:, 2] * 1e-6

# Compute weighted RMS for Tempo2
weights_t2 = 1.0 / (t2_err**2)
weighted_mean_t2 = np.sum(t2_res * weights_t2) / np.sum(weights_t2)
weighted_rms_t2 = np.sqrt(np.sum(weights_t2 * (t2_res - weighted_mean_t2)**2) / np.sum(weights_t2))

# 3. Compare
idx_jug = np.argsort(jug_mjds)
idx_t2 = np.argsort(t2_mjds)

jug_res_sorted = jug_res[idx_jug]
t2_res_sorted = t2_res[idx_t2]

diff = jug_res_sorted - t2_res_sorted
rms_diff = np.sqrt(np.mean(diff**2))

print(f'\nComparison Results (JUG - Tempo2 Pre-Fit):')
print(f'JUG Weighted RMS: {res_jug["weighted_rms_us"]:.6f} us')
print(f'Tempo2 Weighted RMS: {weighted_rms_t2*1e6:.6f} us')
print(f'Difference RMS: {rms_diff*1e6:.6f} us')

if abs(res_jug["weighted_rms_us"] - weighted_rms_t2*1e6) < 0.001:
    print('✅ MATCH: RMS values match!')
else:
    print('❌ MISMATCH: RMS values differ')
