# Test if we can solve WLS manually with longdouble using scipy

import numpy as np
import scipy.linalg

def wls_solve_longdouble_manual(residuals_ld, errors_ld, M_ld):
    """WLS solver in longdouble using manual normal equations.
    
    Solves: (M^T W M) delta = M^T W r
    where W = diag(1/error^2)
    """
    weights_solve = np.longdouble(1.0) / errors_ld
    M_weighted = M_ld * weights_solve[:, None]
    r_weighted = residuals_ld * weights_solve
    
    # Normal equations: A^T A x = A^T b
    ATA = M_weighted.T @ M_weighted
    ATb = M_weighted.T @ r_weighted
    
    # Solve using scipy.linalg.solve (supports longdouble)
    delta_params = scipy.linalg.solve(ATA, ATb)
    return delta_params

# Test with longdouble
M_ld = np.random.randn(100, 2).astype(np.longdouble)
r_ld = np.random.randn(100).astype(np.longdouble)
e_ld = (np.ones(100) * 0.01).astype(np.longdouble)

result_ld = wls_solve_longdouble_manual(r_ld, e_ld, M_ld)
print(f"Longdouble result: {result_ld}")
print(f"Dtype: {result_ld.dtype}")
print("Success!")
