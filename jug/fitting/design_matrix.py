"""Design matrix computation for pulsar timing fitting.

The design matrix contains analytical derivatives ∂(residual)/∂(parameter)
for each timing model parameter. This enables fast Gauss-Newton fitting.
"""

import numpy as np
from typing import Dict, List, Tuple

# Constants
SECS_PER_DAY = 86400.0
K_DM_SEC = 4.148808e3  # DM constant: MHz² pc⁻¹ cm³ s


def compute_design_matrix(
    params: Dict[str, float],
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    errors_us: np.ndarray,
    fit_params: List[str],
    pepoch_mjd: float = None
) -> np.ndarray:
    """Compute analytical design matrix for timing fit.
    
    The design matrix M has shape (n_toas, n_params) where:
        M[i, j] = ∂(residual_i) / ∂(param_j)
    
    All derivatives are computed analytically for speed.
    
    Parameters
    ----------
    params : dict
        Current timing model parameters
    toas_mjd : ndarray
        TOA times in MJD
    freq_mhz : ndarray
        Observing frequencies in MHz
    errors_us : ndarray
        TOA uncertainties in microseconds
    fit_params : list of str
        Names of parameters to fit
    pepoch_mjd : float, optional
        Reference epoch (default: params['PEPOCH'])
    
    Returns
    -------
    design_matrix : ndarray
        Shape (n_toas, n_params), weighted by 1/errors_us
    """
    n_toas = len(toas_mjd)
    n_params = len(fit_params)
    
    if pepoch_mjd is None:
        pepoch_mjd = params.get('PEPOCH', toas_mjd[0])
    
    # Time from epoch in seconds
    dt_sec = (toas_mjd - pepoch_mjd) * SECS_PER_DAY
    
    # Spin frequency (needed for phase -> time conversion)
    f0 = params['F0']
    
    # Initialize design matrix
    M = np.zeros((n_toas, n_params))
    
    # Compute derivatives for each parameter
    for i, param_name in enumerate(fit_params):
        M[:, i] = compute_derivative(
            param_name, params, dt_sec, freq_mhz, f0
        )
    
    # Weight by 1/error (for weighted least squares)
    M = M / errors_us[:, np.newaxis]
    
    return M


def compute_derivative(
    param_name: str,
    params: Dict[str, float],
    dt_sec: np.ndarray,
    freq_mhz: np.ndarray,
    f0: float
) -> np.ndarray:
    """Compute ∂(residual)/∂(parameter) for a single parameter.
    
    Residuals are in microseconds. Derivatives are computed analytically
    in the phase domain, then converted to time domain.
    
    Phase residual: Δφ = φ_obs - φ_model
    Time residual: Δt = Δφ / F0 * 1e6 [μs]
    
    Therefore: ∂(Δt)/∂p = (∂Δφ/∂p) / F0 * 1e6
    
    But since we're fitting the model to observations:
        ∂Δφ/∂p = -∂φ_model/∂p
    
    So: ∂(Δt)/∂p = -(∂φ_model/∂p) / F0 * 1e6
    """
    
    # Spin parameters (affect phase directly)
    if param_name == 'F0':
        # φ = F0*t + 0.5*F1*t² + ...
        # ∂φ/∂F0 = t
        # But also: Δt = Δφ/F0, so ∂(Δt)/∂F0 = -Δφ/F0² + (∂Δφ/∂F0)/F0
        # Simplifies to: ∂(Δt)/∂F0 = -t/F0 (dominant term)
        dphase_dp = dt_sec
        return -(dphase_dp / f0 * 1e6)
    
    elif param_name == 'F1':
        # ∂φ/∂F1 = 0.5 * t²
        dphase_dp = 0.5 * dt_sec**2
        return -(dphase_dp / f0 * 1e6)
    
    elif param_name == 'F2':
        # ∂φ/∂F2 = (1/6) * t³
        dphase_dp = (1.0/6.0) * dt_sec**3
        return -(dphase_dp / f0 * 1e6)
    
    elif param_name == 'F3':
        # ∂φ/∂F3 = (1/24) * t⁴
        dphase_dp = (1.0/24.0) * dt_sec**4
        return -(dphase_dp / f0 * 1e6)
    
    # DM parameters (affect delay directly, not phase)
    elif param_name == 'DM':
        # DM delay: τ = K_DM * DM / freq²
        # ∂τ/∂DM = K_DM / freq²
        # This affects arrival time directly (not phase)
        return K_DM_SEC / (freq_mhz**2)
    
    elif param_name == 'DM1':
        # DM evolution: DM(t) = DM + DM1*t + ...
        # ∂τ/∂DM1 = (K_DM / freq²) * t / SECS_PER_DAY
        # (divided by SECS_PER_DAY because DM1 is in pc cm⁻³ day⁻¹)
        dt_days = dt_sec / SECS_PER_DAY
        return K_DM_SEC / (freq_mhz**2) * dt_days
    
    elif param_name == 'DM2':
        # ∂τ/∂DM2 = (K_DM / freq²) * 0.5 * t² / SECS_PER_DAY²
        dt_days = dt_sec / SECS_PER_DAY
        return K_DM_SEC / (freq_mhz**2) * 0.5 * dt_days**2
    
    # Binary parameters - ELL1 model
    elif param_name == 'PB':
        # Binary period affects orbital phase
        # This requires computing ∂(binary_delay)/∂PB
        # Complex - implement when needed
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'A1':
        # Projected semi-major axis
        # ∂(binary_delay)/∂A1 is complex
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'TASC':
        # Time of ascending node
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name in ['EPS1', 'EPS2']:
        # Eccentricity parameters
        return np.zeros_like(dt_sec)  # Placeholder
    
    # Binary parameters - BT/DD model
    elif param_name == 'T0':
        # Time of periastron
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'ECC':
        # Eccentricity
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'OM':
        # Longitude of periastron
        return np.zeros_like(dt_sec)  # Placeholder
    
    # Astrometric parameters
    elif param_name == 'RAJ':
        # Right ascension
        # Affects barycentric correction
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'DECJ':
        # Declination
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'PMRA':
        # Proper motion in RA
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'PMDEC':
        # Proper motion in DEC
        return np.zeros_like(dt_sec)  # Placeholder
    
    elif param_name == 'PX':
        # Parallax
        return np.zeros_like(dt_sec)  # Placeholder
    
    else:
        raise ValueError(f"Unknown parameter for design matrix: {param_name}")


def get_default_fit_params(params: Dict[str, float]) -> List[str]:
    """Get default parameters to fit based on FIT flags in .par file.
    
    Parameters
    ----------
    params : dict
        Timing model parameters (should include FIT flags)
    
    Returns
    -------
    fit_params : list of str
        Parameters with FIT=1
    """
    fit_params = []
    
    # Check for FIT flags
    for param_name in ['F0', 'F1', 'F2', 'F3', 'DM', 'DM1', 'DM2',
                       'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'T0', 'ECC', 'OM',
                       'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
        fit_flag_name = param_name + '_FIT'
        if params.get(fit_flag_name, 0) == 1:
            fit_params.append(param_name)
    
    # If no FIT flags, default to F0 and F1
    if not fit_params:
        fit_params = ['F0', 'F1']
    
    return fit_params
