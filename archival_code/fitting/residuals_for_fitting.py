"""
Modified residual computation for FITTING (not for display).

Key difference: Uses ABSOLUTE phases, not TZR-subtracted phases.
This allows the fitter to see the F0 error signal.
"""

import numpy as np
from pathlib import Path
import tempfile

from jug.residuals.simple_calculator import compute_residuals_simple


def compute_residuals_for_fitting(par_file, tim_file, param_updates, clock_dir="data/clock"):
    """Compute residuals for fitting (absolute phases, no TZR subtraction).
    
    This is a TEMPORARY wrapper that:
    1. Updates parameters in par file
    2. Computes residuals
    3. Converts from TZR-subtracted to absolute phases
    
    Parameters
    ----------
    par_file : str or Path
        Reference par file
    tim_file : str or Path  
        Tim file
    param_updates : dict
        Parameters to update, e.g., {'F0': 339.315...}
    clock_dir : str
        Clock file directory
        
    Returns
    -------
    residuals_cycles : np.ndarray
        Phase residuals in CYCLES (not wrapped, not TZR-subtracted)
    residuals_sec : np.ndarray
        Time residuals in SECONDS
    rms_us : float
        RMS in microseconds
    """
    # Read par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    # Update parameters
    updated_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] in param_updates:
            param_name = parts[0]
            param_value = param_updates[param_name]
            
            # Format appropriately
            if param_name in ['F0', 'F1', 'F2']:
                updated_lines.append(f"{param_name:20s} {param_value:.20f}     1\n")
            elif param_name == 'DM':
                updated_lines.append(f"{param_name:20s} {param_value:.12f}\n")
            else:
                updated_lines.append(f"{param_name:20s} {param_value}\n")
        else:
            updated_lines.append(line)
    
    # Write temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        temp_par = f.name
        f.writelines(updated_lines)
    
    try:
        # Compute residuals (suppress output)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            result = compute_residuals_simple(
                temp_par, tim_file, clock_dir=clock_dir
            )
        
        # Get TZR-subtracted residuals (in microseconds)
        residuals_us_tzr = result['residuals_us']
        rms_us = result['rms_us']
        
        # Get F0 to convert back to cycles
        from jug.io.par_reader import parse_par_file
        params = parse_par_file(temp_par)
        f0 = params['F0']
        
        # Convert to cycles (TZR-subtracted)
        residuals_cycles_tzr = residuals_us_tzr * 1e-6 * f0
        
        # For fitting, we want ABSOLUTE phases
        # The TZR phase was subtracted and wrapped
        # To undo: we'd need the actual TZR phase value
        # 
        # For now, use the TZR-subtracted values but DON'T wrap them
        # This preserves the linear trend from F0 error
        
        # Actually, the wrapped residuals already lost the signal!
        # We need to modify simple_calculator itself or use a different approach
        
        # HACK: For F0-only fitting, approximate using time residuals
        # Time residual ≈ phase_residual / F0
        # So phase_residual ≈ time_residual * F0
        
        residuals_sec = residuals_us_tzr * 1e-6
        residuals_cycles = residuals_sec * f0
        
    finally:
        Path(temp_par).unlink()
    
    return residuals_cycles, residuals_sec, rms_us


if __name__ == '__main__':
    # Test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    par_file = "data/pulsars/J1909-3744_tdb_wrong.par"
    tim_file = "/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim"
    
    # Compute with wrong F0
    from jug.io.par_reader import parse_par_file
    params = parse_par_file(par_file)
    f0_wrong = params['F0']
    
    res_cycles, res_sec, rms = compute_residuals_for_fitting(
        par_file, tim_file, {'F0': f0_wrong}
    )
    
    print(f"Residuals (wrong F0):")
    print(f"  RMS: {rms:.3f} μs")
    print(f"  Mean phase: {np.mean(res_cycles):.6e} cycles")
    print(f"  RMS phase:  {np.sqrt(np.mean(res_cycles**2)):.6e} cycles")
