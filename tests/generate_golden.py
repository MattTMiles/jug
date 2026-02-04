#!/usr/bin/env python3
"""
Generate golden reference values for test datasets.

Usage:
    python tests/generate_golden.py
    
This regenerates tests/data_golden/J1909_mini_golden.json
Run this whenever the mini dataset changes or JUG's residual
calculation is intentionally modified.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from jug.residuals.simple_calculator import compute_residuals_simple


def generate_mini_golden():
    """Generate golden reference for J1909_mini dataset."""
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        print(f"ERROR: Mini dataset not found at {golden_dir}")
        return False
    
    print(f"Computing residuals for {par.name}...")
    result = compute_residuals_simple(str(par), str(tim), verbose=False)
    
    golden = {
        "_comment": "Golden reference values for J1909_mini test dataset",
        "_generated": "Regenerate with: python tests/generate_golden.py",
        "_purpose": "CI validation - verify JUG produces consistent results",
        
        "n_toas": result['n_toas'],
        "weighted_rms_us": float(result['weighted_rms_us']),
        "unweighted_rms_us": float(result['unweighted_rms_us']),
        "mean_residual_us": float(np.mean(result['residuals_us'])),
        
        "first_5_residuals_ns": [
            round(float(r) * 1000, 3) for r in result['residuals_us'][:5]
        ],
        
        "tolerances": {
            "_comment": "Allowed relative difference for validation",
            "rms_rel_tol": 1e-6,
            "residual_abs_tol_ns": 0.1
        }
    }
    
    output = golden_dir / "J1909_mini_golden.json"
    with open(output, 'w') as f:
        json.dump(golden, f, indent=2)
    
    print(f"Golden reference written to {output}")
    print(f"  N_TOAs: {golden['n_toas']}")
    print(f"  wRMS: {golden['weighted_rms_us']:.6f} µs")
    print(f"  RMS: {golden['unweighted_rms_us']:.6f} µs")
    
    return True


if __name__ == "__main__":
    success = generate_mini_golden()
    sys.exit(0 if success else 1)
