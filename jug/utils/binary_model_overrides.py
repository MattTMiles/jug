"""Binary model override utilities.

Provides consistent handling of binary model overrides (e.g., DDK -> DD aliasing)
across all JUG code paths.
"""

import os
import warnings

# Canonical error message for DDK not implemented
DDK_NOT_IMPLEMENTED_ERROR = """\
DDK binary model is not implemented in JUG.

DDK requires Kopeikin (1995, 1996) annual orbital parallax terms that modify 
the projected semi-major axis (A1) and longitude of periastron (OM) based on 
orbital inclination (KIN), position angle of ascending node (KOM), parallax (PX), 
and proper motion.

Previously, JUG silently aliased DDK to DD, which is INCORRECT and would produce 
wrong science for high-parallax pulsars like J0437-4715.

Options:
  1. Convert your par file to use BINARY DD (if Kopeikin corrections are negligible)
  2. Use PINT or tempo2 for DDK pulsars until JUG implements true DDK support
  3. Set environment variable JUG_ALLOW_DDK_AS_DD=1 to force DD aliasing (NOT RECOMMENDED)
"""

# Canonical warning message for DDK override
DDK_OVERRIDE_WARNING = (
    "JUG_ALLOW_DDK_AS_DD=1: Treating DDK as DD. This is INCORRECT for "
    "high-parallax pulsars and will produce wrong science. Use at your own risk."
)

# Track whether we've already warned about DDK override (to avoid spam in loops)
_ddk_warning_issued = False


def is_ddk_override_allowed() -> bool:
    """Check if DDK->DD aliasing is allowed via environment variable.
    
    Returns True if JUG_ALLOW_DDK_AS_DD is set to '1', 'true', or 'yes'.
    """
    return os.environ.get('JUG_ALLOW_DDK_AS_DD', '').lower() in ('1', 'true', 'yes')


def resolve_binary_model(model: str, warn: bool = True) -> str:
    """Resolve binary model name, handling DDK override.
    
    Parameters
    ----------
    model : str
        Binary model name (e.g., 'DDK', 'DD', 'ELL1')
    warn : bool, optional
        Whether to issue a warning if DDK->DD aliasing is used (default: True)
        Set to False if calling in a loop to avoid duplicate warnings.
        
    Returns
    -------
    str
        Resolved model name. For DDK with override enabled, returns 'DD'.
        
    Raises
    ------
    NotImplementedError
        If model is DDK and override is not enabled.
    """
    global _ddk_warning_issued
    
    model = model.upper()
    
    if model == 'DDK':
        if is_ddk_override_allowed():
            if warn and not _ddk_warning_issued:
                warnings.warn(DDK_OVERRIDE_WARNING, UserWarning, stacklevel=3)
                _ddk_warning_issued = True
            return 'DD'
        else:
            raise NotImplementedError(DDK_NOT_IMPLEMENTED_ERROR)
    
    return model


def reset_ddk_warning():
    """Reset the DDK warning flag (useful for testing)."""
    global _ddk_warning_issued
    _ddk_warning_issued = False
