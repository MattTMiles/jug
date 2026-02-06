"""Binary model override utilities.

Provides consistent handling of binary model overrides (e.g., DDK -> DD aliasing)
across all JUG code paths.

History
-------
As of June 2025, DDK is now fully implemented in JUG with:
  - Forward model: Kopeikin 1995 parallax + K96 proper motion corrections
  - Partial derivatives: KIN/KOM chain rule derivatives through effective A1/OM/SINI
  - Full fitting support in the optimized fitter

The override mechanism remains for backward compatibility but is no longer needed
for normal DDK usage. Users who want to force DD behavior for DDK par files
can still use JUG_ALLOW_DDK_AS_DD=1.
"""

import os
import warnings

# Canonical error message for DDK aliasing (now informational since DDK is implemented)
DDK_ALIASING_INFO = """\
DDK binary model is now FULLY IMPLEMENTED in JUG.

This includes:
  - Kopeikin (1995) annual orbital parallax corrections
  - K96 (Kopeikin 1996) proper motion corrections  
  - Analytic partial derivatives for KIN and KOM parameters
  - Full fitting support in the optimized fitter

If you see this message, the code path calling resolve_binary_model() is outdated.
DDK should be used directly without aliasing.
"""

# Canonical warning message for DDK override
DDK_OVERRIDE_WARNING = (
    "JUG_ALLOW_DDK_AS_DD=1: Forcing DDK to be treated as DD. "
    "This ignores Kopeikin corrections. DDK is now fully implemented, "
    "so this override is no longer needed unless you specifically want DD behavior."
)

# Track whether we've already warned about DDK override (to avoid spam in loops)
_ddk_warning_issued = False


def is_ddk_override_allowed() -> bool:
    """Check if DDK->DD aliasing is allowed via environment variable.
    
    Returns True if JUG_ALLOW_DDK_AS_DD is set to '1', 'true', or 'yes'.
    
    Note: As of June 2025, DDK is fully implemented, so this override is
    only needed if you want to force DD behavior (ignoring Kopeikin corrections).
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
        Otherwise returns the model unchanged (DDK is now fully implemented).
    
    Notes
    -----
    As of June 2025, DDK is fully implemented in JUG, so this function
    typically just returns 'DDK' unchanged. The aliasing only occurs if
    JUG_ALLOW_DDK_AS_DD=1 is explicitly set (to force DD behavior).
    """
    global _ddk_warning_issued
    
    model = model.upper()
    
    if model == 'DDK':
        if is_ddk_override_allowed():
            if warn and not _ddk_warning_issued:
                warnings.warn(DDK_OVERRIDE_WARNING, UserWarning, stacklevel=3)
                _ddk_warning_issued = True
            return 'DD'
        # DDK is now fully implemented - return unchanged
        return 'DDK'
    
    return model


def reset_ddk_warning():
    """Reset the DDK warning flag (useful for testing)."""
    global _ddk_warning_issued
    _ddk_warning_issued = False
