"""
JUG Model Module
================

This module provides the ParameterSpec registry and component graph architecture
for scaling JUG to full PINT+tempo2 parameter coverage without cloning PINT's
Parameter system.

Key components:
- ParameterSpec: Dataclass defining parameter metadata (name, group, units, etc.)
- DerivativeGroup: Enum for routing parameters to appropriate derivative functions
- Codecs: I/O transformations (sexagesimal <-> radians, etc.)
- Components: Thin wrappers around derivative functions

Usage:
    from jug.model.parameter_spec import get_spec, canonicalize_param_name, DerivativeGroup
    from jug.model.codecs import CODECS

Design principles:
1. All angles stored internally as radians; codecs only at I/O boundary
2. ParameterSpec is immutable (frozen dataclass)
3. Components are thin wrappers - they call existing derivative functions
"""

from .parameter_spec import (
    ParameterSpec,
    DerivativeGroup,
    PARAMETER_REGISTRY,
    get_spec,
    canonicalize_param_name,
    list_params_by_group,
    list_fittable_params,
    get_derivative_group,
)

from .codecs import (
    Codec,
    FloatCodec,
    EpochMJDCodec,
    RAJCodec,
    DECJCodec,
    CODECS,
)

__all__ = [
    # Parameter specs
    'ParameterSpec',
    'DerivativeGroup',
    'PARAMETER_REGISTRY',
    'get_spec',
    'canonicalize_param_name',
    'list_params_by_group',
    'list_fittable_params',
    'get_derivative_group',
    # Codecs
    'Codec',
    'FloatCodec',
    'EpochMJDCodec',
    'RAJCodec',
    'DECJCodec',
    'CODECS',
]
