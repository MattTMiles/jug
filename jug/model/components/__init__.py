"""
Timing Model Components
=======================

Components are thin wrappers around derivative functions that provide:
1. Parameter filtering (only pass relevant params to each derivative module)
2. Consistent interface for design matrix assembly
3. Clear separation of concerns

Each component wraps an existing derivative module:
- SpinComponent -> derivatives_spin.py
- DispersionComponent -> derivatives_dm.py
- (Future) AstrometryComponent -> derivatives_astrometry.py
- (Future) BinaryComponent -> derivatives_binary.py

Usage:
    from jug.model.components import get_component, COMPONENT_REGISTRY

    # Get component by name
    spin = get_component('SpinComponent')
    derivs = spin.compute_derivatives(params, toas_mjd, fit_params)

    # Or use registry directly
    for name, component in COMPONENT_REGISTRY.items():
        print(f"{name} provides: {component.provides_params()}")

Design principles:
1. Components are THIN wrappers - they call existing derivative functions exactly as before
2. No new computation logic in components
3. Filtering happens before calling underlying functions
4. Bit-for-bit identical to direct function calls
"""

from .base import TimingComponent
from .spin import SpinComponent
from .dispersion import DispersionComponent


# Component registry
COMPONENT_REGISTRY = {
    'SpinComponent': SpinComponent(),
    'DispersionComponent': DispersionComponent(),
}


def get_component(name: str) -> TimingComponent:
    """
    Get a component by name.

    Parameters
    ----------
    name : str
        Component name (SpinComponent, DispersionComponent, etc.)

    Returns
    -------
    TimingComponent
        The component instance

    Raises
    ------
    KeyError
        If component not found
    """
    return COMPONENT_REGISTRY[name]


def list_components():
    """
    List all registered components.

    Returns
    -------
    list of str
        Component names
    """
    return list(COMPONENT_REGISTRY.keys())


__all__ = [
    'TimingComponent',
    'SpinComponent',
    'DispersionComponent',
    'COMPONENT_REGISTRY',
    'get_component',
    'list_components',
]
