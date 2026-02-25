"""Deterministic signal models for pulsar timing.

This package provides a framework for deterministic astrophysical signals
(continuous gravitational waves, bursts with memory, chromatic transients)
that can be subtracted from timing residuals before fitting.

Each signal type implements the :class:`DeterministicSignal` base class and
is registered in :data:`SIGNAL_REGISTRY` so it is automatically detected
from par file parameters and displayed in the GUI noise panel.
"""

from jug.signals.base import (
    DeterministicSignal,
    SIGNAL_REGISTRY,
    detect_signals,
)

# Import signal modules to trigger @register_signal decorators
from jug.signals import continuous_wave  # noqa: F401
from jug.signals import burst_memory     # noqa: F401
from jug.signals import chromatic_event  # noqa: F401

__all__ = [
    "DeterministicSignal",
    "SIGNAL_REGISTRY",
    "detect_signals",
]
