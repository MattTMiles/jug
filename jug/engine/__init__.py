"""
JUG Engine - The "Brain" of JUG Timing
=======================================

This module provides the core timing engine that can be used by:
- Command-line tools (jug-compute-residuals, jug-fit)
- Qt GUI (local in-process)
- Tauri GUI (via jugd server)
- Python API (direct imports)

Key Features:
- Session-based caching (parse files once, compute fast)
- Clean separation of business logic from UI
- Backward compatible with existing code

Usage Example:
--------------
# New session-based API (fast for multiple operations)
from jug.engine import open_session

session = open_session('pulsar.par', 'pulsar.tim')
result1 = session.compute_residuals()
result2 = session.fit_parameters(['F0', 'F1'])
result3 = session.compute_residuals(params={'F0': result2['final_params']['F0']})

# Legacy API (still works, no caching)
from jug.engine import compute_residuals
result = compute_residuals('pulsar.par', 'pulsar.tim')
"""

from jug.engine.api import (
    open_session,
    compute_residuals,
    fit_parameters,
)

from jug.engine.session import TimingSession

__all__ = [
    'open_session',
    'compute_residuals',
    'fit_parameters',
    'TimingSession',
]
