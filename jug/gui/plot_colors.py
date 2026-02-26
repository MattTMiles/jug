"""
Helper functions for plot coloring -- backend identification and brush generation.

Performance-critical: Uses vectorized numpy operations, no per-TOA Python loops.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor

from jug.engine.flag_mapping import resolve_backends
from jug.gui.theme import PlotTheme


def get_backend_labels(toa_flags: List[Dict[str, str]]) -> np.ndarray:
    """Extract backend labels from TOA flags.
    
    Thin wrapper around flag_mapping.resolve_backends() that returns
    a numpy array for vectorized operations.
    
    Parameters
    ----------
    toa_flags : list of dict
        Per-TOA flag dictionaries (keys without leading dash).
    
    Returns
    -------
    ndarray of str
        Backend identifier for each TOA.
        Missing/unknown backends are labeled "__unknown__".
    """
    backends = resolve_backends(toa_flags)
    return np.array(backends, dtype=str)


def build_backend_brush_array(
    toa_flags: List[Dict[str, str]],
    palette: Optional[List[Tuple[int, int, int, int]]] = None,
    overrides: Optional[Dict[str, QColor]] = None,
    all_backends: Optional[List[str]] = None,
) -> Tuple[List, Dict[str, QColor]]:
    """Build per-TOA brush array for backend coloring (vectorized).
    
    Uses numpy unique + boolean masking for O(N) performance.
    No per-TOA Python loops.
    
    Parameters
    ----------
    toa_flags : list of dict
        Per-TOA flag dictionaries.
    palette : list of (R, G, B, A) tuples, optional
        Color palette. If None, uses PlotTheme.get_backend_palette().
    overrides : dict of backend_name -> QColor, optional
        User-specified color overrides for specific backends.
    all_backends : list of str, optional
        Full sorted list of all backend names (including deleted ones).
        Used to keep color assignments stable after TOA deletion.
        If None, derives from the current toa_flags.
    
    Returns
    -------
    brushes : list of pg.mkBrush
        Brush for each TOA (same length as toa_flags).
    backend_color_map : dict of backend_name -> QColor
        Mapping used for legend (sorted alphabetically).
    """
    if palette is None:
        palette = PlotTheme.get_backend_palette()
    
    if overrides is None:
        overrides = {}
    
    # Get backend labels (vectorized)
    backend_labels = get_backend_labels(toa_flags)
    
    # Use full backend list for stable palette indices, or derive from current data
    if all_backends is not None:
        unique_backends = all_backends
    else:
        unique_backends = sorted(set(backend_labels))
    
    # Build color map: backend -> QColor (indices stable across deletions)
    backend_color_map = {}
    for i, backend in enumerate(unique_backends):
        if backend in overrides:
            backend_color_map[backend] = overrides[backend]
        else:
            r, g, b, a = palette[i % len(palette)]
            backend_color_map[backend] = QColor(r, g, b, a)
    
    # Build brush array (vectorized with boolean masking)
    brushes = [None] * len(backend_labels)
    for backend, color in backend_color_map.items():
        # Boolean mask for this backend
        mask = backend_labels == backend
        indices = np.where(mask)[0]
        
        # Create brush once, assign to all matching TOAs
        brush = pg.mkBrush(color)
        for idx in indices:
            brushes[idx] = brush
    
    return brushes, backend_color_map


def build_noise_colors(
    process_names: List[str],
    overrides: Optional[Dict[str, QColor]] = None,
) -> Dict[str, QColor]:
    """Build color map for noise realization overlays.
    
    Parameters
    ----------
    process_names : list of str
        Noise process names (e.g., ["RedNoise", "DMNoise"]).
    overrides : dict of process_name -> QColor, optional
        User-specified color overrides.
    
    Returns
    -------
    dict of process_name -> QColor
        Color for each noise process.
    """
    if overrides is None:
        overrides = {}
    
    # Get theme-aware defaults
    default_colors = PlotTheme.get_noise_colors()
    
    # Build color map
    noise_color_map = {}
    for name in process_names:
        if name in overrides:
            noise_color_map[name] = overrides[name]
        elif name in default_colors:
            r, g, b, a = default_colors[name]
            noise_color_map[name] = QColor(r, g, b, a)
        else:
            # Fallback: gray with transparency
            noise_color_map[name] = QColor(128, 128, 128, 180)
    
    return noise_color_map
