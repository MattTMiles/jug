"""
Tests for plot coloring functionality.
"""

import pytest
import numpy as np
from PySide6.QtGui import QColor

from jug.gui.plot_colors import (
    get_backend_labels,
    build_backend_brush_array,
    build_noise_colors
)
from jug.gui.theme import PlotTheme, set_theme, LightTheme, SynthwaveTheme


class TestBackendLabels:
    """Test backend label extraction."""
    
    def test_basic_extraction(self):
        """Test basic backend extraction from flags."""
        toa_flags = [
            {'f': 'GUPPI', 'sys': 'AO'},
            {'f': 'ASP', 'sys': 'Parkes'},
            {'f': 'GUPPI', 'sys': 'AO'},
        ]
        labels = get_backend_labels(toa_flags)
        assert len(labels) == 3
        assert labels[0] == 'GUPPI'
        assert labels[1] == 'ASP'
        assert labels[2] == 'GUPPI'
    
    def test_fallback_candidates(self):
        """Test fallback to 'be' and 'backend' keys."""
        toa_flags = [
            {'be': 'BACKEND1'},  # No 'f', use 'be'
            {'backend': 'BACKEND2'},  # No 'f' or 'be', use 'backend'
            {'other': 'value'},  # No backend keys
        ]
        labels = get_backend_labels(toa_flags)
        assert labels[0] == 'BACKEND1'
        assert labels[1] == 'BACKEND2'
        assert labels[2] == '__unknown__'
    
    def test_returns_numpy_array(self):
        """Test that result is numpy array."""
        toa_flags = [{'f': 'TEST'}]
        labels = get_backend_labels(toa_flags)
        assert isinstance(labels, np.ndarray)
        assert labels.dtype.kind == 'U'  # Unicode string


class TestBuildBackendBrushArray:
    """Test backend brush array building."""
    
    def test_basic_brush_array(self):
        """Test basic brush array generation."""
        toa_flags = [
            {'f': 'BACKEND_A'},
            {'f': 'BACKEND_B'},
            {'f': 'BACKEND_A'},
        ]
        brushes, color_map = build_backend_brush_array(toa_flags)
        
        # Check length
        assert len(brushes) == 3
        
        # Check color map has both backends
        assert 'BACKEND_A' in color_map
        assert 'BACKEND_B' in color_map
        assert len(color_map) == 2
        
        # All brushes should be valid
        assert all(b is not None for b in brushes)
    
    def test_stable_color_assignment(self):
        """Test that same backend always gets same color."""
        toa_flags = [{'f': f'BACKEND_{i % 3}'} for i in range(10)]
        
        # Build twice
        brushes1, color_map1 = build_backend_brush_array(toa_flags)
        brushes2, color_map2 = build_backend_brush_array(toa_flags)
        
        # Colors should be identical
        for backend in color_map1:
            assert color_map1[backend].name() == color_map2[backend].name()
    
    def test_user_overrides(self):
        """Test that user overrides are applied."""
        toa_flags = [{'f': 'BACKEND_A'}, {'f': 'BACKEND_B'}]
        
        override_color = QColor(255, 0, 0, 255)  # Red
        overrides = {'BACKEND_A': override_color}
        
        brushes, color_map = build_backend_brush_array(toa_flags, overrides=overrides)
        
        # Check that override was applied
        assert color_map['BACKEND_A'].name() == override_color.name()
    
    def test_many_backends(self):
        """Test with more backends than palette colors (wrapping)."""
        toa_flags = [{'f': f'BACKEND_{i}'} for i in range(20)]
        brushes, color_map = build_backend_brush_array(toa_flags)
        
        assert len(brushes) == 20
        assert len(color_map) == 20


class TestPaletteThemeAwareness:
    """Test that palettes are theme-aware."""
    
    def test_light_dark_palettes_differ(self):
        """Test that light and dark palettes are different."""
        set_theme(LightTheme)
        light_palette = PlotTheme.get_backend_palette()
        
        set_theme(SynthwaveTheme)
        dark_palette = PlotTheme.get_backend_palette()
        
        # Palettes should differ
        assert light_palette != dark_palette
        
        # Both should have 12 colors
        assert len(light_palette) == 12
        assert len(dark_palette) == 12
    
    def test_palette_format(self):
        """Test that palette colors are RGBA tuples."""
        palette = PlotTheme.get_backend_palette()
        
        for color in palette:
            assert isinstance(color, tuple)
            assert len(color) == 4
            # All values should be 0-255
            assert all(0 <= c <= 255 for c in color)


class TestBuildNoiseColors:
    """Test noise color building."""
    
    def test_default_colors(self):
        """Test default noise colors."""
        process_names = ['RedNoise', 'DMNoise']
        color_map = build_noise_colors(process_names)
        
        assert 'RedNoise' in color_map
        assert 'DMNoise' in color_map
        assert len(color_map) == 2
    
    def test_user_overrides(self):
        """Test user color overrides."""
        process_names = ['RedNoise']
        override_color = QColor(0, 255, 0, 200)
        overrides = {'RedNoise': override_color}
        
        color_map = build_noise_colors(process_names, overrides=overrides)
        
        assert color_map['RedNoise'].name() == override_color.name()
    
    def test_unknown_process_fallback(self):
        """Test fallback for unknown noise processes."""
        process_names = ['UnknownNoise']
        color_map = build_noise_colors(process_names)
        
        # Should have a fallback color
        assert 'UnknownNoise' in color_map
        assert color_map['UnknownNoise'].isValid()


class TestSettingsPersistence:
    """Test QSettings persistence (mocked)."""
    
    def test_color_serialization(self):
        """Test that QColor can be serialized to hex."""
        color = QColor(123, 45, 67, 200)
        hex_str = color.name(QColor.HexArgb)
        
        # Deserialize
        restored = QColor(hex_str)
        
        assert restored.red() == 123
        assert restored.green() == 45
        assert restored.blue() == 67
        assert restored.alpha() == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
