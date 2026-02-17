"""
Plot Colors Customization Dialog for JUG GUI.

Allows users to customize backend colors and noise realization colors.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QPushButton, QLabel, QScrollArea, QFrame, QColorDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette

from jug.gui.theme import PlotTheme, get_border_subtle, Colors


class ColorSwatchButton(QPushButton):
    """Button that displays a color swatch and opens color picker on click."""
    
    colorChanged = Signal(QColor)
    
    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(40, 30)
        self._update_style()
        self.clicked.connect(self._pick_color)
    
    def _update_style(self):
        """Update button style to show current color."""
        r, g, b, a = self._color.red(), self._color.green(), self._color.blue(), self._color.alpha()
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba({r}, {g}, {b}, {a});
                border: 2px solid {get_border_subtle()};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid {Colors.ACCENT_PRIMARY};
            }}
        """)
    
    def _pick_color(self):
        """Open color picker dialog."""
        color = QColorDialog.getColor(
            self._color,
            self,
            "Choose Color",
            QColorDialog.ShowAlphaChannel
        )
        if color.isValid():
            self.setColor(color)
            self.colorChanged.emit(color)
    
    def setColor(self, color: QColor):
        """Set the color."""
        self._color = color
        self._update_style()
    
    def color(self) -> QColor:
        """Get the current color."""
        return self._color


class PlotColorsDialog(QDialog):
    """Dialog for customizing plot colors."""
    
    colorsChanged = Signal(dict, dict)  # backend_overrides, noise_overrides
    
    def __init__(self, toa_flags=None, noise_processes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Colors")
        self.setModal(False)  # Allow interaction with main window for live preview
        self.resize(500, 600)
        
        self.toa_flags = toa_flags
        self.noise_processes = noise_processes or []
        
        # Store current overrides
        self.backend_overrides = {}
        self.noise_overrides = {}
        
        # Store original overrides for cancel
        self.original_backend_overrides = {}
        self.original_noise_overrides = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        tabs = QTabWidget()
        
        # Backend colors tab
        backend_tab = self._create_backend_tab()
        tabs.addTab(backend_tab, "Backend Colors")
        
        # Noise colors tab
        noise_tab = self._create_noise_tab()
        tabs.addTab(noise_tab, "Noise Colors")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._on_reset)
        button_layout.addWidget(reset_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        button_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _create_backend_tab(self) -> QWidget:
        """Create backend colors tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        info = QLabel("Customize colors for each backend/receiver:")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Scroll area for backend list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        self.backend_layout = QVBoxLayout(content)
        self.backend_layout.setSpacing(8)
        
        # Populate with backends
        self.backend_swatches = {}
        if self.toa_flags:
            from jug.gui.plot_colors import get_backend_labels
            import numpy as np
            
            backends = get_backend_labels(self.toa_flags)
            unique_backends = sorted(set(backends))
            
            # Get default palette
            palette = PlotTheme.get_backend_palette()
            
            for i, backend in enumerate(unique_backends):
                row = QHBoxLayout()
                
                # Color swatch
                r, g, b, a = palette[i % len(palette)]
                default_color = QColor(r, g, b, a)
                swatch = ColorSwatchButton(default_color)
                swatch.colorChanged.connect(lambda c, be=backend: self._on_backend_color_changed(be, c))
                self.backend_swatches[backend] = swatch
                row.addWidget(swatch)
                
                # Backend name
                label = QLabel(backend)
                label.setMinimumWidth(200)
                row.addWidget(label, 1)
                
                self.backend_layout.addLayout(row)
        else:
            self.backend_layout.addWidget(QLabel("No data loaded"))
        
        self.backend_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return widget
    
    def _create_noise_tab(self) -> QWidget:
        """Create noise colors tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        info = QLabel("Customize colors for noise realization overlays:")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Scroll area for noise processes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content = QWidget()
        self.noise_layout = QVBoxLayout(content)
        self.noise_layout.setSpacing(8)
        
        # Populate with noise processes
        self.noise_swatches = {}
        noise_colors = PlotTheme.get_noise_colors()
        
        # Include common noise processes even if not in current dataset
        all_processes = set(self.noise_processes) | {"RedNoise", "DMNoise", "ECORR"}
        
        for process in sorted(all_processes):
            row = QHBoxLayout()
            
            # Color swatch
            if process in noise_colors:
                r, g, b, a = noise_colors[process]
                default_color = QColor(r, g, b, a)
            else:
                default_color = QColor(128, 128, 128, 180)
            
            swatch = ColorSwatchButton(default_color)
            swatch.colorChanged.connect(lambda c, p=process: self._on_noise_color_changed(p, c))
            self.noise_swatches[process] = swatch
            row.addWidget(swatch)
            
            # Process name
            label = QLabel(process)
            label.setMinimumWidth(200)
            row.addWidget(label, 1)
            
            self.noise_layout.addLayout(row)
        
        self.noise_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return widget
    
    def _on_backend_color_changed(self, backend: str, color: QColor):
        """Handle backend color change."""
        self.backend_overrides[backend] = color
        # Emit signal for live preview
        self.colorsChanged.emit(self.backend_overrides, self.noise_overrides)
    
    def _on_noise_color_changed(self, process: str, color: QColor):
        """Handle noise color change."""
        self.noise_overrides[process] = color
        # Emit signal for live preview
        self.colorsChanged.emit(self.backend_overrides, self.noise_overrides)
    
    def _on_reset(self):
        """Reset all colors to defaults."""
        # Reset backend colors
        if self.toa_flags:
            from jug.gui.plot_colors import get_backend_labels
            backends = get_backend_labels(self.toa_flags)
            unique_backends = sorted(set(backends))
            palette = PlotTheme.get_backend_palette()
            
            for i, backend in enumerate(unique_backends):
                r, g, b, a = palette[i % len(palette)]
                default_color = QColor(r, g, b, a)
                self.backend_swatches[backend].setColor(default_color)
        
        # Reset noise colors
        noise_colors = PlotTheme.get_noise_colors()
        for process, swatch in self.noise_swatches.items():
            if process in noise_colors:
                r, g, b, a = noise_colors[process]
                default_color = QColor(r, g, b, a)
            else:
                default_color = QColor(128, 128, 128, 180)
            swatch.setColor(default_color)
        
        # Clear overrides
        self.backend_overrides = {}
        self.noise_overrides = {}
        
        # Emit signal for live preview
        self.colorsChanged.emit(self.backend_overrides, self.noise_overrides)
    
    def _on_apply(self):
        """Apply changes and close."""
        # Store as original for next open
        self.original_backend_overrides = self.backend_overrides.copy()
        self.original_noise_overrides = self.noise_overrides.copy()
        self.accept()
    
    def _on_cancel(self):
        """Cancel changes and revert."""
        # Revert to original overrides
        self.backend_overrides = self.original_backend_overrides.copy()
        self.noise_overrides = self.original_noise_overrides.copy()
        
        # Emit signal to revert in main window
        self.colorsChanged.emit(self.backend_overrides, self.noise_overrides)
        self.reject()
    
    def setOverrides(self, backend_overrides: dict, noise_overrides: dict):
        """Set current overrides (called when opening dialog)."""
        self.backend_overrides = backend_overrides.copy()
        self.noise_overrides = noise_overrides.copy()
        self.original_backend_overrides = backend_overrides.copy()
        self.original_noise_overrides = noise_overrides.copy()
        
        # Update swatches
        for backend, color in backend_overrides.items():
            if backend in self.backend_swatches:
                self.backend_swatches[backend].setColor(color)
        
        for process, color in noise_overrides.items():
            if process in self.noise_swatches:
                self.noise_swatches[process].setColor(color)
