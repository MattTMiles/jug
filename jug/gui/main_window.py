"""
JUG Main Window - tempo2 plk-style interactive timing GUI.

Modern redesign inspired by Linear, Raycast, and Notion.
"""
from __future__ import annotations
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QLabel, QSplitter, QProgressBar,
    QMessageBox, QMenu, QFileDialog, QComboBox, QLineEdit,
    QDialog, QTextBrowser, QAbstractItemView, QGroupBox,
    QProgressDialog, QFrame, QScrollArea, QSizePolicy, QApplication, QPushButton, QStatusBar
)
from PySide6.QtCore import Qt, QThreadPool, QEvent, QPointF, QTimer, QSize, QRect, QVariantAnimation, QEasingCurve, Slot, Signal
from PySide6.QtGui import QFont, QCursor, QPainter, QBrush, QColor, QAction, QIcon, QKeySequence, QActionGroup

# Import custom widgets
from jug.gui.widgets.colorbar import SimpleColorBar
import pyqtgraph as pg
import numpy as np

from jug.gui.theme import (
    get_main_stylesheet,
    get_placeholder_style,
    get_added_param_style,
    get_section_title_style,
    get_control_panel_style,
    get_primary_button_style,
    get_secondary_button_style,
    get_border_subtle,
    get_border_strong,
    get_rms_emphasis_color,
    configure_plot_widget,
    create_scatter_item,
    create_error_bar_item,
    create_zero_line,
    get_scatter_colors,
    Colors,
    Typography,
    LightTheme,
    SynthwaveTheme,
    set_theme,
    is_dark_mode,
    PlotTheme,
    get_synthwave_variant,
    toggle_synthwave_variant,
    get_dynamic_accent_primary,
    get_light_variant,
    toggle_light_variant,
    get_plot_title_style,
)

# NOTE: jug.engine.stats import is deferred to call sites to avoid
# triggering jug.engine.__init__ → JAX import chain at GUI startup.



# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class ToggleSwitch(QCheckBox):
    """Custom paint-based sliding toggle switch."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._handle_pos = 0.0
        
        # Animation
        self._anim = QVariantAnimation()
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_anim_update)
        self.stateChanged.connect(self._start_anim)
        
    def _start_anim(self):
        start = self._handle_pos
        end = 1.0 if self.isChecked() else 0.0
        self._anim.stop()
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()
        
    def _on_anim_update(self, val):
        self._handle_pos = val
        self.update()
    
    def sizeHint(self):
        return QSize(44, 24)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Use simple color lookup
        if self.isChecked():
            bg_col = QColor(Colors.ACCENT_PRIMARY)
        else:
            bg_col = QColor(Colors.SURFACE_BORDER)
            
        # Draw Track
        rect = self.rect()
        radius = rect.height() / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_col))
        painter.drawRoundedRect(rect, radius, radius)
        
        # Draw Handle
        handle_dia = self.height() - 6
        # Calculate X based on handle_pos (0.0 to 1.0)
        min_x = 3
        max_x = self.width() - 3 - handle_dia
        x = min_x + (max_x - min_x) * self._handle_pos
        
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(int(x), 3, int(handle_dia), int(handle_dia))
        
        painter.end()


class OverlayMenu(QFrame):
    """
    A custom dropdown menu that lives as an overlay widget within the main window.
    This avoids creating new top-level windows (like QMenu/QComboBox do), 
    eliminating X11 forwarding lag/white-box artifacts in remote sessions.
    """
    def __init__(self, parent, options, callback, width=160):
        super().__init__(parent)
        from jug.gui.theme import Colors, get_border_subtle
        self.callback = callback
        self.setFixedWidth(width)
        
        # Style matches the theme's "Surface" look
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.ACCENT_PRIMARY};
                border-radius: 4px;
            }}
            QPushButton {{
                text-align: left;
                padding: 6px 12px;
                border: none;
                background: transparent;
                color: {Colors.TEXT_PRIMARY};
                border-radius: 0px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE_HOVER};
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        # Force frame to resize to fit content (buttons)
        layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        
        for text in options:
            btn = QPushButton(text)
            # Use closure to capture text
            btn.clicked.connect(lambda checked=False, val=text: self._on_select(val))
            layout.addWidget(btn)
            
        # Drop shadow removed for X11 performance (causes significant lag)
        # self.setGraphicsEffect(None)
        
        self.hide()
        
    def _on_select(self, val):
        self.callback(val)
        self.hide()
        
    def toggle_at(self, pos):
        """Toggle visibility at the given local position (relative to parent)."""
        if self.isVisible():
            self.hide()
        else:
            self.move(pos)
            self.raise_()
            self.show()

class MenuBarOverlayMenu(QFrame):
    """
    Enhanced overlay menu for the remote menu bar.

    Supports separators, checkable items, disabled items,
    shortcut text display, and inline submenu expansion.
    Renders as a child QFrame (no top-level X11 windows).
    """
    closed = Signal()

    def __init__(self, parent_window, width=220):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.setFixedWidth(width)
        self._items = {}  # name -> item_data dict
        self._sub_items = {}  # group_name -> list of (name, btn) tuples
        self._apply_style()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 4, 2, 4)
        self._layout.setSpacing(0)
        self._layout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.hide()

    def _apply_style(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.ACCENT_PRIMARY};
                border-radius: 4px;
            }}
        """)

    def _item_style(self):
        return f"""
            QPushButton {{
                text-align: left;
                padding: 5px 8px;
                border: none;
                background: transparent;
                color: {Colors.TEXT_PRIMARY};
                font-family: {Typography.FONT_MONO};
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.SURFACE_HOVER};
            }}
            QPushButton:disabled {{
                color: {Colors.TEXT_MUTED};
            }}
        """

    def _sub_item_style(self):
        """Style for submenu child items — tinted background to show nesting."""
        return f"""
            QPushButton {{
                text-align: left;
                padding: 5px 8px 5px 16px;
                border: none;
                background-color: {Colors.SURFACE_HOVER};
                color: {Colors.TEXT_PRIMARY};
                font-family: {Typography.FONT_MONO};
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_PRIMARY};
                color: #FFFFFF;
            }}
        """

    def _format_text(self, text, shortcut="", checkable=False, checked=False):
        prefix = " \u2713 " if (checkable and checked) else "   "
        if shortcut:
            pad = max(1, 22 - len(text))
            return f"{prefix}{text}{' ' * pad}{shortcut}"
        return f"{prefix}{text}"

    def add_action(self, text, callback, shortcut="", checkable=False,
                   checked=False, enabled=True, name=None):
        item_name = name or text
        btn = QPushButton()
        btn.setEnabled(enabled)
        btn.setText(self._format_text(text, shortcut, checkable, checked))
        btn.setStyleSheet(self._item_style())

        item_data = {
            'text': text, 'callback': callback, 'shortcut': shortcut,
            'checkable': checkable, 'checked': checked, 'btn': btn,
        }

        def on_click():
            if item_data['checkable']:
                item_data['checked'] = not item_data['checked']
                btn.setText(self._format_text(
                    item_data['text'], shortcut, True, item_data['checked']))
                callback(item_data['checked'])
            else:
                callback()
            self.close_menu()

        btn.clicked.connect(on_click)
        self._layout.addWidget(btn)
        self._items[item_name] = item_data
        return item_data

    def add_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {Colors.SURFACE_BORDER}; border: none; margin: 4px 8px;")
        self._layout.addWidget(sep)

    def add_submenu_group(self, label_text, items, group_name):
        """Add an inline expandable submenu group with mutual exclusivity.

        items: list of (text, callback, checked_initially) tuples
        """
        # Header button
        header_btn = QPushButton(f"   {label_text}  \u25b8")
        header_btn.setStyleSheet(self._item_style())
        self._layout.addWidget(header_btn)

        sub_buttons = []
        for sub_text, sub_callback, sub_checked in items:
            btn = QPushButton()
            btn.setText(self._format_text(f"  {sub_text}", "", True, sub_checked))
            btn.setStyleSheet(self._sub_item_style())
            btn.setVisible(False)
            sub_name = f"{group_name}_{sub_text}"
            item_data = {
                'text': sub_text, 'callback': sub_callback,
                'checkable': True, 'checked': sub_checked, 'btn': btn,
                'shortcut': '',
            }
            self._items[sub_name] = item_data
            sub_buttons.append((sub_name, btn, item_data))
            self._layout.addWidget(btn)

        self._sub_items[group_name] = sub_buttons

        def toggle_sub():
            visible = not sub_buttons[0][1].isVisible()
            arrow = "\u25be" if visible else "\u25b8"
            header_btn.setText(f"   {label_text}  {arrow}")
            for _, b, _ in sub_buttons:
                b.setVisible(visible)

        header_btn.clicked.connect(toggle_sub)

        # Wire mutual exclusivity and close-on-select
        for sn, sb, sd in sub_buttons:
            def make_click(name, data):
                def on_click():
                    # Uncheck all in group, check this one
                    for other_name, other_btn, other_data in sub_buttons:
                        other_data['checked'] = (other_name == name)
                        other_btn.setText(self._format_text(
                            f"  {other_data['text']}", "", True, other_data['checked']))
                    data['callback']()
                    self.close_menu()
                return on_click
            sb.clicked.connect(make_click(sn, sd))

    def set_item_text(self, name, new_text):
        item = self._items.get(name)
        if item:
            item['text'] = new_text
            item['btn'].setText(self._format_text(
                new_text, item['shortcut'], item['checkable'], item['checked']))

    def set_item_enabled(self, name, enabled):
        item = self._items.get(name)
        if item:
            item['btn'].setEnabled(enabled)

    def set_item_checked(self, name, checked):
        item = self._items.get(name)
        if item:
            item['checked'] = checked
            item['btn'].setText(self._format_text(
                item['text'], item['shortcut'], True, checked))

    def show_at(self, pos):
        # Clamp to parent bounds
        pw = self.parent_window.width()
        ph = self.parent_window.height()
        x = min(pos.x(), pw - self.width() - 4)
        y = min(pos.y(), ph - self.sizeHint().height() - 4)
        self.move(int(x), int(y))
        self.raise_()
        self.show()

    def close_menu(self):
        self.hide()
        self.closed.emit()

    def update_theme(self):
        self._apply_style()
        style = self._item_style()
        for item in self._items.values():
            item['btn'].setStyleSheet(style)


class MainWindow(QMainWindow):
    """Main window for JUG timing analysis GUI."""

    def __init__(self, fit_params=None):
        super().__init__()
        self.setWindowTitle("JUG Timing Analysis")
        self.setGeometry(100, 100, 1152, 768)
        self.setMinimumSize(1152, 768)
        self.setMaximumSize(2304, 1536)

        # Detect remote environment
        import os
        self.is_remote = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ or os.environ.get('JUG_REMOTE_UI', '').lower() in ('1', 'true', 'yes')

        # Apply modern theme stylesheet
        self.setStyleSheet(get_main_stylesheet())

        # Enforce dark palette for X11 background clearing (prevents white flash)
        from PySide6.QtGui import QPalette, QColor
        from jug.gui.theme import Colors
        palette = self.palette()
        dark_bg = QColor(Colors.BG_PRIMARY)
        palette.setColor(QPalette.Window, dark_bg)
        palette.setColor(QPalette.Base, dark_bg)
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        # Timing session (engine with caching)
        self.session = None
        
        # Data storage
        self.par_file = None
        self.tim_file = None
        self.residuals_us = None
        self.mjd = None
        self.errors_us = None
        self.rms_us = None
        
        # RMS display mode (weighted vs unweighted)
        self.use_weighted_rms = True  # Default to weighted RMS
        self.weighted_rms_us = None
        self.unweighted_rms_us = None
        self.prefit_weighted_rms_us = None
        self.prefit_unweighted_rms_us = None
        
        # Original data (for full restart, before any deletions)
        self.original_mjd = None
        self.original_residuals_us = None
        self.original_errors_us = None
        
        # Deleted TOA indices (track which original TOAs have been deleted)
        self.deleted_indices = set()

        # Fit state
        self.prefit_residuals_us = None
        self.postfit_residuals_us = None
        self.fit_results = None
        self.is_fitted = False
        self.pulsar_name = None

        # Command-line fit parameters
        self.cmdline_fit_params = fit_params or []

        # Available parameters (will be populated from par file)
        self.available_params = []
        
        # Initial parameter values from .par file
        self.initial_params = {}
        
        # Plot items (reused for performance)
        self.scatter_item = None
        self.error_bar_item = None
        self.zero_line = None
        self.show_zero_line = False  # Zero line hidden by default
        self.show_residuals = True   # Residuals visible by default
        self.show_uncertainties = True  # Error bars visible by default

        # Noise realization overlay items {process_name: ScatterPlotItem/ErrorBarItem}
        self._noise_curves: dict = {}
        self._noise_errorbars: dict = {}
        self._noise_realizations: dict = {}

        # Box zoom state
        self.box_zoom_active = False
        self.box_zoom_start = None  # (x, y) in data coordinates
        self.box_zoom_rect = None   # LinearRegionItem or ROI for visual feedback
        
        # Box delete state
        self.box_delete_active = False
        self.box_delete_start = None
        self.box_delete_rect = None

        # Error bar beam update throttling (for smooth pan/zoom over X11)
        self._beam_update_timer = None
        self._pending_beam = None
        
        # Interaction mode tracking (reduces work during continuous pan/zoom)
        self._interaction_active = False
        self._interaction_end_timer = None

        # Cached boolean mask for deleted TOAs (avoids O(N) Python loop)
        self._keep_mask = None  # Will be initialized when data loads

        # Solver mode for fitting (exact or fast)
        self.solver_mode = "exact"  # "exact" or "fast"

        # Noise control panel (populated after session loads)
        self.noise_panel = None

        # Color mode for scatter plot (none/backend/frequency)
        self._color_mode = "none"

        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # Session + compute/fit

        # Setup UI
        self._setup_ui()
        if self.is_remote:
            self._create_remote_menu_bar()
            self._register_shortcuts()
        else:
            self._create_menu_bar()
        self._create_status_bar()

        # Install event filter for key handling
        self.installEventFilter(self)

    def resizeEvent(self, event):
        """Handle window resize events to maintain overlay position."""
        super().resizeEvent(event)
        
        # Update overlay drawer position in remote mode
        if getattr(self, 'is_remote', False) and getattr(self, 'params_drawer_open', False):
            from PySide6.QtCore import QPoint
            
            # Keep pinned to drop-down position below button
            panel_width = 240 # Matches control panel
            w = self.centralWidget().width()
            
            # Map button position
            btn_bottom_global = self.params_drawer_btn.mapTo(self.centralWidget(), QPoint(0, self.params_drawer_btn.height()))
            y_pos = btn_bottom_global.y()
            x_pos = w - panel_width
            h_available = self.centralWidget().height() - y_pos
            
            self.params_drawer.setGeometry(x_pos, y_pos, panel_width, h_available)
    
    def _setup_ui(self):
        """Setup the main user interface with modern styling."""
        central_widget = QWidget()
        from jug.gui.theme import Colors
        central_widget.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")
        self.setCentralWidget(central_widget)

        # Main layout: Plot on left, controls on right
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left side: Plot area with title
        self.plot_container = QWidget()
        self.plot_container.setObjectName("plotContainer")
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(24, 16, 16, 16)
        plot_layout.setSpacing(12)

        # Combined title + axis selector bar (single line)
        title_axis_bar = QHBoxLayout()
        title_axis_bar.setSpacing(12)
        title_axis_bar.setContentsMargins(0, 0, 0, 0)

        # "Noise" button — shown when noise panel is collapsed, opens it back
        self._noise_title_btn = QPushButton("◀  Noise")
        self._noise_title_btn.setFixedWidth(100)
        self._noise_title_btn.setCursor(Qt.PointingHandCursor)
        self._noise_title_btn.setToolTip("Show noise panel")
        self._noise_title_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border: 1px solid {Colors.SURFACE_BORDER}; border-radius: 4px; "
            f"padding: 2px 8px; font-size: {Typography.SIZE_BASE}; font-weight: {Typography.WEIGHT_BOLD}; }}"
            f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; "
            f"border-color: {Colors.TEXT_MUTED}; }}"
        )
        self._noise_title_btn.setVisible(False)
        self._noise_title_btn.clicked.connect(lambda: self._set_noise_panel_visible(True))
        title_axis_bar.addWidget(self._noise_title_btn, 0, Qt.AlignVCenter)

        # Title label (pulsar name and RMS) - centered, stretches to fill space
        self.plot_title_label = QLabel("")
        self.plot_title_label.setObjectName("plotTitleLabel")
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        title_axis_bar.addWidget(self.plot_title_label, 1)  # stretch factor 1

        # Y-axis selector
        y_label = QLabel("Y:")
        y_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px; border: none;")
        title_axis_bar.addWidget(y_label)

        self.y_axis_btn = self._create_overlay_axis_button("Pre-fit (\u03bcs)", [
            "Pre-fit (\u03bcs)", "Post-fit (\u03bcs)",
            "Normalized",
        ], lambda val: self._on_axis_changed(y_val=val), width=140)
        self._y_axis_mode = "Pre-fit (\u03bcs)"
        title_axis_bar.addWidget(self.y_axis_btn)

        title_axis_bar.addSpacing(8)

        # X-axis selector
        x_label = QLabel("X:")
        x_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 12px; border: none;")
        title_axis_bar.addWidget(x_label)

        self.x_axis_btn = self._create_overlay_axis_button("MJD", [
            "MJD", "Serial", "Frequency (MHz)",
            "ToA Error (\u03bcs)", "Orbital Phase",
            "Day of Year", "Year"
        ], lambda val: self._on_axis_changed(x_val=val), width=160)
        self._x_axis_mode = "MJD"
        title_axis_bar.addWidget(self.x_axis_btn)

        plot_layout.addLayout(title_axis_bar)

        # Large residual plot with modern styling
        self.plot_widget = pg.PlotWidget()
        configure_plot_widget(self.plot_widget)
        
        # Connect to view range changes to scale error bar caps
        self.plot_widget.getPlotItem().getViewBox().sigRangeChanged.connect(self._on_view_range_changed)

        # Optimize QGraphicsView for smoother pan/zoom (especially over X11/SSH)
        # PlotWidget inherits from QGraphicsView, so we can call these directly
        from PySide6.QtWidgets import QGraphicsView
        self.plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.plot_widget.setCacheMode(QGraphicsView.CacheBackground)

        # Add subtle rounded corners effect via container
        self.plot_frame = QFrame()
        self.plot_frame.setObjectName("plotFrame")
        plot_frame_layout = QVBoxLayout(self.plot_frame)
        plot_frame_layout.setContentsMargins(8, 8, 8, 8)
        plot_frame_layout.addWidget(self.plot_widget)

        plot_layout.addWidget(self.plot_frame)

        # Right side: Control panel
        control_panel = self._create_control_panel()

        # Add to main layout: noise panel (if any) + plot (80%) + controls (20%)
        # Noise panel placeholder — will be populated in on_session_ready
        from jug.gui.widgets.noise_control_panel import NoiseControlPanel
        self.noise_panel = NoiseControlPanel()
        self.noise_panel.setVisible(False)  # Hidden until session loads
        self.noise_panel.collapse_requested.connect(lambda: self._set_noise_panel_visible(False))
        self.noise_panel.realise_changed.connect(self._on_noise_realise_changed)
        self.noise_panel.subtract_changed.connect(self._on_noise_subtract_changed)
        self.noise_panel.estimate_noise_requested.connect(self._on_estimate_noise)
        self.noise_panel.show_residuals_changed.connect(self._on_toggle_residuals)
        self.noise_panel.show_uncertainties_changed.connect(self._on_toggle_uncertainties)
        main_layout.addWidget(self.noise_panel)
        main_layout.addWidget(self.plot_container, stretch=4)
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self):
        """Create the control panel widget with modern card-based layout."""
        # Main container that holds both control panel and parameter drawer
        self.control_container = QWidget()
        from jug.gui.theme import Colors
        self.control_container.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")
        container_layout = QHBoxLayout(self.control_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Import AnimatedButton
        from jug.gui.theme import AnimatedButton

        # Main control panel
        panel = QWidget()
        panel.setStyleSheet(get_control_panel_style())
        panel.setFixedWidth(240)  # Fixed width to prevent resizing glitches
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        # Primary action: Run Fit (special styling) - Animated (primary)
        self.fit_button = AnimatedButton("▶  Run Fit", role="primary")
        self.fit_button.clicked.connect(self.on_fit_clicked)
        self.fit_button.setEnabled(False)
        layout.addWidget(self.fit_button)

        layout.addSpacing(8)

        # Parameters button (opens drawer) - Animated (secondary)
        self.params_drawer_btn = AnimatedButton("Parameters to Fit  ▾", role="secondary")
        self.params_drawer_btn.clicked.connect(self.on_toggle_params_drawer)
        layout.addWidget(self.params_drawer_btn)

        # Create the parameter drawer (hidden by default)
        self._create_params_drawer()
        self.param_checkboxes = {}

        # Solver mode dropdown (using QPushButton + QMenu for proper styling)
        solver_layout = QHBoxLayout()
        solver_layout.setContentsMargins(0, 0, 0, 0)
        solver_layout.setSpacing(8)
        self.solver_label = QLabel("Solver:")
        self.solver_label.setObjectName("solverLabel")
        solver_layout.addWidget(self.solver_label)

        # Solver button with dropdown menu
        self.solver_button = QPushButton("  Exact ▾")
        self.solver_button.setObjectName("solverButton")
        self.solver_button.setToolTip(
            "Exact: SVD-based, bit-for-bit reproducible\n"
            "Fast: QR-based, faster but may differ slightly"
        )

        if self.is_remote:
            # Use overlay menu to avoid QMenu top-level window on X11
            self._solver_overlay = OverlayMenu(
                self, ["Exact (reproducible)", "Fast"],
                lambda val: self._set_solver_mode(
                    "exact" if "Exact" in val else "fast",
                    "Exact" if "Exact" in val else "Fast"),
                width=180)
            self.solver_button.clicked.connect(
                lambda: self._solver_overlay.toggle_at(
                    self.solver_button.mapTo(
                        self.centralWidget(),
                        QPointF(0, self.solver_button.height()).toPoint())))
        else:
            # Standard QMenu for local desktop
            self.solver_menu = QMenu(self)
            self.solver_menu.setObjectName("solverMenu")
            exact_action = self.solver_menu.addAction("Exact (reproducible)")
            exact_action.setData("exact")
            exact_action.triggered.connect(lambda: self._set_solver_mode("exact", "Exact"))
            fast_action = self.solver_menu.addAction("Fast")
            fast_action.setData("fast")
            fast_action.triggered.connect(lambda: self._set_solver_mode("fast", "Fast"))
            self.solver_button.setMenu(self.solver_menu)
        self.solver_button.setCursor(Qt.PointingHandCursor)
        self.solver_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        solver_layout.addWidget(self.solver_button)
        layout.addLayout(solver_layout)

        # Secondary buttons with consistent styling
        self.fit_report_button = AnimatedButton("📊  Fit Report", role="secondary")
        self.fit_report_button.clicked.connect(self.on_show_fit_report)
        self.fit_report_button.setEnabled(False)
        layout.addWidget(self.fit_report_button)

        self.reset_button = AnimatedButton("↺  Restart", role="secondary")
        self.reset_button.clicked.connect(self.on_restart_clicked)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)

        self.fit_window_button = AnimatedButton("⤢  Fit Window to Data", role="secondary")
        self.fit_window_button.clicked.connect(self.on_zoom_fit)
        self.fit_window_button.setEnabled(False)
        layout.addWidget(self.fit_window_button)

        layout.addSpacing(8)

        # Statistics card
        self.stats_card = QWidget()
        self.stats_card.setObjectName("statsCard")
        self.stats_card.setMinimumHeight(160)
        stats_layout = QVBoxLayout(self.stats_card)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(16, 16, 16, 16)

        self.stats_title = QLabel("Statistics")
        self.stats_title.setObjectName("statsTitle")
        stats_layout.addWidget(self.stats_title)

        # Create stat rows with label + value (store both for theme updates)
        self.rms_label_text, self.rms_label = self._create_stat_row(stats_layout, "wRMS", "--")
        # Set objectName for RMS value (special emphasis styling)
        self.rms_label.setObjectName("rmsValue")
        self.iter_label_text, self.iter_label = self._create_stat_row(stats_layout, "Iterations", "--")
        self.ntoa_label_text, self.ntoa_label = self._create_stat_row(stats_layout, "TOAs", "--")
        self.chi2_label_text, self.chi2_label = self._create_stat_row(stats_layout, "χ²/dof", "--")

        layout.addWidget(self.stats_card)

        # Stretch to push everything to top
        layout.addStretch()

        # Add panel to container
        container_layout.addWidget(panel)
        
        if not self.is_remote:
            container_layout.addWidget(self.params_drawer)
        else:
            # OVERLAY MODE: Drawer floats over content (Remote Only)
            self.params_drawer.setParent(self.centralWidget())
            self.params_drawer.raise_()

        return self.control_container

    def _create_params_drawer(self):
        """Create the slide-out parameter drawer."""
        self.params_drawer = QFrame()
        self.params_drawer.setObjectName("paramsDrawer")
        self.params_drawer.setFixedWidth(0) # Hidden by width (pre-rendered)
        # Explicit background to prevent white flash on X11
        # Use a safe dark default if Colors isn't updated yet, but Colors.BG_SECONDARY should work
        self.params_drawer.setStyleSheet(f"background-color: {Colors.BG_SECONDARY}; border-left: 1px solid {get_border_subtle()};")
        self.params_drawer.setVisible(True)  # Always visible to prevent flash
        self.params_drawer_open = False

        drawer_layout = QVBoxLayout(self.params_drawer)
        drawer_layout.setContentsMargins(12, 8, 12, 12)
        drawer_layout.setSpacing(8)

        # Header with close button
        header_layout = QHBoxLayout()
        # "Lights On" Toggle Switch Cluster
        
        switch_container = QWidget()
        switch_container.setStyleSheet("background: transparent; border: none;")
        switch_layout = QHBoxLayout(switch_container)
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_layout.setSpacing(6)
        
        # Styles
        text_style = f"border: none; color: {Colors.TEXT_PRIMARY}; font-weight: {Typography.WEIGHT_BOLD}; font-size: 13px;"
        muted_style = f"border: none; color: {Colors.TEXT_MUTED}; font-weight: normal; font-size: 13px;"
        
        # Title
        self.lbl_lights = QLabel("Lights On:")
        self.lbl_lights.setStyleSheet(text_style)
        
        # Off/On Labels
        self.lbl_off = QLabel("Off")
        self.lbl_off.setStyleSheet(text_style) 
        
        self.lights_on_switch = ToggleSwitch()
        
        self.lbl_on = QLabel("On")
        self.lbl_on.setStyleSheet(muted_style)
        
        self._saved_param_state = set()

        # Update labels visual state
        def on_switch_toggle(checked):
            self.lbl_off.setStyleSheet(f"border: none; color: {Colors.TEXT_MUTED if checked else Colors.TEXT_PRIMARY}; font-weight: {'normal' if checked else 'bold'}; font-size: 13px;")
            self.lbl_on.setStyleSheet(f"border: none; color: {Colors.TEXT_PRIMARY if checked else Colors.TEXT_MUTED}; font-weight: {'bold' if checked else 'normal'}; font-size: 13px;")
            self._on_lights_on_toggled(checked)

        self.lights_on_switch.toggled.connect(on_switch_toggle)
        
        switch_layout.addWidget(self.lbl_lights)
        switch_layout.addStretch() 
        switch_layout.addWidget(self.lbl_off)
        switch_layout.addWidget(self.lights_on_switch)
        switch_layout.addWidget(self.lbl_on)
        
        header_layout.addWidget(switch_container)
        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {Colors.TEXT_MUTED};
                border: none;
                font-size: 14px;
            }}
            QPushButton:hover {{
                color: {Colors.TEXT_PRIMARY};
            }}
        """)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.on_toggle_params_drawer)
        header_layout.addWidget(close_btn)

        drawer_layout.addLayout(header_layout)

        # QLineEdit search filter for parameters (Phase 4.2)
        self.param_search = QLineEdit()
        self.param_search.setPlaceholderText("Search parameters...")
        self.param_search.setClearButtonEnabled(True)
        self.param_search.setStyleSheet(f"""
            QLineEdit {{
                background-color: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {get_border_subtle()};
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT_PRIMARY};
            }}
        """)
        self.param_search.textChanged.connect(self._on_param_search_changed)
        drawer_layout.addWidget(self.param_search)

        # QTableWidget for parameter list (optimized for performance)
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(1)
        self.param_table.horizontalHeader().setVisible(False)
        self.param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setShowGrid(False)
        self.param_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.param_table.setFocusPolicy(Qt.NoFocus)
        self.param_table.setAlternatingRowColors(False)
        
        self.param_table.verticalHeader().setDefaultSectionSize(36)
        self.param_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Style matches the drawer background
        self.param_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.BG_SECONDARY};
                border: none;
            }}
            QTableWidget::item {{
                padding: 0px;
                border: none;
            }}
        """)

        # Connect itemChanged signal for manual toggles - REMOVED (using QCheckBox signals)
        
        drawer_layout.addWidget(self.param_table)
        
        # Initialize dictionary to map param name to QCheckBox (for fit logic)
        self.param_checkboxes = {}

    def on_toggle_params_drawer(self):
        """Toggle the parameter drawer. Overlay for remote, animated for local."""
        from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QPoint
        
        drawer_width = 250
        
        if self.is_remote:
             # OVERLAY MODE: Drop-down over control panel
             self.params_drawer_open = not self.params_drawer_open
             
             if self.params_drawer_open:
                 # Align to Control Panel (Fixed 240)
                 panel_width = 240
                 w_total = self.centralWidget().width()
                 
                 # Calculate Y position: Bottom of the button
                 # Map button's bottom-left (0, height) to central widget coords
                 btn_bottom_global = self.params_drawer_btn.mapTo(self.centralWidget(), QPoint(0, self.params_drawer_btn.height()))
                 y_pos = btn_bottom_global.y()
                 x_pos = w_total - panel_width # Right aligned
                 
                 h_available = self.centralWidget().height() - y_pos
                 
                 self.params_drawer.setFixedWidth(panel_width)
                 self.params_drawer.setGeometry(x_pos, y_pos, panel_width, h_available)
                 self.params_drawer.raise_()
                 self.params_drawer.setVisible(True)
                 
                 self.params_drawer_btn.setText("Parameters to Fit  ▴")
             else:
                 self.params_drawer.setVisible(False)
                 self.params_drawer_btn.setText("Parameters to Fit  ▾")
             return

        # LOCAL MODE: Smooth Animation
        self.params_drawer_open = not self.params_drawer_open
        
        # Unlock fixed width constraints for animation
        self.params_drawer.setMinimumWidth(0)
        self.params_drawer.setMaximumWidth(drawer_width if self.params_drawer_open else 0)

        # Initialize animation if needed
        if not hasattr(self, '_drawer_anim'):
            self._drawer_anim = QPropertyAnimation(self.params_drawer, b"maximumWidth")
            self._drawer_anim.setDuration(300)
            self._drawer_anim.setEasingCurve(QEasingCurve.OutQuart)
            
        # Clear previous connections
        try: self._drawer_anim.finished.disconnect()
        except: pass

        if self.params_drawer_open:
             # OPENING
            current_size = self.size()
            self.resize(current_size.width() + drawer_width, current_size.height())
            
            self._drawer_anim.setStartValue(0)
            self._drawer_anim.setEndValue(drawer_width)
            self.params_drawer_btn.setText("Parameters to Fit  ▴")
            self._drawer_anim.start()
        else:
            # CLOSING
            self._drawer_anim.setStartValue(drawer_width)
            self._drawer_anim.setEndValue(0)
            self.params_drawer_btn.setText("Parameters to Fit  ▾")
            
            def on_close_finished():
                # Shrink window back
                curr = self.size()
                self.resize(curr.width() - drawer_width, curr.height())
                
            self._drawer_anim.finished.connect(on_close_finished)
            self._drawer_anim.start()

    def _on_lights_on_toggled(self, checked):
        """Toggle all parameters on/off, restoring previous state.
        Defer execution slightly to allow UI (switch animation) to start smoothly.
        """
        from PySide6.QtCore import QTimer
        QTimer.singleShot(20, lambda: self._execute_lights_on_toggle(checked))

    def _execute_lights_on_toggle(self, checked):
        """Execute the actual parameter toggling logic."""
        # Use param_table if available
        target = self.param_table if hasattr(self, 'param_table') else self
        if hasattr(target, 'setUpdatesEnabled'):
            target.setUpdatesEnabled(False)
        
        try:
            if checked:
                # Lights ON: Save state and enable all
                self._saved_param_state.clear()
                for param, cb in self.param_checkboxes.items():
                    if cb.isChecked():
                        self._saved_param_state.add(param)
                    cb.blockSignals(True)
                    cb.setChecked(True)
                    cb.blockSignals(False)
                self.statusBar().showMessage("All parameters enabled", 3000)
            else:
                # Lights OFF: Restore previous state
                for param, cb in self.param_checkboxes.items():
                    should_check = param in self._saved_param_state
                    cb.blockSignals(True)
                    cb.setChecked(should_check)
                    cb.blockSignals(False)
                self._saved_param_state.clear()
                self.statusBar().showMessage("Restored parameter selection", 3000)
        finally:
            if hasattr(target, 'setUpdatesEnabled'):
                target.setUpdatesEnabled(True)

    def _on_param_manual_toggle(self, checked):
        """Handle individual parameter toggle to sync with master switch."""
        # If user manually unchecks a box while "Lights On" is active,
        # we must turn off the master switch to reflect state is no longer "All On".
        
        if not checked and hasattr(self, 'lights_on_switch') and self.lights_on_switch.isChecked():
            # Block signals to prevent recursive restore logic
            self.lights_on_switch.blockSignals(True)
            self.lights_on_switch.setChecked(False)
            
            # Manually reset animation state for custom widget to match checked state
            if hasattr(self.lights_on_switch, '_handle_pos'):
                self.lights_on_switch._handle_pos = 0.0
                self.lights_on_switch.update()
                
            self.lights_on_switch.blockSignals(False)
            
            # Clear saved state as we've exited the clean mode
            self._saved_param_state.clear()

    def _on_param_search_changed(self, text: str):
        """Filter visible parameters in the table by search text (Phase 4.2)."""
        # Fix: don't strip() so backspacing to empty works immediately
        search = text.upper()
        for row in range(self.param_table.rowCount()):
            widget = self.param_table.cellWidget(row, 0)
            if widget is None:
                continue
            param_name = widget.text().upper()
            visible = (not search) or (search in param_name)
            self.param_table.setRowHidden(row, not visible)

    def _create_stat_row(self, parent_layout, label_text, initial_value):
        """Create a styled statistic row with label and value. Returns (label, value) tuple."""
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel(label_text)
        label.setProperty("class", "statLabel")

        value = QLabel(initial_value)
        value.setProperty("class", "statValue")
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addWidget(label)
        row.addStretch()
        row.addWidget(value)

        parent_layout.addLayout(row)

        # Return both label and value for theme updates
        return label, value
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_par_action = file_menu.addAction("Open .par...")
        open_par_action.setShortcut("Ctrl+P")
        open_par_action.triggered.connect(self.on_open_par)
        
        open_tim_action = file_menu.addAction("Open .tim...")
        open_tim_action.setShortcut("Ctrl+T")
        open_tim_action.triggered.connect(self.on_open_tim)
        
        file_menu.addSeparator()
        
        save_par_action = file_menu.addAction("Save .par...")
        save_par_action.setShortcut("Ctrl+S")
        save_par_action.triggered.connect(self.on_save_par)
        save_par_action.setEnabled(False)
        self.save_par_action = save_par_action
        
        self.save_tim_action = file_menu.addAction("Save .tim...")
        self.save_tim_action.setShortcut("Ctrl+Shift+S")
        self.save_tim_action.triggered.connect(self.on_save_tim)
        self.save_tim_action.setEnabled(False)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu("&View")

        params_action = view_menu.addAction("Parameters...")
        params_action.setShortcut("Ctrl+E")
        params_action.triggered.connect(self.on_show_parameters)

        view_menu.addSeparator()

        # Zero line toggle (checkable)
        self.zero_line_action = view_menu.addAction("Show Zero Line")
        self.zero_line_action.setCheckable(True)
        self.zero_line_action.setChecked(self.show_zero_line)
        self.zero_line_action.triggered.connect(self.on_toggle_zero_line)

        # Residuals visibility toggle
        self._show_residuals_action = view_menu.addAction("Show Residuals")
        self._show_residuals_action.setCheckable(True)
        self._show_residuals_action.setChecked(True)
        self._show_residuals_action.setShortcut("R")
        self._show_residuals_action.triggered.connect(self._on_toggle_residuals)

        view_menu.addSeparator()

        # Theme toggle
        self.theme_action = view_menu.addAction("Theme: Light")
        self.theme_action.triggered.connect(self.on_toggle_theme)

        # Color variant toggle (available in both light and dark modes)
        self.variant_action = view_menu.addAction("Data Color: Navy")
        self.variant_action.triggered.connect(self.on_toggle_color_variant)

        # Phase 4.4: Color By submenu (backend / frequency colormap)
        color_by_menu = view_menu.addMenu("Color By")

        # key fix: use QActionGroup for mutual exclusivity
        from PySide6.QtGui import QActionGroup
        color_group = QActionGroup(self)

        self._color_none_action = color_by_menu.addAction("None (single color)")
        self._color_none_action.setCheckable(True)
        self._color_none_action.setChecked(True)
        self._color_none_action.setActionGroup(color_group)
        self._color_none_action.triggered.connect(lambda: self._set_color_mode("none"))

        self._color_backend_action = color_by_menu.addAction("Backend/Receiver")
        self._color_backend_action.setCheckable(True)
        self._color_backend_action.setActionGroup(color_group)
        self._color_backend_action.triggered.connect(lambda: self._set_color_mode("backend"))

        self._color_freq_action = color_by_menu.addAction("Frequency (continuous)")
        self._color_freq_action.setCheckable(True)
        self._color_freq_action.setActionGroup(color_group)
        self._color_freq_action.triggered.connect(lambda: self._set_color_mode("frequency"))

        view_menu.addSeparator()

        # Noise panel toggle
        self.noise_panel_action = view_menu.addAction("Show Noise Panel")
        self.noise_panel_action.setCheckable(True)
        self.noise_panel_action.setChecked(False)
        self.noise_panel_action.triggered.connect(self._on_toggle_noise_panel)

        # Noise subtraction shortcuts (tempo2-style)
        subtract_rn_action = view_menu.addAction("Subtract Red Noise")
        subtract_rn_action.setShortcut("Shift+K")
        subtract_rn_action.triggered.connect(lambda: self._toggle_noise_subtract("RedNoise"))

        subtract_dm_action = view_menu.addAction("Subtract DM Noise")
        subtract_dm_action.setShortcut("Shift+F")
        subtract_dm_action.triggered.connect(lambda: self._toggle_noise_subtract("DMNoise"))

        view_menu.addSeparator()

        zoom_fit_action = view_menu.addAction("Zoom to Fit")
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.on_zoom_fit)

        unzoom_action = view_menu.addAction("Unzoom (Fit to Data)")
        unzoom_action.setShortcut("U")
        unzoom_action.triggered.connect(self.on_zoom_fit)

        box_zoom_action = view_menu.addAction("Box Zoom...")
        box_zoom_action.setShortcut("Z")
        box_zoom_action.triggered.connect(self._handle_box_zoom_key)

        box_delete_action = view_menu.addAction("Box Delete...")
        box_delete_action.setShortcut("Shift+Z")
        box_delete_action.triggered.connect(self._handle_box_delete_key)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        fit_action = tools_menu.addAction("Run Fit")
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self.on_fit_clicked)
        
        restart_action = tools_menu.addAction("Restart")
        restart_action.setShortcut("Ctrl+R")
        restart_action.triggered.connect(self.on_restart_clicked)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = help_menu.addAction("About JUG...")
        about_action.triggered.connect(self.on_about)

    # -----------------------------------------------------------------
    # Remote overlay menu bar (replaces QMenuBar over SSH/X11)
    # -----------------------------------------------------------------

    def _create_remote_menu_bar(self):
        """Create overlay-based menu bar for remote/SSH sessions.

        Replaces QMenuBar to eliminate top-level X11 window creation.
        Each menu is a MenuBarOverlayMenu (child QFrame).
        """
        self.menuBar().setVisible(False)

        bar = QFrame()
        bar.setObjectName("remoteMenuBar")
        bar.setStyleSheet(f"""
            QFrame#remoteMenuBar {{
                background-color: {Colors.BG_SECONDARY};
                border-bottom: 1px solid {get_border_subtle()};
                padding: 0px;
            }}
        """)
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 2, 8, 2)
        bar_layout.setSpacing(0)

        self._active_menu_overlay = None
        self._remote_menu_bar = bar

        # Build the four overlay menus
        self._file_overlay = self._build_file_overlay()
        self._view_overlay = self._build_view_overlay()
        self._tools_overlay = self._build_tools_overlay()
        self._help_overlay = self._build_help_overlay()

        for label, overlay in [
            ("File", self._file_overlay),
            ("View", self._view_overlay),
            ("Tools", self._tools_overlay),
            ("Help", self._help_overlay),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    color: {Colors.TEXT_PRIMARY};
                    padding: 4px 12px;
                    border: none;
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.SURFACE_HOVER};
                    border-radius: 2px;
                }}
            """)
            btn.clicked.connect(
                lambda checked=False, o=overlay, b=btn: self._toggle_menu_overlay(o, b))
            bar_layout.addWidget(btn)

        bar_layout.addStretch()
        self.setMenuWidget(bar)

    def _toggle_menu_overlay(self, overlay, trigger_btn):
        """Toggle an overlay menu, closing any other open overlay first."""
        if self._active_menu_overlay and self._active_menu_overlay is not overlay:
            self._active_menu_overlay.close_menu()

        if overlay.isVisible():
            overlay.close_menu()
            self._active_menu_overlay = None
        else:
            # Use global coords since menu bar widget is not in centralWidget hierarchy
            global_pos = trigger_btn.mapToGlobal(QPointF(0, trigger_btn.height()).toPoint())
            local_pos = self.centralWidget().mapFromGlobal(global_pos)
            overlay.show_at(local_pos)
            self._active_menu_overlay = overlay

    def _build_file_overlay(self):
        m = MenuBarOverlayMenu(self.centralWidget())
        m.add_action("Open .par...", self.on_open_par, shortcut="Ctrl+P")
        m.add_action("Open .tim...", self.on_open_tim, shortcut="Ctrl+T")
        m.add_separator()
        m.add_action("Save .par...", self.on_save_par, shortcut="Ctrl+S",
                      enabled=False, name="save_par")
        m.add_action("Save .tim...", self.on_save_tim, shortcut="Ctrl+Shift+S",
                      enabled=False, name="save_tim")
        m.add_separator()
        m.add_action("Exit", self.close, shortcut="Ctrl+Q")
        return m

    def _build_view_overlay(self):
        m = MenuBarOverlayMenu(self.centralWidget(), width=260)
        m.add_action("Parameters...", self.on_show_parameters, shortcut="Ctrl+E")
        m.add_separator()
        m.add_action("Show Zero Line", self.on_toggle_zero_line,
                      checkable=True, checked=self.show_zero_line, name="zero_line")
        m.add_separator()
        m.add_action("Theme: Light", self.on_toggle_theme, name="theme")
        m.add_action("Data Color: Navy", self.on_toggle_color_variant, name="variant")
        m.add_submenu_group("Color By", [
            ("None (single color)", lambda: self._set_color_mode("none"), True),
            ("Backend/Receiver", lambda: self._set_color_mode("backend"), False),
            ("Frequency (continuous)", lambda: self._set_color_mode("frequency"), False),
        ], group_name="color_by")
        m.add_separator()
        m.add_action("Show Noise Panel", self._on_toggle_noise_panel,
                      checkable=True, checked=False, name="noise_panel")
        m.add_separator()
        m.add_action("Zoom to Fit", self.on_zoom_fit, shortcut="Ctrl+0")
        m.add_action("Unzoom (Fit to Data)", self.on_zoom_fit, shortcut="U")
        m.add_action("Box Zoom...", self._handle_box_zoom_key, shortcut="Z")
        m.add_action("Box Delete...", self._handle_box_delete_key, shortcut="Shift+Z")
        return m

    def _build_tools_overlay(self):
        m = MenuBarOverlayMenu(self.centralWidget())
        m.add_action("Run Fit", self.on_fit_clicked, shortcut="Ctrl+F")
        m.add_action("Restart", self.on_restart_clicked, shortcut="Ctrl+R")
        return m

    def _build_help_overlay(self):
        m = MenuBarOverlayMenu(self.centralWidget())
        m.add_action("About JUG...", self.on_about)
        return m

    def _register_shortcuts(self):
        """Register keyboard shortcuts independent of menu bar.

        In remote mode, QMenuBar is hidden so shortcuts must be registered
        directly on the MainWindow.
        """
        shortcuts = [
            ("Ctrl+P", self.on_open_par),
            ("Ctrl+T", self.on_open_tim),
            ("Ctrl+S", self.on_save_par),
            ("Ctrl+Shift+S", self.on_save_tim),
            ("Ctrl+Q", self.close),
            ("Ctrl+E", self.on_show_parameters),
            ("Ctrl+0", self.on_zoom_fit),
            ("Ctrl+F", self.on_fit_clicked),
            ("Ctrl+R", self.on_restart_clicked),
        ]
        for shortcut_str, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut_str))
            action.triggered.connect(callback)
            self.addAction(action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def on_open_par(self):
        """Handle Open .par file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open .par File", "", "Par Files (*.par);;All Files (*)"
        )
        if filename:
            self.par_file = Path(filename)
            self.status_bar.showMessage(f"Loaded .par: {self.par_file.name}")
            self._update_available_parameters()
            self._check_ready_to_compute()
    
    def on_open_tim(self):
        """Handle Open .tim file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open .tim File", "", "Tim Files (*.tim);;All Files (*)"
        )
        if filename:
            self.tim_file = Path(filename)
            self.status_bar.showMessage(f"Loaded .tim: {self.tim_file.name}")
            self._check_ready_to_compute()
    
    def on_save_par(self):
        """Handle Save .par file."""
        # TODO: Implement in Phase 3
        self.status_bar.showMessage("Save .par not yet implemented (Phase 3)")
    
    def _check_ready_to_compute(self):
        """Check if we have both files and can compute residuals."""
        if self.par_file and self.tim_file:
            self._create_session()
    
    def _create_session(self):
        """Create timing session in background."""
        from jug.gui.workers.session_worker import SessionWorker
        
        self.status_bar.showMessage("Loading files...")
        
        # Disable controls during load
        if hasattr(self, 'fit_button'):
            self.fit_button.setEnabled(False)
        
        # Create session worker
        worker = SessionWorker(self.par_file, self.tim_file)
        worker.signals.result.connect(self.on_session_ready)
        worker.signals.error.connect(self.on_session_error)
        worker.signals.progress.connect(self.on_session_progress)
        worker.signals.finished.connect(self.on_session_finished)
        
        # Start in thread pool
        self.thread_pool.start(worker)
    
    def on_session_ready(self, session):
        """Handle successful session creation."""
        self.session = session
        
        # Get initial params for GUI
        self.initial_params = session.get_initial_params()
        
        # Populate noise panel from par file params
        if self.noise_panel is not None:
            self.noise_panel.populate_from_params(session.params)
            if self.noise_panel.has_noise():
                self._set_noise_panel_visible(True)
            else:
                # No noise yet — keep panel hidden but show title-bar button
                self._set_noise_panel_visible(False)
                if hasattr(self, '_noise_title_btn'):
                    self._noise_title_btn.setVisible(True)
        
        # Now compute initial residuals
        self._compute_initial_residuals()
    
    def on_session_error(self, error_msg):
        """Handle session creation error."""
        QMessageBox.critical(self, "Session Error", f"Failed to load files:\n\n{error_msg}")
        self.status_bar.showMessage("Failed to load files")
    
    def on_session_progress(self, message):
        """Handle session progress updates."""
        self.status_bar.showMessage(message)
    
    def on_session_finished(self):
        """Handle session worker completion."""
        pass  # Re-enable handled in on_session_ready
    
    def _compute_initial_residuals(self):
        """Compute initial residuals using session (background)."""
        if not self.session:
            return
        
        from jug.gui.workers.compute_worker import ComputeWorker
        
        self.status_bar.showMessage("Computing residuals...")
        
        # Create compute worker
        worker = ComputeWorker(self.session)
        worker.signals.result.connect(self.on_compute_complete)
        worker.signals.error.connect(self.on_compute_error)
        worker.signals.progress.connect(self.status_bar.showMessage)
        worker.signals.finished.connect(self.on_compute_finished)
        
        # Start in thread pool
        self.thread_pool.start(worker)
    
    def on_compute_complete(self, result):
        """Handle successful residual computation."""
        # Store data
        self.mjd = result['tdb_mjd']
        self.residuals_us = result['residuals_us']
        self.prefit_residuals_us = result['residuals_us'].copy()
        self.errors_us = result.get('errors_us', None)
        self.toa_freqs = result.get('freq_bary_mhz', None)  # Phase 4.3
        self.toa_flags = result.get('toa_flags', None)       # Phase 4.4
        self.orbital_phase = result.get('orbital_phase', None) # Phase 4.3 fix
        self._update_rms_from_result(result)
        self.prefit_weighted_rms_us = self.weighted_rms_us
        self.prefit_unweighted_rms_us = self.unweighted_rms_us
        self.is_fitted = False
        self.fit_results = None
        
        # Store original data for full restart (only on first load)
        if self.original_mjd is None:
            self.original_mjd = self.mjd.copy()
            self.original_residuals_us = self.residuals_us.copy()
            self.original_errors_us = self.errors_us.copy() if self.errors_us is not None else None
            self.deleted_indices = set()
            # Initialize keep mask (all True = keep all TOAs)
            self._keep_mask = np.ones(len(self.original_mjd), dtype=bool)
        
        # Update plot
        self._update_plot()
        self._update_plot_title()
        
        # Enable controls
        self.fit_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.fit_window_button.setEnabled(True)
        if self.is_remote:
            self._file_overlay.set_item_enabled("save_tim", True)
        else:
            self.save_tim_action.setEnabled(True)

        # Update statistics (now just values, labels are separate)
        self.rms_label.setText(f"{self.rms_us:.6f} μs")
        self.ntoa_label.setText(f"{len(self.mjd)}")
        
        # Update status
        rms_label = "wRMS" if self.use_weighted_rms else "RMS"
        self.status_bar.showMessage(
            f"Loaded {len(self.mjd)} TOAs, Prefit {rms_label} = {self.rms_us:.6f} μs"
        )

        # Schedule JAX warmup in background (avoids first-fit lag)
        self._schedule_jax_warmup()
    
    def _schedule_jax_warmup(self):
        """Schedule JAX warmup in background to avoid first-fit lag.

        Note: Warmup is currently disabled to avoid slowing down initial load.
        The first fit will trigger JIT compilation instead.
        """
        # DISABLED: Warmup can slow down load on some systems
        # Uncomment to enable background JIT warmup:
        #
        # if self.mjd is None or len(self.mjd) == 0:
        #     return
        # from jug.gui.workers.warmup_worker import WarmupWorker
        # worker = WarmupWorker(n_toas=len(self.mjd))
        # worker.signals.progress.connect(lambda msg: None)
        # worker.signals.finished.connect(lambda: None)
        # self.thread_pool.start(worker)
        pass

    def on_compute_error(self, error_msg):
        """Handle computation error."""
        QMessageBox.critical(self, "Compute Error", f"Computation failed:\n\n{error_msg}")
        self.status_bar.showMessage("Computation failed")
    
    def on_compute_finished(self):
        """Handle compute worker completion."""
        pass  # Re-enable handled in on_compute_complete
    
    def _update_plot(self, auto_range=None):
        """
        Update the residual plot.
        
        Parameters
        ----------
        auto_range : bool, optional
            If True, call autoRange(). If None, auto-ranges only on first plot.
        """
        if self.mjd is None or self.residuals_us is None:
            return
        
        # Determine if this is first plot
        is_first_plot = self.scatter_item is None
        if auto_range is None:
            auto_range = is_first_plot
        
        # Block view range signals during data update to prevent cascading
        # redraws (setData triggers sigRangeChanged → beam recalc → redraw)
        view_box = self.plot_widget.getPlotItem().getViewBox()
        view_box.blockSignals(True)
        
        try:
            # Disable auto-range during batch update (prevents multiple range calcs)
            view_box.disableAutoRange()
            
            # Update or create scatter plot with modern styling
            if self.scatter_item is None:
                # First time: create themed scatter item
                self.scatter_item = create_scatter_item()
                self.plot_widget.addItem(self.scatter_item)
            
            # Update scatter data (fast - no recreation)
            x_data, x_label = self._get_x_data()
            y_data, y_label = self._get_y_data()
            
            # Phase 4.4: Get per-point coloring
            # Renamed from _apply_point_colors to prevent double-update
            brushes = self._get_point_brushes(len(x_data))
            
            # Efficient single update: set data AND brush at once
            # This avoids the "clearing per-point brush" issue and is faster (one repaint)
            # If brushes is None, we rely on setSymbolBrush for default color
            self.scatter_item.setData(x=x_data, y=y_data, brush=brushes)

            # Set default symbol brush if no per-point coloring
            if brushes is None:
                from jug.gui.theme import Colors
                self.scatter_item.setBrush(pg.mkBrush(Colors.DATA_POINTS))
                self.scatter_item.setPen(pg.mkPen(None))
            else:
                 # Clear default symbol brush if per-point coloring is active (optional but clean)
                 self.scatter_item.setBrush(None)

            # Update axis labels
            self.plot_widget.setLabel('bottom', x_label)
            self.plot_widget.setLabel('left', y_label)
            
            # Update or create error bars with modern styling
            if self.errors_us is not None:
                if self.error_bar_item is None:
                    # First time: create themed error bar item
                    self.error_bar_item = create_error_bar_item()
                    self.plot_widget.addItem(self.error_bar_item)
                
                # Update error bar data
                eb_height = self.errors_us * 2  # ±1σ
                # If Y axis is normalized, scale error bars accordingly
                y_mode = self._y_axis_mode
                if y_mode == "Normalized":
                    # In normalized mode, error bars are ±1 (by definition)
                    eb_height = np.ones_like(self.errors_us) * 2
                self.error_bar_item.setData(
                    x=x_data,
                    y=y_data,
                    height=eb_height
                )
            elif self.error_bar_item is not None:
                # Remove error bars if no longer needed
                self.plot_widget.removeItem(self.error_bar_item)
                self.error_bar_item = None
            
            # Residuals visibility (scatter + error bars)
            if self.scatter_item is not None:
                self.scatter_item.setVisible(self.show_residuals)
            if self.error_bar_item is not None:
                self.error_bar_item.setVisible(self.show_residuals and self.show_uncertainties)

            # Update noise realization overlay curves
            self._update_noise_curves(x_data, x_label)

            # Add zero line only if enabled (off by default)
            if self.show_zero_line and self.zero_line is None:
                self.zero_line = create_zero_line(self.plot_widget)
            
            # (No explicit _apply_point_colors call anymore)
        finally:
            # Re-enable signals before auto-range so the final range is applied
            view_box.blockSignals(False)
        
        # Auto-range only when requested (not every update!)
        # Single range calc instead of multiple from setData calls
        if auto_range:
            # Guard against all-NaN or empty data which crashes pyqtgraph's autoRange
            if (self.residuals_us is not None
                    and len(self.residuals_us) > 0
                    and np.any(np.isfinite(self.residuals_us))):
                self.plot_widget.autoRange()
            else:
                # Fallback: set a sensible default range so the plot isn't blank
                self.plot_widget.setYRange(-1, 1)
                if self.mjd is not None and len(self.mjd) > 0 and np.any(np.isfinite(self.mjd)):
                    self.plot_widget.setXRange(float(np.nanmin(self.mjd)), float(np.nanmax(self.mjd)))

    # -- Phase 4.3: Axis routing helpers -----------------------------------

    def _create_overlay_axis_button(self, initial_text, options, callback, width=140):
        """Create a styled QPushButton that toggles a custom OverlayMenu."""
        from jug.gui.theme import Colors, get_border_subtle, AnimatedButton
        from PySide6.QtCore import QPoint
        
        # Use AnimatedButton (secondary role) for axis selectors
        btn = AnimatedButton(initial_text + "  ▾", role="secondary")
        btn.setFixedWidth(width)
        # Match height to plot title label by reducing padding and removing min-height
        from jug.gui.theme import Spacing as _Sp
        btn.setStyleSheet(btn.styleSheet().replace("min-height: 44px", "min-height: 0px").replace(
            f"padding: {_Sp.MD} {_Sp.LG}",
            f"padding: {_Sp.SM} {_Sp.LG}"
        ))
        
        # Create the overlay menu
        # Parent to self (MainWindow) to ensure it floats above centralWidget and all children
        overlay = OverlayMenu(self, options, callback, width=width)
        
        # Wrap callback to update button text and toggle chevron
        original_callback = overlay.callback
        def wrapped_callback(val):
            btn.setText(val + "  ▾")
            original_callback(val)
        overlay.callback = wrapped_callback
        
        # Toggle logic
        def toggle_overlay():
            if overlay.isVisible():
                overlay.hide()
                btn.setText(btn.text().replace("▴", "▾"))
            else:
                # Calculate position relative to MainWindow (self)
                # This ensures correct placement even if centralWidget has margins/offsets
                pos_in_window = btn.mapTo(self, QPoint(0, btn.height()))
                
                # Update text to show active state
                btn.setText(btn.text().replace("▾", "▴"))
                overlay.toggle_at(pos_in_window)
                
        btn.clicked.connect(toggle_overlay)
        return btn

    def _on_axis_changed(self, x_val=None, y_val=None):
        """Re-plot when axis selection changes (no re-fitting needed)."""
        if x_val:
            self._x_axis_mode = x_val
        if y_val:
            self._y_axis_mode = y_val
            
        if self.mjd is not None and self.residuals_us is not None:
            # Force auto-range
            self._update_plot(auto_range=True)
            self._update_plot_title()
            
            # Special handling for Orbital Phase: force 0..1 range if auto-range fails or to be explicit
            if self._x_axis_mode == "Orbital Phase":
                 self.plot_widget.setXRange(0.0, 1.0, padding=0.0)

    def _get_x_data(self):
        """Return (x_array, x_label) based on the current X-axis selector."""
        mode = self._x_axis_mode
        mjd = self.mjd

        if mode == "MJD":
            return mjd, "MJD"

        if mode == "Serial":
            return np.arange(len(mjd), dtype=np.float64), "TOA number"

        if mode == "Year":
            # MJD → Year (approximate: MJD 51544.0 = 2000-01-01.5)
            return 2000.0 + (mjd - 51544.0) / 365.25, "Year"

        if mode == "Day of Year":
            # Day-of-year = fractional part of year × 365.25
            year_frac = (mjd - 51544.0) / 365.25
            return (year_frac - np.floor(year_frac)) * 365.25, "Day of Year"

        if mode == "Frequency (MHz)":
            if hasattr(self, 'toa_freqs') and self.toa_freqs is not None:
                return self.toa_freqs, "Frequency (MHz)"
            return mjd, "MJD (freq unavailable)"

        if mode == "ToA Error (μs)":
            if self.errors_us is not None:
                return self.errors_us, "ToA Error (μs)"
            return mjd, "MJD (errors unavailable)"

        if mode == "Orbital Phase":
            if hasattr(self, 'orbital_phase') and self.orbital_phase is not None:
                return self.orbital_phase, "Orbital Phase"
            return mjd, "MJD (orbital phase unavailable)"

        # Fallback
        return mjd, "MJD"

    def _get_y_data(self):
        """Return (y_array, y_label) based on the current Y-axis selector."""
        mode = self._y_axis_mode
        if mode == "Pre-fit (μs)":
            if hasattr(self, 'prefit_residuals_us') and self.prefit_residuals_us is not None:
                return self.prefit_residuals_us, "Pre-fit residual (μs)"
            return self.residuals_us, "Residual (μs) [no pre-fit]"

        if mode == "Normalized":
            if self.errors_us is not None and len(self.errors_us) > 0:
                from jug.engine.stats import compute_normalized_residuals
                norm = compute_normalized_residuals(self.residuals_us, self.errors_us)
                return norm, "Normalized residual"
            return self.residuals_us, "Residual (μs) [no errors]"

        # Default: Post-fit (μs)
        return self.residuals_us, "Post-fit residual (μs)"
    
    def _update_plot_title(self):
        """Update the plot title with pulsar name and RMS matching displayed view."""
        pulsar_str = self.pulsar_name if self.pulsar_name else "Unknown Pulsar"
        rms_label = "wRMS" if self.use_weighted_rms else "RMS"

        # Show RMS corresponding to what's on screen
        if self._y_axis_mode == "Pre-fit (μs)":
            rms_val = self.prefit_weighted_rms_us if self.use_weighted_rms else self.prefit_unweighted_rms_us
        else:
            rms_val = self.weighted_rms_us if self.use_weighted_rms else self.unweighted_rms_us

        rms_str = f"{rms_val:.6f} μs" if rms_val is not None else "--"
        self.plot_title_label.setText(f"✦  {pulsar_str}  ·  {rms_label}: {rms_str}")
    
    def on_fit_clicked(self):
        """Handle Fit button click."""
        if not self.session:
            QMessageBox.warning(self, "No Session", "Please load .par and .tim files first")
            return

        # Get selected parameters, canonicalize, and validate
        from jug.model.parameter_spec import canonicalize_param_name, validate_fit_param
        fit_params = [canonicalize_param_name(param)
                      for param, checkbox in self.param_checkboxes.items()
                      if checkbox.isChecked()]

        if not fit_params:
            QMessageBox.warning(self, "No Parameters",
                              "Please select at least one parameter to fit")
            return

        # Validate all selected parameters
        for param in fit_params:
            try:
                validate_fit_param(param)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Parameter", str(e))
                return

        # Disable fit button during fitting
        self.fit_button.setEnabled(False)
        
        # Use cached TOA mask if we have deleted TOAs (O(1) vs O(N) Python loop)
        toa_mask = None
        n_total = len(self.original_mjd) if self.original_mjd is not None else 0
        n_deleted = len(self.deleted_indices) if self.deleted_indices else 0

        if n_deleted > 0 and n_total > 0:
            # Use pre-computed boolean mask (updated incrementally on delete)
            toa_mask = self._keep_mask
            n_used = np.sum(toa_mask)
            self.status_bar.showMessage(f"Fitting {', '.join(fit_params)} on {n_used} TOAs ({n_deleted} excluded)...")
        else:
            self.status_bar.showMessage(f"Fitting {', '.join(fit_params)}...")

        # Run fit with mask, solver mode, and noise config
        from jug.gui.workers.fit_worker import FitWorker

        noise_config = None
        if self.noise_panel is not None:
            noise_config = self.noise_panel.get_noise_config()

        # Remember which noise was active for the fit report
        self._last_fit_noise_config = noise_config

        worker = FitWorker(self.session, fit_params, toa_mask=toa_mask,
                           solver_mode=self.solver_mode,
                           noise_config=noise_config)
        worker.signals.result.connect(self.on_fit_complete)
        worker.signals.error.connect(self.on_fit_error)
        worker.signals.finished.connect(self.on_fit_finished)

        # Start in thread pool
        self.thread_pool.start(worker)
    
    def on_fit_complete(self, result):
        """
        Handle successful fit completion.

        Parameters
        ----------
        result : dict
            Fit results from fit_parameters_optimized
        """
        # Store fit results
        self.fit_results = result
        self.is_fitted = True

        # Store fit result temporarily for postfit callback
        self._pending_fit_result = result

        # Recompute residuals with fitted parameters (async)
        # Stats and dialog will be shown when postfit completes
        self._compute_postfit_residuals(result)

    def _compute_postfit_residuals(self, result):
        """
        Compute postfit residuals using fitted parameters.
        
        Uses session for fast computation (no file I/O needed!).

        Parameters
        ----------
        result : dict
            Fit results containing final_params
        """
        if not self.session:
            return
        
        # Use ComputeWorker with fitted parameters (background, but will be instant from cache)
        from jug.gui.workers.compute_worker import ComputeWorker
        
        self.status_bar.showMessage("Computing postfit residuals...")
        
        # Create compute worker with fitted parameters
        worker = ComputeWorker(self.session, params=result['final_params'])
        worker.signals.result.connect(self.on_postfit_compute_complete)
        worker.signals.error.connect(self.on_compute_error)
        worker.signals.finished.connect(lambda: None)  # No action needed
        
        # Start in thread pool
        self.thread_pool.start(worker)
    
    def on_postfit_compute_complete(self, result):
        """Handle postfit residual computation completion."""
        # Get full data from session
        full_mjd = result['tdb_mjd']
        full_residuals = result['residuals_us']
        full_errors = result.get('errors_us', None)
        
        # Filter out deleted TOAs if any have been deleted
        if self.deleted_indices and len(self.deleted_indices) > 0:
            # Use pre-computed boolean mask (O(1) vs O(N) Python loop)
            keep_mask = self._keep_mask
            self.mjd = full_mjd[keep_mask]
            self.residuals_us = full_residuals[keep_mask]
            self.postfit_residuals_us = full_residuals[keep_mask].copy()
            self.prefit_residuals_us = self.prefit_residuals_us  # Keep current prefit (already filtered)
            if full_errors is not None:
                self.errors_us = full_errors[keep_mask]
            else:
                self.errors_us = None
            # Recalculate RMS for filtered data using canonical engine stats
            from jug.engine.stats import compute_residual_stats
            stats = compute_residual_stats(self.residuals_us, self.errors_us)
            self._update_rms_from_stats(stats)
        else:
            # No deletions - use all data
            self.mjd = full_mjd
            self.residuals_us = full_residuals
            self.postfit_residuals_us = full_residuals.copy()
            self.errors_us = full_errors
            self._update_rms_from_result(result)
        
        # Update plot (auto-range to show new residual scale after fit)
        # Auto-switch to post-fit view
        self._y_axis_mode = "Post-fit (μs)"
        if hasattr(self, 'y_axis_btn'):
            self.y_axis_btn.setText("Post-fit (μs)")

        # Store GLS noise realizations from the fit (if available)
        if hasattr(self, '_pending_fit_result'):
            nr = self._pending_fit_result.get('noise_realizations', {})
            if nr:
                # Apply TOA mask if needed
                keep = self._keep_mask if self.deleted_indices else None
                for name, real in nr.items():
                    self._noise_realizations[name] = real[keep] if keep is not None else real.copy()

        # If any noise processes had subtract active, re-subtract the NEW
        # realizations from the NEW residuals (Tempo2-like persistent mode).
        _noise_was_subtracted = False
        if hasattr(self, 'noise_panel'):
            for proc_name in self.noise_panel.get_subtract_processes():
                real = self._noise_realizations.get(proc_name)
                if real is not None and self.residuals_us is not None:
                    self.residuals_us = self.residuals_us - real
                    _noise_was_subtracted = True

        # Refresh stats if noise was re-subtracted so RMS/χ² reflect whitened residuals
        if _noise_was_subtracted:
            self._refresh_stats_from_residuals()

        self._update_plot(auto_range=True)
        self._update_plot_title()
        
        # Now update statistics and show dialog (fit result was stored)
        if hasattr(self, '_pending_fit_result'):
            fit_result = self._pending_fit_result
            
            # Update statistics (values only, labels are separate)
            self.rms_label.setText(f"{self.rms_us:.6f} μs")
            self.iter_label.setText(f"{fit_result['iterations']}")

            # Calculate chi-squared if we have errors
            if self.errors_us is not None:
                n_toas = len(self.mjd)
                n_params = len(fit_result['final_params'])
                dof = n_toas - n_params
                # Calculate proper chi-squared with postfit residuals
                chi2 = np.sum((self.residuals_us / self.errors_us) ** 2)
                chi2_dof = chi2 / dof
                self.chi2_label.setText(f"{chi2_dof:.2f}")

            # Pre-build fit report dialog so it opens instantly on button click
            self._prebuild_fit_report(fit_result)
            self.fit_report_button.setEnabled(True)

            # Update status
            param_str = ', '.join(fit_result['final_params'].keys())
            rms_label = "wRMS" if self.use_weighted_rms else "RMS"
            self.status_bar.showMessage(
                f"Fit complete: {param_str} | "
                f"{rms_label} = {self.rms_us:.6f} μs | "
                f"{fit_result['iterations']} iterations | "
                f"{'converged' if fit_result['converged'] else 'not converged'}"
            )
            
            # Clear pending result
            delattr(self, '_pending_fit_result')
        else:
            self.status_bar.showMessage("Postfit residuals computed successfully")

    def on_fit_error(self, error_msg):
        """
        Handle fit error.

        Parameters
        ----------
        error_msg : str
            Error message from fit worker
        """
        QMessageBox.critical(self, "Fit Error", f"Fitting failed:\n\n{error_msg}")
        self.status_bar.showMessage("Fit failed")

    def on_fit_finished(self):
        """Handle fit worker finishing (success or error)."""
        self.fit_button.setEnabled(True)

    def on_show_fit_report(self):
        """Show fit report dialog when button is clicked."""
        if hasattr(self, '_cached_fit_dialog') and self._cached_fit_dialog is not None:
            # Show pre-built dialog instantly (non-blocking)
            self._cached_fit_dialog.show()
            self._cached_fit_dialog.raise_()
            self._cached_fit_dialog.activateWindow()
        elif self.fit_results:
            # Fallback: build on demand
            self._show_fit_results(self.fit_results)

    def _prebuild_fit_report(self, result):
        """Pre-build the fit report dialog so it opens instantly on button click."""
        # Discard any previous cached dialog
        if hasattr(self, '_cached_fit_dialog') and self._cached_fit_dialog is not None:
            self._cached_fit_dialog.deleteLater()
            self._cached_fit_dialog = None

        pulsar_str = self.pulsar_name or 'Unknown'
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{pulsar_str} - Fit Results")
        dialog.setFixedSize(min(int(self.width() * 0.8), 1000),
                            min(int(self.height() * 0.85), 700))

        report_layout = self._build_fit_report_content(result, dialog)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.hide)
        close_btn.setStyleSheet(get_primary_button_style())
        close_btn.setFixedWidth(120)
        close_btn.setCursor(Qt.PointingHandCursor)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        report_layout.addLayout(btn_layout)

        self._cached_fit_dialog = dialog
    
    def _format_param_row(self, param, result):
        """Format a single parameter row for the fit report.

        Returns (param, new_val_str, prev_val_str, change_str, unc_str, unit).
        """
        from jug.io.par_reader import parse_ra, parse_dec, format_ra, format_dec
        from jug.model.parameter_spec import get_display_unit

        new_value = result['final_params'][param]
        uncertainty = result['uncertainties'][param]
        prev_value = self.initial_params.get(param, 0.0)
        unit = get_display_unit(param)

        if param == 'RAJ':
            prev_value_rad = parse_ra(prev_value) if isinstance(prev_value, str) else prev_value
            change = new_value - prev_value_rad
            new_val_str = format_ra(new_value)
            prev_val_str = format_ra(prev_value_rad) if isinstance(prev_value, str) else str(prev_value)
            change_str = f"{change * 180 / 3.14159265 * 3600:+.6f}"
            unit = "Δ arcsec"
        elif param == 'DECJ':
            prev_value_rad = parse_dec(prev_value) if isinstance(prev_value, str) else prev_value
            change = new_value - prev_value_rad
            new_val_str = format_dec(new_value)
            prev_val_str = format_dec(prev_value_rad) if isinstance(prev_value, str) else str(prev_value)
            change_str = f"{change * 180 / 3.14159265 * 3600:+.6f}"
            unit = "Δ arcsec"
        elif param == 'F0':
            change = new_value - prev_value
            new_val_str = f"{new_value:.15f}"
            prev_val_str = f"{prev_value:.15f}"
            change_str = f"{change:+.15f}"
        elif param.startswith('F') and param[1:].isdigit():
            change = new_value - prev_value
            new_val_str = f"{new_value:.6e}"
            prev_val_str = f"{prev_value:.6e}"
            change_str = f"{change:+.6e}"
        elif param.startswith('DM') and param != 'DMEPOCH':
            change = new_value - prev_value
            new_val_str = f"{new_value:.10f}"
            prev_val_str = f"{prev_value:.10f}"
            change_str = f"{change:+.10f}"
        elif param in ['PMRA', 'PMDEC', 'PX']:
            change = new_value - prev_value
            new_val_str = f"{new_value:.6f}"
            prev_val_str = f"{prev_value:.6f}"
            change_str = f"{change:+.6f}"
        elif param == 'PB':
            change = new_value - prev_value
            new_val_str = f"{new_value:.15f}"
            prev_val_str = f"{prev_value:.15f}"
            change_str = f"{change:+.6e}"
        elif param == 'A1':
            change = new_value - prev_value
            new_val_str = f"{new_value:.10f}"
            prev_val_str = f"{prev_value:.10f}"
            change_str = f"{change:+.10f}"
        elif param in ['T0', 'TASC']:
            change = new_value - prev_value
            new_val_str = f"{new_value:.12f}"
            prev_val_str = f"{prev_value:.12f}"
            change_str = f"{change:+.6e}"
        elif param in ['ECC', 'EPS1', 'EPS2']:
            change = new_value - prev_value
            new_val_str = f"{new_value:.12e}"
            prev_val_str = f"{prev_value:.12e}"
            change_str = f"{change:+.6e}"
        elif param in ['PBDOT', 'XDOT', 'OMDOT', 'EDOT', 'GAMMA',
                       'H3', 'H4', 'STIG'] or param.startswith('FD'):
            change = new_value - prev_value
            new_val_str = f"{new_value:.6e}"
            prev_val_str = f"{prev_value:.6e}"
            change_str = f"{change:+.6e}"
        elif param in ['OM', 'KIN', 'KOM']:
            change = new_value - prev_value
            new_val_str = f"{new_value:.10f}"
            prev_val_str = f"{prev_value:.10f}"
            change_str = f"{change:+.6e}"
        elif param == 'M2':
            change = new_value - prev_value
            new_val_str = f"{new_value:.12f}"
            prev_val_str = f"{prev_value:.12f}"
            change_str = f"{change:+.6e}"
        elif param == 'SINI':
            if isinstance(prev_value, str) and prev_value.upper() == 'KIN':
                kin_deg = float(self.initial_params.get('KIN', 0.0))
                prev_value_num = float(np.sin(np.deg2rad(kin_deg)))
            else:
                prev_value_num = float(prev_value)
            change = new_value - prev_value_num
            new_val_str = f"{new_value:.12f}"
            prev_val_str = f"{prev_value_num:.12f}"
            change_str = f"{change:+.6e}"
        else:
            change = new_value - float(prev_value) if isinstance(prev_value, (int, float)) else 0.0
            new_val_str = f"{new_value:.6g}"
            prev_val_str = f"{prev_value}"
            change_str = f"{change:+.6g}"

        unc_str = f"\u00b1{uncertainty:.2e}"
        return (param, new_val_str, prev_val_str, change_str, unc_str, unit)

    def _build_fit_report_content(self, result, container):
        """Build the styled fit report UI into the given container widget.

        Reused by both overlay and dialog modes. Uses QTableWidget with
        batched updates for performance.
        """
        text_neutral = Colors.TEXT_PRIMARY
        text_muted = Colors.TEXT_SECONDARY
        accent = get_dynamic_accent_primary()
        bg_secondary = Colors.BG_SECONDARY
        border_col = Colors.SURFACE_BORDER
        pulsar_str = self.pulsar_name or 'Unknown'

        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)

        # 1. Header
        title_lbl = QLabel(f"{pulsar_str} - Fit Results")
        title_lbl.setStyleSheet(f"color: {accent}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title_lbl)

        # 2. Summary Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {bg_secondary}; border-radius: 8px;")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(16, 16, 16, 16)

        def add_stat(label, value, color=text_neutral, bold=False):
            c = QWidget()
            vbox = QVBoxLayout(c)
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.setSpacing(4)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {text_muted}; font-size: 12px;")
            lbl.setAlignment(Qt.AlignCenter)
            val = QLabel(str(value))
            weight = "bold" if bold else "normal"
            val.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: {weight}; font-family: monospace;")
            val.setAlignment(Qt.AlignCenter)
            vbox.addWidget(lbl)
            vbox.addWidget(val)
            stats_layout.addWidget(c)

        rms_label = "Final wRMS" if self.use_weighted_rms else "Final RMS"
        rms_value = self.rms_us if self.rms_us is not None else result['final_rms']
        add_stat(rms_label, f"{rms_value:.6f} \u03bcs", color=accent, bold=True)
        add_stat("Iterations", result['iterations'])
        conv_text = "Yes" if result['converged'] else "No"
        conv_color = Colors.ACCENT_SUCCESS if result['converged'] else Colors.ACCENT_WARNING
        add_stat("Converged", conv_text, color=conv_color, bold=True)
        add_stat("Time", f"{result['total_time']:.2f}s")
        layout.addWidget(stats_frame)

        # 2b. Noise processes used in fit
        nc = getattr(self, '_last_fit_noise_config', None)
        active_noise = nc.active_processes() if nc is not None else []
        if active_noise:
            noise_frame = QFrame()
            noise_frame.setStyleSheet(
                f"background-color: {bg_secondary}; border-radius: 8px; "
                f"margin-top: 4px;"
            )
            nf_layout = QHBoxLayout(noise_frame)
            nf_layout.setContentsMargins(16, 10, 16, 10)
            noise_lbl = QLabel("Noise in fit:")
            noise_lbl.setStyleSheet(f"color: {text_muted}; font-size: 12px;")
            nf_layout.addWidget(noise_lbl)
            from jug.gui.widgets.noise_control_panel import _PROCESS_INFO
            names = [_PROCESS_INFO.get(p, {}).get("label", p) for p in active_noise]
            noise_val = QLabel(", ".join(names))
            noise_val.setStyleSheet(
                f"color: {accent}; font-size: 13px; font-weight: bold;"
            )
            nf_layout.addWidget(noise_val)
            nf_layout.addStretch()
            layout.addWidget(noise_frame)
        else:
            no_noise_lbl = QLabel("Fit mode: WLS (no noise model)")
            no_noise_lbl.setStyleSheet(f"color: {text_muted}; font-size: 12px; margin-top: 4px;")
            layout.addWidget(no_noise_lbl)

        # 3. Parameters Table (batched for performance)
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Parameter", "New Value", "Previous", "Change", "Uncertainty", "Unit"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setShowGrid(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)

        # Prepare ordered params
        param_order = getattr(self, 'available_params', list(result['final_params'].keys()))
        ordered_params = [p for p in param_order if p in result['final_params']]
        for p in result['final_params']:
            if p not in ordered_params:
                ordered_params.append(p)

        table.setRowCount(len(ordered_params))

        # Pre-create shared font objects (avoid per-cell allocation)
        mono_font = QFont("Monospace")
        mono_font.setStyleHint(QFont.Monospace)
        bold_font = QFont(mono_font)
        bold_font.setBold(True)

        # Batch: suppress repaints during population
        table.setUpdatesEnabled(False)

        for row, param in enumerate(ordered_params):
            name, nv, pv, ch, unc, unit = self._format_param_row(param, result)

            # Parameter name (bold, accent)
            item_p = QTableWidgetItem(name)
            item_p.setForeground(QBrush(QColor(accent)))
            item_p.setFont(bold_font)
            item_p.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 0, item_p)

            for col, (text, color) in enumerate([
                (nv, text_neutral), (pv, text_muted), (ch, text_muted),
                (unc, text_muted), (unit, text_muted)
            ], start=1):
                item = QTableWidgetItem(text)
                item.setForeground(QBrush(QColor(color)))
                if col <= 4:
                    item.setFont(mono_font)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                table.setItem(row, col, item)

        table.setUpdatesEnabled(True)

        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {Colors.BG_PRIMARY};
                border: 1px solid {border_col};
                border-radius: 4px;
                gridline-color: transparent;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_SECONDARY};
                padding: 8px;
                border: none;
                border-bottom: 2px solid {accent};
                color: {text_neutral};
                font-weight: bold;
            }}
            QTableWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {Colors.SURFACE_BORDER};
            }}
            QTableWidget::item:selected {{
                background-color: {Colors.SURFACE_HOVER};
            }}
        """)
        layout.addWidget(table)

        return layout

    def _show_fit_results(self, result):
        """Show fit results (fallback when no pre-built dialog is cached)."""
        self._prebuild_fit_report(result)
        self._cached_fit_dialog.show()
        self._cached_fit_dialog.raise_()
        self._cached_fit_dialog.activateWindow()

    def on_restart_clicked(self):
        """Handle Restart button click - restore original data including deleted TOAs."""
        if self.original_mjd is not None:
            # Restore all original data
            self.mjd = self.original_mjd.copy()
            self.residuals_us = self.original_residuals_us.copy()
            self.prefit_residuals_us = self.original_residuals_us.copy()
            self.errors_us = self.original_errors_us.copy() if self.original_errors_us is not None else None
            self.deleted_indices = set()
            # Reset cached keep mask (all True = keep all TOAs)
            self._keep_mask = np.ones(len(self.original_mjd), dtype=bool)
            
            # Reset session params to original par file values
            # Without this, subsequent fits would start from previously-fitted
            # params instead of the original model, causing stale/NaN residuals.
            if self.session is not None:
                self.session.params = self.session.get_initial_params()
                self.session._cached_result_by_mode.clear()
                self.session._cached_toa_data = None

            # Clear noise state: realizations, overlay curves, error bars
            for name in list(self._noise_curves):
                self._remove_noise_curve(name)
            self._noise_realizations.clear()

            # Reset noise panel toggles to all off
            if self.noise_panel is not None and self.session is not None:
                self.noise_panel.populate_from_params(self.session.params)

            # Restore residuals + uncertainties visibility
            self.show_residuals = True
            self.show_uncertainties = True
            if hasattr(self, 'noise_panel') and self.noise_panel is not None:
                if hasattr(self.noise_panel, '_show_residuals_cb'):
                    self.noise_panel._show_residuals_cb.blockSignals(True)
                    self.noise_panel._show_residuals_cb.setChecked(True)
                    self.noise_panel._show_residuals_cb.blockSignals(False)
                if hasattr(self.noise_panel, '_show_uncertainties_cb'):
                    self.noise_panel._show_uncertainties_cb.blockSignals(True)
                    self.noise_panel._show_uncertainties_cb.setChecked(True)
                    self.noise_panel._show_uncertainties_cb.blockSignals(False)
            if hasattr(self, '_show_residuals_action'):
                self._show_residuals_action.blockSignals(True)
                self._show_residuals_action.setChecked(True)
                self._show_residuals_action.blockSignals(False)

            # Recalculate RMS using canonical engine stats
            from jug.engine.stats import compute_residual_stats
            stats = compute_residual_stats(self.residuals_us, self.errors_us)
            self._update_rms_from_stats(stats)

            # Reset fit state
            self.is_fitted = False
            self.fit_results = None
            self.fit_report_button.setEnabled(False)
            
            # Update plot (auto-range to fit prefit residuals)
            self._update_plot(auto_range=True)
            self._update_plot_title()

            # Update statistics
            self.rms_label.setText(f"{self.rms_us:.6f} μs")
            self.ntoa_label.setText(f"{len(self.mjd)}")
            self.iter_label.setText("--")
            self.chi2_label.setText("--")
            self.status_bar.showMessage(f"Restarted: restored {len(self.mjd)} TOAs")
        elif self.par_file and self.tim_file:
            self._compute_initial_residuals()
    
    def on_show_parameters(self):
        """Show parameter editor dialog."""
        # TODO: Implement in Phase 3
        self.status_bar.showMessage("Parameter editor not yet implemented (Phase 3)")
    
    def on_toggle_zero_line(self, checked):
        """Toggle zero reference line visibility."""
        self.show_zero_line = checked
        if checked:
            # Add zero line if not present
            if self.zero_line is None:
                self.zero_line = create_zero_line(self.plot_widget)
        else:
            # Remove zero line if present
            if self.zero_line is not None:
                self.plot_widget.removeItem(self.zero_line)
                self.zero_line = None

    def _on_toggle_noise_panel(self, checked):
        """Toggle noise panel visibility from menu."""
        self._set_noise_panel_visible(checked)

    def _set_noise_panel_visible(self, visible: bool):
        """Show or hide the noise panel, resizing the window to keep the plot the same size."""
        if self.noise_panel is None:
            return
        was_visible = self.noise_panel.isVisible()
        if visible == was_visible:
            return

        panel_width = self.noise_panel.minimumWidth()  # 220

        if visible:
            geo = self.geometry()
            new_x = max(0, geo.x() - panel_width)
            self.setGeometry(new_x, geo.y(), geo.width() + panel_width, geo.height())
            self.setMinimumSize(self.minimumWidth() + panel_width, self.minimumHeight())
        else:
            geo = self.geometry()
            min_w = self.minimumWidth()
            self.setMinimumSize(max(1152, min_w - panel_width), self.minimumHeight())
            self.setGeometry(geo.x() + panel_width, geo.y(),
                             geo.width() - panel_width, geo.height())

        self.noise_panel.setVisible(visible)

        # Toggle "Noise" button in title bar (visible when panel is hidden)
        if hasattr(self, '_noise_title_btn'):
            self._noise_title_btn.setVisible(not visible)

        # Sync menu checkboxes
        if hasattr(self, 'noise_panel_action'):
            self.noise_panel_action.setChecked(visible)
        if hasattr(self, '_view_overlay'):
            self._view_overlay.set_item_checked("noise_panel", visible)

    def _on_noise_realise_changed(self, process_name: str, show: bool):
        """Handle toggling noise realization overlay for a process.

        When toggled on, overlays the GLS realization as a curve on the plot.
        When toggled off, removes the curve.
        Uses GLS realizations from the fit if available, falls back to
        Wiener filter MAP estimate.
        """
        if not show:
            self._remove_noise_curve(process_name)
            self.status_bar.showMessage(f"Removed {process_name} overlay")
            return

        if self.residuals_us is None or self.session is None:
            return

        # Prefer GLS realization from fit (stored in _noise_realizations after fit)
        realization_us = self._noise_realizations.get(process_name)

        # Fallback to Wiener filter if no GLS realization available
        if realization_us is None:
            realization_us = self._compute_noise_realization_fallback(process_name)

        if realization_us is None:
            self.status_bar.showMessage(
                f"No {process_name} realization available — run fit first or check parameters"
            )
            return

        # Store for overlay rendering (won't overwrite GLS if it exists)
        if process_name not in self._noise_realizations:
            self._noise_realizations[process_name] = realization_us

        self._update_plot(auto_range=False)
        self.status_bar.showMessage(f"Showing {process_name} realization")

    def _on_noise_subtract_changed(self, process_name: str, subtract: bool):
        """Handle subtracting/restoring a noise realization from residuals.

        This modifies the displayed residuals in-place (like tempo2 Shift+K / Shift+F).
        Requires the realization to already be computed (via "Realise" toggle).
        Also updates displayed RMS / wRMS / χ² to reflect the new residuals.
        """
        realization_us = self._noise_realizations.get(process_name)
        if realization_us is None or self.residuals_us is None:
            self.status_bar.showMessage(f"No {process_name} realization to subtract")
            return

        if subtract:
            self.residuals_us = self.residuals_us - realization_us
        else:
            self.residuals_us = self.residuals_us + realization_us

        # Recompute and display statistics on the current residuals
        self._refresh_stats_from_residuals()

        self._update_plot(auto_range=True)
        self._update_plot_title()
        self.status_bar.showMessage(
            f"{'Subtracted' if subtract else 'Restored'} {process_name}"
        )

    def _compute_noise_realization_fallback(self, process_name: str):
        """Wiener filter fallback when GLS realization isn't available."""
        params = self.session.params
        mjd = self.mjd
        current_res_us = self.residuals_us.copy()
        errors_us = self.errors_us
        if errors_us is None:
            return None

        if process_name == "RedNoise":
            from jug.noise.red_noise import realize_red_noise, parse_red_noise_params
            rn = parse_red_noise_params(params)
            if rn is not None:
                real_sec = realize_red_noise(
                    mjd, current_res_us * 1e-6, errors_us * 1e-6,
                    rn.log10_A, rn.gamma, rn.n_harmonics,
                )
                return real_sec * 1e6

        elif process_name == "DMNoise":
            from jug.noise.red_noise import realize_dm_noise, parse_dm_noise_params
            dm = parse_dm_noise_params(params)
            if dm is not None and self.toa_freqs is not None:
                real_sec = realize_dm_noise(
                    mjd, self.toa_freqs, current_res_us * 1e-6,
                    errors_us * 1e-6, dm.log10_A, dm.gamma, dm.n_harmonics,
                )
                return real_sec * 1e6

        elif process_name == "DMX":
            return self._compute_dmx_realization()

        return None

    # Colour palette for noise realization curves
    _NOISE_CURVE_COLORS = {
        "RedNoise": "#e74c3c",   # red
        "DMNoise":  "#2ecc71",   # green
        "DMX":      "#f39c12",   # amber
    }

    def _update_noise_curves(self, x_data, x_label: str):
        """Sync overlay scatter items and error bars with current noise realizations."""
        # Determine which processes should have visible overlays
        active_realise = set()
        if self.noise_panel is not None:
            active_realise = set(self.noise_panel.get_realise_processes())

        # Remove items for processes no longer realised
        for name in list(self._noise_curves):
            if name not in active_realise:
                self._remove_noise_curve(name)

        if not active_realise:
            return

        for name in active_realise:
            real_us = self._noise_realizations.get(name)
            if real_us is None:
                continue
            err_us = self._noise_realizations.get(f'{name}_err')

            color = self._NOISE_CURVE_COLORS.get(name, "#3498db")
            if name in self._noise_curves:
                self._noise_curves[name].setData(x=x_data, y=real_us)
            else:
                scatter = pg.ScatterPlotItem(
                    x=x_data, y=real_us,
                    size=PlotTheme.SCATTER_SIZE,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(color),
                    pxMode=True,
                    useCache=True,
                    hoverable=False,
                    name=name,
                )
                self.plot_widget.addItem(scatter)
                self._noise_curves[name] = scatter

            # Error bars from GLS posterior covariance
            if err_us is not None:
                if name in self._noise_errorbars:
                    self._noise_errorbars[name].setData(
                        x=x_data, y=real_us, height=2 * err_us,
                    )
                else:
                    errbar = pg.ErrorBarItem(
                        x=x_data, y=real_us, height=2 * err_us,
                        beam=0.0,
                        pen=pg.mkPen(color=color, width=1.5),
                    )
                    self.plot_widget.addItem(errbar)
                    self._noise_errorbars[name] = errbar

    def _remove_noise_curve(self, name: str):
        """Remove noise realization scatter and error bars from the plot."""
        curve = self._noise_curves.pop(name, None)
        if curve is not None:
            self.plot_widget.removeItem(curve)
        errbar = self._noise_errorbars.pop(name, None)
        if errbar is not None:
            self.plot_widget.removeItem(errbar)

    def _toggle_noise_subtract(self, process_name: str):
        """Keyboard shortcut handler: toggle subtract for a noise process.

        If the realization isn't computed yet, compute it first (Wiener fallback).
        """
        if process_name not in self._noise_realizations:
            # Try to compute the realization
            real = self._compute_noise_realization_fallback(process_name)
            if real is not None:
                self._noise_realizations[process_name] = real
            else:
                self.status_bar.showMessage(
                    f"No {process_name} realization available — run fit first"
                )
                return

        # Toggle the subtract state
        key = f"_subtract_active_{process_name}"
        currently_subtracting = getattr(self, key, False)
        self._on_noise_subtract_changed(process_name, not currently_subtracting)
        setattr(self, key, not currently_subtracting)

        # Sync the noise panel button if visible
        if self.noise_panel is not None and process_name in self.noise_panel._rows:
            row = self.noise_panel._rows[process_name]
            if row._subtract_btn is not None:
                row._subtract_btn.setChecked(not currently_subtracting)
                row._subtracting = not currently_subtracting

    def _on_estimate_noise(self, selections):
        """Run MAP estimation in background with selected noise processes."""
        if self.session is None:
            self.status_bar.showMessage("No data loaded")
            if self.noise_panel:
                self.noise_panel.set_estimate_complete(False)
            return

        include_red = selections.get('include_red', True)
        include_dm = selections.get('include_dm', True)
        include_ecorr = selections.get('include_ecorr', False)

        self.status_bar.showMessage("Running MAP noise estimation...")

        # Run in background thread to avoid freezing GUI
        from PySide6.QtCore import QThread, QObject, Signal as QtSignal

        class EstimateWorker(QObject):
            finished = QtSignal(object)  # NoiseEstimateResult or Exception
            progress = QtSignal(str)

            def __init__(self, session, inc_red, inc_dm, inc_ecorr):
                super().__init__()
                self.session = session
                self.inc_red = inc_red
                self.inc_dm = inc_dm
                self.inc_ecorr = inc_ecorr

            def run(self):
                try:
                    from jug.noise.map_estimator import estimate_noise_parameters
                    import numpy as np

                    # Get pre-computed residuals
                    result = self.session.compute_residuals(subtract_tzr=True)
                    residuals_us = result['residuals_us']
                    residuals_sec = residuals_us * 1e-6

                    # Original errors (before EFAC/EQUAD)
                    errors_us = np.array([t.error_us for t in self.session.toas_data])
                    errors_sec = errors_us * 1e-6

                    toas_mjd = np.array([
                        t.mjd_int + t.mjd_frac for t in self.session.toas_data
                    ])
                    freq_mhz = result['freq_bary_mhz']
                    toa_flags = [t.flags for t in self.session.toas_data]

                    est_result = estimate_noise_parameters(
                        residuals_sec=residuals_sec,
                        errors_sec=errors_sec,
                        toas_mjd=toas_mjd,
                        freq_mhz=freq_mhz,
                        toa_flags=toa_flags,
                        params=self.session.params,
                        include_red_noise=self.inc_red,
                        include_dm_noise=self.inc_dm,
                        include_ecorr=self.inc_ecorr,
                        batch_size=1000,
                        max_num_batches=30,
                        patience=3,
                    )
                    self.finished.emit(est_result)
                except Exception as e:
                    self.finished.emit(e)

        self._estimate_thread = QThread()
        self._estimate_worker = EstimateWorker(
            self.session, include_red, include_dm, include_ecorr
        )
        self._estimate_worker.moveToThread(self._estimate_thread)
        self._estimate_thread.started.connect(self._estimate_worker.run)
        self._estimate_worker.finished.connect(self._on_estimate_complete)
        self._estimate_worker.finished.connect(self._estimate_thread.quit)
        self._estimate_thread.start()

    def _on_estimate_complete(self, result):
        """Handle MAP estimation completion."""
        if self.noise_panel:
            self.noise_panel.set_estimate_complete(True)

        if isinstance(result, Exception):
            self.status_bar.showMessage(f"Noise estimation failed: {result}")
            return

        # Update par params with estimated values
        from jug.noise.map_estimator import NoiseEstimateResult
        if isinstance(result, NoiseEstimateResult):
            self.status_bar.showMessage(
                f"Noise estimation complete: {len(result.params)} parameters"
            )
            # Log the estimated parameters
            for key, val in sorted(result.params.items()):
                print(f"  MAP estimate: {key} = {val}")

            # Update session params with estimated noise values
            if self.session is not None:
                for key, val in result.params.items():
                    self.session.params[key] = val

                # Rebuild _noise_lines so populate_from_params can find
                # the estimated EFAC/EQUAD/ECORR values.
                existing = list(self.session.params.get('_noise_lines', []))
                for key, val in result.params.items():
                    if key.startswith('EFAC_'):
                        backend = key[len('EFAC_'):]
                        line = f"EFAC -f {backend} {val}"
                        existing = [l for l in existing
                                    if not (l.strip().startswith(('EFAC','T2EFAC'))
                                            and backend in l)]
                        existing.append(line)
                    elif key.startswith('EQUAD_'):
                        backend = key[len('EQUAD_'):]
                        line = f"EQUAD -f {backend} {val}"
                        existing = [l for l in existing
                                    if not (l.strip().startswith(('EQUAD','T2EQUAD'))
                                            and backend in l)]
                        existing.append(line)
                    elif key.startswith('ECORR_'):
                        backend = key[len('ECORR_'):]
                        line = f"ECORR -f {backend} {val}"
                        existing = [l for l in existing
                                    if not (l.strip().startswith(('ECORR','TNECORR'))
                                            and backend in l)]
                        existing.append(line)
                self.session.params['_noise_lines'] = existing

            # Refresh the noise panel to show updated values
            if self.noise_panel and self.session:
                self.noise_panel.populate_from_params(self.session.params)

    def _on_toggle_residuals(self, checked: bool):
        """Toggle visibility of residuals (scatter points + error bars)."""
        self.show_residuals = checked
        if self.scatter_item is not None:
            self.scatter_item.setVisible(checked)
        if self.error_bar_item is not None:
            self.error_bar_item.setVisible(checked and self.show_uncertainties)
        # Keep View menu action and noise panel checkbox in sync
        if hasattr(self, '_show_residuals_action'):
            self._show_residuals_action.blockSignals(True)
            self._show_residuals_action.setChecked(checked)
            self._show_residuals_action.blockSignals(False)
        if hasattr(self, 'noise_panel') and hasattr(self.noise_panel, '_show_residuals_cb'):
            self.noise_panel._show_residuals_cb.blockSignals(True)
            self.noise_panel._show_residuals_cb.setChecked(checked)
            self.noise_panel._show_residuals_cb.blockSignals(False)

    def _on_toggle_uncertainties(self, checked: bool):
        """Toggle visibility of residual error bars independently."""
        self.show_uncertainties = checked
        if self.error_bar_item is not None:
            self.error_bar_item.setVisible(self.show_residuals and checked)
        # Keep noise panel checkbox in sync
        if hasattr(self, 'noise_panel') and hasattr(self.noise_panel, '_show_uncertainties_cb'):
            self.noise_panel._show_uncertainties_cb.blockSignals(True)
            self.noise_panel._show_uncertainties_cb.setChecked(checked)
            self.noise_panel._show_uncertainties_cb.blockSignals(False)

    def _compute_dmx_realization(self) -> Optional[np.ndarray]:
        """Compute DMX delay contribution at each TOA."""
        if self.session is None:
            return None
        params = self.session.params
        from jug.model.dmx import parse_dmx_ranges
        try:
            ranges = parse_dmx_ranges(params)
        except Exception:
            return None
        if not ranges:
            return None

        mjd = self.mjd
        freq_mhz = self.toa_freqs
        if freq_mhz is None:
            return None

        # DMX delay = DMX_value * K_DM / freq^2 (in seconds) → convert to μs
        K_DM = 4.148808e3  # DM constant in MHz^2 s / pc cm^-3
        dmx_delay_us = np.zeros(len(mjd))
        for r in ranges:
            mask = (mjd >= r.r1_mjd) & (mjd <= r.r2_mjd)
            dmx_val = params.get(f"DMX_{r.index:04d}", 0.0)
            if isinstance(dmx_val, str):
                dmx_val = float(dmx_val)
            dmx_delay_us[mask] = dmx_val * K_DM / (freq_mhz[mask] ** 2) * 1e6

        return dmx_delay_us

    def _toggle_rms_mode(self):
        """Toggle between weighted and unweighted RMS display (shortcut: W)."""
        self.use_weighted_rms = not self.use_weighted_rms
        
        # Update the displayed RMS value based on mode
        if self.use_weighted_rms:
            self.rms_us = self.weighted_rms_us
            rms_type = "weighted"
            label_text = "wRMS"
        else:
            # If unweighted_rms_us is missing/None, compute it on-the-fly
            if self.unweighted_rms_us is None or self.unweighted_rms_us == 0.0:
                if self.residuals_us is not None and len(self.residuals_us) > 0:
                    self.unweighted_rms_us = float(np.std(self.residuals_us))
            self.rms_us = self.unweighted_rms_us
            rms_type = "unweighted"
            label_text = "RMS"
        
        # Update label text in statistics panel
        self.rms_label_text.setText(label_text)
        
        # Update displayed value
        if self.rms_us is not None:
            self.rms_label.setText(f"{self.rms_us:.6f} μs")
        
        # Update plot title
        self._update_plot_title()
        
        # Show status message
        if self.rms_us is not None:
            self.status_bar.showMessage(f"Switched to {rms_type} RMS: {self.rms_us:.6f} μs (press W to toggle)")
        else:
            self.status_bar.showMessage(f"Switched to {rms_type} RMS (press W to toggle)")

    def _update_rms_from_stats(self, stats: dict):
        """Update all RMS values from a stats dictionary.
        
        Parameters
        ----------
        stats : dict
            Dictionary containing 'weighted_rms_us' and 'unweighted_rms_us' keys.
        """
        self.weighted_rms_us = stats['weighted_rms_us']
        self.unweighted_rms_us = stats['unweighted_rms_us']
        # Set displayed RMS based on current mode
        self.rms_us = self.weighted_rms_us if self.use_weighted_rms else self.unweighted_rms_us

    def _update_rms_from_result(self, result: dict):
        """Update all RMS values from a compute_residuals result dictionary.
        
        Parameters
        ----------
        result : dict
            Dictionary from compute_residuals_simple containing RMS values.
        """
        # compute_residuals_simple returns 'weighted_rms_us' and 'unweighted_rms_us'
        self.weighted_rms_us = result.get('weighted_rms_us', result.get('rms_us', 0.0))
        self.unweighted_rms_us = result.get('unweighted_rms_us', 0.0)
        # Set displayed RMS based on current mode
        self.rms_us = self.weighted_rms_us if self.use_weighted_rms else self.unweighted_rms_us

    def _refresh_stats_from_residuals(self):
        """Recompute RMS / wRMS / χ² from the current ``self.residuals_us`` and update labels."""
        if self.residuals_us is None:
            return
        r = self.residuals_us
        self.unweighted_rms_us = float(np.std(r))
        if self.errors_us is not None and len(self.errors_us) == len(r):
            w = 1.0 / (self.errors_us ** 2)
            wmean = np.sum(w * r) / np.sum(w)
            self.weighted_rms_us = float(np.sqrt(np.sum(w * (r - wmean) ** 2) / np.sum(w)))
        else:
            self.weighted_rms_us = self.unweighted_rms_us
        self.rms_us = self.weighted_rms_us if self.use_weighted_rms else self.unweighted_rms_us
        if self.rms_us is not None:
            self.rms_label.setText(f"{self.rms_us:.6f} μs")
        # Update χ²/dof
        if self.errors_us is not None:
            n_toas = len(r)
            n_params = len(self.param_checkboxes) if hasattr(self, 'param_checkboxes') else 0
            n_fit = sum(1 for cb in self.param_checkboxes.values() if cb.isChecked()) if n_params else 1
            dof = max(n_toas - n_fit, 1)
            chi2 = float(np.sum((r / self.errors_us) ** 2))
            chi2_dof = chi2 / dof
            self.chi2_label.setText(f"{chi2_dof:.2f}")

    # -- Phase 4.4: Color-by-backend/frequency --------------------------

    def _set_color_mode(self, mode: str):
        """Set coloring mode: 'none', 'backend', 'frequency'."""
        self._color_mode = mode

        # Check corresponding action in group
        if self.is_remote:
            # Update overlay submenu check states
            mode_map = {"none": "None (single color)",
                        "backend": "Backend/Receiver",
                        "frequency": "Frequency (continuous)"}
            for m, label in mode_map.items():
                self._view_overlay.set_item_checked(f"color_by_{label}", m == mode)
        else:
            if mode == "none":
                self._color_none_action.setChecked(True)
            elif mode == "backend":
                self._color_backend_action.setChecked(True)
            elif mode == "frequency":
                self._color_freq_action.setChecked(True)

        # Re-plot to apply changes efficiently
        if self.scatter_item is not None:
             self._update_plot(auto_range=False)

    # Renamed from _apply_point_colors
    def _get_point_brushes(self, n_points):
        """Return list of brushes for current coloring mode, or None for default."""
        if not hasattr(self, 'scatter_item') or self.scatter_item is None:
            return None

        mode = self._color_mode
        brushes = []
        
        if mode == "backend" and self.toa_flags is not None:
            # Backend coloring: use 'f' flag
            # We need a stable mapping of backend string -> color
            # Use 'f' flag or 'i' (instrument) or 'be' (backend)?
            # Standard is '-f' usually for frontend/backend.
            
            # Map unique backends to colors
            # Get list of flags for all TOAs
            flags_list = self.toa_flags # List of dicts
            
            # Extract backend for each point
            # This might be slow if loop.
            # But we only do it when switching mode.
            
            from jug.gui.theme import Colors
            # 8 distinct colors
            palette = [
                Colors.ACCENT_PRIMARY, Colors.ACCENT_SECONDARY,
                "#4CAF50", "#FFC107", "#9C27B0", "#FF5722", "#00BCD4", "#795548"
            ]
            
            backend_map = {}
            next_color_idx = 0
            
            for flags in flags_list:
                be = flags.get('f', flags.get('be', flags.get('i', 'unknown')))
                if be not in backend_map:
                    backend_map[be] = pg.mkBrush(palette[next_color_idx % len(palette)])
                    next_color_idx += 1
                brushes.append(backend_map[be])
            
            # Match length
            if len(brushes) != n_points:
                if len(brushes) > n_points:
                    brushes = brushes[:n_points]
                else:
                    brushes.extend([pg.mkBrush(None)] * (n_points - len(brushes)))

            self._update_freq_legend(False)
            return brushes

        elif mode == "frequency" and self.toa_freqs is not None:
            freqs = self.toa_freqs
            
            # Handle empty/NaN
            if len(freqs) == 0:
                self._update_freq_legend(False)
                return None
                
            f_min, f_max = float(np.nanmin(freqs)), float(np.nanmax(freqs))
            span = f_max - f_min if f_max > f_min else 1.0
            
            # [0, 0, 0.5] -> [1, 1, 0] ?
            # Let's use a better map. Plasma or Viridis.
            # Simple 3-point gradient: Purple -> Red -> Yellow
            # 0.0: (68, 1, 84)   (Purple)
            # 0.5: (193, 60, 100) (Reddish)
            # 1.0: (253, 231, 37) (Yellow)
            
            brushes = []
            for f in freqs:
                t = np.clip((f - f_min) / span, 0.0, 1.0)
                if t < 0.5:
                    # Purple -> Red
                    tt = t * 2.0
                    r = int(68 + tt * (193 - 68))
                    g = int(1 + tt * (60 - 1))
                    b = int(84 + tt * (100 - 84))
                else:
                    # Red -> Yellow
                    tt = (t - 0.5) * 2.0
                    r = int(193 + tt * (253 - 193))
                    g = int(60 + tt * (231 - 60))
                    b = int(100 + tt * (37 - 100))
                brushes.append(pg.mkBrush(r, g, b, 200))
            # Check if we have points to color
            if self.scatter_item.data is None or len(self.scatter_item.data) == 0:
                return None # Return None if no data

            # Ensure brushes match data length (handle race condition or filtered data)
            # n_points is passed in now to be safe
            if len(brushes) != n_points:
                # If we have a mismatch, maybe we are coloring based on full dataset but plotting filtered/downsampled?
                # For now, just slice or skip to avoid crash
                if len(brushes) > n_points:
                    brushes = brushes[:n_points]
                else:
                     # Not enough brushes? Fallback to default for remainder
                     brushes.extend([pg.mkBrush(None)] * (n_points - len(brushes)))
            
            # Show legend
            self._update_freq_legend(True, f_min, f_max)
            return brushes

        # mode == "none" → reset to theme default
        self._update_freq_legend(False)
        return None  # Returning None means "use default symbolBrush"



    def _update_freq_legend(self, show, f_min=0.0, f_max=0.0):
        """Update the frequency pseudo-colorbar legend using SimpleColorBar."""
        # Create if not exists
        if not hasattr(self, 'freq_legend'):
            self.freq_legend = SimpleColorBar(size=(120, 12), offset=(20, 20))
            # Min/max labels are sufficient?
            
            # Add to SCENE directly to avoid ViewBox clipping/scaling issues
            # Only if scene exists (it should if widget is shown)
            scene = self.plot_widget.scene()
            if scene:
                scene.addItem(self.freq_legend)
            
            # Connect range changes to update position (keep top-right)
            def update_pos(*args):
                # Ensure it's in the scene
                if self.freq_legend.scene() is None and self.plot_widget.scene() is not None:
                     self.plot_widget.scene().addItem(self.freq_legend)
                     
                view_box = self.plot_widget.plotItem.getViewBox()
                self.freq_legend.setAnchor(view_box)
            
            # Keep reference to slot to avoid GC?
            self._freq_legend_updater = update_pos
            view_box = self.plot_widget.plotItem.getViewBox()
            view_box.sigStateChanged.connect(update_pos)
            self.plot_widget.plotItem.sigRangeChanged.connect(update_pos)
            # Also update when geometry changes (resize)
            self.plot_widget.plotItem.geometryChanged.connect(update_pos)

        if show:
            # Ensure it is in the scene
            if self.freq_legend.scene() is None and self.plot_widget.scene() is not None:
                self.plot_widget.scene().addItem(self.freq_legend)
            
            self.freq_legend.setVisible(True)
            self.freq_legend.setLabels(f"{int(f_min)} MHz", f"{int(f_max)} MHz")
            # Force update position immediately
            self.freq_legend.setAnchor(self.plot_widget.plotItem.getViewBox())
        else:
            if hasattr(self, 'freq_legend'):
                self.freq_legend.setVisible(False)
                # Remove from scene to stop any processing/events
                if self.freq_legend.scene() is not None:
                    self.plot_widget.scene().removeItem(self.freq_legend)

    def _set_menu_item_text(self, action_attr, overlay_name, text):
        """Set menu item text in either QAction (local) or overlay (remote) mode."""
        if self.is_remote:
            self._view_overlay.set_item_text(overlay_name, text)
        else:
            getattr(self, action_attr).setText(text)

    def on_toggle_color_variant(self):
        """Toggle data color variant (navy/burgundy in light, classic/scilab in dark)."""
        if is_dark_mode():
            new_variant = toggle_synthwave_variant()
            if new_variant == "scilab":
                self._set_menu_item_text("variant_action", "variant", "Data Color: Cyan/Pink")
            else:
                self._set_menu_item_text("variant_action", "variant", "Data Color: Pink/Cyan")
        else:
            new_variant = toggle_light_variant()
            if new_variant == "burgundy":
                self._set_menu_item_text("variant_action", "variant", "Data Color: Burgundy")
            else:
                self._set_menu_item_text("variant_action", "variant", "Data Color: Navy")

        self._apply_theme()

    def on_toggle_theme(self):
        """Toggle between light and dark (Synthwave) themes."""
        if is_dark_mode():
            set_theme(LightTheme)
            self._set_menu_item_text("theme_action", "theme", "Theme: Light")
            variant = get_light_variant()
            if variant == "burgundy":
                self._set_menu_item_text("variant_action", "variant", "Data Color: Burgundy")
            else:
                self._set_menu_item_text("variant_action", "variant", "Data Color: Navy")
        else:
            set_theme(SynthwaveTheme)
            self._set_menu_item_text("theme_action", "theme", "Theme: Synthwave '84")
            variant = get_synthwave_variant()
            if variant == "scilab":
                self._set_menu_item_text("variant_action", "variant", "Data Color: Cyan/Pink")
            else:
                self._set_menu_item_text("variant_action", "variant", "Data Color: Pink/Cyan")

        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme to all UI elements.
        
        OPTIMIZED: Suppresses widget repaints during batch stylesheet updates,
        then triggers a single repaint at the end.
        """
        # Suppress all intermediate repaints during theme switch
        self.setUpdatesEnabled(False)
        
        try:
            # ONE stylesheet update for all widgets (objectName selectors handle specifics)
            self.setStyleSheet(get_main_stylesheet())

            # Update pyqtgraph plot widget (doesn't use QSS)
            configure_plot_widget(self.plot_widget)
            
            # Explicitly style plot containers (force theme application)
            if hasattr(self, 'plot_container'):
                 self.plot_container.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")
                 
            if hasattr(self, 'plot_frame'):
                 self.plot_frame.setStyleSheet(f"background-color: {Colors.PLOT_BG}; border: 1px solid {get_border_strong()}; border-radius: 12px;")
                 
            if hasattr(self, 'plot_title_label'):
                 self.plot_title_label.setStyleSheet(get_plot_title_style())

            # Update control panel (needs get_control_panel_style for the panel widget)
            if hasattr(self, 'control_container'):
                 panel = self.control_container.findChild(QWidget)
                 if panel: panel.setStyleSheet(get_control_panel_style())

            # Update primary/secondary buttons (need special gradient styling in dark mode)
            self.fit_button.setStyleSheet(get_primary_button_style())
            secondary_style = get_secondary_button_style()
            self.fit_report_button.setStyleSheet(secondary_style)
            self.reset_button.setStyleSheet(secondary_style)
            self.fit_window_button.setStyleSheet(secondary_style)
            self.params_drawer_btn.setStyleSheet(secondary_style)

            # Update params header (section title style)
            if hasattr(self, 'params_header_label'):
                 self.params_header_label.setStyleSheet(get_section_title_style())

            # Update Parameter Drawer Background & Table
            if hasattr(self, 'params_drawer'):
                self.params_drawer.setStyleSheet(f"background-color: {Colors.BG_SECONDARY}; border-left: 1px solid {get_border_subtle()};")
                
            if hasattr(self, 'param_table'):
                 # Batch checkbox style — build string once, apply to all
                 checkbox_style = f"""
                    QCheckBox {{
                        color: {Colors.TEXT_PRIMARY};
                        padding: 4px 8px;
                        spacing: 8px;
                        font-size: 13px;
                    }}
                    QCheckBox::indicator {{
                        width: 16px; height: 16px;
                    }}
                 """
                 self.param_table.setStyleSheet(f"""
                    QTableWidget {{
                        background-color: {Colors.BG_SECONDARY};
                        border: none;
                    }}
                    QTableWidget::item {{
                        padding: 0px;
                        border: none;
                    }}
                """)
                 # Update styling of all checkboxes in the table
                 for row in range(self.param_table.rowCount()):
                     widget = self.param_table.cellWidget(row, 0)
                     if widget:
                         widget.setStyleSheet(checkbox_style)
            
            # Update Lights On Switch Labels
            if hasattr(self, 'lbl_lights'):
                text_style = f"border: none; color: {Colors.TEXT_PRIMARY}; font-weight: {Typography.WEIGHT_BOLD}; font-size: 13px;"
                self.lbl_lights.setStyleSheet(text_style)
                
                is_on = self.lights_on_switch.isChecked()
                self.lbl_off.setStyleSheet(f"border: none; color: {Colors.TEXT_MUTED if is_on else Colors.TEXT_PRIMARY}; font-weight: {'normal' if is_on else 'bold'}; font-size: 13px;")
                self.lbl_on.setStyleSheet(f"border: none; color: {Colors.TEXT_PRIMARY if is_on else Colors.TEXT_MUTED}; font-weight: {'bold' if is_on else 'normal'}; font-size: 13px;")

            # Update scatter plot colors (pyqtgraph, not QSS)
            colors = get_scatter_colors()
            if self.scatter_item is not None:
                self.scatter_item.setBrush(pg.mkBrush(*colors['primary']))
                
            # Update error bar colors (pyqtgraph)
            if self.error_bar_item is not None:
                 error_color = PlotTheme.get_error_bar_color()
                 self.error_bar_item.setOpts(pen=pg.mkPen(color=error_color, width=2.0))
                 
            # Update zero line if visible (pyqtgraph)
            if self.zero_line is not None:
                self.plot_widget.removeItem(self.zero_line)
                self.zero_line = create_zero_line(self.plot_widget)

            # Update remote overlay menus if present
            if hasattr(self, '_remote_menu_bar'):
                self._remote_menu_bar.setStyleSheet(f"""
                    QFrame#remoteMenuBar {{
                        background-color: {Colors.BG_SECONDARY};
                        border-bottom: 1px solid {get_border_subtle()};
                        padding: 0px;
                    }}
                """)
                for overlay in [self._file_overlay, self._view_overlay,
                                self._tools_overlay, self._help_overlay]:
                    overlay.update_theme()
        finally:
            # Single repaint for the entire window
            self.setUpdatesEnabled(True)

    def on_zoom_fit(self):
        """Zoom plot to fit data."""
        # Temporarily hide items that should stay hidden so autoRange
        # computes bounds only from visible data.
        hidden_items = []
        if not self.show_residuals:
            for item in (self.scatter_item, self.error_bar_item):
                if item is not None:
                    hidden_items.append(item)
        elif not self.show_uncertainties:
            if self.error_bar_item is not None:
                hidden_items.append(self.error_bar_item)

        self.plot_widget.autoRange()

        # Re-enforce visibility state
        if self.scatter_item is not None:
            self.scatter_item.setVisible(self.show_residuals)
        if self.error_bar_item is not None:
            self.error_bar_item.setVisible(self.show_residuals and self.show_uncertainties)

    def _set_solver_mode(self, mode: str, display_name: str):
        """Set the solver mode from dropdown menu selection."""
        self.solver_mode = mode
        self.solver_button.setText(f"  {display_name} ▾")
        self.status_bar.showMessage(f"Solver mode: {display_name}")

    def _on_view_range_changed(self, view_box, ranges):
        """Handle view range changes to scale error bar caps.

        OPTIMIZED: Uses throttling to coalesce rapid updates during pan/zoom.
        Only updates beam width, NOT the x/y/height arrays (O(1) vs O(N)).
        This makes panning/zooming smooth, especially over X11.
        
        Implements "interaction mode": during continuous drag/zoom, only the
        beam update runs (throttled). A final cleanup runs when interaction ends.
        """
        if self.error_bar_item is None or self.mjd is None:
            return
        # Skip beam updates when error bars are hidden
        if not (self.show_residuals and self.show_uncertainties):
            return

        # Mark interaction as active (will be cleared after idle period)
        self._interaction_active = True

        # Get the x-axis range
        x_range = ranges[0]
        x_span = x_range[1] - x_range[0]

        if x_span <= 0:
            return

        # Calculate beam size as a fraction of the visible x range
        # Target: caps should be about 0.5% of visible width
        beam_fraction = 0.005
        new_beam = x_span * beam_fraction

        # Clamp to reasonable bounds (in MJD days)
        min_beam = 0.001  # Minimum 0.001 days (~1.4 minutes) for tight zoom
        max_beam = 50.0   # Maximum 50 days
        new_beam = max(min_beam, min(max_beam, new_beam))

        # THROTTLED UPDATE: Store pending beam and schedule update
        # This coalesces rapid updates during mouse drag (30 FPS = 33ms)
        self._pending_beam = new_beam

        if self._beam_update_timer is None:
            self._beam_update_timer = QTimer()
            self._beam_update_timer.setSingleShot(True)
            self._beam_update_timer.timeout.connect(self._apply_pending_beam)

        # Restart timer on each range change (coalesces updates)
        if not self._beam_update_timer.isActive():
            self._beam_update_timer.start(33)  # ~30 FPS, safe for X11

        # Schedule interaction end check (runs final cleanup when drag stops)
        self._schedule_interaction_end()

    def _schedule_interaction_end(self):
        """Schedule a check to detect when interaction (pan/zoom) has ended.
        
        When no range changes occur for 150ms, we consider interaction complete
        and run any deferred cleanup (e.g., precise beam calculation).
        """
        if self._interaction_end_timer is None:
            self._interaction_end_timer = QTimer()
            self._interaction_end_timer.setSingleShot(True)
            self._interaction_end_timer.timeout.connect(self._on_interaction_end)

        # Restart the timer on each range change (extends deadline)
        self._interaction_end_timer.stop()
        self._interaction_end_timer.start(150)  # 150ms idle = interaction ended

    def _on_interaction_end(self):
        """Called when pan/zoom interaction has ended (no events for 150ms).
        
        Runs final cleanup: apply any pending beam update with exact value.
        """
        self._interaction_active = False
        
        # Apply any final pending beam update immediately
        if self._pending_beam is not None and self.error_bar_item is not None:
            self.error_bar_item.setOpts(beam=self._pending_beam)
            self._pending_beam = None

    def _apply_pending_beam(self):
        """Apply the pending beam width update to error bars.

        OPTIMIZED: Uses setOpts() to update only the beam width,
        NOT the x/y/height arrays. This is O(1) vs O(N).
        """
        if self._pending_beam is None or self.error_bar_item is None:
            return

        # Update ONLY the beam width - do NOT re-upload x/y/height arrays!
        # ErrorBarItem.setOpts() updates rendering options without array copies
        self.error_bar_item.setOpts(beam=self._pending_beam)
        self._pending_beam = None

    def on_about(self):
        """Show about dialog with modern styling."""
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("About JUG")
        msg.setTextFormat(Qt.RichText)
        accent = get_dynamic_accent_primary()
        msg.setText(
            f"""
            <div style='text-align: center;'>
                <h2 style='color: {accent}; margin-bottom: 8px;'>✦ JUG Timing Analysis</h2>
                <p style='color: {Colors.TEXT_SECONDARY}; font-size: 14px;'>
                    JAX-based pulsar timing software
                </p>
                <p style='color: {Colors.TEXT_MUTED}; font-size: 13px; margin-top: 16px;'>
                    Fast · Independent · Extensible
                </p>
                <hr style='border: none; border-top: 1px solid {Colors.SURFACE_BORDER}; margin: 16px 0;'>
                <p style='font-size: 12px;'>
                    <b>Version:</b> 0.5.0 (GUI Phase 2)<br>
                    <b>Framework:</b> PySide6 + PyQtGraph
                </p>
            </div>
            """
        )
        msg.exec()

    def load_files_from_args(self, par_file: str | None, tim_file: str | None):
        """
        Load .par and .tim files from command-line arguments.

        Parameters
        ----------
        par_file : str or None
            Path to .par file
        tim_file : str or None
            Path to .tim file
        """
        if par_file:
            par_path = Path(par_file).resolve()
            if par_path.exists():
                self.par_file = par_path
                self.status_bar.showMessage(f"Loaded .par: {self.par_file.name}")
                self._update_available_parameters()
            else:
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Cannot find .par file:\n{par_file}"
                )
                return

        if tim_file:
            tim_path = Path(tim_file).resolve()
            if tim_path.exists():
                self.tim_file = tim_path
                self.status_bar.showMessage(f"Loaded .tim: {self.tim_file.name}")
            else:
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"Cannot find .tim file:\n{tim_file}"
                )
                return

        # Compute residuals if both files are loaded
        if self.par_file and self.tim_file:
            self._check_ready_to_compute()

    def _parse_par_file_parameters(self):
        """
        Parse .par file to extract available fittable parameters and fit flags.

        The fit flag is the ``1`` between the value and uncertainty in a
        par-file line.  When present it means "fit this parameter by default".

        Returns
        -------
        dict
            Mapping of parameter name → ``True`` if the fit flag is set,
            ``False`` otherwise.  Only includes parameters recognised as
            fittable by the parameter registry.
        """
        if not self.par_file:
            return {}

        # All fittable parameters from the registry (auto-syncs with derivatives)
        from jug.model.parameter_spec import list_fittable_params
        fittable_params = list_fittable_params()

        found_params = {}  # param_name -> has_fit_flag

        try:
            with open(self.par_file, 'r') as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('#'):
                        continue

                    parts = line_stripped.split()
                    if parts:
                        param_name = parts[0]
                        if param_name in fittable_params:
                            # Detect fit flag: format is PARAM value 1 uncertainty
                            # The '1' at parts[2] is the fit indicator
                            has_fit_flag = False
                            if len(parts) >= 3:
                                try:
                                    flag = int(parts[2])
                                    has_fit_flag = (flag == 1)
                                except ValueError:
                                    pass
                            found_params[param_name] = has_fit_flag

        except Exception as e:
            print(f"Error parsing par file: {e}")
            return {}

        return found_params
    
    def _load_initial_parameter_values(self):
        """
        Load initial parameter values from .par file.
        
        Stores values in self.initial_params for comparison in fit results.
        """
        from jug.io.par_reader import parse_par_file
        
        if not self.par_file:
            return
        
        try:
            params = parse_par_file(self.par_file)
            self.initial_params = dict(params)
            
            # Extract pulsar name (PSRJ or PSR)
            self.pulsar_name = params.get('PSRJ', params.get('PSR', 'Unknown'))
        except Exception as e:
            print(f"Error loading initial parameter values: {e}")
            self.initial_params = {}
            self.pulsar_name = 'Unknown'

    def _update_available_parameters(self):
        """Update the parameter checkboxes based on available parameters in par file."""
        # Parse par file to get available parameters and fit flags
        params_in_file = self._parse_par_file_parameters()  # dict: name -> has_fit_flag
        
        # Load initial parameter values from par file
        self._load_initial_parameter_values()

        # Add command-line fit parameters (append any not already in par file)
        # Preserve par file order, then append any extra cmdline params at the end
        all_params = list(params_in_file)
        for p in self.cmdline_fit_params:
            if p not in all_params:
                all_params.append(p)

        # Clear existing checkboxes and layout
        # Clear existing table content
        self.param_table.setRowCount(0)
        self.param_checkboxes.clear()
        # param_items is no longer needed
        if hasattr(self, 'param_items'): self.param_items.clear()
        
        # Set new row count
        self.param_table.setRowCount(len(all_params))
        
        # Populate table with QCheckBox widgets (restores desired look & feel)
        for row, param in enumerate(all_params):
            checkbox = QCheckBox(param)
            
            # Phase 4.1: Determine initial check state from fit flags
            should_check = False
            if param in self.cmdline_fit_params:
                # Command-line --fit override always wins
                should_check = True
            elif param in params_in_file and params_in_file[param]:
                # Par file has fit flag "1" for this parameter
                should_check = True
            elif not self.cmdline_fit_params and not any(params_in_file.values()):
                # Fallback: if no fit flags at all and no --fit args, default F0/F1
                if param in ['F0', 'F1']:
                    should_check = True
            
            checkbox.setChecked(should_check)

            # Apply styling
            style_base = f"""
                QCheckBox {{
                    padding: 4px 8px;
                    spacing: 8px;
                    font-size: 13px;
                }}
                QCheckBox::indicator {{
                    width: 16px;
                    height: 16px;
                }}
            """
            
            if param not in params_in_file:
                # Highlight added parameters
                checkbox.setToolTip(f"{param} not in original .par file (will be fitted from scratch)")
                checkbox.setStyleSheet(style_base + f"QCheckBox {{ color: {Colors.ACCENT_WARNING}; }}")
            else:
                checkbox.setStyleSheet(style_base + f"QCheckBox {{ color: {Colors.TEXT_PRIMARY}; }}")

            # Connect signal
            checkbox.toggled.connect(self._on_param_manual_toggle)
            
            self.param_table.setCellWidget(row, 0, checkbox)
            self.param_checkboxes[param] = checkbox
            
        # No need to block/unblock signals on table itself since widgets handle signals

        self.available_params = all_params

        # Update status — report fit flags detected
        if all_params:
            n_fit_flagged = sum(1 for v in params_in_file.values() if v)
            status_msg = f"Found {len(params_in_file)} parameters in .par"
            if n_fit_flagged > 0:
                status_msg += f" ({n_fit_flagged} fit-flagged)"
            if self.cmdline_fit_params:
                added = [p for p in self.cmdline_fit_params if p not in params_in_file]
                if added:
                    status_msg += f" (+ {len(added)} from --fit)"
            self.status_bar.showMessage(status_msg)

    # =========================================================================
    # BOX ZOOM FEATURE
    # =========================================================================

    def eventFilter(self, obj, event):
        """Handle key events for box zoom, box delete, unzoom, and overlay dismiss."""
        # Click-outside-to-close for remote overlay menus
        if event.type() == QEvent.MouseButtonPress:
            active = getattr(self, '_active_menu_overlay', None)
            if active and active.isVisible():
                # Map click position to the overlay's parent coordinate space
                click_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
                if hasattr(obj, 'mapTo'):
                    local_pos = obj.mapTo(self.centralWidget(), click_pos)
                else:
                    local_pos = click_pos
                if not active.geometry().contains(local_pos):
                    active.close_menu()
                    self._active_menu_overlay = None

        if event.type() == QEvent.KeyPress:
            modifiers = event.modifiers()
            key = event.key()

            # Close overlay menus on Escape
            active = getattr(self, '_active_menu_overlay', None)
            if active and active.isVisible() and key == Qt.Key_Escape:
                active.close_menu()
                self._active_menu_overlay = None
                return True

            if key == Qt.Key_Z:
                if modifiers & Qt.ShiftModifier:
                    # Shift+Z for box delete
                    self._handle_box_delete_key()
                else:
                    # Z for box zoom
                    self._handle_box_zoom_key()
                return True
            elif key == Qt.Key_U:
                self.on_zoom_fit()
                return True
            elif key == Qt.Key_W:
                self._toggle_rms_mode()
                return True
        return super().eventFilter(obj, event)

    def _handle_box_zoom_key(self):
        """Handle 'z' key press for box zoom."""
        if self.box_delete_active:
            return  # Don't interfere with delete mode
        if not self.box_zoom_active:
            # Start box zoom mode and set first corner at current mouse position
            self._start_box_zoom()
        else:
            # Complete box zoom
            self._complete_box_zoom()

    def _start_box_zoom(self):
        """Start box zoom mode - set first corner at current mouse position."""
        self.box_zoom_active = True
        
        # Get current mouse position in data coordinates
        view_box = self.plot_widget.getPlotItem().getViewBox()
        
        # Get mouse position relative to the plot
        global_pos = QCursor.pos()
        widget_pos = self.plot_widget.mapFromGlobal(global_pos)
        scene_pos = self.plot_widget.mapToScene(widget_pos)
        
        # Check if mouse is within the plot area
        if view_box.sceneBoundingRect().contains(scene_pos):
            mouse_point = view_box.mapSceneToView(scene_pos)
            self.box_zoom_start = (mouse_point.x(), mouse_point.y())
            self._create_zoom_rect(mouse_point.x(), mouse_point.y())
            self.status_bar.showMessage("Box zoom: Move mouse to select region, press 'z' to zoom (Esc to cancel)")
        else:
            # Mouse not in plot - wait for it to enter
            self.box_zoom_start = None
            self.status_bar.showMessage("Box zoom: Move mouse into plot and press 'z' to set corner (Esc to cancel)")
        
        # Connect mouse events on the plot
        view_box.scene().sigMouseMoved.connect(self._on_box_zoom_move)
        
        # Change cursor to crosshair
        self.plot_widget.setCursor(QCursor(Qt.CrossCursor))

    def _on_box_zoom_move(self, scene_pos):
        """Handle mouse move during box zoom to update selection rectangle."""
        if not self.box_zoom_active:
            return
        
        view_box = self.plot_widget.getPlotItem().getViewBox()
        
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
            
        mouse_point = view_box.mapSceneToView(scene_pos)
        
        # If we don't have a start point yet, set it now (mouse entered plot)
        if self.box_zoom_start is None:
            self.box_zoom_start = (mouse_point.x(), mouse_point.y())
            self._create_zoom_rect(mouse_point.x(), mouse_point.y())
            self.status_bar.showMessage("Box zoom: Move mouse to select region, press 'z' to zoom (Esc to cancel)")
        else:
            # Update the zoom rectangle visualization
            self._update_zoom_rect(mouse_point.x(), mouse_point.y())

    def _create_zoom_rect(self, x, y):
        """Create the visual zoom selection rectangle."""
        # Use a semi-transparent rectangle
        # We'll create it as a simple polygon or use LinearRegionItems
        
        # Create horizontal and vertical region items for the box
        accent = get_dynamic_accent_primary()
        
        # Store for later - we'll use a simple ROI approach
        self.box_zoom_rect = pg.RectROI(
            [x, y], [0, 0],
            pen=pg.mkPen(accent, width=2, style=Qt.DashLine),
            movable=False,
            resizable=False
        )
        self.box_zoom_rect.removeHandle(0)  # Remove the resize handle
        self.plot_widget.addItem(self.box_zoom_rect)

    def _update_zoom_rect(self, x, y):
        """Update the zoom rectangle as mouse moves."""
        if self.box_zoom_rect is None or self.box_zoom_start is None:
            return
            
        x0, y0 = self.box_zoom_start
        
        # Calculate width and height (can be negative)
        width = x - x0
        height = y - y0
        
        # RectROI needs positive size, so adjust position if needed
        new_x = min(x0, x)
        new_y = min(y0, y)
        new_width = abs(width)
        new_height = abs(height)
        
        self.box_zoom_rect.setPos([new_x, new_y])
        self.box_zoom_rect.setSize([new_width, new_height])

    def _complete_box_zoom(self):
        """Complete box zoom using current rectangle."""
        if self.box_zoom_rect is None or self.box_zoom_start is None:
            self._cancel_box_zoom()
            return
            
        # Get the rectangle bounds
        pos = self.box_zoom_rect.pos()
        size = self.box_zoom_rect.size()
        
        x_min = pos.x()
        y_min = pos.y()
        x_max = x_min + size.x()
        y_max = y_min + size.y()
        
        # Only zoom if we have a meaningful rectangle
        if size.x() > 0 and size.y() > 0:
            self._zoom_to_box((x_min, y_min), (x_max, y_max))
        
        self._cancel_box_zoom()

    def _zoom_to_box(self, start, end):
        """Zoom the plot to the specified box region."""
        x0, y0 = start
        x1, y1 = end
        
        # Ensure proper ordering
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        
        # Set the view range
        view_box = self.plot_widget.getPlotItem().getViewBox()
        view_box.setRange(xRange=(x_min, x_max), yRange=(y_min, y_max), padding=0)
        
        self.status_bar.showMessage(f"Zoomed to region: MJD [{x_min:.2f}, {x_max:.2f}], Residual [{y_min:.2f}, {y_max:.2f}] μs")

    def _cancel_box_zoom(self):
        """Cancel box zoom mode and clean up."""
        self.box_zoom_active = False
        self.box_zoom_start = None
        
        # Remove the zoom rectangle
        if self.box_zoom_rect is not None:
            self.plot_widget.removeItem(self.box_zoom_rect)
            self.box_zoom_rect = None
        
        # Disconnect mouse events
        try:
            view_box = self.plot_widget.getPlotItem().getViewBox()
            view_box.scene().sigMouseMoved.disconnect(self._on_box_zoom_move)
        except (TypeError, RuntimeError):
            pass  # Already disconnected
        
        # Restore cursor
        self.plot_widget.setCursor(QCursor(Qt.ArrowCursor))
        
        if "Box zoom" in self.status_bar.currentMessage():
            self.status_bar.showMessage("Ready")

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Escape:
            if self.box_zoom_active:
                self._cancel_box_zoom()
                self.status_bar.showMessage("Box zoom cancelled")
                return
            if self.box_delete_active:
                self._cancel_box_delete()
                self.status_bar.showMessage("Box delete cancelled")
                return
        super().keyPressEvent(event)

    # =========================================================================
    # BOX DELETE FEATURE
    # =========================================================================

    def _handle_box_delete_key(self):
        """Handle 'Shift+Z' key press for box delete."""
        if self.box_zoom_active:
            return  # Don't interfere with zoom mode
        if not self.box_delete_active:
            self._start_box_delete()
        else:
            self._complete_box_delete()

    def _start_box_delete(self):
        """Start box delete mode - set first corner at current mouse position."""
        self.box_delete_active = True
        
        # Get current mouse position in data coordinates
        view_box = self.plot_widget.getPlotItem().getViewBox()
        
        # Get mouse position relative to the plot
        global_pos = QCursor.pos()
        widget_pos = self.plot_widget.mapFromGlobal(global_pos)
        scene_pos = self.plot_widget.mapToScene(widget_pos)
        
        # Check if mouse is within the plot area
        if view_box.sceneBoundingRect().contains(scene_pos):
            mouse_point = view_box.mapSceneToView(scene_pos)
            self.box_delete_start = (mouse_point.x(), mouse_point.y())
            self._create_delete_rect(mouse_point.x(), mouse_point.y())
            self.status_bar.showMessage("Box delete: Move mouse to select region, press 'Shift+Z' to delete (Esc to cancel)")
        else:
            self.box_delete_start = None
            self.status_bar.showMessage("Box delete: Move mouse into plot and press 'Shift+Z' to set corner (Esc to cancel)")
        
        # Connect mouse events on the plot
        view_box.scene().sigMouseMoved.connect(self._on_box_delete_move)
        
        # Change cursor to crosshair
        self.plot_widget.setCursor(QCursor(Qt.CrossCursor))

    def _on_box_delete_move(self, scene_pos):
        """Handle mouse move during box delete to update selection rectangle."""
        if not self.box_delete_active:
            return
        
        view_box = self.plot_widget.getPlotItem().getViewBox()
        
        if not view_box.sceneBoundingRect().contains(scene_pos):
            return
            
        mouse_point = view_box.mapSceneToView(scene_pos)
        
        # If we don't have a start point yet, set it now (mouse entered plot)
        if self.box_delete_start is None:
            self.box_delete_start = (mouse_point.x(), mouse_point.y())
            self._create_delete_rect(mouse_point.x(), mouse_point.y())
            self.status_bar.showMessage("Box delete: Move mouse to select region, press 'Shift+Z' to delete (Esc to cancel)")
        else:
            # Update the delete rectangle visualization
            self._update_delete_rect(mouse_point.x(), mouse_point.y())

    def _create_delete_rect(self, x, y):
        """Create the visual delete selection rectangle (red/warning color)."""
        # Use error/warning color for delete
        delete_color = Colors.ACCENT_ERROR
        
        self.box_delete_rect = pg.RectROI(
            [x, y], [0, 0],
            pen=pg.mkPen(delete_color, width=2, style=Qt.DashLine),
            movable=False,
            resizable=False
        )
        self.box_delete_rect.removeHandle(0)
        self.plot_widget.addItem(self.box_delete_rect)

    def _update_delete_rect(self, x, y):
        """Update the delete rectangle as mouse moves."""
        if self.box_delete_rect is None or self.box_delete_start is None:
            return
            
        x0, y0 = self.box_delete_start
        
        new_x = min(x0, x)
        new_y = min(y0, y)
        new_width = abs(x - x0)
        new_height = abs(y - y0)
        
        self.box_delete_rect.setPos([new_x, new_y])
        self.box_delete_rect.setSize([new_width, new_height])

    def _complete_box_delete(self):
        """Complete box delete - remove points within the rectangle."""
        if self.box_delete_rect is None or self.box_delete_start is None:
            self._cancel_box_delete()
            return
            
        # Get the rectangle bounds
        pos = self.box_delete_rect.pos()
        size = self.box_delete_rect.size()
        
        x_min = pos.x()
        y_min = pos.y()
        x_max = x_min + size.x()
        y_max = y_min + size.y()
        
        # Only delete if we have a meaningful rectangle
        if size.x() > 0 and size.y() > 0:
            self._delete_points_in_box(x_min, x_max, y_min, y_max)
        
        self._cancel_box_delete()

    def _delete_points_in_box(self, x_min, x_max, y_min, y_max):
        """Delete data points within the specified box region."""
        if self.mjd is None or self.residuals_us is None:
            return
        
        # Find points inside the box
        inside_mask = (
            (self.mjd >= x_min) & (self.mjd <= x_max) &
            (self.residuals_us >= y_min) & (self.residuals_us <= y_max)
        )
        
        n_delete = np.sum(inside_mask)
        if n_delete == 0:
            self.status_bar.showMessage("No points in selection")
            return
        
        # Track deleted indices relative to original data
        # Find which original indices these correspond to
        current_to_original = self._get_current_to_original_mapping()
        for i, is_inside in enumerate(inside_mask):
            if is_inside:
                orig_idx = current_to_original[i]
                self.deleted_indices.add(orig_idx)
                # Update cached mask (O(1) update)
                if self._keep_mask is not None:
                    self._keep_mask[orig_idx] = False
        
        # Keep points outside the box
        keep_mask = ~inside_mask
        self.mjd = self.mjd[keep_mask]
        self.residuals_us = self.residuals_us[keep_mask]
        self.prefit_residuals_us = self.prefit_residuals_us[keep_mask]
        if self.errors_us is not None:
            self.errors_us = self.errors_us[keep_mask]
        
        # Recalculate RMS using canonical engine stats
        from jug.engine.stats import compute_residual_stats
        stats = compute_residual_stats(self.residuals_us, self.errors_us)
        self._update_rms_from_stats(stats)

        # Update plot
        self._update_plot()
        self._update_plot_title()

        # Update statistics
        self.rms_label.setText(f"{self.rms_us:.6f} μs")
        self.ntoa_label.setText(f"{len(self.mjd)}")
        
        self.status_bar.showMessage(f"Deleted {n_delete} TOAs ({len(self.mjd)} remaining)")

    def _get_current_to_original_mapping(self):
        """Get mapping from current data indices to original data indices."""
        if self.original_mjd is None:
            return np.arange(len(self.mjd))
        
        # Use pre-computed boolean mask for O(1) lookup via numpy
        return np.where(self._keep_mask)[0]

    def _cancel_box_delete(self):
        """Cancel box delete mode and clean up."""
        self.box_delete_active = False
        self.box_delete_start = None
        
        # Remove the delete rectangle
        if self.box_delete_rect is not None:
            self.plot_widget.removeItem(self.box_delete_rect)
            self.box_delete_rect = None
        
        # Disconnect mouse events
        try:
            view_box = self.plot_widget.getPlotItem().getViewBox()
            view_box.scene().sigMouseMoved.disconnect(self._on_box_delete_move)
        except (TypeError, RuntimeError):
            pass
        
        # Restore cursor
        self.plot_widget.setCursor(QCursor(Qt.ArrowCursor))
        
        if "Box delete" in self.status_bar.currentMessage():
            self.status_bar.showMessage("Ready")

    # =========================================================================
    # SAVE TIM FILE
    # =========================================================================

    def on_save_tim(self):
        """Save current TOAs to a new .tim file (excluding deleted points)."""
        if self.tim_file is None:
            QMessageBox.warning(self, "No Data", "No .tim file loaded")
            return
        
        # Suggest a filename
        original_name = self.tim_file.stem
        suggested_name = f"{original_name}_filtered.tim"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save .tim File", suggested_name, "Tim Files (*.tim);;All Files (*)"
        )
        
        if not filename:
            return  # User cancelled
        
        try:
            self._write_filtered_tim(filename)
            self.status_bar.showMessage(f"Saved {len(self.mjd)} TOAs to {Path(filename).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save .tim file:\n\n{e}")

    def _write_filtered_tim(self, output_path):
        """Write filtered TOAs to a new .tim file."""
        # Read original .tim file
        with open(self.tim_file, 'r') as f:
            lines = f.readlines()
        
        # Parse and filter lines
        output_lines = []
        toa_index = 0  # Index into original TOA data
        
        for line in lines:
            stripped = line.strip()
            
            # Keep comment lines and FORMAT/MODE lines
            if not stripped or stripped.startswith('C ') or stripped.upper().startswith('FORMAT') or stripped.upper().startswith('MODE'):
                output_lines.append(line)
                continue
            
            # Check if this is a TOA line (not a comment)
            # TOA lines typically start with a filename or have specific format
            parts = stripped.split()
            if len(parts) >= 3:
                # This looks like a TOA line
                if toa_index not in self.deleted_indices:
                    output_lines.append(line)
                toa_index += 1
            else:
                # Not a TOA line, keep it
                output_lines.append(line)
        
        # Write output file
        with open(output_path, 'w') as f:
            f.writelines(output_lines)
