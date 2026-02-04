"""
JUG Main Window - tempo2 plk-style interactive timing GUI.

Modern redesign inspired by Linear, Raycast, and Notion.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QStatusBar,
    QCheckBox, QGroupBox, QMessageBox, QProgressDialog,
    QFrame, QScrollArea, QSizePolicy, QApplication, QMenu,
    QDialog, QTextBrowser, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, QThreadPool, QEvent, QPointF, QTimer, QSize, QRect, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QFont, QCursor, QPainter, QBrush, QColor
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

# Import canonical stats function (engine is source of truth)
from jug.engine.stats import compute_residual_stats


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

        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # Session + compute/fit

        # Setup UI
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
        
        # Install event filter for key handling
        self.installEventFilter(self)
    
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

        # Title label above plot (pulsar name and RMS) - card style
        self.plot_title_label = QLabel("")
        self.plot_title_label.setObjectName("plotTitleLabel")
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_title_label)

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

        # Add to main layout (plot takes 80%, controls 20%)
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

        # Create dropdown menu
        self.solver_menu = QMenu(self)
        self.solver_menu.setObjectName("solverMenu")

        # Add menu items
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

        view_menu.addSeparator()

        # Theme toggle
        self.theme_action = view_menu.addAction("Theme: Light")
        self.theme_action.triggered.connect(self.on_toggle_theme)

        # Color variant toggle (available in both light and dark modes)
        self.variant_action = view_menu.addAction("Data Color: Navy")
        self.variant_action.triggered.connect(self.on_toggle_color_variant)

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
        self._update_rms_from_result(result)
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
        # DEBUG
        print(f"[DEBUG] Compute error: {error_msg}")

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
        
        # Update or create scatter plot with modern styling
        if self.scatter_item is None:
            # First time: create themed scatter item
            self.scatter_item = create_scatter_item()
            self.plot_widget.addItem(self.scatter_item)
        
        # Update scatter data (fast - no recreation)
        self.scatter_item.setData(x=self.mjd, y=self.residuals_us)
        
        # Update or create error bars with modern styling
        if self.errors_us is not None:
            if self.error_bar_item is None:
                # First time: create themed error bar item
                self.error_bar_item = create_error_bar_item()
                self.plot_widget.addItem(self.error_bar_item)
            
            # Update error bar data
            self.error_bar_item.setData(
                x=self.mjd,
                y=self.residuals_us,
                height=self.errors_us * 2  # ±1σ
            )
        elif self.error_bar_item is not None:
            # Remove error bars if no longer needed
            self.plot_widget.removeItem(self.error_bar_item)
            self.error_bar_item = None
        
        # Add zero line only if enabled (off by default)
        if self.show_zero_line and self.zero_line is None:
            self.zero_line = create_zero_line(self.plot_widget)
        
        # Auto-range only when requested (not every update!)
        if auto_range:
            self.plot_widget.autoRange()
    
    def _update_plot_title(self):
        """Update the plot title with pulsar name and RMS."""
        pulsar_str = self.pulsar_name if self.pulsar_name else "Unknown Pulsar"
        rms_str = f"{self.rms_us:.6f} μs" if self.rms_us is not None else "--"
        rms_label = "wRMS" if self.use_weighted_rms else "RMS"
        # Use styled separator
        self.plot_title_label.setText(f"✦  {pulsar_str}  ·  {rms_label}: {rms_str}")
    
    def on_fit_clicked(self):
        """Handle Fit button click."""
        if not self.session:
            QMessageBox.warning(self, "No Session", "Please load .par and .tim files first")
            return

        # Get selected parameters
        fit_params = [param for param, checkbox in self.param_checkboxes.items()
                      if checkbox.isChecked()]

        if not fit_params:
            QMessageBox.warning(self, "No Parameters",
                              "Please select at least one parameter to fit")
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

        # Run fit with mask and solver mode
        from jug.gui.workers.fit_worker import FitWorker

        worker = FitWorker(self.session, fit_params, toa_mask=toa_mask,
                           solver_mode=self.solver_mode)
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
        
        # DEBUG
        print(f"[DEBUG] Computing postfit with params: {list(result['final_params'].keys())}")
        print(f"[DEBUG] F0 = {result['final_params'].get('F0', 'N/A')}")
        
        # Create compute worker with fitted parameters
        worker = ComputeWorker(self.session, params=result['final_params'])
        worker.signals.result.connect(self.on_postfit_compute_complete)
        worker.signals.error.connect(self.on_compute_error)
        worker.signals.finished.connect(lambda: None)  # No action needed
        
        # Start in thread pool
        self.thread_pool.start(worker)
    
    def on_postfit_compute_complete(self, result):
        """Handle postfit residual computation completion."""
        # DEBUG
        print(f"[DEBUG] Postfit compute complete: RMS = {result['rms_us']:.6f} μs")
        print(f"[DEBUG] Residuals range: [{result['residuals_us'].min():.3f}, {result['residuals_us'].max():.3f}]")
        
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
            stats = compute_residual_stats(self.residuals_us, self.errors_us)
            self._update_rms_from_stats(stats)
        else:
            # No deletions - use all data
            self.mjd = full_mjd
            self.residuals_us = full_residuals
            self.postfit_residuals_us = full_residuals.copy()
            self.errors_us = full_errors
            self._update_rms_from_result(result)
        
        # DEBUG
        print(f"[DEBUG] Updated GUI RMS: {self.rms_us:.6f} μs (after filtering {len(self.deleted_indices)} deleted TOAs)")
        
        # Update plot (auto-range to show new residual scale after fit)
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

            # Enable fit report button
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
        if self.fit_results:
            self._show_fit_results(self.fit_results)
    
    def _show_fit_results(self, result):
        """
        Show fit results in a styled dialog using QTableWidget for performance.

        Parameters
        ----------
        result : dict
            Fit results
        """
        # Styling constants
        text_neutral = Colors.TEXT_PRIMARY
        text_muted = Colors.TEXT_SECONDARY
        accent = get_dynamic_accent_primary()
        bg_surface = Colors.SURFACE
        bg_secondary = Colors.BG_SECONDARY
        border_col = Colors.SURFACE_BORDER
        
        # Dialog Setup
        dialog = QDialog(self)
        pulsar_str = self.pulsar_name if self.pulsar_name else 'Unknown'
        dialog.setWindowTitle(f"{pulsar_str} - Fit Results")
        
        # Size
        main_height = self.height()
        main_width = self.width()
        dialog.setFixedSize(min(int(main_width * 0.8), 1000), min(int(main_height * 0.85), 700))
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # 1. Header (Title)
        title_lbl = QLabel(f"{pulsar_str} - Fit Results")
        title_lbl.setStyleSheet(f"color: {accent}; font-size: 20px; font-weight: bold;")
        layout.addWidget(title_lbl)
        
        # 2. Summary Stats (Grid/HBox)
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {bg_secondary}; border-radius: 8px;")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        
        def add_stat(label, value, color=text_neutral, bold=False):
            container = QWidget()
            vbox = QVBoxLayout(container)
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
            stats_layout.addWidget(container)
            
        # Use current RMS value (weighted or unweighted based on mode)
        rms_label = "Final wRMS" if self.use_weighted_rms else "Final RMS"
        rms_value = self.rms_us if self.rms_us is not None else result['final_rms']
        add_stat(rms_label, f"{rms_value:.6f} μs", color=accent, bold=True)
        add_stat("Iterations", result['iterations'])
        conv_text = "Yes" if result['converged'] else "No"
        conv_color = Colors.ACCENT_SUCCESS if result['converged'] else Colors.ACCENT_WARNING
        add_stat("Converged", conv_text, color=conv_color, bold=True)
        add_stat("Time", f"{result['total_time']:.2f}s")
        
        layout.addWidget(stats_frame)
        
        # 3. Parameters Table
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels(["Parameter", "New Value", "Previous", "Change", "Uncertainty", "Unit"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setShowGrid(False)
        
        # Header Styling
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True) 
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        
        # Prepare Data
        from jug.io.par_reader import parse_ra, parse_dec, format_ra, format_dec
        
        param_order = getattr(self, 'available_params', list(result['final_params'].keys()))
        ordered_params = [p for p in param_order if p in result['final_params']]
        for p in result['final_params'].keys():
            if p not in ordered_params:
                ordered_params.append(p)
                
        table.setRowCount(len(ordered_params))
        
        for row, param in enumerate(ordered_params):
            new_value = result['final_params'][param]
            uncertainty = result['uncertainties'][param]
            prev_value = self.initial_params.get(param, 0.0)
            
            # Logic copied from original HTML generator
            new_val_str = ""
            prev_val_str = ""
            change_str = ""
            unit = ""
            
            if param == 'RAJ':
                if isinstance(prev_value, str): prev_value_rad = parse_ra(prev_value)
                else: prev_value_rad = prev_value
                change = new_value - prev_value_rad
                new_val_str = format_ra(new_value)
                prev_val_str = format_ra(prev_value_rad) if isinstance(prev_value, str) else prev_value
                change_str = f"{change * 180 / 3.14159265 * 3600:+.6f}"
                unit = "Δ arcsec"
            elif param == 'DECJ':
                if isinstance(prev_value, str): prev_value_rad = parse_dec(prev_value)
                else: prev_value_rad = prev_value
                change = new_value - prev_value_rad
                new_val_str = format_dec(new_value)
                prev_val_str = format_dec(prev_value_rad) if isinstance(prev_value, str) else prev_value
                change_str = f"{change * 180 / 3.14159265 * 3600:+.6f}"
                unit = "Δ arcsec"
            elif param.startswith('F'):
                change = new_value - prev_value
                if param == 'F0':
                    new_val_str = f"{new_value:.15f}"
                    prev_val_str = f"{prev_value:.15f}"
                    change_str = f"{change:+.15f}"
                    unit = "Hz"
                else:
                    new_val_str = f"{new_value:.6e}"
                    prev_val_str = f"{prev_value:.6e}"
                    change_str = f"{change:+.6e}"
                    unit = "Hz/s"
            elif param.startswith('DM'):
                change = new_value - prev_value
                new_val_str = f"{new_value:.10f}"
                prev_val_str = f"{prev_value:.10f}"
                change_str = f"{change:+.10f}"
                unit = "pc/cm³"
            elif param in ['PMRA', 'PMDEC']:
                change = new_value - prev_value
                new_val_str = f"{new_value:.6f}"
                prev_val_str = f"{prev_value:.6f}"
                change_str = f"{change:+.6f}"
                unit = "mas/yr"
            elif param == 'PX':
                change = new_value - prev_value
                new_val_str = f"{new_value:.6f}"
                prev_val_str = f"{prev_value:.6f}"
                change_str = f"{change:+.6f}"
                unit = "mas"
            elif param == 'PB':
                change = new_value - prev_value
                new_val_str = f"{new_value:.15f}"
                prev_val_str = f"{prev_value:.15f}"
                change_str = f"{change:+.6e}"
                unit = "days"
            elif param == 'PBDOT':
                change = new_value - prev_value
                new_val_str = f"{new_value:.6e}"
                prev_val_str = f"{prev_value:.6e}"
                change_str = f"{change:+.6e}"
                unit = "s/s"
            elif param == 'A1':
                change = new_value - prev_value
                new_val_str = f"{new_value:.10f}"
                prev_val_str = f"{prev_value:.10f}"
                change_str = f"{change:+.10f}"
                unit = "lt-s"
            elif param in ['T0', 'TASC']:
                change = new_value - prev_value
                new_val_str = f"{new_value:.12f}"
                prev_val_str = f"{prev_value:.12f}"
                change_str = f"{change:+.6e}"
                unit = "MJD"
            elif param in ['ECC', 'EPS1', 'EPS2']:
                change = new_value - prev_value
                new_val_str = f"{new_value:.12e}"
                prev_val_str = f"{prev_value:.12e}"
                change_str = f"{change:+.6e}"
                unit = ""
            elif param == 'OM':
                change = new_value - prev_value
                new_val_str = f"{new_value:.10f}"
                prev_val_str = f"{prev_value:.10f}"
                change_str = f"{change:+.6e}"
                unit = "deg"
            elif param == 'M2':
                change = new_value - prev_value
                new_val_str = f"{new_value:.12f}"
                prev_val_str = f"{prev_value:.12f}"
                change_str = f"{change:+.6e}"
                unit = "M☉"
            elif param == 'SINI':
                change = new_value - prev_value
                new_val_str = f"{new_value:.12f}"
                prev_val_str = f"{prev_value:.12f}"
                change_str = f"{change:+.6e}"
                unit = ""
            else:
                change = new_value - float(prev_value) if isinstance(prev_value, (int, float)) else 0.0
                new_val_str = f"{new_value:.6g}"
                prev_val_str = f"{prev_value}"
                change_str = f"{change:+.6g}"
                unit = ""

            unc_str = f"±{uncertainty:.2e}"
            
            # Populate Row
            def create_item(text, color=text_neutral, font_mono=False):
                item = QTableWidgetItem(text)
                item.setForeground(QBrush(QColor(color)))
                if font_mono:
                    font = QFont("Monospace")
                    font.setStyleHint(QFont.Monospace)
                    item.setFont(font)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                return item

            # Parameter
            item_p = create_item(param, color=accent) # Parameter name accent color
            font_p = item_p.font()
            font_p.setBold(True)
            item_p.setFont(font_p)
            table.setItem(row, 0, item_p)
            
            table.setItem(row, 1, create_item(new_val_str, font_mono=True))
            table.setItem(row, 2, create_item(prev_val_str, color=text_muted, font_mono=True))
            table.setItem(row, 3, create_item(change_str, color=text_muted, font_mono=True))
            table.setItem(row, 4, create_item(unc_str, color=text_muted, font_mono=True))
            table.setItem(row, 5, create_item(unit, color=text_muted))

        # Table Styling
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
        
        # 4. Close Button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet(get_primary_button_style())
        close_btn.setFixedWidth(120)
        close_btn.setCursor(Qt.PointingHandCursor)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        dialog.exec()

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
            
            # Recalculate RMS using canonical engine stats
            stats = compute_residual_stats(self.residuals_us, self.errors_us)
            self._update_rms_from_stats(stats)
            
            # Reset fit state
            self.is_fitted = False
            self.fit_results = None
            self.fit_report_button.setEnabled(False)
            
            # Update plot
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

    def _toggle_rms_mode(self):
        """Toggle between weighted and unweighted RMS display (shortcut: W)."""
        self.use_weighted_rms = not self.use_weighted_rms
        
        # Update the displayed RMS value based on mode
        if self.use_weighted_rms:
            self.rms_us = self.weighted_rms_us
            rms_type = "weighted"
            label_text = "wRMS"
        else:
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

    def on_toggle_color_variant(self):
        """Toggle data color variant (navy/burgundy in light, classic/scilab in dark)."""
        if is_dark_mode():
            # Dark mode: toggle between classic and scilab
            new_variant = toggle_synthwave_variant()
            if new_variant == "scilab":
                self.variant_action.setText("Data Color: Cyan/Pink")
            else:
                self.variant_action.setText("Data Color: Pink/Cyan")
        else:
            # Light mode: toggle between navy and burgundy
            new_variant = toggle_light_variant()
            if new_variant == "burgundy":
                self.variant_action.setText("Data Color: Burgundy")
            else:
                self.variant_action.setText("Data Color: Navy")

        # Refresh the entire UI with new variant colors
        self._apply_theme()

    def on_toggle_theme(self):
        """Toggle between light and dark (Synthwave) themes."""
        if is_dark_mode():
            set_theme(LightTheme)
            self.theme_action.setText("Theme: Light")
            # Update variant action text for light mode
            variant = get_light_variant()
            if variant == "burgundy":
                self.variant_action.setText("Data Color: Burgundy")
            else:
                self.variant_action.setText("Data Color: Navy")
        else:
            set_theme(SynthwaveTheme)
            self.theme_action.setText("Theme: Synthwave '84")
            # Update variant action text for dark mode
            variant = get_synthwave_variant()
            if variant == "scilab":
                self.variant_action.setText("Data Color: Cyan/Pink")
            else:
                self.variant_action.setText("Data Color: Pink/Cyan")

        # Refresh the entire UI
        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme to all UI elements.
        
        OPTIMIZED: Uses ONE top-level stylesheet for all QSS styling.
        Only pyqtgraph plot elements and inline-styled widgets need direct updates.
        """
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
                     # Re-apply text color
                     widget.setStyleSheet(f"""
                        QCheckBox {{
                            color: {Colors.TEXT_PRIMARY};
                            padding: 4px 8px;
                            spacing: 8px;
                            font-size: 13px;
                        }}
                        QCheckBox::indicator {{
                            width: 16px; height: 16px;
                        }}
                     """)
        
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

    def on_zoom_fit(self):
        """Zoom plot to fit data."""
        self.plot_widget.autoRange()

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
        Parse .par file to extract available fittable parameters.

        Returns
        -------
        list of str
            List of parameter names that exist in the par file
        """
        if not self.par_file:
            return []

        # Common fittable parameters we look for
        fittable_params = [
            'F0', 'F1', 'F2', 'F3', 'F4', 'F5',  # Spin
            'DM', 'DM1', 'DM2', 'DM3',  # Dispersion
            'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX',  # Astrometry
            'PB', 'A1', 'ECC', 'OM', 'T0', 'TASC',  # Binary
            'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT'  # More binary
        ]

        found_params = []

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
                            found_params.append(param_name)

        except Exception as e:
            print(f"Error parsing par file: {e}")
            return []

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
        # Parse par file to get available parameters
        params_in_file = self._parse_par_file_parameters()
        
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
            
            # Determine initial check state
            should_check = False
            if param in self.cmdline_fit_params:
                should_check = True
            elif not self.cmdline_fit_params and param in ['F0', 'F1']:
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

        # Update status
        if all_params:
            status_msg = f"Found {len(params_in_file)} fittable parameters in .par file"
            if self.cmdline_fit_params:
                added = [p for p in self.cmdline_fit_params if p not in params_in_file]
                if added:
                    status_msg += f" (+ {len(added)} from --fit)"
            self.status_bar.showMessage(status_msg)

    # =========================================================================
    # BOX ZOOM FEATURE
    # =========================================================================

    def eventFilter(self, obj, event):
        """Handle key events for box zoom, box delete, and unzoom."""
        if event.type() == QEvent.KeyPress:
            modifiers = event.modifiers()
            key = event.key()
            
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
            return list(range(len(self.mjd)))
        
        # Build list of original indices that haven't been deleted
        original_indices = [i for i in range(len(self.original_mjd)) if i not in self.deleted_indices]
        return original_indices

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
