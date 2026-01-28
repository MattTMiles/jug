"""
JUG Main Window - tempo2 plk-style interactive timing GUI.

Modern redesign inspired by Linear, Raycast, and Notion.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QStatusBar,
    QCheckBox, QGroupBox, QMessageBox, QProgressDialog,
    QFrame, QScrollArea, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QFont
import pyqtgraph as pg
import numpy as np

from jug.gui.theme import (
    get_main_stylesheet,
    get_plot_title_style,
    get_stats_card_style,
    get_stat_label_style,
    get_stat_value_style,
    get_placeholder_style,
    get_added_param_style,
    get_section_title_style,
    get_control_panel_style,
    get_primary_button_style,
    get_secondary_button_style,
    configure_plot_widget,
    create_scatter_item,
    create_error_bar_item,
    create_zero_line,
    get_scatter_colors,
    Colors,
    Typography,
    Spacing,
    LightTheme,
    SynthwaveTheme,
    set_theme,
    is_dark_mode,
    PlotTheme,
)


class MainWindow(QMainWindow):
    """Main window for JUG timing analysis GUI."""

    def __init__(self, fit_params=None):
        super().__init__()
        self.setWindowTitle("JUG Timing Analysis")
        self.setGeometry(100, 100, 1152, 768)
        self.setMinimumSize(1152, 768)
        self.setMaximumSize(2304, 1536)

        # Apply modern theme stylesheet
        self.setStyleSheet(get_main_stylesheet())

        # Timing session (engine with caching)
        self.session = None
        
        # Data storage
        self.par_file = None
        self.tim_file = None
        self.residuals_us = None
        self.mjd = None
        self.errors_us = None
        self.rms_us = None

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
        self.data_point_color = "primary"  # "primary" or "alt" (theme-dependent)

        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # Session + compute/fit

        # Setup UI
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
    
    def _setup_ui(self):
        """Setup the main user interface with modern styling."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout: Plot on left, controls on right
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left side: Plot area with title
        self.plot_container = QWidget()
        self.plot_container.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")
        plot_layout = QVBoxLayout(self.plot_container)
        plot_layout.setContentsMargins(24, 16, 16, 16)
        plot_layout.setSpacing(12)

        # Title label above plot (pulsar name and RMS) - card style
        self.plot_title_label = QLabel("")
        self.plot_title_label.setStyleSheet(get_plot_title_style())
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_title_label)

        # Large residual plot with modern styling
        self.plot_widget = pg.PlotWidget()
        configure_plot_widget(self.plot_widget)

        # Add subtle rounded corners effect via container
        self.plot_frame = QFrame()
        self.plot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.PLOT_BG};
                border: 1px solid {Colors.SURFACE_BORDER};
                border-radius: 12px;
            }}
        """)
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
        container_layout = QHBoxLayout(self.control_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Main control panel
        panel = QWidget()
        panel.setStyleSheet(get_control_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        # Parameters button (opens drawer)
        self.params_drawer_btn = QPushButton("Parameters  ▶")
        self.params_drawer_btn.setStyleSheet(get_secondary_button_style())
        self.params_drawer_btn.setCursor(Qt.PointingHandCursor)
        self.params_drawer_btn.clicked.connect(self.on_toggle_params_drawer)
        layout.addWidget(self.params_drawer_btn)

        layout.addSpacing(8)

        # Create the parameter drawer (hidden by default)
        self._create_params_drawer()

        self.param_checkboxes = {}

        # Action buttons section
        self.action_title = QLabel("Actions")
        self.action_title.setStyleSheet(get_section_title_style())
        layout.addWidget(self.action_title)

        # Primary action: Run Fit (special styling)
        self.fit_button = QPushButton("▶  Run Fit")
        self.fit_button.setStyleSheet(get_primary_button_style())
        self.fit_button.setCursor(Qt.PointingHandCursor)
        self.fit_button.clicked.connect(self.on_fit_clicked)
        self.fit_button.setEnabled(False)
        layout.addWidget(self.fit_button)

        # Secondary buttons with consistent styling
        secondary_style = get_secondary_button_style()

        self.fit_report_button = QPushButton("📊  Fit Report")
        self.fit_report_button.setStyleSheet(secondary_style)
        self.fit_report_button.setCursor(Qt.PointingHandCursor)
        self.fit_report_button.clicked.connect(self.on_show_fit_report)
        self.fit_report_button.setEnabled(False)
        layout.addWidget(self.fit_report_button)

        self.reset_button = QPushButton("↺  Reset to Prefit")
        self.reset_button.setStyleSheet(secondary_style)
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)

        self.fit_window_button = QPushButton("⤢  Fit Window to Data")
        self.fit_window_button.setStyleSheet(secondary_style)
        self.fit_window_button.setCursor(Qt.PointingHandCursor)
        self.fit_window_button.clicked.connect(self.on_zoom_fit)
        self.fit_window_button.setEnabled(False)
        layout.addWidget(self.fit_window_button)

        layout.addSpacing(16)

        # Statistics card
        stats_card = QWidget()
        stats_card.setStyleSheet(f"""
            QWidget {{
                background-color: {Colors.SURFACE};
                border: 1px solid {Colors.SURFACE_BORDER};
                border-radius: 12px;
            }}
            QLabel {{
                background: transparent;
                border: none;
            }}
        """)
        stats_card.setMinimumHeight(160)
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(16, 16, 16, 16)

        self.stats_title = QLabel("Statistics")
        self.stats_title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {Colors.TEXT_PRIMARY};
            background: transparent;
            border: none;
            padding-bottom: 8px;
        """)
        stats_layout.addWidget(self.stats_title)

        # Create stat rows with label + value
        self.rms_label = self._create_stat_row(stats_layout, "RMS", "--")
        # Highlight RMS value - contrasts with data point color (burgundy when points are navy)
        self.rms_label.setStyleSheet(f"""
            font-family: monospace;
            font-size: 14px;
            font-weight: 600;
            color: #5E1803;
            background: transparent;
            border: none;
        """)
        self.iter_label = self._create_stat_row(stats_layout, "Iterations", "--")
        self.ntoa_label = self._create_stat_row(stats_layout, "TOAs", "--")
        self.chi2_label = self._create_stat_row(stats_layout, "χ²/dof", "--")

        layout.addWidget(stats_card)

        # Stretch to push everything to top
        layout.addStretch()

        # Add panel to container
        container_layout.addWidget(panel)
        container_layout.addWidget(self.params_drawer)

        return self.control_container

    def _create_params_drawer(self):
        """Create the slide-out parameter drawer."""
        self.params_drawer = QFrame()
        self.params_drawer.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border-left: 1px solid {Colors.SURFACE_BORDER};
            }}
        """)
        self.params_drawer.setFixedWidth(200)
        self.params_drawer.setVisible(False)  # Hidden by default
        self.params_drawer_open = False

        drawer_layout = QVBoxLayout(self.params_drawer)
        drawer_layout.setContentsMargins(12, 16, 12, 16)
        drawer_layout.setSpacing(8)

        # Header with close button
        header_layout = QHBoxLayout()
        self.params_header_label = QLabel("Parameters to Fit")
        self.params_header_label.setStyleSheet(get_section_title_style())
        header_layout.addWidget(self.params_header_label)
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

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        # Container for checkboxes
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.param_layout.setSpacing(4)
        self.param_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder
        self.param_placeholder = QLabel("Load .par file to see parameters")
        self.param_placeholder.setStyleSheet(get_placeholder_style())
        self.param_placeholder.setWordWrap(True)
        self.param_layout.addWidget(self.param_placeholder)
        self.param_layout.addStretch()

        scroll.setWidget(self.param_container)
        drawer_layout.addWidget(scroll)

    def on_toggle_params_drawer(self):
        """Toggle the parameter drawer open/closed, expanding window to fit."""
        self.params_drawer_open = not self.params_drawer_open
        drawer_width = 200  # Fixed drawer width

        if self.params_drawer_open:
            # Expand window to accommodate drawer
            current_size = self.size()
            self.resize(current_size.width() + drawer_width, current_size.height())
            self.params_drawer.setVisible(True)
            self.params_drawer_btn.setText("Parameters  ◀")
        else:
            # Set drawer width to 0 first to prevent layout redistribution
            self.params_drawer.setFixedWidth(0)
            self.params_drawer.setVisible(False)
            # Shrink window
            current_size = self.size()
            self.resize(current_size.width() - drawer_width, current_size.height())
            # Restore drawer width for next open
            self.params_drawer.setFixedWidth(drawer_width)
            self.params_drawer_btn.setText("Parameters  ▶")

    def _create_stat_row(self, parent_layout, label_text, initial_value):
        """Create a styled statistic row with label and value. Returns (label, value) tuple."""
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel(label_text)
        label.setStyleSheet(f"""
            font-size: 12px;
            color: {Colors.TEXT_SECONDARY};
            background: transparent;
            border: none;
        """)

        value = QLabel(initial_value)
        value.setStyleSheet(f"""
            font-family: monospace;
            font-size: 14px;
            color: {Colors.TEXT_PRIMARY};
            background: transparent;
            border: none;
        """)
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

        # Data point color toggle
        self.color_toggle_action = view_menu.addAction("Data Points: Navy")
        self.color_toggle_action.triggered.connect(self.on_toggle_data_color)

        view_menu.addSeparator()

        # Theme toggle
        self.theme_action = view_menu.addAction("Theme: Light")
        self.theme_action.triggered.connect(self.on_toggle_theme)

        view_menu.addSeparator()

        zoom_fit_action = view_menu.addAction("Zoom to Fit")
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.on_zoom_fit)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        fit_action = tools_menu.addAction("Run Fit")
        fit_action.setShortcut("Ctrl+F")
        fit_action.triggered.connect(self.on_fit_clicked)
        
        reset_action = tools_menu.addAction("Reset to Initial")
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.on_reset_clicked)
        
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
        self.rms_us = result['rms_us']
        self.is_fitted = False
        self.fit_results = None
        
        # Update plot
        self._update_plot()
        self._update_plot_title()
        
        # Enable controls
        self.fit_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.fit_window_button.setEnabled(True)
        
        # Update statistics (now just values, labels are separate)
        self.rms_label.setText(f"{self.rms_us:.6f} μs")
        self.ntoa_label.setText(f"{len(self.mjd)}")
        
        # Update status
        self.status_bar.showMessage(
            f"Loaded {len(self.mjd)} TOAs, Prefit RMS = {self.rms_us:.6f} μs"
        )
    
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
        # Use styled separator
        self.plot_title_label.setText(f"✦  {pulsar_str}  ·  RMS: {rms_str}")
    
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
        self.status_bar.showMessage(f"Fitting {', '.join(fit_params)}...")

        # Create and start fit worker (uses session)
        from jug.gui.workers.fit_worker import FitWorker

        worker = FitWorker(self.session, fit_params)
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
        
        # Update ALL data (MJDs, residuals, errors, RMS)
        self.mjd = result['tdb_mjd']
        self.residuals_us = result['residuals_us']
        self.postfit_residuals_us = result['residuals_us'].copy()
        self.errors_us = result.get('errors_us', None)
        self.rms_us = result['rms_us']
        
        # DEBUG
        print(f"[DEBUG] Updated GUI RMS: {self.rms_us:.6f} μs")
        
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
            self.status_bar.showMessage(
                f"Fit complete: {param_str} | "
                f"RMS = {self.rms_us:.6f} μs | "
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
        Show fit results in a styled dialog.

        Parameters
        ----------
        result : dict
            Fit results
        """
        # Build title with pulsar name and RMS
        pulsar_str = self.pulsar_name if self.pulsar_name else 'Unknown'
        title = f"{pulsar_str} - Fit Results"

        # Colors for styling
        text_neutral = "#111827"
        text_muted = "#6b7280"
        accent = Colors.ACCENT_PRIMARY
        border = Colors.SURFACE_BORDER
        converged_color = Colors.ACCENT_SUCCESS if result['converged'] else Colors.ACCENT_WARNING
        converged_text = "Yes" if result['converged'] else "No"

        # Build parameter rows
        param_rows = ""
        for param, new_value in result['final_params'].items():
            uncertainty = result['uncertainties'][param]
            prev_value = self.initial_params.get(param, 0.0)
            change = new_value - prev_value

            # Format based on parameter type
            if param.startswith('F'):
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
                new_val_str = f"{new_value:.10f}"
                prev_val_str = f"{prev_value:.10f}"
                change_str = f"{change:+.10f}"
                unit = "pc/cm³"
            else:
                new_val_str = f"{new_value:.6e}"
                prev_val_str = f"{prev_value:.6e}"
                change_str = f"{change:+.6e}"
                unit = ""

            unc_str = f"±{uncertainty:.2e}"

            param_rows += f"""
            <tr>
                <td style='padding: 8px; border-bottom: 1px solid {border};'><b style='color: {accent};'>{param}</b></td>
                <td style='padding: 8px; border-bottom: 1px solid {border}; font-family: monospace;'>{new_val_str}</td>
                <td style='padding: 8px; border-bottom: 1px solid {border}; font-family: monospace; color: {text_muted};'>{prev_val_str}</td>
                <td style='padding: 8px; border-bottom: 1px solid {border}; font-family: monospace;'>{change_str}</td>
                <td style='padding: 8px; border-bottom: 1px solid {border}; font-family: monospace; color: {text_muted};'>{unc_str}</td>
                <td style='padding: 8px; border-bottom: 1px solid {border}; color: {text_muted}; white-space: nowrap;'>{unit}</td>
            </tr>
            """

        msg = f"""
        <h2 style='color: {accent}; margin-bottom: 16px;'>Fit Results</h2>

        <table style='border-collapse: collapse; width: 100%;'>
            <thead>
                <tr style='background-color: {Colors.BG_SECONDARY};'>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent};'>Parameter</th>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent};'>New Value</th>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent};'>Previous</th>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent};'>Change</th>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent};'>Uncertainty</th>
                    <th style='padding: 10px 8px; text-align: left; border-bottom: 2px solid {accent}; white-space: nowrap;'>Unit</th>
                </tr>
            </thead>
            <tbody>
                {param_rows}
            </tbody>
        </table>

        <br><br>

        <table style='width: 100%; background-color: {Colors.BG_SECONDARY}; border-radius: 8px;'>
            <tr>
                <td style='padding: 16px; text-align: center;'>
                    <span style='color: {text_muted}; font-size: 12px;'>Final RMS</span><br>
                    <span style='font-family: monospace; font-size: 18px; color: {accent}; font-weight: bold;'>{result['final_rms']:.6f} μs</span>
                </td>
                <td style='padding: 16px; text-align: center;'>
                    <span style='color: {text_muted}; font-size: 12px;'>Iterations</span><br>
                    <span style='font-family: monospace; font-size: 18px; color: {text_neutral};'>{result['iterations']}</span>
                </td>
                <td style='padding: 16px; text-align: center;'>
                    <span style='color: {text_muted}; font-size: 12px;'>Converged</span><br>
                    <span style='font-size: 18px; color: {converged_color}; font-weight: bold;'>{converged_text}</span>
                </td>
                <td style='padding: 16px; text-align: center;'>
                    <span style='color: {text_muted}; font-size: 12px;'>Time</span><br>
                    <span style='font-family: monospace; font-size: 18px; color: {text_neutral};'>{result['total_time']:.2f}s</span>
                </td>
            </tr>
        </table>
        """

        msgbox = QMessageBox(self)
        msgbox.setWindowTitle(title)
        msgbox.setTextFormat(Qt.RichText)
        msgbox.setText(msg)

        # Set dialog size
        screen = QApplication.primaryScreen().geometry()
        dialog_height = int(screen.height() * 0.6)
        dialog_width = 950

        msgbox.show()
        msgbox.setFixedSize(dialog_width, dialog_height)
        msgbox.exec()

    def on_reset_clicked(self):
        """Handle Reset button click."""
        if self.prefit_residuals_us is not None:
            # Reset to prefit residuals
            self.residuals_us = self.prefit_residuals_us.copy()
            self.is_fitted = False
            self.fit_results = None
            self.fit_report_button.setEnabled(False)
            self._update_plot()
            self._update_plot_title()

            # Update statistics (values only)
            self.rms_label.setText(f"{self.rms_us:.6f} μs")
            self.iter_label.setText("--")
            self.chi2_label.setText("--")
            self.status_bar.showMessage("Reset to prefit residuals")
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

    def on_toggle_data_color(self):
        """Toggle data point color between primary and alternate."""
        colors = get_scatter_colors()

        if self.data_point_color == "primary":
            self.data_point_color = "alt"
            point_color = colors['alt']
            error_color = PlotTheme.get_error_bar_color_alt()
            if is_dark_mode():
                self.color_toggle_action.setText("Data Points: Pink")
                rms_color = "#36f9f6"  # Cyan when points are pink
            else:
                self.color_toggle_action.setText("Data Points: Burgundy")
                rms_color = "#2b4162"  # Navy when points are burgundy
        else:
            self.data_point_color = "primary"
            point_color = colors['primary']
            error_color = PlotTheme.get_error_bar_color()
            if is_dark_mode():
                self.color_toggle_action.setText("Data Points: Cyan")
                rms_color = "#ff7edb"  # Pink when points are cyan
            else:
                self.color_toggle_action.setText("Data Points: Navy")
                rms_color = "#5E1803"  # Burgundy when points are navy

        # Update scatter plot if it exists
        if self.scatter_item is not None:
            self.scatter_item.setBrush(pg.mkBrush(*point_color))

        # Update error bars if they exist
        if self.error_bar_item is not None:
            self.error_bar_item.setData(pen=pg.mkPen(color=error_color, width=2.0))

        # Update RMS label color to contrast with data points
        self.rms_label.setStyleSheet(f"""
            font-family: monospace;
            font-size: 14px;
            font-weight: 600;
            color: {rms_color};
            background: transparent;
            border: none;
        """)

    def on_toggle_theme(self):
        """Toggle between light and dark (Synthwave) themes."""
        if is_dark_mode():
            set_theme(LightTheme)
            self.theme_action.setText("Theme: Light")
        else:
            set_theme(SynthwaveTheme)
            self.theme_action.setText("Theme: Synthwave '84")

        # Refresh the entire UI
        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme to all UI elements."""
        # Update main stylesheet
        self.setStyleSheet(get_main_stylesheet())

        # Update plot widget
        configure_plot_widget(self.plot_widget)

        # Update plot container background
        self.plot_container.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")

        # Update plot title
        self.plot_title_label.setStyleSheet(get_plot_title_style())

        # Update plot frame
        self.plot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.PLOT_BG};
                border: 1px solid {Colors.SURFACE_BORDER};
                border-radius: 12px;
            }}
        """)

        # Update control panel
        self.control_container.findChild(QWidget).setStyleSheet(get_control_panel_style())

        # Update buttons
        self.fit_button.setStyleSheet(get_primary_button_style())
        secondary_style = get_secondary_button_style()
        self.fit_report_button.setStyleSheet(secondary_style)
        self.reset_button.setStyleSheet(secondary_style)
        self.fit_window_button.setStyleSheet(secondary_style)
        self.params_drawer_btn.setStyleSheet(secondary_style)

        # Update params drawer
        self.params_drawer.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.SURFACE};
                border-left: 1px solid {Colors.SURFACE_BORDER};
            }}
        """)

        # Update stats card styling
        # Find the stats card and update it
        for widget in self.control_container.findChildren(QWidget):
            if hasattr(widget, 'minimumHeight') and widget.minimumHeight() == 160:
                widget.setStyleSheet(f"""
                    QWidget {{
                        background-color: {Colors.SURFACE};
                        border: 1px solid {Colors.SURFACE_BORDER};
                        border-radius: 12px;
                    }}
                    QLabel {{
                        background: transparent;
                        border: none;
                    }}
                """)

        # Update scatter plot colors
        colors = get_scatter_colors()
        if self.scatter_item is not None:
            if self.data_point_color == "primary":
                self.scatter_item.setBrush(pg.mkBrush(*colors['primary']))
            else:
                self.scatter_item.setBrush(pg.mkBrush(*colors['alt']))

        # Update error bar colors
        if self.error_bar_item is not None:
            if self.data_point_color == "primary":
                error_color = PlotTheme.get_error_bar_color()
            else:
                error_color = PlotTheme.get_error_bar_color_alt()
            self.error_bar_item.setData(pen=pg.mkPen(color=error_color, width=2.0))

        # Update zero line if visible
        if self.zero_line is not None:
            self.plot_widget.removeItem(self.zero_line)
            self.zero_line = create_zero_line(self.plot_widget)

        # Update color toggle menu text
        if is_dark_mode():
            if self.data_point_color == "primary":
                self.color_toggle_action.setText("Data Points: Cyan")
            else:
                self.color_toggle_action.setText("Data Points: Pink")
        else:
            if self.data_point_color == "primary":
                self.color_toggle_action.setText("Data Points: Navy")
            else:
                self.color_toggle_action.setText("Data Points: Burgundy")

        # Update RMS label color
        if is_dark_mode():
            rms_color = "#36f9f6" if self.data_point_color == "alt" else "#ff7edb"
        else:
            rms_color = "#2b4162" if self.data_point_color == "alt" else "#5E1803"

        self.rms_label.setStyleSheet(f"""
            font-family: monospace;
            font-size: 14px;
            font-weight: 600;
            color: {rms_color};
            background: transparent;
            border: none;
        """)

    def on_zoom_fit(self):
        """Zoom plot to fit data."""
        self.plot_widget.autoRange()
    
    def on_about(self):
        """Show about dialog with modern styling."""
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("About JUG")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            f"""
            <div style='text-align: center;'>
                <h2 style='color: {Colors.ACCENT_PRIMARY}; margin-bottom: 8px;'>✦ JUG Timing Analysis</h2>
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

        # Add command-line fit parameters
        all_params = list(set(params_in_file + self.cmdline_fit_params))

        # Sort parameters by type and order
        def param_sort_key(p):
            # Sort order: F params, DM params, astrometry, binary
            if p.startswith('F'):
                return (0, int(p[1:]) if p[1:].isdigit() else 99)
            elif p.startswith('DM'):
                return (1, int(p[2:]) if len(p) > 2 and p[2:].isdigit() else 0)
            elif p in ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX']:
                return (2, ['RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX'].index(p))
            else:
                return (3, 0)

        all_params.sort(key=param_sort_key)

        # Clear existing checkboxes and layout
        for checkbox in self.param_checkboxes.values():
            checkbox.deleteLater()
        self.param_checkboxes.clear()

        if self.param_placeholder:
            self.param_placeholder.deleteLater()
            self.param_placeholder = None

        # Clear any remaining items from layout (like stretch)
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create new checkboxes
        for param in all_params:
            checkbox = QCheckBox(param)

            # Pre-select if in command-line fit params
            if param in self.cmdline_fit_params:
                checkbox.setChecked(True)
            # Default to F0, F1 if no command-line params specified
            elif not self.cmdline_fit_params and param in ['F0', 'F1']:
                checkbox.setChecked(True)

            # Add note if parameter not in original par file
            if param not in params_in_file:
                checkbox.setStyleSheet(get_added_param_style())
                checkbox.setToolTip(f"{param} not in original .par file (will be fitted from scratch)")

            self.param_checkboxes[param] = checkbox
            self.param_layout.addWidget(checkbox)

        # Add stretch at end
        self.param_layout.addStretch()

        self.available_params = all_params

        # Update status
        if all_params:
            status_msg = f"Found {len(params_in_file)} fittable parameters in .par file"
            if self.cmdline_fit_params:
                added = [p for p in self.cmdline_fit_params if p not in params_in_file]
                if added:
                    status_msg += f" (+ {len(added)} from --fit)"
            self.status_bar.showMessage(status_msg)
