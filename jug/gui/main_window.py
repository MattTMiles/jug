"""
JUG Main Window - tempo2 plk-style interactive timing GUI.

Modern redesign inspired by Linear, Raycast, and Notion.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QStatusBar,
    QCheckBox, QGroupBox, QMessageBox, QProgressDialog,
    QFrame, QScrollArea, QSizePolicy, QApplication, QComboBox
)
from PySide6.QtCore import Qt, QThreadPool, QEvent, QPointF, QTimer
from PySide6.QtGui import QFont, QCursor
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
    Spacing,
    LightTheme,
    SynthwaveTheme,
    set_theme,
    is_dark_mode,
    PlotTheme,
    get_synthwave_variant,
    toggle_synthwave_variant,
    get_synthwave_rms_color,
    get_dynamic_accent_primary,
    get_light_variant,
    toggle_light_variant,
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
        
        # Connect to view range changes to scale error bar caps
        self.plot_widget.getPlotItem().getViewBox().sigRangeChanged.connect(self._on_view_range_changed)

        # Add subtle rounded corners effect via container
        self.plot_frame = QFrame()
        self.plot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.PLOT_BG};
                border: 1px solid {get_border_strong()};
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

        # Solver mode dropdown
        solver_layout = QHBoxLayout()
        solver_layout.setSpacing(8)
        self.solver_label = QLabel("Solver:")
        self.solver_label.setStyleSheet(f"""
            font-size: 12px;
            color: {Colors.TEXT_SECONDARY};
        """)
        solver_layout.addWidget(self.solver_label)

        self.solver_combo = QComboBox()
        self.solver_combo.addItem("Exact (reproducible)", "exact")
        self.solver_combo.addItem("Fast", "fast")
        self.solver_combo.setCurrentIndex(0)  # Default to exact
        self.solver_combo.currentIndexChanged.connect(self._on_solver_mode_changed)
        self.solver_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {Colors.SURFACE};
                border: 1px solid {get_border_subtle()};
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
                color: {Colors.TEXT_PRIMARY};
            }}
            QComboBox:hover {{
                border-color: {get_border_strong()};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
        """)
        self.solver_combo.setToolTip(
            "Exact: SVD-based, bit-for-bit reproducible\n"
            "Fast: QR-based, faster but may differ slightly"
        )
        solver_layout.addWidget(self.solver_combo)
        solver_layout.addStretch()
        layout.addLayout(solver_layout)

        # Secondary buttons with consistent styling
        secondary_style = get_secondary_button_style()

        self.fit_report_button = QPushButton("📊  Fit Report")
        self.fit_report_button.setStyleSheet(secondary_style)
        self.fit_report_button.setCursor(Qt.PointingHandCursor)
        self.fit_report_button.clicked.connect(self.on_show_fit_report)
        self.fit_report_button.setEnabled(False)
        layout.addWidget(self.fit_report_button)

        self.reset_button = QPushButton("↺  Restart")
        self.reset_button.setStyleSheet(secondary_style)
        self.reset_button.setCursor(Qt.PointingHandCursor)
        self.reset_button.clicked.connect(self.on_restart_clicked)
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
                border: 1px solid {get_border_subtle()};
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

        # Create stat rows with label + value (store both for theme updates)
        self.rms_label_text, self.rms_label = self._create_stat_row(stats_layout, "RMS", "--")
        # Highlight RMS value with teal (emphasis color) - uses theme-aware accent
        rms_color = get_rms_emphasis_color()
        self.rms_label.setStyleSheet(f"""
            font-family: monospace;
            font-size: 14px;
            font-weight: 600;
            color: {rms_color};
            background: transparent;
            border: none;
        """)
        self.iter_label_text, self.iter_label = self._create_stat_row(stats_layout, "Iterations", "--")
        self.ntoa_label_text, self.ntoa_label = self._create_stat_row(stats_layout, "TOAs", "--")
        self.chi2_label_text, self.chi2_label = self._create_stat_row(stats_layout, "χ²/dof", "--")

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
                border-left: 1px solid {get_border_subtle()};
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
        self.rms_us = result['rms_us']
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
        self.status_bar.showMessage(
            f"Loaded {len(self.mjd)} TOAs, Prefit RMS = {self.rms_us:.6f} μs"
        )

        # Schedule JAX warmup in background (avoids first-fit lag)
        self._schedule_jax_warmup()
    
    def _schedule_jax_warmup(self):
        """Schedule JAX warmup in background to avoid first-fit lag."""
        if self.mjd is None or len(self.mjd) == 0:
            return

        from jug.gui.workers.warmup_worker import WarmupWorker

        # Create warmup worker with actual data size
        worker = WarmupWorker(n_toas=len(self.mjd))
        worker.signals.progress.connect(lambda msg: None)  # Silent warmup
        worker.signals.finished.connect(lambda: None)

        # Run in thread pool (low priority)
        self.thread_pool.start(worker)

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
            # Recalculate RMS for filtered data
            self.rms_us = np.sqrt(np.mean(self.residuals_us**2))
        else:
            # No deletions - use all data
            self.mjd = full_mjd
            self.residuals_us = full_residuals
            self.postfit_residuals_us = full_residuals.copy()
            self.errors_us = full_errors
            self.rms_us = result['rms_us']
        
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

        # Colors for styling (use dynamic accent for theme/variant awareness)
        text_neutral = "#111827"
        text_muted = "#6b7280"
        accent = get_dynamic_accent_primary()
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
            
            # Recalculate RMS
            self.rms_us = np.sqrt(np.mean(self.residuals_us**2))
            
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
                border: 1px solid {get_border_strong()};
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
                border-left: 1px solid {get_border_subtle()};
            }}
        """)

        # Update stats card styling
        # Find the stats card and update it
        for widget in self.control_container.findChildren(QWidget):
            if hasattr(widget, 'minimumHeight') and widget.minimumHeight() == 160:
                widget.setStyleSheet(f"""
                    QWidget {{
                        background-color: {Colors.SURFACE};
                        border: 1px solid {get_border_subtle()};
                        border-radius: 12px;
                    }}
                    QLabel {{
                        background: transparent;
                        border: none;
                    }}
                """)

        # Update section titles
        self.action_title.setStyleSheet(get_section_title_style())
        self.stats_title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {Colors.TEXT_PRIMARY};
            background: transparent;
            border: none;
            padding-bottom: 8px;
        """)
        self.params_header_label.setStyleSheet(get_section_title_style())

        # Update stat labels (not values)
        stat_label_style = f"""
            font-size: 12px;
            color: {Colors.TEXT_SECONDARY};
            background: transparent;
            border: none;
        """
        self.rms_label_text.setStyleSheet(stat_label_style)
        self.iter_label_text.setStyleSheet(stat_label_style)
        self.ntoa_label_text.setStyleSheet(stat_label_style)
        self.chi2_label_text.setStyleSheet(stat_label_style)

        # Update stat values (except RMS which has special styling)
        stat_value_style = f"""
            font-family: monospace;
            font-size: 14px;
            color: {Colors.TEXT_PRIMARY};
            background: transparent;
            border: none;
        """
        self.iter_label.setStyleSheet(stat_value_style)
        self.ntoa_label.setStyleSheet(stat_value_style)
        self.chi2_label.setStyleSheet(stat_value_style)

        # Update scatter plot colors (always use primary - variant controls what "primary" means)
        colors = get_scatter_colors()
        if self.scatter_item is not None:
            self.scatter_item.setBrush(pg.mkBrush(*colors['primary']))

        # Update error bar colors (always visible, muted)
        if self.error_bar_item is not None:
            error_color = PlotTheme.get_error_bar_color()
            self.error_bar_item.setData(pen=pg.mkPen(color=error_color, width=2.0))

        # Update zero line if visible
        if self.zero_line is not None:
            self.plot_widget.removeItem(self.zero_line)
            self.zero_line = create_zero_line(self.plot_widget)

        # Update RMS label color (teal in light mode, variant-aware in dark mode)
        rms_color = get_rms_emphasis_color()

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

    def _on_solver_mode_changed(self, index):
        """Handle solver mode dropdown change."""
        self.solver_mode = self.solver_combo.itemData(index)
        mode_name = "Exact" if self.solver_mode == "exact" else "Fast"
        self.status_bar.showMessage(f"Solver mode: {mode_name}")

    def _on_view_range_changed(self, view_box, ranges):
        """Handle view range changes to scale error bar caps.

        OPTIMIZED: Uses throttling to coalesce rapid updates during pan/zoom.
        Only updates beam width, NOT the x/y/height arrays (O(1) vs O(N)).
        This makes panning/zooming smooth, especially over X11.
        """
        if self.error_bar_item is None or self.mjd is None:
            return

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
        min_beam = 0.1    # Minimum 0.1 days
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
        
        # Recalculate RMS
        if len(self.residuals_us) > 0:
            self.rms_us = np.sqrt(np.mean(self.residuals_us**2))
        else:
            self.rms_us = 0.0
        
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
