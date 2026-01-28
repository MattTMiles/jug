"""
JUG Main Window - tempo2 plk-style interactive timing GUI.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QStatusBar,
    QCheckBox, QGroupBox, QMessageBox, QProgressDialog
)
from PySide6.QtCore import Qt, QThreadPool
import pyqtgraph as pg
import numpy as np


class MainWindow(QMainWindow):
    """Main window for JUG timing analysis GUI."""

    def __init__(self, fit_params=None):
        super().__init__()
        self.setWindowTitle("JUG Timing Analysis")
        self.setGeometry(100, 100, 1152, 768)
        self.setMinimumSize(1152, 768)
        self.setMaximumSize(2304, 1536)

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

        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)  # Session + compute/fit

        # Setup UI
        self._setup_ui()
        self._create_menu_bar()
        self._create_status_bar()
    
    def _setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout: Plot on left, controls on right
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Plot area with title
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 5, 0, 0)
        plot_layout.setSpacing(15)
        
        # Title label above plot (pulsar name and RMS)
        self.plot_title_label = QLabel("")
        self.plot_title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px; margin-bottom: 5px;")
        self.plot_title_label.setAlignment(Qt.AlignCenter)
        plot_layout.addWidget(self.plot_title_label)
        
        # Large residual plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Residual', units='μs')
        self.plot_widget.setLabel('bottom', 'MJD (TDB)', units='days')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('w')  # White background
        
        # Configure plot appearance for all axes
        for axis in ['left', 'bottom', 'top', 'right']:
            ax = self.plot_widget.getPlotItem().getAxis(axis)
            ax.setPen(pg.mkPen('k', width=2))
            if axis in ['left', 'bottom']:
                ax.setTextPen('k')
            else:
                # Top and right: show border but no tick labels
                ax.setStyle(showValues=False)
                ax.show()
        
        # Disable SI prefix scaling on x-axis (show days, not kdays)
        self.plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
        
        # Add border around view box with thicker pen
        self.plot_widget.getPlotItem().getViewBox().setBorder(pg.mkPen('k', width=2))
        
        plot_layout.addWidget(self.plot_widget)
        
        # Right side: Control panel
        control_panel = self._create_control_panel()
        
        # Add to main layout (plot takes 80%, controls 20%)
        main_layout.addWidget(plot_container, stretch=4)
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self):
        """Create the control panel widget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Parameter selection (will be populated when par file is loaded)
        self.param_group = QGroupBox("Parameters to Fit")
        self.param_layout = QVBoxLayout()
        self.param_group.setLayout(self.param_layout)
        layout.addWidget(self.param_group)

        self.param_checkboxes = {}

        # Initially show placeholder message
        self.param_placeholder = QLabel("Load .par file to see available parameters")
        self.param_placeholder.setStyleSheet("color: gray; font-style: italic;")
        self.param_layout.addWidget(self.param_placeholder)

        layout.addSpacing(10)

        # Fit button
        self.fit_button = QPushButton("Run Fit")
        self.fit_button.setMinimumHeight(40)
        self.fit_button.clicked.connect(self.on_fit_clicked)
        self.fit_button.setEnabled(False)
        layout.addWidget(self.fit_button)

        # Fit Report button
        self.fit_report_button = QPushButton("Fit Report")
        self.fit_report_button.setMinimumHeight(40)
        self.fit_report_button.clicked.connect(self.on_show_fit_report)
        self.fit_report_button.setEnabled(False)
        layout.addWidget(self.fit_report_button)

        # Reset button
        self.reset_button = QPushButton("Reset to Prefit")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)

        # Fit Window to Data button
        self.fit_window_button = QPushButton("Fit Window to Data")
        self.fit_window_button.setMinimumHeight(40)
        self.fit_window_button.clicked.connect(self.on_zoom_fit)
        self.fit_window_button.setEnabled(False)
        layout.addWidget(self.fit_window_button)

        layout.addSpacing(20)

        # Statistics section
        stats_title = QLabel("<b>Statistics</b>")
        stats_title.setStyleSheet("font-size: 14px;")
        layout.addWidget(stats_title)

        self.rms_label = QLabel("RMS: --")
        self.rms_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.rms_label)

        self.iter_label = QLabel("Iterations: --")
        self.iter_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.iter_label)

        self.ntoa_label = QLabel("TOAs: --")
        self.ntoa_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.ntoa_label)

        self.chi2_label = QLabel("χ²/dof: --")
        self.chi2_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.chi2_label)

        # Stretch to push everything to top
        layout.addStretch()

        return panel
    
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
        
        # Update statistics
        self.rms_label.setText(f"RMS: {self.rms_us:.6f} μs")
        self.ntoa_label.setText(f"TOAs: {len(self.mjd)}")
        
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
        
        # Update or create scatter plot
        if self.scatter_item is None:
            # First time: create scatter item
            self.scatter_item = pg.ScatterPlotItem(
                size=5,
                pen=pg.mkPen(None),
                brush=pg.mkBrush(100, 100, 255, 120)
            )
            self.plot_widget.addItem(self.scatter_item)
        
        # Update scatter data (fast - no recreation)
        self.scatter_item.setData(x=self.mjd, y=self.residuals_us)
        
        # Update or create error bars (optional for performance)
        if self.errors_us is not None:
            if self.error_bar_item is None:
                # First time: create error bar item
                self.error_bar_item = pg.ErrorBarItem(
                    beam=0.5,
                    pen=pg.mkPen(color=(100, 100, 100, 100), width=1)
                )
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
        
        # Add zero line (only once)
        if self.zero_line is None:
            self.zero_line = self.plot_widget.addLine(
                y=0,
                pen=pg.mkPen('r', style=Qt.DashLine, width=2)
            )
        
        # Auto-range only when requested (not every update!)
        if auto_range:
            self.plot_widget.autoRange()
    
    def _update_plot_title(self):
        """Update the plot title with pulsar name and RMS."""
        pulsar_str = self.pulsar_name if self.pulsar_name else "Unknown Pulsar"
        rms_str = f"{self.rms_us:.6f} μs" if self.rms_us is not None else "--"
        self.plot_title_label.setText(f"{pulsar_str}  |  RMS: {rms_str}")
    
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
            
            # Update statistics
            self.rms_label.setText(f"RMS: {self.rms_us:.6f} μs")
            self.iter_label.setText(f"Iterations: {fit_result['iterations']}")

            # Calculate chi-squared if we have errors
            if self.errors_us is not None:
                n_toas = len(self.mjd)
                n_params = len(fit_result['final_params'])
                dof = n_toas - n_params
                # Calculate proper chi-squared with postfit residuals
                chi2 = np.sum((self.residuals_us / self.errors_us) ** 2)
                chi2_dof = chi2 / dof
                self.chi2_label.setText(f"χ²/dof: {chi2_dof:.2f}")

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
        Show fit results in a dialog.

        Parameters
        ----------
        result : dict
            Fit results
        """
        # Build title with pulsar name and RMS
        pulsar_str = self.pulsar_name if self.pulsar_name else 'Unknown'
        title = f"{pulsar_str} - RMS: {result['final_rms']:.6f} μs"
        
        msg = "<h3>Fit Results</h3>"
        msg += "<table border='1' cellpadding='5' style='border-collapse: collapse;'>"
        msg += "<tr><th>Parameter</th><th>New Value</th><th>Previous Value</th><th>Change</th><th>Uncertainty</th></tr>"

        for param, new_value in result['final_params'].items():
            uncertainty = result['uncertainties'][param]
            
            # Get previous value (0.0 if not in original par file)
            prev_value = self.initial_params.get(param, 0.0)
            change = new_value - prev_value
            
            # Format based on parameter type
            if param.startswith('F'):
                if param == 'F0':
                    new_val_str = f"{new_value:.15f} Hz"
                    prev_val_str = f"{prev_value:.15f} Hz"
                    change_str = f"{change:.15f} Hz"
                else:
                    new_val_str = f"{new_value:.6e} Hz/s"
                    prev_val_str = f"{prev_value:.6e} Hz/s"
                    change_str = f"{change:.6e} Hz/s"
            elif param.startswith('DM'):
                new_val_str = f"{new_value:.10f} pc cm⁻³"
                prev_val_str = f"{prev_value:.10f} pc cm⁻³"
                change_str = f"{change:.10f} pc cm⁻³"
            else:
                new_val_str = f"{new_value:.6e}"
                prev_val_str = f"{prev_value:.6e}"
                change_str = f"{change:.6e}"
            
            unc_str = f"{uncertainty:.2e}"
            
            msg += f"<tr><td><b>{param}</b></td><td>{new_val_str}</td><td>{prev_val_str}</td><td>{change_str}</td><td>{unc_str}</td></tr>"

        msg += "</table><br>"
        msg += f"<b>Final RMS:</b> {result['final_rms']:.6f} μs<br>"
        msg += f"<b>Iterations:</b> {result['iterations']}<br>"
        msg += f"<b>Converged:</b> {result['converged']}<br>"
        msg += f"<b>Time:</b> {result['total_time']:.2f} s"

        msgbox = QMessageBox(self)
        msgbox.setWindowTitle(title)
        msgbox.setTextFormat(Qt.RichText)
        msgbox.setText(msg)
        
        # Set dialog to 60% of screen height
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        dialog_height = int(screen.height() * 0.6)
        dialog_width = 800  # Fixed reasonable width
        
        # Must set size after show() for QMessageBox
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

            # Update statistics
            self.rms_label.setText(f"RMS: {self.rms_us:.6f} μs")
            self.iter_label.setText("Iterations: --")
            self.chi2_label.setText("χ²/dof: --")
            self.status_bar.showMessage("Reset to prefit residuals")
        elif self.par_file and self.tim_file:
            self._compute_initial_residuals()
    
    def on_show_parameters(self):
        """Show parameter editor dialog."""
        # TODO: Implement in Phase 3
        self.status_bar.showMessage("Parameter editor not yet implemented (Phase 3)")
    
    def on_zoom_fit(self):
        """Zoom plot to fit data."""
        self.plot_widget.autoRange()
    
    def on_about(self):
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About JUG",
            "<h3>JUG Timing Analysis</h3>"
            "<p>JAX-based pulsar timing software</p>"
            "<p>Fast, independent, and extensible</p>"
            "<p><b>Version:</b> 0.5.0 (GUI Phase 2)</p>"
            "<p><b>Framework:</b> PySide6 + pyqtgraph</p>"
        )

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

        # Clear existing checkboxes
        for checkbox in self.param_checkboxes.values():
            checkbox.deleteLater()
        self.param_checkboxes.clear()

        if self.param_placeholder:
            self.param_placeholder.deleteLater()
            self.param_placeholder = None

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
                checkbox.setStyleSheet("color: #0066cc;")  # Blue for added params
                checkbox.setToolTip(f"{param} not in original .par file (will be fitted from scratch)")

            self.param_checkboxes[param] = checkbox
            self.param_layout.addWidget(checkbox)

        self.available_params = all_params

        # Update status
        if all_params:
            status_msg = f"Found {len(params_in_file)} fittable parameters in .par file"
            if self.cmdline_fit_params:
                added = [p for p in self.cmdline_fit_params if p not in params_in_file]
                if added:
                    status_msg += f" (+ {len(added)} from --fit)"
            self.status_bar.showMessage(status_msg)
