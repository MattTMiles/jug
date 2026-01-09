"""
JUG Main Window - tempo2 plk-style interactive timing GUI.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QLabel, QPushButton, QStatusBar
)
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np


class MainWindow(QMainWindow):
    """Main window for JUG timing analysis GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JUG Timing Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.par_file = None
        self.tim_file = None
        self.residuals_us = None
        self.mjd = None
        self.errors_us = None
        self.rms_us = None
        
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
        
        # Left side: Large residual plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Residual', units='μs')
        self.plot_widget.setLabel('bottom', 'MJD (TDB)', units='days')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setBackground('w')  # White background
        
        # Configure plot appearance
        self.plot_widget.getAxis('left').setPen('k')
        self.plot_widget.getAxis('bottom').setPen('k')
        self.plot_widget.getAxis('left').setTextPen('k')
        self.plot_widget.getAxis('bottom').setTextPen('k')
        
        # Right side: Control panel
        control_panel = self._create_control_panel()
        
        # Add to main layout (plot takes 80%, controls 20%)
        main_layout.addWidget(self.plot_widget, stretch=4)
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self):
        """Create the control panel widget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("<b>Fit Controls</b>")
        title.setStyleSheet("font-size: 14px;")
        layout.addWidget(title)
        
        # Fit button
        self.fit_button = QPushButton("Fit")
        self.fit_button.setMinimumHeight(40)
        self.fit_button.clicked.connect(self.on_fit_clicked)
        self.fit_button.setEnabled(False)
        layout.addWidget(self.fit_button)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)
        
        layout.addSpacing(30)
        
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
            self._compute_initial_residuals()
    
    def _compute_initial_residuals(self):
        """Compute and display initial residuals."""
        from jug.residuals.simple_calculator import compute_residuals_simple
        
        self.status_bar.showMessage("Computing residuals...")
        
        try:
            result = compute_residuals_simple(
                par_file=self.par_file,
                tim_file=self.tim_file,
                verbose=False
            )
            
            # Store data
            self.mjd = result['tdb_mjd']
            self.residuals_us = result['residuals_us']
            self.errors_us = result.get('errors_us', None)
            self.rms_us = result['rms_us']
            
            # Update plot
            self._update_plot()
            
            # Enable controls
            self.fit_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            
            # Update statistics
            self.rms_label.setText(f"RMS: {self.rms_us:.6f} μs")
            self.ntoa_label.setText(f"TOAs: {len(self.mjd)}")
            
            # Update status
            self.status_bar.showMessage(
                f"Loaded {len(self.mjd)} TOAs, Prefit RMS = {self.rms_us:.6f} μs"
            )
            
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}")
            print(f"Error computing residuals: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_plot(self):
        """Update the residual plot."""
        self.plot_widget.clear()
        
        if self.mjd is None or self.residuals_us is None:
            return
        
        # Add error bars if available
        if self.errors_us is not None:
            error_bar = pg.ErrorBarItem(
                x=self.mjd,
                y=self.residuals_us,
                height=self.errors_us * 2,  # ±1σ
                beam=0.5,
                pen=pg.mkPen(color=(100, 100, 100, 100), width=1)
            )
            self.plot_widget.addItem(error_bar)
        
        # Scatter plot of residuals
        scatter = pg.ScatterPlotItem(
            x=self.mjd,
            y=self.residuals_us,
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120)
        )
        self.plot_widget.addItem(scatter)
        
        # Zero line
        self.plot_widget.addLine(y=0, pen=pg.mkPen('r', style=Qt.DashLine, width=2))
        
        # Auto-range
        self.plot_widget.autoRange()
    
    def on_fit_clicked(self):
        """Handle Fit button click."""
        # TODO: Implement fitting in Phase 2
        self.status_bar.showMessage("Fitting not yet implemented (Phase 2)")
    
    def on_reset_clicked(self):
        """Handle Reset button click."""
        if self.par_file and self.tim_file:
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
            "<p><b>Version:</b> 0.5.0 (GUI Phase 1)</p>"
            "<p><b>Framework:</b> PySide6 + pyqtgraph</p>"
        )
