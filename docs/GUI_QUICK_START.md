# JUG GUI Quick Start Guide

**Created**: 2026-01-09
**Status**: Ready to begin Phase 1 (MVP)

This document provides quick-start instructions for continuing GUI development if interrupted.

---

## Current Status

✅ **Research Complete**: Framework decision made (PySide6 + pyqtgraph)
✅ **Design Complete**: Architecture and layout finalized
⏸️ **Phase 1 (MVP)**: Ready to start

**Next Step**: Create main window skeleton

---

## Technology Stack

- **Framework**: PySide6 6.6+
- **Plotting**: pyqtgraph 0.13+
- **Architecture**: Simple layered + reactive (signals/slots)
- **Threading**: QThreadPool

---

## Quick Start: Phase 1 (MVP)

### 1. Install Dependencies

```bash
cd /home/mmiles/soft/jug
pip install PySide6>=6.6.0 pyqtgraph>=0.13.0
```

Or add to `pyproject.toml`:
```toml
[project.optional-dependencies]
gui = [
    "PySide6>=6.6.0",
    "pyqtgraph>=0.13.0",
]
```

Then: `pip install -e .[gui]`

### 2. Create Directory Structure

```bash
mkdir -p jug/gui/widgets jug/gui/models jug/gui/workers
touch jug/gui/__init__.py
touch jug/gui/main.py
touch jug/gui/main_window.py
touch jug/gui/widgets/__init__.py
```

### 3. Create Main Entry Point

**File**: `jug/gui/main.py`

```python
#!/usr/bin/env python3
"""
JUG GUI entry point.
"""
import sys
from PySide6.QtWidgets import QApplication
from jug.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("JUG Timing")
    app.setOrganizationName("Pulsar Timing")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

### 4. Create Main Window Skeleton

**File**: `jug/gui/main_window.py`

```python
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QFileDialog, QLabel, QPushButton
)
from PySide6.QtCore import Qt
import pyqtgraph as pg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JUG Timing Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.par_file = None
        self.tim_file = None
        self.residuals = None
        self.mjd = None
        self.errors = None
        
        self._setup_ui()
        self._create_menu_bar()
    
    def _setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left side: Large plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Residual', units='μs')
        self.plot_widget.setLabel('bottom', 'MJD', units='days')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Right side: Control panel
        control_panel = self._create_control_panel()
        
        # Add to main layout
        main_layout.addWidget(self.plot_widget, stretch=4)
        main_layout.addWidget(control_panel, stretch=1)
    
    def _create_control_panel(self):
        """Create the control panel widget."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Fit button
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.on_fit_clicked)
        self.fit_button.setEnabled(False)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.reset_button.setEnabled(False)
        
        # Stats display
        self.rms_label = QLabel("RMS: -- μs")
        self.iter_label = QLabel("Iterations: --")
        
        # Add widgets to layout
        layout.addWidget(QLabel("<b>Fit Controls</b>"))
        layout.addWidget(self.fit_button)
        layout.addWidget(self.reset_button)
        layout.addSpacing(20)
        layout.addWidget(QLabel("<b>Statistics</b>"))
        layout.addWidget(self.rms_label)
        layout.addWidget(self.iter_label)
        layout.addStretch()
        
        return panel
    
    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_par_action = file_menu.addAction("Open .par...")
        open_par_action.triggered.connect(self.on_open_par)
        
        open_tim_action = file_menu.addAction("Open .tim...")
        open_tim_action.triggered.connect(self.on_open_tim)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        params_action = view_menu.addAction("Parameters...")
        params_action.triggered.connect(self.on_show_parameters)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        fit_action = tools_menu.addAction("Run Fit")
        fit_action.triggered.connect(self.on_fit_clicked)
    
    def on_open_par(self):
        """Handle Open .par file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open .par File", "", "Par Files (*.par);;All Files (*)"
        )
        if filename:
            self.par_file = filename
            self.statusBar().showMessage(f"Loaded: {filename}")
            self._check_ready_to_compute()
    
    def on_open_tim(self):
        """Handle Open .tim file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open .tim File", "", "Tim Files (*.tim);;All Files (*)"
        )
        if filename:
            self.tim_file = filename
            self.statusBar().showMessage(f"Loaded: {filename}")
            self._check_ready_to_compute()
    
    def _check_ready_to_compute(self):
        """Check if we have both files and can compute residuals."""
        if self.par_file and self.tim_file:
            self._compute_initial_residuals()
    
    def _compute_initial_residuals(self):
        """Compute and display initial residuals."""
        from pathlib import Path
        from jug.residuals.simple_calculator import compute_residuals_simple
        
        try:
            result = compute_residuals_simple(
                par_file=Path(self.par_file),
                tim_file=Path(self.tim_file),
                verbose=False
            )
            
            self.mjd = result['tdb_mjd']
            self.residuals = result['residuals_us']
            self.errors = result.get('errors_us', None)
            
            self._update_plot()
            self.fit_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            
            rms = result['rms_us']
            self.rms_label.setText(f"RMS: {rms:.6f} μs")
            self.statusBar().showMessage(
                f"Loaded {len(self.mjd)} TOAs, RMS = {rms:.6f} μs"
            )
            
        except Exception as e:
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def _update_plot(self):
        """Update the residual plot."""
        self.plot_widget.clear()
        
        if self.errors is not None:
            # Plot with error bars
            error_bar = pg.ErrorBarItem(
                x=self.mjd, y=self.residuals,
                height=self.errors * 2,  # ±1σ
                beam=0.5
            )
            self.plot_widget.addItem(error_bar)
        
        # Scatter plot
        scatter = pg.ScatterPlotItem(
            x=self.mjd, y=self.residuals,
            size=5, pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120)
        )
        self.plot_widget.addItem(scatter)
        
        # Zero line
        self.plot_widget.addLine(y=0, pen=pg.mkPen('r', style=Qt.DashLine))
    
    def on_fit_clicked(self):
        """Handle Fit button click."""
        # TODO: Implement fitting in Phase 2
        self.statusBar().showMessage("Fitting not yet implemented (Phase 2)")
    
    def on_reset_clicked(self):
        """Handle Reset button click."""
        if self.par_file and self.tim_file:
            self._compute_initial_residuals()
    
    def on_show_parameters(self):
        """Show parameter editor dialog."""
        # TODO: Implement in Phase 3
        self.statusBar().showMessage("Parameter editor not yet implemented (Phase 3)")
```

### 5. Add CLI Entry Point

Update `pyproject.toml`:

```toml
[project.scripts]
jug-compute-residuals = "jug.scripts.compute_residuals:main"
jug-fit = "jug.scripts.fit_parameters:main"
jug-gui = "jug.gui.main:main"  # ADD THIS LINE
```

Then reinstall: `pip install -e .`

### 6. Test MVP

```bash
jug-gui
```

Then:
1. File → Open .par (select data/pulsars/J1909-3744_tdb.par)
2. File → Open .tim (select data/pulsars/J1909-3744.tim)
3. Should see 10,408 TOAs plotted

---

## Phase 2: Fit Integration (Next Steps)

After MVP works, add fitting:

1. Create `jug/gui/workers/fit_worker.py`:
```python
from PySide6.QtCore import QRunnable, QObject, Signal
from pathlib import Path

class WorkerSignals(QObject):
    result = Signal(dict)
    error = Signal(str)
    finished = Signal()

class FitWorker(QRunnable):
    def __init__(self, par_file, tim_file, fit_params):
        super().__init__()
        self.signals = WorkerSignals()
        self.par_file = par_file
        self.tim_file = tim_file
        self.fit_params = fit_params
    
    def run(self):
        from jug.fitting.optimized_fitter import fit_parameters_optimized
        try:
            result = fit_parameters_optimized(
                par_file=Path(self.par_file),
                tim_file=Path(self.tim_file),
                fit_params=self.fit_params,
                verbose=False
            )
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()
```

2. Update `on_fit_clicked()` in `main_window.py` to use worker

---

## Documentation References

- **Comprehensive research**: `docs/GUI_ARCHITECTURE_RESEARCH.md`
- **Implementation guide**: `docs/JUG_implementation_guide.md` (Milestone 5)
- **Progress tracker**: `docs/JUG_PROGRESS_TRACKER.md` (M5 section)

---

## Tips for Continuation

1. **If stuck on Qt**: Check Qt documentation: https://doc.qt.io/qtforpython/
2. **If stuck on pyqtgraph**: Check examples: http://www.pyqtgraph.org/
3. **Common issues**: See "Potential Pitfalls" in GUI_ARCHITECTURE_RESEARCH.md
4. **Testing**: Use J1909-3744 data (data/pulsars/)

---

## Expected Timeline

- **Phase 1 (MVP)**: 4-6 hours → Load data, view residuals
- **Phase 2 (Fit)**: 4-6 hours → Run fits from GUI
- **Phase 3 (Params)**: 4-6 hours → Interactive parameter editing
- **Phase 4 (Polish)**: 8-12 hours → Professional appearance

**Total to production**: 20-30 hours (~2 weeks part-time)

---

**Status**: Ready to begin! Start with Phase 1 MVP.
