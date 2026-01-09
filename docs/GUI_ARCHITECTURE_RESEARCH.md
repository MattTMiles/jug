# Modern Python GUI Architecture Research

## Research Questions
1. What are the current best Python GUI frameworks (2024-2026)?
2. Which frameworks are best for scientific/data visualization applications?
3. What are the architectural patterns for modern GUI apps?
4. What are the pros/cons of each framework?
5. Which integrates best with scientific Python stack (NumPy, JAX, etc.)?
6. What do similar scientific tools use?

## Starting Research...

## PART 1: MAJOR PYTHON GUI FRAMEWORKS (2024-2026)
====================================================

### 1. Qt-based Frameworks
--------------------------

#### PyQt6 / PySide6 (Qt 6)
**Status**: Industry standard, actively maintained
**License**: 
  - PyQt6: GPL/Commercial (Riverbank Computing)
  - PySide6: LGPL (official Qt binding)
**Latest Version**: Qt 6.x (2024+)

**Pros**:
+ Most mature, feature-complete GUI framework
+ Excellent documentation and community
+ Native look-and-feel on all platforms
+ Built-in designer tool (Qt Designer)
+ Comprehensive widget library
+ Strong MVC architecture support
+ Thread-safe with signals/slots
+ Excellent for complex applications

**Cons**:
- Steeper learning curve
- Larger application size
- GPL license (PyQt6) can be restrictive
- Some Qt concepts feel "heavy" for simple apps

**Scientific Use**:
+ pyqtgraph: Fast scientific plotting (1M+ points)
+ qwt: Scientific widgets
+ Large scientific community (CERN, NASA tools)
+ Integrates well with NumPy, Matplotlib

**Examples**: Spyder IDE, Orange, QGIS, LabPlot


#### PyQt5 / PySide2 (Qt 5)
**Status**: Legacy but still widely used
**Note**: Qt 5 in LTS until 2025, but Qt 6 is the future


### 2. Modern Web-based Frameworks
-----------------------------------

#### Streamlit
**Status**: Very popular for data apps
**License**: Apache 2.0
**Latest**: 1.x (2024)

**Pros**:
+ Extremely simple - pure Python, no HTML/CSS/JS
+ Auto-reloading on code changes
+ Great for dashboards and data apps
+ Built-in widgets for data science
+ Easy deployment (Streamlit Cloud)
+ Fast prototyping

**Cons**:
- Not a "true" desktop app (web-based)
- Limited customization
- Less control over layout
- Not good for complex interactions
- Requires server even for local use

**Scientific Use**:
+ Excellent for quick data exploration
+ Many ML/data science examples
+ Good for sharing with non-technical users

**Examples**: Data dashboards, ML demos, internal tools


#### Plotly Dash
**Status**: Enterprise-grade data apps
**License**: MIT
**Latest**: 2.x (2024)

**Pros**:
+ Built on React.js (modern web framework)
+ Beautiful, interactive plots (Plotly)
+ Good for dashboards
+ Enterprise support available
+ More customizable than Streamlit

**Cons**:
- Web-based (not native desktop)
- Steeper learning curve than Streamlit
- Requires understanding of callbacks
- Server required

**Scientific Use**:
+ Industry standard for interactive dashboards
+ Strong in financial, scientific domains

**Examples**: Corporate dashboards, data exploration tools


#### Gradio
**Status**: Rising star for ML interfaces
**License**: Apache 2.0
**Latest**: 4.x (2024)

**Pros**:
+ Designed for ML model interfaces
+ Extremely simple API
+ Auto-generates web UI
+ Great for demos

**Cons**:
- Very limited compared to full GUI frameworks
- Web-based
- Not suitable for complex applications


### 3. Native Python GUI Frameworks
------------------------------------

#### Tkinter
**Status**: Built into Python
**License**: Python Software Foundation License

**Pros**:
+ Ships with Python (no installation)
+ Simple for basic GUIs
+ Lightweight

**Cons**:
- Looks dated (1990s aesthetic)
- Limited widgets
- Poor for scientific plotting
- Not recommended for modern apps

**Verdict**: ❌ Not suitable for scientific application in 2024


#### wxPython (Phoenix)
**Status**: Mature but declining usage
**License**: wxWindows (LGPL-like)

**Pros**:
+ Native look on all platforms
+ Good widget library
+ Permissive license

**Cons**:
- Smaller community than Qt
- Documentation not as good
- Less modern than Qt 6
- Scientific plotting support limited

**Examples**: Some older scientific tools


#### Kivy
**Status**: Focus on mobile/touch
**License**: MIT

**Pros**:
+ Cross-platform (including mobile)
+ Modern look
+ Touch-friendly

**Cons**:
- Not native look
- Smaller community
- Less suitable for desktop scientific apps

**Verdict**: ❌ Not ideal for desktop timing software


### 4. Emerging/Alternative Frameworks
---------------------------------------

#### Dear PyGui
**Status**: Modern, GPU-accelerated
**License**: MIT
**Latest**: 1.x (2024)

**Pros**:
+ Very fast (GPU-accelerated)
+ Modern immediate-mode GUI
+ Great for real-time visualization
+ Simple API
+ Good for scientific/gaming apps

**Cons**:
- Younger, smaller community
- Different paradigm (immediate mode)
- Less comprehensive widget library
- Documentation still growing

**Scientific Use**:
+ Fast plotting (GPU-based)
+ Good for real-time data
+ Growing in scientific community


#### Flet
**Status**: Very new (2022+)
**License**: Apache 2.0
**Based on**: Flutter (Google)

**Pros**:
+ Modern, beautiful UI
+ Cross-platform (desktop, web, mobile)
+ Flutter's Material Design
+ Simple Python API

**Cons**:
- Very new (immature)
- Small community
- Stability concerns
- Limited scientific plotting


#### Textual
**Status**: TUI (Terminal UI) framework
**License**: MIT

**Pros**:
+ Beautiful terminal interfaces
+ No GUI dependencies
+ Great for SSH/remote work

**Cons**:
- Terminal-only (not GUI)
- Limited for scientific plotting

**Verdict**: ❌ Not suitable for residual visualization



## PART 2: SCIENTIFIC PLOTTING LIBRARIES
=========================================

### For Qt (PyQt6/PySide6)
---------------------------

#### pyqtgraph
**Status**: THE choice for fast scientific plotting in Qt
**License**: MIT
**Performance**: Can handle millions of points

**Pros**:
+ Extremely fast (GPU-accelerated where available)
+ Real-time plotting (scrolling, updating)
+ Interactive (zoom, pan, crosshairs)
+ Built specifically for Qt
+ Scientific widgets (ROI, image analysis)
+ Minimal dependencies (NumPy + Qt)

**Cons**:
- Less "polished" than Matplotlib
- Smaller feature set than Matplotlib
- Documentation could be better

**Perfect for**: Real-time data, large datasets, interactive visualization


#### Matplotlib (Qt backend)
**Status**: Standard plotting library
**License**: PSF-like

**Pros**:
+ Publication-quality plots
+ Huge feature set
+ Excellent documentation
+ Familiar to scientists
+ Can embed in Qt (FigureCanvasQTAgg)

**Cons**:
- Slower than pyqtgraph
- Not designed for real-time
- Higher memory usage
- Interaction can be clunky

**Good for**: Static plots, publication figures, complex plotting


#### VisPy
**Status**: High-performance visualization
**License**: BSD

**Pros**:
+ OpenGL-based (very fast)
+ Designed for large datasets
+ Modern architecture

**Cons**:
- More complex API
- Smaller community than pyqtgraph
- Overkill for simple plotting


### For Web-based
------------------

#### Plotly
**Status**: Industry standard for web
**License**: MIT

**Pros**:
+ Beautiful, interactive
+ Web-native
+ Great for dashboards
+ Easy sharing

**Cons**:
- Web-only
- Less suitable for desktop apps


## PART 3: ARCHITECTURE PATTERNS FOR SCIENTIFIC GUIs
=====================================================

### 1. Model-View-Controller (MVC)
-----------------------------------
**Best for**: Qt applications
**Pattern**:
- Model: Data (parameters, TOAs, residuals)
- View: Qt widgets (plot, tables, buttons)
- Controller: Business logic (fitting, I/O)

**Example**:
```
jug/gui/
  ├── models/
  │   ├── timing_model.py    (stores parameters, TOAs)
  │   └── fit_results.py     (stores fit results)
  ├── views/
  │   ├── main_window.py     (main GUI)
  │   ├── residual_plot.py   (plot widget)
  │   └── parameter_dialog.py
  └── controllers/
      ├── fit_controller.py  (handles fitting)
      └── io_controller.py   (file operations)
```

**Pros**:
+ Clean separation of concerns
+ Easy to test
+ Maintainable
+ Qt's signals/slots fit naturally

**Cons**:
- More boilerplate
- Can feel over-engineered for simple apps


### 2. Model-View-ViewModel (MVVM)
-----------------------------------
**Best for**: Reactive frameworks
**Pattern**:
- Model: Core data
- View: UI elements
- ViewModel: Mediates, provides data bindings

**Qt Implementation**: Use Qt's property system + signals/slots

**Pros**:
+ Excellent data binding
+ View doesn't know about model
+ Very testable

**Cons**:
- More complex than MVC
- Requires good understanding of reactive programming


### 3. Simple Layered Architecture
------------------------------------
**Best for**: Small to medium scientific apps
**Pattern**:
```
GUI Layer (Qt widgets)
    ↓
Application Layer (business logic)
    ↓
Core Layer (JUG fitting/residuals)
```

**Example**:
```
jug/gui/
  ├── main_window.py        (GUI + thin glue logic)
  ├── widgets/
  │   ├── residual_plot.py  (custom widget)
  │   └── fit_controls.py   (custom widget)
  └── app_logic.py          (wraps JUG core for GUI)

jug/fitting/                (existing - no GUI deps)
jug/residuals/              (existing - no GUI deps)
```

**Pros**:
+ Simple, pragmatic
+ Less boilerplate
+ Fast to develop
+ Easy to understand

**Cons**:
- Can get messy as app grows
- Business logic may leak into GUI


### 4. Plugin-based Architecture
----------------------------------
**Best for**: Extensible applications
**Pattern**: Core + Plugins

**Example**: Like Spyder, Orange
- Core provides GUI framework
- Plugins add analysis capabilities

**Pros**:
+ Very extensible
+ Users can add features
+ Modular

**Cons**:
- Complex to set up
- Overkill for single-purpose app


## PART 4: WHAT DO SIMILAR SCIENTIFIC TOOLS USE?
=================================================

### Astronomy/Astrophysics Tools

#### Ginga (astronomical image viewer)
- Framework: Qt (PyQt/PySide)
- Plotting: Custom OpenGL + Matplotlib fallback
- Architecture: Plugin-based
- **Takeaway**: Qt + custom visualization for performance

#### Glue (multi-dimensional data visualization)
- Framework: Qt (PyQt/PySide)
- Plotting: Matplotlib + custom OpenGL
- Architecture: Plugin-based, MVC-like
- **Takeaway**: Qt is standard in astronomy

#### DS9 (SAOImage DS9 - FITS viewer)
- Framework: Tk/Tcl (legacy)
- **Takeaway**: Old but functional - dated UI

#### Spyder (Python IDE)
- Framework: Qt (PyQt)
- Architecture: Plugin-based
- **Takeaway**: Qt for complex scientific tools

#### Orange (data mining suite)
- Framework: Qt (PyQt)
- Architecture: Widget-based, visual programming
- **Takeaway**: Qt + custom widgets


### Timing-Specific Tools

#### PINT (pulsar timing - Python)
- GUI: None (command-line only)
- Plotting: Matplotlib (when needed)
- **Takeaway**: CLI-first, plots via Matplotlib

#### tempo2
- GUI: PLK (pgplot-based)
- **Takeaway**: Legacy graphics, functional but dated

#### PSRCHIVE (pulsar data reduction)
- GUI: pgplot-based viewers
- **Takeaway**: Legacy but works


### Physics/Engineering

#### PyDM (Python Display Manager - accelerator control)
- Framework: Qt (PyQt)
- Architecture: Widget-based
- **Takeaway**: Qt standard for control room apps

#### Mantid (neutron/muon data analysis)
- Framework: Qt (PyQt)
- Plotting: Matplotlib + custom
- Architecture: Layered + plugins
- **Takeaway**: Large scientific app uses Qt


## PART 5: MODERN ARCHITECTURAL BEST PRACTICES
===============================================

### 1. Separation of Concerns
------------------------------
**Principle**: GUI code separate from core logic

**Implementation**:
```
jug/
  ├── core/           (NO GUI dependencies)
  │   ├── fitting/
  │   ├── residuals/
  │   └── io/
  └── gui/            (depends on core, not vice versa)
      └── ...
```

**Benefits**:
+ Can use core without GUI
+ Easier testing
+ CLI and GUI share same core


### 2. Reactive/Event-Driven
-----------------------------
**Principle**: Data changes trigger UI updates

**Qt Implementation**: Signals & Slots
```python
class TimingModel(QObject):
    parameters_changed = pyqtSignal()
    
    def set_parameter(self, name, value):
        self.params[name] = value
        self.parameters_changed.emit()

# In GUI:
model.parameters_changed.connect(plot.update_residuals)
```

**Benefits**:
+ Automatic UI synchronization
+ Loose coupling
+ Easy to add new features


### 3. Async/Threading for Long Operations
-------------------------------------------
**Principle**: Never block the UI

**Qt Implementation**: QThread or QRunnable
```python
class FitWorker(QRunnable):
    def run(self):
        result = fit_parameters_optimized(...)
        self.signals.result.emit(result)

# Start in thread pool
pool.start(FitWorker())
```

**Benefits**:
+ Responsive UI
+ Show progress bars
+ Can cancel operations


### 4. Dependency Injection
----------------------------
**Principle**: Pass dependencies, don't create them

**Example**:
```python
class MainWindow(QMainWindow):
    def __init__(self, fitter, residual_calculator):
        self.fitter = fitter
        self.calc = residual_calculator
```

**Benefits**:
+ Easy testing (mock dependencies)
+ Flexible configuration
+ Clear dependencies


### 5. State Management
------------------------
**Principle**: Centralized application state

**Implementation**:
```python
class AppState:
    def __init__(self):
        self.par_file = None
        self.tim_file = None
        self.parameters = {}
        self.fit_results = None
```

**Benefits**:
+ Single source of truth
+ Easy to serialize (save/load session)
+ Predictable updates



## PART 6: RECOMMENDATION FOR JUG TIMING GUI
=============================================

### Primary Recommendation: **PySide6 + pyqtgraph**
---------------------------------------------------

#### Why PySide6 over PyQt6?
- **License**: LGPL (more permissive than GPL)
- **Official**: Qt Company's official Python binding
- **Future-proof**: Better long-term support
- **Free**: No licensing concerns for distribution
- **API**: Nearly identical to PyQt6 (easy to switch if needed)

#### Why pyqtgraph over Matplotlib?
- **Performance**: 100-1000x faster for interactive plotting
- **Real-time**: Designed for live data updates
- **Scientific**: Built-in crosshairs, ROI, image analysis
- **Qt integration**: Native Qt widget, perfect fit
- **10,408 TOAs**: Will handle easily (tested to millions)

#### Architecture: **Simple Layered + Signals/Slots**

**Rationale**:
- Not complex enough for full MVC
- Not simple enough for monolithic code
- Leverage Qt's signals/slots for reactivity
- Keep core JUG code separate from GUI


### Recommended Structure
--------------------------

```
jug/
├── fitting/           (existing - NO GUI dependencies)
├── residuals/         (existing - NO GUI dependencies)
├── io/                (existing - NO GUI dependencies)
├── utils/             (existing - NO GUI dependencies)
│
└── gui/               (NEW - depends on core)
    ├── __init__.py
    ├── main.py        (entry point - jug-gui command)
    │
    ├── main_window.py (QMainWindow)
    │   - Menu bar
    │   - Layout management
    │   - Signal/slot connections
    │
    ├── widgets/
    │   ├── __init__.py
    │   ├── residual_plot.py    (pyqtgraph widget)
    │   ├── fit_controls.py     (buttons + stats panel)
    │   └── parameter_dialog.py (separate window)
    │
    ├── models/
    │   ├── __init__.py
    │   └── app_state.py        (application state)
    │
    └── workers/
        ├── __init__.py
        └── fit_worker.py       (QRunnable for fitting)
```


### Dependencies to Add
------------------------

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
gui = [
    "PySide6>=6.6.0",
    "pyqtgraph>=0.13.0",
]
```

Install with: `pip install -e .[gui]`


### Key Design Decisions
-------------------------

1. **Framework**: PySide6
   - Reason: LGPL license, official Qt binding, modern

2. **Plotting**: pyqtgraph
   - Reason: Fast, interactive, perfect for scientific data

3. **Architecture**: Simple layered + reactive (signals/slots)
   - Reason: Right complexity level for this app

4. **Threading**: QThreadPool + QRunnable
   - Reason: Non-blocking fitting, responsive UI

5. **State**: Centralized AppState class
   - Reason: Single source of truth, easy to manage

6. **No Qt Designer**: Code-based UI
   - Reason: More maintainable, version-controllable, flexible


### Implementation Strategy
----------------------------

#### Phase 1: Minimal Viable GUI (MVP)
**Goal**: Load data, view residuals
**Time**: ~4-6 hours

1. Create main window skeleton
2. Add pyqtgraph residual plot
3. Add file menu (Open .par, Open .tim)
4. Display prefit residuals
5. Test with J1909-3744 data

**Deliverable**: Can visualize timing residuals

#### Phase 2: Fit Integration
**Goal**: Run fits from GUI
**Time**: ~4-6 hours

1. Add fit control panel
2. Create fit worker (QRunnable)
3. Connect "Fit" button to worker
4. Update plot with postfit residuals
5. Display convergence stats

**Deliverable**: Can run fits and see results

#### Phase 3: Parameter Editing
**Goal**: Interactive parameter adjustment
**Time**: ~4-6 hours

1. Create parameter dialog (QDialog)
2. Populate from .par file
3. Connect edits to residual updates
4. Add debouncing (300ms delay)
5. Save .par functionality

**Deliverable**: Full interactive workflow

#### Phase 4: Polish & Features
**Goal**: Professional, publication-ready
**Time**: ~8-12 hours

1. Improve styling (colors, fonts)
2. Add keyboard shortcuts
3. Add plot export (PNG, PDF)
4. Error handling and validation
5. Progress indicators
6. Status bar with info

**Deliverable**: Production-ready GUI


### Alternative: Quick Prototype with Streamlit
------------------------------------------------

**If you want to test the idea VERY quickly** (~2 hours):

```python
import streamlit as st
from jug.fitting.optimized_fitter import fit_parameters_optimized

st.title("JUG Timing GUI (Prototype)")

par_file = st.file_uploader("Upload .par file")
tim_file = st.file_uploader("Upload .tim file")

if st.button("Fit"):
    result = fit_parameters_optimized(...)
    st.write(f"RMS: {result['final_rms']:.6f} μs")
    st.line_chart(result['residuals_us'])
```

**Pros**: Extremely fast to build
**Cons**: Not a real desktop app, limited customization

**Verdict**: Good for proof-of-concept, but PySide6 for production


## PART 7: POTENTIAL PITFALLS & SOLUTIONS
==========================================

### Pitfall 1: GUI Blocking During Fit
**Problem**: Fit takes 3-4 seconds, UI freezes
**Solution**: Use QThreadPool + QRunnable
```python
class FitWorker(QRunnable):
    def __init__(self, par_file, tim_file, params):
        super().__init__()
        self.signals = WorkerSignals()
        self.par_file = par_file
        self.tim_file = tim_file
        self.params = params
    
    def run(self):
        try:
            result = fit_parameters_optimized(
                self.par_file, self.tim_file, self.params
            )
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
```

### Pitfall 2: Memory Leaks with Plots
**Problem**: Recreating plots causes memory growth
**Solution**: Update data, don't recreate widgets
```python
# BAD:
def update_plot(self):
    self.plot_widget = pg.PlotWidget()  # Creates new widget each time!

# GOOD:
def update_plot(self, new_residuals):
    self.plot_item.setData(self.mjd, new_residuals)  # Just update data
```

### Pitfall 3: Slow Parameter Updates
**Problem**: Editing parameters recalculates residuals immediately, feels sluggish
**Solution**: Debouncing with QTimer
```python
self.update_timer = QTimer()
self.update_timer.setSingleShot(True)
self.update_timer.timeout.connect(self.update_residuals)

def on_parameter_changed(self):
    self.update_timer.start(300)  # Wait 300ms before updating
```

### Pitfall 4: Thread-Safety with NumPy/JAX
**Problem**: NumPy arrays shared between threads can cause corruption
**Solution**: Copy data when passing between threads
```python
def run(self):
    result = fit_parameters_optimized(...)
    # Copy arrays before emitting to main thread
    residuals_copy = np.array(result['residuals_us'])
    self.signals.result.emit(residuals_copy)
```

### Pitfall 5: Large File Loading
**Problem**: Loading 100k TOAs blocks UI
**Solution**: Load in background thread
```python
class FileLoadWorker(QRunnable):
    def run(self):
        toas = parse_tim_file(self.filename)
        self.signals.loaded.emit(toas)
```


## PART 8: COMPARISON TABLE - FINAL DECISION MATRIX
====================================================

| Criterion | PySide6 + pyqtgraph | Streamlit | Dear PyGui | wxPython |
|-----------|---------------------|-----------|------------|----------|
| **Maturity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Scientific Plotting** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Ease of Learning** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Desktop App** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Community** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **License** | ⭐⭐⭐⭐⭐ (LGPL) | ⭐⭐⭐⭐⭐ (Apache) | ⭐⭐⭐⭐⭐ (MIT) | ⭐⭐⭐⭐ |
| **Sci Ecosystem** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Long-term** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **TOTAL** | **48/50** | **37/50** | **37/50** | **33/50** |

**Winner**: **PySide6 + pyqtgraph**


## FINAL RECOMMENDATION
========================

### For JUG Timing GUI:

**Framework**: PySide6 6.6+
**Plotting**: pyqtgraph 0.13+
**Architecture**: Simple layered + reactive (signals/slots)
**Threading**: QThreadPool for long operations

### Reasoning:
1. **Industry standard** in scientific Python community
2. **Proven** in astronomy tools (Ginga, Glue, Spyder)
3. **Fast** - pyqtgraph handles 10k+ points easily
4. **Professional** - native look, publication-quality
5. **LGPL** - no licensing concerns
6. **Future-proof** - Qt 6 is actively developed
7. **Right complexity** - not overkill, not limiting

### Next Steps:
1. Add PySide6 + pyqtgraph to dependencies
2. Create basic main_window.py skeleton
3. Add pyqtgraph residual plot
4. Test with J1909-3744 data
5. Iterate from there

**Estimated Time to MVP**: 4-6 hours
**Estimated Time to Production**: 20-30 hours

