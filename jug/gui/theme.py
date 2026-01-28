"""
JUG Modern Theme - Pulsar Timing Analysis
Inspired by Linear, Raycast, Notion, and modern SaaS apps.

Design Philosophy:
- Sophisticated, inviting color palette (not sterile lab equipment)
- Generous spacing and breathing room
- Subtle depth with shadows and rounded corners
- Delightful micro-interactions
"""

# =============================================================================
# COLOR PALETTE
# =============================================================================

# =============================================================================
# THEME DEFINITIONS
# =============================================================================

class LightTheme:
    """
    Custom Light - Cream & Teal palette.
    Light, warm theme with earthy accents.
    """
    NAME = "light"

    # Background colors - cream-based light theme
    BG_PRIMARY = "#F3EFE6"
    BG_SECONDARY = "#ebe6db"
    BG_TERTIARY = "#e3ded3"
    BG_ELEVATED = "#ffffff"

    # Surface colors
    SURFACE = "#ffffff"
    SURFACE_HOVER = "#f9f7f3"
    SURFACE_BORDER = "#CCB999"

    # Text colors
    TEXT_PRIMARY = "#1F4D4A"
    TEXT_SECONDARY = "#3d6b67"
    TEXT_MUTED = "#788176"
    TEXT_ACCENT = "#2b4162"

    # Accent colors
    ACCENT_PRIMARY = "#1F4D4A"
    ACCENT_SECONDARY = "#2b4162"
    ACCENT_TERTIARY = "#CCB999"
    ACCENT_SUCCESS = "#1F4D4A"
    ACCENT_WARNING = "#CCB999"
    ACCENT_ERROR = "#9e2a2b"

    # Plot-specific colors
    PLOT_BG = "#ffffff"
    PLOT_GRID = "#e3ded3"
    PLOT_AXES = "#1F4D4A"
    PLOT_ZERO_LINE = "#9e2a2b"

    # Data visualization
    DATA_POINTS = "#2b4162"
    DATA_POINTS_GLOW = "#4a6fa5"
    ERROR_BARS = "#78817680"

    # Button states
    BTN_PRIMARY_BG = "#1F4D4A"
    BTN_PRIMARY_HOVER = "#2a635f"
    BTN_PRIMARY_PRESSED = "#163836"
    BTN_PRIMARY_TEXT = "#F3EFE6"

    BTN_SECONDARY_BG = "#ffffff"
    BTN_SECONDARY_HOVER = "#f9f7f3"
    BTN_SECONDARY_PRESSED = "#ebe6db"
    BTN_SECONDARY_TEXT = "#1F4D4A"

    # Special states
    DISABLED_BG = "#e3ded3"
    DISABLED_TEXT = "#a8a49c"
    FOCUS_RING = "#1F4D4A40"


class SynthwaveTheme:
    """
    Synthwave '84 - Retro neon dark theme.
    Inspired by the VS Code Synthwave '84 extension.
    """
    NAME = "synthwave"

    # Background colors - deep purple/dark
    BG_PRIMARY = "#262335"
    BG_SECONDARY = "#2a2139"
    BG_TERTIARY = "#34294f"
    BG_ELEVATED = "#3b2d5e"

    # Surface colors
    SURFACE = "#34294f"
    SURFACE_HOVER = "#3b2d5e"
    SURFACE_BORDER = "#495495"

    # Text colors
    TEXT_PRIMARY = "#f4eee4"
    TEXT_SECONDARY = "#b6b1c4"
    TEXT_MUTED = "#848bbd"
    TEXT_ACCENT = "#ff7edb"

    # Accent colors - neon
    ACCENT_PRIMARY = "#ff7edb"    # Neon pink
    ACCENT_SECONDARY = "#36f9f6"  # Neon cyan
    ACCENT_TERTIARY = "#fede5d"   # Neon yellow
    ACCENT_SUCCESS = "#72f1b8"    # Neon green
    ACCENT_WARNING = "#fede5d"    # Neon yellow
    ACCENT_ERROR = "#fe4450"      # Neon red

    # Plot-specific colors
    PLOT_BG = "#1a1525"           # Darker purple for plot
    PLOT_GRID = "#34294f"
    PLOT_AXES = "#b6b1c4"
    PLOT_ZERO_LINE = "#fe4450"

    # Data visualization - neon cyan
    DATA_POINTS = "#36f9f6"
    DATA_POINTS_GLOW = "#72f1b8"
    ERROR_BARS = "#848bbd80"

    # Button states
    BTN_PRIMARY_BG = "#ff7edb"
    BTN_PRIMARY_HOVER = "#ff9de4"
    BTN_PRIMARY_PRESSED = "#e066c0"
    BTN_PRIMARY_TEXT = "#262335"

    BTN_SECONDARY_BG = "#34294f"
    BTN_SECONDARY_HOVER = "#3b2d5e"
    BTN_SECONDARY_PRESSED = "#2a2139"
    BTN_SECONDARY_TEXT = "#f4eee4"

    # Special states
    DISABLED_BG = "#2a2139"
    DISABLED_TEXT = "#495495"
    FOCUS_RING = "#ff7edb40"


# Active theme (can be switched at runtime)
_current_theme = LightTheme


def get_current_theme():
    """Get the currently active theme."""
    return _current_theme


def set_theme(theme_class):
    """Set the active theme."""
    global _current_theme
    _current_theme = theme_class


def is_dark_mode():
    """Check if current theme is dark mode."""
    return _current_theme == SynthwaveTheme


class Colors:
    """
    Dynamic color accessor that delegates to current theme.
    """

    @classmethod
    def _get(cls, attr):
        return getattr(_current_theme, attr)

    # Background colors
    @property
    def BG_PRIMARY(self):
        return _current_theme.BG_PRIMARY

    @property
    def BG_SECONDARY(self):
        return _current_theme.BG_SECONDARY

    # ... this approach is verbose, let's use a simpler class

# Simpler approach - use module-level properties
class _ColorProxy:
    """Proxy that fetches colors from current theme."""
    def __getattr__(self, name):
        return getattr(_current_theme, name)

Colors = _ColorProxy()


# =============================================================================
# TYPOGRAPHY
# =============================================================================

class Typography:
    """Font definitions - using system fonts for reliability."""

    # Font family stack (system fonts that work on Linux)
    FONT_FAMILY = "'Inter', 'Segoe UI', 'SF Pro Display', 'Ubuntu', 'Roboto', system-ui, sans-serif"
    FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Ubuntu Mono', 'Consolas', monospace"

    # Font sizes
    SIZE_XS = "11px"
    SIZE_SM = "12px"
    SIZE_BASE = "14px"
    SIZE_LG = "16px"
    SIZE_XL = "18px"
    SIZE_2XL = "24px"
    SIZE_3XL = "32px"

    # Font weights
    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"


# =============================================================================
# SPACING & LAYOUT
# =============================================================================

class Spacing:
    """Consistent spacing values."""

    XS = "4px"
    SM = "8px"
    MD = "12px"
    LG = "16px"
    XL = "24px"
    XXL = "32px"
    XXXL = "48px"


class BorderRadius:
    """Border radius values."""

    SM = "4px"
    MD = "8px"
    LG = "12px"
    XL = "16px"
    FULL = "9999px"


# =============================================================================
# QT STYLESHEETS
# =============================================================================

def get_main_stylesheet():
    """Generate the complete application stylesheet."""
    return f"""
    /* ===== GLOBAL STYLES ===== */
    QMainWindow {{
        background-color: {Colors.BG_PRIMARY};
    }}

    QWidget {{
        background-color: transparent;
        color: {Colors.TEXT_PRIMARY};
        font-family: {Typography.FONT_FAMILY};
        font-size: {Typography.SIZE_BASE};
    }}

    /* ===== MENU BAR ===== */
    QMenuBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_PRIMARY};
        padding: {Spacing.SM} {Spacing.MD};
        border-bottom: 1px solid {Colors.SURFACE_BORDER};
    }}

    QMenuBar::item {{
        background: transparent;
        padding: {Spacing.SM} {Spacing.MD};
        border-radius: {BorderRadius.SM};
    }}

    QMenuBar::item:selected {{
        background-color: {Colors.SURFACE_HOVER};
    }}

    QMenu {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.SM};
    }}

    QMenu::item {{
        padding: {Spacing.SM} {Spacing.XL};
        border-radius: {BorderRadius.SM};
    }}

    QMenu::item:selected {{
        background-color: {Colors.ACCENT_PRIMARY};
        color: {Colors.BTN_PRIMARY_TEXT};
    }}

    QMenu::separator {{
        height: 1px;
        background: {Colors.SURFACE_BORDER};
        margin: {Spacing.SM} 0;
    }}

    /* ===== STATUS BAR ===== */
    QStatusBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_SECONDARY};
        border-top: 1px solid {Colors.SURFACE_BORDER};
        padding: {Spacing.SM} {Spacing.MD};
        font-size: {Typography.SIZE_SM};
    }}

    /* ===== BUTTONS ===== */
    QPushButton {{
        background-color: {Colors.BTN_PRIMARY_BG};
        color: {Colors.BTN_PRIMARY_TEXT};
        border: none;
        border-radius: {BorderRadius.MD};
        padding: {Spacing.MD} {Spacing.LG};
        font-size: {Typography.SIZE_BASE};
        font-weight: {Typography.WEIGHT_MEDIUM};
        min-height: 44px;
    }}

    QPushButton:hover {{
        background-color: {Colors.BTN_PRIMARY_HOVER};
    }}

    QPushButton:pressed {{
        background-color: {Colors.BTN_PRIMARY_PRESSED};
    }}

    QPushButton:disabled {{
        background-color: {Colors.DISABLED_BG};
        color: {Colors.DISABLED_TEXT};
        border: 1px solid {Colors.SURFACE_BORDER};
    }}

    QPushButton:focus {{
        outline: none;
        border: 2px solid {Colors.ACCENT_SECONDARY};
    }}

    /* Secondary buttons */
    QPushButton.secondary {{
        background-color: {Colors.BTN_SECONDARY_BG};
        color: {Colors.BTN_SECONDARY_TEXT};
        border: 1px solid {Colors.SURFACE_BORDER};
    }}

    QPushButton.secondary:hover {{
        background-color: {Colors.BTN_SECONDARY_HOVER};
        border-color: {Colors.ACCENT_PRIMARY};
    }}

    /* ===== GROUP BOX (Cards) ===== */
    QGroupBox {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.LG};
        margin-top: {Spacing.XL};
        padding: {Spacing.LG};
        padding-top: {Spacing.XXL};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: {Spacing.LG};
        top: {Spacing.SM};
        padding: 0 {Spacing.SM};
        color: {Colors.TEXT_PRIMARY};
        font-size: {Typography.SIZE_BASE};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
    }}

    /* ===== CHECKBOXES ===== */
    QCheckBox {{
        color: {Colors.TEXT_PRIMARY};
        spacing: {Spacing.SM};
        padding: {Spacing.SM} 0;
        font-size: {Typography.SIZE_BASE};
    }}

    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border-radius: {BorderRadius.SM};
        border: 2px solid {Colors.SURFACE_BORDER};
        background-color: {Colors.BG_PRIMARY};
    }}

    QCheckBox::indicator:hover {{
        border-color: {Colors.ACCENT_PRIMARY};
        background-color: {Colors.SURFACE_HOVER};
    }}

    QCheckBox::indicator:checked {{
        background-color: {Colors.ACCENT_PRIMARY};
        border-color: {Colors.ACCENT_PRIMARY};
        image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMiAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEgNUw0LjUgOC41TDExIDEiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
    }}

    QCheckBox::indicator:disabled {{
        border-color: {Colors.DISABLED_TEXT};
        background-color: {Colors.DISABLED_BG};
    }}

    QCheckBox:disabled {{
        color: {Colors.DISABLED_TEXT};
    }}

    /* Added params (from --fit) */
    QCheckBox.added-param {{
        color: {Colors.ACCENT_SECONDARY};
    }}

    /* ===== LABELS ===== */
    QLabel {{
        color: {Colors.TEXT_PRIMARY};
        background: transparent;
    }}

    QLabel.title {{
        font-size: {Typography.SIZE_XL};
        font-weight: {Typography.WEIGHT_BOLD};
        color: {Colors.TEXT_PRIMARY};
    }}

    QLabel.subtitle {{
        font-size: {Typography.SIZE_BASE};
        color: {Colors.TEXT_SECONDARY};
    }}

    QLabel.muted {{
        color: {Colors.TEXT_MUTED};
        font-style: italic;
    }}

    QLabel.stat-value {{
        font-family: {Typography.FONT_MONO};
        font-size: {Typography.SIZE_BASE};
        color: {Colors.TEXT_PRIMARY};
    }}

    QLabel.stat-label {{
        font-size: {Typography.SIZE_SM};
        color: {Colors.TEXT_SECONDARY};
    }}

    /* ===== SCROLL AREA ===== */
    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background-color: {Colors.BG_PRIMARY};
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {Colors.SURFACE_BORDER};
        border-radius: 6px;
        min-height: 40px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {Colors.TEXT_MUTED};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}

    /* ===== TOOLTIPS ===== */
    QToolTip {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.SM} {Spacing.MD};
        font-size: {Typography.SIZE_SM};
    }}

    /* ===== MESSAGE BOX ===== */
    QMessageBox {{
        background-color: {Colors.SURFACE};
    }}

    QMessageBox QLabel {{
        color: {Colors.TEXT_PRIMARY};
    }}

    /* ===== FILE DIALOG ===== */
    QFileDialog {{
        background-color: {Colors.BG_PRIMARY};
    }}

    /* ===== PROGRESS DIALOG ===== */
    QProgressDialog {{
        background-color: {Colors.SURFACE};
    }}

    QProgressBar {{
        background-color: {Colors.BG_PRIMARY};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.SM};
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {Colors.ACCENT_PRIMARY};
        border-radius: {BorderRadius.SM};
    }}
    """


def get_plot_title_style():
    """Style for the plot title label."""
    return f"""
        font-size: {Typography.SIZE_LG};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        background: {Colors.BG_PRIMARY};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.MD} {Spacing.LG};
        margin: {Spacing.SM};
    """


def get_stats_card_style():
    """Style for statistics card container."""
    return f"""
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.SURFACE_BORDER};
        border-radius: {BorderRadius.LG};
        padding: {Spacing.LG};
    """


def get_stat_label_style():
    """Style for individual stat labels."""
    return f"""
        font-size: {Typography.SIZE_SM};
        color: {Colors.TEXT_SECONDARY};
        background: transparent;
        padding: {Spacing.XS} 0;
    """


def get_stat_value_style():
    """Style for stat values."""
    return f"""
        font-family: {Typography.FONT_MONO};
        font-size: {Typography.SIZE_BASE};
        color: {Colors.TEXT_PRIMARY};
        background: transparent;
        font-weight: {Typography.WEIGHT_MEDIUM};
    """


def get_placeholder_style():
    """Style for placeholder text."""
    return f"""
        color: {Colors.TEXT_MUTED};
        font-style: italic;
        padding: {Spacing.MD};
    """


def get_added_param_style():
    """Style for parameters added via --fit flag."""
    return f"""
        color: {Colors.ACCENT_SECONDARY};
    """


def get_section_title_style():
    """Style for section titles."""
    return f"""
        font-size: {Typography.SIZE_BASE};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        padding: {Spacing.SM} 0;
        margin-top: {Spacing.MD};
    """


def get_control_panel_style():
    """Style for the control panel container."""
    return f"""
        background-color: {Colors.BG_SECONDARY};
        border-left: 1px solid {Colors.SURFACE_BORDER};
    """


def get_primary_button_style():
    """Style for primary action button (Run Fit)."""
    return f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {Colors.ACCENT_SECONDARY}, stop:1 {Colors.ACCENT_PRIMARY});
            color: {Colors.BTN_PRIMARY_TEXT};
            border: none;
            border-radius: {BorderRadius.MD};
            padding: {Spacing.MD} {Spacing.LG};
            font-size: {Typography.SIZE_BASE};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
            min-height: 48px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {Colors.ACCENT_TERTIARY}, stop:1 {Colors.ACCENT_SECONDARY});
        }}
        QPushButton:pressed {{
            background: {Colors.BTN_PRIMARY_PRESSED};
        }}
        QPushButton:disabled {{
            background: {Colors.DISABLED_BG};
            color: {Colors.DISABLED_TEXT};
            border: 1px solid {Colors.SURFACE_BORDER};
        }}
    """


def get_secondary_button_style():
    """Style for secondary action buttons."""
    return f"""
        QPushButton {{
            background-color: {Colors.BTN_SECONDARY_BG};
            color: {Colors.BTN_SECONDARY_TEXT};
            border: 1px solid {Colors.SURFACE_BORDER};
            border-radius: {BorderRadius.MD};
            padding: {Spacing.MD} {Spacing.LG};
            font-size: {Typography.SIZE_BASE};
            font-weight: {Typography.WEIGHT_MEDIUM};
            min-height: 44px;
        }}
        QPushButton:hover {{
            background-color: {Colors.BTN_SECONDARY_HOVER};
            border-color: {Colors.ACCENT_PRIMARY};
        }}
        QPushButton:pressed {{
            background-color: {Colors.BTN_SECONDARY_PRESSED};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED_BG};
            color: {Colors.DISABLED_TEXT};
            border: 1px solid {Colors.SURFACE_BORDER};
        }}
    """


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

class PlotTheme:
    """PyQtGraph plot styling - adapts to current theme."""

    AXIS_PEN_WIDTH = 2
    SCATTER_SIZE = 7
    ERROR_BAR_WIDTH = 2.0
    ERROR_BAR_BEAM = 0.6
    ZERO_LINE_WIDTH = 2
    ZERO_LINE_STYLE = "dash"
    LABEL_FONT_FAMILY = "Inter, Ubuntu, sans-serif"
    LABEL_FONT_SIZE = "12pt"

    @classmethod
    def get_background(cls):
        return Colors.PLOT_BG

    @classmethod
    def get_axis_pen_color(cls):
        return Colors.PLOT_AXES

    @classmethod
    def get_axis_text_color(cls):
        return Colors.PLOT_AXES

    @classmethod
    def get_grid_alpha(cls):
        return 0.3 if is_dark_mode() else 0.5

    @classmethod
    def get_scatter_color(cls):
        """Get scatter color based on theme."""
        if is_dark_mode():
            return (54, 249, 246, 220)  # Neon cyan #36f9f6
        else:
            return (43, 65, 98, 220)    # Navy #2b4162

    @classmethod
    def get_scatter_color_alt(cls):
        """Get alternate scatter color (for toggle)."""
        if is_dark_mode():
            return (255, 126, 219, 220)  # Neon pink #ff7edb
        else:
            return (94, 24, 3, 220)      # Burgundy #5E1803

    @classmethod
    def get_error_bar_color(cls):
        if is_dark_mode():
            return (132, 139, 189, 90)  # #848bbd
        else:
            return (43, 65, 98, 90)     # Navy with transparency

    @classmethod
    def get_error_bar_color_alt(cls):
        if is_dark_mode():
            return (255, 126, 219, 90)  # Pink with transparency
        else:
            return (94, 24, 3, 90)      # Burgundy with transparency

    @classmethod
    def get_zero_line_color(cls):
        return Colors.PLOT_ZERO_LINE

    @classmethod
    def get_label_color(cls):
        return Colors.PLOT_AXES


def get_plot_axis_style():
    """Get axis styling dict for PyQtGraph."""
    import pyqtgraph as pg
    return {
        'color': PlotTheme.AXIS_TEXT_COLOR,
        'font-size': PlotTheme.LABEL_FONT_SIZE,
    }


def configure_plot_widget(plot_widget):
    """
    Apply modern theme to a PyQtGraph PlotWidget.

    Parameters
    ----------
    plot_widget : pg.PlotWidget
        The plot widget to configure
    """
    import pyqtgraph as pg
    from PySide6.QtGui import QFont

    # Set background
    plot_widget.setBackground(PlotTheme.get_background())

    # Configure all axes
    for axis in ['left', 'bottom', 'top', 'right']:
        ax = plot_widget.getPlotItem().getAxis(axis)
        ax.setPen(pg.mkPen(PlotTheme.get_axis_pen_color(), width=PlotTheme.AXIS_PEN_WIDTH))

        if axis in ['left', 'bottom']:
            ax.setTextPen(PlotTheme.get_axis_text_color())
            # Set font
            font = QFont("Inter, Ubuntu, sans-serif", 11)
            ax.setStyle(tickFont=font)
        else:
            ax.setStyle(showValues=False)
            ax.show()

    # Grid disabled
    plot_widget.showGrid(x=False, y=False)

    # Set axis labels with modern font
    label_style = {'color': PlotTheme.get_axis_text_color(), 'font-size': '12pt'}
    plot_widget.setLabel('left', 'Residual', units='μs', **label_style)
    plot_widget.setLabel('bottom', 'MJD (TDB)', units='days', **label_style)

    # Disable SI prefix on x-axis
    plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

    # Add subtle border
    plot_widget.getPlotItem().getViewBox().setBorder(
        pg.mkPen(PlotTheme.get_axis_pen_color(), width=1)
    )


def create_scatter_item():
    """Create a styled scatter plot item."""
    import pyqtgraph as pg

    return pg.ScatterPlotItem(
        size=PlotTheme.SCATTER_SIZE,
        pen=pg.mkPen(None),
        brush=pg.mkBrush(*PlotTheme.get_scatter_color())
    )


def create_error_bar_item():
    """Create a styled error bar item matching data point color."""
    import pyqtgraph as pg

    return pg.ErrorBarItem(
        beam=PlotTheme.ERROR_BAR_BEAM,
        pen=pg.mkPen(color=PlotTheme.get_error_bar_color(), width=PlotTheme.ERROR_BAR_WIDTH)
    )


def get_error_bar_color_for_data(data_point_color, use_alt=False):
    """Get error bar color matching the current data point color."""
    if use_alt:
        return PlotTheme.get_error_bar_color_alt()
    else:
        return PlotTheme.get_error_bar_color()


def get_scatter_colors():
    """Get primary and alternate scatter colors for current theme."""
    return {
        'primary': PlotTheme.get_scatter_color(),
        'alt': PlotTheme.get_scatter_color_alt()
    }


def create_zero_line(plot_widget):
    """Create a styled zero reference line."""
    import pyqtgraph as pg
    from PySide6.QtCore import Qt

    return plot_widget.addLine(
        y=0,
        pen=pg.mkPen(
            PlotTheme.get_zero_line_color(),
            style=Qt.DashLine,
            width=PlotTheme.ZERO_LINE_WIDTH
        )
    )
