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
    Warm Paper-like Professional Light Theme.
    
    Design Philosophy:
    - Primary text is warm near-black (neutral, readable)
    - Teal (#1F4D4A) reserved for emphasis and interaction ONLY:
      - Primary CTA (Run Fit button)
      - Focus rings and active states
      - Checked checkbox fills
      - Key metric emphasis (RMS value)
    - Paper-like warm backgrounds with 4 distinct surface levels
    - Subtle borders by default, strong borders for major containers
    """
    NAME = "light"

    # ==========================================================================
    # BACKGROUND COLORS - 4 elevation levels (paper-like depth)
    # ==========================================================================
    BG_PRIMARY = "#F5F1E8"        # App background (warm paper)
    BG_SECONDARY = "#EBE7DD"      # Sidebar background (slightly darker)
    BG_TERTIARY = "#E3DFD5"       # Panel backgrounds
    BG_ELEVATED = "#FDFCFA"       # Elevated cards (near-white warm)

    # ==========================================================================
    # SURFACE COLORS
    # ==========================================================================
    SURFACE = "#FDFCFA"           # Card surfaces (warm white)
    SURFACE_HOVER = "#F7F5F0"     # Hover state (gentle warm tint)
    SURFACE_BORDER = "#D4C9B5"    # Strong border (major containers)

    # Border tokens for hierarchy
    BORDER_SUBTLE = "#D4C9B580"   # Subtle border (50% opacity) - most components
    BORDER_STRONG = "#D4C9B5"     # Strong border - plot frame, selected containers

    # ==========================================================================
    # TEXT COLORS - Warm neutrals (NOT teal for primary text)
    # ==========================================================================
    TEXT_PRIMARY = "#201E1B"      # Primary text: warm near-black
    TEXT_SECONDARY = "#4A4641"    # Secondary text: warm dark gray
    TEXT_MUTED = "#7A756D"        # Muted text: warm medium gray
    TEXT_ACCENT = "#1F4D4A"       # Accent text: teal (for emphasis ONLY)

    # ==========================================================================
    # ACCENT COLORS - Teal for interaction/emphasis
    # ==========================================================================
    ACCENT_PRIMARY = "#1F4D4A"    # Teal - CTA, focus, checked states, RMS value
    ACCENT_SECONDARY = "#2b4162"  # Navy - data points, secondary emphasis
    ACCENT_TERTIARY = "#CCB999"   # Warm tan - warnings, info
    ACCENT_SUCCESS = "#2D6A4F"    # Muted green - success states
    ACCENT_WARNING = "#B5890A"    # Warm amber - warning states
    ACCENT_ERROR = "#9e2a2b"      # Muted red - error states

    # ==========================================================================
    # PLOT COLORS - Warm paper integration
    # ==========================================================================
    PLOT_BG = "#FDFCFA"           # Warm off-white (matches elevated surface)
    PLOT_GRID = "#E8E4DA"         # Subtle warm grid
    PLOT_AXES = "#4A4641"         # Neutral warm gray (NOT teal)
    PLOT_ZERO_LINE = "#9e2a2b"    # Semantic red

    # Data visualization
    DATA_POINTS = "#2b4162"       # Navy (scientific, clear)
    DATA_POINTS_GLOW = "#4a6fa5"
    ERROR_BARS = "#2b416260"      # Navy with 38% opacity (muted but visible)

    # ==========================================================================
    # BUTTON STATES
    # ==========================================================================
    # Primary button: Teal (CTA emphasis)
    BTN_PRIMARY_BG = "#1F4D4A"
    BTN_PRIMARY_HOVER = "#2A635F"
    BTN_PRIMARY_PRESSED = "#163836"
    BTN_PRIMARY_TEXT = "#FDFCFA"  # Warm white on teal

    # Secondary button: Neutral with subtle border
    BTN_SECONDARY_BG = "#FDFCFA"
    BTN_SECONDARY_HOVER = "#F7F5F0"
    BTN_SECONDARY_PRESSED = "#EBE7DD"
    BTN_SECONDARY_TEXT = "#201E1B"  # Warm near-black (not teal)

    # ==========================================================================
    # SPECIAL STATES
    # ==========================================================================
    DISABLED_BG = "#E3DFD5"
    DISABLED_TEXT = "#A8A49C"
    FOCUS_RING = "#1F4D4A60"      # Teal focus ring (60% opacity)


class SynthwaveTheme:
    """
    Synthwave '84 - Retro neon dark theme.
    Inspired by the VS Code Synthwave '84 extension.

    Two variants controlled by _synthwave_variant:
    - "classic": Pink-forward brand, cyan data (default)
    - "scilab": Cyan-forward brand, pink data
    """
    NAME = "synthwave"

    # ==========================================================================
    # SURFACE COLORS (4 elevation levels)
    # ==========================================================================
    BG_PRIMARY = "#262335"        # Main window background
    BG_SECONDARY = "#241b2f"      # Sidebar / right panel background
    BG_TERTIARY = "#1a1325"       # Panel backgrounds, plot container
    BG_ELEVATED = "#34294f"       # Hover/selection surfaces

    # ==========================================================================
    # SURFACE & BORDER COLORS
    # ==========================================================================
    SURFACE = "#1a1325"           # Card backgrounds (use darkest for depth)
    SURFACE_HOVER = "#34294f"     # Hover state
    SURFACE_BORDER = "#49549580"  # Faded purple with transparency (subtle)

    # Additional border tokens
    BORDER_SUBTLE = "#49549560"   # Even more subtle for dividers
    GRID_SUBTLE = "#34294f66"     # Plot grid (very subtle)

    # ==========================================================================
    # TEXT COLORS
    # ==========================================================================
    TEXT_PRIMARY = "#ffffff"      # Primary headings
    TEXT_SECONDARY = "#f4eee4"    # Body text
    TEXT_MUTED = "#b6b1c4"        # Labels
    TEXT_INACTIVE = "#848bbd"     # Muted/inactive
    TEXT_ACCENT = "#ff7edb"       # Accent text (overridden by variant)

    # ==========================================================================
    # NEON ACCENT PALETTE (static reference)
    # ==========================================================================
    NEON_PINK = "#ff7edb"
    NEON_CYAN = "#36f9f6"
    NEON_YELLOW = "#fede5d"
    NEON_ORANGE = "#f97e72"
    NEON_GREEN = "#72f1b8"
    NEON_RED = "#fe4450"
    NEON_BLUE = "#03edf9"
    ELECTRIC_BLUE = "#00b3f0"

    # ==========================================================================
    # ACCENT ROLES (variant-dependent, set by _get_variant_color)
    # ==========================================================================
    # These are defaults for "classic" mode - will be dynamically read
    ACCENT_PRIMARY = "#ff7edb"    # Brand/CTA (pink in classic)
    ACCENT_SECONDARY = "#36f9f6"  # Data emphasis (cyan in classic)
    ACCENT_TERTIARY = "#fede5d"   # Warning/info
    ACCENT_SUCCESS = "#72f1b8"    # Success
    ACCENT_WARNING = "#fede5d"    # Warning
    ACCENT_ERROR = "#fe4450"      # Error

    # ==========================================================================
    # PLOT COLORS
    # ==========================================================================
    PLOT_BG = "#1a1325"           # Dark purple plot background
    PLOT_GRID = "#34294f50"       # Subtle grid
    PLOT_AXES = "#b6b1c4"         # Lavender axes/ticks
    PLOT_ZERO_LINE = "#fe4450"    # Neon red

    # Data visualization
    DATA_POINTS = "#36f9f6"       # Cyan (classic default)
    DATA_POINTS_ALT = "#ff7edb"   # Pink (alternate)
    DATA_POINTS_GLOW = "#72f1b8"
    ERROR_BARS = "#848bbd80"      # Muted purple, always visible

    # ==========================================================================
    # BUTTON STATES
    # ==========================================================================
    # Primary button: pink gradient (classic) - will be adjusted by variant
    BTN_PRIMARY_BG = "#ff7edb"
    BTN_PRIMARY_HOVER = "#ff9de4"
    BTN_PRIMARY_PRESSED = "#e066c0"
    BTN_PRIMARY_TEXT = "#1a1325"  # Dark text on bright button

    # Secondary button: subtle surface
    BTN_SECONDARY_BG = "#241b2f"
    BTN_SECONDARY_HOVER = "#34294f"
    BTN_SECONDARY_PRESSED = "#1a1325"
    BTN_SECONDARY_TEXT = "#f4eee4"

    # ==========================================================================
    # SPECIAL STATES
    # ==========================================================================
    DISABLED_BG = "#241b2f"
    DISABLED_TEXT = "#495495"
    FOCUS_RING = "#03edf940"      # Neon blue focus ring


# =============================================================================
# SYNTHWAVE VARIANT SYSTEM
# =============================================================================
_synthwave_variant = "classic"  # "classic" or "scilab"


def get_synthwave_variant():
    """Get current synthwave variant."""
    return _synthwave_variant


def set_synthwave_variant(variant):
    """Set synthwave variant: 'classic' or 'scilab'."""
    global _synthwave_variant
    if variant in ("classic", "scilab"):
        _synthwave_variant = variant


def toggle_synthwave_variant():
    """Toggle between classic and scilab variants."""
    global _synthwave_variant
    _synthwave_variant = "scilab" if _synthwave_variant == "classic" else "classic"
    return _synthwave_variant


def get_synthwave_accent_primary():
    """Get primary accent based on variant."""
    if _synthwave_variant == "scilab":
        return SynthwaveTheme.NEON_CYAN
    return SynthwaveTheme.NEON_PINK


def get_synthwave_accent_secondary():
    """Get secondary accent (data color) based on variant."""
    if _synthwave_variant == "scilab":
        return SynthwaveTheme.NEON_PINK
    return SynthwaveTheme.NEON_CYAN


def get_synthwave_btn_gradient():
    """Get button gradient colors based on variant."""
    if _synthwave_variant == "scilab":
        return ("#36f9f6", "#03edf9", "#00b3f0")  # Cyan gradient
    return ("#ff7edb", "#ff9de4", "#e066c0")      # Pink gradient


def get_synthwave_rms_color():
    """Get RMS highlight color (contrasts with data points)."""
    if _synthwave_variant == "scilab":
        return SynthwaveTheme.NEON_CYAN  # Cyan RMS when pink data
    return SynthwaveTheme.NEON_PINK      # Pink RMS when cyan data


def get_synthwave_data_colors():
    """Get data point colors for current variant."""
    if _synthwave_variant == "scilab":
        return {
            'primary': (255, 126, 219, 220),    # Pink primary
            'alt': (54, 249, 246, 220),         # Cyan alt
            'primary_hex': "#ff7edb",
            'alt_hex': "#36f9f6",
        }
    return {
        'primary': (54, 249, 246, 220),         # Cyan primary
        'alt': (255, 126, 219, 220),            # Pink alt
        'primary_hex': "#36f9f6",
        'alt_hex': "#ff7edb",
    }


# Active theme (can be switched at runtime)
# Default to dark mode (Synthwave) for better visibility
_current_theme = SynthwaveTheme

# Light mode variant: "navy" (default) or "burgundy"
_light_variant = "navy"


def get_light_variant():
    """Get current light mode variant."""
    return _light_variant


def set_light_variant(variant):
    """Set light mode variant: 'navy' or 'burgundy'."""
    global _light_variant
    if variant in ("navy", "burgundy"):
        _light_variant = variant


def toggle_light_variant():
    """Toggle between navy and burgundy variants in light mode."""
    global _light_variant
    _light_variant = "burgundy" if _light_variant == "navy" else "navy"
    return _light_variant


def get_light_data_colors():
    """Get data point colors for current light mode variant."""
    if _light_variant == "burgundy":
        return {
            'primary': (94, 24, 3, 220),       # Burgundy primary
            'alt': (43, 65, 98, 220),          # Navy alt
            'primary_hex': "#5E1803",
            'alt_hex': "#2b4162",
        }
    return {
        'primary': (43, 65, 98, 220),          # Navy primary (default)
        'alt': (94, 24, 3, 220),               # Burgundy alt
        'primary_hex': "#2b4162",
        'alt_hex': "#5E1803",
    }


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

def get_dynamic_accent_primary():
    """Get the current accent primary color (variant-aware for dark mode)."""
    if is_dark_mode():
        return get_synthwave_accent_primary()
    return Colors.ACCENT_PRIMARY


def get_dynamic_accent_secondary():
    """Get the current accent secondary color (variant-aware for dark mode)."""
    if is_dark_mode():
        return get_synthwave_accent_secondary()
    return Colors.ACCENT_SECONDARY


def get_rms_emphasis_color():
    """Get RMS value emphasis color (teal in light mode, variant-aware in dark mode)."""
    if is_dark_mode():
        return get_synthwave_rms_color()
    return LightTheme.ACCENT_PRIMARY  # Teal for emphasis


def get_border_subtle():
    """Get the subtle border color (with fallback for themes without it)."""
    return getattr(_current_theme, 'BORDER_SUBTLE', Colors.SURFACE_BORDER)


def get_border_strong():
    """Get the strong border color."""
    return getattr(_current_theme, 'BORDER_STRONG', Colors.SURFACE_BORDER)


def get_main_stylesheet():
    """Generate the complete application stylesheet."""
    # Get dynamic accent colors based on current variant
    accent_primary = get_dynamic_accent_primary()
    accent_secondary = get_dynamic_accent_secondary()

    # Get border colors (use subtle by default, strong for major containers)
    border_subtle = get_border_subtle()
    border_strong = get_border_strong()

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
        border-bottom: 1px solid {border_subtle};
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
        border: 1px solid {border_subtle};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.SM};
    }}

    QMenu::item {{
        padding: {Spacing.SM} {Spacing.XL};
        border-radius: {BorderRadius.SM};
    }}

    QMenu::item:selected {{
        background-color: {accent_primary};
        color: {Colors.BTN_PRIMARY_TEXT};
    }}

    QMenu::separator {{
        height: 1px;
        background: {border_subtle};
        margin: {Spacing.SM} 0;
    }}

    /* ===== STATUS BAR ===== */
    QStatusBar {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_SECONDARY};
        border-top: 1px solid {border_subtle};
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
        border: 1px solid {border_subtle};
    }}

    QPushButton:focus {{
        outline: none;
        border: 2px solid {accent_primary};
    }}

    /* Secondary buttons */
    QPushButton.secondary {{
        background-color: {Colors.BTN_SECONDARY_BG};
        color: {Colors.BTN_SECONDARY_TEXT};
        border: 1px solid {border_subtle};
    }}

    QPushButton.secondary:hover {{
        background-color: {Colors.BTN_SECONDARY_HOVER};
        border-color: {accent_primary};
    }}

    /* ===== GROUP BOX (Cards) ===== */
    QGroupBox {{
        background-color: {Colors.SURFACE};
        border: 1px solid {border_subtle};
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
        border: 2px solid {border_subtle};
        background-color: {Colors.BG_PRIMARY};
    }}

    QCheckBox::indicator:hover {{
        border-color: {accent_primary};
        background-color: {Colors.SURFACE_HOVER};
    }}

    QCheckBox::indicator:checked {{
        background-color: {accent_primary};
        border-color: {accent_primary};
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
        color: {accent_secondary};
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
        background-color: {border_subtle};
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

    /* ===== COMBOBOX ===== */
    QComboBox {{
        background-color: {Colors.BG_SECONDARY};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {border_subtle};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.SM} {Spacing.MD};
        font-size: {Typography.SIZE_SM};
        min-height: 28px;
    }}

    QComboBox:hover {{
        border-color: {accent_primary};
    }}

    QComboBox:on {{
        border-color: {accent_primary};
    }}

    QComboBox::drop-down {{
        border: none;
        padding-right: {Spacing.SM};
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {Colors.TEXT_SECONDARY};
        margin-right: {Spacing.SM};
    }}

    /* Dropdown popup frame */
    QComboBox QFrame {{
        background-color: {Colors.SURFACE};
        border: 1px solid {border_subtle};
    }}

    /* Dropdown list view */
    QComboBox QAbstractItemView {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_PRIMARY};
        border: none;
        outline: none;
        selection-background-color: {accent_primary};
        selection-color: {Colors.BTN_PRIMARY_TEXT};
        padding: 4px;
    }}

    QComboBox QAbstractItemView::item {{
        min-height: 24px;
        padding: 4px 8px;
    }}

    QComboBox QAbstractItemView::item:hover {{
        background-color: {Colors.SURFACE_HOVER};
    }}

    QComboBox QAbstractItemView::item:selected {{
        background-color: {accent_primary};
        color: {Colors.BTN_PRIMARY_TEXT};
    }}

    QComboBox QScrollBar:vertical {{
        width: 0px;
    }}

    /* ===== TOOLTIPS ===== */
    QToolTip {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {border_subtle};
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
        border: 1px solid {border_subtle};
        border-radius: {BorderRadius.SM};
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {accent_primary};
        border-radius: {BorderRadius.SM};
    }}
    """


def get_plot_title_style():
    """Style for the plot title label."""
    border_subtle = get_border_subtle()
    return f"""
        font-size: {Typography.SIZE_LG};
        font-weight: {Typography.WEIGHT_SEMIBOLD};
        color: {Colors.TEXT_PRIMARY};
        background: {Colors.SURFACE};
        border: 1px solid {border_subtle};
        border-radius: {BorderRadius.MD};
        padding: {Spacing.MD} {Spacing.LG};
        margin: {Spacing.SM};
    """


def get_stats_card_style():
    """Style for statistics card container."""
    border_subtle = get_border_subtle()
    return f"""
        background-color: {Colors.SURFACE};
        border: 1px solid {border_subtle};
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
    border_subtle = get_border_subtle()
    return f"""
        background-color: {Colors.BG_SECONDARY};
        border-left: 1px solid {border_subtle};
    """


def get_primary_button_style():
    """Style for primary action button (Run Fit) - teal CTA in light mode, gradient in dark."""
    if is_dark_mode():
        # Synthwave: variant-aware gradient
        gradient = get_synthwave_btn_gradient()
        grad_start, grad_mid, grad_end = gradient
        return f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {grad_start}, stop:1 {grad_end});
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
                    stop:0 {grad_mid}, stop:1 {grad_start});
            }}
            QPushButton:pressed {{
                background: {Colors.BTN_PRIMARY_PRESSED};
            }}
            QPushButton:disabled {{
                background: {Colors.DISABLED_BG};
                color: {Colors.DISABLED_TEXT};
                border: 1px solid {get_border_subtle()};
            }}
        """
    else:
        # Light mode: solid teal CTA (emphasis color)
        return f"""
            QPushButton {{
                background-color: {Colors.BTN_PRIMARY_BG};
                color: {Colors.BTN_PRIMARY_TEXT};
                border: none;
                border-radius: {BorderRadius.MD};
                padding: {Spacing.MD} {Spacing.LG};
                font-size: {Typography.SIZE_BASE};
                font-weight: {Typography.WEIGHT_SEMIBOLD};
                min-height: 48px;
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
                border: 1px solid {get_border_subtle()};
            }}
        """


def get_secondary_button_style():
    """Style for secondary action buttons."""
    accent_primary = get_dynamic_accent_primary()
    border_subtle = get_border_subtle()
    return f"""
        QPushButton {{
            background-color: {Colors.BTN_SECONDARY_BG};
            color: {Colors.BTN_SECONDARY_TEXT};
            border: 1px solid {border_subtle};
            border-radius: {BorderRadius.MD};
            padding: {Spacing.MD} {Spacing.LG};
            font-size: {Typography.SIZE_BASE};
            font-weight: {Typography.WEIGHT_MEDIUM};
            min-height: 44px;
        }}
        QPushButton:hover {{
            background-color: {Colors.BTN_SECONDARY_HOVER};
            border-color: {accent_primary};
        }}
        QPushButton:pressed {{
            background-color: {Colors.BTN_SECONDARY_PRESSED};
        }}
        QPushButton:disabled {{
            background-color: {Colors.DISABLED_BG};
            color: {Colors.DISABLED_TEXT};
            border: 1px solid {border_subtle};
        }}
    """


# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

class PlotTheme:
    """PyQtGraph plot styling - adapts to current theme and variant."""

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
        return 0.25 if is_dark_mode() else 0.5

    @classmethod
    def get_scatter_color(cls):
        """Get primary scatter color based on theme and variant."""
        if is_dark_mode():
            colors = get_synthwave_data_colors()
            return colors['primary']
        else:
            colors = get_light_data_colors()
            return colors['primary']

    @classmethod
    def get_scatter_color_alt(cls):
        """Get alternate scatter color (for toggle)."""
        if is_dark_mode():
            colors = get_synthwave_data_colors()
            return colors['alt']
        else:
            colors = get_light_data_colors()
            return colors['alt']

    @classmethod
    def get_error_bar_color(cls):
        """Error bars: muted, always visible, matches data color."""
        if is_dark_mode():
            return (132, 139, 189, 128)  # #848bbd with good visibility
        else:
            colors = get_light_data_colors()
            # Extract RGB from primary and add transparency
            r, g, b, _ = colors['primary']
            return (r, g, b, 90)

    @classmethod
    def get_error_bar_color_alt(cls):
        """Error bars for alt data color."""
        if is_dark_mode():
            return (132, 139, 189, 128)  # Same muted purple (always visible)
        else:
            colors = get_light_data_colors()
            r, g, b, _ = colors['alt']
            return (r, g, b, 90)

    @classmethod
    def get_zero_line_color(cls):
        return Colors.PLOT_ZERO_LINE

    @classmethod
    def get_label_color(cls):
        return Colors.PLOT_AXES

    @classmethod
    def get_rms_highlight_color(cls):
        """Get RMS highlight color based on theme and variant.
        
        In light mode: teal (ACCENT_PRIMARY) for emphasis.
        In dark mode: variant-aware (pink or cyan).
        """
        if is_dark_mode():
            return get_synthwave_rms_color()
        else:
            # Light mode: teal for key metric emphasis
            return LightTheme.ACCENT_PRIMARY  # #1F4D4A teal


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
