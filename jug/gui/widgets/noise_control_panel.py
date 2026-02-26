"""Noise Control Panel -- interactive panel for toggling and editing noise processes.

Shows noise processes detected from the .par file (EFAC, EQUAD, ECORR,
Red Noise, DM Noise) as toggleable rows. Active processes affect
the fit when "Run Fit" is clicked. Parameter values are editable in-place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QSizePolicy, QLineEdit, QPushButton, QMenu, QCheckBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QDoubleValidator

from jug.gui.theme import Colors, Typography
from jug.engine.noise_mode import (
    NOISE_REGISTRY, get_noise_label, get_noise_tooltip, get_noise_display_order,
)

if TYPE_CHECKING:
    from jug.engine.noise_mode import NoiseConfig


class NoiseProcessRow(QFrame):
    """A single toggleable noise process with expandable parameter details."""

    toggled = Signal(str, bool)         # (process_name, is_enabled)
    param_changed = Signal(str, str, str)  # (process_name, param_key, new_value)
    remove_requested = Signal(str)      # (process_name)
    realise_toggled = Signal(str, bool)  # (process_name, show_overlay)
    subtract_toggled = Signal(str, bool)  # (process_name, should_subtract)

    def __init__(self, name: str, enabled: bool, params: List[dict],
                 can_subtract: bool = False, parent=None):
        """
        Parameters
        ----------
        name : str
            Canonical process name (e.g. "EFAC").
        enabled : bool
            Initial toggle state.
        params : list of dict
            Parameter entries, each with keys: 'key', 'label', 'value', 'editable'.
        can_subtract : bool
            Whether this process supports subtract-from-residuals toggle.
        """
        super().__init__(parent)
        self.process_name = name
        self._enabled = enabled
        self._expanded = False
        self._realising = False
        self._subtracting = False
        self._params = params
        self._param_edits: Dict[str, QLineEdit] = {}

        info = {"label": get_noise_label(name), "tip": get_noise_tooltip(name)}

        self.setObjectName("noiseProcessRow")
        self.setFrameShape(QFrame.NoFrame)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Header row: toggle + label + realise + subtract + badge + arrow ---
        header = QWidget()
        header.setCursor(Qt.PointingHandCursor)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(10, 2, 6, 2)
        hl.setSpacing(4)

        # Checkbox toggle (matches "Parameters to Fit" style)
        from PySide6.QtWidgets import QCheckBox
        self.toggle_btn = QCheckBox(info["label"])
        self.toggle_btn.setChecked(enabled)
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.setToolTip(info["tip"])
        self.toggle_btn.setStyleSheet(
            f"QCheckBox {{ padding: 2px 0; spacing: 6px; font-size: 12px; "
            f"font-weight: {Typography.WEIGHT_MEDIUM}; color: {Colors.TEXT_PRIMARY}; border: none; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        self.toggle_btn.toggled.connect(self._on_toggle_check)
        hl.addWidget(self.toggle_btn, 1)

        # Realise + Subtract buttons (B3: two-button workflow)
        self._realise_btn = None
        self._subtract_btn = None
        _action_btn_style = (
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border: 1px solid {Colors.SURFACE_BORDER}; border-radius: 3px; "
            f"font-size: 9px; padding: 0 3px; }}"
            f"QPushButton:checked {{ background-color: {Colors.ACCENT_PRIMARY}; "
            f"color: white; border: none; }}"
            f"QPushButton:hover {{ border-color: {Colors.TEXT_MUTED}; }}"
            f"QPushButton:disabled {{ color: {Colors.SURFACE_BORDER}; "
            f"border-color: {Colors.SURFACE_BORDER}; }}"
        )
        if can_subtract:
            self._realise_btn = QPushButton("R")
            self._realise_btn.setFixedSize(18, 16)
            self._realise_btn.setCheckable(True)
            self._realise_btn.setCursor(Qt.PointingHandCursor)
            self._realise_btn.setToolTip("Show realization on plot")
            self._realise_btn.setStyleSheet(_action_btn_style)
            self._realise_btn.clicked.connect(self._on_realise_toggle)
            hl.addWidget(self._realise_btn)

            self._subtract_btn = QPushButton("−")
            self._subtract_btn.setFixedSize(18, 16)
            self._subtract_btn.setCheckable(True)
            self._subtract_btn.setCursor(Qt.PointingHandCursor)
            self._subtract_btn.setToolTip("Subtract realization from residuals")
            self._subtract_btn.setStyleSheet(_action_btn_style)
            self._subtract_btn.setEnabled(True)
            self._subtract_btn.clicked.connect(self._on_subtract_toggle)
            hl.addWidget(self._subtract_btn)

        # Count badge (number of sub-params)
        if params:
            badge = QLabel(str(len(params)))
            badge.setAlignment(Qt.AlignCenter)
            badge.setFixedSize(18, 14)
            badge.setStyleSheet(
                f"background-color: {Colors.SURFACE_HOVER}; color: {Colors.TEXT_MUTED}; "
                f"border-radius: 7px; font-size: 10px; border: none;"
            )
            hl.addWidget(badge)

        # Expand arrow
        self.arrow_lbl = QLabel("▸")
        self.arrow_lbl.setFixedWidth(12)
        self.arrow_lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px; border: none;")
        hl.addWidget(self.arrow_lbl)

        root.addWidget(header)

        # Click on header toggles expand
        header.mousePressEvent = self._on_header_click

        # --- Expandable detail section ---
        self.detail_widget = QWidget()
        self.detail_widget.setVisible(False)
        detail_layout = QVBoxLayout(self.detail_widget)
        detail_layout.setContentsMargins(30, 2, 10, 6)
        detail_layout.setSpacing(2)

        for p in params:
            row = QHBoxLayout()
            row.setSpacing(6)
            lbl = QLabel(p["label"])
            lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px; border: none;")
            lbl.setFixedWidth(90)
            row.addWidget(lbl)

            if p.get("editable", True):
                edit = QLineEdit(str(p["value"]))
                edit.setFixedHeight(20)
                edit.setStyleSheet(
                    f"background-color: {Colors.SURFACE}; color: {Colors.TEXT_PRIMARY}; "
                    f"border: 1px solid {Colors.SURFACE_BORDER}; border-radius: 3px; "
                    f"padding: 1px 4px; font-size: 11px; font-family: monospace;"
                )
                edit.setValidator(QDoubleValidator())
                key = p["key"]
                edit.editingFinished.connect(lambda k=key, e=edit: self._on_param_edit(k, e))
                self._param_edits[key] = edit
                row.addWidget(edit, 1)
            else:
                val = QLabel(str(p["value"]))
                val.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px; border: none;")
                row.addWidget(val, 1)

            detail_layout.addLayout(row)

        # A4: "* Remove" link at bottom of detail section
        remove_btn = QPushButton("× Remove")
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border: none; font-size: 10px; padding: 2px 0; text-align: left; }}"
            f"QPushButton:hover {{ color: {Colors.ACCENT_ERROR}; }}"
        )
        def on_remove_clicked():
            self.remove_requested.emit(self.process_name)
        remove_btn.clicked.connect(on_remove_clicked)
        detail_layout.addWidget(remove_btn)

        root.addWidget(self.detail_widget)

        # Bottom separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.SURFACE_BORDER}; border: none; max-height: 1px; "
                          f"background-color: {Colors.SURFACE_BORDER};")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        self._update_opacity()

    # -- Styling helpers ---------------------------------------------------

    def _update_opacity(self):
        opacity = "1.0" if self._enabled else "0.5"
        self.toggle_btn.setStyleSheet(
            f"QCheckBox {{ padding: 2px 0; spacing: 6px; font-size: 12px; "
            f"font-weight: {Typography.WEIGHT_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; border: none; opacity: {opacity}; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )

    # -- Slots -------------------------------------------------------------

    def _on_toggle_check(self, checked: bool):
        self._enabled = checked
        self._update_opacity()
        self.toggled.emit(self.process_name, self._enabled)

    def _on_header_click(self, event):
        self._expanded = not self._expanded
        self.detail_widget.setVisible(self._expanded)
        self.arrow_lbl.setText("▾" if self._expanded else "▸")

    def _on_param_edit(self, key: str, edit: QLineEdit):
        self.param_changed.emit(self.process_name, key, edit.text())

    def _on_subtract_toggle(self):
        self._subtracting = self._subtract_btn.isChecked()
        self.subtract_toggled.emit(self.process_name, self._subtracting)

    def _on_realise_toggle(self):
        self._realising = self._realise_btn.isChecked()
        self.realise_toggled.emit(self.process_name, self._realising)

    # -- Public API --------------------------------------------------------

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        self.toggle_btn.setChecked(enabled)
        self._update_opacity()

    def is_enabled(self) -> bool:
        return self._enabled

    def is_subtracting(self) -> bool:
        return self._subtracting

    def is_realising(self) -> bool:
        return self._realising


class NoiseControlPanel(QWidget):
    """Panel showing all noise processes with toggles and editable parameters.

    Signals
    -------
    noise_config_changed(NoiseConfig)
        Emitted whenever a toggle changes.
    param_value_changed(str, str, str)
        Emitted when a user edits a noise parameter value.
    collapse_requested()
        Emitted when user clicks collapse arrow.
    subtract_changed(str, bool)
        Emitted when a subtract toggle changes. Args: (process_name, should_subtract)
    """

    noise_config_changed = Signal(object)  # NoiseConfig
    param_value_changed = Signal(str, str, str)
    collapse_requested = Signal()
    realise_changed = Signal(str, bool)   # (process_name, show_overlay)
    subtract_changed = Signal(str, bool)  # (process_name, subtract_from_residuals)
    estimate_noise_requested = Signal(dict)  # user clicked "Estimate" with selections
    show_residuals_changed = Signal(bool)     # user toggled "Show Residuals" checkbox
    show_uncertainties_changed = Signal(bool)  # user toggled "Show Uncertainties" checkbox

    # Processes that support subtract-from-residuals
    _SUBTRACTABLE = {"RedNoise", "DMNoise", "ChromaticNoise", "CW", "BWM", "ChromaticEvent"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("noiseControlPanel")
        self.setFixedWidth(220)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        self._rows: Dict[str, NoiseProcessRow] = {}
        self._noise_config = None
        self._params = None

        # Root layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.setStyleSheet(f"""
            QWidget#noiseControlPanel {{
                background-color: {Colors.BG_SECONDARY};
                border-right: 1px solid {Colors.SURFACE_BORDER};
            }}
        """)

        # Header -- clickable "Noise >" button that collapses the panel
        self._header_btn = QPushButton("Noise ▶")
        self._header_btn.setCursor(Qt.PointingHandCursor)
        self._header_btn.setToolTip("Collapse noise panel")
        self._header_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_PRIMARY}; "
            f"font-size: 14px; font-weight: {Typography.WEIGHT_BOLD}; "
            f"border: none; padding: 10px 12px 8px 12px; text-align: left; }}"
            f"QPushButton:hover {{ color: {Colors.ACCENT_PRIMARY}; }}"
        )
        self._header_btn.clicked.connect(self._on_collapse)
        outer.addWidget(self._header_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {Colors.SURFACE_BORDER}; border: none;")
        outer.addWidget(sep)

        # Units toggle -- mus vs log10(s) for EQUAD/ECORR display
        self._units_mode = 0  # 0 = mus, 1 = log10(s)
        _main_units_labels = ["mus", "log10(s)"]
        self._main_units_btn = QPushButton(f"Units: {_main_units_labels[0]}  v")
        self._main_units_btn.setCursor(Qt.PointingHandCursor)
        self._main_units_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border: none; font-size: 10px; padding: 4px 12px 2px 12px; text-align: left; }}"
            f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; }}"
        )
        self._main_units_btn.clicked.connect(self._toggle_main_units_list)
        outer.addWidget(self._main_units_btn)

        self._main_units_list = QWidget()
        self._main_units_list.setVisible(False)
        mul_layout = QVBoxLayout(self._main_units_list)
        mul_layout.setContentsMargins(20, 0, 0, 0)
        mul_layout.setSpacing(0)
        mu_opt_style = (
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border: none; font-size: 10px; padding: 2px 0; text-align: left; }}"
            f"QPushButton:hover {{ color: {Colors.ACCENT_PRIMARY}; }}"
        )
        for idx, label in enumerate(_main_units_labels):
            opt = QPushButton(label)
            opt.setCursor(Qt.PointingHandCursor)
            opt.setStyleSheet(mu_opt_style)
            opt.clicked.connect(lambda checked=False, i=idx, l=label: self._select_main_units(i, l))
            mul_layout.addWidget(opt)
        outer.addWidget(self._main_units_list)

        # "Estimate Noise" -- expandable dropdown with process checkboxes
        self._estimate_section = QWidget()
        est_layout = QVBoxLayout(self._estimate_section)
        est_layout.setContentsMargins(0, 0, 0, 0)
        est_layout.setSpacing(0)

        # Toggle button (like "Parameters to Fit v")
        self._estimate_toggle_btn = QPushButton("⚡ Estimate Noise  ▾")
        self._estimate_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._estimate_toggle_btn.setToolTip(
            "Select noise processes and run MAP estimation"
        )
        self._estimate_toggle_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.ACCENT_PRIMARY}; "
            f"border: 1px solid {Colors.ACCENT_PRIMARY}; border-radius: 4px; "
            f"padding: 4px 12px; font-size: 11px; margin: 4px 12px; }}"
            f"QPushButton:hover {{ background-color: {Colors.ACCENT_PRIMARY}; "
            f"color: {Colors.BG_PRIMARY}; }}"
        )
        self._estimate_toggle_btn.clicked.connect(self._toggle_estimate_list)
        est_layout.addWidget(self._estimate_toggle_btn)

        # Expandable checklist (hidden by default)
        self._estimate_list = QWidget()
        self._estimate_list.setVisible(False)
        el_layout = QVBoxLayout(self._estimate_list)
        el_layout.setContentsMargins(16, 4, 12, 4)
        el_layout.setSpacing(2)

        from PySide6.QtWidgets import QCheckBox
        cb_style = (
            f"QCheckBox {{ color: {Colors.TEXT_PRIMARY}; font-size: 11px; "
            f"spacing: 4px; padding: 2px 0; }}"
        )
        self._est_cb_efac = QCheckBox("EFAC (error scale)")
        self._est_cb_efac.setChecked(True)
        self._est_cb_efac.setStyleSheet(cb_style)
        el_layout.addWidget(self._est_cb_efac)

        self._est_cb_equad = QCheckBox("EQUAD (added variance)")
        self._est_cb_equad.setChecked(True)
        self._est_cb_equad.setStyleSheet(cb_style)
        el_layout.addWidget(self._est_cb_equad)

        self._est_cb_ecorr = QCheckBox("ECORR (epoch-correlated)")
        self._est_cb_ecorr.setChecked(False)
        self._est_cb_ecorr.setStyleSheet(cb_style)
        el_layout.addWidget(self._est_cb_ecorr)

        self._est_cb_red = QCheckBox("Achromatic Red Noise")
        self._est_cb_red.setChecked(True)
        self._est_cb_red.setStyleSheet(cb_style)
        el_layout.addWidget(self._est_cb_red)

        self._est_cb_dm = QCheckBox("DM Noise")
        self._est_cb_dm.setChecked(True)
        self._est_cb_dm.setStyleSheet(cb_style)
        el_layout.addWidget(self._est_cb_dm)

        # "Estimate" action button
        self._est_run_btn = QPushButton(">  Estimate")
        self._est_run_btn.setCursor(Qt.PointingHandCursor)
        self._est_run_btn.setStyleSheet(
            f"QPushButton {{ background: {Colors.ACCENT_PRIMARY}; "
            f"color: {Colors.BG_PRIMARY}; border: none; border-radius: 4px; "
            f"padding: 5px 12px; font-size: 11px; font-weight: bold; "
            f"margin: 6px 0 2px 0; }}"
            f"QPushButton:hover {{ background: {Colors.TEXT_PRIMARY}; }}"
            f"QPushButton:disabled {{ background: {Colors.SURFACE_BORDER}; "
            f"color: {Colors.TEXT_MUTED}; }}"
        )
        self._est_run_btn.clicked.connect(self._on_estimate_clicked)
        el_layout.addWidget(self._est_run_btn)

        est_layout.addWidget(self._estimate_list)
        outer.addWidget(self._estimate_section)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: transparent; }}"
            f"QScrollBar:vertical {{ width: 6px; background: transparent; }}"
            f"QScrollBar::handle:vertical {{ background: {Colors.TEXT_MUTED}; "
            f"border-radius: 3px; min-height: 30px; }}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}"
        )

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 4, 0, 4)
        self._content_layout.setSpacing(0)
        self._content_layout.addStretch()

        scroll.setWidget(self._content)
        outer.addWidget(scroll, 1)

        # "Show Residuals" checkbox
        self._show_residuals_cb = QCheckBox("Show Residuals")
        self._show_residuals_cb.setChecked(True)
        self._show_residuals_cb.setStyleSheet(
            f"QCheckBox {{ color: {Colors.TEXT_MUTED}; font-size: 11px; "
            f"padding: 6px 12px; }}"
            f"QCheckBox:hover {{ color: {Colors.TEXT_PRIMARY}; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        self._show_residuals_cb.toggled.connect(self.show_residuals_changed.emit)
        outer.addWidget(self._show_residuals_cb)

        # "Show Uncertainties" checkbox
        self._show_uncertainties_cb = QCheckBox("Show Uncertainties")
        self._show_uncertainties_cb.setChecked(True)
        self._show_uncertainties_cb.setStyleSheet(
            f"QCheckBox {{ color: {Colors.TEXT_MUTED}; font-size: 11px; "
            f"padding: 2px 12px 6px 12px; }}"
            f"QCheckBox:hover {{ color: {Colors.TEXT_PRIMARY}; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; }}"
        )
        self._show_uncertainties_cb.toggled.connect(self.show_uncertainties_changed.emit)
        outer.addWidget(self._show_uncertainties_cb)

        # "Add Noise Process" -- foldable list (A3: list unfurls above button)
        self._add_section = QWidget()
        add_layout = QVBoxLayout(self._add_section)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_layout.setSpacing(0)

        # Foldable list of addable processes (above the button)
        self._add_list = QWidget()
        self._add_list.setVisible(False)
        self._add_list_layout = QVBoxLayout(self._add_list)
        self._add_list_layout.setContentsMargins(12, 4, 12, 4)
        self._add_list_layout.setSpacing(2)
        add_layout.addWidget(self._add_list)

        # Add button (anchored at bottom)
        self._add_btn = QPushButton("+ Add Noise Process")
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {Colors.TEXT_MUTED}; "
            f"border-top: 1px solid {Colors.SURFACE_BORDER}; border-radius: 0px; "
            f"padding: 8px 12px; font-size: 11px; text-align: left; }}"
            f"QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; "
            f"background-color: {Colors.SURFACE_HOVER}; }}"
        )
        self._add_btn.clicked.connect(self._toggle_add_list)
        add_layout.addWidget(self._add_btn)

        outer.addWidget(self._add_section)

    # -- Public API --------------------------------------------------------

    def populate_from_params(self, params: dict):
        """Populate panel from parsed par file parameters."""
        from jug.engine.noise_mode import NoiseConfig
        from jug.noise.white import parse_noise_lines

        self._params = params
        self._noise_config = NoiseConfig.from_par(params)

        # Clear existing rows
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        while self._content_layout.count() > 0:
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        noise_lines = params.get("_noise_lines", [])
        entries = parse_noise_lines(noise_lines) if noise_lines else []

        for proc_name in get_noise_display_order():
            detected = self._noise_config.is_enabled(proc_name)
            proc_params = self._extract_params(proc_name, params, entries)

            if not detected:
                continue

            can_sub = proc_name in self._SUBTRACTABLE
            # Default OFF -- user must explicitly enable before fitting
            row = NoiseProcessRow(proc_name, False, proc_params,
                                  can_subtract=can_sub)
            row.toggled.connect(self._on_process_toggled)
            row.param_changed.connect(self._on_param_value_changed)
            row.remove_requested.connect(self._on_remove_process)
            row.realise_toggled.connect(self._on_realise_toggled)
            row.subtract_toggled.connect(self._on_subtract_toggled)
            self._rows[proc_name] = row
            self._content_layout.addWidget(row)

        # Start with all noise disabled in the config (matches row default)
        self._noise_config.disable_all()

        self._content_layout.addStretch()
        self._rebuild_add_list()

        # Update estimate checkboxes to match detected noise
        from jug.noise.red_noise import parse_red_noise_params, parse_dm_noise_params
        has_red = parse_red_noise_params(params) is not None
        has_dm = parse_dm_noise_params(params) is not None
        has_ecorr = any(e.kind.upper() == 'ECORR' for e in entries)
        self._est_cb_red.setChecked(has_red)
        self._est_cb_dm.setChecked(has_dm)
        self._est_cb_ecorr.setChecked(has_ecorr)

    def get_noise_config(self) -> Optional["NoiseConfig"]:
        return self._noise_config

    def has_noise(self) -> bool:
        return len(self._rows) > 0

    def get_subtract_processes(self) -> List[str]:
        """Return list of process names with subtract toggled on."""
        return [name for name, row in self._rows.items() if row.is_subtracting()]

    def get_realise_processes(self) -> List[str]:
        """Return list of process names with realise toggled on."""
        return [name for name, row in self._rows.items() if row.is_realising()]

    def set_subtract_all(self, active: bool):
        """Set the subtract toggle on all rows without emitting signals.

        Used by the main window to sync UI after auto-subtracting noise
        realizations following a GLS fit (Tempo2-like behaviour).
        """
        for row in self._rows.values():
            row._subtract_btn.blockSignals(True)
            row._subtract_btn.setChecked(active)
            row._subtracting = active
            row._subtract_btn.blockSignals(False)

    # -- Internal ----------------------------------------------------------

    @property
    def _use_log_units(self) -> bool:
        """True when EQUAD/ECORR should be displayed as log_1_0(seconds)."""
        return self._units_mode == 1

    def _toggle_main_units_list(self):
        """Toggle the main units option list."""
        vis = not self._main_units_list.isVisible()
        self._main_units_list.setVisible(vis)
        current = "mus" if self._units_mode == 0 else "log10(s)"
        chevron = "^" if vis else "v"
        self._main_units_btn.setText(f"Units: {current}  {chevron}")

    def _select_main_units(self, index: int, label: str):
        """Select a units mode and collapse the list."""
        self._units_mode = index
        self._main_units_list.setVisible(False)
        self._main_units_btn.setText(f"Units: {label}  v")
        self._on_units_changed(index)

    def _on_units_changed(self, _index: int):
        """Re-populate panel when units are toggled."""
        if self._params is not None:
            self.populate_from_params(self._params)

    def _format_white_value(self, value_us: float) -> str:
        """Format an EQUAD/ECORR value according to current units setting."""
        import math
        if self._use_log_units:
            val_sec = value_us * 1e-6
            if val_sec > 0:
                return f"{math.log10(val_sec):.4f}"
            return "-inf"
        return f"{value_us:.6g}"

    def _extract_params(self, proc_name: str, params: dict, entries) -> List[dict]:
        """Extract displayable parameter entries for a noise process."""
        from jug.engine.noise_mode import get_impl_param_defs

        result = []

        if proc_name in ("EFAC", "EQUAD", "ECORR"):
            # White noise uses per-backend entries from noise lines -- special case
            for e in entries:
                if e.kind.upper() == proc_name.upper() or \
                   (proc_name == "EFAC" and e.kind.upper() in ("EFAC", "T2EFAC")) or \
                   (proc_name == "EQUAD" and e.kind.upper() in ("EQUAD", "T2EQUAD")):
                    label = f"-{e.flag_name} {e.flag_value}" if e.flag_name else "global"
                    if proc_name in ("EQUAD", "ECORR"):
                        display_val = self._format_white_value(e.value)
                    else:
                        display_val = f"{e.value:.6g}"
                    result.append({
                        "key": f"{proc_name}_{e.flag_name}_{e.flag_value}",
                        "label": label,
                        "value": display_val,
                        "editable": True,
                    })

        elif proc_name == "RedNoise" and "RNAMP" in params and "RNIDX" in params:
            # Tempo2 RNAMP/RNIDX format -- show converted + original values
            import math
            rnamp_str = str(params["RNAMP"]).replace("D", "e").replace("d", "e")
            rnamp = float(rnamp_str)
            from jug.utils.constants import SECS_PER_YEAR
            _SEC_PER_YR = SECS_PER_YEAR
            log10_A = math.log10(2.0 * math.pi * math.sqrt(3.0) / (_SEC_PER_YR * 1e6) * rnamp)
            gamma = -float(params["RNIDX"])
            n_harmonics = int(params.get("RNC", params.get("TNREDC", params.get("TNRedC", 30))))
            result.append({"key": "RNAMP_converted", "label": "log_1_0(A) [conv]", "value": f"{log10_A:.4f}", "editable": False})
            result.append({"key": "RNIDX_converted", "label": "gamma [conv]", "value": f"{gamma:.4f}", "editable": False})
            result.append({"key": "RNC", "label": "N harmonics", "value": str(n_harmonics), "editable": False})
            result.append({"key": "RNAMP", "label": "RNAMP (orig)", "value": f"{rnamp:.5g}", "editable": True})
            result.append({"key": "RNIDX", "label": "RNIDX (orig)", "value": f"{float(params['RNIDX']):.4f}", "editable": True})

        else:
            # Generic: derive from impl class via the registry
            result = get_impl_param_defs(proc_name, params)

        return result

    def _on_process_toggled(self, name: str, enabled: bool):
        if self._noise_config is None:
            return
        if enabled:
            self._noise_config.enable(name)
        else:
            self._noise_config.disable(name)
        self.noise_config_changed.emit(self._noise_config)

    def _on_param_value_changed(self, proc_name: str, key: str, value: str):
        # Convert log_1_0(s) edits back to mus before emitting
        if self._use_log_units and proc_name in ("EQUAD", "ECORR"):
            try:
                value = str(10 ** float(value) * 1e6)
            except (ValueError, OverflowError):
                pass
        self.param_value_changed.emit(proc_name, key, value)

    def _on_subtract_toggled(self, name: str, active: bool):
        self.subtract_changed.emit(name, active)

    def _on_realise_toggled(self, name: str, active: bool):
        self.realise_changed.emit(name, active)

    def _on_collapse(self):
        self.collapse_requested.emit()

    def _toggle_estimate_list(self):
        """Toggle the estimate noise checklist dropdown."""
        vis = not self._estimate_list.isVisible()
        self._estimate_list.setVisible(vis)
        chevron = "^" if vis else "v"
        self._estimate_toggle_btn.setText(f"* Estimate Noise  {chevron}")

    def _on_estimate_clicked(self):
        """Emit signal to run MAP noise estimation with selected processes."""
        self._est_run_btn.setEnabled(False)
        self._est_run_btn.setText("* Estimating...")
        self._estimate_toggle_btn.setEnabled(False)
        selections = {
            'include_efac': self._est_cb_efac.isChecked(),
            'include_equad': self._est_cb_equad.isChecked(),
            'include_ecorr': self._est_cb_ecorr.isChecked(),
            'include_red': self._est_cb_red.isChecked(),
            'include_dm': self._est_cb_dm.isChecked(),
        }
        self.estimate_noise_requested.emit(selections)

    def set_estimate_complete(self, success: bool = True):
        """Re-enable the estimate button after estimation completes."""
        self._est_run_btn.setEnabled(True)
        self._est_run_btn.setText(">  Estimate")
        self._estimate_toggle_btn.setEnabled(True)

    def _on_remove_process(self, name: str):
        """Remove a noise process row."""
        row = self._rows.pop(name, None)
        if row is None:
            return
        row.setParent(None)
        row.deleteLater()
        if self._noise_config:
            self._noise_config.disable(name)
            self.noise_config_changed.emit(self._noise_config)
        self._rebuild_add_list()

    def _toggle_add_list(self):
        """Toggle the foldable add-process list."""
        visible = not self._add_list.isVisible()
        self._add_list.setVisible(visible)
        self._add_btn.setText("- Add Noise Process" if visible else "+ Add Noise Process")

    def _rebuild_add_list(self):
        """Rebuild the list of addable processes."""
        # Clear
        while self._add_list_layout.count() > 0:
            item = self._add_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        any_available = False
        for name in get_noise_display_order():
            if name not in self._rows:
                any_available = True
                btn = QPushButton(get_noise_label(name))
                btn.setCursor(Qt.PointingHandCursor)
                btn.setStyleSheet(
                    f"QPushButton {{ background: transparent; color: {Colors.TEXT_SECONDARY}; "
                    f"border: none; padding: 4px 8px; font-size: 11px; text-align: left; "
                    f"border-radius: 3px; }}"
                    f"QPushButton:hover {{ background-color: {Colors.SURFACE_HOVER}; "
                    f"color: {Colors.TEXT_PRIMARY}; }}"
                )
                btn.clicked.connect(lambda checked, n=name: self._add_process(n))
                self._add_list_layout.addWidget(btn)

        if not any_available:
            lbl = QLabel("All processes added")
            lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 11px; "
                              f"padding: 4px 8px; border: none;")
            self._add_list_layout.addWidget(lbl)

    def _add_process(self, name: str):
        """Add a new noise process row (initially disabled)."""
        from jug.engine.noise_mode import NoiseConfig

        if name in self._rows:
            return

        if self._noise_config is None:
            self._noise_config = NoiseConfig()

        from jug.engine.noise_mode import get_impl_param_defs

        # White noise has special per-backend defaults
        _white_defaults = {
            "EFAC": [{"key": "EFAC_default", "label": "global", "value": "1.0", "editable": True}],
            "EQUAD": [{"key": "EQUAD_default", "label": "global", "value": "0.0", "editable": True}],
            "ECORR": [{"key": "ECORR_default", "label": "global", "value": "0.0", "editable": True}],
        }

        can_sub = name in self._SUBTRACTABLE
        # Use registry defaults for any process with an impl class, else white noise defaults
        proc_params = _white_defaults.get(name) or get_impl_param_defs(name)
        row = NoiseProcessRow(name, False, proc_params, can_subtract=can_sub)
        row.toggled.connect(self._on_process_toggled)
        row.param_changed.connect(self._on_param_value_changed)
        row.remove_requested.connect(self._on_remove_process)
        row.realise_toggled.connect(self._on_realise_toggled)
        row.subtract_toggled.connect(self._on_subtract_toggled)
        self._rows[name] = row

        # Insert before the stretch
        idx = self._content_layout.count() - 1
        self._content_layout.insertWidget(idx, row)
        self._rebuild_add_list()

        # Write default values to session.params via the same signal
        for p in proc_params:
            self.param_value_changed.emit(name, p["key"], p["value"])
