"""Dialog for residual averaging options."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QPushButton,
)
from PySide6.QtCore import Qt
from jug.gui.theme import Colors, Typography, get_border_subtle


class AveragingDialog(QDialog):
    """Lightweight dialog for averaging settings.

    Uses styled toggle buttons that match the app's parameter-fit buttons.
    """

    def __init__(self, parent=None, n_toas: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Average Residuals")
        self.setFixedWidth(320)
        self._build_ui(n_toas)

    def _build_ui(self, n_toas: int):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 12, 16, 12)

        # Apply dialog background to match app theme
        self.setStyleSheet(f"""
            QDialog {{
                background: {Colors.BG_PRIMARY};
                color: {Colors.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {Colors.TEXT_PRIMARY};
                font-family: {Typography.FONT_FAMILY};
            }}
        """)

        # Info
        info = QLabel(f"{n_toas} residuals")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: {Typography.SIZE_SM};")
        layout.addWidget(info)

        # Mode toggle buttons (mutually exclusive, styled like app buttons)
        mode_label = QLabel("Averaging mode")
        mode_label.setStyleSheet(f"font-weight: bold; font-size: {Typography.SIZE_SM};")
        layout.addWidget(mode_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)

        self._mode_buttons = {}
        modes = [
            ("time", "Time"),
            ("frequency", "Frequency"),
            ("backend", "Backend"),
        ]
        for key, label in modes:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda checked, k=key: self._select_mode(k))
            self._mode_buttons[key] = btn
            mode_row.addWidget(btn)

        self._mode_buttons["time"].setChecked(True)
        self._selected_mode = "time"
        self._apply_mode_styles()
        layout.addLayout(mode_row)

        # Window size — time presets (swapped to freq presets when mode changes)
        self._window_label = QLabel("Max gap between TOAs in group")
        self._window_label.setStyleSheet(f"font-size: {Typography.SIZE_SM};")
        layout.addWidget(self._window_label)

        self._window_row = QHBoxLayout()
        self._window_row.setSpacing(6)

        # Preset buttons container (rebuilt on mode change)
        self._preset_container = QHBoxLayout()
        self._preset_container.setSpacing(6)
        self._window_row.addLayout(self._preset_container)

        self._spin = QDoubleSpinBox()
        self._spin.setFixedHeight(28)
        self._spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {get_border_subtle()};
                border-radius: 4px;
                padding: 2px 6px;
                font-family: {Typography.FONT_MONO};
                font-size: {Typography.SIZE_SM};
            }}
        """)
        self._spin.valueChanged.connect(self._on_spin_changed)
        self._window_row.addWidget(self._spin)

        self._window_buttons = {}
        self._selected_window = 1.0
        self._build_window_presets_for_mode("time")
        layout.addLayout(self._window_row)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(32)
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.setStyleSheet(self._action_btn_style(primary=False))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton("Average")
        ok_btn.setFixedHeight(32)
        ok_btn.setDefault(True)
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.setStyleSheet(self._action_btn_style(primary=True))
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)

        layout.addLayout(btn_row)

    def _toggle_btn_style(self, checked: bool) -> str:
        if checked:
            return f"""
                QPushButton {{
                    background: {Colors.ACCENT_PRIMARY};
                    color: {Colors.TEXT_PRIMARY};
                    border: 1px solid {Colors.ACCENT_PRIMARY};
                    border-radius: 4px;
                    font-family: {Typography.FONT_FAMILY};
                    font-size: {Typography.SIZE_SM};
                    font-weight: bold;
                    padding: 4px 10px;
                }}
            """
        else:
            return f"""
                QPushButton {{
                    background: {Colors.BG_SECONDARY};
                    color: {Colors.TEXT_MUTED};
                    border: 1px solid {get_border_subtle()};
                    border-radius: 4px;
                    font-family: {Typography.FONT_FAMILY};
                    font-size: {Typography.SIZE_SM};
                    padding: 4px 10px;
                }}
                QPushButton:hover {{
                    color: {Colors.TEXT_PRIMARY};
                    border-color: {Colors.TEXT_MUTED};
                }}
            """

    def _action_btn_style(self, primary: bool) -> str:
        if primary:
            return f"""
                QPushButton {{
                    background: {Colors.ACCENT_PRIMARY};
                    color: {Colors.TEXT_PRIMARY};
                    border: none;
                    border-radius: 4px;
                    font-family: {Typography.FONT_FAMILY};
                    font-size: {Typography.SIZE_BASE};
                    font-weight: bold;
                    padding: 6px 20px;
                }}
                QPushButton:hover {{ background: {Colors.ACCENT_SUCCESS}; }}
            """
        else:
            return f"""
                QPushButton {{
                    background: transparent;
                    color: {Colors.TEXT_MUTED};
                    border: 1px solid {get_border_subtle()};
                    border-radius: 4px;
                    font-family: {Typography.FONT_FAMILY};
                    font-size: {Typography.SIZE_BASE};
                    padding: 6px 16px;
                }}
                QPushButton:hover {{ color: {Colors.TEXT_PRIMARY}; }}
            """

    def _select_mode(self, key: str):
        self._selected_mode = key
        for k, btn in self._mode_buttons.items():
            btn.setChecked(k == key)
        self._apply_mode_styles()
        self._build_window_presets_for_mode(key)

    def _apply_mode_styles(self):
        for k, btn in self._mode_buttons.items():
            btn.setStyleSheet(self._toggle_btn_style(k == self._selected_mode))

    def _select_window(self, val: float):
        self._selected_window = val
        self._spin.blockSignals(True)
        self._spin.setValue(val)
        self._spin.blockSignals(False)
        for v, btn in self._window_buttons.items():
            btn.setChecked(v == val)
        self._apply_window_styles()

    def _on_spin_changed(self, val: float):
        self._selected_window = val
        # Deselect preset buttons if value doesn't match any
        for v, btn in self._window_buttons.items():
            btn.setChecked(abs(v - val) < 0.001)
        self._apply_window_styles()

    def _apply_window_styles(self):
        for v, btn in self._window_buttons.items():
            btn.setStyleSheet(self._toggle_btn_style(btn.isChecked()))

    def _build_window_presets_for_mode(self, mode: str):
        """Rebuild preset buttons and spin range for the selected mode."""
        # Clear existing preset buttons
        while self._preset_container.count():
            item = self._preset_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._window_buttons.clear()

        if mode == "frequency":
            presets = [("100", 100.0), ("200", 200.0), ("500", 500.0)]
            self._window_label.setText("Frequency bandwidth (MHz)")
            self._spin.blockSignals(True)
            self._spin.setRange(1.0, 10000.0)
            self._spin.setDecimals(0)
            self._spin.setSuffix(" MHz")
            self._spin.setValue(100.0)
            self._spin.blockSignals(False)
            self._selected_window = 100.0
        else:
            presets = [("1d", 1.0), ("7d", 7.0), ("30d", 30.0)]
            self._window_label.setText("Max gap between TOAs in group")
            self._spin.blockSignals(True)
            self._spin.setRange(0.001, 3650.0)
            self._spin.setDecimals(2)
            self._spin.setSuffix(" d")
            self._spin.setValue(1.0)
            self._spin.blockSignals(False)
            self._selected_window = 1.0

        default_val = presets[0][1]
        for label, val in presets:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.clicked.connect(lambda checked, v=val: self._select_window(v))
            self._window_buttons[val] = btn
            self._preset_container.addWidget(btn)

        self._window_buttons[default_val].setChecked(True)
        self._apply_window_styles()

    def get_settings(self) -> dict:
        """Return the selected averaging settings."""
        if self._selected_mode == "frequency":
            return {
                "mode": self._selected_mode,
                "window_days": 1.0,
                "window_mhz": self._selected_window,
            }
        return {
            "mode": self._selected_mode,
            "window_days": self._selected_window,
            "window_mhz": 100.0,
        }
