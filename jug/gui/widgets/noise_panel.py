"""Noise diagnostics dialog.

Shows a formatted noise/backend report using data from
``jug.engine.diagnostics`` and ``jug.engine.validation``.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QPushButton,
    QLabel, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from jug.gui.theme import Colors, Typography


class NoiseDiagnosticsDialog(QDialog):
    """Modal dialog displaying the noise diagnostics report."""

    def __init__(self, report_text: str, validation_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Noise & Backend Diagnostics")
        self.setMinimumSize(560, 480)
        self.resize(640, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title = QLabel("📋  Noise & Backend Report")
        title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 16px;
            font-weight: {Typography.WEIGHT_BOLD};
        """)
        layout.addWidget(title)

        # Report content (monospace text browser)
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(False)
        self.text_browser.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {Colors.BG_SECONDARY};
                color: {Colors.TEXT_PRIMARY};
                border: 1px solid {Colors.SURFACE_BORDER};
                border-radius: 8px;
                padding: 16px;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                selection-background-color: {Colors.ACCENT_PRIMARY};
            }}
        """)

        # Build combined report
        full_report = ""
        if validation_text:
            full_report += "═══ DATA VALIDATION ═══\n\n"
            full_report += validation_text + "\n\n"
        if report_text:
            full_report += "═══ NOISE DIAGNOSTICS ═══\n\n"
            full_report += report_text
        if not full_report:
            full_report = "No diagnostics available. Load a .par and .tim file first."

        self.text_browser.setPlainText(full_report)
        layout.addWidget(self.text_browser)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BTN_SECONDARY_BG};
                color: {Colors.BTN_SECONDARY_TEXT};
                border: 1px solid {Colors.SURFACE_BORDER};
                border-radius: 6px;
                padding: 8px 24px;
                font-size: 13px;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {Colors.BTN_SECONDARY_HOVER};
            }}
        """)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        # Dialog styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_PRIMARY};
            }}
        """)


def build_noise_report(session) -> tuple:
    """Build the noise diagnostics report from a session.

    Returns (report_text, validation_text) tuple.
    """
    report_text = ""
    validation_text = ""

    if session is None:
        return report_text, validation_text

    try:
        from jug.engine.diagnostics import build_diagnostics_report, format_diagnostics_text
        from jug.engine.validation import validate_toa_data

        params = session.params
        toa_data = session._get_toa_data()

        # Validation
        try:
            issues = validate_toa_data(
                mjd_values=toa_data['tdb_mjd'],
                freq_values=toa_data.get('freq_mhz', None),
                error_values=toa_data.get('errors_us', None),
                toa_flags=toa_data.get('toa_flags', None),
                strict=False,
            )
            if issues:
                validation_text = "\n".join(f"⚠  {issue}" for issue in issues)
            else:
                validation_text = "✓  All TOA data checks passed."
        except Exception as e:
            validation_text = f"Validation error: {e}"

        # Diagnostics
        try:
            # Collect noise entries and flags
            noise_entries = params.get('_noise_entries', [])
            toa_flags = toa_data.get('toa_flags', [])
            backends = []
            for f in toa_flags:
                be = f.get('be', f.get('f', f.get('sys', 'unknown')))
                backends.append(be)

            report = build_diagnostics_report(
                backends=backends,
                noise_entries=noise_entries,
            )
            report_text = format_diagnostics_text(report)
        except Exception as e:
            report_text = f"Diagnostics error: {e}"

    except ImportError as e:
        report_text = f"Import error: {e}"

    return report_text, validation_text
