#!/usr/bin/env python3
"""
JUG GUI entry point.

Launch the JUG timing analysis GUI.
"""
import sys
from PySide6.QtWidgets import QApplication
from jug.gui.main_window import MainWindow


def main():
    """Main entry point for jug-gui command."""
    app = QApplication(sys.argv)
    app.setApplicationName("JUG Timing")
    app.setOrganizationName("Pulsar Timing")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
