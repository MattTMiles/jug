#!/usr/bin/env python3
"""
JUG GUI entry point.

Launch the JUG timing analysis GUI.
"""
import sys
import os
import argparse
import platform
from pathlib import Path

# NOTE: JAX and Astropy initialization are intentionally NOT done here.
# They are deferred to background workers (SessionWorker, WarmupWorker)
# so the GUI window appears instantly. JAX_PLATFORMS env var is set
# inside main() before any JAX import occurs.


def _startup_cookie_path():
    """Path of the marker file used to detect a crash during startup."""
    cache = os.environ.get('XDG_CACHE_HOME') or os.path.join(Path.home(), '.cache')
    return Path(cache) / 'jug' / 'startup.lock'


def _check_and_arm_crash_guard():
    """Detect a previous startup crash and arm the guard for this launch.

    A GUI crash before the first paint (bad GL driver, broken style plugin,
    a segfault inside a Qt platform plugin) cannot be caught in-process -- the
    process is simply gone. So we leave a marker on disk before starting the
    event loop and remove it once the window has painted. If the marker is
    still there at the next launch, the previous one died on the way up and we
    start in a stripped-down, maximally-portable rendering mode.

    Returns True if this launch should run in safe mode.
    """
    forced = os.environ.get('JUG_SAFE_MODE', '').lower()
    cookie = _startup_cookie_path()

    try:
        crashed = cookie.exists()
    except OSError:
        crashed = False

    if forced in ('1', 'true', 'yes'):
        safe_mode = True
    elif forced in ('0', 'false', 'no'):
        safe_mode = False
    else:
        safe_mode = crashed

    try:
        cookie.parent.mkdir(parents=True, exist_ok=True)
        cookie.write_text(f"{platform.node()} {platform.platform()}\n")
    except OSError:
        # Read-only or unwritable home: no crash detection, but never fatal.
        pass

    return safe_mode


def _disarm_crash_guard():
    """Remove the startup marker: this launch got far enough to be usable."""
    try:
        _startup_cookie_path().unlink(missing_ok=True)
    except OSError:
        pass


def main():
    """Main entry point for jug-gui command."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='JUG Timing Analysis GUI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jug-gui                                      # Launch empty GUI
  jug-gui pulsar.par pulsar.tim                # Load files on startup
  jug-gui pulsar.par pulsar.tim --fit F0 F1    # Load and pre-select F0, F1 for fitting
  jug-gui pulsar.par pulsar.tim --fit F0 F1 F2 DM1  # Fit multiple parameters
  jug-gui --gpu pulsar.par pulsar.tim          # Load files with GPU mode
  jug-gui --opengl pulsar.par pulsar.tim       # Opt in to an OpenGL plot viewport
  jug-gui --help                               # Show this help message

Note: CPU is faster for typical pulsar timing (<100k TOAs).
      GPU becomes beneficial for very large datasets (>100k TOAs) or PTAs.
      --gpu selects the JAX compute backend; --opengl only affects plot
      rendering. They are independent.
"""
    )
    parser.add_argument(
        'par_file',
        nargs='?',
        help='Path to .par file (optional)'
    )
    parser.add_argument(
        'tim_file',
        nargs='?',
        help='Path to .tim file (optional)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (default: CPU, which is faster for typical datasets)'
    )
    parser.add_argument(
        '--fit',
        nargs='+',
        metavar='PARAM',
        help='Parameters to fit (e.g., --fit F0 F1 DM). Pre-selects these in GUI.'
    )
    parser.add_argument(
        '--opengl',
        action='store_true',
        help='Render plots through an OpenGL viewport (off by default; needs a '
             'working local GL driver, and gives no measurable speedup)'
    )
    # Older READMEs documented --par/--tim. Unknown options are handed to Qt,
    # which ignores them, so that form used to launch an empty GUI without a
    # word of complaint. Accept both spellings instead.
    parser.add_argument('--par', dest='par_opt', metavar='FILE',
                        help='Path to .par file (same as the first positional argument)')
    parser.add_argument('--tim', dest='tim_opt', metavar='FILE',
                        help='Path to .tim file (same as the second positional argument)')

    args, remaining_args = parser.parse_known_args()

    # Positional wins if both spellings are given.
    args.par_file = args.par_file or args.par_opt
    args.tim_file = args.tim_file or args.tim_opt

    # Set JAX platform based on argument
    # Default to CPU (faster for typical pulsar timing)
    if args.gpu:
        os.environ['JAX_PLATFORMS'] = 'cuda'
        print("JUG GUI: Using GPU acceleration")
    else:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        # Don't print message for default behavior

    # Did the previous launch die before its window appeared?
    safe_mode = _check_and_arm_crash_guard()
    if safe_mode:
        print("JUG GUI: Previous launch crashed during startup - starting in "
              "safe mode (no OpenGL, no antialiasing).", flush=True)
        print("JUG GUI: This is automatic and clears itself once a launch "
              "succeeds. Force it any time with JUG_SAFE_MODE=1.", flush=True)

    # Show progress before X11 connection (QApplication takes ~4s over SSH)
    if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
        print("JUG GUI: Connecting to X11 display...", flush=True)

    # Configure pyqtgraph BEFORE importing Qt widgets
    import pyqtgraph as pg

    # Detect remote/SSH environment for performance optimization
    is_remote = os.environ.get('JUG_REMOTE_UI', '').lower() in ('1', 'true', 'yes')
    is_ssh = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ

    # OpenGL is OFF by default and must be asked for explicitly.
    #
    # Enabling it swaps the QGraphicsView viewport for a QOpenGLWidget. That
    # does NOT accelerate anything JUG draws: pyqtgraph's item-level GL paths
    # exist only in PlotCurveItem/PColorMeshItem, and the residual plot is
    # ScatterPlotItem + ErrorBarItem, which always paint via QPainter. Measured
    # gain is within run-to-run noise, while the GL context costs ~440 ms of
    # startup, renders a blank window over SSH (invalid context), and is the
    # one place at startup that hands control to a vendor GL driver -- which
    # segfaults outright when a conda GL stack meets system Mesa drivers.
    env_opengl = os.environ.get('JUG_PG_USE_OPENGL', '').lower()

    if env_opengl in ('1', 'true', 'yes'):
        use_opengl = True
    elif env_opengl in ('0', 'false', 'no'):
        use_opengl = False
    else:
        use_opengl = args.opengl

    # Antialiasing looks better but costs performance
    use_antialias = True
    if is_remote or is_ssh:
        # Remote mode: disable AA (the actual win over X11 forwarding)
        use_antialias = False
        print("JUG GUI: Remote mode detected - antialiasing disabled for speed")
        if use_opengl and not safe_mode:
            print("JUG GUI: Warning - OpenGL requested over SSH/Remote. "
                  "The GL context is usually invalid there (blank window).")

    use_experimental = True
    if safe_mode:
        # Strip every optional rendering path, whichever one killed the last run.
        if use_opengl:
            print("JUG GUI: Safe mode - ignoring the OpenGL request")
        use_opengl = False
        use_antialias = False
        use_experimental = False

    pg_opts = {
        'useOpenGL': use_opengl,
        'antialias': use_antialias,
        'enableExperimental': use_experimental
    }

    if use_opengl:
        print("JUG GUI: OpenGL viewport enabled (opt-in)")

    pg.setConfigOptions(**pg_opts)

    # Import after setting JAX_PLATFORMS and pyqtgraph config
    from PySide6.QtWidgets import QApplication
    from jug.gui.main_window import MainWindow

    # Pass remaining args to QApplication (for Qt-specific args like -platform)
    app = QApplication([sys.argv[0]] + remaining_args)
    app.setApplicationName("JUG Timing")
    app.setOrganizationName("Pulsar Timing")

    # Optimization: Disable menu animations on remote connections (fixes "white box" lag)
    if is_remote or is_ssh:
        from PySide6.QtCore import Qt
        app.setEffectEnabled(Qt.UI_AnimateMenu, False)
        app.setEffectEnabled(Qt.UI_FadeMenu, False)
        app.setEffectEnabled(Qt.UI_AnimateCombo, False)
        app.setEffectEnabled(Qt.UI_AnimateTooltip, False)

    # Create main window and optionally load files
    window = MainWindow(fit_params=args.fit)
    window.show()

    # Disarm the crash guard once the plot has actually painted -- that is the
    # step which touches the GL driver, so surviving it is what "started up
    # fine" means. Belt and braces: a timer in case no paint event ever reaches
    # our filter, and quit, in case the user closes the window very quickly.
    from PySide6.QtCore import QTimer, QObject, QEvent

    class _FirstPaintWatcher(QObject):
        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Paint:
                _disarm_crash_guard()
                obj.removeEventFilter(self)
            return False

    _paint_watcher = _FirstPaintWatcher()
    _paint_target = getattr(window, 'plot_widget', None)
    _paint_target = _paint_target.viewport() if _paint_target is not None else window
    _paint_target.installEventFilter(_paint_watcher)
    QTimer.singleShot(5000, _disarm_crash_guard)
    app.aboutToQuit.connect(_disarm_crash_guard)

    # Load files via QTimer to allow UI to render first (perceived speedup)
    if args.par_file or args.tim_file:
        QTimer.singleShot(100, lambda: window.load_files_from_args(args.par_file, args.tim_file))

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
