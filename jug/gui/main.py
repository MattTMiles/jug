#!/usr/bin/env python3
"""
JUG GUI entry point.

Launch the JUG timing analysis GUI.
"""
import sys
import os
import argparse

# Configure JAX (x64 precision + compilation cache) EARLY
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

# Configure Astropy for deterministic behavior (before any Astropy imports)
from jug.utils.astropy_config import configure_astropy
configure_astropy()


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
  jug-gui --help                               # Show this help message

Note: CPU is faster for typical pulsar timing (<100k TOAs).
      GPU becomes beneficial for very large datasets (>100k TOAs) or PTAs.
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

    args, remaining_args = parser.parse_known_args()

    # Set JAX platform based on argument
    # Default to CPU (faster for typical pulsar timing)
    if args.gpu:
        os.environ['JAX_PLATFORMS'] = 'cuda'
        print("JUG GUI: Using GPU acceleration")
    else:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        # Don't print message for default behavior

    # Configure pyqtgraph BEFORE importing Qt widgets
    import pyqtgraph as pg

    # Detect remote/SSH environment for performance optimization
    is_remote = os.environ.get('JUG_REMOTE_UI', '').lower() in ('1', 'true', 'yes')
    is_ssh = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ

    # Performance tuning: OpenGL is crucial for >1k points
    # Default to True unless explicitly disabled
    env_opengl = os.environ.get('JUG_PG_USE_OPENGL', '').lower()
    
    if env_opengl == 'true':
        use_opengl = True
    elif env_opengl == 'false':
        use_opengl = False
    else:
        # Auto-detect: Enable locally, disable remotely (causes white screen/hangs)
        use_opengl = not (is_remote or is_ssh)

    # Antialiasing looks better but costs performance
    use_antialias = True
    if is_remote or is_ssh:
        # Remote mode: disable AA and OpenGL defaults
        use_antialias = False
        if use_opengl:
             print("JUG GUI: Warning - OpenGL enabled over SSH/Remote. This may cause hangs.")
        else:
             print("JUG GUI: Remote mode detected - OpenGL disabled for stability")

    pg_opts = {
        'useOpenGL': use_opengl, 
        'antialias': use_antialias,
        'enableExperimental': True
    }
    
    if use_opengl:
        print("JUG GUI: OpenGL acceleration enabled")

    pg.setConfigOptions(**pg_opts)

    # Import after setting JAX_PLATFORMS and pyqtgraph config
    from PySide6.QtWidgets import QApplication
    from jug.gui.main_window import MainWindow

    # Pass remaining args to QApplication (for Qt-specific args like -platform)
    app = QApplication([sys.argv[0]] + remaining_args)
    app.setApplicationName("JUG Timing")
    app.setOrganizationName("Pulsar Timing")

    # Create main window and optionally load files
    window = MainWindow(fit_params=args.fit)

    # Load files if provided
    if args.par_file or args.tim_file:
        window.load_files_from_args(args.par_file, args.tim_file)

    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
