#!/usr/bin/env python3
"""
JUG Server CLI (jugd)
=====================

Command-line interface for starting the JUG timing server.

Usage:
------
# Start server on default port (8080)
jugd serve

# Custom port
jugd serve --port 9000

# Development mode with auto-reload
jugd serve --reload

# Bind to all interfaces (NOT RECOMMENDED for security)
jugd serve --host 0.0.0.0
"""

import argparse
import sys


def main():
    """Main entry point for jugd CLI."""
    parser = argparse.ArgumentParser(
        description="JUG Timing Server (jugd)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port
  jugd serve
  
  # Custom port
  jugd serve --port 9000
  
  # Development mode
  jugd serve --reload
  
  # Remote access via SSH tunnel
  # On cluster:
  jugd serve --port 8080
  
  # On local machine:
  ssh -L 8080:localhost:8080 user@cluster
  
  # Then connect to http://localhost:8080
  # API docs at http://localhost:8080/docs

Security:
  - Server binds to localhost (127.0.0.1) by default
  - For remote access, use SSH tunnel (never bind to 0.0.0.0)
  - All file operations are restricted to server filesystem
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start JUG server')
    serve_parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1 for security)'
    )
    serve_parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to bind to (default: 8080)'
    )
    serve_parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        try:
            from jug.server import run_server
        except ImportError as e:
            print("Error: FastAPI not installed", file=sys.stderr)
            print("Install with: pip install jug-timing[server]", file=sys.stderr)
            sys.exit(1)
        
        # Warn if binding to all interfaces
        if args.host not in ['127.0.0.1', 'localhost']:
            print("WARNING: Binding to non-localhost address", file=sys.stderr)
            print("This exposes the server to your network!", file=sys.stderr)
            print("For remote access, use SSH tunnel instead.", file=sys.stderr)
            print()
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
        
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
