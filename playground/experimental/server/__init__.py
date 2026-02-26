"""
JUG Server (jugd) - Remote Engine Access
=========================================

This module provides a REST API server for remote access to the JUG
timing engine. The server can run on a cluster login node and be
accessed via SSH tunnel from a remote Tauri GUI.

Architecture:
- FastAPI REST API
- Session management (multiple concurrent sessions)
- File system browsing (cluster paths)
- Progress streaming (WebSocket/SSE for fits)
- Secure by default (localhost only, SSH tunnel for remote)

Usage:
------
# Start server on cluster
jugd serve --port 8080

# Or from Python
from jug.server import run_server
run_server(host='localhost', port=8080)

# Client connects via SSH tunnel
ssh -L 8080:localhost:8080 user@cluster
# Then connect to http://localhost:8080
"""

from jug.server.app import create_app, run_server

__all__ = ['create_app', 'run_server']
