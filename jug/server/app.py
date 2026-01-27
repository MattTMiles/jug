"""
FastAPI Server Application
===========================

Provides REST API for remote JUG engine access.

API Endpoints:
--------------
GET  /health          - Health check
GET  /version         - API version + engine version
GET  /list_dir        - List directory contents (with path param)
POST /open_session    - Open timing session
POST /compute         - Compute residuals
POST /fit             - Fit parameters (with progress)
POST /save            - Save results
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

# FastAPI will be optional dependency
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

from jug.engine import open_session, TimingSession
import jug


# Pydantic models for request/response
if HAS_FASTAPI:
    class OpenSessionRequest(BaseModel):
        par_path: str
        tim_path: str
        clock_dir: Optional[str] = None
        
    class ComputeRequest(BaseModel):
        session_id: str
        params: Optional[Dict[str, float]] = None
        
    class FitRequest(BaseModel):
        session_id: str
        fit_params: List[str]
        max_iter: int = 25
        
    class ListDirResponse(BaseModel):
        path: str
        entries: List[Dict[str, Any]]


# Global session storage (TODO: move to Redis for multi-process)
_sessions: Dict[str, TimingSession] = {}


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns
    -------
    app : FastAPI
        Configured FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install jug-timing[server]"
        )
    
    app = FastAPI(
        title="JUG Timing Server",
        description="Remote access to JUG pulsar timing engine",
        version="0.1.0",
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/version")
    async def version():
        """Get API and engine version."""
        return {
            "api_version": "0.1.0",
            "engine_version": jug.__version__ if hasattr(jug, '__version__') else "unknown",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    @app.get("/list_dir")
    async def list_dir(path: str = Query(".", description="Directory path to list")):
        """
        List directory contents.
        
        Allows remote file browsing for selecting .par and .tim files.
        """
        try:
            dir_path = Path(path).expanduser().resolve()
            
            if not dir_path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {path}")
            
            if not dir_path.is_dir():
                raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
            
            entries = []
            for item in sorted(dir_path.iterdir()):
                try:
                    entries.append({
                        "name": item.name,
                        "path": str(item),
                        "is_dir": item.is_dir(),
                        "is_file": item.is_file(),
                        "size": item.stat().st_size if item.is_file() else None,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    })
                except (PermissionError, OSError):
                    # Skip files we can't access
                    continue
            
            return {
                "path": str(dir_path),
                "parent": str(dir_path.parent) if dir_path.parent != dir_path else None,
                "entries": entries
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/open_session")
    async def open_session_endpoint(request: OpenSessionRequest):
        """
        Open a new timing session.
        
        Returns a session_id for subsequent operations.
        """
        try:
            # Validate paths
            par_path = Path(request.par_path).expanduser().resolve()
            tim_path = Path(request.tim_path).expanduser().resolve()
            
            if not par_path.exists():
                raise HTTPException(status_code=404, detail=f"Par file not found: {request.par_path}")
            if not tim_path.exists():
                raise HTTPException(status_code=404, detail=f"Tim file not found: {request.tim_path}")
            
            # Create session
            session = open_session(
                par_file=par_path,
                tim_file=tim_path,
                clock_dir=request.clock_dir,
                verbose=False
            )
            
            # Generate session ID and store
            session_id = str(uuid.uuid4())
            _sessions[session_id] = session
            
            return {
                "session_id": session_id,
                "par_file": str(par_path),
                "tim_file": str(tim_path),
                "ntoas": session.get_toa_count(),
                "created": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/compute")
    async def compute_residuals_endpoint(request: ComputeRequest):
        """
        Compute residuals for a session.
        """
        session = _sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
        
        try:
            result = session.compute_residuals(params=request.params)
            
            # Convert numpy arrays to lists for JSON serialization
            return {
                "session_id": request.session_id,
                "rms_us": float(result['rms_us']),
                "ntoas": len(result['residuals_us']),
                "residuals_us": result['residuals_us'].tolist(),
                "tdb_mjd": result['tdb_mjd'].tolist(),
                "errors_us": result.get('errors_us', [None]*len(result['residuals_us']))
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/fit")
    async def fit_parameters_endpoint(request: FitRequest):
        """
        Fit parameters for a session.
        
        TODO: Add progress streaming via WebSocket
        """
        session = _sessions.get(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
        
        try:
            result = session.fit_parameters(
                fit_params=request.fit_params,
                max_iter=request.max_iter,
                verbose=False
            )
            
            return {
                "session_id": request.session_id,
                "final_params": result['final_params'],
                "uncertainties": result['uncertainties'],
                "final_rms": float(result['final_rms']),
                "iterations": int(result['iterations']),
                "converged": bool(result['converged']),
                "total_time": float(result['total_time'])
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    reload: bool = False
):
    """
    Run the JUG server.
    
    Parameters
    ----------
    host : str, default "127.0.0.1"
        Host to bind to (localhost by default for security)
    port : int, default 8080
        Port to bind to
    reload : bool, default False
        Enable auto-reload for development
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI not installed. Install with: pip install jug-timing[server]"
        )
    
    import uvicorn
    
    app = create_app()
    
    print(f"Starting JUG server on {host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"Health check: http://{host}:{port}/health")
    print()
    print("For remote access, use SSH tunnel:")
    print(f"  ssh -L {port}:localhost:{port} user@cluster")
    print()
    
    uvicorn.run(app, host=host, port=port, reload=reload)
