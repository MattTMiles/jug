"""Disk memoization for geometry arrays.

This module provides disk caching for expensive geometry computations
(observatory position/velocity at SSB, etc.) to accelerate cold starts
when reopening the same dataset.

The cache is keyed by:
- Hash of TDB times array
- Observatory coordinates
- Ephemeris selection (e.g., "de440")
- Astropy/ERFA versions

This ensures bit-for-bit identical results when loading from cache.

Usage:
    from jug.utils.geom_cache import GeometryDiskCache
    cache = GeometryDiskCache()
    
    # Try to load cached geometry
    cached = cache.load(tdb_mjd, obs_itrf_km, ephemeris="de440")
    if cached is not None:
        ssb_obs_pos, ssb_obs_vel = cached
    else:
        # Compute and save
        ssb_obs_pos, ssb_obs_vel = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km)
        cache.save(tdb_mjd, obs_itrf_km, ssb_obs_pos, ssb_obs_vel, ephemeris="de440")

Environment Variables:
    JUG_GEOM_CACHE_DIR: Override cache directory
    JUG_GEOM_CACHE_DISABLE: Set to "1" to disable disk caching
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def get_cache_directory() -> Path:
    """Get the geometry cache directory.
    
    Priority:
    1. JUG_GEOM_CACHE_DIR environment variable
    2. $TMPDIR/jug_geom_cache (if TMPDIR set and writable)
    3. ~/.cache/jug/geometry (default)
    """
    # Priority 1: Explicit env var
    if env_dir := os.environ.get('JUG_GEOM_CACHE_DIR'):
        return Path(env_dir)
    
    # Priority 2: TMPDIR
    if tmpdir := os.environ.get('TMPDIR'):
        tmpdir_path = Path(tmpdir)
        if tmpdir_path.exists() and os.access(tmpdir_path, os.W_OK):
            return tmpdir_path / "jug_geom_cache"
    
    # Priority 3: User cache
    return Path.home() / ".cache" / "jug" / "geometry"


def compute_array_hash(arr: np.ndarray) -> str:
    """Compute a stable hash of a numpy array.
    
    Uses the raw bytes of the array for exact matching.
    """
    # Ensure contiguous float64 array
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def get_version_info() -> dict:
    """Get version information for cache validation."""
    info = {
        'numpy': np.__version__,
        'astropy': None,
        'erfa': None,
    }
    
    try:
        import astropy
        info['astropy'] = astropy.__version__
    except ImportError:
        pass
    
    try:
        import erfa
        info['erfa'] = erfa.__version__
    except ImportError:
        pass
    
    return info


class GeometryDiskCache:
    """Disk cache for geometry arrays.
    
    Caches the outputs of compute_ssb_obs_pos_vel and related functions
    to accelerate cold starts when reopening the same dataset.
    
    The cache uses .npz files with metadata JSON for validation.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the geometry cache.
        
        Parameters
        ----------
        cache_dir : Path, optional
            Override cache directory. If None, uses get_cache_directory().
        """
        self.enabled = os.environ.get('JUG_GEOM_CACHE_DISABLE', '').strip() != '1'
        
        if cache_dir is None:
            cache_dir = get_cache_directory()
        
        self.cache_dir = Path(cache_dir)
        
        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.debug(f"Could not create cache directory: {e}")
                self.enabled = False
    
    def _compute_cache_key(
        self,
        tdb_mjd: np.ndarray,
        obs_itrf_km: np.ndarray,
        ephemeris: str = "de440"
    ) -> str:
        """Compute the cache key for a geometry computation.
        
        The key includes:
        - Hash of TDB times
        - Hash of observatory coordinates
        - Ephemeris name
        """
        tdb_hash = compute_array_hash(tdb_mjd)
        obs_hash = compute_array_hash(obs_itrf_km)
        
        return f"geom_{tdb_hash}_{obs_hash}_{ephemeris}"
    
    def _get_cache_paths(self, key: str) -> Tuple[Path, Path]:
        """Get paths for cache data and metadata files."""
        data_path = self.cache_dir / f"{key}.npz"
        meta_path = self.cache_dir / f"{key}.json"
        return data_path, meta_path
    
    def _validate_metadata(self, metadata: dict) -> bool:
        """Check if cached metadata matches current environment."""
        current_versions = get_version_info()
        
        # Check version compatibility
        cached_versions = metadata.get('versions', {})
        
        # Astropy version must match (affects IERS/coordinate transforms)
        if cached_versions.get('astropy') != current_versions.get('astropy'):
            return False
        
        # ERFA version must match (core coordinate routines)
        if cached_versions.get('erfa') != current_versions.get('erfa'):
            return False
        
        return True
    
    def load(
        self,
        tdb_mjd: np.ndarray,
        obs_itrf_km: np.ndarray,
        ephemeris: str = "de440"
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Try to load cached geometry arrays.
        
        Parameters
        ----------
        tdb_mjd : np.ndarray
            TDB times in MJD
        obs_itrf_km : np.ndarray
            Observatory ITRF position (km)
        ephemeris : str
            Ephemeris name (e.g., "de440")
        
        Returns
        -------
        tuple or None
            (ssb_obs_pos, ssb_obs_vel) if cache hit, None otherwise.
            Arrays are float64, shapes (n_times, 3).
        """
        if not self.enabled:
            return None
        
        key = self._compute_cache_key(tdb_mjd, obs_itrf_km, ephemeris)
        data_path, meta_path = self._get_cache_paths(key)
        
        # Check if cache files exist
        if not data_path.exists() or not meta_path.exists():
            return None
        
        try:
            # Load and validate metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            if not self._validate_metadata(metadata):
                logger.debug(f"Cache metadata mismatch for {key}")
                return None
            
            # Verify array hashes match
            if metadata.get('tdb_hash') != compute_array_hash(tdb_mjd):
                logger.debug(f"TDB hash mismatch for {key}")
                return None
            
            if metadata.get('obs_hash') != compute_array_hash(obs_itrf_km):
                logger.debug(f"Observatory hash mismatch for {key}")
                return None
            
            # Load arrays
            with np.load(data_path) as data:
                ssb_obs_pos = data['ssb_obs_pos'].astype(np.float64)
                ssb_obs_vel = data['ssb_obs_vel'].astype(np.float64)
            
            # Verify shapes
            if ssb_obs_pos.shape != (len(tdb_mjd), 3):
                logger.debug(f"Position shape mismatch for {key}")
                return None
            
            if ssb_obs_vel.shape != (len(tdb_mjd), 3):
                logger.debug(f"Velocity shape mismatch for {key}")
                return None
            
            logger.debug(f"Loaded geometry from cache: {key}")
            return ssb_obs_pos, ssb_obs_vel
            
        except Exception as e:
            logger.debug(f"Failed to load cache {key}: {e}")
            return None
    
    def save(
        self,
        tdb_mjd: np.ndarray,
        obs_itrf_km: np.ndarray,
        ssb_obs_pos: np.ndarray,
        ssb_obs_vel: np.ndarray,
        ephemeris: str = "de440"
    ) -> bool:
        """Save geometry arrays to disk cache.
        
        Parameters
        ----------
        tdb_mjd : np.ndarray
            TDB times in MJD
        obs_itrf_km : np.ndarray
            Observatory ITRF position (km)
        ssb_obs_pos : np.ndarray
            SSB observatory position (km), shape (n_times, 3)
        ssb_obs_vel : np.ndarray
            SSB observatory velocity (km/s), shape (n_times, 3)
        ephemeris : str
            Ephemeris name
        
        Returns
        -------
        bool
            True if saved successfully
        """
        if not self.enabled:
            return False
        
        key = self._compute_cache_key(tdb_mjd, obs_itrf_km, ephemeris)
        data_path, meta_path = self._get_cache_paths(key)
        
        try:
            # Prepare metadata
            metadata = {
                'tdb_hash': compute_array_hash(tdb_mjd),
                'obs_hash': compute_array_hash(obs_itrf_km),
                'ephemeris': ephemeris,
                'n_times': len(tdb_mjd),
                'versions': get_version_info(),
            }
            
            # Ensure arrays are float64
            ssb_obs_pos = np.asarray(ssb_obs_pos, dtype=np.float64)
            ssb_obs_vel = np.asarray(ssb_obs_vel, dtype=np.float64)
            
            # Write atomically using temp file + rename
            # First, write metadata
            with tempfile.NamedTemporaryFile(
                mode='w', 
                dir=self.cache_dir, 
                suffix='.json',
                delete=False
            ) as f:
                json.dump(metadata, f, indent=2)
                temp_meta = f.name
            
            # Then write data
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir,
                suffix='.npz',
                delete=False
            ) as f:
                temp_data = f.name
            
            np.savez(temp_data, ssb_obs_pos=ssb_obs_pos, ssb_obs_vel=ssb_obs_vel)
            
            # Atomic rename
            os.replace(temp_meta, meta_path)
            os.replace(temp_data, data_path)
            
            logger.debug(f"Saved geometry to cache: {key}")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to save cache {key}: {e}")
            # Clean up temp files
            try:
                if 'temp_meta' in locals():
                    os.unlink(temp_meta)
                if 'temp_data' in locals():
                    os.unlink(temp_data)
            except Exception:
                pass
            return False
    
    def clear(self) -> int:
        """Clear all cached geometry files.
        
        Returns
        -------
        int
            Number of cache entries removed
        """
        count = 0
        
        if not self.cache_dir.exists():
            return count
        
        for f in self.cache_dir.glob("geom_*.npz"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        
        for f in self.cache_dir.glob("geom_*.json"):
            try:
                f.unlink()
            except Exception:
                pass
        
        return count
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache.
        
        Returns
        -------
        dict
            Cache statistics including size, entry count, etc.
        """
        stats = {
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'entry_count': 0,
            'total_size_mb': 0.0
        }
        
        if not self.cache_dir.exists():
            return stats
        
        total_size = 0
        for f in self.cache_dir.glob("geom_*.npz"):
            stats['entry_count'] += 1
            total_size += f.stat().st_size
        
        stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats


# Global cache instance
_default_cache: Optional[GeometryDiskCache] = None


def get_geometry_cache() -> GeometryDiskCache:
    """Get the default geometry cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = GeometryDiskCache()
    return _default_cache
