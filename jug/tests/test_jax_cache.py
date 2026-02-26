"""Tests for JAX compilation cache configuration.

Verifies that configure_jax_compilation_cache() works correctly and
doesn't crash under various conditions.
"""

import os
import tempfile
from pathlib import Path


class TestJaxCacheConfiguration:
    """Test JAX compilation cache configuration."""

    def test_configure_does_not_crash(self):
        """Test that configure_jax_compilation_cache() doesn't crash."""
        from jug.utils.jax_cache import configure_jax_compilation_cache
        
        # Should not raise any exceptions
        result = configure_jax_compilation_cache()
        
        # Should return True (success) or False (graceful failure)
        assert isinstance(result, bool)

    def test_get_cache_directory_with_env_var(self):
        """Test cache directory selection with JUG_JAX_CACHE_DIR set."""
        from jug.utils.jax_cache import get_cache_directory
        
        test_dir = "/tmp/test_jug_cache_12345"
        original = os.environ.get('JUG_JAX_CACHE_DIR')
        
        try:
            os.environ['JUG_JAX_CACHE_DIR'] = test_dir
            result = get_cache_directory()
            assert result == Path(test_dir)
        finally:
            if original is None:
                os.environ.pop('JUG_JAX_CACHE_DIR', None)
            else:
                os.environ['JUG_JAX_CACHE_DIR'] = original

    def test_get_cache_directory_with_tmpdir(self):
        """Test cache directory selection with TMPDIR set."""
        from jug.utils.jax_cache import get_cache_directory
        
        # Save original values
        original_jug = os.environ.get('JUG_JAX_CACHE_DIR')
        original_tmpdir = os.environ.get('TMPDIR')
        
        try:
            # Clear JUG_JAX_CACHE_DIR so TMPDIR is used
            os.environ.pop('JUG_JAX_CACHE_DIR', None)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ['TMPDIR'] = tmpdir
                result = get_cache_directory()
                assert result == Path(tmpdir) / "jug_jax_cache"
        finally:
            # Restore original values
            if original_jug is not None:
                os.environ['JUG_JAX_CACHE_DIR'] = original_jug
            if original_tmpdir is not None:
                os.environ['TMPDIR'] = original_tmpdir

    def test_get_cache_directory_default(self):
        """Test cache directory selection falls back to ~/.cache."""
        from jug.utils.jax_cache import get_cache_directory
        
        # Save and clear environment
        original_jug = os.environ.get('JUG_JAX_CACHE_DIR')
        original_tmpdir = os.environ.get('TMPDIR')
        
        try:
            os.environ.pop('JUG_JAX_CACHE_DIR', None)
            os.environ.pop('TMPDIR', None)
            
            result = get_cache_directory()
            assert result == Path.home() / ".cache" / "jug" / "jax_compilation"
        finally:
            if original_jug is not None:
                os.environ['JUG_JAX_CACHE_DIR'] = original_jug
            if original_tmpdir is not None:
                os.environ['TMPDIR'] = original_tmpdir

    def test_get_cache_info(self):
        """Test get_cache_info returns expected structure."""
        from jug.utils.jax_cache import get_cache_info
        
        info = get_cache_info()
        
        assert isinstance(info, dict)
        assert 'configured' in info
        assert 'cache_dir' in info
        assert 'jax_version' in info
        assert isinstance(info['configured'], bool)

    def test_idempotent_configuration(self):
        """Test that calling configure multiple times is safe."""
        from jug.utils.jax_cache import configure_jax_compilation_cache
        
        # Call multiple times
        result1 = configure_jax_compilation_cache()
        result2 = configure_jax_compilation_cache()
        result3 = configure_jax_compilation_cache()
        
        # Should all return same result (True after first successful config)
        assert result1 == result2 == result3


def run_tests():
    """Run all JAX cache tests."""
    test = TestJaxCacheConfiguration()
    
    print("Testing JAX compilation cache...")
    
    test.test_configure_does_not_crash()
    print("[x] configure_jax_compilation_cache() doesn't crash")
    
    test.test_get_cache_directory_with_env_var()
    print("[x] Cache directory respects JUG_JAX_CACHE_DIR")
    
    test.test_get_cache_directory_with_tmpdir()
    print("[x] Cache directory respects TMPDIR")
    
    test.test_get_cache_directory_default()
    print("[x] Cache directory falls back to ~/.cache")
    
    test.test_get_cache_info()
    print("[x] get_cache_info() returns expected structure")
    
    test.test_idempotent_configuration()
    print("[x] Multiple configure calls are idempotent")
    
    print("\nAll JAX cache tests passed!")


if __name__ == '__main__':
    run_tests()
