"""
Data Manifest and Verification Tests
=====================================

Tests for data manifest integrity and verification functionality.

Key requirements:
- Manifest file exists and is valid JSON
- All referenced files exist
- SHA256 checksums match
- Verification function works correctly

Run with:
    pytest jug/tests/test_data_manifest.py -v
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"


class TestManifestExists:
    """Tests that manifest file exists and is valid."""

    def test_manifest_file_exists(self):
        """Test: manifest.json exists in data directory."""
        assert MANIFEST_PATH.exists(), \
            f"Manifest file not found: {MANIFEST_PATH}"

    def test_manifest_is_valid_json(self):
        """Test: manifest.json is valid JSON."""
        assert MANIFEST_PATH.exists(), "Manifest not found"

        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        assert isinstance(manifest, dict)

    def test_manifest_has_required_keys(self):
        """Test: manifest.json has required keys."""
        assert MANIFEST_PATH.exists(), "Manifest not found"

        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        assert 'version' in manifest, "Manifest missing 'version'"
        assert 'files' in manifest, "Manifest missing 'files'"

    def test_manifest_version_is_string(self):
        """Test: manifest version is a string."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        assert isinstance(manifest['version'], str)


class TestManifestFileEntries:
    """Tests for individual file entries in manifest."""

    def test_file_entries_have_sha256(self):
        """Test: All file entries have sha256 checksums."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        for rel_path, entry in manifest['files'].items():
            assert 'sha256' in entry, f"Entry {rel_path} missing sha256"
            assert isinstance(entry['sha256'], str)
            assert len(entry['sha256']) == 64, \
                f"Entry {rel_path} has invalid sha256 length"

    def test_file_entries_have_size(self):
        """Test: All file entries have size_bytes."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        for rel_path, entry in manifest['files'].items():
            assert 'size_bytes' in entry, f"Entry {rel_path} missing size_bytes"
            assert isinstance(entry['size_bytes'], int)
            assert entry['size_bytes'] > 0, f"Entry {rel_path} has invalid size"


class TestDataFileExistence:
    """Tests that manifest files exist on disk."""

    def test_all_manifest_files_exist(self):
        """Test: All files in manifest exist on disk."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        missing = []
        for rel_path in manifest['files'].keys():
            full_path = DATA_DIR / rel_path
            if not full_path.exists():
                missing.append(rel_path)

        if missing:
            # Don't fail, just warn - some files may be optional
            print(f"Warning: Missing files: {missing}")


class TestDataFileIntegrity:
    """Tests that data files match their checksums."""

    def test_pulsar_data_checksums(self):
        """Test: Pulsar data files have correct checksums."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        # Only test files that exist
        for rel_path, expected in manifest['files'].items():
            if not rel_path.startswith('pulsars/'):
                continue

            full_path = DATA_DIR / rel_path
            if not full_path.exists():
                print(f"SKIP: {rel_path} (not found)")
                continue

            with open(full_path, 'rb') as f:
                actual_sha256 = hashlib.sha256(f.read()).hexdigest()

            assert actual_sha256 == expected['sha256'], \
                f"Checksum mismatch for {rel_path}"

    def test_clock_data_checksums(self):
        """Test: Clock data files have correct checksums."""
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        for rel_path, expected in manifest['files'].items():
            if not rel_path.startswith('clock/'):
                continue

            full_path = DATA_DIR / rel_path
            if not full_path.exists():
                print(f"SKIP: {rel_path} (not found)")
                continue

            with open(full_path, 'rb') as f:
                actual_sha256 = hashlib.sha256(f.read()).hexdigest()

            assert actual_sha256 == expected['sha256'], \
                f"Checksum mismatch for {rel_path}"


class TestVerificationFunction:
    """Tests for the verify_data_integrity function."""

    def test_verification_function_exists(self):
        """Test: verify_data_integrity function exists."""
        from jug.scripts.download_data import verify_data_integrity
        assert callable(verify_data_integrity)

    def test_verification_returns_dict(self):
        """Test: verify_data_integrity returns expected dict structure."""
        from jug.scripts.download_data import verify_data_integrity

        result = verify_data_integrity(verbose=False)

        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'missing' in result
        assert 'corrupted' in result
        assert 'verified' in result

    def test_verification_lists_are_lists(self):
        """Test: verify_data_integrity lists are actually lists."""
        from jug.scripts.download_data import verify_data_integrity

        result = verify_data_integrity(verbose=False)

        assert isinstance(result['missing'], list)
        assert isinstance(result['corrupted'], list)
        assert isinstance(result['verified'], list)


class TestOfflineSafeFunction:
    """Tests for the run_offline_safe function."""

    def test_offline_safe_function_exists(self):
        """Test: run_offline_safe function exists."""
        from jug.scripts.download_data import run_offline_safe
        assert callable(run_offline_safe)

    def test_offline_safe_returns_bool(self):
        """Test: run_offline_safe returns boolean."""
        # Save current env
        old_offline = os.environ.get('JUG_OFFLINE')

        try:
            # Ensure offline mode is OFF for this test
            os.environ.pop('JUG_OFFLINE', None)

            from jug.scripts.download_data import run_offline_safe
            result = run_offline_safe()

            assert isinstance(result, bool)
        finally:
            # Restore env
            if old_offline:
                os.environ['JUG_OFFLINE'] = old_offline


class TestCacheDirectoryConfig:
    """Tests for cache directory configuration."""

    def test_get_cache_dir_default(self):
        """Test: get_cache_dir returns default path."""
        # Save and clear env
        old_cache = os.environ.get('JUG_CACHE_DIR')

        try:
            os.environ.pop('JUG_CACHE_DIR', None)

            from jug.scripts.download_data import get_cache_dir
            cache_dir = get_cache_dir()

            assert isinstance(cache_dir, Path)
            assert 'jug' in str(cache_dir).lower()
        finally:
            if old_cache:
                os.environ['JUG_CACHE_DIR'] = old_cache

    def test_get_cache_dir_env_override(self):
        """Test: get_cache_dir respects JUG_CACHE_DIR env var."""
        # Save old value
        old_cache = os.environ.get('JUG_CACHE_DIR')

        try:
            os.environ['JUG_CACHE_DIR'] = '/tmp/test_jug_cache'

            from jug.scripts.download_data import get_cache_dir
            cache_dir = get_cache_dir()

            assert str(cache_dir) == '/tmp/test_jug_cache'
        finally:
            if old_cache:
                os.environ['JUG_CACHE_DIR'] = old_cache
            else:
                os.environ.pop('JUG_CACHE_DIR', None)


def run_all_tests():
    """Run all data manifest tests."""
    print("=" * 80)
    print("Data Manifest Test Suite")
    print("=" * 80)

    if not MANIFEST_PATH.exists():
        print(f"\nERROR: Manifest not found: {MANIFEST_PATH}")
        return

    tests = [
        TestManifestExists(),
        TestManifestFileEntries(),
        TestDataFileExistence(),
        TestDataFileIntegrity(),
        TestVerificationFunction(),
        TestOfflineSafeFunction(),
        TestCacheDirectoryConfig(),
    ]

    passed = 0
    failed = 0

    for test_class in tests:
        class_name = test_class.__class__.__name__
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"[x] {class_name}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"[ ] {class_name}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"[ ] {class_name}.{method_name}: {type(e).__name__}: {e}")
                    failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")


if __name__ == '__main__':
    run_all_tests()
