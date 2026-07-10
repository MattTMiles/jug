"""Tempo2 dev-oracle parity tests (not shipped in pint-only portable build)."""

import pytest

pytestmark = pytest.mark.dev_oracle


def test_dev_oracle_wsrt167_skipped_in_pint_only_build():
    pytest.skip("tempo2 dev-oracle harness not available in pint-only build")