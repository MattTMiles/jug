"""Tempo2 native-chain runtime evidence (not shipped in pint-only portable build)."""

import pytest

pytestmark = pytest.mark.dev_oracle


def test_phase2_runtime_evidence_skipped_in_pint_only_build():
    pytest.skip("tempo2 native-chain evidence tests not available in pint-only build")