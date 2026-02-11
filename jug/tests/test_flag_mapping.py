"""Tests for jug.engine.flag_mapping — flag aliasing + backend resolution."""

import pytest

from jug.engine.flag_mapping import (
    FlagMappingConfig,
    resolve_backend_for_toa,
    resolve_backends,
    resolve_flag_for_toa,
)


# ---------------------------------------------------------------------------
# FlagMappingConfig defaults
# ---------------------------------------------------------------------------

class TestFlagMappingConfig:
    def test_default_candidates(self):
        cfg = FlagMappingConfig()
        assert cfg.candidates == ["f", "be", "backend"]

    def test_default_fallback(self):
        cfg = FlagMappingConfig()
        assert cfg.fallback == "__unknown__"


# ---------------------------------------------------------------------------
# resolve_backend_for_toa
# ---------------------------------------------------------------------------

class TestResolveBackendForToa:
    def test_first_candidate_wins(self):
        flags = {"f": "L-wide", "be": "GUPPI"}
        result = resolve_backend_for_toa(flags)
        # "f" is first candidate in default config
        assert result == "L-wide"

    def test_second_candidate_if_first_missing(self):
        flags = {"be": "GUPPI"}
        result = resolve_backend_for_toa(flags)
        assert result == "GUPPI"

    def test_third_candidate(self):
        flags = {"backend": "PUPPI"}
        result = resolve_backend_for_toa(flags)
        assert result == "PUPPI"

    def test_fallback_when_none_match(self):
        flags = {"something_else": "value"}
        result = resolve_backend_for_toa(flags)
        assert result == "__unknown__"

    def test_empty_flags(self):
        result = resolve_backend_for_toa({})
        assert result == "__unknown__"

    def test_custom_config(self):
        cfg = FlagMappingConfig(
            candidates=["sys"],
            aliases={},
            fallback="NONE",
        )
        flags = {"sys": "PUPPI"}
        assert resolve_backend_for_toa(flags, cfg) == "PUPPI"

    def test_alias_applied(self):
        cfg = FlagMappingConfig(
            candidates=["be"],
            aliases={"KAT": "MKBF"},
        )
        flags = {"be": "KAT"}
        assert resolve_backend_for_toa(flags, cfg) == "MKBF"

    def test_alias_passthrough_for_unknown_value(self):
        """Values not in aliases are returned as-is."""
        cfg = FlagMappingConfig(
            candidates=["be"],
            aliases={"KAT": "MKBF"},
        )
        flags = {"be": "GUPPI"}
        assert resolve_backend_for_toa(flags, cfg) == "GUPPI"


# ---------------------------------------------------------------------------
# resolve_backends (batch)
# ---------------------------------------------------------------------------

class TestResolveBackends:
    def test_batch_resolution(self):
        toa_list = [
            {"f": "L-wide", "be": "GUPPI"},
            {"be": "PUPPI"},
            {},
        ]
        result = resolve_backends(toa_list)
        assert result == ["L-wide", "PUPPI", "__unknown__"]

    def test_batch_with_config(self):
        cfg = FlagMappingConfig(
            candidates=["be"],
            aliases={"KAT": "MKBF"},
        )
        toa_list = [{"be": "KAT"}, {"be": "GUPPI"}]
        result = resolve_backends(toa_list, cfg)
        assert result == ["MKBF", "GUPPI"]


# ---------------------------------------------------------------------------
# resolve_flag_for_toa (generic)
# ---------------------------------------------------------------------------

class TestResolveFlagForToa:
    def test_generic_resolution(self):
        flags = {"fe": "L-wide"}
        result = resolve_flag_for_toa(flags, ["fe", "frontend"])
        assert result == "L-wide"

    def test_generic_with_alias(self):
        flags = {"fe": "KAT"}
        result = resolve_flag_for_toa(
            flags, ["fe"], aliases={"KAT": "L-band"}
        )
        assert result == "L-band"

    def test_generic_fallback(self):
        result = resolve_flag_for_toa({}, ["fe"], fallback="NONE")
        assert result == "NONE"


# ---------------------------------------------------------------------------
# Mixed inputs (like real TIM files)
# ---------------------------------------------------------------------------

class TestMixedInputs:
    def test_meerkat_style_flags(self):
        """MPTC/MeerKAT TIM files use -fe and -be flags."""
        flags = {"fe": "KAT", "be": "MKBF"}
        # Default config tries "f" first, then "be"
        result = resolve_backend_for_toa(flags)
        assert result == "MKBF"  # "be" is second candidate

    def test_nanograv_style_flags(self):
        """NANOGrav TIM files use -f flag."""
        flags = {"f": "L-wide.PUPPI"}
        result = resolve_backend_for_toa(flags)
        assert result == "L-wide.PUPPI"

    def test_mixed_heterogeneous_dataset(self):
        """Different styles in the same dataset."""
        toa_list = [
            {"f": "L-wide.PUPPI"},          # NANOGrav style
            {"fe": "KAT", "be": "MKBF"},    # MeerKAT style
            {"backend": "RCVR_800"},         # generic
            {},                              # no flags
        ]
        result = resolve_backends(toa_list)
        assert result == ["L-wide.PUPPI", "MKBF", "RCVR_800", "__unknown__"]
