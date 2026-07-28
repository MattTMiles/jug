"""Tests for shared timing CLI argument helpers."""

from __future__ import annotations

import argparse

from jug.residuals.tempo2.graph_config import TEMPO2_GRAPH_FIXED_STATE_STRIPPED
from jug.timing.cli import (
    DEFAULT_COMPATIBILITY,
    add_timing_cli_arguments,
    timing_kwargs_from_namespace,
)


def test_timing_cli_defaults():
    parser = argparse.ArgumentParser()
    add_timing_cli_arguments(parser)
    args = parser.parse_args([])
    kwargs = timing_kwargs_from_namespace(args)
    assert kwargs == {
        "compatibility": DEFAULT_COMPATIBILITY,
        "tempo2_native": None,
    }


def test_timing_cli_tempo2_mode():
    parser = argparse.ArgumentParser()
    add_timing_cli_arguments(parser)
    args = parser.parse_args(
        ["--compatibility", "tempo2", "--tempo2-native", "staged_bclt"]
    )
    kwargs = timing_kwargs_from_namespace(args)
    assert kwargs == {
        "compatibility": "tempo2",
        "tempo2_native": "staged_bclt",
    }


def test_timing_cli_tempo2_default_graph_mode():
    parser = argparse.ArgumentParser()
    add_timing_cli_arguments(parser)
    args = parser.parse_args(["-c", "tempo2"])
    kwargs = timing_kwargs_from_namespace(args)
    assert kwargs["compatibility"] == "tempo2"
    assert kwargs["tempo2_native"] == TEMPO2_GRAPH_FIXED_STATE_STRIPPED
