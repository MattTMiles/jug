"""Shared CLI arguments for JUG timing compatibility and tempo2 graph mode."""

from __future__ import annotations

import argparse

from jug.residuals.tempo2.graph_config import (
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
    TEMPO2_GRAPH_FULL,
    TEMPO2_GRAPH_MODE_DEFAULT,
    TEMPO2_GRAPH_STAGED_BCLT,
)
from jug.timing import validate_tempo2_graph_mode

DEFAULT_COMPATIBILITY = "pint"
DEFAULT_TEMPO2_NATIVE = TEMPO2_GRAPH_MODE_DEFAULT

COMPATIBILITY_CHOICES = ("pint", "tempo2")
TEMPO2_NATIVE_CHOICES = (
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_STAGED_BCLT,
    TEMPO2_GRAPH_FULL,
)


def add_timing_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_tempo2_native: bool = True,
) -> None:
    """Register ``--compatibility`` / ``-c`` and optional ``--tempo2-native``."""
    parser.add_argument(
        "-c",
        "--compatibility",
        choices=COMPATIBILITY_CHOICES,
        default=DEFAULT_COMPATIBILITY,
        help=(
            "Timing compatibility mode: pint (default) or tempo2 "
            "(tempo2/libstempo conventions)"
        ),
    )
    if include_tempo2_native:
        parser.add_argument(
            "--tempo2-native",
            "--tempo2_native",
            dest="tempo2_native",
            choices=TEMPO2_NATIVE_CHOICES,
            default=DEFAULT_TEMPO2_NATIVE,
            help=(
                "Tempo2-native JAX graph mode when --compatibility tempo2 "
                f"(default: {DEFAULT_TEMPO2_NATIVE})"
            ),
        )


def timing_kwargs_from_namespace(args: argparse.Namespace) -> dict[str, str | None]:
    """Extract ``compatibility`` and ``tempo2_native`` kwargs from parsed CLI args."""
    compatibility = str(getattr(args, "compatibility", DEFAULT_COMPATIBILITY)).lower()
    tempo2_native = getattr(args, "tempo2_native", None)
    if compatibility != "tempo2":
        return {"compatibility": compatibility, "tempo2_native": None}
    if tempo2_native is None:
        tempo2_native = DEFAULT_TEMPO2_NATIVE
    return {
        "compatibility": compatibility,
        "tempo2_native": validate_tempo2_graph_mode(str(tempo2_native)),
    }
