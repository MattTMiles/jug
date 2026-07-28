"""The phase gauge: choice of representative for a residual vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

GaugeMode = Literal["none", "mean", "constant"]

_LEGAL_MODES = ("none", "mean", "constant")


@dataclass(frozen=True)
class ReferenceGauge:
    """Boundary-only reconstruction descriptor for one reference residual vector.

    Carries the *data* needed to reproduce the gauge, not just its name: a
    bare string cannot reconstruct a weighted mean, because the normalized
    TOA weights are an (n_toa,) array. ``weights`` is None for the unweighted
    mean and for ``mode`` in {"none", "constant"}.
    """

    mode: GaugeMode
    weights: np.ndarray | None = None  # normalized, sums to 1
    offset_sec: float | None = None  # only for mode="constant"

    def __post_init__(self) -> None:
        """Canonicalize on the host, so every construction path is frozen alike.

        `frozen=True` protects the field binding, not the array it points at,
        so copy the weights and mark them read-only here (D19 policy) rather
        than relying on each construction site to remember.

        This makes ``ReferenceGauge`` **host-only by construction**: the copy
        would raise on a JAX tracer. That is deliberate — traced code calls
        ``_gauge_offset_values`` directly and never builds a descriptor.
        """
        if self.weights is not None:
            w = np.array(self.weights, dtype=np.float64, copy=True)
            w.setflags(write=False)
            object.__setattr__(self, "weights", w)


def _gauge_offset_values(residual_sec, *, mode, weights=None, offset_sec=None, xp):
    """Tracer-safe kernel: the scalar c, from raw values rather than a descriptor.

    The ONLY implementation of the gauge arithmetic. Performs no validation and
    no host-side array canonicalization, so it is safe to call with JAX tracers.
    Private: callers outside this module use the public entry points below,
    which validate first.
    """
    if mode == "none":
        return xp.asarray(0.0, dtype=xp.float64)
    if mode == "constant":
        return xp.asarray(offset_sec, dtype=xp.float64)
    # mode == "mean"
    r = xp.asarray(residual_sec)
    if weights is None:
        return xp.mean(r)
    w = xp.asarray(weights)
    return xp.sum(r * w) / xp.sum(w)


def _validate_gauge(residual_sec, gauge: ReferenceGauge) -> None:
    """Validate ``gauge`` against ``residual_sec`` in the four locked stages."""
    mode = gauge.mode
    weights = gauge.weights
    offset_sec = gauge.offset_sec
    residual = np.asarray(residual_sec)

    # Stage 1 — mode validity
    if mode not in _LEGAL_MODES:
        raise ValueError(f"Unknown gauge mode {mode!r}; expected one of " f"{list(_LEGAL_MODES)}")

    # Stage 2 — payload compatibility (independent of array contents)
    if mode == "constant":
        if offset_sec is None:
            raise ValueError("ReferenceGauge(mode='constant') requires offset_sec")
        if weights is not None:
            raise ValueError(
                "ReferenceGauge(mode='constant') must not carry weights "
                "(a constant anchor takes no weights)"
            )
    elif mode == "mean":
        if offset_sec is not None:
            raise ValueError("ReferenceGauge(mode='mean') must not carry offset_sec")
    elif mode == "none":
        if weights is not None or offset_sec is not None:
            raise ValueError("ReferenceGauge(mode='none') must not carry weights or offset_sec")

    if weights is not None:
        w = np.asarray(weights)
        if w.shape != residual.shape:
            raise ValueError(
                f"gauge weights shape {w.shape} does not match " f"residual shape {residual.shape}"
            )

    # Stage 3 — empty short-circuit (caller handles return; no Stage 4)
    n = int(residual.size)
    if n == 0:
        return

    # Stage 4 — weight contents (only when n > 0 and weights is not None)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if not np.all(np.isfinite(w)):
            raise ValueError("gauge weights must be finite")
        if np.any(w < 0.0):
            raise ValueError(
                "gauge weights must be non-negative "
                "(1/sigma^2 weights are positive by construction)"
            )
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum == 0.0:
            raise ValueError("gauge weights must have a positive finite sum")


def gauge_offset_sec(residual_sec, gauge: ReferenceGauge, *, xp=np):
    """Return the scalar c such that the gauged residual is r - c.

    Validates ``gauge`` (host-side descriptor) and delegates the arithmetic to
    ``_gauge_offset_values``. ``mode="mean"`` with ``weights=None`` is the
    unweighted mean (tempo2 ``REFPHS MEAN``); with weights it is the weighted
    mean (PINT family). ``mode="constant"`` returns ``gauge.offset_sec``
    (tempo2 ``REFPHS TZR``).
    """
    residual = np.asarray(residual_sec)
    _validate_gauge(residual, gauge)

    n = int(residual.size)
    if n == 0 and gauge.mode in ("none", "mean"):
        return 0.0
    if n == 0 and gauge.mode == "constant":
        return float(gauge.offset_sec)

    offset = _gauge_offset_values(
        residual,
        mode=gauge.mode,
        weights=gauge.weights,
        offset_sec=gauge.offset_sec,
        xp=xp,
    )
    # Host API: always a Python float (traced call sites use the kernel).
    return float(np.asarray(offset))


def apply_phase_gauge(residual_sec, gauge: ReferenceGauge, *, xp=np):
    """Return ``residual_sec - gauge_offset_sec(residual_sec, gauge)``.

    ``mode="none"`` returns the input unchanged with no copy; callers own copies.
    """
    residual = np.asarray(residual_sec)
    _validate_gauge(residual, gauge)

    if gauge.mode == "none":
        return residual_sec

    n = int(residual.size)
    if n == 0:
        return residual_sec

    offset = _gauge_offset_values(
        residual,
        mode=gauge.mode,
        weights=gauge.weights,
        offset_sec=gauge.offset_sec,
        xp=xp,
    )
    return xp.asarray(residual_sec) - offset


def reconstruct_absolute_residuals(
    reference_sec, residual_delta_sec, gauge: ReferenceGauge, *, xp=np
):
    """r_theta in the reference's gauge, per the absolute-residual recipe.

    The ONLY supported way to combine a gauged reference with a gauge-free
    delta anywhere in the stack. Applies the delta raw for "none",
    mean-projected for "mean", and raises NotImplementedError for "constant".
    """
    reference = np.asarray(reference_sec)
    delta = np.asarray(residual_delta_sec)
    _validate_gauge(reference, gauge)
    if delta.shape != reference.shape:
        raise ValueError(
            f"residual_delta shape {delta.shape} does not match "
            f"reference shape {reference.shape}"
        )

    if gauge.mode == "constant":
        raise NotImplementedError(
            "reconstruct_absolute_residuals does not support mode='constant' "
            "(tempo2 REFPHS TZR): that anchor depends on the spin parameters, "
            "so reproducing a theta-dependent anchor requires machinery the "
            "frozen model does not carry."
        )

    ref_xp = xp.asarray(reference_sec)
    delta_xp = xp.asarray(residual_delta_sec)

    if gauge.mode == "none":
        return ref_xp + delta_xp

    # mode == "mean": r_ref^g + P(delta)
    projected = apply_phase_gauge(
        residual_delta_sec,
        ReferenceGauge(mode="mean", weights=gauge.weights),
        xp=xp,
    )
    return ref_xp + xp.asarray(projected)
