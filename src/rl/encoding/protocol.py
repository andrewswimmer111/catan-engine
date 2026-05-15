"""Observation encoder protocol used by the env, agent, and trainer.

The trainer and runtime path are agnostic to which encoder is in use —
both :class:`FlatObservationEncoder` and :class:`GraphObservationEncoder`
satisfy this structural protocol. Adding a new encoder (e.g. a
heterogeneous-graph variant in a later phase) just requires conforming to
this shape; no call sites change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from domain.engine.player_view import PlayerView

__all__ = ["ObservationEncoder"]


@runtime_checkable
class ObservationEncoder(Protocol):
    """Encodes a :class:`PlayerView` as a flat float32 vector.

    Implementations expose ``out_shape`` (so the trainer can size buffers
    without instantiating) and ``layout_version`` (so checkpoints can
    refuse mismatched encoders at load time).
    """

    out_shape: tuple[int, ...]
    layout_version: int

    def encode(self, view: PlayerView) -> np.ndarray: ...
