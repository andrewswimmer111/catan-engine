"""Lazy TensorBoard wrapper.

The trainer logs hundreds of scalars per iteration; importing
``torch.utils.tensorboard`` eagerly drags in TensorBoard's protobuf stack
even for users who just want to play a game. We import inside the
constructor so callers who don't instantiate :class:`TBLogger` pay nothing.

If TensorBoard isn't installed, :class:`TBLogger` falls back to a no-op so
the trainer still runs end-to-end — useful in CI environments where TB
isn't available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["TBLogger", "NoOpLogger"]


class NoOpLogger:
    """Logger that silently discards every event."""

    def log_scalar(self, name: str, value: float, step: int) -> None:
        return

    def log_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        return

    def log_histogram(self, name: str, values: Any, step: int) -> None:
        return

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class TBLogger:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    Logs flatten dict prefixes into ``"prefix/key"`` scalar names — the
    convention TensorBoard's scalar UI groups by.
    """

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter  # local import on purpose

        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def log_scalars(self, prefix: str, values: dict[str, float], step: int) -> None:
        for k, v in values.items():
            self._writer.add_scalar(f"{prefix}/{k}", v, step)

    def log_histogram(self, name: str, values: Any, step: int) -> None:
        self._writer.add_histogram(name, values, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def make_logger(log_dir: str | Path | None) -> TBLogger | NoOpLogger:
    """Return a real TB logger if ``log_dir`` is provided and TB is available."""
    if log_dir is None:
        return NoOpLogger()
    try:
        return TBLogger(log_dir)
    except ImportError:
        return NoOpLogger()
