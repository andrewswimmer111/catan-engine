"""Single source of truth for the ``--device`` CLI flag.

Both :mod:`scripts.train` (PPO driver) and :mod:`scripts.train_alphazero`
(az-007) expose the same ``--device {cpu, mps, cuda}`` flag, and
:func:`rl.cli._cmd_evaluate` mirrors it. Centralising the validation
here keeps the three entry points from drifting on "what does mps
unavailable mean" semantics.
"""

from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(spec: str) -> torch.device:
    """Validate ``spec`` against the actually-available backends.

    Fails fast (:class:`SystemExit`) if the user asked for a device the
    host can't provide, rather than silently falling back to CPU (which
    would surprise the user and silently halve the run's throughput
    target).
    """
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "error: --device cuda but torch.cuda.is_available()=False"
            )
        return torch.device("cuda")
    if spec == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not torch.backends.mps.is_available():
            raise SystemExit(
                "error: --device mps but torch MPS backend is unavailable "
                "(macOS + Metal-supporting GPU required)"
            )
        return torch.device("mps")
    if spec == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"error: --device must be one of cpu/mps/cuda, got {spec!r}")
