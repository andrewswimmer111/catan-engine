"""GUI-side factory for the trained-checkpoint opponent.

Lives in :mod:`gui` rather than :mod:`controller` to keep RL imports out of
``controller``. The returned object satisfies the ``controller.agents.Agent``
protocol so the orchestrator wires it in unchanged.

Greedy by default: ``PolicyAgent.stochastic_play=False`` makes ``choose``
take ``argmax`` over the masked policy — the canonical eval behaviour.
"""

from __future__ import annotations

from pathlib import Path

from controller.agents import Agent
from rl.training.checkpoint import load_checkpoint

__all__ = ["make_ai_agent", "find_latest_checkpoint"]


def make_ai_agent(checkpoint_path: Path, *, device: str = "cpu") -> Agent:
    agent, _meta = load_checkpoint(Path(checkpoint_path), device=device)
    agent.stochastic_play = False
    return agent


def find_latest_checkpoint(runs_dir: Path) -> Path | None:
    """Return the most recently modified ``*.pt`` under ``runs_dir`` or None.

    Recursive — checkpoints live one or two levels deep (``runs/<run>/final.pt``,
    ``runs/<run>/checkpoints/iter_*.pt``). mtime is preferred over name parsing
    because run-dir naming conventions have drifted across the project.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return None
    candidates = list(runs_dir.rglob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
