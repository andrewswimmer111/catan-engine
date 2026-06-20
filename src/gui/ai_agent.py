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

__all__ = ["make_ai_agent"]


def make_ai_agent(checkpoint_path: Path, *, device: str = "cpu") -> Agent:
    agent, _meta = load_checkpoint(Path(checkpoint_path), device=device)
    agent.stochastic_play = False
    return agent
