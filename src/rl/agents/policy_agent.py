"""Adapter wiring :class:`MLPPolicyValue` into both the engine and the trainer.

A :class:`PolicyAgent` exposes two entry points:

- :meth:`act` (training path): consumes the pre-encoded ``(obs, mask)`` the
  env emits and returns an :class:`ActStep` carrying the action index, the
  selected action's log-probability under the policy, the value estimate,
  and the policy entropy. PPO needs all four.
- :meth:`choose` (engine path): satisfies the existing ``Agent`` protocol so
  the agent slots into the tournament harness and the GUI without shims. It
  re-encodes the snapshot, runs ``act`` in deterministic mode, and decodes
  the integer back into a typed :class:`Action`. Discards are resolved with
  :func:`heuristic_discard` mirroring the env's own behaviour.

The model lives on a configurable device; the agent moves single-row inputs
to that device, runs inference under ``torch.no_grad()``, and returns plain
Python scalars so callers don't drag tensors out of this boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.engine.player_view import make_player_view
from domain.enums import TurnPhase
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending
from rl.agents.base import ActStep
from rl.agents.heuristic_agent import heuristic_discard
from rl.encoding.action import ActionEncoder, DiscardSentinel
from rl.encoding.observation import FlatObservationEncoder
from rl.models.mlp import MLPPolicyValue

__all__ = ["PolicyAgent"]


def _acting_player(state: GameState) -> PlayerID:
    """Same convention as ``CatanEnv.current_agent`` — picks the discard owner
    during DISCARD phase, otherwise the dice-roller / current_player.
    """
    if state.phase is TurnPhase.DISCARD and isinstance(state.pending, DiscardPending):
        return next(iter(state.pending.cards_to_discard))
    return state.current_player


class PolicyAgent:
    """Trainable policy that doubles as a play-time :class:`Agent`."""

    def __init__(
        self,
        model: MLPPolicyValue,
        action_encoder: ActionEncoder,
        obs_encoder: FlatObservationEncoder | None = None,
        device: str | torch.device = "cpu",
        *,
        stochastic_play: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._action_encoder = action_encoder
        self._obs_encoder = obs_encoder or FlatObservationEncoder()
        # ``choose`` defaults to deterministic argmax (the canonical eval
        # policy). Opponent-inference siblings flip this to sample from the
        # masked categorical so self-play opponents stay exploration-friendly
        # rather than collapsing onto a greedy strategy that's easy to game.
        self.stochastic_play = stochastic_play

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> ActStep:
        """Sample an action index from the masked policy.

        ``obs`` is a 1-D float array; ``mask`` is a 1-D boolean array of
        length ``action_dim``. Returns an :class:`ActStep` whose fields are
        all Python scalars. Inference runs under ``torch.no_grad`` — gradient
        flow during PPO updates happens through a fresh forward pass on the
        stored batch, not through these per-step calls.
        """
        if not mask.any():
            raise ValueError("action mask is all False; no legal action to sample")

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device).unsqueeze(0)

        with torch.no_grad():
            out = self._model(obs_t, mask_t)
            dist = Categorical(logits=out.logits)
            if deterministic:
                action_t = torch.argmax(out.logits, dim=-1)
            else:
                action_t = dist.sample()
            logp_t = dist.log_prob(action_t)
            entropy_t = dist.entropy()

        return ActStep(
            action_idx=int(action_t.item()),
            logp=float(logp_t.item()),
            value=float(out.value.item()),
            entropy=float(entropy_t.item()),
        )

    def act_with_dist(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[ActStep, np.ndarray]:
        """Like :meth:`act` but also returns the full masked softmax dist.

        Used by the replay archiver — illegal actions stay at ~0 (the mask
        fill sends their logits to -1e9 before softmax), so the returned
        vector is safe to feed straight into a top-K display.
        """
        if not mask.any():
            raise ValueError("action mask is all False; no legal action to sample")

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device).unsqueeze(0)

        with torch.no_grad():
            out = self._model(obs_t, mask_t)
            probs = torch.softmax(out.logits, dim=-1)
            dist = Categorical(probs=probs)
            if deterministic:
                action_t = torch.argmax(out.logits, dim=-1)
            else:
                action_t = dist.sample()
            logp_t = dist.log_prob(action_t)
            entropy_t = dist.entropy()

        action_dist = probs.squeeze(0).cpu().numpy()
        return (
            ActStep(
                action_idx=int(action_t.item()),
                logp=float(logp_t.item()),
                value=float(out.value.item()),
                entropy=float(entropy_t.item()),
            ),
            action_dist,
        )

    # ------------------------------------------------------------------
    # Engine / play path
    # ------------------------------------------------------------------

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        """Pick a typed action for the engine.

        Re-encodes the snapshot from the acting player's perspective, runs
        :meth:`act` deterministically, decodes the integer back to an
        :class:`Action`, and resolves discards via the same heuristic the
        env uses. If no legal action is representable in the discrete head
        (e.g. only domestic-trade proposals remain), falls back to ``legal[0]``
        so the game doesn't stall.
        """
        if not legal:
            return None

        mask = self._action_encoder.mask(legal)
        if not mask.any():
            return legal[0]

        acting = _acting_player(snap.state)
        view = make_player_view(snap.state, acting)
        obs = self._obs_encoder.encode(view)

        step = self.act(obs, mask, deterministic=not self.stochastic_play)
        decoded = self._action_encoder.decode(step.action_idx, snap.state)
        if isinstance(decoded, DiscardSentinel):
            return heuristic_discard(snap.state, decoded.player_id)
        return decoded

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def model(self) -> MLPPolicyValue:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def action_encoder(self) -> ActionEncoder:
        return self._action_encoder

    @property
    def obs_encoder(self) -> FlatObservationEncoder:
        return self._obs_encoder

    def state_dict(self) -> dict[str, Any]:
        return self._model.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._model.load_state_dict(sd)
