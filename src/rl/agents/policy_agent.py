"""Adapter wiring a policy/value network into both the engine and the trainer.

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
import torch.nn as nn
from torch.distributions import Categorical

from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.engine.player_view import make_player_view
from rl.acting_player import acting_player
from rl.agents.base import ActStep
from rl.agents.heuristic_agent import heuristic_discard
from rl.encoding.action import ActionEncoder, DiscardSentinel
from rl.encoding.observation import FlatObservationEncoder
from rl.encoding.protocol import ObservationEncoder

__all__ = ["PolicyAgent"]


class PolicyAgent:
    """Trainable policy that doubles as a play-time :class:`Agent`."""

    def __init__(
        self,
        model: nn.Module,
        action_encoder: ActionEncoder,
        obs_encoder: ObservationEncoder | None = None,
        device: str | torch.device = "cpu",
        *,
        stochastic_play: bool = False,
    ) -> None:
        self._device = torch.device(device)
        self._model = model.to(self._device)
        self._action_encoder = action_encoder
        self._obs_encoder: ObservationEncoder = (
            obs_encoder if obs_encoder is not None else FlatObservationEncoder()
        )
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

    def act_batch(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        generators: list[torch.Generator] | None = None,
        deterministic: bool = False,
    ) -> list[ActStep]:
        """Batched sample of actions for ``B`` parallel envs.

        ``obs`` is shape ``(B, obs_dim)`` and ``mask`` is ``(B, action_dim)``.
        Returns ``B`` :class:`ActStep` records in input order. The vec env
        path uses this to amortise a single GPU forward over ``num_envs``
        decisions.

        ``generators`` is a list of per-row :class:`torch.Generator`. When
        provided, each row is sampled with its own generator so the
        per-env trajectory is independent of how many other envs are in the
        batch — required for the determinism test that rolls a vec env
        against a sequential reference run.
        """
        if obs.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"act_batch requires 2-D obs/mask, got {obs.shape} / {mask.shape}"
            )
        if obs.shape[0] != mask.shape[0]:
            raise ValueError(
                f"obs/mask batch dims disagree: {obs.shape[0]} vs {mask.shape[0]}"
            )

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device)

        with torch.no_grad():
            out = self._model(obs_t, mask_t)
            probs = torch.softmax(out.logits, dim=-1)
            if deterministic:
                action_t = torch.argmax(out.logits, dim=-1)
            elif generators is None:
                action_t = Categorical(probs=probs).sample()
            else:
                if len(generators) != obs.shape[0]:
                    raise ValueError(
                        f"generators length {len(generators)} != batch size {obs.shape[0]}"
                    )
                # Per-row sampling so each env's action draw is deterministic
                # under its own generator regardless of batch order.
                action_idxs = [
                    int(torch.multinomial(probs[i], 1, generator=gen).item())
                    for i, gen in enumerate(generators)
                ]
                action_t = torch.tensor(action_idxs, device=self._device)
            log_probs = torch.log_softmax(out.logits, dim=-1)
            logp_t = log_probs.gather(-1, action_t.unsqueeze(-1)).squeeze(-1)
            entropy_t = -(probs * log_probs).sum(dim=-1)

        return [
            ActStep(
                action_idx=int(action_t[i].item()),
                logp=float(logp_t[i].item()),
                value=float(out.value[i].item()),
                entropy=float(entropy_t[i].item()),
            )
            for i in range(obs.shape[0])
        ]

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

        acting = acting_player(snap.state)
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
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def action_encoder(self) -> ActionEncoder:
        return self._action_encoder

    @property
    def obs_encoder(self) -> ObservationEncoder:
        return self._obs_encoder

    def state_dict(self) -> dict[str, Any]:
        return self._model.state_dict()

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._model.load_state_dict(sd)
