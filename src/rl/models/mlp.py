"""MLP policy/value network with action masking.

The network is a shared trunk producing a hidden representation, with two
independent heads: a policy head over the full discrete action space and a
scalar value head. ``forward`` applies the boolean action mask by setting
illegal-action logits to ``-1e9`` *before* softmax — the entire policy
probability sits on legal actions and gradients flow only through legal logits.

This module is intentionally a thin shell; rollout sampling and log-prob
computation live in :class:`rl.agents.policy_agent.PolicyAgent` and the PPO
update (rl-013) so the model itself stays a pure tensor function.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rl.models.nets import (
    POLICY_HEAD_GAIN,
    VALUE_HEAD_GAIN,
    build_mlp_trunk,
    orthogonal_init,
)

__all__ = ["MLPPolicyValue", "ModelOutput", "MASK_FILL_VALUE"]

MASK_FILL_VALUE: float = -1e9


@dataclass
class ModelOutput:
    """Masked logits and state value for one forward pass.

    ``logits`` has the same shape as the input mask; entries where ``mask``
    is False are set to :data:`MASK_FILL_VALUE`. ``value`` is ``(B,)`` for
    a scalar-value model (the default for :class:`MLPPolicyValue`) or
    ``(B, N_PLAYERS)`` for a per-seat vector-value model (see
    :class:`rl.models.gnn.GNNPolicyValue` with ``value_kind="vector"``).
    """

    logits: torch.Tensor
    value: torch.Tensor


class MLPPolicyValue(nn.Module):
    """Shared-trunk policy/value network operating on flat observations."""

    # Static: the flat-encoder MLP is scalar-value-only. The vector value
    # head is a Phase-3 (AlphaZero) feature on the GNN encoder.
    value_kind: str = "scalar"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (512, 512, 512),
    ) -> None:
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one layer width")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = tuple(hidden)

        self.trunk = build_mlp_trunk(obs_dim, self.hidden)
        self.policy_head = orthogonal_init(
            nn.Linear(self.hidden[-1], action_dim), gain=POLICY_HEAD_GAIN
        )
        self.value_head = orthogonal_init(
            nn.Linear(self.hidden[-1], 1), gain=VALUE_HEAD_GAIN
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        """Compute masked logits and value for a batch of observations.

        ``obs`` is ``(B, obs_dim)`` float; ``mask`` is ``(B, action_dim)``
        boolean. The returned logits already have illegal entries set to
        ``-1e9``; callers should pass them straight to ``Categorical`` or
        ``log_softmax`` without re-masking.
        """
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"obs shape {tuple(obs.shape)} incompatible with obs_dim={self.obs_dim}"
            )
        if mask.shape != (obs.size(0), self.action_dim):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} incompatible with "
                f"(batch={obs.size(0)}, action_dim={self.action_dim})"
            )

        h = self.trunk(obs)
        raw_logits = self.policy_head(h)
        logits = raw_logits.masked_fill(~mask.bool(), MASK_FILL_VALUE)
        value = self.value_head(h).squeeze(-1)
        return ModelOutput(logits=logits, value=value)
