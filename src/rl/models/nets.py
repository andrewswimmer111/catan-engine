"""Shared init helpers for policy/value networks.

PPO is sensitive to initialization. Two conventions ship here:

- ``orthogonal_init`` — orthogonal weight init with a configurable gain. The
  standard choice for the trunk (gain=sqrt(2) ≈ 1.414, matching ReLU's gain)
  and the value head (gain=1.0). Bias is zeroed.
- The policy head uses ``gain=0.01`` to keep initial logits near zero, so the
  starting policy is nearly uniform over legal actions. This avoids early
  spurious confidence that PPO's clip ratio can't recover from.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["orthogonal_init", "TRUNK_GAIN", "VALUE_HEAD_GAIN", "POLICY_HEAD_GAIN"]

TRUNK_GAIN: float = math.sqrt(2.0)
VALUE_HEAD_GAIN: float = 1.0
POLICY_HEAD_GAIN: float = 0.01


def orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal weight init + zero bias. Returns ``layer`` for chaining."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def build_mlp_trunk(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Linear → Tanh stack with orthogonal init.

    Tanh activations (over ReLU) are standard in PPO implementations because
    they keep activations bounded and tend to give more stable value-function
    learning.
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        linear = orthogonal_init(nn.Linear(prev, h), gain=TRUNK_GAIN)
        layers.append(linear)
        layers.append(nn.Tanh())
        prev = h
    return nn.Sequential(*layers)
