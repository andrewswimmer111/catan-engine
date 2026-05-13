"""PPO update — clipped surrogate loss, masked log-prob recomputation, GAE returns.

Single entry point: :func:`ppo_update`. Given a full-rollout
:class:`TrajectoryBatch` (already-finalised by ``compute_advantages``), it
runs ``cfg.n_epochs`` of SGD over shuffled minibatches of size
``cfg.minibatch_size``, with KL-based early stopping between epochs and
gradient-norm clipping per step.

Two implementation notes worth flagging:

1. **Masked log-prob recomputation.** The mask used at action-selection
   time is stored alongside the action; the update path passes the *same*
   mask back through the model so old- and new-policy log-probs are
   directly comparable. Without this, an action that became "illegal"
   between collection and update would have an undefined new log-prob and
   the clip ratio would be meaningless.
2. **Advantage normalisation.** Per minibatch ``(adv - mean) / (std + 1e-8)``
   — standard CleanRL trick. Normalising over the full rollout is also
   common but per-minibatch keeps the gradient magnitude stable when
   minibatches differ in variance.
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.models.mlp import MLPPolicyValue
from rl.replay.buffer import TrajectoryBatch
from rl.training.config import PPOConfig

__all__ = ["ppo_update"]


_DeviceLike = Union[str, torch.device]


def ppo_update(
    model: MLPPolicyValue,
    optimizer: torch.optim.Optimizer,
    batch: TrajectoryBatch,
    cfg: PPOConfig,
    device: _DeviceLike = "cpu",
) -> dict[str, float]:
    """Run a full PPO update — ``cfg.n_epochs`` × minibatching — on ``batch``.

    Returns a metrics dict averaged over all minibatch steps actually
    executed. If KL early-stop fires, the metrics reflect only the work
    done before the stop. ``n_updates`` and ``early_stopped`` let the
    caller see whether the full schedule ran.
    """
    if cfg.minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be positive, got {cfg.minibatch_size}")
    if cfg.n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {cfg.n_epochs}")

    dev = torch.device(device)
    obs = batch.obs.to(dev)
    actions = batch.actions.to(dev)
    masks = batch.masks.to(dev)
    old_logps = batch.old_logps.to(dev)
    advantages_full = batch.advantages.to(dev)
    returns = batch.returns.to(dev)

    N = obs.shape[0]
    if N == 0:
        return _zero_metrics(cfg)

    metrics_sum = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    n_updates = 0
    early_stopped = False

    for _epoch in range(cfg.n_epochs):
        perm = torch.randperm(N, device=dev)
        epoch_kls: list[float] = []

        for start in range(0, N, cfg.minibatch_size):
            mb_idx = perm[start : start + cfg.minibatch_size]
            step_metrics = _one_minibatch_step(
                model=model,
                optimizer=optimizer,
                obs=obs[mb_idx],
                actions=actions[mb_idx],
                masks=masks[mb_idx],
                old_logps=old_logps[mb_idx],
                advantages=advantages_full[mb_idx],
                returns=returns[mb_idx],
                cfg=cfg,
            )
            for k, v in step_metrics.items():
                metrics_sum[k] += v
            n_updates += 1
            epoch_kls.append(step_metrics["approx_kl"])

        if cfg.target_kl is not None and epoch_kls:
            mean_epoch_kl = sum(epoch_kls) / len(epoch_kls)
            if mean_epoch_kl > cfg.target_kl:
                early_stopped = True
                break

    out = {k: (v / n_updates if n_updates else 0.0) for k, v in metrics_sum.items()}
    out["n_updates"] = float(n_updates)
    out["early_stopped"] = 1.0 if early_stopped else 0.0
    out["lr"] = _current_lr(optimizer)
    return out


def _one_minibatch_step(
    model: MLPPolicyValue,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    """One PPO gradient step over a single minibatch."""
    # Per-minibatch advantage normalisation. Single-element batches keep std=0,
    # which would divide-by-epsilon and inflate gradients — guard against that.
    adv = advantages
    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    out = model(obs, masks)
    dist = Categorical(logits=out.logits)
    new_logps = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    log_ratio = new_logps - old_logps
    ratio = torch.exp(log_ratio)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = 0.5 * (out.value - returns).pow(2).mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    optimizer.step()

    # Schulman's "approximate KL" — unbiased and always non-negative.
    # See http://joschu.net/blog/kl-approx.html.
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
        "grad_norm": float(grad_norm.item()),
    }


def _zero_metrics(cfg: PPOConfig) -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
        "n_updates": 0.0,
        "early_stopped": 0.0,
        "lr": cfg.lr,
    }


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Pull the LR from the first param group (the only group we use)."""
    return float(optimizer.param_groups[0]["lr"])
