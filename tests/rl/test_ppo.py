"""Tests for the PPO update step (rl-013).

Three contracts:

* One PPO update on a synthetic batch finishes without errors and produces
  finite, well-formed metrics. We also check that the policy loss is
  finite and that the value head improves on a degenerate (constant
  returns) signal.
* When ``target_kl`` is set artificially low, KL early-stop fires:
  ``early_stopped == 1.0`` and ``n_updates < n_epochs × n_minibatches``.
* Gradient-norm clipping enforces ``max_grad_norm`` even when the synthetic
  inputs would otherwise produce a much larger norm.
"""

from __future__ import annotations

import torch

from rl.encoding.action import ACTION_SPACE_SIZE
from rl.encoding.observation import OBS_SHAPE
from rl.models.mlp import MLPPolicyValue
from rl.replay.buffer import TrajectoryBatch
from rl.training.config import PPOConfig
from rl.training.ppo import ppo_update


def _make_synthetic_batch(
    n: int = 128,
    obs_dim: int = OBS_SHAPE[0],
    action_dim: int = ACTION_SPACE_SIZE,
    seed: int = 0,
) -> tuple[TrajectoryBatch, MLPPolicyValue]:
    """Build a model, sample old logps from it, and produce a batch.

    Old logps come from the model itself so ratios start at exactly 1.0 —
    this makes early KLs near zero and gives the test signal a clean
    baseline to deviate from.
    """
    torch.manual_seed(seed)
    model = MLPPolicyValue(obs_dim, action_dim, hidden=(32, 32))

    obs = torch.randn(n, obs_dim)
    masks = torch.zeros(n, action_dim, dtype=torch.bool)
    # 16 legal actions per sample, randomly chosen.
    for i in range(n):
        legal_idx = torch.randperm(action_dim)[:16]
        masks[i, legal_idx] = True

    with torch.no_grad():
        out = model(obs, masks)
        from torch.distributions import Categorical
        dist = Categorical(logits=out.logits)
        actions = dist.sample()
        old_logps = dist.log_prob(actions)
        old_values = out.value

    advantages = torch.randn(n)
    returns = old_values + advantages  # mimic GAE returns layout

    batch = TrajectoryBatch(
        obs=obs,
        actions=actions,
        masks=masks,
        old_logps=old_logps,
        advantages=advantages,
        returns=returns,
    )
    return batch, model


def test_ppo_update_runs_and_returns_finite_metrics() -> None:
    batch, model = _make_synthetic_batch(n=128, seed=1)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    cfg = PPOConfig(n_epochs=2, minibatch_size=32, target_kl=None)

    metrics = ppo_update(model, optim, batch, cfg)

    for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                "clip_fraction", "grad_norm"):
        v = metrics[key]
        assert v == v, f"{key} is NaN"   # NaN check
        assert v != float("inf") and v != float("-inf"), f"{key} is inf"

    expected_updates = cfg.n_epochs * ((128 + cfg.minibatch_size - 1) // cfg.minibatch_size)
    assert metrics["n_updates"] == expected_updates
    assert metrics["early_stopped"] == 0.0


def test_ppo_update_reduces_value_loss_when_returns_are_constant() -> None:
    """Value head should learn a constant target — a basic learning sanity check."""
    batch, model = _make_synthetic_batch(n=128, seed=2)
    # Replace returns with a constant target; the value head should regress to it.
    target = 1.5
    batch = TrajectoryBatch(
        obs=batch.obs,
        actions=batch.actions,
        masks=batch.masks,
        old_logps=batch.old_logps,
        advantages=torch.zeros_like(batch.advantages),  # no policy gradient pressure
        returns=torch.full_like(batch.returns, target),
    )

    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = PPOConfig(n_epochs=10, minibatch_size=32, target_kl=None, entropy_coef=0.0)

    with torch.no_grad():
        v0 = model(batch.obs, batch.masks).value.mean().item()
    initial_dist = abs(v0 - target)

    ppo_update(model, optim, batch, cfg)

    with torch.no_grad():
        v1 = model(batch.obs, batch.masks).value.mean().item()
    final_dist = abs(v1 - target)
    assert final_dist < initial_dist, (
        f"value head did not improve: {initial_dist:.3f} -> {final_dist:.3f}"
    )


def test_kl_early_stop_triggers_with_low_target() -> None:
    """Force early stop: target_kl=0 means any drift halts after one epoch."""
    batch, model = _make_synthetic_batch(n=64, seed=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)  # large LR forces drift
    cfg = PPOConfig(
        n_epochs=6,
        minibatch_size=16,
        target_kl=0.0,
        entropy_coef=0.0,
    )

    metrics = ppo_update(model, optim, batch, cfg)

    expected_full = cfg.n_epochs * ((64 + cfg.minibatch_size - 1) // cfg.minibatch_size)
    assert metrics["early_stopped"] == 1.0
    assert metrics["n_updates"] < expected_full


def test_kl_no_early_stop_when_target_is_none() -> None:
    batch, model = _make_synthetic_batch(n=64, seed=4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    cfg = PPOConfig(
        n_epochs=3,
        minibatch_size=16,
        target_kl=None,
    )

    metrics = ppo_update(model, optim, batch, cfg)
    expected = cfg.n_epochs * ((64 + cfg.minibatch_size - 1) // cfg.minibatch_size)
    assert metrics["early_stopped"] == 0.0
    assert metrics["n_updates"] == expected


def test_grad_norm_clipping_is_enforced() -> None:
    """Set max_grad_norm very small; metrics["grad_norm"] reflects pre-clip norm.

    The post-clip norm is what the optimizer steps with, so we verify
    behaviour by checking the per-step grad_norm value reported equals the
    pre-clip norm and that clip_grad_norm_ would have rescaled. We can't
    directly observe the post-clip gradient from metrics, so we instead
    check that parameter movement is bounded by max_grad_norm * lr.
    """
    batch, model = _make_synthetic_batch(n=32, seed=5)
    # Make advantages huge so gradients explode without clipping.
    batch = TrajectoryBatch(
        obs=batch.obs,
        actions=batch.actions,
        masks=batch.masks,
        old_logps=batch.old_logps,
        advantages=batch.advantages * 1e6,
        returns=batch.returns * 1e6,
    )

    lr = 1e-3
    max_grad = 0.1
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    cfg = PPOConfig(
        n_epochs=1,
        minibatch_size=32,
        target_kl=None,
        max_grad_norm=max_grad,
        entropy_coef=0.0,
    )

    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    ppo_update(model, optim, batch, cfg)
    after = {k: v.detach().clone() for k, v in model.named_parameters()}

    # SGD step = -lr * grad. With grad-norm clipping to `max_grad`, the
    # total L2 norm of the update across all params must be ≤ lr * max_grad.
    total_delta_sq = 0.0
    for k in before:
        total_delta_sq += (after[k] - before[k]).pow(2).sum().item()
    total_delta = total_delta_sq ** 0.5
    # Allow a small tolerance for fp32.
    assert total_delta <= lr * max_grad + 1e-5, (
        f"total update norm {total_delta} exceeded lr*max_grad_norm={lr * max_grad}"
    )


def test_lr_in_metrics_matches_optimizer() -> None:
    batch, model = _make_synthetic_batch(n=16, seed=6)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    cfg = PPOConfig(n_epochs=1, minibatch_size=16, target_kl=None)
    metrics = ppo_update(model, optim, batch, cfg)
    assert metrics["lr"] == 5e-4
