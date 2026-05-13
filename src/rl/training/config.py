"""Hyperparameter configs for the PPO trainer.

Two configs ship here. :class:`PPOConfig` carries the update-step
hyperparameters consumed by :func:`rl.training.ppo.ppo_update`.
:class:`TrainConfig` carries the orchestration-level hyperparameters used
by :class:`rl.training.trainer.Trainer` — rollout length, eval cadence,
checkpoint cadence, etc.

``DEFAULT_BASELINE_CONFIG`` will be populated in rl-015 after the
hyperparameter sweep converges; for now the defaults are reasonable
starting points borrowed from CleanRL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PPOConfig", "TrainConfig", "DEFAULT_BASELINE_CONFIG"]


@dataclass
class PPOConfig:
    """PPO update-step hyperparameters."""

    lr: float = 3e-4
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float | None = 0.02
    """Approx-KL between old and new policy at which to early-stop an epoch.

    Set to ``None`` to disable early stopping. The classic PPO papers use
    ``0.01–0.03`` for environments with relatively short episodes.
    """


@dataclass
class TrainConfig:
    """Trainer orchestration hyperparameters.

    These are consumed by :class:`Trainer` and :class:`RolloutWorker`, not
    by the inner update loop. ``ppo`` carries the update-step config.
    """

    ppo: PPOConfig = field(default_factory=PPOConfig)
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    hidden_sizes: tuple[int, ...] = (512, 512, 512)
    eval_every: int = 50_000
    eval_n_games: int = 30
    checkpoint_every: int = 100_000
    log_every: int = 1
    seed: int = 0


DEFAULT_BASELINE_CONFIG: TrainConfig = TrainConfig(
    ppo=PPOConfig(
        lr=3e-4,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        minibatch_size=256,
        target_kl=0.02,
    ),
    rollout_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,
    hidden_sizes=(512, 512, 512),
    eval_every=50_000,
    eval_n_games=30,
    checkpoint_every=100_000,
    log_every=1,
    seed=0,
)
"""Starting-point baseline for the rl-015 convergence sweep.

These are reasonable defaults borrowed from CleanRL-style PPO recipes
adapted for the Catan domain (large discrete action space, long
episodes, sparse terminal reward). They have **not** been swept against
the rl-015 grid yet; revise once empirical results are in.

Sweep grid from the rl-015 plan:

* ``lr`` ∈ {1e-4, 3e-4, 1e-3}
* ``entropy_coef`` ∈ {0.001, 0.01, 0.05}
* ``clip_range`` ∈ {0.1, 0.2}
* ``hidden_sizes`` ∈ {(256, 256), (512, 512, 512)}

Pick the config with the highest win rate vs random at 10M steps.
"""
