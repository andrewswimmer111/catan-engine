"""AlphaZero training loop.

One :meth:`AlphaZeroTrainer.train_iteration` does, in order:

1. **Self-play.** Generate ``games_per_iter`` games with the current
   learner using MCTS at ``config.mcts`` (Dirichlet noise enabled).
   Append every transition to the bounded replay buffer.
2. **Update.** Run ``batches_per_iter`` Adam steps over uniform mini-batches
   drawn from the buffer. Loss is policy cross-entropy against the MCTS
   visit distribution plus value MSE against the rotated per-seat
   outcome; L2 regularisation lives in the Adam ``weight_decay`` knob.
3. **Eval (cadenced).** Every ``eval_every_iters`` iterations, play
   ``eval_games`` tournament games vs random and heuristic anchors; log
   win-rates to TB.
4. **Snapshot (cadenced).** Every ``snapshot_every_iters`` iterations,
   save a versioned checkpoint into ``snapshot_dir``.

Design choices specific to Catan + Phase 3:

* The learner's value head is ``value_kind="vector"`` (az-001) so a
  single forward at the leaf gives the whole per-seat backup vector.
  The loss is direct MSE against the rotated value target stored on
  every :class:`rl.training.self_play.SelfPlayTransition`.
* Stalemate target is the soft-penalty ``-0.25`` baked into
  ``SelfPlayConfig.stalemate_value`` per the Phase 3 design memo.
* The trainer is **continuous** (no promotion gate). Each iteration's
  network plays the next iteration's self-play; the prior-best
  comparison is purely informational. Promotion is left to az-008.

Known v1 warts (documented per project convention):

* **Single-process self-play.** No parallelism across games — that's
  az-009. Throughput will be poor on large rollout budgets; tune
  ``games_per_iter`` * ``mcts.rollouts`` accordingly.
* **Buffer not persisted across runs.** ``--init-from`` carries the
  weights but starts the buffer empty; the first iteration trains on
  only its own self-play data.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.az_evaluator import AZEvaluator
from rl.evaluation.tournament import Tournament  # noqa: F401  re-exported for callers
from rl.search.mcts import MCTSConfig  # noqa: F401  re-exported via SelfPlayConfig
from rl.training.az_buffer import AZBatch, AZReplayBuffer
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    model_arch_from,
    obs_layout_version_for,
    save_checkpoint,
)
from rl.training.self_play import (
    SelfPlayConfig,
    SelfPlayGame,
    play_self_play_game,
)
from rl.utils.logging import NoOpLogger, TBLogger, make_logger

__all__ = ["AlphaZeroTrainer", "AZTrainConfig"]


_DEFAULT_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


@dataclass(frozen=True)
class AZTrainConfig:
    """All hyperparameters for one :class:`AlphaZeroTrainer` run.

    Defaults sized for an iteratable single-process loop on CPU — they
    are not the AlphaGo-Zero defaults. Larger rollout budgets and
    buffer sizes belong on the GPU / MPS run via az-007 + az-002.
    """

    # Self-play
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    games_per_iter: int = 20

    # Replay buffer
    buffer_capacity: int = 50_000

    # Optimisation
    lr: float = 3e-4
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    batch_size: int = 128
    batches_per_iter: int = 100
    max_grad_norm: float = 5.0

    # Cadence
    eval_every_iters: int = 5
    eval_games: int = 20
    snapshot_every_iters: int = 5
    log_every_iters: int = 1

    # Seeding
    seed: int = 0

    # Identity (carried into checkpoint meta)
    player_ids: tuple[PlayerID, ...] = _DEFAULT_PLAYER_IDS


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------


class AlphaZeroTrainer:
    """Self-play → buffer → update → snapshot loop with periodic eval.

    The trainer drives a single :class:`PolicyAgent` (the learner)
    forward across iterations. The learner is also the source of
    self-play games, so each iteration's network plays itself. There
    is no opponent pool (vs PPO) — symmetry of self-play replaces it.
    """

    def __init__(
        self,
        learner: PolicyAgent,
        config: AZTrainConfig,
        env_factory: Callable[[int], CatanEnv] | None = None,
        log_dir: str | Path | None = None,
        snapshot_dir: str | Path | None = None,
        evaluator: AZEvaluator | None = None,
        logger: TBLogger | NoOpLogger | None = None,
    ) -> None:
        # The model's value head must be the per-seat vector kind — that's
        # the entire AZ contract (one forward at the leaf gives the
        # backup vector). Caller error is loud here, not silent later.
        if getattr(learner.model, "value_kind", "scalar") != "vector":
            raise ValueError(
                "AlphaZeroTrainer requires a learner with a vector value head; "
                "got value_kind="
                f"{getattr(learner.model, 'value_kind', 'scalar')!r}. "
                "Build the GNN with arch.value_kind='vector' (the default) "
                "to satisfy this."
            )

        self._learner = learner
        self._cfg = config
        self._device = learner.device

        self._buffer = AZReplayBuffer(
            capacity=config.buffer_capacity,
            rng=random.Random(config.seed),
        )
        # Distinct RNG for self-play action sampling so the buffer's
        # sampling stream is independent of the self-play stream.
        self._self_play_rng = random.Random(config.seed + 1)

        self._optimizer = torch.optim.Adam(
            self._learner.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self._iteration = 0
        self._global_step = 0  # one per gradient step
        self._n_self_play_transitions = 0
        # ``logger`` takes precedence over ``log_dir`` so a caller that
        # wants to share one SummaryWriter across the trainer + evaluator
        # can pass a pre-built logger; otherwise we build one off log_dir
        # (or NoOpLogger when log_dir is None).
        if logger is not None and log_dir is not None:
            raise ValueError(
                "Pass either ``logger`` or ``log_dir`` to AlphaZeroTrainer, not both."
            )
        self._logger: TBLogger | NoOpLogger = (
            logger if logger is not None else make_logger(log_dir)
        )

        self._snapshot_dir: Path | None = (
            Path(snapshot_dir) if snapshot_dir else None
        )
        if self._snapshot_dir is not None:
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot_path: Path | None = None

        self._env_factory: Callable[[int], CatanEnv] = (
            env_factory if env_factory is not None else self._default_env_factory
        )
        # Optional eval driver. ``None`` falls back to the lite anchor
        # eval below (vs random + vs heuristic only, no Elo / promotion).
        # az-007's CLI wires up a full :class:`AZEvaluator` by default.
        self._evaluator: AZEvaluator | None = evaluator

        self._model_arch = model_arch_from(learner.model)
        self._obs_layout_version = obs_layout_version_for(
            self._model_arch.encoder_kind
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def learner(self) -> PolicyAgent:
        return self._learner

    @property
    def logger(self) -> TBLogger | NoOpLogger:
        return self._logger

    @property
    def env_factory(self) -> Callable[[int], CatanEnv]:
        return self._env_factory

    @property
    def evaluator(self) -> AZEvaluator | None:
        return self._evaluator

    def train(self, total_iters: int) -> None:
        """Run ``total_iters`` AlphaZero iterations end-to-end."""
        if total_iters <= 0:
            raise ValueError(f"total_iters must be positive (got {total_iters})")
        try:
            for _ in range(total_iters):
                self.train_iteration()
        finally:
            self._logger.flush()

    def train_iteration(self) -> dict[str, float]:
        """One AZ iteration: self-play → update → cadenced eval/snapshot.

        Returns the flat scalar dict logged to TB so callers (tests,
        scripts) can assert on it.
        """
        t0 = time.time()
        self_play_summary = self._run_self_play()
        update_summary = self._run_updates()
        eval_summary = self._maybe_eval()
        self._maybe_snapshot()
        self._iteration += 1

        wall = time.time() - t0
        summary = {
            "iter": float(self._iteration),
            "wall_seconds": wall,
            **{f"self_play/{k}": v for k, v in self_play_summary.items()},
            **{f"train/{k}": v for k, v in update_summary.items()},
            **{f"eval/{k}": v for k, v in eval_summary.items()},
            "buffer/size": float(len(self._buffer)),
        }
        self._log_iteration(summary)
        return summary

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a versioned :class:`CheckpointMeta`-tagged checkpoint to ``path``."""
        save_checkpoint(self._learner, Path(path), self._build_meta())

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def _run_self_play(self) -> dict[str, float]:
        """Generate ``games_per_iter`` games into the buffer; return stats."""
        self._learner.model.eval()
        winners: list[int | None] = []
        n_moves: list[int] = []
        n_transitions = 0
        for g_idx in range(self._cfg.games_per_iter):
            game_seed = (
                self._cfg.seed * 1_000_003
                + self._iteration * 10_007
                + g_idx
            )
            game = play_self_play_game(
                self._learner,
                self._cfg.self_play,
                self._self_play_rng,
                game_seed=game_seed,
            )
            self._buffer.extend(game.transitions)
            winners.append(game.winner_seat_idx)
            n_moves.append(game.n_moves)
            n_transitions += len(game.transitions)

        stalemates = sum(1 for w in winners if w is None)
        summary = {
            "n_games": float(self._cfg.games_per_iter),
            "n_transitions": float(n_transitions),
            "stalemate_rate": (
                stalemates / self._cfg.games_per_iter
                if self._cfg.games_per_iter > 0
                else 0.0
            ),
            "mean_moves_per_game": (
                float(np.mean(n_moves)) if n_moves else 0.0
            ),
        }
        self._n_self_play_transitions += n_transitions
        return summary

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def _run_updates(self) -> dict[str, float]:
        """Draw ``batches_per_iter`` mini-batches and step the optimiser."""
        if len(self._buffer) == 0:
            # Defensive: a self-play iteration that produces no transitions
            # (all games hit a domestic-trade-only loop) — skip the update.
            return _zero_update_metrics()

        self._learner.model.train()
        sums = _zero_update_metrics()
        for _ in range(self._cfg.batches_per_iter):
            batch = self._buffer.sample(self._cfg.batch_size)
            step_metrics = self._train_step(batch)
            for k in sums:
                sums[k] += step_metrics[k]
            self._global_step += 1

        n = max(self._cfg.batches_per_iter, 1)
        return {k: v / n for k, v in sums.items()}

    def _train_step(self, batch: AZBatch) -> dict[str, float]:
        """One Adam step on a single mini-batch; returns scalar metrics."""
        obs_t = torch.as_tensor(batch.obs, dtype=torch.float32, device=self._device)
        mask_t = torch.as_tensor(batch.action_mask, dtype=torch.bool, device=self._device)
        policy_target_t = torch.as_tensor(
            batch.policy_target, dtype=torch.float32, device=self._device
        )
        value_target_t = torch.as_tensor(
            batch.value_target, dtype=torch.float32, device=self._device
        )

        out = self._learner.model(obs_t, mask_t)
        # Policy CE = -sum(pi_target * log pi_model). The model's forward
        # already masks illegal slots to MASK_FILL_VALUE (-1e9) so the
        # log-softmax at those slots is ~-inf. mcts_policy is 0 on
        # illegals (we built it that way in self_play.py), but
        # ``0 * -inf == nan`` in float so we explicitly zero illegal
        # log-probs before the dot product.
        log_probs = torch.log_softmax(out.logits, dim=-1)
        log_probs = torch.where(mask_t, log_probs, torch.zeros_like(log_probs))
        policy_loss = -(policy_target_t * log_probs).sum(dim=-1).mean()

        # Value MSE over the per-seat vector; mean over batch and seats.
        value_loss = (out.value - value_target_t).pow(2).mean()

        loss = policy_loss + self._cfg.value_coef * value_loss

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._learner.model.parameters(), self._cfg.max_grad_norm
        )
        self._optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()),
        }

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _maybe_eval(self) -> dict[str, float]:
        """Run cadenced anchor-evals; return a flat scalar dict (possibly empty).

        When an :class:`AZEvaluator` was supplied at construction, that
        evaluator owns the eval logic (anchors + prior-snapshot +
        Elo). Without one, we fall back to a lite vs-random / vs-heuristic
        anchor eval on the trainer-level cadence — sufficient for the
        smoke tests in :mod:`tests.rl.test_alphazero_trainer`.
        """
        next_iter = self._iteration + 1  # post-increment view

        if self._evaluator is not None:
            self._learner.model.eval()
            result = self._evaluator.maybe_run(self._learner, next_iter)
            return result or {}

        if self._cfg.eval_every_iters <= 0:
            return {}
        if next_iter % self._cfg.eval_every_iters != 0:
            return {}
        self._learner.model.eval()
        anchor_results = {
            "vs_random": self._eval_against(
                lambda rng: RandomAgent(rng, skip_proposals=True),
                seed=next_iter * 7919,
            ),
            "vs_heuristic": self._eval_against(
                lambda _rng: HeuristicAgent(),
                seed=next_iter * 7919 + 1,
            ),
        }
        flat: dict[str, float] = {}
        for label, win_rate in anchor_results.items():
            flat[f"{label}/win_rate"] = win_rate
        return flat

    def _eval_against(
        self,
        opponent_factory: Callable[[random.Random], Agent],
        *,
        seed: int,
    ) -> float:
        """Play ``eval_games`` games of learner vs the supplied factory.

        The learner takes seat 0; the other three seats are filled by
        independent calls to ``opponent_factory``. Returns the learner's
        win rate.
        """
        rng = random.Random(seed)
        learner_seat = self._cfg.player_ids[0]
        agents: dict[PlayerID, Agent] = {learner_seat: self._learner}
        for pid in self._cfg.player_ids[1:]:
            agents[pid] = opponent_factory(random.Random(rng.randrange(2**32)))
        result = Tournament(self._env_factory).play(
            agents, n_games=self._cfg.eval_games, base_seed=seed
        )
        return float(result.win_rates.get(learner_seat, 0.0))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _maybe_snapshot(self) -> None:
        """Periodic snapshot to ``snapshot_dir``; refresh prior-snapshot.

        When an :class:`AZEvaluator` is attached, every snapshot also
        offers the current learner as the new prior-snapshot opponent.
        The evaluator's promotion gate decides whether to accept it.
        """
        if self._cfg.snapshot_every_iters <= 0:
            return
        next_iter = self._iteration + 1
        if next_iter % self._cfg.snapshot_every_iters != 0:
            return

        if self._snapshot_dir is not None:
            path = self._snapshot_dir / f"iter_{next_iter}.pt"
            self.save_checkpoint(path)
            self._last_snapshot_path = path

        if self._evaluator is not None:
            promoted = self._evaluator.refresh_prior_snapshot(
                self._learner, next_iter
            )
            if promoted:
                self._logger.log_scalar(
                    "eval/promotion_event", 1.0, self._global_step
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_iteration(self, summary: dict[str, float]) -> None:
        if self._cfg.log_every_iters <= 0:
            return
        if self._iteration % self._cfg.log_every_iters != 0:
            return
        for name, value in summary.items():
            self._logger.log_scalar(name, value, self._global_step)

    def _build_meta(self) -> CheckpointMeta:
        return CheckpointMeta(
            obs_layout_version=self._obs_layout_version,
            action_layout_version=ACTION_LAYOUT_VERSION,
            model_arch=self._model_arch,
            train_step=self._global_step,
            timestamp=time.time(),
            config_hash="alphazero",  # AZTrainConfig drifts independently of TrainConfig
        )

    def _default_env_factory(self, seed: int) -> CatanEnv:
        """Build a CatanEnv matching the learner's obs encoder.

        The encoder choice (flat vs graph) is read off the learner; the
        reward function doesn't matter for AZ-side eval because we read
        the env's terminal state directly (win-rate, not return).
        """
        return CatanEnv(seed=seed, obs_encoder=self._learner.obs_encoder)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zero_update_metrics() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
        "grad_norm": 0.0,
    }
