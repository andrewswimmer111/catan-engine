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
* Stalemate target is computed by :class:`StalemateValueConfig` on
  ``SelfPlayConfig.stalemate`` (default ``"vp_linear"`` band
  ``[-0.5, -0.1]``). The same instance lives on ``mcts.stalemate`` so
  search backups match training targets.
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
    load_checkpoint,
    model_arch_from,
    obs_layout_version_for,
    save_checkpoint,
)
from rl.training.parallel_self_play import generate_games_parallel
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
    aux_value_coef: float = 0.0
    """Coefficient on the per-step VP auxiliary value loss.

    When > 0, the trainer adds
    ``aux_value_coef * MSE(value_pred, batch.vp_aux_target)`` to the
    total loss. The aux target is each seat's VP / 10 at the
    transition's state, rotated to acting-as-slot-0 (same layout as
    ``value_target``). Use ~0.1-0.5 to break the cold-bootstrap loop
    where the value head never observes a real win (run #AZ-1's and
    #011 pilot's failure mode); 0.0 (default) reproduces canonical AZ.
    """
    bc_anchor_path: Path | None = None
    bc_anchor_coef: float = 0.0
    """KL-to-BC-anchor regularisation: pull learner toward a frozen clone.

    When ``bc_anchor_coef > 0`` and ``bc_anchor_path`` points at a
    graph-encoder checkpoint, the trainer loads that checkpoint at
    init time, freezes it, and on every batch adds
    ``bc_anchor_coef * KL(anchor || learner)`` (summed over legal
    actions, mean over batch) to the total loss. The intent is to
    prevent AZ self-play from eroding habits learned during a
    behavioural-cloning warm-start — observed on az_019 where roads
    built/game regressed from bc_001's 13.3 back to the az_015
    plateau's 10.5 over 20 iterations. Both fields must be set
    together or both left at their defaults; the constructor errors
    otherwise.
    """
    batch_size: int = 128
    batches_per_iter: int = 100
    max_grad_norm: float = 5.0

    # Cadence
    # ``eval_games`` should be a multiple of 4 — the evaluator rotates the
    # learner across all 4 seats, and remainders go to the lower-indexed
    # seats (so 48 → 12/12/12/12, 50 → 13/13/12/12). 48 games per anchor
    # gives ~±11% 2σ resolution on a 17% win-rate, vs. ±26% at 8 games —
    # large enough to distinguish iter-to-iter movement from noise.
    eval_every_iters: int = 4
    eval_games: int = 48
    snapshot_every_iters: int = 4
    log_every_iters: int = 1

    # Parallel self-play (az-009).
    # ``0`` keeps the single-process loop (current default); ``>=1``
    # spawns that many subprocess workers and round-robins self-play
    # games across them on every iteration.
    n_self_play_workers: int = 0

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
        progress_path: str | Path | None = None,
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
        # ``logger`` takes precedence over ``log_dir``; if neither is
        # supplied we build a NoOpLogger off log_dir=None.
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

        # Human-readable progress file. Rewritten in full after every
        # iteration (small file; atomic-rewrite is simpler than appending
        # and getting partial-write corner cases right). ``train_alphazero.py``
        # wires this to ``<output_dir>/progress.md`` by default.
        self._progress_path: Path | None = (
            Path(progress_path) if progress_path else None
        )
        self._history: list[dict[str, float]] = []
        self._train_start_time: float | None = None
        self._train_total_iters: int | None = None

        self._model_arch = model_arch_from(learner.model)
        self._obs_layout_version = obs_layout_version_for(
            self._model_arch.encoder_kind
        )

        # KL-to-BC-anchor regulariser. Loaded once, frozen, kept in eval
        # mode for the lifetime of the trainer. Both fields must agree:
        # a coef without a path can't act, and a path without a coef
        # would silently waste a forward pass per batch.
        self._bc_anchor: PolicyAgent | None = self._maybe_load_bc_anchor()

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
        self._train_total_iters = total_iters
        self._train_start_time = time.time()
        completed = False
        try:
            for _ in range(total_iters):
                self.train_iteration()
            completed = True
        finally:
            self._logger.flush()
            self._write_progress(status="done" if completed else "interrupted")

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
        self._history.append(summary)
        self._write_progress(status="running")
        return summary

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a versioned :class:`CheckpointMeta`-tagged checkpoint to ``path``."""
        save_checkpoint(self._learner, Path(path), self._build_meta())

    # ------------------------------------------------------------------
    # Self-play
    # ------------------------------------------------------------------

    def _run_self_play(self) -> dict[str, float]:
        """Generate ``games_per_iter`` games into the buffer; return stats.

        Dispatches to the parallel subprocess path
        (:func:`generate_games_parallel`) when ``n_self_play_workers > 0``;
        otherwise runs the games sequentially in this process.
        """
        self._learner.model.eval()
        game_seeds = [
            self._cfg.seed * 1_000_003 + self._iteration * 10_007 + g_idx
            for g_idx in range(self._cfg.games_per_iter)
        ]

        if self._cfg.n_self_play_workers > 0:
            games = generate_games_parallel(
                self._learner,
                self._cfg.self_play,
                game_seeds,
                n_workers=self._cfg.n_self_play_workers,
            )
        else:
            games = [
                play_self_play_game(
                    self._learner,
                    self._cfg.self_play,
                    self._self_play_rng,
                    game_seed=seed,
                )
                for seed in game_seeds
            ]

        winners: list[int | None] = []
        n_moves: list[int] = []
        n_transitions = 0
        for game in games:
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
        vp_aux_target_t = torch.as_tensor(
            batch.vp_aux_target, dtype=torch.float32, device=self._device
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
        # Per-step VP auxiliary: MSE between the value head and the
        # rotated VP-normalised vector at the current state. Mixed in
        # at coefficient ``aux_value_coef`` (default 0; non-zero
        # required to break the cold-bootstrap where the value head
        # never observes terminal wins).
        aux_value_loss = (out.value - vp_aux_target_t).pow(2).mean()

        # KL-to-BC-anchor: pull the learner toward a frozen behavioural
        # clone on every batch. Skipped (zero scalar, no anchor forward)
        # when the anchor isn't configured so canonical AZ stays a free
        # path. KL(anchor || learner) over the masked legal-action
        # distribution; illegal slots already carry ~0 mass after the
        # model's MASK_FILL_VALUE softmax, but we ``where``-mask the
        # log-probs explicitly to avoid ``0 * -inf -> NaN``.
        if self._bc_anchor is not None:
            with torch.no_grad():
                anchor_logits = self._bc_anchor.model(obs_t, mask_t).logits
            anchor_log_probs = torch.log_softmax(anchor_logits, dim=-1)
            anchor_probs = anchor_log_probs.exp()
            safe_anchor_log = torch.where(
                mask_t, anchor_log_probs, torch.zeros_like(anchor_log_probs)
            )
            kl_anchor_loss = (
                anchor_probs * (safe_anchor_log - log_probs)
            ).sum(dim=-1).mean()
        else:
            kl_anchor_loss = torch.zeros((), device=self._device)

        loss = (
            policy_loss
            + self._cfg.value_coef * value_loss
            + self._cfg.aux_value_coef * aux_value_loss
            + self._cfg.bc_anchor_coef * kl_anchor_loss
        )

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self._learner.model.parameters(), self._cfg.max_grad_norm
        )
        self._optimizer.step()

        # Diagnostic: std of value_pred across the batch tells us
        # whether the value head is differentiating states (good) or
        # collapsing to a near-constant (the #012 pilot failure mode
        # where the aux loss got drowned out by the terminal stalemate
        # signal). MCTS Q-differentiation is impossible when this is
        # near zero, so it's the canonical metric to watch when the
        # aux-value coef knob is exercised.
        value_pred_std = float(out.value.detach().std().item())

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "aux_value_loss": float(aux_value_loss.item()),
            "kl_anchor_loss": float(kl_anchor_loss.item()),
            "value_pred_std": value_pred_std,
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

    def _maybe_load_bc_anchor(self) -> PolicyAgent | None:
        """Load + freeze the BC anchor for the KL regulariser, or None.

        Raises ``ValueError`` if the two ``bc_anchor_*`` config fields
        disagree (one set, the other at default) or the anchor isn't a
        graph-encoder checkpoint (would consume a different obs layout).
        """
        path = self._cfg.bc_anchor_path
        coef = self._cfg.bc_anchor_coef
        if path is None and coef == 0.0:
            return None
        if path is None:
            raise ValueError(
                "bc_anchor_coef > 0 requires bc_anchor_path to be set"
            )
        if coef <= 0.0:
            raise ValueError(
                "bc_anchor_path is set but bc_anchor_coef <= 0; the "
                "anchor would be loaded without contributing to the loss"
            )
        anchor, meta = load_checkpoint(Path(path), device=self._device)
        if meta.model_arch.encoder_kind != "graph":
            raise ValueError(
                f"bc_anchor_path {path} has encoder_kind="
                f"{meta.model_arch.encoder_kind!r}; AZ training requires "
                "a graph-encoder anchor (same obs layout as the learner)"
            )
        anchor.model.eval()
        for p in anchor.model.parameters():
            p.requires_grad_(False)
        return anchor

    def _default_env_factory(self, seed: int) -> CatanEnv:
        """Build a CatanEnv matching the learner's obs encoder.

        The encoder choice (flat vs graph) is read off the learner; the
        reward function doesn't matter for AZ-side eval because we read
        the env's terminal state directly (win-rate, not return).
        """
        return CatanEnv(seed=seed, obs_encoder=self._learner.obs_encoder)

    # ------------------------------------------------------------------
    # Progress file
    # ------------------------------------------------------------------

    def _write_progress(self, *, status: str) -> None:
        """Atomically rewrite ``progress.md`` with the latest run state."""
        if self._progress_path is None:
            return
        content = self._render_progress_md(status=status)
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._progress_path.write_text(content)

    def _render_progress_md(self, *, status: str) -> str:
        """Build the human-readable run-progress markdown.

        The layout is: header (status + ETA + config) followed by a
        one-row-per-iteration table. Designed for ``cat progress.md`` /
        ``watch -n 30 cat progress.md`` during a long run.
        """
        iter_done = len(self._history)
        total = self._train_total_iters or 0
        elapsed = (
            time.time() - self._train_start_time
            if self._train_start_time is not None
            else 0.0
        )

        lines: list[str] = ["# AlphaZero run progress", ""]
        lines.append(f"- **Status**: {status}")
        lines.append(
            f"- **Iteration**: {iter_done} / {total}"
            if total
            else f"- **Iteration**: {iter_done}"
        )
        lines.append(f"- **Global step**: {self._global_step}")
        lines.append(f"- **Elapsed**: {_fmt_duration(elapsed)}")
        if iter_done > 0 and status == "running" and total > iter_done:
            mean_iter = elapsed / iter_done
            eta = mean_iter * (total - iter_done)
            lines.append(f"- **Mean per-iter**: {_fmt_duration(mean_iter)}")
            lines.append(f"- **ETA**: {_fmt_duration(eta)}")
        lines.append(f"- **Buffer size**: {len(self._buffer)} / {self._cfg.buffer_capacity}")
        if self._evaluator is not None and self._evaluator.prior_snapshot_iter is not None:
            lines.append(
                f"- **Prior snapshot**: iter {self._evaluator.prior_snapshot_iter}"
            )

        lines += ["", "## Config", ""]
        sp = self._cfg.self_play
        mcts = sp.mcts
        lines += [
            f"- device: `{self._device}`",
            f"- self-play workers: {self._cfg.n_self_play_workers}",
            f"- games / iter: {self._cfg.games_per_iter}",
            f"- batches / iter: {self._cfg.batches_per_iter}",
            f"- batch size: {self._cfg.batch_size}",
            f"- buffer capacity: {self._cfg.buffer_capacity}",
            f"- mcts rollouts / move: {mcts.rollouts}",
            f"- mcts c_puct: {mcts.c_puct}",
            f"- dirichlet (α, ε): ({mcts.dirichlet_alpha}, {mcts.dirichlet_epsilon})",
            f"- temperature schedule: T={sp.temperature_initial} for first "
            f"{sp.temperature_threshold_moves} moves, then T={sp.temperature_final}",
            f"- stalemate: {_render_stalemate(sp.stalemate)}",
            f"- max moves: {sp.max_moves}",
            f"- win VP: {sp.victory_point_target}",
            f"- lr: {self._cfg.lr}",
            f"- weight decay (L2): {self._cfg.weight_decay}",
            f"- value coef: {self._cfg.value_coef}",
            f"- aux value coef (per-step VP): {self._cfg.aux_value_coef}",
            f"- bc anchor: {_render_bc_anchor(self._cfg)}",
            f"- eval / snapshot cadence (iters): "
            f"{self._cfg.eval_every_iters} / {self._cfg.snapshot_every_iters}",
        ]

        lines += ["", "## Iterations", ""]
        cols = [
            "iter", "wall", "stale%", "moves/game",
            "pol_loss", "val_loss", "aux_loss", "kl_anc", "val_std",
            "vs_rand", "vs_heur", "vs_prior",
            "buffer",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for h in self._history:
            lines.append(_render_iter_row(h))
        lines.append("")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zero_update_metrics() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "aux_value_loss": 0.0,
        "kl_anchor_loss": 0.0,
        "value_pred_std": 0.0,
        "total_loss": 0.0,
        "grad_norm": 0.0,
    }


# ----------------------------------------------------------------------
# Progress rendering helpers
# ----------------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    """Compact ``Xh Ym Zs`` formatter, dropping the largest zero unit.

    Three tiers so the field stays narrow in the progress table:
    sub-minute → ``12.3s``, sub-hour → ``5m 12s``, otherwise ``1h 23m``.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _fmt_pct(value: float | None) -> str:
    """Percent with no decimals, or ``-`` when the eval didn't fire."""
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def _render_stalemate(stalemate) -> str:
    """One-liner for the progress.md config block."""
    if stalemate.shape == "flat":
        return f"flat={stalemate.flat_value}"
    return f"{stalemate.shape} [{stalemate.low}, {stalemate.high}]"


def _render_bc_anchor(cfg: AZTrainConfig) -> str:
    """One-liner for the progress.md config block."""
    if cfg.bc_anchor_path is None and cfg.bc_anchor_coef == 0.0:
        return "off"
    return f"{cfg.bc_anchor_path} (coef {cfg.bc_anchor_coef})"


def _render_iter_row(summary: dict[str, float]) -> str:
    """One markdown table row from a per-iteration summary dict."""
    def _get(key: str) -> float | None:
        v = summary.get(key)
        return None if v is None else float(v)

    cells = [
        f"{int(summary.get('iter', 0))}",
        _fmt_duration(float(summary.get("wall_seconds", 0.0))),
        f"{float(summary.get('self_play/stalemate_rate', 0.0)) * 100:.0f}",
        f"{float(summary.get('self_play/mean_moves_per_game', 0.0)):.0f}",
        f"{float(summary.get('train/policy_loss', 0.0)):.3f}",
        f"{float(summary.get('train/value_loss', 0.0)):.3f}",
        f"{float(summary.get('train/aux_value_loss', 0.0)):.3f}",
        f"{float(summary.get('train/kl_anchor_loss', 0.0)):.3f}",
        f"{float(summary.get('train/value_pred_std', 0.0)):.3f}",
        _fmt_pct(_get("eval/vs_random/win_rate")),
        _fmt_pct(_get("eval/vs_heuristic/win_rate")),
        _fmt_pct(_get("eval/vs_prior_snapshot/win_rate")),
        f"{int(summary.get('buffer/size', 0))}",
    ]
    return "| " + " | ".join(cells) + " |"
