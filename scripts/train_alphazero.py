#!/usr/bin/env python3
"""Train driver for AlphaZero-style training.

Mirrors :mod:`scripts.train` for the AZ loop: every interesting
hyperparameter is exposed as a flag, the run is written to a timestamped
directory under ``runs/``, and the resulting ``final.pt`` is shape-
compatible with the existing :func:`rl.training.checkpoint.load_checkpoint`
path so the existing eval scripts (``rl.cli evaluate``, the GUI replay
viewer) load AZ-trained checkpoints with no special-casing.

Outputs (under ``--output-dir``, default ``runs/<timestamp>``):

* ``tb/`` — TensorBoard event files.
* ``snapshots/`` — periodic ``iter_N.pt`` checkpoints.
* ``config.json`` — pinned snapshot of the CLI flags + resolved
  hyperparameters.
* ``progress.md`` — human-readable run progress: status (running /
  done / interrupted), ETA, per-iteration table of losses + eval
  win-rates. Rewritten after every iteration. ``cat progress.md`` (or
  ``watch -n 30 cat progress.md``) is the canonical way to follow a
  long run without spinning up TensorBoard.
* ``elo.json`` — Elo trajectory across iterations.
* ``final.pt`` — final checkpoint written on clean exit.

Usage::

    python scripts/train_alphazero.py --total-iters 30 --device mps \\
        --games-per-iter 16 --batches-per-iter 100 --mcts-rollouts 50 \\
        --output-dir runs/az_001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace as _dc_replace
from pathlib import Path

import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.action import ActionEncoder
from rl.encoding.graph_observation import (
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.env.catan_env import CatanEnv
from rl.evaluation.az_evaluator import AZEvalConfig, AZEvaluator
from rl.evaluation.elo import EloTracker
from rl.models.gnn import DEFAULT_GNN_ARCH, GNNPolicyValue
from rl.search.mcts import MCTSConfig
from rl.stalemate_value import StalemateValueConfig
from rl.training.alphazero import AlphaZeroTrainer, AZTrainConfig
from rl.training.checkpoint import load_checkpoint
from rl.training.self_play import SelfPlayConfig
from rl.utils.device import resolve_device
from rl.utils.logging import make_logger


_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree. Exposed for testing the flag schema."""
    p = argparse.ArgumentParser(
        prog="train_alphazero",
        description="AlphaZero-style training driver.",
    )

    # Loop.
    p.add_argument(
        "--total-iters",
        type=int,
        required=True,
        help="number of AlphaZero iterations to run",
    )
    p.add_argument(
        "--games-per-iter",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["games_per_iter"].default,
        help="self-play games generated each iteration",
    )
    p.add_argument(
        "--batches-per-iter",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["batches_per_iter"].default,
        help="gradient steps each iteration (uniform sampling from buffer)",
    )

    # Optimisation.
    p.add_argument("--lr", type=float, default=AZTrainConfig.__dataclass_fields__["lr"].default)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=AZTrainConfig.__dataclass_fields__["weight_decay"].default,
        help="Adam weight_decay; the L2 term in the AZ loss.",
    )
    p.add_argument(
        "--value-coef",
        type=float,
        default=AZTrainConfig.__dataclass_fields__["value_coef"].default,
        help="loss weight on the per-seat value MSE term.",
    )
    p.add_argument(
        "--aux-value-coef",
        type=float,
        default=AZTrainConfig.__dataclass_fields__["aux_value_coef"].default,
        help="loss weight on the per-step VP auxiliary value MSE term. "
             "0.0 (default) = canonical AZ. Non-zero (try 0.1-0.5) "
             "bridges the cold-bootstrap loop where the value head "
             "never observes terminal wins by regressing it against "
             "current VP / 10 at every state.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["batch_size"].default,
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=AZTrainConfig.__dataclass_fields__["max_grad_norm"].default,
    )

    # Buffer.
    p.add_argument(
        "--buffer-capacity",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["buffer_capacity"].default,
        help="bounded ring-buffer size in transitions",
    )

    # MCTS / self-play.
    mcts_defaults = MCTSConfig()
    p.add_argument("--mcts-rollouts", type=int, default=mcts_defaults.rollouts)
    p.add_argument("--mcts-cpuct", type=float, default=mcts_defaults.c_puct)
    p.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=mcts_defaults.dirichlet_alpha,
        help="Dirichlet concentration; effective only when epsilon > 0",
    )
    p.add_argument(
        "--dirichlet-epsilon",
        type=float,
        default=0.25,
        help="root-noise mix weight (AZ default 0.25 for self-play)",
    )

    sp_defaults = SelfPlayConfig()
    p.add_argument(
        "--temperature-initial",
        type=float,
        default=sp_defaults.temperature_initial,
    )
    p.add_argument(
        "--temperature-final",
        type=float,
        default=sp_defaults.temperature_final,
    )
    p.add_argument(
        "--temperature-threshold-moves",
        type=int,
        default=sp_defaults.temperature_threshold_moves,
        help="after this many moves, switch from initial to final temperature.",
    )
    stale_defaults = StalemateValueConfig()
    p.add_argument(
        "--stalemate-shape",
        choices=("flat", "vp_linear"),
        default=stale_defaults.shape,
        help="how to compute the per-seat stalemate value target. "
             "'flat' uses --stalemate-flat-value for every seat (legacy "
             "constant target). 'vp_linear' (default) maps each seat "
             "into [--stalemate-low, --stalemate-high] by combining its "
             "VP rank and absolute VP at game end — restores variance "
             "in the value target distribution so the value loss "
             "doesn't collapse onto a constant.",
    )
    p.add_argument(
        "--stalemate-flat-value",
        type=float,
        default=stale_defaults.flat_value,
        help="constant per-seat target when --stalemate-shape=flat "
             "(default -0.25; the legacy Phase 3 design pass value).",
    )
    p.add_argument(
        "--stalemate-low",
        type=float,
        default=stale_defaults.low,
        help="worst-case target in the vp_linear band (default -0.5).",
    )
    p.add_argument(
        "--stalemate-high",
        type=float,
        default=stale_defaults.high,
        help="best-case (leader, high VP) target in the vp_linear band "
             "(default -0.1). Must satisfy max(stalemate-high) < min(win=1.0) "
             "so the policy still prefers actual wins.",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=sp_defaults.max_moves,
        help="hard cap on per-self-play-game moves; truncates as stalemate",
    )

    # Cadence.
    p.add_argument(
        "--eval-every",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["eval_every_iters"].default,
        help="iterations between vs-random / vs-heuristic anchor evals",
    )
    p.add_argument(
        "--eval-games",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["eval_games"].default,
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=AZTrainConfig.__dataclass_fields__["snapshot_every_iters"].default,
        help="iterations between checkpoint snapshots",
    )
    p.add_argument(
        "--no-eval-random",
        action="store_true",
        help="skip the vs-random anchor benchmark",
    )
    p.add_argument(
        "--no-eval-heuristic",
        action="store_true",
        help="skip the vs-heuristic anchor benchmark",
    )
    p.add_argument(
        "--no-eval-prior-snapshot",
        action="store_true",
        help="skip the vs-prior-snapshot benchmark",
    )
    p.add_argument(
        "--promote-threshold",
        type=float,
        default=None,
        help="optional promotion gate: only refresh the prior-snapshot "
             "opponent when learner_win_rate > THIS value. Default None "
             "(continuous training; refresh on every snapshot).",
    )
    p.add_argument(
        "--self-play-workers",
        type=int,
        default=0,
        help="parallel-self-play subprocess workers. 0 (default) keeps "
             "the single-process loop; >=1 spawns that many CPU workers "
             "via the spawn start method. The parent's MPS / CUDA model "
             "is untouched — workers receive a CPU-only state-dict copy.",
    )

    # I/O.
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="torch device for the learner model.",
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="optional checkpoint to warm-start weights from; must be a "
             "graph-encoder, vector-value-head checkpoint (i.e. another "
             "AZ run).",
    )
    p.add_argument(
        "--init-from-encoder",
        type=Path,
        default=None,
        help="optional checkpoint to partially warm-start from: copies "
             "encoder + policy-head weights only, re-initializes the "
             "value head. Use this to seed an AZ run from a PPO-trained "
             "scalar-head GNN checkpoint (e.g. run #5's final.pt) — the "
             "encoder is the expensive part and worth carrying over, but "
             "the value head's training target shape differs. Mutually "
             "exclusive with --init-from.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run directory; defaults to runs/az_<timestamp>",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> AZTrainConfig:
    # Single source of truth for the stalemate target — the same
    # instance is plumbed into both MCTSConfig and SelfPlayConfig so the
    # network's training target and MCTS's terminal backup are identical
    # by construction.
    stalemate = StalemateValueConfig(
        shape=args.stalemate_shape,
        flat_value=args.stalemate_flat_value,
        low=args.stalemate_low,
        high=args.stalemate_high,
    )
    mcts = MCTSConfig(
        rollouts=args.mcts_rollouts,
        c_puct=args.mcts_cpuct,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature=1.0,  # policy_distribution uses an explicit T; this is a default.
        stalemate=stalemate,
    )
    self_play = SelfPlayConfig(
        mcts=mcts,
        temperature_initial=args.temperature_initial,
        temperature_final=args.temperature_final,
        temperature_threshold_moves=args.temperature_threshold_moves,
        stalemate=stalemate,
        max_moves=args.max_moves,
        player_ids=_PLAYER_IDS,
    )
    return AZTrainConfig(
        self_play=self_play,
        games_per_iter=args.games_per_iter,
        buffer_capacity=args.buffer_capacity,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_coef=args.value_coef,
        aux_value_coef=args.aux_value_coef,
        batch_size=args.batch_size,
        batches_per_iter=args.batches_per_iter,
        max_grad_norm=args.max_grad_norm,
        eval_every_iters=args.eval_every,
        eval_games=args.eval_games,
        snapshot_every_iters=args.snapshot_every,
        log_every_iters=1,
        n_self_play_workers=args.self_play_workers,
        seed=args.seed,
        player_ids=_PLAYER_IDS,
    )


def _build_learner(device: torch.device) -> PolicyAgent:
    """Construct a fresh GNN learner with the AZ-shape (vector) value head."""
    # DEFAULT_GNN_ARCH already defaults to value_kind="vector"; explicit
    # to make the AZ contract loud at the construction site.
    arch = _dc_replace(DEFAULT_GNN_ARCH, value_kind="vector")
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=arch,
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(list(_PLAYER_IDS)),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
        device=device,
    )


def _resolve_warm_start(
    path: Path | None, device: torch.device
) -> PolicyAgent | None:
    """Load ``--init-from``, validate it's AZ-shaped, and return its agent.

    Refuses to warm-start across encoder kinds OR value-head kinds: a
    flat or a scalar-head checkpoint can't seed an AZ run with this
    path (state_dicts don't line up). For the scalar-head case, use
    ``--init-from-encoder`` instead — it drops the value-head keys
    so the destination keeps its freshly-initialized vector head.
    """
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"error: --init-from path does not exist: {path}")
    agent, meta = load_checkpoint(path, device=device)
    if meta.model_arch.encoder_kind != "graph":
        raise SystemExit(
            f"error: --init-from {path} is encoder_kind="
            f"{meta.model_arch.encoder_kind!r}; AlphaZero training "
            "requires a graph-encoder checkpoint."
        )
    assert meta.model_arch.gnn_arch is not None  # validated by checkpoint loader
    if meta.model_arch.gnn_arch.value_kind != "vector":
        raise SystemExit(
            f"error: --init-from {path} has value_kind="
            f"{meta.model_arch.gnn_arch.value_kind!r}; AlphaZero training "
            "requires a vector-value-head checkpoint. Pass the same path "
            "via --init-from-encoder to copy only encoder + policy-head "
            "weights and re-initialize the value head."
        )
    return agent


# Keys whose shape depends on ``value_kind`` and so are intentionally
# dropped during the partial warm-start path.
_VALUE_HEAD_STATE_KEYS: frozenset[str] = frozenset(
    {"value_head.weight", "value_head.bias"}
)


def _resolve_partial_warm_start(
    path: Path | None, device: torch.device
) -> dict | None:
    """Load ``--init-from-encoder`` and return the encoder+policy state-dict.

    The source checkpoint must be a graph-encoder checkpoint; its
    ``value_kind`` is irrelevant (the value head's weights are dropped
    either way). Returns the filtered state-dict ready for
    ``model.load_state_dict(sd, strict=False)``; the caller is
    responsible for verifying that only the expected keys
    (``_VALUE_HEAD_STATE_KEYS``) end up missing.
    """
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(
            f"error: --init-from-encoder path does not exist: {path}"
        )
    agent, meta = load_checkpoint(path, device=device)
    if meta.model_arch.encoder_kind != "graph":
        raise SystemExit(
            f"error: --init-from-encoder {path} is encoder_kind="
            f"{meta.model_arch.encoder_kind!r}; AlphaZero training "
            "requires a graph-encoder checkpoint."
        )
    sd = agent.state_dict()
    return {k: v for k, v in sd.items() if k not in _VALUE_HEAD_STATE_KEYS}


def _config_to_jsonable(cfg: AZTrainConfig, args: argparse.Namespace) -> dict:
    """Flatten the config and the CLI for the on-disk run record."""
    return {
        "az": {
            "games_per_iter": cfg.games_per_iter,
            "buffer_capacity": cfg.buffer_capacity,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "value_coef": cfg.value_coef,
            "aux_value_coef": cfg.aux_value_coef,
            "batch_size": cfg.batch_size,
            "batches_per_iter": cfg.batches_per_iter,
            "max_grad_norm": cfg.max_grad_norm,
            "eval_every_iters": cfg.eval_every_iters,
            "eval_games": cfg.eval_games,
            "snapshot_every_iters": cfg.snapshot_every_iters,
            "seed": cfg.seed,
        },
        "mcts": {
            "rollouts": cfg.self_play.mcts.rollouts,
            "c_puct": cfg.self_play.mcts.c_puct,
            "dirichlet_alpha": cfg.self_play.mcts.dirichlet_alpha,
            "dirichlet_epsilon": cfg.self_play.mcts.dirichlet_epsilon,
            "seed": cfg.self_play.mcts.seed,
        },
        "self_play": {
            "temperature_initial": cfg.self_play.temperature_initial,
            "temperature_final": cfg.self_play.temperature_final,
            "temperature_threshold_moves": cfg.self_play.temperature_threshold_moves,
            "max_moves": cfg.self_play.max_moves,
        },
        "stalemate": {
            "shape": cfg.self_play.stalemate.shape,
            "flat_value": cfg.self_play.stalemate.flat_value,
            "low": cfg.self_play.stalemate.low,
            "high": cfg.self_play.stalemate.high,
        },
        "cli": {
            "device": args.device,
            "init_from": str(args.init_from) if args.init_from else None,
            "total_iters": args.total_iters,
        },
    }


def _env_factory_for(learner: PolicyAgent):
    """Build a seed→CatanEnv factory wired to the learner's obs encoder."""

    def factory(seed: int) -> CatanEnv:
        return CatanEnv(seed=seed, obs_encoder=learner.obs_encoder)

    return factory


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_iters <= 0:
        print(
            f"error: --total-iters must be positive (got {args.total_iters})",
            file=sys.stderr,
        )
        return 2
    if args.init_from is not None and args.init_from_encoder is not None:
        print(
            "error: --init-from and --init-from-encoder are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    output_dir = (
        args.output_dir
        or Path("runs") / f"az_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "tb"
    snapshot_dir = output_dir / "snapshots"

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    learner = _build_learner(device)
    warm = _resolve_warm_start(args.init_from, device)
    if warm is not None:
        # Only weights carry over; the optimiser, buffer, and iteration
        # counter all restart from scratch by construction.
        learner.load_state_dict(warm.state_dict())
        print(f"[az] warm-started from {args.init_from}", flush=True)

    partial_sd = _resolve_partial_warm_start(args.init_from_encoder, device)
    if partial_sd is not None:
        result = learner.model.load_state_dict(partial_sd, strict=False)
        if result.unexpected_keys:
            print(
                f"error: --init-from-encoder source has unexpected keys "
                f"not present in the destination model: "
                f"{sorted(result.unexpected_keys)}",
                file=sys.stderr,
            )
            return 2
        extra_missing = set(result.missing_keys) - _VALUE_HEAD_STATE_KEYS
        if extra_missing:
            print(
                f"error: --init-from-encoder source is missing keys "
                f"beyond the value-head set: {sorted(extra_missing)}",
                file=sys.stderr,
            )
            return 2
        print(
            f"[az] partially warm-started encoder + policy head from "
            f"{args.init_from_encoder}; value head re-initialized "
            f"({len(partial_sd)} tensors copied, "
            f"{len(_VALUE_HEAD_STATE_KEYS)} re-initialized)",
            flush=True,
        )

    cfg = _build_config(args)
    (output_dir / "config.json").write_text(
        json.dumps(_config_to_jsonable(cfg, args), indent=2)
    )

    print(f"[az] output_dir={output_dir} device={device}", flush=True)
    print(
        f"[az] total_iters={args.total_iters} "
        f"games/iter={cfg.games_per_iter} batches/iter={cfg.batches_per_iter} "
        f"mcts_rollouts={cfg.self_play.mcts.rollouts}",
        flush=True,
    )

    # Build the AZ evaluator once so the trainer's logger is shared with
    # the evaluator's TB scalars (single SummaryWriter, single TB log dir).
    logger = make_logger(log_dir)
    env_factory = _env_factory_for(learner)
    eval_cfg = AZEvalConfig(
        every_iters=cfg.eval_every_iters,
        n_games=cfg.eval_games,
        bench_random=not args.no_eval_random,
        bench_heuristic=not args.no_eval_heuristic,
        bench_prior_snapshot=not args.no_eval_prior_snapshot,
        promotion_threshold=args.promote_threshold,
        player_ids=cfg.player_ids,
    )
    evaluator = AZEvaluator(
        config=eval_cfg,
        env_factory=env_factory,
        logger=logger,
        elo=EloTracker(),
        ratings_path=output_dir / "elo.json",
    )

    trainer = AlphaZeroTrainer(
        learner=learner,
        config=cfg,
        env_factory=env_factory,
        snapshot_dir=snapshot_dir,
        evaluator=evaluator,
        logger=logger,
        progress_path=output_dir / "progress.md",
    )

    t0 = time.time()
    try:
        trainer.train(total_iters=args.total_iters)
    except KeyboardInterrupt:
        print(
            "\n[az] KeyboardInterrupt — saving final checkpoint and exiting",
            flush=True,
        )
    finally:
        trainer.save_checkpoint(output_dir / "final.pt")
    dt = time.time() - t0
    print(
        f"[az] done: {trainer.iteration} iters / {trainer.global_step} "
        f"grad steps in {dt:.1f}s → {output_dir / 'final.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
