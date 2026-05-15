#!/usr/bin/env python3
"""Train driver for the PPO learner.

This is the scripted entry point referenced in ``rl.cli``'s docstring — the
CLI's ``train`` subcommand is a thin smoke wrapper, this is what real runs
(sweeps, long convergence runs, anything that needs anything other than the
defaults) go through.

What it does on top of the CLI:

* Exposes every interesting hyperparameter as a flag (``--lr``,
  ``--entropy-coef``, ``--clip-range``, ``--hidden-sizes``, ``--rollout-steps``,
  ``--num-envs``, ``--eval-every``, ``--snapshot-every``, …).
* Wires an :class:`EvalScheduler` into the trainer with the three baseline
  benchmark families (vs random / vs heuristic / vs pool) and an Elo tracker
  whose ratings are persisted next to the TB log.
* Switches the rollout collector to :class:`SubprocVecEnv` when ``--num-envs >
  1`` using ``rl.training.vec_factories.random_opponents_factory``.

Outputs (under ``--output-dir``, default ``runs/<timestamp>``):

* ``tb/`` — TensorBoard event files.
* ``snapshots/`` — periodic ``.pt`` checkpoints.
* ``elo.json`` — Elo trajectory across evals.
* ``final.pt`` — final checkpoint written on clean exit.

Usage::

    python scripts/train.py --total-steps 1_000_000 --num-envs 4 \\
        --output-dir runs/exp1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.env.rewards import RewardFn, ShapedReward, SparseWinReward
from rl.evaluation.elo import EloTracker
from rl.evaluation.scheduler import (
    EvalScheduler,
    make_bench_vs_heuristic,
    make_bench_vs_pool,
    make_bench_vs_random,
)
from rl.models.mlp import MLPPolicyValue
from rl.training.checkpoint import CheckpointMeta, load_checkpoint
from rl.training.config import DEFAULT_BASELINE_CONFIG, PPOConfig, TrainConfig
from rl.training.opponent_pool import OpponentPool
from rl.training.trainer import Trainer
from rl.training.vec_factories import (
    heuristic_opponents_factory,
    random_opponents_factory,
)


_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]


def _make_reward_fn(kind: str, stalemate_penalty: float) -> RewardFn:
    if kind == "shaped":
        return ShapedReward(stalemate_penalty=stalemate_penalty)
    return SparseWinReward()


def _make_env_factory(
    reward_kind: str, stalemate_penalty: float
) -> Callable[[int], CatanEnv]:
    """Build a seed→CatanEnv factory closing over the chosen reward kind."""
    def factory(seed: int) -> CatanEnv:
        return CatanEnv(
            seed=seed,
            reward_fn=_make_reward_fn(reward_kind, stalemate_penalty),
        )
    return factory


def _parse_hidden(spec: str) -> tuple[int, ...]:
    """Parse ``"512,512,512"`` into ``(512, 512, 512)``."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"empty hidden-sizes spec: {spec!r}")
    return tuple(int(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/train.py",
        description="Train the PPO learner against a self-play pool.",
    )

    # Training duration / determinism.
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)

    # Run output.
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="root for tb/, snapshots/, elo.json. Default: runs/<timestamp>",
    )

    # Warm-start.
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="path to a prior checkpoint (.pt). Copies model weights into the "
             "freshly-constructed learner; optimizer, global_step, OpponentPool, "
             "and EvalScheduler all start fresh. The checkpoint's hidden-sizes "
             "override --hidden-sizes (with a heads-up if they differ).",
    )

    # PPO update hyperparameters.
    p.add_argument("--lr", type=float, default=DEFAULT_BASELINE_CONFIG.ppo.lr)
    p.add_argument(
        "--clip-range", type=float, default=DEFAULT_BASELINE_CONFIG.ppo.clip_range
    )
    p.add_argument(
        "--value-coef", type=float, default=DEFAULT_BASELINE_CONFIG.ppo.value_coef
    )
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=DEFAULT_BASELINE_CONFIG.ppo.entropy_coef,
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_BASELINE_CONFIG.ppo.max_grad_norm,
    )
    p.add_argument(
        "--n-epochs", type=int, default=DEFAULT_BASELINE_CONFIG.ppo.n_epochs
    )
    p.add_argument(
        "--minibatch-size",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.ppo.minibatch_size,
    )
    p.add_argument(
        "--target-kl",
        type=float,
        default=DEFAULT_BASELINE_CONFIG.ppo.target_kl,
        help="set <= 0 to disable early-stop",
    )

    # Trainer orchestration.
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.rollout_steps,
    )
    p.add_argument("--gamma", type=float, default=DEFAULT_BASELINE_CONFIG.gamma)
    p.add_argument(
        "--gae-lambda", type=float, default=DEFAULT_BASELINE_CONFIG.gae_lambda
    )
    p.add_argument(
        "--hidden-sizes",
        type=_parse_hidden,
        default=DEFAULT_BASELINE_CONFIG.hidden_sizes,
        help="comma-separated MLP widths, e.g. 512,512,512",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="parallel rollout subprocesses. >1 enables SubprocVecEnv path.",
    )
    p.add_argument(
        "--vec-opponents",
        choices=("random", "heuristic"),
        default="random",
        help="opponent setup baked into each vec worker (ignored if num-envs==1)",
    )

    # Snapshots / self-play pool.
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.snapshot_every,
    )
    p.add_argument(
        "--promote-every-n-snapshots",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.promote_every_n_snapshots,
    )

    # Eval.
    p.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.eval_every,
    )
    p.add_argument(
        "--eval-games",
        type=int,
        default=DEFAULT_BASELINE_CONFIG.eval_n_games,
    )
    p.add_argument(
        "--no-eval-heuristic",
        action="store_true",
        help="skip the vs-heuristic benchmark (full games are ~3× slower than vs-random)",
    )
    p.add_argument(
        "--no-eval-pool",
        action="store_true",
        help="skip the vs-pool benchmark",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="optional replay archive root (off by default)",
    )

    # Reward shaping.
    p.add_argument(
        "--reward",
        choices=("sparse", "shaped"),
        default="sparse",
        help="reward function for training and eval envs. sparse=±1 at terminal; "
             "shaped adds vp_coef × Δvp per step + small turn penalty.",
    )
    p.add_argument(
        "--stalemate-penalty",
        type=float,
        default=0.5,
        help="ShapedReward only: per-seat penalty when the game stalemates. "
             "Keeps stalling from beating losing on the per-episode return; "
             "set to 0 to reproduce the pre-penalty contract.",
    )

    # Heartbeat / watchdog.
    p.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="iterations between stdout heartbeat lines (0 to disable)",
    )
    p.add_argument(
        "--watchdog-zero-wins-iters",
        type=int,
        default=0,
        help="abort if rollout/wins is 0 for this many consecutive iterations "
             "(0 = no watchdog). With cfg.rollout_steps=2048, 50 ≈ 100k env steps.",
    )

    # Pool baseline seeding.
    p.add_argument(
        "--pool-baselines",
        type=int,
        default=0,
        help="seed the OpponentPool with this many HeuristicAgent instances. "
             "Only active for --num-envs 1 (vec mode uses module-level factories).",
    )
    p.add_argument(
        "--baseline-weight",
        type=float,
        default=0.4,
        help="OpponentPool mix weight for the baseline bucket (default 0.4). "
             "Ignored if --pool-baselines == 0.",
    )

    return p


def _build_config(args: argparse.Namespace) -> TrainConfig:
    target_kl = args.target_kl if args.target_kl > 0 else None
    return TrainConfig(
        ppo=PPOConfig(
            lr=args.lr,
            clip_range=args.clip_range,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            n_epochs=args.n_epochs,
            minibatch_size=args.minibatch_size,
            target_kl=target_kl,
        ),
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        hidden_sizes=tuple(args.hidden_sizes),
        eval_every=args.eval_every,
        eval_n_games=args.eval_games,
        snapshot_every=args.snapshot_every,
        promote_every_n_snapshots=args.promote_every_n_snapshots,
        log_every=1,
        print_every=args.print_every,
        watchdog_zero_wins_iters=args.watchdog_zero_wins_iters,
        seed=args.seed,
        num_envs=args.num_envs,
    )


def _resolve_warm_start(
    args: argparse.Namespace,
) -> tuple[PolicyAgent | None, CheckpointMeta | None]:
    """Load the ``--init-from`` checkpoint, if any, and log the inheritance.

    Returns ``(agent, meta)`` on success or ``(None, None)`` when no
    warm-start was requested. The agent is held only to forward its
    ``state_dict`` into the freshly-constructed learner below — its own
    encoder, device, and life-cycle are otherwise discarded.
    """
    if args.init_from is None:
        return None, None
    path = Path(args.init_from)
    agent, meta = load_checkpoint(path)
    ckpt_hidden = meta.model_arch.hidden
    requested_hidden = tuple(args.hidden_sizes)
    if requested_hidden != ckpt_hidden:
        # The checkpoint's architecture is load-bearing for state_dict shape
        # match. Override silently but tell the user we did so.
        print(
            f"[train] --init-from: overriding --hidden-sizes "
            f"{requested_hidden} with checkpoint architecture {ckpt_hidden}",
            flush=True,
        )
    print(
        f"[train] warm-starting from {path} "
        f"(train_step={meta.train_step}, hidden={ckpt_hidden})",
        flush=True,
    )
    return agent, meta


def _build_pool(args: argparse.Namespace) -> OpponentPool:
    """Construct the OpponentPool, optionally seeded with heuristic baselines."""
    rng = random.Random(args.seed)
    baselines: list[Agent] = []
    baseline_weight = 0.0
    if args.pool_baselines > 0:
        baselines = [HeuristicAgent() for _ in range(args.pool_baselines)]
        baseline_weight = args.baseline_weight
        if args.num_envs > 1:
            # Vec workers run in subprocesses and can't reach the in-main-process
            # pool. Warning rather than error because the user might still want
            # the eval-side vs_pool benchmark to see baselines.
            print(
                "[train] warning: --pool-baselines set but --num-envs > 1; "
                "vec rollout workers use module-level factories and won't "
                "consult the pool. Baselines will only affect eval.",
                file=sys.stderr,
            )
    return OpponentPool(
        baselines=baselines,
        baseline_weight=baseline_weight,
        rng=rng,
    )


def _build_scheduler(
    trainer: Trainer,
    pool: OpponentPool,
    env_factory: Callable[[int], CatanEnv],
    output_dir: Path,
    args: argparse.Namespace,
) -> EvalScheduler:
    benchmarks = {
        "vs_random": make_bench_vs_random(
            env_factory=env_factory,
            n_games=args.eval_games,
        ),
    }
    if not args.no_eval_heuristic:
        benchmarks["vs_heuristic"] = make_bench_vs_heuristic(
            env_factory=env_factory,
            n_games=args.eval_games,
        )
    if not args.no_eval_pool:
        benchmarks["vs_pool"] = make_bench_vs_pool(
            env_factory=env_factory,
            pool=pool,
            n_games=args.eval_games,
        )
    return EvalScheduler(
        trainer=trainer,
        every_steps=args.eval_every,
        benchmarks=benchmarks,
        elo=EloTracker(),
        ratings_path=output_dir / "elo.json",
        archive_root=args.archive_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.total_steps <= 0:
        print(f"error: --total-steps must be positive (got {args.total_steps})", file=sys.stderr)
        return 2
    if args.num_envs <= 0:
        print(f"error: --num-envs must be positive (got {args.num_envs})", file=sys.stderr)
        return 2

    output_dir = args.output_dir or Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "tb"
    snapshot_dir = output_dir / "snapshots"

    if args.stalemate_penalty < 0:
        print(
            f"error: --stalemate-penalty must be non-negative (got {args.stalemate_penalty})",
            file=sys.stderr,
        )
        return 2

    if args.init_from is not None and not Path(args.init_from).is_file():
        print(
            f"error: --init-from path does not exist: {args.init_from}",
            file=sys.stderr,
        )
        return 2
    init_agent, init_meta = _resolve_warm_start(args)
    if init_meta is not None:
        args.hidden_sizes = init_meta.model_arch.hidden

    cfg = _build_config(args)
    (output_dir / "config.json").write_text(
        json.dumps(
            _config_to_jsonable(
                cfg, args.reward, args.stalemate_penalty, args.init_from
            ),
            indent=2,
        )
    )
    print(f"[train] output_dir={output_dir}", flush=True)
    print(
        f"[train] cfg={cfg} reward={args.reward} "
        f"stalemate_penalty={args.stalemate_penalty}",
        flush=True,
    )

    # Subprocess vec workers can't accept closure arguments, so the reward
    # kind and the stalemate-penalty knob both ride through env vars.
    # Setting them before SubprocVecEnv.spawn() lets workers pick up the
    # same choices as the main-process env factory below.
    os.environ["CATAN_RL_REWARD"] = args.reward
    os.environ["CATAN_RL_STALEMATE_PENALTY"] = str(args.stalemate_penalty)

    torch.manual_seed(cfg.seed)
    model = MLPPolicyValue(
        obs_dim=OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        hidden=cfg.hidden_sizes,
    )
    learner = PolicyAgent(model, ActionEncoder(_PLAYER_IDS))
    if init_agent is not None:
        # Warm-start: only the model weights carry over. The Trainer below
        # builds a fresh Adam and starts at global_step=0, and OpponentPool /
        # EvalScheduler are constructed below — all of those state machines
        # restart from scratch by construction, not by an explicit reset here.
        learner.load_state_dict(init_agent.state_dict())
    pool = _build_pool(args)
    env_factory = _make_env_factory(args.reward, args.stalemate_penalty)

    vec_factory = None
    if cfg.num_envs > 1:
        vec_factory = (
            heuristic_opponents_factory
            if args.vec_opponents == "heuristic"
            else random_opponents_factory
        )

    trainer = Trainer(
        env_factory=env_factory,
        learner=learner,
        opponent_pool=pool,
        cfg=cfg,
        log_dir=log_dir,
        snapshot_dir=snapshot_dir,
        vec_factory=vec_factory,
    )
    trainer.eval_scheduler = _build_scheduler(
        trainer, pool, env_factory, output_dir, args
    )

    t0 = time.time()
    try:
        trainer.train(total_steps=args.total_steps)
    except KeyboardInterrupt:
        print("\n[train] KeyboardInterrupt — saving final checkpoint and exiting", flush=True)
    finally:
        trainer.save_checkpoint(output_dir / "final.pt")
    dt = time.time() - t0
    steps_per_s = trainer.global_step / dt if dt > 0 else float("inf")
    print(
        f"[train] done: {trainer.global_step} steps in {dt:.1f}s "
        f"({steps_per_s:.1f} steps/s) → {output_dir / 'final.pt'}",
        flush=True,
    )
    return 0


def _config_to_jsonable(
    cfg: TrainConfig,
    reward_kind: str,
    stalemate_penalty: float,
    init_from: Path | None,
) -> dict:
    """Flatten ``TrainConfig`` to a JSON-friendly dict for the run record."""
    return {
        "ppo": {
            "lr": cfg.ppo.lr,
            "clip_range": cfg.ppo.clip_range,
            "value_coef": cfg.ppo.value_coef,
            "entropy_coef": cfg.ppo.entropy_coef,
            "max_grad_norm": cfg.ppo.max_grad_norm,
            "n_epochs": cfg.ppo.n_epochs,
            "minibatch_size": cfg.ppo.minibatch_size,
            "target_kl": cfg.ppo.target_kl,
        },
        "rollout_steps": cfg.rollout_steps,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "hidden_sizes": list(cfg.hidden_sizes),
        "eval_every": cfg.eval_every,
        "eval_n_games": cfg.eval_n_games,
        "snapshot_every": cfg.snapshot_every,
        "promote_every_n_snapshots": cfg.promote_every_n_snapshots,
        "log_every": cfg.log_every,
        "print_every": cfg.print_every,
        "watchdog_zero_wins_iters": cfg.watchdog_zero_wins_iters,
        "seed": cfg.seed,
        "num_envs": cfg.num_envs,
        "reward": reward_kind,
        "stalemate_penalty": stalemate_penalty,
        "init_from": str(init_from) if init_from is not None else None,
    }


if __name__ == "__main__":
    sys.exit(main())
