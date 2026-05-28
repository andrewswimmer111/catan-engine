#!/usr/bin/env python3
"""Heuristic behavioural-cloning pretrain driver.

Rolls out expert games with the rule-based
:class:`rl.agents.heuristic_agent.HeuristicAgent` in all four seats,
clones its decisions into a fresh GNN policy/value network (masked policy
cross-entropy + per-seat value MSE), and writes a checkpoint that
``train_alphazero.py --init-from`` consumes verbatim — the learner is a
graph-encoder, vector-value-head model, exactly the AZ shape.

This is the Phase-0 pretrain the AZ plan originally skipped. The #015
self-play diagnostic localised the residual AZ plateau to a
settlement-EXPANSION ceiling; the heuristic does not have that problem, so
cloning it seeds AZ with the road→settlement build chain. See
:mod:`rl.training.bc_pretrain` for the full rationale and warts.

Outputs (under ``--output-dir``, default ``runs/bc_<timestamp>``):

* ``config.json`` — pinned CLI flags + resolved hyperparameters.
* ``progress.md`` — human-readable per-epoch table, rewritten each epoch.
* ``final.pt`` — the cloned checkpoint (AZ ``--init-from`` compatible).

Usage::

    # Clone the heuristic at the current vp=6 curriculum, then continue AZ:
    venv/bin/python scripts/pretrain_bc.py --win-vp 6 --n-games 400 \\
        --epochs 15 --device mps --output-dir runs/bc_001
    venv/bin/python scripts/train_alphazero.py --total-iters 15 --win-vp 6 \\
        --init-from runs/bc_001/final.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.action import ActionEncoder
from rl.encoding.graph_observation import GRAPH_OBS_SHAPE, GraphObservationEncoder
from rl.models.gnn import DEFAULT_GNN_ARCH, GNNPolicyValue
from rl.stalemate_value import StalemateValueConfig
from rl.training.bc_pretrain import BCConfig, generate_bc_transitions, train_bc
from rl.training.checkpoint import (
    ACTION_LAYOUT_VERSION,
    CheckpointMeta,
    model_arch_from,
    obs_layout_version_for,
    save_checkpoint,
)
from rl.utils.device import resolve_device

_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree. Exposed for testing the flag schema."""
    p = argparse.ArgumentParser(
        prog="pretrain_bc",
        description="Heuristic behavioural-cloning pretrain driver.",
    )
    defaults = BCConfig()

    # Data generation.
    p.add_argument(
        "--n-games",
        type=int,
        default=defaults.n_games,
        help="heuristic self-play games to clone from (all four seats are "
        "the heuristic; variety comes from per-game board/dice seeds)",
    )
    p.add_argument(
        "--win-vp",
        type=int,
        default=defaults.victory_point_target,
        help="victory-point threshold for ending a game in a real win "
        "(default 10 — standard Catan). MATCH this to the downstream AZ "
        "run's --win-vp (e.g. 6) so the cloned value head is calibrated "
        "for the same threshold; at 10 the heuristic stalemates ~99%% of "
        "games and the value target collapses into the stalemate band "
        "(the policy clone is unaffected either way).",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=defaults.max_moves,
        help="hard per-game move cap; reaching it truncates as a stalemate",
    )
    stale_defaults = StalemateValueConfig()
    p.add_argument(
        "--stalemate-shape",
        choices=("flat", "vp_linear"),
        default=stale_defaults.shape,
        help="per-seat stalemate value-target shape (kept identical to the "
        "AZ default so warm-started value targets match)",
    )
    p.add_argument(
        "--stalemate-flat-value", type=float, default=stale_defaults.flat_value
    )
    p.add_argument("--stalemate-low", type=float, default=stale_defaults.low)
    p.add_argument("--stalemate-high", type=float, default=stale_defaults.high)

    # Optimisation.
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument(
        "--value-coef",
        type=float,
        default=defaults.value_coef,
        help="weight on the per-seat value MSE; 0.0 = policy-only clone",
    )
    p.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)

    # I/O.
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="torch device for the learner model (generation is CPU-only)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="run directory; defaults to runs/bc_<timestamp>",
    )
    p.add_argument("--seed", type=int, default=defaults.seed)
    return p


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> BCConfig:
    stalemate = StalemateValueConfig(
        shape=args.stalemate_shape,
        flat_value=args.stalemate_flat_value,
        low=args.stalemate_low,
        high=args.stalemate_high,
    )
    return BCConfig(
        n_games=args.n_games,
        victory_point_target=args.win_vp,
        max_moves=args.max_moves,
        stalemate=stalemate,
        player_ids=_PLAYER_IDS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )


def _build_learner(device: torch.device) -> PolicyAgent:
    """Construct a fresh GNN learner with the AZ-shape (vector) value head."""
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=DEFAULT_GNN_ARCH,  # value_kind="vector" by default
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(list(_PLAYER_IDS)),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
        device=device,
    )


def _config_to_jsonable(cfg: BCConfig, args: argparse.Namespace) -> dict:
    return {
        "bc": {
            "n_games": cfg.n_games,
            "victory_point_target": cfg.victory_point_target,
            "max_moves": cfg.max_moves,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "value_coef": cfg.value_coef,
            "max_grad_norm": cfg.max_grad_norm,
            "seed": cfg.seed,
        },
        "stalemate": {
            "shape": cfg.stalemate.shape,
            "flat_value": cfg.stalemate.flat_value,
            "low": cfg.stalemate.low,
            "high": cfg.stalemate.high,
        },
        "cli": {"device": args.device},
    }


def _build_meta(agent: PolicyAgent, train_step: int) -> CheckpointMeta:
    arch = model_arch_from(agent.model)
    return CheckpointMeta(
        obs_layout_version=obs_layout_version_for(arch.encoder_kind),
        action_layout_version=ACTION_LAYOUT_VERSION,
        model_arch=arch,
        train_step=train_step,
        timestamp=time.time(),
        config_hash="bc_pretrain",  # BCConfig drifts independently of TrainConfig
    )


# ----------------------------------------------------------------------
# Progress file
# ----------------------------------------------------------------------


def _render_progress_md(
    cfg: BCConfig,
    args: argparse.Namespace,
    history: list[dict[str, float]],
    *,
    status: str,
    n_samples: int,
    start_time: float,
) -> str:
    elapsed = time.time() - start_time
    lines = ["# Heuristic BC pretrain progress", ""]
    lines.append(f"- **Status**: {status}")
    lines.append(f"- **Epoch**: {len(history)} / {cfg.epochs}")
    lines.append(f"- **Samples (expert decisions)**: {n_samples}")
    lines.append(f"- **Elapsed**: {elapsed:.1f}s")
    lines += ["", "## Config", ""]
    lines += [
        f"- device: `{args.device}`",
        f"- games: {cfg.n_games}",
        f"- win VP: {cfg.victory_point_target}",
        f"- max moves: {cfg.max_moves}",
        f"- epochs: {cfg.epochs}",
        f"- batch size: {cfg.batch_size}",
        f"- lr: {cfg.lr}",
        f"- weight decay (L2): {cfg.weight_decay}",
        f"- value coef: {cfg.value_coef}",
        f"- stalemate: {cfg.stalemate.shape}",
    ]
    lines += ["", "## Epochs", ""]
    cols = ["epoch", "pol_loss", "val_loss", "accuracy", "grad_norm"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for h in history:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{int(h['epoch'])}",
                    f"{h['policy_loss']:.3f}",
                    f"{h['value_loss']:.3f}",
                    f"{h['accuracy']:.3f}",
                    f"{h['grad_norm']:.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_games <= 0:
        print(f"error: --n-games must be positive (got {args.n_games})", file=sys.stderr)
        return 2
    if args.epochs <= 0:
        print(f"error: --epochs must be positive (got {args.epochs})", file=sys.stderr)
        return 2

    output_dir = args.output_dir or Path("runs") / f"bc_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.md"

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cfg = _build_config(args)
    (output_dir / "config.json").write_text(
        json.dumps(_config_to_jsonable(cfg, args), indent=2)
    )

    learner = _build_learner(device)

    print(f"[bc] output_dir={output_dir} device={device}", flush=True)
    print(
        f"[bc] generating {cfg.n_games} heuristic games "
        f"(win_vp={cfg.victory_point_target}, max_moves={cfg.max_moves})...",
        flush=True,
    )
    t_gen = time.time()
    transitions = generate_bc_transitions(cfg)
    print(
        f"[bc] generated {len(transitions)} expert decisions "
        f"in {time.time() - t_gen:.1f}s; training {cfg.epochs} epochs...",
        flush=True,
    )

    start_time = time.time()
    history: list[dict[str, float]] = []

    def _on_epoch_end(summary: dict[str, float]) -> None:
        history.append(summary)
        print(
            f"[bc] epoch {int(summary['epoch'])}/{cfg.epochs} "
            f"pol_loss={summary['policy_loss']:.3f} "
            f"val_loss={summary['value_loss']:.3f} "
            f"acc={summary['accuracy']:.3f}",
            flush=True,
        )
        progress_path.write_text(
            _render_progress_md(
                cfg,
                args,
                history,
                status="running",
                n_samples=len(transitions),
                start_time=start_time,
            )
        )

    completed = False
    try:
        summary = train_bc(learner, transitions, cfg, on_epoch_end=_on_epoch_end)
        completed = True
    finally:
        train_step = int(history[-1]["global_step"]) if history else 0
        save_checkpoint(learner, output_dir / "final.pt", _build_meta(learner, train_step))
        progress_path.write_text(
            _render_progress_md(
                cfg,
                args,
                history,
                status="done" if completed else "interrupted",
                n_samples=len(transitions),
                start_time=start_time,
            )
        )

    print(
        f"[bc] done: final accuracy={summary['accuracy']:.3f} "
        f"→ {output_dir / 'final.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
