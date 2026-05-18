#!/usr/bin/env python3
"""Measurement of the MCTS per-rollout hotspots on the current checkpoint.

Reports per-call latency for the operations MCTS will pay on every
expansion step:

* ``engine.apply_action(state, action)`` — one deepcopy + rule transition;
  paid once per leaf expansion.
* ``policy.act_with_dist(obs, mask)`` — one GNN forward; gives the policy
  prior and the value head for the leaf's acting player. Paid once per
  new tree node (prior) and once per leaf (value).
* 4-batched GNN forward — simulates the per-player vector backup:
  evaluate the leaf from each seat's perspective in a single batched
  forward call.

Re-run after architectural changes (MPS, smaller GNN, batched leaf
evaluation across rollouts) to re-project Phase 2 eval wall-time.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from domain.actions.base import Action  # noqa: E402
from domain.engine.game_engine import GameEngine  # noqa: E402
from domain.engine.player_view import make_player_view  # noqa: E402
from domain.engine.randomizer import SeededRandomizer  # noqa: E402
from domain.game.config import GameConfig  # noqa: E402
from domain.ids import PlayerID  # noqa: E402
from rl.acting_player import acting_player  # noqa: E402
from rl.training.checkpoint import load_checkpoint  # noqa: E402


def _time_calls(label: str, fn, *, n: int) -> float:
    """Time ``fn()`` ``n`` times; return microseconds per call."""
    # Warm-up: torch JITs the first forward and the OS warms caches.
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    us_per_call = dt / n * 1e6
    print(f"  {label:<48s} {us_per_call:>8.1f} µs/call   ({n} calls in {dt:.3f}s)")
    return us_per_call


def _advance_to_clean_mid_state(engine: GameEngine, state, action_encoder, rng_seed: int):
    """Walk forward until we land on a non-terminal state with a non-empty mask.

    Random-action walks can leave the engine in DISCARD or domestic-trade
    states whose mask is all-False (the encoder declares those off-discrete);
    those aren't representative of MCTS expansion. Keep walking until we
    find a clean state.
    """
    rng = np.random.default_rng(rng_seed)
    s = state
    for _ in range(500):
        if s.is_terminal():
            break
        legal = engine.legal_actions(s)
        if not legal:
            break
        mask = action_encoder.mask(legal)
        if mask.any() and s.phase.value == "main":
            return s
        action = legal[int(rng.integers(0, len(legal)))]
        s = engine.apply_action(s, action).state
    raise RuntimeError("could not find a clean MAIN-phase mid-state in 500 steps")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/overnight_20260515_2006/final.pt"),
        help="checkpoint to measure against (default: run #5).",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="torch device for the loaded model. Use 'mps' to re-project "
             "Phase-3 wall-times under Metal.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    ckpt_path: Path = args.checkpoint
    if not ckpt_path.is_file():
        print(f"error: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[measure] loading {ckpt_path} (device={args.device})", flush=True)
    agent, meta = load_checkpoint(ckpt_path, device=args.device)
    print(
        f"[measure] encoder={meta.model_arch.encoder_kind} "
        f"train_step={meta.train_step}",
        flush=True,
    )
    agent.model.eval()
    obs_encoder = agent.obs_encoder
    action_encoder = agent.action_encoder

    pids = [PlayerID(i) for i in range(1, 5)]
    cfg = GameConfig(player_ids=pids, seed=0)
    engine = GameEngine(SeededRandomizer(0))
    initial_state = engine.new_game(cfg)
    mid_state = _advance_to_clean_mid_state(
        engine, initial_state, action_encoder, rng_seed=42
    )
    print(
        f"[measure] mid-state phase={mid_state.phase.value} "
        f"turn={mid_state.turn_number}",
        flush=True,
    )

    print("\n[measure] copy.deepcopy(GameState):")
    deepcopy_us_initial = _time_calls(
        "initial state", lambda: copy.deepcopy(initial_state), n=500
    )
    deepcopy_us_mid = _time_calls(
        "mid-game state", lambda: copy.deepcopy(mid_state), n=500
    )

    print("\n[measure] engine.apply_action (deepcopy + transition):")
    initial_legal = engine.legal_actions(initial_state)
    mid_legal = engine.legal_actions(mid_state)
    apply_initial_us = _time_calls(
        "apply on initial state",
        lambda: engine.apply_action(initial_state, initial_legal[0]),
        n=500,
    )
    apply_mid_us = _time_calls(
        "apply on mid-game state",
        lambda: engine.apply_action(mid_state, mid_legal[0]),
        n=500,
    )

    print("\n[measure] policy.act_with_dist (single forward, gives logits+value):")
    acting = acting_player(mid_state)
    view = make_player_view(mid_state, acting)
    obs = obs_encoder.encode(view)
    mask = action_encoder.mask(mid_legal)
    fwd_us = _time_calls(
        "single forward", lambda: agent.act_with_dist(obs, mask), n=200
    )

    print("\n[measure] batched 4-perspective forward (per-player value vector):")
    obs_batch = np.stack(
        [obs_encoder.encode(make_player_view(mid_state, p)) for p in pids]
    )
    mask_batch = np.tile(mask, (4, 1))
    device = agent.device
    fwd4_us = _time_calls(
        "batched 4-way forward",
        lambda: _run_batched_forward(agent.model, obs_batch, mask_batch, device=device),
        n=200,
    )

    print("\n[measure] === per-rollout cost ===")
    # Per rollout: 1 apply_action (leaf expansion) + 1 batched 4-way forward
    # (leaf value evaluation, 4 perspectives). Per-rollout descent through
    # existing tree is dict lookups + PUCT score arithmetic; negligible vs
    # apply + forward.
    per_rollout_us = apply_mid_us + fwd4_us
    print(f"  apply_action + 4-way forward     {per_rollout_us:>8.1f} µs/rollout")

    print("\n[measure] === wall-time projection ===")
    # Eval workload: 200 games × 32 learner moves × 100 rollouts (default).
    rollouts = 100
    learner_moves_per_game = 32
    games = 200
    total_rollouts = rollouts * learner_moves_per_game * games
    eval_seconds = total_rollouts * per_rollout_us / 1e6
    print(
        f"  {rollouts} rollouts/move × {learner_moves_per_game} learner-moves/game × "
        f"{games} games"
    )
    print(
        f"  = {total_rollouts:,} rollouts × {per_rollout_us:.0f} µs "
        f"= {eval_seconds:.0f}s "
        f"= {eval_seconds / 60:.1f} min "
        f"= {eval_seconds / 3600:.2f} h"
    )
    # Plus opponent moves (~96/game): naked policy forward each.
    opp_moves_per_game = 96
    opp_seconds = opp_moves_per_game * games * fwd_us / 1e6
    print(
        f"  opponent overhead: {opp_moves_per_game}/game × {games} games × "
        f"{fwd_us:.0f} µs = {opp_seconds:.0f}s ({opp_seconds / 60:.1f} min)"
    )
    total = eval_seconds + opp_seconds
    print(f"  total projected eval wall-time:  {total / 3600:.2f} h")


def _run_batched_forward(
    model, obs: np.ndarray, mask: np.ndarray, *, device: torch.device
) -> None:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device)
    with torch.no_grad():
        model(obs_t, mask_t)


if __name__ == "__main__":
    main()
