"""Command-line entry point for the RL layer.

Subcommands:

* ``evaluate`` — head-to-head learner-vs-opponent tournament with seat
  rotation. Prints a one-page Markdown report (see :mod:`rl.evaluation.report`).
* ``benchmark`` — runs the three reproducible baseline benchmarks
  (random-vs-random, heuristic-vs-random, heuristic-vs-heuristic) and prints
  a compact table.
* ``train`` — invokes :class:`Trainer.train` with the baseline config. The CLI
  is intentionally thin here: real sweeps go through a config file or a
  scripted harness, not this entrypoint.
* ``play`` — plays a single game between two named agents and prints the
  final outcome (cheap smoke-test for shipping an agent without firing up
  pytest).
* ``replay`` — loads a :class:`ReplayLog` from disk, replays it, and prints
  its summary.

Argparse is used directly — no ``click``/``typer`` dependency.

Opponent specs
--------------

``--learner`` always names a checkpoint path. ``--opponent`` accepts a
checkpoint path *or* one of the literal strings ``"random"`` / ``"heuristic"``,
so the same CLI covers both checkpoint-vs-checkpoint and checkpoint-vs-baseline
comparisons.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

from controller.agents import Agent
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.hybrid_agent import PlacementOverrideAgent
from rl.agents.policy_agent import PolicyAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.search_agent import SearchAgent
from rl.search.mcts import MCTSConfig
from rl.env.catan_env import CatanEnv
from rl.evaluation.benchmarks import (
    bench_heuristic_vs_heuristic,
    bench_heuristic_vs_random,
    bench_random_vs_random,
)
from rl.evaluation.report import (
    RotationResult,
    aggregate_evaluation,
    format_evaluation_report,
)
from rl.evaluation.tournament import Tournament
from rl.training.checkpoint import load_checkpoint

__all__ = ["build_parser", "main"]


_PLAYER_IDS: list[PlayerID] = [PlayerID(i) for i in range(1, 5)]
_AgentFactory = Callable[[random.Random], Agent]


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse tree. Exposed for testing."""
    parser = argparse.ArgumentParser(prog="rl.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="learner vs opponent tournament")
    p_eval.add_argument("--learner", required=True, type=Path)
    p_eval.add_argument(
        "--opponent",
        required=True,
        help='checkpoint path or one of "random" / "heuristic"',
    )
    p_eval.add_argument("--games", type=int, default=100)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="if set, sample-game replays are written here",
    )
    p_eval.add_argument(
        "--placement-agent",
        choices=("self", "heuristic"),
        default="self",
        help="who plays the learner's opening placements (INITIAL_SETTLEMENT / "
             "INITIAL_ROAD phases). 'self' (default) uses the learner end-to-end; "
             "'heuristic' delegates only the opening to HeuristicAgent and the "
             "learner takes over from MAIN onwards. Diagnostic for isolating "
             "opening-quality bottlenecks from in-game play.",
    )
    p_eval.add_argument(
        "--use-mcts",
        action="store_true",
        help="wrap the learner with PUCT MCTS at inference (no retraining). "
             "Uses the policy's masked softmax as the prior and the value head "
             "as the leaf evaluator. Composes with --placement-agent.",
    )
    p_eval.add_argument(
        "--mcts-rollouts",
        type=int,
        default=100,
        help="MCTS rollouts per decision. Default 100. Ignored without --use-mcts.",
    )
    p_eval.add_argument(
        "--mcts-cpuct",
        type=float,
        default=2.0,
        help="PUCT exploration coefficient. Default 2.0. Ignored without --use-mcts.",
    )
    p_eval.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
        help="torch device for the learner (and any checkpoint opponent). "
             "Defaults to CPU; 'mps' enables Metal on macOS, 'cuda' on a "
             "CUDA GPU host.",
    )

    p_bench = sub.add_parser("benchmark", help="run all baseline benchmarks")
    p_bench.add_argument("--games", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=0)

    p_train = sub.add_parser("train", help="train the PPO learner")
    p_train.add_argument("--total-steps", type=int, default=10_000)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--log-dir", type=Path, default=None)
    p_train.add_argument("--snapshot-dir", type=Path, default=None)

    p_play = sub.add_parser("play", help="play one game between two agents")
    p_play.add_argument("--learner", required=True, type=Path)
    p_play.add_argument("--opponent", required=True)
    p_play.add_argument("--seed", type=int, default=0)

    p_replay = sub.add_parser("replay", help="replay a saved game log")
    p_replay.add_argument("path", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "play":
        return _cmd_play(args)
    if args.command == "replay":
        return _cmd_replay(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


# ----------------------------------------------------------------------
# Subcommand handlers
# ----------------------------------------------------------------------


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if args.games <= 0:
        print(f"error: --games must be positive (got {args.games})", file=sys.stderr)
        return 2
    device = _resolve_eval_device(args.device)
    learner_agent, learner_label = _load_learner(args.learner, device=device)
    if args.use_mcts:
        cfg = MCTSConfig(
            rollouts=args.mcts_rollouts,
            c_puct=args.mcts_cpuct,
            seed=args.seed,
        )
        learner_agent = SearchAgent(learner_agent, cfg)
        learner_label = (
            f"{learner_label}+mcts(r={args.mcts_rollouts},c={args.mcts_cpuct:g})"
        )
    if args.placement_agent == "heuristic":
        learner_agent = PlacementOverrideAgent(learner_agent, HeuristicAgent())
        learner_label = f"{learner_label}+heuristic_placement"
    opponent_factory, opponent_label = _resolve_opponent(args.opponent, device=device)
    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    rotations = _run_rotated_tournament(
        learner=learner_agent,
        opponent_factory=opponent_factory,
        n_games=args.games,
        base_seed=args.seed,
        replay_dir=output_dir,
    )
    comparison = aggregate_evaluation(
        learner_label=learner_label,
        opponent_label=opponent_label,
        rotations=rotations,
    )
    print(format_evaluation_report(comparison))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    n = args.games
    seed = args.seed
    results = {
        "random_vs_random": bench_random_vs_random(n_games=n, base_seed=seed),
        "heuristic_vs_random": bench_heuristic_vs_random(n_games=n, base_seed=seed),
        "heuristic_vs_heuristic": bench_heuristic_vs_heuristic(
            n_games=n, base_seed=seed
        ),
    }
    print("# Baseline benchmarks")
    print()
    print("| benchmark | mean turns | top win rate |")
    print("| --- | --- | --- |")
    for name, res in results.items():
        top = max(res.win_rates.values()) if res.win_rates else 0.0
        print(f"| {name} | {res.mean_turns:.1f} | {top:.3f} |")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    # Lazy imports so the CLI doesn't pull in torch when only running
    # evaluate/benchmark.
    from rl.training.config import DEFAULT_BASELINE_CONFIG
    from rl.training.opponent_pool import OpponentPool
    from rl.training.trainer import Trainer
    from rl.encoding.action import ACTION_SPACE_SIZE
    from rl.encoding.observation import OBS_SHAPE
    from rl.encoding.action import ActionEncoder
    from rl.models.mlp import MLPPolicyValue

    cfg = DEFAULT_BASELINE_CONFIG
    cfg.seed = args.seed

    model = MLPPolicyValue(
        obs_dim=OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        hidden=cfg.hidden_sizes,
    )
    learner = PolicyAgent(model, ActionEncoder(_PLAYER_IDS))
    pool = OpponentPool()
    trainer = Trainer(
        env_factory=_env_factory,
        learner=learner,
        opponent_pool=pool,
        cfg=cfg,
        log_dir=args.log_dir,
        snapshot_dir=args.snapshot_dir,
    )
    trainer.train(total_steps=args.total_steps)
    print(f"trained for {trainer.global_step} steps")
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    learner_agent, learner_label = _load_learner(args.learner)
    opponent_factory, opponent_label = _resolve_opponent(args.opponent)
    rng = random.Random(args.seed)
    learner_seat = _PLAYER_IDS[0]
    agents: dict[PlayerID, Agent] = {learner_seat: learner_agent}
    for pid in _PLAYER_IDS[1:]:
        agents[pid] = opponent_factory(random.Random(rng.randrange(2**32)))
    result = Tournament(_env_factory).play(agents, n_games=1, base_seed=args.seed)
    g = result.games[0]
    winner = "stalemate" if g.winner is None else f"P{int(g.winner)}"
    print(
        f"{learner_label} vs {opponent_label}: winner={winner} "
        f"turns={g.turn_count} end={g.end_reason.name}"
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from domain.engine.game_engine import GameEngine
    from domain.engine.randomizer import SeededRandomizer
    from serialization.replay import load_replay, replay_game

    log = load_replay(str(args.path))
    engine = GameEngine(SeededRandomizer(seed=log.config.seed))
    results = replay_game(log, engine)
    final = results[-1].state if results else None
    if final is None:
        print(f"empty replay at {args.path}")
        return 0
    winner = "stalemate" if final.winner is None else f"P{int(final.winner)}"
    print(
        f"replay {args.path}: {len(log.actions)} actions, winner={winner}"
    )
    return 0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _resolve_eval_device(spec: str):
    """Mirror of :func:`scripts.train._resolve_device` for the eval CLI."""
    import torch

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "error: --device cuda but torch.cuda.is_available()=False"
            )
        return torch.device("cuda")
    if spec == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise SystemExit(
                "error: --device mps but torch MPS backend is unavailable "
                "(macOS + Metal-supporting GPU required)"
            )
        return torch.device("mps")
    return torch.device("cpu")


def _load_learner(path: Path, *, device=None) -> tuple[PolicyAgent, str]:
    agent, _ = load_checkpoint(path, device=device or "cpu")
    return agent, f"checkpoint[{path.name}]"


def _resolve_opponent(spec: str, *, device=None) -> tuple[_AgentFactory, str]:
    """Turn an opponent spec into ``(factory, human_label)``.

    ``factory`` is called once per opponent seat per rotation with a fresh
    ``random.Random``. For checkpoint opponents the factory ignores its rng
    arg and returns a shared ``PolicyAgent`` instance — sharing weights is
    fine because :meth:`PolicyAgent.choose` is stateless. ``device``
    propagates to checkpoint opponents so all networks live on the same
    backend as the learner.
    """
    if spec == "random":
        return lambda rng: RandomAgent(rng, skip_proposals=True), "random"
    if spec == "heuristic":
        return lambda _rng: HeuristicAgent(), "heuristic"

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"opponent spec {spec!r} is neither 'random'/'heuristic' nor an existing path"
        )
    agent, _ = load_checkpoint(path, device=device or "cpu")
    # Opponent inference is stochastic — greedy opponents are easy to game.
    agent.stochastic_play = True

    def factory(_rng: random.Random) -> Agent:
        return agent

    return factory, f"checkpoint[{path.name}]"


def _run_rotated_tournament(
    *,
    learner: Agent,
    opponent_factory: _AgentFactory,
    n_games: int,
    base_seed: int,
    replay_dir: Path | None,
) -> list[RotationResult]:
    """Run ``n_games`` total games rotating the learner through every seat.

    Games are split as evenly as possible across the four rotations; if
    ``n_games`` isn't divisible by 4, the leftover rounds up to give the
    first few seats one extra game.

    Replays are written to ``replay_dir`` if provided — one JSON per game,
    keyed by ``f"r{rot_idx}_g{game_idx}.json"``. The paths are returned
    parallel to ``rotation.result.games`` so the report can reference them.
    """
    per_rotation = _split_evenly(n_games, k=len(_PLAYER_IDS))
    rotations: list[RotationResult] = []
    cursor_seed = base_seed
    rng = random.Random(base_seed)

    for rot_idx, learner_seat in enumerate(_PLAYER_IDS):
        n_round = per_rotation[rot_idx]
        if n_round == 0:
            continue
        agents: dict[PlayerID, Agent] = {learner_seat: learner}
        for pid in _PLAYER_IDS:
            if pid != learner_seat:
                agents[pid] = opponent_factory(random.Random(rng.randrange(2**32)))

        if replay_dir is not None:
            paths, result = _play_with_replay(
                agents=agents,
                n_games=n_round,
                base_seed=cursor_seed,
                replay_dir=replay_dir,
                prefix=f"r{rot_idx}",
            )
        else:
            result = Tournament(_env_factory).play(
                agents, n_games=n_round, base_seed=cursor_seed
            )
            paths = [None] * len(result.games)

        rotations.append(RotationResult(
            learner_seat=learner_seat,
            result=result,
            replay_paths=paths,
        ))
        cursor_seed += n_round

    return rotations


def _split_evenly(n: int, k: int) -> list[int]:
    """Return ``k`` non-negative ints summing to ``n``, max-min ≤ 1."""
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def _play_with_replay(
    *,
    agents: dict[PlayerID, Agent],
    n_games: int,
    base_seed: int,
    replay_dir: Path,
    prefix: str,
) -> tuple[list[Path | None], "TournamentResult"]:  # noqa: F821
    """Tournament play that side-channels per-game replays to disk.

    Plays each game once on a single ``CatanEnv`` and reconstructs a
    :class:`ReplayLog` from the action stream we fed in plus the events the
    env returned. That lets the GUI's existing replay loader pick the JSON
    back up without us standing up a parallel ``GameSession``.
    """
    from collections import defaultdict

    from controller.session import GameSnapshot
    from domain.enums import EndReason
    from domain.rules.victory import compute_victory_points
    from rl.evaluation.metrics import GameStats
    from rl.evaluation.tournament import aggregate_games
    from serialization.codec import encode_action, encode_event
    from serialization.replay import ReplayLog, save_replay

    games: list[GameStats] = []
    replay_paths: list[Path | None] = []
    player_ids: list[PlayerID] = []

    for i in range(n_games):
        seed = base_seed + i
        env = _env_factory(seed)
        player_ids = list(env.state.config.player_ids)
        config = env.state.config

        action_counts: dict[str, int] = defaultdict(int)
        per_seat_counts: dict[PlayerID, dict[str, int]] = {
            pid: defaultdict(int) for pid in player_ids
        }
        actions_encoded: list[dict] = []
        events_encoded: list[list[dict]] = []
        step_index = 0
        snap = GameSnapshot(state=env.state, step_index=0, last_action=None, last_events=())

        done = False
        while not done:
            legal = env.legal_actions()
            acting = env.current_agent
            action = agents[acting].choose(snap, legal)
            if action is None:
                break
            name = type(action).__name__
            action_counts[name] += 1
            per_seat_counts[acting][name] += 1
            actions_encoded.append(encode_action(action))
            _, _, done, info = env.step(action)
            events_encoded.append([encode_event(e) for e in info["last_events"]])
            step_index += 1
            snap = GameSnapshot(
                state=env.state,
                step_index=step_index,
                last_action=action,
                last_events=tuple(info["last_events"]),
            )

        state = env.state
        games.append(GameStats(
            winner=state.winner,
            final_vps={
                pid: compute_victory_points(state, pid) for pid in player_ids
            },
            turn_count=state.turn_number,
            end_reason=state.end_reason or EndReason.STALEMATE_NO_PROGRESS,
            action_histogram=dict(action_counts),
            per_seat_action_histogram={
                pid: dict(per_seat_counts[pid]) for pid in per_seat_counts
            },
        ))

        path = replay_dir / f"{prefix}_g{i}.json"
        save_replay(
            ReplayLog(config=config, actions=actions_encoded, events=events_encoded),
            str(path),
        )
        replay_paths.append(path)

    result = aggregate_games(games, player_ids)
    return replay_paths, result


if __name__ == "__main__":
    sys.exit(main())
