#!/usr/bin/env python3
"""Diagnostic: play N self-play games at a checkpoint and report what happened.

Use this to discriminate between two plateau failure modes that look
the same in the AZ progress table:

* "Bad tactics" — policy builds toward 6 VP but in scattered locations,
  loses to opponents who out-build, then stalemates. More MCTS rollouts
  may help.
* "Bad prior" — policy spends moves on trades / dev cards / robber
  actions rather than building. More MCTS won't fix this; needs prior
  changes (Dirichlet, temperature schedule) or BC pretrain.

Reads the run dir's ``config.json`` so self-play knobs match training
exactly. Prints per-game summary + aggregate action-category histogram.

Usage::

    venv/bin/python scripts/diagnose_self_play.py \\
        runs/az_015_consolidate_20260526_173234 [--n-games 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections import Counter
from pathlib import Path

from domain.engine.game_engine import GameEngine
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from domain.rules.victory import compute_victory_points
from rl.acting_player import acting_player
from rl.agents.search_agent import NetworkEvaluator
from rl.search.mcts import MCTSConfig, run_mcts
from rl.stalemate_value import StalemateValueConfig
from rl.training.checkpoint import load_checkpoint


# Coarse-grained action categories. Class names are matched as substrings
# so we don't have to import every concrete Action subclass here.
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("placement", ("PlaceSettlement", "PlaceRoad")),
    ("build", ("BuildSettlement", "BuildRoad", "BuildCity")),
    ("dev_card", ("BuyDevCard", "PlayKnight", "PlayRoadBuilding",
                  "PlayYearOfPlenty", "PlayMonopoly")),
    ("trade", ("MaritimeTrade", "DomesticTrade")),
    ("robber", ("MoveRobber", "StealResource")),
    ("admin", ("RollDice", "EndTurn", "DiscardResources")),
)


def _categorise(action_cls_name: str) -> str:
    for category, needles in _CATEGORY_RULES:
        if any(n in action_cls_name for n in needles):
            return category
    return "other"


def _vp_snapshot(state, pids: list[PlayerID]) -> list[int]:
    return [compute_victory_points(state, pid) for pid in pids]


def _play_one_game(
    *, agent, sp_cfg: dict, mcts_cfg: dict, game_seed: int, rng: random.Random,
) -> dict:
    pids = [PlayerID(i) for i in range(1, 5)]
    engine = GameEngine(SeededRandomizer(game_seed))
    state = engine.new_game(
        GameConfig(
            player_ids=pids,
            seed=game_seed,
            victory_point_target=sp_cfg["victory_point_target"],
        )
    )

    agent.model.eval()
    evaluator = NetworkEvaluator(agent)

    base_mcts = MCTSConfig(
        rollouts=mcts_cfg["rollouts"],
        c_puct=mcts_cfg["c_puct"],
        seed=mcts_cfg["seed"],
        dirichlet_alpha=mcts_cfg["dirichlet_alpha"],
        dirichlet_epsilon=mcts_cfg["dirichlet_epsilon"],
        temperature=1.0,
        stalemate=StalemateValueConfig(
            shape=sp_cfg["stalemate"]["shape"],
            flat_value=sp_cfg["stalemate"]["flat_value"],
            low=sp_cfg["stalemate"]["low"],
            high=sp_cfg["stalemate"]["high"],
        ),
    )

    action_counts_per_player: dict[int, Counter] = {
        i: Counter() for i in range(len(pids))
    }
    vp_milestones: dict[int, dict[int, int]] = {
        i: {} for i in range(len(pids))
    }  # seat → {vp_level: move_idx}

    move_idx = 0
    n_forced = 0
    # Build-opportunity tracking, over non-forced decisions only. The
    # engine lists a Build* action as legal *only* when the player can
    # afford it and has a valid location, so "build was legal" is a
    # direct affordability/opportunity signal. Separating "build legal"
    # from "build chosen when legal" splits resource-starvation (build
    # rarely legal) from a prior problem (build legal but skipped).
    decision_points = 0
    build_opportunities = 0
    build_taken_when_available = 0
    while not state.is_terminal() and move_idx < sp_cfg["max_moves"]:
        legal = engine.legal_actions(state)
        if not legal:
            break
        acting_pid = acting_player(state)
        acting_idx = pids.index(acting_pid)

        if len(legal) == 1:
            chosen = legal[0]
            n_forced += 1
        else:
            move_cfg = dataclasses.replace(
                base_mcts, seed=base_mcts.seed + move_idx
            )
            result = run_mcts(state, evaluator, move_cfg, engine=engine)
            t_initial = sp_cfg["temperature_initial"]
            t_final = sp_cfg["temperature_final"]
            threshold = sp_cfg["temperature_threshold_moves"]
            temperature = t_initial if move_idx < threshold else t_final
            dist = result.policy_distribution(temperature)
            if temperature == 0.0:
                sample_idx = int(dist.argmax())
            else:
                r = rng.random()
                cum = 0.0
                sample_idx = len(dist) - 1
                for i, p in enumerate(dist):
                    cum += float(p)
                    if r < cum:
                        sample_idx = i
                        break
            chosen = result.legal_actions[sample_idx]

            decision_points += 1
            if any(_categorise(type(a).__name__) == "build" for a in legal):
                build_opportunities += 1
                if _categorise(type(chosen).__name__) == "build":
                    build_taken_when_available += 1

        action_counts_per_player[acting_idx][type(chosen).__name__] += 1

        state = engine.apply_action(state, chosen).state
        move_idx += 1

        # Snapshot VPs after the move, recording first move at which each
        # player reaches each VP level.
        for seat, vp in enumerate(_vp_snapshot(state, pids)):
            for level in (2, 3, 4, 5, 6, 7, 8, 9, 10):
                if vp >= level and level not in vp_milestones[seat]:
                    vp_milestones[seat][level] = move_idx

    winner_seat = None
    if state.winner is not None:
        winner_seat = pids.index(state.winner)
    end_reason = (
        state.end_reason.name if state.end_reason is not None else "max_moves"
    )

    return {
        "seed": game_seed,
        "n_moves": move_idx,
        "n_forced": n_forced,
        "winner_seat": winner_seat,
        "end_reason": end_reason,
        "final_vps": _vp_snapshot(state, pids),
        "action_counts_per_player": action_counts_per_player,
        "vp_milestones": vp_milestones,
        "decision_points": decision_points,
        "build_opportunities": build_opportunities,
        "build_taken_when_available": build_taken_when_available,
    }


def _summarise_game(g: dict) -> str:
    lines = []
    winner = (
        f"seat {g['winner_seat']}" if g["winner_seat"] is not None else "—"
    )
    lines.append(
        f"  seed={g['seed']} moves={g['n_moves']} (forced={g['n_forced']}) "
        f"winner={winner} end={g['end_reason']} "
        f"final_vps={g['final_vps']}"
    )

    # Aggregate per-game action category counts (across all 4 seats).
    cat = Counter()
    for seat_counter in g["action_counts_per_player"].values():
        for action_name, n in seat_counter.items():
            cat[_categorise(action_name)] += n
    total = sum(cat.values()) or 1
    cat_line = "  actions: " + "  ".join(
        f"{k}={v} ({100*v/total:.0f}%)"
        for k, v in sorted(cat.items(), key=lambda kv: -kv[1])
    )
    lines.append(cat_line)

    # VP timing: first move at which any player hit each level.
    earliest = {}
    for seat_ms in g["vp_milestones"].values():
        for level, m in seat_ms.items():
            if level not in earliest or m < earliest[level]:
                earliest[level] = m
    if earliest:
        ms_line = "  first-to-VP: " + " ".join(
            f"{lvl}@{earliest[lvl]}" for lvl in sorted(earliest)
        )
        lines.append(ms_line)

    dp = g["decision_points"] or 1
    bo = g["build_opportunities"]
    if bo:
        taken_rate = 100 * g["build_taken_when_available"] / bo
    else:
        taken_rate = 0.0
    lines.append(
        f"  build-opportunity: legal in {bo}/{g['decision_points']} "
        f"non-forced decisions ({100*bo/dp:.0f}%); "
        f"chosen {g['build_taken_when_available']}/{bo} when legal "
        f"({taken_rate:.0f}%)"
    )

    return "\n".join(lines)


def _summarise_aggregate(games: list[dict]) -> str:
    total_cat = Counter()
    total_actions_per_game = []
    total_builds_per_game = []
    n_wins = 0
    n_max_moves = 0
    avg_moves = 0
    final_vp_winners = []
    final_vp_losers = []
    for g in games:
        per_game = Counter()
        for seat_counter in g["action_counts_per_player"].values():
            for action_name, n in seat_counter.items():
                per_game[_categorise(action_name)] += n
                total_cat[_categorise(action_name)] += n
        total_actions_per_game.append(sum(per_game.values()))
        total_builds_per_game.append(per_game.get("build", 0))
        if g["winner_seat"] is not None:
            n_wins += 1
            final_vp_winners.append(g["final_vps"][g["winner_seat"]])
        else:
            final_vp_losers.extend(g["final_vps"])
        if g["end_reason"] == "max_moves":
            n_max_moves += 1
        avg_moves += g["n_moves"]
    avg_moves /= len(games)

    grand = sum(total_cat.values()) or 1
    lines = ["", "## Aggregate", ""]
    lines.append(
        f"  games={len(games)}  wins={n_wins}  "
        f"stalemates={len(games)-n_wins} (max_moves hits={n_max_moves})"
    )
    lines.append(f"  avg moves/game = {avg_moves:.1f}")
    if total_builds_per_game:
        avg_builds = sum(total_builds_per_game) / len(total_builds_per_game)
        lines.append(f"  avg build-actions/game (across all 4 seats) = {avg_builds:.1f}")
    lines.append("  action-category mix across all games:")
    for cat, n in sorted(total_cat.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {cat:12s} {n:5d}  ({100*n/grand:5.1f}%)")
    if final_vp_winners:
        lines.append(
            f"  winner final VPs: {final_vp_winners} "
            f"(avg {sum(final_vp_winners)/len(final_vp_winners):.2f})"
        )
    if final_vp_losers:
        lines.append(
            f"  stalemate final VPs (all seats, all stalemate games): "
            f"min={min(final_vp_losers)} max={max(final_vp_losers)} "
            f"avg={sum(final_vp_losers)/len(final_vp_losers):.2f}"
        )

    total_dp = sum(g["decision_points"] for g in games) or 1
    total_bo = sum(g["build_opportunities"] for g in games)
    total_bt = sum(g["build_taken_when_available"] for g in games)
    lines.append("")
    lines.append("  build-opportunity (the deciding signal):")
    lines.append(
        f"    build was LEGAL in {total_bo}/{total_dp} non-forced "
        f"decisions ({100*total_bo/total_dp:.1f}%)  "
        f"← low ⇒ resource starvation"
    )
    if total_bo:
        lines.append(
            f"    build CHOSEN when legal: {total_bt}/{total_bo} "
            f"({100*total_bt/total_bo:.1f}%)  ← low ⇒ prior under-weights building"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="diagnose_self_play")
    p.add_argument(
        "run_dir",
        type=Path,
        help="run directory (must contain final.pt and config.json)",
    )
    p.add_argument("--n-games", type=int, default=5)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed; game N uses seed+N for both engine and sampler",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="final.pt",
        help="checkpoint filename inside run_dir (default: final.pt)",
    )
    args = p.parse_args(argv)

    ckpt_path = args.run_dir / args.checkpoint
    cfg_path = args.run_dir / "config.json"
    if not ckpt_path.is_file():
        print(f"error: missing checkpoint {ckpt_path}", file=sys.stderr)
        return 2
    if not cfg_path.is_file():
        print(f"error: missing config {cfg_path}", file=sys.stderr)
        return 2

    full_cfg = json.loads(cfg_path.read_text())
    sp_cfg = {**full_cfg["self_play"], "stalemate": full_cfg["stalemate"]}
    mcts_cfg = full_cfg["mcts"]

    print(f"# Diagnostic for {ckpt_path}")
    print(
        f"#   win_vp={sp_cfg['victory_point_target']} "
        f"max_moves={sp_cfg['max_moves']} "
        f"mcts_rollouts={mcts_cfg['rollouts']} "
        f"dirichlet_ε={mcts_cfg['dirichlet_epsilon']} "
        f"T_initial={sp_cfg['temperature_initial']} "
        f"T_threshold_moves={sp_cfg['temperature_threshold_moves']}"
    )
    print(f"#   n_games={args.n_games} base_seed={args.seed}")
    print()

    agent, _meta = load_checkpoint(ckpt_path, device="cpu")
    rng = random.Random(args.seed)
    games = []
    for i in range(args.n_games):
        game_seed = args.seed + i
        print(f"## game {i+1}/{args.n_games}")
        g = _play_one_game(
            agent=agent,
            sp_cfg=sp_cfg,
            mcts_cfg=mcts_cfg,
            game_seed=game_seed,
            rng=rng,
        )
        games.append(g)
        print(_summarise_game(g))
        print()

    print(_summarise_aggregate(games))
    return 0


if __name__ == "__main__":
    sys.exit(main())
