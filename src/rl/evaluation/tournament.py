from __future__ import annotations

from collections import defaultdict
from typing import Callable

from controller.session import GameSnapshot
from domain.enums import EndReason
from domain.ids import PlayerID
from domain.rules.victory import compute_victory_points
from rl.agents.base import RLAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.metrics import GameStats, TournamentResult

__all__ = ["Tournament", "aggregate_games"]


class Tournament:
    def __init__(self, env_factory: Callable[[int], CatanEnv]) -> None:
        self._env_factory = env_factory

    def play(
        self,
        agents: dict[PlayerID, RLAgent],
        n_games: int,
        base_seed: int,
    ) -> TournamentResult:
        games = [self._play_one(agents, base_seed + i) for i in range(n_games)]
        return aggregate_games(games, list(agents.keys()))

    def _play_one(
        self,
        agents: dict[PlayerID, RLAgent],
        seed: int,
    ) -> GameStats:
        env = self._env_factory(seed)
        step_index = 0
        snap = GameSnapshot(
            state=env.state,
            step_index=step_index,
            last_action=None,
            last_events=(),
        )
        action_counts: dict[str, int] = defaultdict(int)
        per_seat_counts: dict[PlayerID, dict[str, int]] = {
            pid: defaultdict(int) for pid in env.state.config.player_ids
        }
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
            _, _, done, info = env.step(action)
            step_index += 1
            snap = GameSnapshot(
                state=env.state,
                step_index=step_index,
                last_action=action,
                last_events=tuple(info["last_events"]),
            )

        state = env.state
        return GameStats(
            winner=state.winner,
            final_vps={
                pid: compute_victory_points(state, pid)
                for pid in state.config.player_ids
            },
            turn_count=state.turn_number,
            end_reason=state.end_reason or EndReason.STALEMATE_NO_PROGRESS,
            action_histogram=dict(action_counts),
            per_seat_action_histogram={
                pid: dict(per_seat_counts[pid]) for pid in per_seat_counts
            },
        )


def aggregate_games(games: list[GameStats], player_ids: list[PlayerID]) -> TournamentResult:
    n = len(games)
    win_counts: dict[PlayerID, int] = defaultdict(int)
    vp_totals: dict[PlayerID, int] = defaultdict(int)
    turn_total = 0

    for g in games:
        if g.winner is not None:
            win_counts[g.winner] += 1
        for pid in player_ids:
            vp_totals[pid] += g.final_vps.get(pid, 0)
        turn_total += g.turn_count

    return TournamentResult(
        games=games,
        win_rates={pid: win_counts[pid] / n for pid in player_ids},
        mean_vp={pid: vp_totals[pid] / n for pid in player_ids},
        mean_turns=turn_total / n,
    )
