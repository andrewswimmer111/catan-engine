"""
End-to-end smoke driver: play a 4-player game with random-but-legal moves.

Run from the project root:

    python -m src.main          # or
    PYTHONPATH=src python src/main.py

The script picks the first legal action at each step (deterministic given the
seed) and prints a final summary. It demonstrates that the engine can drive a
complete game from setup through victory or a sane move-cap fallback.
"""

from __future__ import annotations

import argparse
import random

from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.game.config import GameConfig
from domain.game.state import GameState
from domain.ids import PlayerID
from domain.rules import victory


def _pick_action(actions: list[Action], rng: random.Random) -> Action:
    """Prefer ending the turn over endlessly proposing trades; otherwise pick at random."""
    end_turn = [a for a in actions if isinstance(a, A.EndTurnAction)]
    proposals = {A.ProposeDomesticTradeAction, A.MaritimeTradeAction}
    non_trade = [a for a in actions if type(a) not in proposals]
    if end_turn and rng.random() < 0.35:
        return end_turn[0]
    if non_trade:
        return rng.choice(non_trade)
    return rng.choice(actions)


def play(seed: int = 0, n_players: int = 4, max_steps: int = 5000) -> GameState:
    pids = [PlayerID(i) for i in range(n_players)]
    cfg = GameConfig(player_ids=pids, seed=seed)
    engine = GameEngine(SeededRandomizer(seed))
    state = engine.new_game(cfg)
    rng = random.Random(seed)
    for step in range(max_steps):
        if state.is_terminal():
            break
        legals = engine.legal_actions(state)
        if not legals:
            break
        action = _pick_action(legals, rng)
        state = engine.apply_action(state, action).state
    return state


def _print_summary(state: GameState) -> None:
    print(f"phase: {state.phase.name}")
    print(f"turn:  {state.turn_number}")
    print(f"winner: {state.winner}")
    print("victory points:")
    for pid in state.config.player_ids:
        vp = victory.compute_victory_points(state, pid)
        p = state.players[pid]
        print(
            f"  P{int(pid)}  vp={vp:>2}  "
            f"settlements={p.settlements_built}  cities={p.cities_built}  "
            f"roads={p.roads_built}  knights={p.knights_played}"
        )
    print(f"longest road hold"
          f"er: {state.longest_road_holder}")
    print(f"largest army holder: {state.largest_army_holder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke 4-player Catan game.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=4, choices=(3, 4))
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    state = play(seed=args.seed, n_players=args.players, max_steps=args.max_steps)
    _print_summary(state)


if __name__ == "__main__":
    main()
