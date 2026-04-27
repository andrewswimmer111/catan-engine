"""Property-based checks: resource counts, occupancy, and terminality."""

from __future__ import annotations

from copy import deepcopy

from hypothesis import given, settings
from hypothesis import strategies as st

from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.game.state import GameState
from tests.fixtures.states import _first_setup_action, fresh_game_state, post_setup_state


def _bank_and_hands_non_negative(s: GameState) -> bool:
    for c in s.bank.resources.values():
        if c < 0:
            return False
    for p in s.players.values():
        for c in p.resources.values():
            if c < 0:
                return False
    return True


def _at_most_one_occupant_per_slot(s: GameState) -> bool:
    return len(s.occupancy.roads) == len(s.occupancy.roads.items())


@settings(max_examples=15, deadline=None)
@given(
    st.integers(0, 5),
    st.integers(1, 20),
    st.booleans(),
)
def test_random_legal_ply_sequence_preserves_invariants(
    game_seed: int, num_steps: int, start_post_setup: bool
) -> None:
    s: GameState = post_setup_state(game_seed) if start_post_setup else fresh_game_state(
        4, game_seed
    )
    s = deepcopy(s)
    eng = GameEngine(SeededRandomizer(game_seed))
    for _ in range(num_steps):
        if s.is_terminal():
            assert s.winner is not None or s.phase is TurnPhase.STALEMATE
            break
        acts = eng.legal_actions(s)
        if not acts:
            break
        if s.phase in (TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD):
            a = _first_setup_action(acts)
        else:
            a = acts[0]
        s = eng.apply_action(s, a).state
        assert _bank_and_hands_non_negative(s)
        assert _at_most_one_occupant_per_slot(s)
        if s.is_terminal():
            assert s.winner is not None or s.phase is TurnPhase.STALEMATE
