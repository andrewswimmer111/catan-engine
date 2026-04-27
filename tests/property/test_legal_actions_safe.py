"""Property: every legal action applies without an exception (sampled)."""

from __future__ import annotations

from copy import deepcopy

from hypothesis import given, settings
from hypothesis import strategies as st

from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.game.state import GameState
from tests.fixtures.states import fresh_game_state, post_setup_state


@settings(max_examples=10, deadline=None)
@given(
    st.integers(0, 6),
    st.booleans(),
)
def test_every_legal_action_in_sampled_state_is_appliable(
    seed: int, after_setup: bool
) -> None:
    s0: GameState = post_setup_state(seed) if after_setup else fresh_game_state(4, seed)
    eng = GameEngine(SeededRandomizer(seed))
    for a in list(eng.legal_actions(s0)):
        s_try = deepcopy(s0)
        eng.apply_action(s_try, a)
