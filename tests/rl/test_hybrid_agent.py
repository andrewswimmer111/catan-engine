"""Tests for :class:`rl.agents.hybrid_agent.PlacementOverrideAgent`."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

import pytest

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import TurnPhase
from domain.ids import PlayerID
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.hybrid_agent import PLACEMENT_PHASES, PlacementOverrideAgent
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.tournament import Tournament
from tests.fixtures.states import post_setup_state

PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


# ----------------------------------------------------------------------
# Spy agent — records every (phase, action) it returned.
# ----------------------------------------------------------------------


@dataclass
class _SpyAgent:
    """Agent that delegates to ``inner`` and records each call's phase."""

    inner: object  # any Agent
    calls: list[TurnPhase] = field(default_factory=list)

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        self.calls.append(snap.state.phase)
        return self.inner.choose(snap, legal)


# ----------------------------------------------------------------------
# Dispatch unit tests
# ----------------------------------------------------------------------


def test_placement_phases_constant_matches_initial_phases():
    """The frozenset is the two initial-placement phases — nothing else."""
    assert PLACEMENT_PHASES == frozenset(
        {TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD}
    )


def _settlement_snap_and_legal() -> tuple[GameSnapshot, list[Action]]:
    engine = GameEngine(SeededRandomizer(42))
    from domain.game.config import GameConfig

    cfg = GameConfig(player_ids=PLAYER_IDS, seed=42)
    state = engine.new_game(cfg)
    assert state.phase is TurnPhase.INITIAL_SETTLEMENT
    legal = engine.legal_actions(state)
    snap = GameSnapshot(state=state, step_index=0, last_action=None, last_events=())
    return snap, legal


def test_initial_settlement_phase_routes_to_placement_agent():
    snap, legal = _settlement_snap_and_legal()
    main = _SpyAgent(inner=HeuristicAgent())
    placement = _SpyAgent(inner=HeuristicAgent())
    agent = PlacementOverrideAgent(main_agent=main, placement_agent=placement)

    action = agent.choose(snap, legal)

    assert isinstance(action, A.PlaceSettlementAction)
    assert placement.calls == [TurnPhase.INITIAL_SETTLEMENT]
    assert main.calls == []


def test_initial_road_phase_routes_to_placement_agent():
    snap, legal0 = _settlement_snap_and_legal()
    # Apply the settlement so the engine advances to INITIAL_ROAD.
    engine = GameEngine(SeededRandomizer(42))
    settle = HeuristicAgent().choose(snap, legal0)
    assert isinstance(settle, A.PlaceSettlementAction)
    result = engine.apply_action(snap.state, settle)
    state1 = result.state
    assert state1.phase is TurnPhase.INITIAL_ROAD
    legal1 = engine.legal_actions(state1)
    snap1 = GameSnapshot(
        state=state1, step_index=1, last_action=settle, last_events=()
    )

    main = _SpyAgent(inner=HeuristicAgent())
    placement = _SpyAgent(inner=HeuristicAgent())
    agent = PlacementOverrideAgent(main_agent=main, placement_agent=placement)

    pick = agent.choose(snap1, legal1)

    assert isinstance(pick, A.PlaceRoadAction)
    assert placement.calls == [TurnPhase.INITIAL_ROAD]
    assert main.calls == []


def test_main_phase_routes_to_main_agent():
    state = copy.deepcopy(post_setup_state(seed=0))
    state.phase = TurnPhase.MAIN
    state.pending = None
    # Pick a current_player whose acting context is MAIN.
    me = state.config.player_ids[0]
    state.current_player = me

    engine = GameEngine(SeededRandomizer(0))
    legal = engine.legal_actions(state)
    snap = GameSnapshot(state=state, step_index=0, last_action=None, last_events=())

    main = _SpyAgent(inner=RandomAgent(random.Random(0), skip_proposals=True))
    placement = _SpyAgent(inner=HeuristicAgent())
    agent = PlacementOverrideAgent(main_agent=main, placement_agent=placement)

    agent.choose(snap, legal)

    assert main.calls == [TurnPhase.MAIN]
    assert placement.calls == []


@pytest.mark.parametrize(
    "phase",
    [
        TurnPhase.ROLL,
        TurnPhase.DISCARD,
        TurnPhase.MOVE_ROBBER,
        TurnPhase.STEAL,
        TurnPhase.BUILD_ROADS,
        TurnPhase.YEAR_OF_PLENTY_SELECT,
        TurnPhase.MONOPOLY_SELECT,
    ],
)
def test_non_placement_phases_route_to_main_agent(phase):
    """Even build-road from a Road-Building card must NOT hit placement_agent."""
    main = _SpyAgent(inner=HeuristicAgent())
    placement = _SpyAgent(inner=HeuristicAgent())
    agent = PlacementOverrideAgent(main_agent=main, placement_agent=placement)

    # Synthetic snapshot with the phase we want to probe. choose() routes by
    # phase alone; the agents' inner choose() never runs because we hand them
    # a stub list — they pick legal[0] as a last resort. The dispatch is what
    # we're verifying here.
    state = copy.deepcopy(post_setup_state(seed=0))
    state.phase = phase
    snap = GameSnapshot(state=state, step_index=0, last_action=None, last_events=())

    # Provide a single dummy action so the inner agent's last-resort branch
    # returns it cleanly. The dispatch happens before the inner agent looks
    # at legal, so the action's type doesn't matter.
    dummy = object()
    try:
        agent.choose(snap, [dummy])  # type: ignore[list-item]
    except Exception:
        # Some inner agents will refuse to handle a dummy action; that's fine —
        # we only care which spy's calls list grew.
        pass

    assert placement.calls == []
    assert main.calls == [phase]


# ----------------------------------------------------------------------
# Full-game smoke
# ----------------------------------------------------------------------


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def test_full_game_smoke_heuristic_placement_random_main():
    """A hybrid agent (heuristic openings, random main) finishes a game.

    The point isn't who wins — it's that the wrapper composes cleanly with
    the Tournament harness and never produces an illegal/None action mid-game.
    """
    hybrid_seat = PLAYER_IDS[0]
    agents = {
        hybrid_seat: PlacementOverrideAgent(
            main_agent=RandomAgent(random.Random(0), skip_proposals=True),
            placement_agent=HeuristicAgent(),
        )
    }
    for pid in PLAYER_IDS[1:]:
        agents[pid] = RandomAgent(random.Random(int(pid)), skip_proposals=True)

    result = Tournament(_env_factory).play(agents, n_games=2, base_seed=7)

    assert len(result.games) == 2
    # The hybrid seat's settlement and road counts include the two of each
    # from initial placement, so the action histogram is well-formed.
    for game in result.games:
        per_seat = game.per_seat_action_histogram[hybrid_seat]
        assert per_seat.get("PlaceSettlementAction", 0) >= 2
        assert per_seat.get("PlaceRoadAction", 0) >= 2
