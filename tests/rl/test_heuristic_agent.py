from __future__ import annotations

import copy
import random

import pytest

from controller.session import GameSnapshot
from domain.actions import all_actions as A
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import SeededRandomizer
from domain.enums import Resource, TurnPhase
from domain.game.config import GameConfig
from domain.ids import PlayerID
from domain.turn.pending import DiscardPending
from rl.agents._heuristic_rules import vertex_pip_count
from rl.agents.heuristic_agent import HeuristicAgent, heuristic_discard
from rl.agents.random_agent import RandomAgent
from rl.env.catan_env import CatanEnv
from rl.evaluation.tournament import Tournament
from tests.fixtures.states import post_setup_state

PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
HEUR_SEAT = PLAYER_IDS[0]
RANDOM_SEATS = PLAYER_IDS[1:]


def _env_factory(seed: int) -> CatanEnv:
    return CatanEnv(seed=seed)


def _mixed_agents(seed: int) -> dict[PlayerID, object]:
    """One heuristic seat; the rest are RandomAgents that skip trade proposals
    so games don't burn wall-clock in unconverted trade flows."""
    rng = random.Random(seed)
    agents: dict[PlayerID, object] = {}
    for pid in PLAYER_IDS:
        sub = random.Random(rng.randrange(2**32))
        if pid == HEUR_SEAT:
            agents[pid] = HeuristicAgent(sub)
        else:
            agents[pid] = RandomAgent(sub, skip_proposals=True)
    return agents


def _all_heuristic(seed: int) -> dict[PlayerID, HeuristicAgent]:
    rng = random.Random(seed)
    return {
        pid: HeuristicAgent(random.Random(rng.randrange(2**32)))
        for pid in PLAYER_IDS
    }


# ----------------------------------------------------------------------
# Setup placement
# ----------------------------------------------------------------------


def test_setup_settlement_picks_max_pip_vertex():
    cfg = GameConfig(player_ids=PLAYER_IDS, seed=42)
    engine = GameEngine(SeededRandomizer(42))
    state = engine.new_game(cfg)
    legal = engine.legal_actions(state)
    settlements = [a for a in legal if isinstance(a, A.PlaceSettlementAction)]
    best_pip = max(vertex_pip_count(state, a.vertex_id) for a in settlements)

    agent = HeuristicAgent(random.Random(0))
    snap = GameSnapshot(state=state, step_index=0, last_action=None, last_events=())
    pick = agent.choose(snap, legal)

    assert isinstance(pick, A.PlaceSettlementAction)
    assert vertex_pip_count(state, pick.vertex_id) == best_pip


# ----------------------------------------------------------------------
# Discard
# ----------------------------------------------------------------------


def _state_with_hand(hand: dict[Resource, int]) -> tuple[object, PlayerID]:
    state = copy.deepcopy(post_setup_state(seed=0, n_players=4))
    me = state.config.player_ids[0]
    state.players[me].resources = dict(hand)
    state.phase = TurnPhase.DISCARD
    state.pending = DiscardPending(cards_to_discard={me: sum(hand.values()) // 2})
    return state, me


@pytest.mark.parametrize(
    "hand",
    [
        {Resource.WOOD: 4, Resource.BRICK: 4},                           # 8 → keep 4
        {Resource.WHEAT: 5, Resource.ORE: 4},                            # 9 → keep 5
        {Resource.SHEEP: 4, Resource.WOOD: 4, Resource.BRICK: 4},        # 12 → keep 6
        {Resource.SHEEP: 5, Resource.WOOD: 5, Resource.WHEAT: 4},        # 14 → keep 7
    ],
)
def test_heuristic_discard_leaves_at_most_seven(hand):
    state, me = _state_with_hand(hand)
    total = sum(hand.values())
    need = total // 2

    action = heuristic_discard(state, me)

    assert action.player_id == me
    assert sum(action.resources.values()) == need
    assert total - need <= 7
    for r, c in action.resources.items():
        assert c > 0
        assert c <= hand.get(r, 0)


def test_heuristic_discard_drops_most_abundant_first():
    state, me = _state_with_hand(
        {Resource.SHEEP: 6, Resource.WHEAT: 1, Resource.ORE: 1}
    )

    action = heuristic_discard(state, me)

    # Hand size 8 → discard 4. All four come from the most-abundant pile.
    assert action.resources == {Resource.SHEEP: 4}


# ----------------------------------------------------------------------
# Smoke (fast — runs in the default suite)
# ----------------------------------------------------------------------


def test_smoke_heuristic_outscores_randoms_in_three_games():
    """Three games terminate and the heuristic's mean VP beats every random's.

    VP-comparison (not win-rate) keeps this test non-flaky: the engine's
    50-turn VP-stall threshold means many games end before anyone hits 10
    VP, but the heuristic still accumulates 6–8 VP while randoms stall at 2–4.
    """
    t = Tournament(_env_factory)
    result = t.play(_mixed_agents(seed=0), n_games=3, base_seed=100)

    assert len(result.games) == 3
    for g in result.games:
        assert g.turn_count > 0
    heur_vp = result.mean_vp[HEUR_SEAT]
    for pid in RANDOM_SEATS:
        assert heur_vp > result.mean_vp[pid], result.mean_vp


# ----------------------------------------------------------------------
# Slow (regression checks — run with `pytest -m slow`)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_heuristic_dominates_random_over_30_games():
    """Heuristic crushes random in both wins and mean VP.

    Threshold uses VP rather than win-rate so stalemates (caused by the
    engine's 50-turn no-progress rule) don't make this flaky. Random's mean
    VP stays around 3 in completed and stalemated games alike, while the
    heuristic averages 7–8.
    """
    t = Tournament(_env_factory)
    result = t.play(_mixed_agents(seed=1), n_games=30, base_seed=200)

    heur_wins = result.win_rates[HEUR_SEAT]
    random_wins_total = sum(result.win_rates[pid] for pid in RANDOM_SEATS)
    assert heur_wins > random_wins_total, result.win_rates

    heur_vp = result.mean_vp[HEUR_SEAT]
    best_random_vp = max(result.mean_vp[pid] for pid in RANDOM_SEATS)
    assert heur_vp > 2 * best_random_vp, result.mean_vp


@pytest.mark.slow
def test_heuristic_self_play_is_seat_symmetric():
    """4 heuristic agents: each seat's mean VP stays within 2.0 of the others.

    Self-play typically stalemates under the env's 50-turn VP-stall rule, so
    seat win-rate is trivially symmetric (all zeros). Mean VP is the more
    informative symmetry signal; 2.0 VP slack absorbs the ~0.3-VP standard
    error from 100 games.
    """
    t = Tournament(_env_factory)
    result = t.play(_all_heuristic(seed=99), n_games=100, base_seed=500)

    vps = [result.mean_vp[pid] for pid in PLAYER_IDS]
    spread = max(vps) - min(vps)
    assert spread < 2.0, result.mean_vp
