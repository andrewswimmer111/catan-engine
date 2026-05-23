"""Tests for :mod:`rl.training.self_play`.

Two layers:

* **Fast unit tests** for the rotation arithmetic, terminal-outcome
  computation, MCTS-target projection, and temperature schedule. These
  use synthetic state-vector inputs (no real games) so they run in
  milliseconds and pin the core invariants.
* **Slow integration test** that drives :func:`play_self_play_game`
  end-to-end with a tiny GNN and a small rollout budget. Confirms the
  whole loop terminates, emits a non-empty transition list, and
  attaches consistent value targets across every transition.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from domain.enums import EndReason, TurnPhase  # noqa: E402
from domain.ids import PlayerID  # noqa: E402
from rl.agents.policy_agent import PolicyAgent  # noqa: E402
from rl.encoding._action_layout import ACTION_SPACE_SIZE  # noqa: E402
from rl.encoding.action import ActionEncoder  # noqa: E402
from rl.encoding.graph_observation import (  # noqa: E402
    GRAPH_OBS_SHAPE,
    GraphObservationEncoder,
)
from rl.models.gnn import GNNArch, GNNPolicyValue  # noqa: E402
from rl.search.mcts import MCTSConfig  # noqa: E402
from rl.stalemate_value import StalemateValueConfig  # noqa: E402
from rl.training.self_play import (  # noqa: E402
    SelfPlayConfig,
    SelfPlayGame,
    SelfPlayTransition,
    play_self_play_game,
)
from rl.training.self_play import (  # noqa: E402  pull internals for unit tests
    _mcts_target_distribution,
    _rotate_to_acting_seat,
    _temperature_for,
    _terminal_outcome,
    _vp_aux_target,
)


_FLAT_QUARTER_PENALTY = StalemateValueConfig(shape="flat", flat_value=-0.25)
_FLAT_ZERO = StalemateValueConfig(shape="flat", flat_value=0.0)


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]
N_PLAYERS = len(PLAYER_IDS)


# ----------------------------------------------------------------------
# Rotation arithmetic
# ----------------------------------------------------------------------


def test_rotate_to_acting_seat_winner_at_specific_seat() -> None:
    """Winner-only outcome rotates so the winner's slot lands at
    ``(acting+k) % n`` for the seat that was acting at this transition."""
    # Seat 2 won; this transition was taken by seat 1.
    outcome = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    rotated = _rotate_to_acting_seat(outcome, acting_seat_idx=1)
    expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(rotated, expected)


def test_rotate_to_acting_seat_winner_at_acting_seat() -> None:
    """When the acting seat itself won, slot 0 of the rotated vector is 1."""
    outcome = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    rotated = _rotate_to_acting_seat(outcome, acting_seat_idx=1)
    assert rotated[0] == 1.0
    assert rotated[1:].sum() == 0.0


def test_rotate_to_acting_seat_is_invariant_for_symmetric_outcomes() -> None:
    """All-equal outcomes (e.g. stalemate) are invariant under rotation."""
    outcome = np.full(N_PLAYERS, -0.25, dtype=np.float32)
    for acting_idx in range(N_PLAYERS):
        rotated = _rotate_to_acting_seat(outcome, acting_seat_idx=acting_idx)
        np.testing.assert_array_equal(rotated, outcome)


# ----------------------------------------------------------------------
# Terminal outcome
# ----------------------------------------------------------------------


def _fake_state(winner: PlayerID | None, phase: TurnPhase):
    """Minimal duck-typed stand-in for a GameState in outcome tests."""

    class _Config:
        player_ids = list(PLAYER_IDS)

    class _State:
        def __init__(self):
            self.winner = winner
            self.phase = phase
            self.config = _Config()

    return _State()


def test_terminal_outcome_winner_is_one_hot() -> None:
    state = _fake_state(winner=PLAYER_IDS[2], phase=TurnPhase.GAME_OVER)
    out = _terminal_outcome(state, N_PLAYERS, _FLAT_QUARTER_PENALTY)
    expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_terminal_outcome_stalemate_uses_flat_value() -> None:
    """``shape='flat'`` returns ``flat_value`` for every seat — the legacy
    constant-target regime, retained for back-compat and A/B baselines."""
    state = _fake_state(winner=None, phase=TurnPhase.STALEMATE)
    out = _terminal_outcome(state, N_PLAYERS, _FLAT_QUARTER_PENALTY)
    np.testing.assert_allclose(out, np.full(N_PLAYERS, -0.25))


def test_terminal_outcome_stalemate_flat_value_is_tunable() -> None:
    """Caller can swap in a different flat constant (e.g. 0.0 to reproduce
    the canonical pure-AZ draw-is-zero baseline)."""
    state = _fake_state(winner=None, phase=TurnPhase.STALEMATE)
    out = _terminal_outcome(state, N_PLAYERS, _FLAT_ZERO)
    np.testing.assert_array_equal(out, np.zeros(N_PLAYERS, dtype=np.float32))


# ----------------------------------------------------------------------
# StalemateValueConfig — vp_linear shape (the new default)
# ----------------------------------------------------------------------


def test_vp_linear_endpoints_bracket_the_band() -> None:
    """Construct an artificial VP vector that lights up both corners of
    the band; targets must sit exactly at ``high`` (leader at 10+ VP)
    and ``low`` (last-place at 0 VP)."""
    cfg = StalemateValueConfig(shape="vp_linear", low=-0.5, high=-0.1)
    vps = np.array([10.0, 5.0, 3.0, 0.0])
    out = cfg._vp_linear(vps)
    # Leader: rank_score=1, vp_score=clip(10/10,0,1)=1 → combined=1 → high.
    assert pytest.approx(out[0], abs=1e-6) == -0.1
    # Last: rank_score=0, vp_score=0 → combined=0 → low.
    assert pytest.approx(out[-1], abs=1e-6) == -0.5
    # Middle seats sit strictly inside the band, monotonically by VP.
    assert -0.5 < out[2] < out[1] < -0.1


def test_vp_linear_all_zero_vp_yields_band_midpoint() -> None:
    """Symmetric case: every seat at 0 VP → all share rank=0 → all the
    same target. Should sit at the midpoint of [low, high] because
    rank_score=1 (everyone tied at the top) and vp_score=0."""
    cfg = StalemateValueConfig(shape="vp_linear", low=-0.5, high=-0.1)
    vps = np.zeros(4)
    out = cfg._vp_linear(vps)
    # combined = 0.5 * 1 + 0.5 * 0 = 0.5 → −0.5 + 0.4*0.5 = −0.3.
    np.testing.assert_allclose(out, np.full(4, -0.3), atol=1e-6)


def test_vp_linear_tied_seats_get_identical_targets() -> None:
    """Two seats at 5 VP should both get the same target — competitive
    ranking ties → identical rank score → identical VP score → equal."""
    cfg = StalemateValueConfig(shape="vp_linear")
    vps = np.array([5.0, 5.0, 3.0, 0.0])
    out = cfg._vp_linear(vps)
    assert out[0] == pytest.approx(out[1])
    assert out[0] > out[2] > out[3]


def test_vp_linear_clips_vp_above_winning_threshold() -> None:
    """A 12 VP terminal stalemate (rare but possible via dev cards +
    truncation) should clip to the 10 VP score — otherwise the band
    semantics break."""
    cfg = StalemateValueConfig(shape="vp_linear")
    vps_capped = cfg._vp_linear(np.array([10.0, 5.0, 3.0, 0.0]))
    vps_over = cfg._vp_linear(np.array([12.0, 5.0, 3.0, 0.0]))
    np.testing.assert_allclose(vps_capped, vps_over)


def test_stalemate_config_rejects_inverted_band() -> None:
    with pytest.raises(ValueError, match="high"):
        StalemateValueConfig(shape="vp_linear", low=-0.1, high=-0.5)


def test_stalemate_config_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError, match="unknown stalemate shape"):
        StalemateValueConfig(shape="rank_only")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# VP-aux target — rotation arithmetic + normalisation
# ----------------------------------------------------------------------


def test_vp_aux_target_uses_real_engine_state_with_zero_vp() -> None:
    """At a fresh post-setup state every seat has 2 settlement VP from
    the initial-placement phase. The aux target is VP/10 rotated to
    acting-as-slot-0; with everyone tied, every slot is 0.2."""
    from tests.fixtures.states import post_setup_state

    s = post_setup_state(seed=0, n_players=4)
    pids = list(s.config.player_ids)
    out = _vp_aux_target(s, pids, acting_seat_idx=2)
    np.testing.assert_allclose(out, np.full(N_PLAYERS, 0.2), atol=1e-6)
    assert out.dtype == np.float32


def test_vp_aux_target_rotates_acting_seat_to_slot_zero() -> None:
    """Seat 0 has 5 VP, seat 1 has 3, seat 2 has 2, seat 3 has 1 (built
    via the fixture state, then forced via raw counters). Aux target
    for acting_seat=1 must put seat 1 (3 VP → 0.3) in slot 0, seat 2
    in slot 1, ... seat 0 in slot 3."""
    from tests.fixtures.states import post_setup_state

    s = post_setup_state(seed=0, n_players=4)
    pids = list(s.config.player_ids)
    # Force a non-uniform VP layout. Bypass the engine and set raw
    # counters — the aux target reads through compute_victory_points
    # which sums settlements + 2×cities + dev VP + special awards.
    s.players[pids[0]].settlements_built = 5
    s.players[pids[1]].settlements_built = 3
    s.players[pids[2]].settlements_built = 2
    s.players[pids[3]].settlements_built = 1
    out = _vp_aux_target(s, pids, acting_seat_idx=1)
    # Absolute-seat VP (after /10): [0.5, 0.3, 0.2, 0.1]
    # Rotated so seat 1 lands at slot 0: [0.3, 0.2, 0.1, 0.5]
    np.testing.assert_allclose(out, [0.3, 0.2, 0.1, 0.5], atol=1e-6)


# ----------------------------------------------------------------------
# Temperature schedule
# ----------------------------------------------------------------------


def test_temperature_schedule_steps_at_threshold() -> None:
    cfg = SelfPlayConfig(
        temperature_initial=1.0,
        temperature_final=0.0,
        temperature_threshold_moves=30,
    )
    # Below threshold → initial.
    assert _temperature_for(0, cfg) == 1.0
    assert _temperature_for(29, cfg) == 1.0
    # At and above threshold → final.
    assert _temperature_for(30, cfg) == 0.0
    assert _temperature_for(99, cfg) == 0.0


def test_temperature_schedule_supports_nonzero_final() -> None:
    """A custom final temperature (e.g. T=0.1 for slight late-game noise)
    must be honoured."""
    cfg = SelfPlayConfig(
        temperature_initial=1.0,
        temperature_final=0.1,
        temperature_threshold_moves=5,
    )
    assert _temperature_for(10, cfg) == 0.1


# ----------------------------------------------------------------------
# MCTS target distribution (full-action-space projection)
# ----------------------------------------------------------------------


def test_mcts_target_distribution_sums_to_one_over_encoded_actions() -> None:
    """The projected target must remain a valid distribution after
    dropping unencodable actions."""
    from rl.encoding.action import ActionEncoder
    from rl.search.mcts import MCTSResult
    from domain.actions.all_actions import EndTurnAction, RollDiceAction

    # Two encodable actions, visit counts [3, 1] → proportional [0.75, 0.25].
    legal = [RollDiceAction(PLAYER_IDS[0]), EndTurnAction(PLAYER_IDS[0])]
    result = MCTSResult(
        action=legal[0],
        visit_counts=np.array([3, 1], dtype=np.int64),
        legal_actions=list(legal),
    )
    encoder = ActionEncoder(PLAYER_IDS)
    target = _mcts_target_distribution(result, encoder)
    assert target.shape == (ACTION_SPACE_SIZE,)
    assert pytest.approx(target.sum(), rel=1e-6) == 1.0
    # The two acting indices share all the mass at the AZ T=1 ratio.
    roll_idx = encoder.encode(legal[0])
    end_idx = encoder.encode(legal[1])
    assert pytest.approx(target[roll_idx], rel=1e-6) == 0.75
    assert pytest.approx(target[end_idx], rel=1e-6) == 0.25


def test_mcts_target_distribution_zero_visits_yields_uniform() -> None:
    """Degenerate: every count is zero (caller mistake or empty search).
    :meth:`MCTSResult.policy_distribution` falls back to uniform over the
    legal actions, and the target projection preserves that — the result
    is still a normalised distribution."""
    from rl.encoding.action import ActionEncoder
    from rl.search.mcts import MCTSResult
    from domain.actions.all_actions import EndTurnAction, RollDiceAction

    legal = [RollDiceAction(PLAYER_IDS[0]), EndTurnAction(PLAYER_IDS[0])]
    result = MCTSResult(
        action=legal[0],
        visit_counts=np.array([0, 0], dtype=np.int64),
        legal_actions=list(legal),
    )
    target = _mcts_target_distribution(result, ActionEncoder(PLAYER_IDS))
    assert pytest.approx(target.sum(), rel=1e-6) == 1.0


# ----------------------------------------------------------------------
# play_self_play_game integration
# ----------------------------------------------------------------------


_TINY_GNN = GNNArch(
    node_hidden=16,
    player_hidden=8,
    n_mp_layers=1,
    n_heads=4,
    global_mlp_hidden=16,
)


def _make_tiny_policy(seed: int = 0) -> PolicyAgent:
    torch.manual_seed(seed)
    model = GNNPolicyValue(
        obs_dim=GRAPH_OBS_SHAPE[0],
        action_dim=ACTION_SPACE_SIZE,
        arch=_TINY_GNN,  # default value_kind="vector" — exercises the AZ path
    )
    return PolicyAgent(
        model,  # type: ignore[arg-type]
        ActionEncoder(PLAYER_IDS),
        obs_encoder=GraphObservationEncoder(),  # type: ignore[arg-type]
    )


def _assert_game_is_self_consistent(game: SelfPlayGame) -> None:
    """Invariants every well-formed SelfPlayGame must satisfy."""
    assert isinstance(game, SelfPlayGame)
    assert len(game.transitions) == game.n_moves or game.n_moves > len(game.transitions)
    # ^ inequality covers the rare "forced-unencodable" skip path.

    for t in game.transitions:
        assert isinstance(t, SelfPlayTransition)
        assert t.obs.shape == (GRAPH_OBS_SHAPE[0],)
        assert t.action_mask.shape == (ACTION_SPACE_SIZE,)
        assert t.mcts_policy.shape == (ACTION_SPACE_SIZE,)
        assert 0 <= t.acting_seat_idx < N_PLAYERS
        assert t.value_target.shape == (N_PLAYERS,)
        # mcts_policy is a valid distribution.
        assert t.mcts_policy.sum() == pytest.approx(1.0, rel=1e-5)
        assert (t.mcts_policy >= 0).all()
        # mcts_policy mass only on legal slots.
        legal_mask = t.action_mask.astype(bool)
        assert (t.mcts_policy[~legal_mask] == 0).all()


@pytest.mark.slow
def test_play_self_play_game_completes_and_records_transitions() -> None:
    """End-to-end smoke: a tiny GNN + small rollout budget should still
    produce a well-formed game in reasonable wall-time."""
    network = _make_tiny_policy(seed=11)
    cfg = SelfPlayConfig(
        mcts=MCTSConfig(rollouts=4, c_puct=2.0, seed=0, dirichlet_epsilon=0.25),
        temperature_threshold_moves=5,
        max_moves=120,
    )
    rng = random.Random(0)
    game = play_self_play_game(network, cfg, rng, game_seed=1)

    _assert_game_is_self_consistent(game)
    # The game made *some* progress — even a stall produces > 8 transitions
    # (placements). If this drops below the threshold, something
    # short-circuited the loop unexpectedly.
    assert game.n_moves >= 8
    # end_reason is set by the engine on natural termination (winner /
    # stalemate); a max_moves-truncated game leaves it None. Either is
    # acceptable here — the goal is just that the loop ran cleanly.


@pytest.mark.slow
def test_play_self_play_value_targets_match_outcome() -> None:
    """The per-transition value target's rotation must be self-consistent
    with the (single) game outcome — every transition's slot 0 must
    equal the *acting* seat's outcome."""
    network = _make_tiny_policy(seed=21)
    cfg = SelfPlayConfig(
        mcts=MCTSConfig(rollouts=4, c_puct=2.0, seed=0, dirichlet_epsilon=0.25),
        temperature_threshold_moves=4,
        max_moves=80,
    )
    rng = random.Random(3)
    game = play_self_play_game(network, cfg, rng, game_seed=7)

    # Reconstruct absolute outcome from any transition's rotated target.
    # absolute[i] = rotated_target[(i - acting_idx) % n]
    if not game.transitions:
        pytest.skip("game emitted no transitions; nothing to verify")

    # Use the first transition to reconstruct.
    first = game.transitions[0]
    absolute = np.empty(N_PLAYERS, dtype=np.float32)
    for i in range(N_PLAYERS):
        absolute[i] = first.value_target[(i - first.acting_seat_idx) % N_PLAYERS]

    # Every other transition's rotated target must agree with the same absolute.
    for t in game.transitions[1:]:
        for i in range(N_PLAYERS):
            assert t.value_target[(i - t.acting_seat_idx) % N_PLAYERS] == pytest.approx(
                absolute[i], rel=1e-5
            )

    # Outcome-vector shape: one winner slot at 1.0 + rest 0.0, OR every
    # entry in the stalemate target band (vp_linear default → [-0.5, -0.1]).
    if game.winner_seat_idx is not None:
        expected = np.zeros(N_PLAYERS, dtype=np.float32)
        expected[game.winner_seat_idx] = 1.0
        np.testing.assert_allclose(absolute, expected)
    else:
        stalemate = cfg.stalemate
        if stalemate.shape == "flat":
            np.testing.assert_allclose(
                absolute, np.full(N_PLAYERS, stalemate.flat_value)
            )
        else:
            assert ((absolute >= stalemate.low - 1e-6)
                    & (absolute <= stalemate.high + 1e-6)).all()


@pytest.mark.slow
def test_play_self_play_max_moves_truncates_as_stalemate() -> None:
    """When max_moves trips before terminal, the game truncates and the
    outcome falls back to the stalemate-value penalty (no winner)."""
    network = _make_tiny_policy(seed=31)
    # Tiny rollout budget + extremely small max_moves so the cap fires
    # before the engine reaches GAME_OVER or natural STALEMATE.
    cfg = SelfPlayConfig(
        mcts=MCTSConfig(rollouts=2, c_puct=2.0, seed=0),
        temperature_threshold_moves=2,
        max_moves=12,
        stalemate=_FLAT_QUARTER_PENALTY,
    )
    game = play_self_play_game(network, cfg, random.Random(0), game_seed=2)
    # We capped at 12 moves; on a near-untrained tiny net we should hit
    # the cap rather than terminate naturally.
    assert game.n_moves == 12
    assert game.winner_seat_idx is None
    # Every transition's rotated value target equals the flat-shape value
    # for every seat.
    for t in game.transitions:
        np.testing.assert_allclose(
            t.value_target, np.full(N_PLAYERS, -0.25)
        )
