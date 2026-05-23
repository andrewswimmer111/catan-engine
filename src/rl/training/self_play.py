"""AlphaZero-style self-play game generator.

One :func:`play_self_play_game` call drives a single Catan game forward
on the pure engine (no env wrapper) using a shared
:class:`rl.agents.policy_agent.PolicyAgent` for all four seats. At each
decision point MCTS is run from the current state; the visit
distribution becomes both the training policy target (always at
``temperature=1.0``) and — softened by the temperature schedule — the
distribution we sample the actually-played action from.

The output is a :class:`SelfPlayGame` carrying one
:class:`SelfPlayTransition` per move. Each transition records:

* the observation encoded from the acting seat's perspective,
* the legal-action mask,
* the per-state MCTS visit-proportional policy target (mapped onto the
  full ``ACTION_SPACE_SIZE`` slots; unencodable actions like domestic
  trade proposals are renormalised out),
* the absolute acting-seat index, and
* the per-seat value target — the terminal outcome vector, **rotated**
  so the acting seat sits in slot 0 (matching the
  ``value_kind="vector"`` model output convention from az-001).

Stalemate targets are produced by :class:`StalemateValueConfig` —
default ``"vp_linear"`` (band ``[-0.5, -0.1]``, rank/VP combined). See
:mod:`rl.stalemate_value` for the why behind moving off the constant
``-0.25`` target after run #AZ-1 showed it collapsed the value loss.

Known v1 warts (documented per project convention):

* **DISCARD phase.** The action encoder collapses all
  :class:`DiscardResourcesAction` legals to one slot, so MCTS gets a
  flat prior over discard combinations and the policy target collapses
  them all onto ``DISCARD_INDEX``. Training learns "discard yes/no"
  rather than which cards to drop; the env's heuristic-discard takes
  over at play time anyway. Reconsider if discard policy starts
  visibly mattering.
* **Forced moves.** Single-legal-action states (``RollDice``,
  ``EndTurn``, single robber sequels) skip MCTS entirely. The recorded
  transition has a one-hot policy target on that action; this is a
  cheap learning signal that the network can't predict the wrong
  action even if it tries.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.enums import EndReason
from domain.game.config import GameConfig
from domain.ids import PlayerID
from domain.rules.victory import compute_victory_points
from rl.acting_player import acting_player
from rl.agents.policy_agent import PolicyAgent
from rl.agents.search_agent import NetworkEvaluator
from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.search.mcts import MCTSConfig, MCTSResult, run_mcts
from rl.stalemate_value import StalemateValueConfig

__all__ = [
    "SelfPlayConfig",
    "SelfPlayTransition",
    "SelfPlayGame",
    "play_self_play_game",
]


_DEFAULT_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


@dataclass(frozen=True)
class SelfPlayConfig:
    """Knobs for one :func:`play_self_play_game` call.

    The defaults match the Phase 3 design pass:

    * ``temperature_threshold_moves=30`` — ``T=1.0`` for the first 30
      moves (covers placements + early game), then ``T=0.0`` (argmax)
      thereafter. Training policy targets always use ``T=1.0``
      regardless; this schedule only shapes the sampled action.
    * ``stalemate`` defaults to the ``vp_linear`` shape in
      :class:`StalemateValueConfig`. Each seat's stalemate target lands
      in ``[-0.5, -0.1]`` based on its rank and VP at the terminal
      state, restoring variance in the value target distribution.
    * ``mcts.dirichlet_epsilon`` is **not** set here — the caller
      builds an ``MCTSConfig`` that turns Dirichlet on for self-play
      (typically ``epsilon=0.25``) and leaves it off for evaluation.

    The same ``StalemateValueConfig`` instance should also live on
    ``mcts.stalemate`` so MCTS's terminal backup matches the training
    target. ``train_alphazero.py`` does this wiring at config-build
    time.
    """

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temperature_initial: float = 1.0
    temperature_final: float = 0.0
    temperature_threshold_moves: int = 30
    stalemate: StalemateValueConfig = field(default_factory=StalemateValueConfig)
    max_moves: int = 1000
    player_ids: tuple[PlayerID, ...] = _DEFAULT_PLAYER_IDS


@dataclass(frozen=True)
class SelfPlayTransition:
    """One (state, MCTS-policy, value-target) training sample.

    All fields are plain numpy arrays / Python scalars so a
    :class:`SelfPlayGame` is trivially pickle-able for the (future)
    parallel self-play workers in az-009.
    """

    obs: np.ndarray
    """Encoded observation from the acting seat's perspective."""

    action_mask: np.ndarray
    """Legal-action mask for this state, shape ``(ACTION_SPACE_SIZE,)``."""

    mcts_policy: np.ndarray
    """MCTS visit-proportional policy target, shape ``(ACTION_SPACE_SIZE,)``.

    Always computed at ``temperature=1.0`` regardless of what
    temperature was used to sample the played action — this is the AZ
    training-target convention.
    """

    acting_seat_idx: int
    """Absolute seat index (0..n_players-1) of the player who acted."""

    value_target: np.ndarray
    """Per-seat outcome rotated to acting-seat-as-slot-0; shape ``(n_players,)``.

    Matches the ``value_kind="vector"`` model's output convention so
    the value loss is a direct MSE against this vector.
    """

    vp_aux_target: np.ndarray
    """Per-seat VP at the *current state*, normalised by the 10-VP win
    threshold and rotated to acting-seat-as-slot-0; shape ``(n_players,)``.

    Used by the optional per-step VP auxiliary value loss
    (:attr:`AZTrainConfig.aux_value_coef` > 0): the value head is also
    regressed against this vector with a small coefficient so it
    learns that high VP is intrinsically valuable, independent of
    whether the bootstrap loop has yet observed terminal wins. Set
    coefficient to 0 to ignore the field entirely (it's still
    populated; the cost is negligible).
    """


@dataclass(frozen=True)
class SelfPlayGame:
    """One self-play game's worth of training samples plus terminal info."""

    transitions: tuple[SelfPlayTransition, ...]
    winner_seat_idx: Optional[int]
    """Absolute seat index of the winner, or ``None`` for a stalemate."""
    n_moves: int
    end_reason: Optional[EndReason]


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def play_self_play_game(
    network: PolicyAgent,
    config: SelfPlayConfig,
    rng: random.Random,
    *,
    game_seed: int = 0,
) -> SelfPlayGame:
    """Drive one self-play game forward; return its training samples.

    ``network`` is the shared policy/value agent used for every seat —
    self-play is symmetric. ``rng`` controls action sampling from the
    temperature-softened MCTS distribution; ``game_seed`` seeds the
    engine and the per-move MCTS seed derivation (so two calls with the
    same ``game_seed`` and ``rng`` state produce the same game).
    """
    pids = list(config.player_ids)
    n_players = len(pids)
    engine = GameEngine(SeededRandomizer(game_seed))
    state = engine.new_game(GameConfig(player_ids=pids, seed=game_seed))

    network.model.eval()
    evaluator = NetworkEvaluator(network)
    action_encoder = network.action_encoder
    obs_encoder = network.obs_encoder

    pending: list[_PendingTransition] = []
    move_idx = 0
    while not state.is_terminal() and move_idx < config.max_moves:
        legal = engine.legal_actions(state)
        if not legal:
            break

        acting_pid = acting_player(state)
        acting_idx = pids.index(acting_pid)
        obs = obs_encoder.encode(make_player_view(state, acting_pid))
        mask = action_encoder.mask(legal)
        # Per-state VP snapshot (rotated to acting-as-slot-0). Cheap to
        # compute (engine traversal over 4 seats) and the only way the
        # value head can get win-shaped signal in regimes where games
        # don't naturally terminate in a winner.
        vp_aux = _vp_aux_target(state, pids, acting_idx)

        chosen_action, policy_target = _decide_move(
            state=state,
            legal=legal,
            evaluator=evaluator,
            engine=engine,
            config=config,
            action_encoder=action_encoder,
            move_idx=move_idx,
            rng=rng,
        )
        if policy_target is None:
            # Forced move whose only legal action wasn't encodable
            # (extremely rare; e.g. a domestic-trade-only legal set).
            # Apply but skip the training sample.
            state = engine.apply_action(state, chosen_action).state
            move_idx += 1
            continue

        pending.append(
            _PendingTransition(
                obs=obs,
                action_mask=mask,
                mcts_policy=policy_target,
                acting_seat_idx=acting_idx,
                vp_aux_target=vp_aux,
            )
        )
        state = engine.apply_action(state, chosen_action).state
        move_idx += 1

    outcome = _terminal_outcome(state, n_players, config.stalemate)
    winner_idx: Optional[int] = None
    if state.winner is not None:
        winner_idx = pids.index(state.winner)

    transitions = tuple(
        SelfPlayTransition(
            obs=p.obs,
            action_mask=p.action_mask,
            mcts_policy=p.mcts_policy,
            acting_seat_idx=p.acting_seat_idx,
            value_target=_rotate_to_acting_seat(outcome, p.acting_seat_idx),
            vp_aux_target=p.vp_aux_target,
        )
        for p in pending
    )
    return SelfPlayGame(
        transitions=transitions,
        winner_seat_idx=winner_idx,
        n_moves=move_idx,
        end_reason=state.end_reason,
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


@dataclass
class _PendingTransition:
    """Mutable accumulator before the terminal outcome is known."""

    obs: np.ndarray
    action_mask: np.ndarray
    mcts_policy: np.ndarray
    acting_seat_idx: int
    vp_aux_target: np.ndarray


def _decide_move(
    *,
    state,
    legal: list[Action],
    evaluator: NetworkEvaluator,
    engine: GameEngine,
    config: SelfPlayConfig,
    action_encoder,
    move_idx: int,
    rng: random.Random,
) -> tuple[Action, np.ndarray | None]:
    """Choose the next action and produce its policy target.

    Returns ``(chosen_action, policy_target)``. ``policy_target`` is
    ``None`` when the move is forced AND the single legal action is not
    encodable in the discrete head — the caller skips recording the
    transition in that case.
    """
    if len(legal) == 1:
        action = legal[0]
        try:
            idx = action_encoder.encode(action)
        except ValueError:
            return action, None
        target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        target[idx] = 1.0
        return action, target

    # Fresh MCTSConfig per move so the Dirichlet sample diverges across
    # the game (otherwise every move shares config.mcts.seed and gets
    # the same noise sample, defeating the exploration purpose).
    move_cfg = dataclasses.replace(
        config.mcts, seed=config.mcts.seed + move_idx
    )
    result = run_mcts(state, evaluator, move_cfg, engine=engine)
    temperature = _temperature_for(move_idx, config)
    sample_idx = _sample_legal_index(result, temperature, rng)
    chosen = result.legal_actions[sample_idx]
    target = _mcts_target_distribution(result, action_encoder)
    # If every legal action was unencodable (e.g. a domestic-trade-only
    # state), the target collapsed to zeros. There's no training signal
    # to record here, so emit a skip.
    if float(target.sum()) <= 0.0:
        return chosen, None
    return chosen, target


def _temperature_for(move_idx: int, config: SelfPlayConfig) -> float:
    """Step temperature schedule: ``initial`` early, ``final`` after the threshold."""
    return (
        config.temperature_initial
        if move_idx < config.temperature_threshold_moves
        else config.temperature_final
    )


def _sample_legal_index(
    result: MCTSResult, temperature: float, rng: random.Random
) -> int:
    """Sample an index into ``result.legal_actions`` at ``temperature``."""
    dist = result.policy_distribution(temperature)
    if temperature == 0.0:
        return int(dist.argmax())
    # Inverse-CDF sample over a small discrete dist; faster (and more
    # debuggable) than numpy.random.choice for this length.
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(dist):
        cum += float(p)
        if r < cum:
            return i
    # Numerical-edge case: r ≈ 1.0 and rounding undershot the sum.
    return len(dist) - 1


def _mcts_target_distribution(
    result: MCTSResult, action_encoder
) -> np.ndarray:
    """Project the visit-proportional MCTS distribution onto ``ACTION_SPACE_SIZE``.

    Unencodable legal actions (domestic trade proposals) get their
    probability mass dropped; the remaining distribution is
    renormalised. Multiple legal actions colliding onto the same
    encoder index (e.g. discard combinations all → ``DISCARD_INDEX``)
    are summed.
    """
    full = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    target = result.policy_distribution(1.0)
    for action, p in zip(result.legal_actions, target):
        try:
            idx = action_encoder.encode(action)
        except ValueError:
            continue
        full[idx] += float(p)
    total = float(full.sum())
    if total > 0.0:
        full /= total
    return full


def _terminal_outcome(
    state, n_players: int, stalemate: StalemateValueConfig
) -> np.ndarray:
    """Per-seat outcome vector in absolute seat order.

    Winner case: ``1.0`` for the winner's slot, ``0.0`` elsewhere.
    Stalemate (including ``max_moves`` truncation): ``stalemate.compute(...)``
    — see :class:`StalemateValueConfig` for the shape.
    """
    if state.winner is None:
        return stalemate.compute(state, n_players)
    out = np.zeros(n_players, dtype=np.float32)
    for i, pid in enumerate(state.config.player_ids):
        if pid == state.winner:
            out[i] = 1.0
    return out


def _vp_aux_target(
    state, pids: list[PlayerID], acting_seat_idx: int
) -> np.ndarray:
    """Per-seat VP / 10 at ``state``, rotated to acting-as-slot-0.

    Range per slot: ``[0.0, 0.9]`` for non-terminal states (any seat at
    10 VP would have triggered :func:`check_winner` and ended the
    game). Normalisation matches the win value target (``+1.0`` for a
    real winner) so the auxiliary loss and the terminal loss live on
    the same scale.
    """
    raw = np.fromiter(
        (compute_victory_points(state, pid) for pid in pids),
        dtype=np.float32,
        count=len(pids),
    )
    raw /= 10.0
    return np.roll(raw, -acting_seat_idx).astype(np.float32, copy=False)


def _rotate_to_acting_seat(
    outcome: np.ndarray, acting_seat_idx: int
) -> np.ndarray:
    """Roll an absolute-seat outcome so the acting seat lands in slot 0.

    Matches the ``value_kind="vector"`` model's output convention: slot
    0 is always the encoder's viewer (== acting seat for transitions
    where the acting seat encoded the observation). For a winner-only
    outcome ``[0, 0, 1, 0]`` and ``acting_seat_idx=1`` this returns
    ``[0, 1, 0, 0]`` — i.e. ``outcome[(acting_seat_idx + k) % n] = out[k]``.
    """
    return np.roll(outcome, -acting_seat_idx).astype(np.float32, copy=False)
