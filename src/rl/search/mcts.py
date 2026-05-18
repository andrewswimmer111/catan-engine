"""PUCT-style MCTS over :class:`GameState`.

The search runs purely on the engine — no env wrapper — and is decoupled
from the network through the :class:`Evaluator` Protocol. That lets
``test_mcts`` exercise the tree mechanics with stub evaluators while
:class:`rl.agents.search_agent.SearchAgent` injects a real
:class:`PolicyAgent` at play time.

Design choices (see Phase 2 design pass):

* **Perfect-info MCTS on the full** :class:`GameState`. The engine's hidden
  info surface is narrow (opp dev card types, deck order) and most Catan
  MCTS references use perfect-info; ISMCTS is a future extension.
* **Per-player value vector backup.** The evaluator returns a value of
  shape ``(n_players,)`` so each ancestor node uses its acting player's
  slot when computing Q. This is the multi-player generalisation of the
  standard zero-sum scalar backup.
* **Dice chance nodes are enumerated lazily** with the 2d6 PMF. The
  ``RollDiceAction`` edge holds 11 (probability, child) outcomes; each
  visit samples one outcome by its PMF weight and expands lazily on
  first touch.
* **Dev card draws and steal-resource picks are open-loop** — engine
  samples internally on each apply; we cache the first sampled child and
  reuse it. This is a documented v1 wart; replace with explicit chance
  nodes if these turn out to bias estimates.
* **Tree is not reused across moves.** A fresh root is built each call.

The PUCT score follows the AlphaZero form:

::

    a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · sqrt(ΣN(s,b)) / (1 + N(s,a)) ]

where ``Q(s,a)`` is the running mean of the acting player's slot of the
value vector accumulated through edge ``(s, a)``, ``N(s,a)`` is its visit
count, and ``P(s,a)`` is the policy prior over legal actions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from domain.actions.all_actions import RollDiceAction
from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.randomizer import Randomizer
from domain.game.state import GameState
from domain.ids import PlayerID
from rl.acting_player import acting_player

__all__ = [
    "DICE_SUM_PMF",
    "Evaluator",
    "MCTSConfig",
    "MCTSResult",
    "run_mcts",
]


# 2d6 PMF over sums 2..12 (index 0 = sum 2, index 10 = sum 12).
# Counts: 1,2,3,4,5,6,5,4,3,2,1 out of 36. Stored normalised.
DICE_SUM_PMF: tuple[float, ...] = tuple(
    c / 36.0 for c in (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1)
)
# Canonical (d1, d2) representative per sum — the engine's transitions only
# read the sum, so any pair will do.
_DICE_REPR: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
)


class Evaluator(Protocol):
    """Plug-in for the network-driven leaf evaluation step.

    Returns priors over the supplied ``legal`` actions (length matched, sums
    to 1 over legal) and a per-player value vector indexed by
    :attr:`GameConfig.player_ids` order. The implementation typically runs
    the policy/value net once with the leaf's acting player as the viewer
    (for priors) and once per seat (for the value vector); a single batched
    forward across all four perspectives is the recommended pattern.
    """

    def evaluate(
        self, state: GameState, legal: list[Action]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(priors, value_vec)``.

        * ``priors`` — shape ``(len(legal),)``, sums to 1 over legal actions.
        * ``value_vec`` — shape ``(n_players,)``, slot ``i`` is the predicted
          win probability for ``state.config.player_ids[i]``.
        """
        ...


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for one MCTS call.

    * ``rollouts`` — number of simulations per move.
    * ``c_puct`` — exploration coefficient in the PUCT formula. Higher
      values explore more; AlphaZero used 1.25 but Catan's high branching
      factor benefits from ~2.0 in early experiments.
    * ``seed`` — controls dice-outcome sampling and Dirichlet noise
      sampling for reproducibility.
    * ``dirichlet_alpha`` — concentration parameter of the Dirichlet
      distribution used to inject root-only exploration noise. AlphaZero
      uses ``alpha ≈ 10 / avg_legal_moves``; Catan's branching factor
      varies wildly (~5–50), so the canonical chess value ``0.3`` is a
      reasonable starting point. Only effective when
      ``dirichlet_epsilon > 0``.
    * ``dirichlet_epsilon`` — mix weight for Dirichlet noise:
      ``prior ← (1 - eps) * prior + eps * Dir(alpha)``. ``0`` (the
      default) disables noise entirely — required during evaluation, but
      AlphaZero self-play sets it to ``0.25``.
    * ``temperature`` — softening exponent for
      :meth:`MCTSResult.policy_distribution`. ``1.0`` (the default)
      returns visit-proportional probabilities suitable for sampling
      during early-game self-play; ``0.0`` returns a one-hot on the
      argmax suitable for late-game / evaluation play. ``run_mcts``
      itself always returns the argmax action; the caller decides
      whether to sample from :meth:`policy_distribution` instead.
    """

    rollouts: int = 100
    c_puct: float = 2.0
    seed: int = 0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0
    temperature: float = 1.0


@dataclass
class MCTSResult:
    """Output of one ``run_mcts`` call.

    * ``action`` — the chosen typed action (most-visited child of the root).
    * ``visit_counts`` — ``N(s, a)`` per legal action, parallel to
      ``legal_actions``.
    * ``legal_actions`` — the legal actions at the root, in the order used
      for the prior and visit-count vectors.
    """

    action: Action
    visit_counts: np.ndarray
    legal_actions: list[Action]

    def policy_distribution(self, temperature: float) -> np.ndarray:
        """Visit-count distribution softened by ``temperature``.

        * ``temperature == 0.0`` returns a one-hot on the most-visited
          action (deterministic exploitation). Ties are broken by the
          first index (numpy ``argmax`` semantics) so the result is
          stable.
        * ``temperature == 1.0`` returns
          ``visit_counts / visit_counts.sum()`` (sample proportional to
          N).
        * Other ``temperature > 0`` returns
          ``visit_counts ** (1/T) / sum(...)``: ``T → 0⁺`` concentrates
          on argmax, ``T → ∞`` flattens to uniform.

        Used as the policy *target* for AlphaZero training (always at
        ``T == 1.0``, regardless of the action actually sampled during
        self-play). The same distribution can be used at sampling time
        with a different ``T`` for the temperature schedule.
        """
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {temperature}")
        n = self.visit_counts.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if temperature == 0.0:
            out = np.zeros(n, dtype=np.float64)
            out[int(self.visit_counts.argmax())] = 1.0
            return out
        # ``counts`` is non-negative integer-valued; the ** below stays
        # finite for any temperature > 0.
        counts = self.visit_counts.astype(np.float64, copy=False)
        powered = counts ** (1.0 / temperature)
        total = powered.sum()
        if total <= 0.0:
            # No visits anywhere — return uniform so the distribution still sums to 1.
            return np.full(n, 1.0 / n, dtype=np.float64)
        return powered / total


# ----------------------------------------------------------------------
# Internal tree data structures
# ----------------------------------------------------------------------


@dataclass
class _Edge:
    """One action from a :class:`_DecisionNode`.

    Edges are stochastic-aware: for deterministic transitions and for
    closed-loop stochastic actions (dev-card draws, steals), the edge holds
    a single ``child``. For ``RollDiceAction`` the edge holds 11 chance
    outcomes weighted by :data:`DICE_SUM_PMF`, expanded lazily on first
    visit by sum.
    """

    action: Action
    prior: float
    n_players: int
    # Visit accumulators (Q is derived as W / N when needed).
    N: int = 0
    W: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # Deterministic / sampled-once: single child reference; None until expanded.
    child: "_DecisionNode | None" = None
    # Dice chance children: indexed by sum (2..12 → index 0..10). Lazy.
    dice_children: list["_DecisionNode | None"] | None = None

    def __post_init__(self) -> None:
        if self.W.size == 0:
            self.W = np.zeros(self.n_players, dtype=np.float64)
        if isinstance(self.action, RollDiceAction):
            self.dice_children = [None] * 11

    @property
    def is_chance(self) -> bool:
        return self.dice_children is not None

    def q_value(self, seat_idx: int) -> float:
        """Mean value of this edge from ``seat_idx``'s perspective."""
        return float(self.W[seat_idx] / self.N) if self.N > 0 else 0.0


@dataclass
class _DecisionNode:
    """One state in the tree, expanded once on first visit.

    Edges live in :attr:`edges`, parallel to :attr:`legal_actions`. A node is
    *terminal* iff the engine reports the state as such; terminal nodes are
    never expanded and back up the deterministic outcome instead of the
    network's value head.
    """

    state: GameState
    legal_actions: list[Action]
    edges: list[_Edge]
    acting_seat_idx: int
    is_terminal: bool
    n_players: int
    # The 4-vector value used to back up at this node (network or terminal).
    leaf_value: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ----------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------


def run_mcts(
    root_state: GameState,
    evaluator: Evaluator,
    config: MCTSConfig,
    *,
    engine: GameEngine | None = None,
) -> MCTSResult:
    """Run PUCT MCTS rooted at ``root_state``; return the most-visited action.

    The caller passes a :class:`GameEngine` so the search can reuse one
    randomizer across all simulated transitions (relevant for reproducibility
    of any non-dice stochasticity that the engine resolves internally). If
    omitted, a fresh seeded engine is constructed.
    """
    if engine is None:
        from domain.engine.randomizer import SeededRandomizer

        engine = GameEngine(SeededRandomizer(config.seed))

    n_players = len(root_state.config.player_ids)
    rng = random.Random(config.seed)

    root = _make_node(root_state, evaluator, engine, n_players)
    if root.is_terminal:
        raise ValueError("run_mcts called on a terminal state")
    if not root.legal_actions:
        raise ValueError("run_mcts called on a state with no legal actions")

    # Inject Dirichlet exploration noise at the root only — children
    # expanded during simulation keep their network priors untouched.
    # This is the standard AlphaZero pattern; without it self-play
    # converges on a narrow set of openings because the prior is the
    # only thing distinguishing equally-Q'd actions on the first visit.
    if config.dirichlet_epsilon > 0.0:
        # Use a numpy RNG seeded from the same source as the dice RNG so
        # the whole search is reproducible from one ``config.seed``.
        np_rng = np.random.default_rng(config.seed)
        _apply_root_dirichlet_noise(root, config, np_rng)

    for _ in range(config.rollouts):
        _simulate(root, evaluator, engine, config, rng)

    visit_counts = np.array([e.N for e in root.edges], dtype=np.int64)
    # Tie-break by prior (visit_counts argmax may have ties on tiny searches).
    best = int(visit_counts.argmax())
    if visit_counts.max() == 0:
        # Shouldn't happen with rollouts > 0, but degenerate safety.
        priors = np.array([e.prior for e in root.edges])
        best = int(priors.argmax())
    return MCTSResult(
        action=root.legal_actions[best],
        visit_counts=visit_counts,
        legal_actions=list(root.legal_actions),
    )


# ----------------------------------------------------------------------
# Simulation step
# ----------------------------------------------------------------------


def _simulate(
    root: _DecisionNode,
    evaluator: Evaluator,
    engine: GameEngine,
    config: MCTSConfig,
    rng: random.Random,
) -> None:
    """One MCTS simulation: select → expand → back up.

    Walks down through expanded nodes via PUCT, expands the first leaf it
    reaches, then backs up the leaf's value vector along the visited edges.
    """
    path: list[tuple[_DecisionNode, _Edge]] = []
    node = root
    while True:
        if node.is_terminal:
            # Terminal: back up the resolved outcome.
            value = node.leaf_value
            break
        edge_idx = _select_edge(node, config.c_puct)
        edge = node.edges[edge_idx]
        path.append((node, edge))
        child = _descend(node, edge, engine, evaluator, rng)
        # New leaf? Expand and stop.
        if child.leaf_value.size == 0:
            # Not yet evaluated → expand now.
            _evaluate_leaf(child, evaluator)
            value = child.leaf_value
            break
        node = child

    _backup(path, value)


def _select_edge(node: _DecisionNode, c_puct: float) -> int:
    """Pick the edge with the highest PUCT score from ``node``."""
    n_total = sum(e.N for e in node.edges)
    sqrt_total = math.sqrt(n_total) if n_total > 0 else 1.0
    best_idx = 0
    best_score = -math.inf
    seat = node.acting_seat_idx
    for i, edge in enumerate(node.edges):
        q = edge.q_value(seat)
        u = c_puct * edge.prior * sqrt_total / (1 + edge.N)
        score = q + u
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _descend(
    parent: _DecisionNode,
    edge: _Edge,
    engine: GameEngine,
    evaluator: Evaluator,
    rng: random.Random,
) -> _DecisionNode:
    """Resolve the child node we land on after taking ``edge`` from ``parent``.

    For deterministic / closed-loop edges, returns (and creates on first
    visit) the cached single child. For dice chance edges, samples a sum by
    the 2d6 PMF and creates the child for that sum lazily.
    """
    if edge.is_chance:
        sum_idx = _sample_dice_sum(rng)
        assert edge.dice_children is not None
        child = edge.dice_children[sum_idx]
        if child is None:
            d1, d2 = _DICE_REPR[sum_idx]
            new_state = _apply_with_fixed_dice(
                engine, parent.state, edge.action, d1, d2
            )
            child = _make_node(
                new_state, evaluator, engine, parent.n_players, lazy=True
            )
            edge.dice_children[sum_idx] = child
        return child

    if edge.child is None:
        new_state = engine.apply_action(parent.state, edge.action).state
        edge.child = _make_node(
            new_state, evaluator, engine, parent.n_players, lazy=True
        )
    return edge.child


def _sample_dice_sum(rng: random.Random) -> int:
    """Sample a dice-sum index 0..10 from the 2d6 PMF using ``rng``."""
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(DICE_SUM_PMF):
        cum += p
        if r < cum:
            return i
    return len(DICE_SUM_PMF) - 1


def _apply_with_fixed_dice(
    engine: GameEngine,
    state: GameState,
    action: Action,
    d1: int,
    d2: int,
) -> GameState:
    """Apply a ``RollDiceAction`` with predetermined dice values.

    Builds a transient :class:`GameEngine` whose randomizer returns the
    supplied dice on the first call and delegates everything else to the
    engine's existing randomizer. The original engine is untouched.
    """
    transient = GameEngine(_OneShotDiceRandomizer(engine._rng, (d1, d2)))  # noqa: SLF001
    return transient.apply_action(state, action).state


class _OneShotDiceRandomizer:
    """Returns predetermined dice once, then delegates to a base randomizer."""

    def __init__(self, base: Randomizer, dice: tuple[int, int]) -> None:
        self._base = base
        self._dice: tuple[int, int] | None = dice

    def roll_dice(self) -> tuple[int, int]:
        if self._dice is not None:
            out = self._dice
            self._dice = None
            return out
        return self._base.roll_dice()

    def shuffle_dev_deck(self, cards):
        return self._base.shuffle_dev_deck(cards)

    def choose_stolen_resource(self, resources):
        return self._base.choose_stolen_resource(resources)

    def shuffled(self, items):
        return self._base.shuffled(items)

    def choice(self, items):
        return self._base.choice(items)


# ----------------------------------------------------------------------
# Node construction / leaf evaluation
# ----------------------------------------------------------------------


def _make_node(
    state: GameState,
    evaluator: Evaluator,
    engine: GameEngine,
    n_players: int,
    *,
    lazy: bool = False,
) -> _DecisionNode:
    """Build a tree node from a state.

    For the root we always evaluate immediately (``lazy=False``). For
    children created during descent we defer evaluation until the
    simulation actually reaches them as a leaf (``lazy=True``); this keeps
    expansion cost paid only along visited paths.
    """
    is_terminal = state.is_terminal()
    if is_terminal:
        legal: list[Action] = []
        edges: list[_Edge] = []
        seat_idx = 0  # Unused; terminal nodes never PUCT-select.
        node = _DecisionNode(
            state=state,
            legal_actions=legal,
            edges=edges,
            acting_seat_idx=seat_idx,
            is_terminal=True,
            n_players=n_players,
            leaf_value=_terminal_value(state, n_players),
        )
        return node

    legal = engine.legal_actions(state)
    acting_pid = acting_player(state)
    seat_idx = _seat_index(state, acting_pid)

    node = _DecisionNode(
        state=state,
        legal_actions=legal,
        edges=[],  # filled by _evaluate_leaf when the node is first visited
        acting_seat_idx=seat_idx,
        is_terminal=False,
        n_players=n_players,
    )
    if not lazy:
        _evaluate_leaf(node, evaluator)
    return node


def _apply_root_dirichlet_noise(
    root: _DecisionNode,
    config: MCTSConfig,
    np_rng: np.random.Generator,
) -> None:
    """Blend Dirichlet noise into the root's edge priors in-place.

    Formula: ``P'(s, a) = (1 - eps) * P(s, a) + eps * eta(a)`` where
    ``eta ~ Dir(alpha, ..., alpha)`` over the root's legal actions.
    Mutates ``root.edges[i].prior`` directly; children expanded later
    during simulation are unaffected.
    """
    n = len(root.edges)
    if n == 0:
        return
    eps = config.dirichlet_epsilon
    eta = np_rng.dirichlet([config.dirichlet_alpha] * n)
    for i, edge in enumerate(root.edges):
        edge.prior = float((1.0 - eps) * edge.prior + eps * eta[i])


def _evaluate_leaf(node: _DecisionNode, evaluator: Evaluator) -> None:
    """Run the evaluator on ``node`` and populate priors + leaf value."""
    if node.is_terminal:
        return  # leaf_value already set in _make_node
    priors, value_vec = evaluator.evaluate(node.state, node.legal_actions)
    if priors.shape != (len(node.legal_actions),):
        raise ValueError(
            f"evaluator priors shape {priors.shape} != "
            f"(len(legal)={len(node.legal_actions)},)"
        )
    if value_vec.shape != (node.n_players,):
        raise ValueError(
            f"evaluator value shape {value_vec.shape} != ({node.n_players},)"
        )
    node.edges = [
        _Edge(action=a, prior=float(p), n_players=node.n_players)
        for a, p in zip(node.legal_actions, priors)
    ]
    node.leaf_value = value_vec.astype(np.float64, copy=False)


def _terminal_value(state: GameState, n_players: int) -> np.ndarray:
    """Per-player value at a terminal state.

    Winner gets 1.0; everyone else gets 0.0. Stalemate gets 0.0 for all
    seats (no one wins) — same convention the sparse reward uses.
    """
    out = np.zeros(n_players, dtype=np.float64)
    if state.winner is None:
        return out
    for i, pid in enumerate(state.config.player_ids):
        if pid == state.winner:
            out[i] = 1.0
    return out


# ----------------------------------------------------------------------
# Backup
# ----------------------------------------------------------------------


def _backup(path: list[tuple[_DecisionNode, _Edge]], value: np.ndarray) -> None:
    """Propagate the leaf's per-player value vector up the visited edges."""
    for _node, edge in path:
        edge.N += 1
        edge.W += value


def _seat_index(state: GameState, pid: PlayerID) -> int:
    """Index of ``pid`` in the state's canonical player order."""
    for i, p in enumerate(state.config.player_ids):
        if p == pid:
            return i
    raise ValueError(f"player {pid} not in state.config.player_ids")
