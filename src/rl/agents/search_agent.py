"""MCTS-wrapped policy as an engine-facing :class:`Agent`.

:class:`SearchAgent` takes a trained :class:`PolicyAgent` and wraps it with
PUCT MCTS at inference: every ``choose`` runs ``rollouts`` simulations
rooted at the current state, using the policy net's masked softmax as the
prior and its value head (from each seat's perspective) as the leaf
evaluator. Returns the most-visited root action as a typed
:class:`Action`.

Composition: ``SearchAgent`` satisfies the :class:`Agent` protocol and
slots into the tournament harness, the CLI eval path, and
:class:`PlacementOverrideAgent` interchangeably — no other call site
changes.

Forced-action short-circuit: when ``len(legal) == 1`` the search is
skipped and the single legal action is returned. MCTS contributes nothing
on forced moves (``RollDice``, ``EndTurn``, single robber sequels) and
short-circuiting saves the per-call evaluator cost.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.engine.game_engine import GameEngine
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.game.state import GameState
from rl.agents.policy_agent import PolicyAgent
from rl.search.mcts import MCTSConfig, run_mcts

__all__ = ["SearchAgent", "NetworkEvaluator"]


# Used for the fallback prior when the encoder represents no legal action
# (e.g. only domestic-trade proposals remain). The network can't produce a
# meaningful prior in this case; uniform over legal is the right default.
_UNIFORM_PRIOR_FALLBACK_TAG: Final = "uniform-fallback"


class SearchAgent:
    """PUCT MCTS wrapped around a trained :class:`PolicyAgent`.

    The wrapped policy is queried once per leaf expansion via
    :class:`NetworkEvaluator`. Models with a per-seat ``vector`` value
    head only need a single forward (the acting seat's perspective) per
    leaf; models with a ``scalar`` value head need one batched forward
    across all four player perspectives to recover the per-seat value
    vector MCTS backs up. In both cases the leaf's acting player's
    masked policy logits are softmaxed and gathered onto the
    legal-action list to produce the MCTS prior.
    """

    def __init__(
        self,
        policy: PolicyAgent,
        config: MCTSConfig | None = None,
    ) -> None:
        self._policy = policy
        self._config = config if config is not None else MCTSConfig()
        self._evaluator = NetworkEvaluator(policy)

    @property
    def policy(self) -> PolicyAgent:
        return self._policy

    @property
    def config(self) -> MCTSConfig:
        return self._config

    def choose(self, snap: GameSnapshot, legal: list[Action]) -> Action | None:
        """Pick a typed action via MCTS rooted at ``snap.state``.

        Short-circuits without running MCTS when ``legal`` has zero or one
        entries — MCTS adds no value when the move is forced.
        """
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        # Fresh engine per call: state cloning is already free inside
        # apply_action, but we want a fresh randomizer for any non-dice
        # stochasticity the engine resolves internally during simulated
        # transitions. Reusing the play-time engine would entangle search
        # randomness with game-time randomness.
        engine = GameEngine(SeededRandomizer(self._config.seed))
        result = run_mcts(snap.state, self._evaluator, self._config, engine=engine)
        return result.action


class NetworkEvaluator:
    """:class:`rl.search.mcts.Evaluator` backed by a :class:`PolicyAgent`.

    Two leaf-eval paths depending on the wrapped model's value head:

    * **Vector head** (``value_kind="vector"``, AZ-style). A single
      forward from the acting seat's perspective gives both the acting
      seat's masked logits (priors) and the full ``(N_PLAYERS,)`` value
      vector — already rotated to viewer-as-slot-0. The vector is rolled
      back to absolute seat indexing for MCTS backup.
    * **Scalar head** (``value_kind="scalar"``, PPO back-compat). One
      batched forward across all four seat perspectives gives the
      per-seat value vector slot-by-slot; the acting seat's row supplies
      priors.

    If no legal action is encodable (mask is all False), the priors fall
    back to uniform over ``legal`` — the same defensive convention
    :meth:`PolicyAgent.choose` uses.
    """

    def __init__(self, policy: PolicyAgent) -> None:
        self._policy = policy
        self._model = policy.model
        self._obs_encoder = policy.obs_encoder
        self._action_encoder = policy.action_encoder
        self._device = policy.device
        # Per the model contract: GNN exposes ``value_kind`` via the arch
        # dataclass; MLP exposes it as a class attribute. Defaulting to
        # ``"scalar"`` is safe for any future model that doesn't declare.
        self._value_kind: str = getattr(self._model, "value_kind", "scalar")

    def evaluate(
        self, state: GameState, legal: list[Action]
    ) -> tuple[np.ndarray, np.ndarray]:
        pids = list(state.config.player_ids)
        n_players = len(pids)
        acting_idx = _find_acting_seat(state)
        mask = self._action_encoder.mask(legal)

        if self._value_kind == "vector":
            value_vec, logits_acting = self._eval_vector(
                state, pids[acting_idx], mask, acting_idx, n_players
            )
        else:
            value_vec, logits_acting = self._eval_scalar(
                state, pids, mask, acting_idx, n_players
            )

        priors = _priors_over_legal(
            logits_acting=logits_acting,
            mask=mask,
            legal=legal,
            action_encoder=self._action_encoder,
        )
        return priors, value_vec

    def _eval_vector(
        self,
        state: GameState,
        acting_pid,
        mask: np.ndarray,
        acting_idx: int,
        n_players: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """One forward at the acting seat; unrotate the value vec to absolute seats."""
        obs = self._obs_encoder.encode(make_player_view(state, acting_pid))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self._device).unsqueeze(0)
        with torch.no_grad():
            out = self._model(obs_t, mask_t)

        # The value head outputs (1, n_players) with slot 0 = acting seat,
        # slot k = the seat at turn-order offset k from acting. Roll to
        # absolute seat order so MCTS backup can index by seat directly.
        rotated = out.value[0].detach().cpu().numpy().astype(np.float64)
        if rotated.shape != (n_players,):
            raise ValueError(
                f"vector value-head produced shape {rotated.shape}, "
                f"expected ({n_players},)"
            )
        value_vec = np.roll(rotated, acting_idx)
        return value_vec, out.logits[0]

    def _eval_scalar(
        self,
        state: GameState,
        pids,
        mask: np.ndarray,
        acting_idx: int,
        n_players: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Four perspectives in one batch; concatenate the scalar values."""
        obs_batch = np.stack(
            [self._obs_encoder.encode(make_player_view(state, p)) for p in pids]
        )
        mask_batch = np.tile(mask, (n_players, 1))

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self._device)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self._device)
        with torch.no_grad():
            out = self._model(obs_t, mask_t)

        value_vec = out.value.detach().cpu().numpy().astype(np.float64)
        return value_vec, out.logits[acting_idx]


def _find_acting_seat(state: GameState) -> int:
    """Index of the player who must act next at this state."""
    from rl.acting_player import acting_player

    acting = acting_player(state)
    for i, pid in enumerate(state.config.player_ids):
        if pid == acting:
            return i
    raise ValueError(f"acting player {acting} not in state's player_ids")


def _priors_over_legal(
    *,
    logits_acting: torch.Tensor,
    mask: np.ndarray,
    legal: list[Action],
    action_encoder,
) -> np.ndarray:
    """Project the masked policy distribution onto the ``legal`` list.

    Returns a length-``len(legal)`` non-negative array summing to 1. For
    legal actions the encoder can't represent (e.g. domestic trade
    proposals), assigns prior 0 then renormalises. If no legal action is
    representable, falls back to uniform.
    """
    n = len(legal)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if not mask.any():
        return np.full(n, 1.0 / n, dtype=np.float64)

    masked_logits = logits_acting.clone()
    masked_logits[~torch.as_tensor(mask, dtype=torch.bool, device=logits_acting.device)] = -1e9
    probs = torch.softmax(masked_logits, dim=-1).detach().cpu().numpy().astype(np.float64)

    priors = np.zeros(n, dtype=np.float64)
    for i, action in enumerate(legal):
        try:
            idx = action_encoder.encode(action)
        except ValueError:
            continue
        priors[i] = probs[idx]

    total = priors.sum()
    if total <= 0.0:
        # The encoder represents none of the legal actions; fall uniform.
        return np.full(n, 1.0 / n, dtype=np.float64)
    return priors / total
