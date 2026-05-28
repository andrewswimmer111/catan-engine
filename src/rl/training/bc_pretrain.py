"""Behavioural-cloning pretrain from the rule-based :class:`HeuristicAgent`.

Generates expert trajectories by playing the deterministic
:class:`rl.agents.heuristic_agent.HeuristicAgent` in all four seats on the
pure engine, records every genuine decision as a
:class:`rl.training.self_play.SelfPlayTransition` whose policy target is a
one-hot on the expert's chosen action, and trains a GNN policy/value
network to imitate it via masked policy cross-entropy plus per-seat value
MSE.

Why this exists
---------------

The AZ self-play loop plateaued (#015) on a settlement-EXPANSION ceiling:
the learner builds roads that don't open new settlement vertices, so it
places ~1 settlement/game versus the heuristic's ~2 and its VP caps at
~4-5. The heuristic does not have this problem — it scores roads by the
settlement potential they unlock
(:func:`rl.agents._heuristic_rules.best_road`). Cloning the heuristic
teaches the road→settlement build chain directly; the resulting
checkpoint is a warm start for ``train_alphazero.py --init-from``.

A BC sample is shaped exactly like a self-play transition (one-hot policy
target instead of an MCTS visit distribution), so the supervised loss
reuses the AZ loss formula verbatim: masked policy CE + per-seat value
MSE. The value target is the rotated terminal outcome, computed by the
same helpers self-play uses, so the warm-started value head sits on the
same scale as AZ's and a vector→vector ``--init-from`` carries both heads.

Known warts (documented per project convention):

* **Expert-state distribution only.** All four seats play the
  deterministic heuristic, so the visited-state distribution is the
  heuristic's own. The cloned policy may behave arbitrarily off this
  distribution; that is acceptable for an AZ *warm start* (AZ's own
  exploration refines off-distribution behaviour). State diversity comes
  from per-game board + dice seed variation, not from action noise.
* **Forced / discard moves skipped.** Only decisions with more than one
  *encodable* legal action are recorded (``action_mask.sum() > 1``). This
  drops forced moves (lone ``RollDice`` / ``EndTurn`` / robber sequels)
  and the encoder's collapsed discard slot — neither carries a learnable
  policy signal, and including them would swamp the cross-entropy with
  trivial single-class examples.
* **Value-target calibration tracks ``victory_point_target``.** At the
  standard 10-VP threshold the heuristic stalemates ~99% of games, so the
  value target collapses into the stalemate band (low variance). Run BC at
  the same ``victory_point_target`` as the downstream AZ run (e.g. 6) for a
  value head that actually saw ``+1`` wins; the policy clone is unaffected
  either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from controller.session import GameSnapshot
from domain.engine.game_engine import GameEngine
from domain.engine.player_view import make_player_view
from domain.engine.randomizer import SeededRandomizer
from domain.game.config import GameConfig
from domain.ids import PlayerID
from rl.acting_player import acting_player
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.policy_agent import PolicyAgent
from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.action import ActionEncoder
from rl.encoding.graph_observation import GraphObservationEncoder
from rl.stalemate_value import StalemateValueConfig
from rl.training.self_play import (
    SelfPlayTransition,
    # The value-target convention (terminal outcome, viewer-as-slot-0
    # rotation, VP-aux snapshot) MUST match self-play exactly or a
    # vector→vector ``--init-from`` warm start would feed AZ value targets
    # on a different scale. Import the canonical helpers rather than
    # re-deriving them so there is a single source of truth.
    _rotate_to_acting_seat,
    _terminal_outcome,
    _vp_aux_target,
)

__all__ = [
    "BCConfig",
    "generate_bc_transitions",
    "train_bc",
]


_DEFAULT_PLAYER_IDS: tuple[PlayerID, ...] = tuple(PlayerID(i) for i in range(1, 5))


@dataclass(frozen=True)
class BCConfig:
    """Knobs for one heuristic behavioural-cloning pretrain.

    The data-generation block controls the expert trajectory corpus; the
    optimisation block controls the supervised fit over it. Defaults are
    sized for a fast CPU run (heuristic self-play needs no network forward
    and no MCTS, so generation is cheap).
    """

    # --- Data generation ---
    n_games: int = 300
    """Heuristic self-play games to roll out. Each game's four seats all
    play the heuristic, so every game contributes the decisions of all
    four seats on one board/dice seed."""

    victory_point_target: int = 10
    """VP threshold for ending a game in a real win, plumbed into the
    per-game :class:`GameConfig`. Match this to the downstream AZ run's
    ``--win-vp`` so the cloned value head is calibrated for the same
    threshold (see the module docstring's calibration wart)."""

    max_moves: int = 1000
    """Hard per-game move cap; reaching it truncates the game as a
    stalemate (same convention as self-play)."""

    stalemate: StalemateValueConfig = field(default_factory=StalemateValueConfig)
    """Shape of the per-seat stalemate value target — kept identical to
    the AZ default so warm-started value targets match."""

    player_ids: tuple[PlayerID, ...] = _DEFAULT_PLAYER_IDS

    # --- Optimisation ---
    epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    value_coef: float = 1.0
    """Weight on the per-seat value MSE. The policy CE is the primary BC
    signal (it teaches settlement expansion); the value term warms the
    AZ value head. Set to 0.0 for a policy-only clone."""
    max_grad_norm: float = 5.0
    seed: int = 0


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------


@dataclass
class _PendingBC:
    """Per-decision accumulator before the terminal outcome is known."""

    obs: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    acting_seat_idx: int
    vp_aux_target: np.ndarray


def generate_bc_transitions(
    config: BCConfig,
    *,
    seed: int | None = None,
) -> list[SelfPlayTransition]:
    """Roll out ``config.n_games`` heuristic games; return the BC samples.

    Every game is played by the deterministic heuristic in all four seats.
    Each genuine decision (more than one encodable legal action) becomes a
    :class:`SelfPlayTransition` with a one-hot ``mcts_policy`` on the
    expert's chosen action and a value target rotated to acting-as-slot-0.
    Game variety comes from the per-game engine seed (board layout + dice
    rolls); the policy itself is deterministic.
    """
    base_seed = config.seed if seed is None else seed
    agent = HeuristicAgent()
    # Build the encoders once and reuse across games; they're stateless
    # apart from the seat→index map, which is fixed by ``player_ids``.
    obs_encoder = GraphObservationEncoder()
    action_encoder = ActionEncoder(list(config.player_ids))

    transitions: list[SelfPlayTransition] = []
    for g in range(config.n_games):
        game_seed = base_seed * 1_000_003 + g
        transitions.extend(
            _play_expert_game(
                agent=agent,
                obs_encoder=obs_encoder,
                action_encoder=action_encoder,
                config=config,
                game_seed=game_seed,
            )
        )
    return transitions


def _play_expert_game(
    *,
    agent: HeuristicAgent,
    obs_encoder,
    action_encoder,
    config: BCConfig,
    game_seed: int,
) -> list[SelfPlayTransition]:
    """Drive one all-heuristic game; return its recorded BC transitions."""
    pids = list(config.player_ids)
    n_players = len(pids)
    engine = GameEngine(SeededRandomizer(game_seed))
    state = engine.new_game(
        GameConfig(
            player_ids=pids,
            seed=game_seed,
            victory_point_target=config.victory_point_target,
        )
    )

    pending: list[_PendingBC] = []
    move_idx = 0
    while not state.is_terminal() and move_idx < config.max_moves:
        legal = engine.legal_actions(state)
        if not legal:
            break

        acting_pid = acting_player(state)
        acting_idx = pids.index(acting_pid)
        snap = GameSnapshot(
            state=state, step_index=move_idx, last_action=None, last_events=()
        )
        chosen = agent.choose(snap, legal)
        if chosen is None:
            break

        mask = action_encoder.mask(legal)
        # Record only states with a genuine choice in the encoded action
        # space. ``mask.sum() <= 1`` covers forced moves and the collapsed
        # discard slot — both are single-class and carry no policy signal.
        if int(mask.sum()) > 1:
            try:
                idx = action_encoder.encode(chosen)
            except ValueError:
                idx = None
            if idx is not None and bool(mask[idx]):
                obs = obs_encoder.encode(make_player_view(state, acting_pid))
                policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                policy_target[idx] = 1.0
                pending.append(
                    _PendingBC(
                        obs=obs,
                        action_mask=mask,
                        policy_target=policy_target,
                        acting_seat_idx=acting_idx,
                        vp_aux_target=_vp_aux_target(state, pids, acting_idx),
                    )
                )

        state = engine.apply_action(state, chosen).state
        move_idx += 1

    outcome = _terminal_outcome(state, n_players, config.stalemate)
    return [
        SelfPlayTransition(
            obs=p.obs,
            action_mask=p.action_mask,
            mcts_policy=p.policy_target,
            acting_seat_idx=p.acting_seat_idx,
            value_target=_rotate_to_acting_seat(outcome, p.acting_seat_idx),
            vp_aux_target=p.vp_aux_target,
        )
        for p in pending
    ]


# ----------------------------------------------------------------------
# Supervised training
# ----------------------------------------------------------------------


def train_bc(
    agent: PolicyAgent,
    transitions: list[SelfPlayTransition],
    config: BCConfig,
    *,
    on_epoch_end: Callable[[dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Fit ``agent``'s policy/value network to the expert ``transitions``.

    Runs ``config.epochs`` passes over the shuffled dataset in mini-batches
    (without replacement within an epoch), optimising masked policy
    cross-entropy + ``value_coef`` * per-seat value MSE with Adam. Returns
    the last epoch's summary metrics; ``on_epoch_end`` (if given) is called
    with the per-epoch summary after every epoch — the driver uses it to
    print progress and rewrite ``progress.md``.

    Requires a vector-value-head model: the value target is a per-seat
    vector, matching ``GNNPolicyValue(value_kind="vector")``. A scalar-head
    model is rejected loudly (mirrors :class:`AlphaZeroTrainer`).
    """
    if getattr(agent.model, "value_kind", "scalar") != "vector":
        raise ValueError(
            "train_bc requires a learner with a vector value head; got "
            f"value_kind={getattr(agent.model, 'value_kind', 'scalar')!r}. "
            "Build the GNN with arch.value_kind='vector' so the BC value "
            "target (a per-seat vector) lines up and the checkpoint can "
            "warm-start AZ via --init-from."
        )
    if not transitions:
        raise ValueError("train_bc received an empty transition list")

    model = agent.model
    device = agent.device
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    n = len(transitions)
    order = list(range(n))
    shuffle_rng = random.Random(config.seed)

    last_summary: dict[str, float] = {}
    global_step = 0
    for epoch in range(config.epochs):
        shuffle_rng.shuffle(order)
        model.train()
        sums = _zero_metrics()
        n_batches = 0
        for start in range(0, n, config.batch_size):
            batch = [transitions[i] for i in order[start : start + config.batch_size]]
            step_metrics = _bc_train_step(model, batch, optimizer, config, device)
            for k in sums:
                sums[k] += step_metrics[k]
            n_batches += 1
            global_step += 1

        denom = max(n_batches, 1)
        last_summary = {
            "epoch": float(epoch + 1),
            "global_step": float(global_step),
            "n_samples": float(n),
            **{k: v / denom for k, v in sums.items()},
        }
        if on_epoch_end is not None:
            on_epoch_end(last_summary)

    return last_summary


def _bc_train_step(
    model: nn.Module,
    batch: list[SelfPlayTransition],
    optimizer: torch.optim.Optimizer,
    config: BCConfig,
    device: torch.device,
) -> dict[str, float]:
    """One Adam step over a mini-batch of expert transitions."""
    obs_t = torch.as_tensor(
        np.stack([t.obs for t in batch]), dtype=torch.float32, device=device
    )
    mask_t = torch.as_tensor(
        np.stack([t.action_mask for t in batch]), dtype=torch.bool, device=device
    )
    policy_target_t = torch.as_tensor(
        np.stack([t.mcts_policy for t in batch]), dtype=torch.float32, device=device
    )
    value_target_t = torch.as_tensor(
        np.stack([t.value_target for t in batch]), dtype=torch.float32, device=device
    )

    out = model(obs_t, mask_t)
    # Policy CE against the one-hot expert target. The forward masks illegal
    # slots to MASK_FILL_VALUE so their log-prob is ~-inf; the expert target
    # is 0 on illegal slots, but ``0 * -inf == nan``, so zero illegal
    # log-probs before the dot product (identical to the AZ trainer).
    log_probs = torch.log_softmax(out.logits, dim=-1)
    log_probs = torch.where(mask_t, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(policy_target_t * log_probs).sum(dim=-1).mean()

    value_loss = (out.value - value_target_t).pow(2).mean()
    loss = policy_loss + config.value_coef * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()

    # Top-1 imitation accuracy: the model's argmax (logits already masked to
    # legal) vs the expert's chosen index. The headline "is the clone
    # working" metric — policy_loss alone is hard to read across action
    # spaces of different legal-set sizes.
    with torch.no_grad():
        pred = out.logits.argmax(dim=-1)
        expert = policy_target_t.argmax(dim=-1)
        accuracy = (pred == expert).float().mean()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(loss.item()),
        "grad_norm": float(grad_norm.item()),
        "accuracy": float(accuracy.item()),
    }


def _zero_metrics() -> dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
        "grad_norm": 0.0,
        "accuracy": 0.0,
    }
