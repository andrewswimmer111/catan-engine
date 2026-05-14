"""Play one game and emit an :class:`EpisodeRecord` for offline analysis.

The recorder is the inverse of the rollout worker — it doesn't store
trainable transitions, just the per-step learner output the GUI overlay
needs (action distribution, value estimate, action index) plus a faithful
:class:`ReplayLog` for the engine to replay later.

Only the learner's steps appear in the resulting :class:`StepRecord` list.
Opponent steps are still applied to the env, so the replay is complete, but
they don't produce step records — the GUI's overlay aligns steps to the
``last_action.player == learner_seat`` boundary inside the replay.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from controller.agents import Agent
from controller.session import GameSnapshot
from domain.engine.player_view import make_player_view
from domain.enums import EndReason
from domain.ids import PlayerID
from domain.rules.victory import compute_victory_points
from rl.agents.heuristic_agent import heuristic_discard
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import DiscardSentinel
from rl.env.catan_env import CatanEnv
from rl.replay.dataset import EpisodeRecord, StepRecord
from serialization.codec import encode_action, encode_event
from serialization.replay import ReplayLog

__all__ = ["play_episode"]


def play_episode(
    env: CatanEnv,
    learner: PolicyAgent,
    learner_seat: PlayerID,
    opponents: dict[PlayerID, Agent],
    metadata: dict[str, Any] | None = None,
) -> EpisodeRecord:
    """Drive one game on ``env`` and return its archived record.

    ``env`` is consumed — callers should pass a freshly-constructed env.
    Opponents are dispatched through their :meth:`Agent.choose`; the
    learner's per-step distribution is captured via
    :meth:`PolicyAgent.act_with_dist` so the resulting :class:`StepRecord`
    contains the full masked softmax for overlay display.
    """
    config = env.state.config
    player_ids = list(config.player_ids)
    actions_encoded: list[dict] = []
    events_encoded: list[list[dict]] = []
    steps: list[StepRecord] = []
    learner_step_indices: list[int] = []

    snap = GameSnapshot(
        state=env.state, step_index=0, last_action=None, last_events=()
    )
    step_index = 0
    done = False
    while not done:
        acting = env.current_agent
        legal = env.legal_actions()
        if acting == learner_seat:
            step_record, action = _record_learner_step(env, learner, learner_seat)
            steps.append(step_record)
            learner_step_indices.append(step_index)
        else:
            agent = opponents[acting]
            action = agent.choose(snap, legal)
            if action is None:
                break

        actions_encoded.append(encode_action(action))
        _, reward, done, info = env.step(action)
        events_encoded.append([encode_event(e) for e in info["last_events"]])

        if acting == learner_seat and steps:
            # Patch the actual realised reward onto the just-recorded step.
            steps[-1] = _with_reward(steps[-1], float(reward))

        step_index += 1
        snap = GameSnapshot(
            state=env.state,
            step_index=step_index,
            last_action=action,
            last_events=tuple(info["last_events"]),
        )

    final_state = env.state
    final_vps = {pid: compute_victory_points(final_state, pid) for pid in player_ids}
    md = dict(metadata or {})
    md.setdefault("learner_seat", int(learner_seat))
    md.setdefault("learner_step_indices", list(learner_step_indices))
    md.setdefault("end_reason", (final_state.end_reason or EndReason.STALEMATE_NO_PROGRESS).name)

    return EpisodeRecord(
        replay_log=ReplayLog(
            config=config, actions=actions_encoded, events=events_encoded
        ),
        steps=steps,
        final_vps=final_vps,
        winner=final_state.winner,
        metadata=md,
    )


def _record_learner_step(
    env: CatanEnv,
    learner: PolicyAgent,
    learner_seat: PlayerID,
) -> tuple[StepRecord, Any]:
    """Compute the learner's choice and return ``(StepRecord, typed_action)``.

    The encoder may decode to a :class:`DiscardSentinel`; the env's
    ``_resolve_action`` path is replayed here so the typed action returned
    matches what env.step will actually apply.
    """
    view = make_player_view(env.state, learner_seat)
    obs = learner.obs_encoder.encode(view)
    mask = env.action_mask()

    if not mask.any():
        # Edge case: no representable legal action. Fall through with a
        # zero distribution; the engine will apply the first legal typed
        # action returned by the env's fallback (legal[0]).
        legal = env.legal_actions()
        zero_dist = np.zeros_like(mask, dtype=np.float32)
        record = StepRecord(
            obs=obs,
            action=-1,
            mask=mask,
            action_dist=zero_dist,
            value=0.0,
            reward=0.0,
            agent=learner_seat,
        )
        return record, legal[0]

    step_out, action_dist = learner.act_with_dist(obs, mask, deterministic=False)
    decoded = learner.action_encoder.decode(step_out.action_idx, env.state)
    if isinstance(decoded, DiscardSentinel):
        typed_action = heuristic_discard(env.state, decoded.player_id)
    else:
        typed_action = decoded

    record = StepRecord(
        obs=obs,
        action=step_out.action_idx,
        mask=mask,
        action_dist=action_dist,
        value=step_out.value,
        reward=0.0,
        agent=learner_seat,
    )
    return record, typed_action


def _with_reward(record: StepRecord, reward: float) -> StepRecord:
    """Frozen dataclasses can't be mutated; return a copy with reward set."""
    return StepRecord(
        obs=record.obs,
        action=record.action,
        mask=record.mask,
        action_dist=record.action_dist,
        value=record.value,
        reward=reward,
        agent=record.agent,
    )
