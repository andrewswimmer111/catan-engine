"""Tests for the rl-007 encoded env API: tensor obs + action mask + int|Action step."""

from __future__ import annotations

import random

import numpy as np
import pytest

from domain.actions.all_actions import EndTurnAction, RollDiceAction
from domain.engine.game_engine import IllegalActionError
from domain.enums import TurnPhase
from domain.ids import PlayerID
from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv


# ---------------------------------------------------------------------------
# Reset shape + mask cardinality
# ---------------------------------------------------------------------------


def _encoded_indices(env: CatanEnv, legal) -> set[int]:
    """Encode every action in ``legal`` that the encoder represents.

    Domestic-trade actions raise ``ValueError`` (not in the layout per rl-005);
    the mask silently skips them and so do we.
    """
    out: set[int] = set()
    for a in legal:
        try:
            out.add(env.action_encoder.encode(a))
        except ValueError:
            pass
    return out


def test_reset_returns_encoded_obs_and_mask() -> None:
    env = CatanEnv(seed=0)
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == OBS_SHAPE
    assert obs.dtype == np.float32

    mask = info["action_mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == np.bool_

    # Mask True count equals the number of distinct encoded indices over the
    # representable legal actions. In INITIAL_SETTLEMENT every legal action is
    # representable (no domestic trade), and each maps to a distinct index.
    assert mask.sum() == len(_encoded_indices(env, info["legal_actions"]))


def test_info_has_all_required_keys() -> None:
    env = CatanEnv(seed=1)
    _, info = env.reset()
    for key in ("current_agent", "action_mask", "legal_actions", "last_events", "phase"):
        assert key in info, f"missing info key: {key!r}"


# ---------------------------------------------------------------------------
# step accepts both int and Action
# ---------------------------------------------------------------------------


def test_step_accepts_int_index_when_mask_true() -> None:
    env = CatanEnv(seed=2)
    _, info = env.reset()
    mask = info["action_mask"]
    # Pick the first masked-True index.
    [idx] = np.argwhere(mask)[:1]
    obs, reward, done, info2 = env.step(int(idx[0]))
    assert isinstance(obs, np.ndarray)
    assert obs.shape == OBS_SHAPE
    assert reward == 0.0
    assert isinstance(info2["action_mask"], np.ndarray)


def test_step_accepts_typed_action_for_backwards_compat() -> None:
    env = CatanEnv(seed=2)
    env.reset()
    legal = env.legal_actions()
    obs, _, _, _ = env.step(legal[0])  # typed Action passes through
    assert obs.shape == OBS_SHAPE


def test_step_accepts_numpy_integer_index() -> None:
    """np.argmax returns numpy.int64 — make sure the env accepts it directly."""
    env = CatanEnv(seed=2)
    _, info = env.reset()
    idx = np.argmax(info["action_mask"])  # numpy scalar
    env.step(idx)


# ---------------------------------------------------------------------------
# Masked-False indices raise
# ---------------------------------------------------------------------------


def test_step_with_masked_false_index_raises() -> None:
    env = CatanEnv(seed=0)
    _, info = env.reset()
    mask = info["action_mask"]
    false_indices = np.argwhere(~mask).flatten()
    assert len(false_indices) > 0
    # Use any masked-False index that decodes to a typed action (skip the
    # discard sentinel — out-of-DISCARD it raises IllegalActionError too via
    # the env's sentinel-resolution path).
    idx = int(false_indices[0])
    with pytest.raises(IllegalActionError):
        env.step(idx)


# ---------------------------------------------------------------------------
# Tournament smoke with the encoded API
# ---------------------------------------------------------------------------


def _play_one_encoded(seed: int, max_steps: int = 2000) -> tuple[bool, int]:
    """Drive a single game to completion using ``env.step(int_index)`` only.

    Returns ``(done, n_steps)``.
    """
    env = CatanEnv(seed=seed)
    _, info = env.reset(seed=seed)
    rng = random.Random(seed)
    done = False
    steps = 0
    while not done and steps < max_steps:
        mask = info["action_mask"]
        idxs = np.argwhere(mask).flatten().tolist()
        if not idxs:
            break
        idx = rng.choice(idxs)
        _, _, done, info = env.step(int(idx))
        steps += 1
    return done, steps


def test_encoded_tournament_smoke_50_games() -> None:
    """50 random seeds, masked-int actions only — no exceptions, all advance."""
    for seed in range(50):
        done, steps = _play_one_encoded(seed=seed, max_steps=300)
        assert steps > 0


# ---------------------------------------------------------------------------
# Encoder injection
# ---------------------------------------------------------------------------


def test_user_supplied_action_encoder_is_preserved_across_resets() -> None:
    """A user-provided ActionEncoder must not be replaced on reset."""
    from rl.encoding.action import ActionEncoder

    pids = [PlayerID(i) for i in range(1, 5)]
    custom = ActionEncoder(pids)
    env = CatanEnv(seed=0, action_encoder=custom)
    env.reset(seed=0)
    assert env.action_encoder is custom
    env.reset(seed=1)
    assert env.action_encoder is custom


def test_default_action_encoder_tracks_config_player_ids() -> None:
    """Without a user-supplied encoder, the env builds one matching the config."""
    env = CatanEnv(seed=0)
    env.reset()
    legal = env.legal_actions()
    # Round-trip every representable legal action via the default encoder.
    enc = env.action_encoder
    for a in legal:
        idx = enc.encode(a)
        assert 0 <= idx < ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# Mask matches legal_actions identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3, 7, 11])
def test_mask_is_consistent_with_legal_actions(seed: int) -> None:
    env = CatanEnv(seed=seed)
    _, info = env.reset(seed=seed)
    rng = random.Random(seed)
    for _ in range(40):
        mask = info["action_mask"]
        legal = info["legal_actions"]
        encoded = _encoded_indices(env, legal)
        assert int(mask.sum()) == len(encoded)
        for idx in encoded:
            assert mask[idx]
        if not legal:
            break
        _, _, done, info = env.step(rng.choice(legal))
        if done:
            break


# ---------------------------------------------------------------------------
# Reaching MAIN via roll then end-turn through the encoded API
# ---------------------------------------------------------------------------


def test_can_drive_setup_and_main_via_int_indices() -> None:
    env = CatanEnv(seed=4)
    _, info = env.reset(seed=4)
    rng = random.Random(4)

    # Walk through setup with the int API.
    while env.state.phase in (TurnPhase.INITIAL_SETTLEMENT, TurnPhase.INITIAL_ROAD):
        idxs = np.argwhere(info["action_mask"]).flatten().tolist()
        idx = rng.choice(idxs)
        _, _, done, info = env.step(int(idx))
        if done:
            break

    assert env.state.phase is TurnPhase.ROLL

    # Roll then end turn.
    roll_idx = env.action_encoder.encode(RollDiceAction(player_id=env.state.current_player))
    _, _, _, info = env.step(roll_idx)

    if env.state.phase is TurnPhase.MAIN:
        end_idx = env.action_encoder.encode(EndTurnAction(player_id=env.state.current_player))
        assert info["action_mask"][end_idx]
        env.step(end_idx)
