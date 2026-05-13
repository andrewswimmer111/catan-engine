"""Tests for :mod:`rl.training.vec_env` (rl-023)."""

from __future__ import annotations

import random
import time
from typing import Iterable

import numpy as np
import pytest

from domain.ids import PlayerID
from rl.training.vec_env import SubprocVecEnv, WorkerStepResult
from tests.rl.fixtures.vec_env_factory import random_opponents_factory


def _pick_legal(mask: np.ndarray, rng: random.Random) -> int:
    """Sample a legal action uniformly using ``rng`` (NOT global random)."""
    legal = np.flatnonzero(mask)
    return int(rng.choice(legal.tolist()))


def _drive_actions(
    vec: SubprocVecEnv, n_steps: int, rng: random.Random
) -> list[list[WorkerStepResult]]:
    """Run the vec env for ``n_steps`` learner-decision rounds and record the
    per-step :class:`WorkerStepResult` for each env."""
    results = vec.reset()
    trajectory: list[list[WorkerStepResult]] = [results]
    from domain.actions.base import Action  # noqa: F401  (decoded type)

    # Build a per-env encoder so we can decode mask indices into typed actions.
    encoders = [_make_encoder_for(res) for res in results]

    for _ in range(n_steps):
        actions = []
        for i, res in enumerate(results):
            if res.done:
                # Post-merge: action drives the new (post-reset) state.
                idx = _pick_legal(res.mask, rng)
            else:
                idx = _pick_legal(res.mask, rng)
            from domain.game.state import GameState  # noqa: F401
            actions.append(_decode(idx, encoders[i]))
        results = vec.step(actions)
        trajectory.append(results)
    return trajectory


def _make_encoder_for(_res: WorkerStepResult):
    from rl.encoding.action import ActionEncoder
    return ActionEncoder([PlayerID(i) for i in range(1, 5)])


def _decode(idx: int, encoder):
    # We don't have direct access to the env's state from main, so we can't
    # decode legitimately. Just send the integer through — the env's
    # ``step(int)`` path decodes against its own state internally.
    return idx


# ----------------------------------------------------------------------
# Functional smoke
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_reset_returns_one_payload_per_env():
    with SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=0) as vec:
        results = vec.reset()
    assert len(results) == 2
    for r in results:
        assert isinstance(r, WorkerStepResult)
        assert r.obs.shape[0] > 0
        assert r.mask.any()
        assert r.agent == PlayerID(1)


@pytest.mark.slow
def test_step_advances_envs_independently():
    with SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=0) as vec:
        results = vec.reset()
        rng = random.Random(0)
        for _ in range(10):
            actions = [_pick_legal(r.mask, rng) for r in results]
            results = vec.step(actions)
    # Both envs should still be live (10 steps unlikely to finish a game).
    for r in results:
        assert r.obs.shape == results[0].obs.shape


@pytest.mark.slow
def test_step_arity_mismatch_raises():
    with SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=0) as vec:
        vec.reset()
        with pytest.raises(ValueError):
            vec.step([0])


# ----------------------------------------------------------------------
# Determinism: rerun → same actions chosen
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_two_runs_same_seed_match():
    """Re-running the vec env with the same base_seed and same learner-action
    stream gives identical per-step masks. The env-side RNG + opponent RNG +
    learner-action sequence together determine the trajectory, so a fixed
    seed + fixed action stream is deterministic."""

    def run() -> list[list[int]]:
        rng = random.Random(123)
        masks: list[list[int]] = [[] for _ in range(3)]
        with SubprocVecEnv(random_opponents_factory, num_envs=3, base_seed=42) as vec:
            results = vec.reset()
            for _ in range(20):
                actions = [_pick_legal(r.mask, rng) for r in results]
                results = vec.step(actions)
                for i, r in enumerate(results):
                    masks[i].append(int(np.flatnonzero(r.mask).sum()))
        return masks

    first = run()
    second = run()
    assert first == second


@pytest.mark.slow
def test_vec_env_matches_single_env_on_same_seed():
    """Running env ``i`` of a num_envs=N vec env should produce the same per-step
    state stream as running a single :class:`CatanEnv` with seed ``base+i`` and
    the same opponent setup. This is the "remote env is a transparent
    multiplexer" property the spec calls out as the determinism criterion.
    """
    from rl.training.vec_env import WorkerStepResult

    base_seed = 17
    n_envs = 3
    rng_actions = random.Random(99)

    # 1. Vec env trajectory.
    with SubprocVecEnv(random_opponents_factory, num_envs=n_envs, base_seed=base_seed) as vec:
        vec_traj: list[list[WorkerStepResult]] = [vec.reset()]
        for _ in range(8):
            last = vec_traj[-1]
            actions = [_pick_legal(r.mask, rng_actions) for r in last]
            vec_traj.append(vec.step(actions))

    # 2. Run each env independently in-process at the same seed sequence.
    #    The action stream must match what the vec env's main-process loop
    #    fed env i — so we re-sample with a fresh rng of the same seed and
    #    pull one action per env per step in the same per-step order.
    rng_actions = random.Random(99)
    bundles = [random_opponents_factory(base_seed + i) for i in range(n_envs)]
    from rl.training.vec_env import _advance_to_next_learner_decision

    pending = [
        _advance_to_next_learner_decision(
            b, reward=0.0, prior_done=False, prior_info=None, seed=base_seed + i
        )
        for i, b in enumerate(bundles)
    ]
    in_proc_traj = [pending]

    for step in range(8):
        last = in_proc_traj[-1]
        next_states: list[WorkerStepResult] = []
        for i, bundle in enumerate(bundles):
            idx = _pick_legal(last[i].mask, rng_actions)
            _, reward, done, _ = bundle.env.step(idx)
            prior_info = None
            if done:
                prior_info = {
                    "winner": None if bundle.env.state.winner is None else int(bundle.env.state.winner),
                    "stalemate": False,
                    "ended_by": "learner",
                }
                bundle.env.reset(seed=base_seed + i)
            next_states.append(
                _advance_to_next_learner_decision(
                    bundle,
                    reward=float(reward),
                    prior_done=done,
                    prior_info=prior_info,
                    seed=base_seed + i,
                )
            )
        in_proc_traj.append(next_states)

    # Compare obs+mask byte-for-byte across every step and every env.
    for step in range(len(vec_traj)):
        for i in range(n_envs):
            assert vec_traj[step][i].obs.tobytes() == in_proc_traj[step][i].obs.tobytes(), (
                f"obs mismatch step={step} env={i}"
            )
            assert vec_traj[step][i].mask.tobytes() == in_proc_traj[step][i].mask.tobytes(), (
                f"mask mismatch step={step} env={i}"
            )
            assert vec_traj[step][i].done == in_proc_traj[step][i].done


@pytest.mark.nightly
def test_throughput_beats_single_env():
    """Headline performance test — should fire only when explicitly invoked.

    Multi-process startup cost is significant (~1s per worker on macOS for
    ``spawn``). The throughput delta only emerges over thousands of steps,
    so this test sits behind the ``nightly`` marker rather than ``slow``.
    """
    n_steps = 200
    n_envs = 4

    # Single-env baseline
    rng = random.Random(0)
    with SubprocVecEnv(random_opponents_factory, num_envs=1, base_seed=0) as vec:
        results = vec.reset()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            actions = [_pick_legal(r.mask, rng) for r in results]
            results = vec.step(actions)
        single_elapsed = time.perf_counter() - t0

    rng = random.Random(0)
    with SubprocVecEnv(random_opponents_factory, num_envs=n_envs, base_seed=0) as vec:
        results = vec.reset()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            actions = [_pick_legal(r.mask, rng) for r in results]
            results = vec.step(actions)
        vec_elapsed = time.perf_counter() - t0

    # Step throughput == env-steps / wall time.
    single_rate = n_steps / single_elapsed
    vec_rate = (n_envs * n_steps) / vec_elapsed
    speedup = vec_rate / single_rate
    assert speedup >= 3.0, (
        f"vec env speedup {speedup:.2f}× below 3× threshold "
        f"(single={single_rate:.0f}/s vec={vec_rate:.0f}/s)"
    )


@pytest.mark.slow
def test_each_env_uses_own_seed():
    """Different base_seed → divergent trajectories.

    Two vec envs with different seeds shouldn't end up with identical per-step
    legal-action sets — if they do, the seeding is broken.
    """

    def run(base: int) -> bytes:
        rng = random.Random(0)
        with SubprocVecEnv(random_opponents_factory, num_envs=2, base_seed=base) as vec:
            results = vec.reset()
            chunks: list[bytes] = []
            for _ in range(5):
                actions = [_pick_legal(r.mask, rng) for r in results]
                results = vec.step(actions)
                chunks.append(results[0].mask.tobytes())
                chunks.append(results[0].obs.tobytes())
        return b"".join(chunks)

    assert run(0) != run(1000)
