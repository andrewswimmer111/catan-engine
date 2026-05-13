"""Tests for :class:`rl.evaluation.scheduler.EvalScheduler` (rl-019)."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pytest
import torch

from domain.enums import EndReason
from domain.ids import PlayerID
from rl.agents.policy_agent import PolicyAgent
from rl.encoding.action import ACTION_SPACE_SIZE, ActionEncoder
from rl.encoding.observation import OBS_SHAPE
from rl.env.catan_env import CatanEnv
from rl.evaluation.elo import EloTracker
from rl.evaluation.metrics import GameStats, TournamentResult
from rl.evaluation.scheduler import (
    EvalScheduler,
    make_bench_vs_heuristic,
    make_bench_vs_pool,
    make_bench_vs_random,
)
from rl.models.mlp import MLPPolicyValue
from rl.training.opponent_pool import OpponentPool


PLAYER_IDS = [PlayerID(i) for i in range(1, 5)]


def _make_learner() -> PolicyAgent:
    torch.manual_seed(0)
    model = MLPPolicyValue(OBS_SHAPE[0], ACTION_SPACE_SIZE, hidden=(32, 32))
    return PolicyAgent(model, ActionEncoder(PLAYER_IDS))


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.calls.append((name, value, step))

    def log_scalars(self, prefix: str, values: dict, step: int) -> None:
        pass

    def log_histogram(self, name: str, values, step: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeTrainer:
    """Minimal Trainer stand-in — scheduler only needs ``.learner`` and ``.logger``."""

    def __init__(self, learner: PolicyAgent) -> None:
        self.learner = learner
        self.logger = _RecordingLogger()


def _fake_bench(name: str) -> Callable[[PolicyAgent], TournamentResult]:
    """Build a benchmark callable that returns a canned TournamentResult.

    Generates two games: learner (seat 1) wins one, loses the other. This
    keeps the test independent of actual game-playing performance.
    """

    def bench(_learner: PolicyAgent) -> TournamentResult:
        games = [
            GameStats(
                winner=PlayerID(1),
                final_vps={PlayerID(1): 10},
                turn_count=20,
                end_reason=EndReason.WINNER,
                action_histogram={},
                per_seat_action_histogram={},
            ),
            GameStats(
                winner=PlayerID(2),
                final_vps={PlayerID(2): 10},
                turn_count=22,
                end_reason=EndReason.WINNER,
                action_histogram={},
                per_seat_action_histogram={},
            ),
        ]
        win_counts: dict[PlayerID, int] = defaultdict(int)
        for g in games:
            if g.winner is not None:
                win_counts[g.winner] += 1
        return TournamentResult(
            games=games,
            win_rates={pid: win_counts[pid] / len(games) for pid in PLAYER_IDS},
            mean_vp={pid: 5.0 for pid in PLAYER_IDS},
            mean_turns=21.0,
        )

    return bench


# ----------------------------------------------------------------------
# Firing cadence
# ----------------------------------------------------------------------


def test_scheduler_does_not_fire_between_intervals() -> None:
    trainer = _FakeTrainer(_make_learner())
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=1000,
        benchmarks={"fake": _fake_bench("fake")},
    )
    # The first call at step=0 fires (last_eval_step initialized to -every_steps).
    assert sched.maybe_run(0) is not None
    # Subsequent calls below 1000 do nothing.
    assert sched.maybe_run(100) is None
    assert sched.maybe_run(500) is None
    assert sched.maybe_run(999) is None
    # At step 1000 it fires again.
    assert sched.maybe_run(1000) is not None


def test_scheduler_first_call_fires_at_step_zero() -> None:
    trainer = _FakeTrainer(_make_learner())
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=5000,
        benchmarks={"fake": _fake_bench("fake")},
    )
    metrics = sched.maybe_run(0)
    assert metrics is not None
    assert "fake/win_rate" in metrics
    assert "fake/elo" in metrics


def test_scheduler_invalid_every_steps_raises() -> None:
    trainer = _FakeTrainer(_make_learner())
    with pytest.raises(ValueError):
        EvalScheduler(trainer=trainer, every_steps=0, benchmarks={})  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Elo updates
# ----------------------------------------------------------------------


def test_scheduler_updates_elo_after_each_eval() -> None:
    trainer = _FakeTrainer(_make_learner())
    elo = EloTracker()
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=100,
        benchmarks={"fake": _fake_bench("fake")},
        elo=elo,
    )
    sched.maybe_run(0)
    # The learner ID for step 0 should now exist in the tracker.
    assert "learner@step_0" in elo.ratings
    assert "fake" in elo.ratings


def test_scheduler_logs_win_rate_and_elo_to_tb() -> None:
    trainer = _FakeTrainer(_make_learner())
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=100,
        benchmarks={"fake_a": _fake_bench("fake_a"), "fake_b": _fake_bench("fake_b")},
    )
    sched.maybe_run(500)

    names = {c[0] for c in trainer.logger.calls}
    assert "eval/fake_a/win_rate" in names
    assert "eval/fake_a/elo" in names
    assert "eval/fake_b/win_rate" in names
    assert "eval/fake_b/elo" in names


def test_ratings_persisted_to_disk(tmp_path: Path) -> None:
    trainer = _FakeTrainer(_make_learner())
    ratings_path = tmp_path / "ratings.json"
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=100,
        benchmarks={"fake": _fake_bench("fake")},
        ratings_path=ratings_path,
    )
    sched.maybe_run(0)
    assert ratings_path.exists()
    loaded = EloTracker.load(ratings_path)
    assert "learner@step_0" in loaded.ratings


# ----------------------------------------------------------------------
# Default benchmark factories — wiring smoke
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_scheduler_archives_interesting_games(tmp_path: Path) -> None:
    """Wiring smoke for rl-021 archive hook.

    The archive should accumulate a directory per game it decides was
    interesting. We accept a wide range (1..n_games inclusive) because
    interestingness is data-dependent and we only want to confirm the
    wiring actually fires and writes something.
    """
    trainer = _FakeTrainer(_make_learner())
    archive_root = tmp_path / "episodes"
    sched = EvalScheduler(
        trainer=trainer,  # type: ignore[arg-type]
        every_steps=100,
        benchmarks={"fake": _fake_bench("fake")},
        archive_root=archive_root,
        archive_n_games=3,
    )
    metrics = sched.maybe_run(0)
    assert metrics is not None
    assert "archive/written" in metrics
    from rl.replay.dataset import ReplayDataset
    ds = ReplayDataset(archive_root)
    n_written = len(ds.list_episodes())
    assert n_written == int(metrics["archive/written"])
    assert 0 <= n_written <= 3


def test_default_benchmark_factories_produce_callables(tmp_path: Path) -> None:
    """Sanity-check the default benchmark constructors return callables that
    actually play games and return a non-empty TournamentResult. The games
    are short to keep this fast."""
    env_factory: Callable[[int], CatanEnv] = lambda s: CatanEnv(seed=s)
    learner = _make_learner()
    pool = OpponentPool(rng=random.Random(0))

    bench_rand = make_bench_vs_random(env_factory, n_games=1, base_seed=10)
    bench_heur = make_bench_vs_heuristic(env_factory, n_games=1, base_seed=20)
    bench_pool = make_bench_vs_pool(env_factory, pool, n_games=1, base_seed=30)

    for bench in (bench_rand, bench_heur, bench_pool):
        result = bench(learner)
        assert len(result.games) == 1
        assert result.mean_turns >= 0
