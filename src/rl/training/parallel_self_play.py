"""Subprocess-parallel self-play for AlphaZero.

:func:`generate_games_parallel` distributes ``n_games`` of self-play
across ``n_workers`` subprocesses. Each subprocess rebuilds the
:class:`rl.agents.policy_agent.PolicyAgent` from a serialised
``(ModelArch, state_dict)`` pair (no checkpoint file written), runs its
share of games independently with its own MCTS tree, and streams
:class:`rl.training.self_play.SelfPlayGame` instances back over a
shared :class:`multiprocessing.Queue`.

This is the first parallel variant from az-009 — independent workers,
no shared forward pass. Each subprocess runs its own (CPU) MCTS;
inference happens N times in parallel rather than in one batched
forward. The trade-off is documented in the Phase 3 design memo:

* **Pros**: trivial to reason about, no inter-worker synchronisation,
  no model-weight sharing complexity, no IPC during MCTS expansion.
* **Cons**: no batched-forward GPU/MPS utilisation; each subprocess has
  a full model copy in memory; weights are re-serialised to every
  worker every iteration.

Switch to a shared-forward design (the alternative variant from the
design memo) only if measurements show forward-pass latency dominates
per-rollout cost on the accelerator.

Concurrency safety / determinism notes
--------------------------------------

* Subprocesses run on ``device="cpu"`` regardless of the parent's
  device — MPS / CUDA contexts don't survive a ``fork`` and the
  ``spawn`` start method (used here for macOS safety) ships a CPU
  state-dict to children. The parent's MPS / CUDA model is untouched.
* Each game's action-sampling RNG is seeded from its ``game_seed``
  inside the worker, so a fixed set of seeds produces a deterministic
  set of games regardless of worker scheduling order.
* The order of returned :class:`SelfPlayGame` instances is **not**
  guaranteed; callers that need a deterministic order should sort by
  ``game_seed`` or carry it in the game record (we don't, yet).
"""

from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass
from typing import Iterable

from rl.agents.policy_agent import PolicyAgent
from rl.training.checkpoint import (
    ModelArch,
    build_agent_for_arch,
    model_arch_from,
)
from rl.training.self_play import (
    SelfPlayConfig,
    SelfPlayGame,
    play_self_play_game,
)

__all__ = ["generate_games_parallel"]


# ----------------------------------------------------------------------
# Worker entry (module-level so ``spawn`` start method can import it)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _WorkerError:
    """Surfaced from a worker when ``play_self_play_game`` raises.

    Carries the traceback string so the parent can re-raise with the
    full subprocess stack for debugging (subprocess exceptions otherwise
    vanish silently into the multiprocessing layer).
    """

    worker_id: int
    game_seed: int
    traceback_str: str


def _worker_main(
    *,
    worker_id: int,
    arch: ModelArch,
    state_dict: dict,
    config: SelfPlayConfig,
    game_seeds: list[int],
    out_queue: "mp.Queue",
) -> None:
    """Subprocess entry: rebuild the learner, run games, push results.

    Each game gets its own ``random.Random(game_seed)`` so the action-
    sampling stream is fully reproducible per game_seed regardless of
    worker assignment.
    """
    try:
        agent = build_agent_for_arch(arch, device="cpu")
        agent.load_state_dict(state_dict)
        agent.model.eval()
    except Exception:
        out_queue.put(_WorkerError(
            worker_id=worker_id,
            game_seed=-1,
            traceback_str=traceback.format_exc(),
        ))
        out_queue.put(_DONE)
        return

    for seed in game_seeds:
        try:
            game = play_self_play_game(
                agent,
                config,
                random.Random(seed),
                game_seed=seed,
            )
            out_queue.put(game)
        except Exception:
            out_queue.put(_WorkerError(
                worker_id=worker_id,
                game_seed=seed,
                traceback_str=traceback.format_exc(),
            ))
            break

    out_queue.put(_DONE)


# Sentinel sent by each worker when it finishes its assigned seed list.
# Using a distinct singleton so callers can disambiguate from None.
_DONE = "__SELF_PLAY_DONE__"


# ----------------------------------------------------------------------
# Public entry
# ----------------------------------------------------------------------


def generate_games_parallel(
    learner: PolicyAgent,
    config: SelfPlayConfig,
    game_seeds: Iterable[int],
    *,
    n_workers: int,
) -> list[SelfPlayGame]:
    """Run ``len(game_seeds)`` self-play games across ``n_workers``.

    Each worker reconstructs the learner from a serialised arch +
    state-dict snapshot on CPU, plays its share of seeds, and streams
    completed :class:`SelfPlayGame` instances back. Returns the
    aggregated list (order not guaranteed).

    Raises ``RuntimeError`` if any worker errors during play —
    re-raises with the subprocess traceback in the message.
    """
    seeds = list(game_seeds)
    if n_workers <= 0:
        raise ValueError(f"n_workers must be positive (got {n_workers})")
    if not seeds:
        return []

    arch = model_arch_from(learner.model)
    state_dict = {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()}

    # ``spawn`` is the macOS / Python 3.13 default and avoids the
    # fork-after-torch-init pitfalls (CUDA / MPS state, file handles).
    ctx = mp.get_context("spawn")
    out_queue: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []

    for worker_id, worker_seeds in enumerate(_split_seeds(seeds, n_workers)):
        if not worker_seeds:
            continue
        p = ctx.Process(
            target=_worker_main,
            kwargs=dict(
                worker_id=worker_id,
                arch=arch,
                state_dict=state_dict,
                config=config,
                game_seeds=worker_seeds,
                out_queue=out_queue,
            ),
        )
        p.start()
        procs.append(p)

    games: list[SelfPlayGame] = []
    workers_remaining = len(procs)
    first_error: _WorkerError | None = None
    while workers_remaining > 0:
        item = out_queue.get()
        if item == _DONE:
            workers_remaining -= 1
        elif isinstance(item, _WorkerError):
            if first_error is None:
                first_error = item
        else:
            games.append(item)

    for p in procs:
        p.join(timeout=30.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)

    if first_error is not None:
        raise RuntimeError(
            f"self-play worker {first_error.worker_id} failed on "
            f"game_seed={first_error.game_seed}:\n{first_error.traceback_str}"
        )

    return games


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _split_seeds(seeds: list[int], n_workers: int) -> list[list[int]]:
    """Round-robin split so workload imbalance is bounded by 1 game."""
    buckets: list[list[int]] = [[] for _ in range(n_workers)]
    for i, seed in enumerate(seeds):
        buckets[i % n_workers].append(seed)
    return buckets
