"""Subprocess-based vectorized env for parallel rollout collection.

The biggest single training-speedup lever is parallelising env stepping: a
GPU-bound learner sits idle while ``CatanEnv`` runs its Python-heavy
domain logic for ~1ms per step. :class:`SubprocVecEnv` shards ``num_envs``
envs across worker subprocesses and runs them concurrently while the main
process batches policy inference across all of them with a single forward
pass per "decision round".

Protocol
--------

Each subprocess owns a :class:`CatanEnv` plus three opponent
:class:`~controller.agents.Agent` instances. The subprocess runs all
opponent moves locally — only the learner's decisions require a round-trip
to the main process. The protocol is therefore:

1. ``reset()`` resets every env and advances each subprocess until the
   learner's turn (or termination). Each subprocess sends back an
   ``(obs, mask, done, info)`` payload.
2. ``step(actions)`` sends each subprocess the learner's chosen action and
   waits for the same payload back, after the subprocess has applied the
   action and then run opponents until the next learner turn.
3. ``close()`` joins all subprocesses cleanly.

Determinism
-----------

Each subprocess seeds its own env and opponent RNGs from ``base_seed +
env_index`` so two runs with the same ``base_seed`` produce identical
per-env trajectories. The learner's stochastic sampling uses per-env
:class:`torch.Generator`s (see :meth:`PolicyAgent.act_batch`) so a vec env
with ``num_envs=N`` produces the same per-env trajectory as ``N`` separate
single-env runs would — order of multiplexing doesn't leak into the result.

What we don't do
----------------

This implementation deliberately keeps the env *and* the opponents in the
subprocess. The alternative — running opponents in the main process too —
needs another round-trip per opponent move (3× more IPC per env step) and
only pays off if opponents are GPU-heavy. With CPU random/heuristic
opponents and small policy opponents on CPU, the current split wins.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Callable

import numpy as np

from controller.agents import Agent
from controller.session import GameSnapshot
from domain.actions.base import Action
from domain.engine.player_view import make_player_view
from domain.ids import PlayerID
from rl.env.catan_env import CatanEnv

__all__ = ["SubprocVecEnv", "VecEnvFactory", "WorkerBundle", "WorkerStepResult"]


@dataclass(frozen=True)
class WorkerStepResult:
    """One env's per-decision payload returned from a subprocess.

    ``obs`` / ``mask`` describe the state the learner is about to act on.
    When ``done`` is True the env has terminated; ``obs`` / ``mask`` then
    reflect the *new* (post-reset) state so the next ``step`` can use them
    directly without another round-trip.
    """

    obs: np.ndarray
    mask: np.ndarray
    reward: float
    done: bool
    agent: PlayerID
    info: dict


# A "vec env factory" hands back a self-contained per-env bundle: an env, a
# learner seat assignment, and the three opponent agents the worker will
# dispatch the non-learner turns to.
@dataclass
class WorkerBundle:
    env: CatanEnv
    learner_seat: PlayerID
    opponents: dict[PlayerID, Agent]


VecEnvFactory = Callable[[int], WorkerBundle]


class SubprocVecEnv:
    """Run ``num_envs`` :class:`CatanEnv` instances in parallel subprocesses.

    Construct with a ``factory`` callable that, given an integer seed,
    returns a fresh :class:`WorkerBundle`. The factory must be picklable
    (importable from a module path) — closures over local variables won't
    survive the ``spawn`` startup method.
    """

    def __init__(
        self,
        factory: VecEnvFactory,
        num_envs: int,
        base_seed: int = 0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")

        self._num_envs = num_envs
        self._closed = False
        ctx = mp.get_context("spawn")
        self._parent_conns: list[Connection] = []
        self._procs: list[mp.process.BaseProcess] = []
        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, factory, base_seed + i),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def reset(self) -> list[WorkerStepResult]:
        """Reset every env and wait for each to surface its first learner turn."""
        for conn in self._parent_conns:
            conn.send(("reset", None))
        return [_recv(conn) for conn in self._parent_conns]

    def step(self, actions: list[Action]) -> list[WorkerStepResult]:
        """Apply ``actions[i]`` to env ``i`` and collect the next learner turn.

        ``len(actions)`` must equal ``num_envs``. The subprocess applies the
        learner's typed action, then advances through opponent turns
        locally until the next learner decision (or termination), and
        responds with the resulting state.
        """
        if len(actions) != self._num_envs:
            raise ValueError(
                f"step() requires {self._num_envs} actions, got {len(actions)}"
            )
        for conn, action in zip(self._parent_conns, actions):
            conn.send(("step", action))
        return [_recv(conn) for conn in self._parent_conns]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
            conn.close()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()

    def __enter__(self) -> "SubprocVecEnv":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _recv(conn: Connection) -> WorkerStepResult:
    payload = conn.recv()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"vec env worker raised: {payload['error']}")
    return payload


# ----------------------------------------------------------------------
# Worker process
# ----------------------------------------------------------------------


def _worker_loop(conn: Connection, factory: VecEnvFactory, seed: int) -> None:
    """Subprocess entry point.

    Builds its bundle once, then loops on commands from the main process:

    * ``reset`` → reset env, advance through opponent turns until learner's
      decision, send :class:`WorkerStepResult` back.
    * ``step action`` → apply ``action`` (the learner's typed move),
      advance through opponent turns, send :class:`WorkerStepResult`.
    * ``close`` → break.

    Every response carries an obs/mask describing a state the learner is
    about to act on — terminations are handled internally (reset + advance)
    so the main process never receives an empty-mask payload. The ``done``
    flag tells the caller "an episode ended between this and the previous
    response" — load the final reward, then act on the new obs.

    Any exception is captured and sent back as an error payload so the
    main process can re-raise instead of hanging on a closed pipe.
    """
    try:
        bundle = factory(seed)
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                bundle.env.reset(seed=seed)
                result = _advance_to_next_learner_decision(
                    bundle, reward=0.0, prior_done=False, prior_info=None, seed=seed
                )
                conn.send(result)
            elif cmd == "step":
                _, reward, done, _ = bundle.env.step(data)
                prior_info: dict | None = None
                if done:
                    prior_info = _terminal_info(bundle, ended_by="learner")
                    bundle.env.reset(seed=seed)
                result = _advance_to_next_learner_decision(
                    bundle,
                    reward=float(reward),
                    prior_done=done,
                    prior_info=prior_info,
                    seed=seed,
                )
                conn.send(result)
            else:
                raise RuntimeError(f"unknown vec env cmd: {cmd}")
    except Exception as exc:  # pragma: no cover - subprocess error path
        try:
            conn.send({"error": repr(exc)})
        finally:
            conn.close()


def _advance_to_next_learner_decision(
    bundle: WorkerBundle,
    *,
    reward: float,
    prior_done: bool,
    prior_info: dict | None,
    seed: int,
) -> WorkerStepResult:
    """Run opponent moves until the learner has to act in a *live* env.

    If an opponent's move terminates the game, the env is reset internally
    and the loop continues — the ``done`` flag is OR'd across all
    terminations so the caller knows an episode boundary was crossed.
    """
    env = bundle.env
    done = prior_done
    info = prior_info
    while env.current_agent != bundle.learner_seat:
        snap = GameSnapshot(
            state=env.state, step_index=0, last_action=None, last_events=()
        )
        legal = env.legal_actions()
        action = bundle.opponents[env.current_agent].choose(snap, legal)
        if action is None:
            # No legal action for the opponent — count as stalemate, reset.
            info = _terminal_info(bundle, ended_by="opponent", stalemate=True)
            done = True
            env.reset(seed=seed)
            continue
        _, _, opp_done, _ = env.step(action)
        if opp_done:
            info = _terminal_info(bundle, ended_by="opponent")
            done = True
            env.reset(seed=seed)

    view = make_player_view(env.state, bundle.learner_seat)
    obs = env.obs_encoder.encode(view)
    return WorkerStepResult(
        obs=obs,
        mask=env.action_mask(),
        reward=reward,
        done=done,
        agent=bundle.learner_seat,
        info=info or {"phase": env.state.phase.name},
    )


def _terminal_info(
    bundle: WorkerBundle, *, ended_by: str, stalemate: bool = False
) -> dict:
    state = bundle.env.state
    return {
        "winner": None if state.winner is None else int(state.winner),
        "stalemate": stalemate,
        "ended_by": ended_by,
    }
