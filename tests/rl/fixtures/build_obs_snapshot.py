"""Regenerate the observation snapshot fixture.

Run when ``OBS_LAYOUT_VERSION`` is bumped:

    PYTHONPATH=src .venv/bin/python tests/rl/fixtures/build_obs_snapshot.py

The snapshot is built from a deterministic seeded rollout — see
``_build_snapshot_view`` below for the recipe. Keep it small and stable so
diffs are reviewable.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from domain.engine.player_view import PlayerView
from rl.encoding.observation import FlatObservationEncoder, OBS_LAYOUT_VERSION
from rl.env.catan_env import CatanEnv

SNAPSHOT_SEED = 42
SNAPSHOT_STEPS = 20  # past initial setup; lands in MAIN/ROLL territory
SNAPSHOT_PATH = Path(__file__).parent / f"obs_snapshot_v{OBS_LAYOUT_VERSION}.npy"


def _build_snapshot_view() -> PlayerView:
    env = CatanEnv(seed=SNAPSHOT_SEED)
    env.reset(seed=SNAPSHOT_SEED)
    rng = random.Random(SNAPSHOT_SEED)
    for _ in range(SNAPSHOT_STEPS):
        legal = env.legal_actions()
        if not legal:
            break
        env.step(rng.choice(legal))
    # Always encode from seat 0's perspective for stability.
    viewer = env.state.config.player_ids[0]
    return env._engine.player_view(env.state, viewer)


def main() -> None:
    view = _build_snapshot_view()
    obs = FlatObservationEncoder().encode(view)
    np.save(SNAPSHOT_PATH, obs, allow_pickle=False)
    print(f"wrote {SNAPSHOT_PATH} shape={obs.shape} sum={float(obs.sum()):.4f}")


if __name__ == "__main__":
    main()
