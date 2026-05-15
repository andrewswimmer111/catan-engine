"""Regenerate the graph observation snapshot fixture.

Run when ``GRAPH_OBS_LAYOUT_VERSION`` is bumped:

    PYTHONPATH=src venv/bin/python tests/rl/fixtures/build_graph_obs_snapshot.py

The snapshot is built from the same deterministic seeded rollout the flat
encoder uses (``build_obs_snapshot.py``), so the two fixtures pin
*identical* game state — they only differ in encoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl.encoding.graph_observation import (
    GRAPH_OBS_LAYOUT_VERSION,
    GraphObservationEncoder,
)
from tests.rl.fixtures.build_obs_snapshot import _build_snapshot_view

SNAPSHOT_PATH = (
    Path(__file__).parent / f"graph_obs_snapshot_v{GRAPH_OBS_LAYOUT_VERSION}.npy"
)


def main() -> None:
    view = _build_snapshot_view()
    obs = GraphObservationEncoder().encode(view)
    np.save(SNAPSHOT_PATH, obs, allow_pickle=False)
    print(f"wrote {SNAPSHOT_PATH} shape={obs.shape} sum={float(obs.sum()):.4f}")


if __name__ == "__main__":
    main()
