"""Discrete action-space and flat observation encoding for the RL agent."""

from rl.encoding._action_layout import ACTION_SPACE_SIZE
from rl.encoding.action import ActionEncoder, DecodedAction, DiscardSentinel
from rl.encoding.observation import (
    FlatObservationEncoder,
    OBS_LAYOUT_VERSION,
    OBS_SHAPE,
)

__all__ = [
    "ACTION_SPACE_SIZE",
    "ActionEncoder",
    "DecodedAction",
    "DiscardSentinel",
    "FlatObservationEncoder",
    "OBS_LAYOUT_VERSION",
    "OBS_SHAPE",
]
