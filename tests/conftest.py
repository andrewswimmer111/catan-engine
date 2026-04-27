"""Shared pytest configuration: fixtures are defined under ``tests.fixtures``."""

from __future__ import annotations

pytest_plugins = [
    "tests.fixtures.boards",
    "tests.fixtures.states",
]
