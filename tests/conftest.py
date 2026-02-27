"""
Shared pytest fixtures for slangpy_renderer tests.
"""
import pytest
import numpy as np
from pathlib import Path

from slangpy_renderer import OffscreenContext


@pytest.fixture(scope="session")
def offscreen_ctx():
    """Session-scoped offscreen rendering context (256x256)."""
    ctx = OffscreenContext(width=256, height=256)
    yield ctx


@pytest.fixture(scope="session")
def assets_path():
    """Absolute path to the installed package's assets directory."""
    return Path(__file__).resolve().parent.parent / "slangpy_renderer" / "assets"


@pytest.fixture
def view_matrix(offscreen_ctx):
    """Default view matrix: camera at (3,3,3) looking at origin."""
    return offscreen_ctx.default_view_matrix(
        eye=(3.0, 3.0, 3.0), center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)
    )


@pytest.fixture
def proj_matrix(offscreen_ctx):
    """Default perspective projection (60 deg fov, aspect 1.0, near 0.1, far 100)."""
    return offscreen_ctx.default_proj_matrix(fov=60.0, near=0.1, far=100.0)
