"""
Offscreen rendering tests: render different scene types and verify output.
"""
import pytest
import numpy as np
from pathlib import Path

from slangpy_renderer import ColoredMesh, Mesh, OffscreenContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cube_obj_path() -> str:
    """Return absolute path to the cube.obj asset."""
    return str(
        Path(__file__).resolve().parent.parent
        / "slangpy_renderer"
        / "assets"
        / "models"
        / "cube.obj"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["cube_mesh", "coord_axes", "colored_triangle"])
def scene(request, offscreen_ctx):
    """
    Parametrised fixture: adds one renderable, yields its name, cleans up.
    """
    name = request.param

    if name == "cube_mesh":
        mesh = Mesh.from_obj(offscreen_ctx.device, _cube_obj_path())
        offscreen_ctx.add_renderable("test", mesh)

    elif name == "coord_axes":
        axes = ColoredMesh.create_axis3d(offscreen_ctx.device, scale=2.0)
        offscreen_ctx.add_renderable("test", axes)

    elif name == "colored_triangle":
        positions = np.array(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32
        )
        colors = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        indices = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint16)
        cm = ColoredMesh(
            device=offscreen_ctx.device,
            positions=positions,
            colors=colors,
            indices=indices,
            sync_gpu=True,
        )
        offscreen_ctx.add_renderable("test", cm)

    yield name
    offscreen_ctx.remove_renderable("test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_render_produces_output(scene, offscreen_ctx, view_matrix, proj_matrix):
    """Rendering each scene type must produce a non-black image."""
    image = offscreen_ctx.render_frame(view_matrix, proj_matrix)
    assert image.shape == (256, 256, 4), f"Unexpected shape: {image.shape}"
    assert image.max() > 0, f"Image is all black for scene '{scene}'"


def test_render_deterministic(scene, offscreen_ctx, view_matrix, proj_matrix):
    """Consecutive renders with the same inputs must produce identical output."""
    img1 = offscreen_ctx.render_frame(view_matrix, proj_matrix)
    img2 = offscreen_ctx.render_frame(view_matrix, proj_matrix)
    np.testing.assert_array_equal(img1, img2)


def test_clear_color(offscreen_ctx, view_matrix, proj_matrix):
    """Rendering an empty scene with a non-black clear color must fill the buffer."""
    offscreen_ctx.clear()
    image = offscreen_ctx.render_frame(
        view_matrix, proj_matrix, clear_color=(0.0, 0.0, 1.0, 1.0)
    )
    # Blue channel should be dominant
    assert image[:, :, 2].mean() > 200


def test_render_frame_shape(offscreen_ctx, view_matrix, proj_matrix):
    """render_frame must return (H, W, 4) uint8 array."""
    offscreen_ctx.clear()
    image = offscreen_ctx.render_frame(view_matrix, proj_matrix)
    assert image.dtype == np.uint8
    assert image.shape == (256, 256, 4)
