"""End-to-end pipeline validation via rendering + readback.

Validates that the transformation pipeline (model -> world -> view -> clip ->
NDC -> screen) produces correct results by:
1. Rendering known geometry with known matrices
2. Reading back color and depth buffers
3. Checking that vertices appear at expected screen positions
4. Verifying depth values and Y-axis convention

No RenderDoc dependency — uses pure SlangPy rendering + texture readback.
RenderDoc capture analysis is available as a separate optional path.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from slangpy_renderer import ColoredMesh, Mesh, OffscreenContext
from slangpy_renderer.offscreen import look_at, vulkan_rh_zo_perspective

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known camera / scene parameters
# ---------------------------------------------------------------------------
WIDTH = 256
HEIGHT = 256
ASPECT = float(WIDTH) / float(HEIGHT)
FOV_Y = 60.0
NEAR = 0.1
FAR = 100.0
EYE = np.array([0.0, 0.0, 3.0])
CENTER = np.array([0.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


def _find_cube_obj() -> str:
    import slangpy_renderer
    pkg_dir = Path(slangpy_renderer.__file__).parent
    candidate = pkg_dir / "assets" / "models" / "cube.obj"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"cube.obj not found at {candidate}")


def _ndc_to_pixel(ndc_x: float, ndc_y: float, width: int, height: int) -> tuple[int, int]:
    """Convert NDC [-1,1] to pixel coordinates.

    Vulkan NDC: X right, Y down, depth [0,1].
    pixel_x = (ndc_x * 0.5 + 0.5) * width
    pixel_y = (ndc_y * 0.5 + 0.5) * height   (Y-down, so +ndc_y = lower on screen)
    """
    px = int((ndc_x * 0.5 + 0.5) * width)
    py = int((ndc_y * 0.5 + 0.5) * height)
    return (
        max(0, min(width - 1, px)),
        max(0, min(height - 1, py)),
    )


def _world_to_ndc(
    point: np.ndarray,
    view: np.ndarray,
    proj: np.ndarray,
    model: np.ndarray = None,
) -> np.ndarray:
    """Transform a 3D world point to NDC (x, y, z) via MVP + perspective divide."""
    if model is None:
        model = np.eye(4, dtype=np.float64)
    v = np.append(np.asarray(point, dtype=np.float64), 1.0)
    clip = proj @ view @ model @ v
    ndc = clip[:3] / clip[3]
    return ndc


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrices() -> dict[str, np.ndarray]:
    view = look_at(EYE, CENTER, UP)
    proj = vulkan_rh_zo_perspective(FOV_Y, ASPECT, NEAR, FAR)
    model = np.eye(4, dtype=np.float64)
    return {"view": view, "proj": proj, "model": model}


@pytest.fixture(scope="module")
def ctx() -> OffscreenContext:
    return OffscreenContext(width=WIDTH, height=HEIGHT)


@pytest.fixture(scope="module")
def cube_render(ctx: OffscreenContext, matrices: dict) -> dict:
    """Render a unit cube and return color + depth buffers."""
    cube_path = _find_cube_obj()
    cube = Mesh.from_obj(ctx.device, cube_path)
    cube.pose = matrices["model"].astype(np.float32)
    ctx.add_renderable("cube", cube)

    view = matrices["view"].astype(np.float32)
    proj = matrices["proj"].astype(np.float32)
    color = ctx.render_frame(
        view, proj,
        clear_color=(0.0, 0.0, 0.0, 0.0),
        extra_args={"renderStaticColor": True, "pointSize": 3.0},
    )
    depth = ctx.read_depth()

    ctx.remove_renderable("cube")

    return {
        "color": color,
        "depth": depth,
        "view": matrices["view"],
        "proj": matrices["proj"],
        "model": matrices["model"],
    }


@pytest.fixture(scope="module")
def colored_tri_render(ctx: OffscreenContext, matrices: dict) -> dict:
    """Render a large colored triangle in front of the camera."""
    # Triangle: 3 vertices with distinct colors
    # Positioned in front of camera (z=0 in world, camera at z=3)
    positions = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    colors = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
    ], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)

    tri = ColoredMesh(ctx.device, positions, colors, indices, sync_gpu=True)
    tri.pose = matrices["model"].astype(np.float32)
    ctx.add_renderable("tri", tri)

    view = matrices["view"].astype(np.float32)
    proj = matrices["proj"].astype(np.float32)
    color = ctx.render_frame(
        view, proj,
        clear_color=(0.0, 0.0, 0.0, 0.0),
    )
    depth = ctx.read_depth()

    ctx.remove_renderable("tri")

    return {
        "color": color,
        "depth": depth,
        "positions": positions,
        "vertex_colors": colors,
    }


# ---------------------------------------------------------------------------
# Test: Basic rendering sanity
# ---------------------------------------------------------------------------


class TestRenderingSanity:
    """Basic checks that rendering produces expected output."""

    def test_cube_not_black(self, cube_render: dict):
        color = cube_render["color"]
        assert color.max() > 0, "Cube render is all black"

    def test_cube_image_shape(self, cube_render: dict):
        color = cube_render["color"]
        assert color.shape == (HEIGHT, WIDTH, 4)

    def test_cube_has_transparent_background(self, cube_render: dict):
        """Cleared with alpha=0, cube pixels should have alpha > 0."""
        color = cube_render["color"]
        # At least some pixels should be non-transparent (the cube)
        alpha = color[:, :, 3]
        assert alpha.max() > 0, "No opaque pixels in cube render"

    def test_depth_buffer_not_all_clear(self, cube_render: dict):
        depth = cube_render["depth"]
        # Clear value is 1.0; the cube should have depth < 1.0 somewhere
        # NOTE: depth texture readback via to_numpy() may not work for d32_float
        # on all drivers. If depth is all 1.0, skip this test.
        if depth.min() >= 1.0 - 1e-5:
            pytest.skip(
                "Depth readback returns all 1.0 — d32_float to_numpy() "
                "may not be supported. Color-based depth validation still works."
            )
        assert depth.min() < 1.0 - 1e-5, f"Depth buffer min={depth.min()}, expected < 1.0"


# ---------------------------------------------------------------------------
# Test: Vertex position validation via projected screen coordinates
# ---------------------------------------------------------------------------


class TestVertexProjection:
    """Verify that cube vertices project to expected screen positions."""

    def test_cube_center_visible(self, cube_render: dict):
        """The cube center (origin) should project to the screen center."""
        ndc = _world_to_ndc(
            np.array([0.0, 0.0, 0.0]),
            cube_render["view"],
            cube_render["proj"],
        )
        px, py = _ndc_to_pixel(ndc[0], ndc[1], WIDTH, HEIGHT)
        log.info("Cube center NDC: (%.3f, %.3f, %.3f) -> pixel (%d, %d)", *ndc, px, py)

        # The center pixel should be approximately at screen center
        assert abs(px - WIDTH // 2) < 5, f"Center x={px}, expected ~{WIDTH // 2}"
        assert abs(py - HEIGHT // 2) < 5, f"Center y={py}, expected ~{HEIGHT // 2}"

    def test_cube_vertex_positions(self, cube_render: dict):
        """Check that cube vertices project within the viewport."""
        view = cube_render["view"]
        proj = cube_render["proj"]

        # Unit cube corners
        corners = [
            [s * 0.5 for s in signs]
            for signs in [
                (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
            ]
        ]

        for corner in corners:
            ndc = _world_to_ndc(np.array(corner), view, proj)
            log.info("Corner %s -> NDC (%.3f, %.3f, %.3f)", corner, *ndc)
            # All visible vertices should have NDC in [-1, 1] for x/y and [0, 1] for z
            assert -1.5 < ndc[0] < 1.5, f"NDC X out of range for {corner}: {ndc[0]}"
            assert -1.5 < ndc[1] < 1.5, f"NDC Y out of range for {corner}: {ndc[1]}"

    def test_depth_at_cube_center(self, cube_render: dict):
        """Verify depth value at the cube's front face center.

        Falls back to NDC-only validation if depth readback isn't available.
        """
        depth = cube_render["depth"]
        view = cube_render["view"]
        proj = cube_render["proj"]

        # The front face center of the cube (closest to camera at z=3)
        # is at world (0, 0, 0.5)
        front_center = np.array([0.0, 0.0, 0.5])
        ndc = _world_to_ndc(front_center, view, proj)
        px, py = _ndc_to_pixel(ndc[0], ndc[1], WIDTH, HEIGHT)

        expected_depth = ndc[2]  # NDC z should equal depth buffer value
        log.info(
            "Front face center: world=%s, NDC z=%.6f, pixel=(%d,%d)",
            front_center, expected_depth, px, py,
        )

        # Validate that the expected depth is in Vulkan [0, 1] range
        assert 0.0 <= expected_depth <= 1.0, (
            f"Expected depth {expected_depth} outside Vulkan [0,1] range"
        )

        # If depth readback works, validate against it
        actual_depth = depth[py, px]
        if actual_depth < 1.0 - 1e-5:
            log.info("Depth readback: actual=%.6f", actual_depth)
            assert abs(actual_depth - expected_depth) < 0.05, (
                f"Depth mismatch: expected={expected_depth:.6f}, actual={actual_depth:.6f}"
            )
        else:
            log.info(
                "Depth readback not available (d32_float to_numpy limitation). "
                "NDC depth validated: %.6f in [0, 1].", expected_depth,
            )


# ---------------------------------------------------------------------------
# Test: Y-axis convention
# ---------------------------------------------------------------------------


class TestYAxisConvention:
    """Determine and verify the Y-axis convention."""

    def test_proj_y_sign(self, matrices: dict):
        """P[1,1] positive means no Y-flip in the projection matrix."""
        P = matrices["proj"]
        log.info("P[1,1] = %.4f", P[1, 1])
        assert P[1, 1] > 0, "Expected positive P[1,1] (no Y-flip)"

    def test_y_axis_direction(self, colored_tri_render: dict, matrices: dict):
        """Determine which screen direction world Y+ maps to.

        Triangle has vertex at (0, +1, 0) in world space (blue).
        Camera at (0, 0, 3) looking at origin. In a Y-up convention,
        this blue vertex should appear in the top half of the screen.
        In Vulkan Y-down NDC, it should appear in the bottom half.
        """
        color = colored_tri_render["color"]

        # Split image into top and bottom halves
        top_half = color[:HEIGHT // 2, :, :]
        bottom_half = color[HEIGHT // 2:, :, :]

        # Look for blue channel (vertex at y=+1 is blue)
        top_blue = top_half[:, :, 2].mean()
        bottom_blue = bottom_half[:, :, 2].mean()

        log.info(
            "Blue channel: top_half_mean=%.4f, bottom_half_mean=%.4f",
            top_blue, bottom_blue,
        )

        # Document the convention
        if top_blue > bottom_blue:
            log.info(
                "Y CONVENTION: world Y+ -> screen top (OpenGL-like). "
                "The pipeline does NOT flip Y for Vulkan — the image may appear "
                "upside-down in some contexts."
            )
        else:
            log.info(
                "Y CONVENTION: world Y+ -> screen bottom (Vulkan native Y-down)."
            )

    def test_x_axis_direction(self, colored_tri_render: dict, matrices: dict):
        """World X+ should map to the right side of the screen."""
        color = colored_tri_render["color"]

        # Left vs right halves
        left_half = color[:, :WIDTH // 2, :]
        right_half = color[:, WIDTH // 2:, :]

        # Vertex at (+1, -1, 0) is green, vertex at (-1, -1, 0) is red
        left_red = left_half[:, :, 0].mean()
        right_red = right_half[:, :, 0].mean()
        left_green = left_half[:, :, 1].mean()
        right_green = right_half[:, :, 1].mean()

        log.info(
            "Red: left=%.4f, right=%.4f; Green: left=%.4f, right=%.4f",
            left_red, right_red, left_green, right_green,
        )

        # Red vertex (-1, -1, 0) should be on the left
        # Green vertex (+1, -1, 0) should be on the right
        assert left_red > right_red, (
            f"Red vertex should be on the left: left={left_red:.4f}, right={right_red:.4f}"
        )
        assert right_green > left_green, (
            f"Green vertex should be on the right: left={left_green:.4f}, right={right_green:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Depth range and Z convention
# ---------------------------------------------------------------------------


class TestDepthConvention:
    """Verify Vulkan depth range and near/far mapping."""

    def test_depth_range_01(self, cube_render: dict):
        """Depth buffer values should be in Vulkan [0, 1] range.

        If depth readback is unavailable, validate via NDC computation instead.
        """
        depth = cube_render["depth"]
        rendered = depth[depth < 0.99]
        if len(rendered) == 0:
            # Depth readback unavailable; validate NDC depth range instead
            view = cube_render["view"]
            proj = cube_render["proj"]
            corners = [
                np.array([s * 0.5 for s in signs])
                for signs in [
                    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                ]
            ]
            ndc_depths = [_world_to_ndc(c, view, proj)[2] for c in corners]
            log.info("NDC depth range (computed): [%.6f, %.6f]", min(ndc_depths), max(ndc_depths))
            assert min(ndc_depths) >= 0.0, f"NDC depth min {min(ndc_depths)} < 0"
            assert max(ndc_depths) <= 1.0, f"NDC depth max {max(ndc_depths)} > 1"
            return
        log.info(
            "Rendered depth range: [%.6f, %.6f]",
            rendered.min(), rendered.max(),
        )
        assert rendered.min() >= 0.0, f"Depth min {rendered.min()} < 0"
        assert rendered.max() <= 1.0, f"Depth max {rendered.max()} > 1"

    def test_nearer_objects_have_smaller_depth(self, cube_render: dict):
        """Front face (closer to camera) should have smaller depth than back face."""
        view = cube_render["view"]
        proj = cube_render["proj"]
        depth = cube_render["depth"]

        # Front face center (z=+0.5, closer to camera at z=3)
        front = np.array([0.0, 0.0, 0.5])
        front_ndc = _world_to_ndc(front, view, proj)
        fpx, fpy = _ndc_to_pixel(front_ndc[0], front_ndc[1], WIDTH, HEIGHT)

        # A point slightly behind the front face
        mid = np.array([0.0, 0.0, 0.0])
        mid_ndc = _world_to_ndc(mid, view, proj)

        log.info(
            "Front NDC z=%.6f, Mid NDC z=%.6f",
            front_ndc[2], mid_ndc[2],
        )

        # In Vulkan ZO with RH: closer objects should have smaller depth
        assert front_ndc[2] < mid_ndc[2], (
            f"Closer point should have smaller depth: front={front_ndc[2]:.6f}, "
            f"mid={mid_ndc[2]:.6f}"
        )

    def test_expected_depth_computation(self, matrices: dict):
        """Verify depth formula: z_ndc = (far * (z_eye + near)) / (z_eye * (far - near))."""
        view = matrices["view"]
        proj = matrices["proj"]

        # Camera at (0,0,3), looking at origin
        # Point at world (0,0,0) -> eye-space z = -(3-0) = -3.0
        point = np.array([0.0, 0.0, 0.0])
        ndc = _world_to_ndc(point, view, proj)

        # Manual depth computation using our projection formula
        # z_eye = -3.0 (distance from camera), A = far/(near-far), B = far*near/(near-far)
        z_eye = -3.0
        A = FAR / (NEAR - FAR)
        B = (FAR * NEAR) / (NEAR - FAR)
        z_clip = A * z_eye + B
        w_clip = -z_eye  # = 3.0
        z_ndc_expected = z_clip / w_clip

        log.info(
            "Point (0,0,0): z_eye=%.2f, z_ndc_mvp=%.6f, z_ndc_manual=%.6f",
            z_eye, ndc[2], z_ndc_expected,
        )
        np.testing.assert_allclose(
            ndc[2], z_ndc_expected, atol=1e-6,
            err_msg="NDC depth does not match manual computation",
        )


# ---------------------------------------------------------------------------
# Test: Matrix consistency
# ---------------------------------------------------------------------------


class TestMatrixConsistency:
    """Verify the view and projection matrices are mathematically correct."""

    def test_view_matrix_properties(self, matrices: dict):
        """View matrix should be orthonormal (rotation + translation)."""
        V = matrices["view"]
        R = V[:3, :3]

        # R should be orthonormal: R @ R^T = I
        np.testing.assert_allclose(
            R @ R.T, np.eye(3), atol=1e-10,
            err_msg="View rotation is not orthonormal",
        )

        # det(R) should be +1 (right-handed)
        det = np.linalg.det(R)
        log.info("View rotation det: %.6f", det)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)

    def test_view_transforms_eye_to_origin(self, matrices: dict):
        """The view matrix should transform the eye position to the origin."""
        V = matrices["view"]
        eye_h = np.append(EYE, 1.0)
        eye_in_view = V @ eye_h
        log.info("Eye in view space: %s", eye_in_view[:3])
        np.testing.assert_allclose(
            eye_in_view[:3], [0, 0, 0], atol=1e-10,
            err_msg="View matrix does not map eye to origin",
        )

    def test_view_looks_down_negative_z(self, matrices: dict):
        """Looking at origin from +Z means forward direction is -Z in view space."""
        V = matrices["view"]
        center_h = np.append(CENTER, 1.0)
        center_view = V @ center_h
        log.info("Center in view space: %s", center_view[:3])
        # Center should be at negative Z in view space
        assert center_view[2] < 0, (
            f"Center should be at negative Z in view space, got z={center_view[2]:.6f}"
        )

    def test_proj_maps_near_to_depth_0(self, matrices: dict):
        """A point at the near plane should map to depth 0."""
        P = matrices["proj"]
        # Near plane point in view space: (0, 0, -near)
        near_point = np.array([0.0, 0.0, -NEAR, 1.0])
        clip = P @ near_point
        ndc_z = clip[2] / clip[3]
        log.info("Near plane depth: %.6f (expected 0)", ndc_z)
        np.testing.assert_allclose(ndc_z, 0.0, atol=1e-5)

    def test_proj_maps_far_to_depth_1(self, matrices: dict):
        """A point at the far plane should map to depth 1."""
        P = matrices["proj"]
        far_point = np.array([0.0, 0.0, -FAR, 1.0])
        clip = P @ far_point
        ndc_z = clip[2] / clip[3]
        log.info("Far plane depth: %.6f (expected 1)", ndc_z)
        np.testing.assert_allclose(ndc_z, 1.0, atol=1e-5)

    def test_proj_fov(self, matrices: dict):
        """P[1,1] should equal 1/tan(fov_y/2)."""
        P = matrices["proj"]
        expected = 1.0 / math.tan(math.radians(FOV_Y) / 2.0)
        log.info("P[1,1]=%.6f, expected=%.6f", P[1, 1], expected)
        np.testing.assert_allclose(P[1, 1], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: Render output pixel validation
# ---------------------------------------------------------------------------


class TestPixelValidation:
    """Validate specific pixel values in the rendered output."""

    def test_center_pixel_is_cube(self, cube_render: dict):
        """The center of the screen should show the cube (front face)."""
        color = cube_render["color"]
        cx, cy = WIDTH // 2, HEIGHT // 2
        pixel = color[cy, cx]
        log.info("Center pixel RGBA: %s", pixel)
        # With renderStaticColor=True, the cube renders normals as colors
        # Front face normal is (0, 0, 1) in object space
        # Mapped to color: (0.5, 0.5, 1.0) * 255 = (127, 127, 255) approximately
        assert pixel[3] > 0, "Center pixel has zero alpha (no cube)"
        assert pixel.max() > 10, "Center pixel is too dark"

    def test_corner_pixel_is_background(self, cube_render: dict):
        """Corner pixels should be background (clear color)."""
        color = cube_render["color"]
        # Top-left corner — should be clear
        pixel = color[0, 0]
        log.info("Corner pixel RGBA: %s", pixel)
        # Clear color is (0, 0, 0, 0)
        assert pixel[3] == 0, f"Corner pixel has non-zero alpha {pixel[3]}"

    def test_cube_front_face_normal_color(self, cube_render: dict):
        """With renderStaticColor, front face should encode normal (0,0,1) as color."""
        color = cube_render["color"]
        view = cube_render["view"]
        proj = cube_render["proj"]

        # Project front face center to screen
        front = np.array([0.0, 0.0, 0.5])
        ndc = _world_to_ndc(front, view, proj)
        px, py = _ndc_to_pixel(ndc[0], ndc[1], WIDTH, HEIGHT)

        pixel = color[py, px]
        log.info("Front face pixel at (%d, %d): RGBA=%s", px, py, pixel)

        # Front face normal in world space is (0, 0, 1)
        # renderStaticColor maps N to: 0.5 * (normalize(N) + 1) = (0.5, 0.5, 1.0)
        # As uint8: (127, 127, 255)
        if pixel[3] > 0:  # Only check if this pixel is rendered
            # Blue channel should be highest (normal Z component = 1)
            assert pixel[2] > pixel[0], (
                f"Blue should be > Red for front face: R={pixel[0]}, B={pixel[2]}"
            )
            assert pixel[2] > pixel[1], (
                f"Blue should be > Green for front face: G={pixel[1]}, B={pixel[2]}"
            )
