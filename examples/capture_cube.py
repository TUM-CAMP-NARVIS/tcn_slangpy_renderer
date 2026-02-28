#!/usr/bin/env python3
"""Render a unit cube with known camera parameters and capture with RenderDoc.

This script:
1. Loads the RenderDoc in-app API (requires librenderdoc.so)
2. Creates an OffscreenContext and loads the unit cube
3. Captures one frame using StartFrameCapture/EndFrameCapture
4. Saves a reference PNG and expected matrices JSON

Usage::

    python examples/capture_cube.py [--output-dir /tmp/captures]

The librenderdoc.so must be loadable (set RENDERDOC_PYTHON_PATH or have it in
the default location ~/.local/renderdoc/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from slangpy_renderer import Mesh, OffscreenContext
from slangpy_renderer.debug.renderdoc_api import RenderDocAPI
from slangpy_renderer.offscreen import look_at, vulkan_rh_zo_perspective


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Capture a unit cube frame with RenderDoc.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/rdc_captures"),
        help="Directory to save capture files",
    )
    parser.add_argument(
        "--width", type=int, default=256, help="Render width"
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Render height"
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load RenderDoc in-app API ---
    print("Loading RenderDoc in-app API...")
    rdoc = RenderDocAPI()
    capture_template = str(output_dir / "cube_capture")
    rdoc.set_capture_path(capture_template)

    # --- Step 2: Set up known camera parameters ---
    eye = np.array([0.0, 0.0, 3.0])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    fov_y = 60.0
    near = 0.1
    far = 100.0
    width = args.width
    height = args.height
    aspect = float(width) / float(height)

    view_matrix = look_at(eye, center, up)
    proj_matrix = vulkan_rh_zo_perspective(fov_y, aspect, near, far)
    model_matrix = np.eye(4, dtype=np.float64)

    # --- Step 3: Create OffscreenContext and load cube ---
    print(f"Creating OffscreenContext ({width}x{height})...")
    ctx = OffscreenContext(width=width, height=height)

    asset_root = Path(__file__).resolve().parent.parent / "slangpy_renderer" / "assets"
    cube_path = asset_root / "models" / "cube.obj"
    if not cube_path.exists():
        # Try installed package path
        import slangpy_renderer
        pkg_dir = Path(slangpy_renderer.__file__).parent
        cube_path = pkg_dir / "assets" / "models" / "cube.obj"

    print(f"Loading cube from {cube_path}")
    cube = Mesh.from_obj(ctx.device, str(cube_path))
    cube.pose = model_matrix.astype(np.float32)
    ctx.add_renderable("cube", cube)

    # --- Step 4: Capture frame ---
    print("Starting frame capture...")
    rdoc.start_capture()

    image = ctx.render_frame(
        view_matrix.astype(np.float32),
        proj_matrix.astype(np.float32),
        clear_color=(0.2, 0.2, 0.2, 1.0),
        extra_args={"renderStaticColor": True, "pointSize": 3.0},
    )

    success = rdoc.end_capture()
    print(f"Frame capture ended (success={success})")

    # --- Step 5: Retrieve capture path ---
    num_captures = rdoc.get_num_captures()
    if num_captures == 0:
        print("ERROR: No captures produced!", file=sys.stderr)
        sys.exit(1)

    capture_path = rdoc.get_capture(num_captures - 1)
    print(f"Capture saved to: {capture_path}")

    # --- Step 6: Save reference PNG ---
    try:
        from PIL import Image
        png_path = output_dir / "cube_reference.png"
        Image.fromarray(image).save(str(png_path))
        print(f"Reference PNG saved to: {png_path}")
    except ImportError:
        print("Pillow not installed, skipping PNG save")

    # --- Step 7: Save expected matrices as JSON ---
    expected = {
        "eye": eye.tolist(),
        "center": center.tolist(),
        "up": up.tolist(),
        "fov_y": fov_y,
        "near": near,
        "far": far,
        "width": width,
        "height": height,
        "view": view_matrix.tolist(),
        "proj": proj_matrix.tolist(),
        "model": model_matrix.tolist(),
        "capture_path": capture_path,
    }
    json_path = output_dir / "expected_matrices.json"
    with open(json_path, "w") as f:
        json.dump(expected, f, indent=2)
    print(f"Expected matrices saved to: {json_path}")

    print("\nDone! To analyze the capture:")
    print(f"  rdc open {capture_path}")
    print(f"  rdc draws")
    print(f"  rdc pipeline <EID>")


if __name__ == "__main__":
    main()
