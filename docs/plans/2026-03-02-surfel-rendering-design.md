# Surfel Rendering Design

Normal-oriented, textured hexagonal surfels for structured pointcloud rendering.

## Problem

Current pointcloud rendering uses either screen-aligned billboard sprites or naive surface
triangulation. Billboards don't respect surface orientation, causing visual artifacts at
grazing angles. The surface triangulation mode has depth-consistency heuristics but no true
normal computation.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Normal storage | Separate `StructuredBuffer<float3>` | Matches existing position/texcoord pattern |
| Sprite shape | Hexagon (6 edges, 18 verts) | Good disc approximation, reasonable GS output |
| UV projection | Per-vertex in geometry shader | Accurate for oriented sprites, handles distortion |
| Sprite sizing | Auto from depth + camera intrinsics | Fills gaps correctly at all depths |
| Code organization | New shader + renderer pair | Clean separation from legacy Holoscan code |

## Components

### 1. Normal Computation (`depth_unproject.slang`)

New compute kernel `compute_normals`:

- **Input**: `position_buffer` (StructuredBuffer<float3>), depth width/height
- **Output**: `normal_buffer` (StructuredBuffer<float3>)
- **Algorithm**: Central-difference cross-product on structured grid
  - `right = P[i+1,j] - P[i-1,j]`, `down = P[i,j+1] - P[i,j-1]`
  - `normal = normalize(cross(right, down))`
  - One-sided differences at boundaries, zero normal for invalid neighbors
- **Dispatch**: [16,16,1] thread groups, after `compute_pointcloud`

### 2. Shared Camera Math (`camera_math.slang`)

Extract from `depth_unproject.slang` into shared include:

- `CameraIntrinsics` struct (fx, fy, cx, cy, k1-k6, p1, p2, max_radius)
- `ColorProjectionParams` struct (intrinsics, image dims, depth_to_color matrix)
- `project_forward()` function (3D point -> 2D UV with Brown-Conrady distortion)

Both `depth_unproject.slang` and `pointcloud_surfels.slang` import this file.

### 3. Surfel Shader (`pointcloud_surfels.slang`)

**Vertex shader**: Pass-through `SV_VertexID`.

**Geometry shader** (point -> triangle_strip, max 18 verts):

1. Read position P, normal N from structured buffers. Skip if N = zero.
2. Compute sprite radius: `radius = depth * (1.0 / fy) * sprite_scale`
   - `fy` from depth camera intrinsics, `sprite_scale` uniform (default 1.5)
3. Build tangent frame: `T = normalize(cross(ref, N))`, `B = cross(N, T)`
   - Reference vector: `(0,1,0)` unless nearly parallel to N, then `(1,0,0)`
4. Generate hexagon vertices: `P + radius * (cos(k*60) * T + sin(k*60) * B)` for k=0..5
5. Project each 3D vertex through color camera model to compute UV
6. Emit center + 6 outer vertices as triangle fan (encoded as triangle strip)

**Fragment shader**: Sample color texture at interpolated UV, discard if UV out of [0,1].

### 4. Renderer (`pointcloud_surfel_renderer.py`)

- Pipeline: `triangle_strip`, no input layout, depth test with `d32_float`
- Structured buffer bindings: positions, normals, UVs (from compute shader center UVs, unused by surfels but kept for fallback), color texture + sampler
- Uniforms: model/view/proj matrices, `ColorProjectionParams` struct, depth camera `fy`, `sprite_scale`
- Draw call: `vertex_count` = number of points (GS expands each)

### 5. DepthUnprojector Changes

- New `m_normal_buffer` (StructuredBuffer<float3>)
- New `compute_normals` kernel compiled at init
- Dispatched after pointcloud computation in `unproject()`
- New `normal_buffer` property and `normals_to_numpy()` readback

### 6. view_depth_pointcloud.py Changes

- Use `PointcloudSurfelRenderer` instead of `PointcloudSpritesRenderer`
- Pass `unprojector.normal_buffer` to pointcloud renderable
- Pass color camera parameters to renderer for geometry-shader UV projection
- Add `sprite_scale` to extra_args for interactive tuning

## Data Flow

```
Depth Image (uint16)
    |
    v
compute_pointcloud (existing) --> position_buffer (float3)
    |                              texcoord_buffer (float2)  [center UVs]
    v
compute_normals (NEW)          --> normal_buffer (float3)
    |
    v
Pointcloud Renderable (holds position + normal + UV + color texture)
    |
    v
PointcloudSurfelRenderer
    |
    v
Geometry Shader: point --> hexagon (6 triangles)
  - oriented by normal (tangent frame)
  - auto-sized from depth
  - UV per vertex via color camera projection
    |
    v
Fragment Shader: sample color texture at projected UV
```

## Testing

- Unit test: normals computation (known geometry, verify normal directions)
- Visual test: view_depth_pointcloud.py with surfel renderer
- Compare: billboard vs surfel rendering at grazing angles
