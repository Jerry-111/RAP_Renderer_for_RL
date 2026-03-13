# Torch + nvdiffrast MVP Plan for RAP Renderer

## Scope

This plan is for a parity-first GPU migration starting from
[`renderer_nvdiffrast.py`](/root/RAP_Renderer_for_RL/third_party/RAP/process_data/helpers/renderer_nvdiffrast.py),
which is a frozen copy of the original RAP NumPy/OpenCV renderer.

The goal of this phase is not a full rewrite. The goal is to introduce a clean
path that can replace the CPU raster core incrementally while preserving the
original renderer's meaning.

## Frozen semantics to preserve

The following behavior should be treated as fixed during the MVP:

- `world -> lidar -> camera` transform contract in `world_to_camera_T`.
- Yaw-only object orientation in `yaw_to_rot`.
- Pinhole projection and visibility conventions in `project_points_cam`.
- `camera_params` and the current empirical camera offsets applied in
  `ScenarioRenderer.observe`.
- `COLOR_TABLE`.
- Current depth-based attenuation formulas used by:
  - `draw_polyline_depth`
  - `draw_polygon_depth`
  - `draw_cuboids_with_occlusion`
  - `draw_cuboids_depth`

These are scene semantics or visual policy. They should move to Torch later,
but they should not be redesigned in this milestone sequence.

## Parts of the original file that are reusable as-is

- Scene extraction logic in `ScenarioRenderer.observe`.
- Camera and transform helpers: `build_se3`, `yaw_to_rot`,
  `world_to_camera_T`, `project_points_cam`.
- Static calibration data: `camera_params`.
- Semantic color mapping: `COLOR_TABLE`.
- Cuboid topology and face ordering rules:
  - `vehicle_corners_local`
  - `face_indices` inside `draw_cuboids_with_occlusion`
  - `face_idxs` inside `draw_cuboid_at`

These should be copied forward into the GPU backend until parity is validated.

## Parts of the original file that should be replaced

- CPU line raster via `cv2.line` and `cv2.clipLine`.
- CPU filled polygon raster via `cv2.fillConvexPoly`.
- CPU arrow raster via `cv2.arrowedLine`.
- CPU face-level painter sort in `draw_cuboids_with_occlusion`.

Those are implementation details of the raster layer, not the semantics layer.

## File-level migration strategy

The safest way to evolve `renderer_nvdiffrast.py` is to keep the public
`ScenarioRenderer` contract stable while splitting the implementation into five
internal layers:

- Scene extraction
- Primitive lowering
- Raster submission
- Visual policy application
- Output assembly

The current file mixes these concerns. The MVP should separate them only as
much as required to make GPU replacement controlled and testable.

## Concrete refactor plan against the copied file

### Stage 0: keep the file importable without nvdiffrast

Before any rendering logic changes:

- Keep `renderer.py` and `renderer_jax.py` untouched.
- Keep `renderer_nvdiffrast.py` importable even if `nvdiffrast` is missing.
- Defer `torch` and `nvdiffrast.torch` imports to the new GPU-specific code
  path, not module import time.
- Do not add a new bridge backend flag until the first single-camera MVP path
  can render a frame.

Reason:

- This keeps the new file safe to land early and prevents half-built backend
  wiring from breaking existing workflows.

### Stage 1: define backend-neutral scene packets

Add small data containers near the top of `renderer_nvdiffrast.py` that capture
what one camera render needs, without committing yet to a specific raster API.

Suggested packet split:

- `CameraState`
  - `camera_id`
  - `width`, `height`
  - `K`
  - `T_w2c`
  - `depth_max`
- `LineOverlay`
  - semantic tag
  - world points `(N, 3)`
  - pixel-space target width
  - base color
  - attenuation mode
- `FilledRegion`
  - semantic tag
  - world hull `(N, 3)`
  - base color
  - attenuation mode
- `CuboidInstance`
  - center/position
  - dimensions
  - yaw
  - face color policy
- `ArrowOverlay`
  - world tail/head inputs or origin + yaw
  - width
  - head size
  - color
- `CameraScenePacket`
  - `camera`
  - `line_overlays`
  - `filled_regions`
  - `cuboids`
  - `arrows`

What to extract from the original file:

- The loops inside `ScenarioRenderer.observe` currently perform extraction and
  rasterization in one pass.
- First split them so `observe` builds a `CameraScenePacket`, then calls a
  renderer function.

Reason:

- This is the cleanest way to preserve semantics while changing only raster
  implementation details.

### Stage 2: add primitive-lowering helpers before any GPU submission

Do not start with a monolithic GPU render function. First prove that every
original primitive family can be lowered into a raster-friendly form.

Add separate helpers:

- `_lower_line_overlay_to_ribbon_tris(...)`
- `_lower_filled_region_to_projected_tris(...)`
- `_lower_cuboid_to_world_tris(...)`
- `_lower_arrow_to_ribbon_and_head_tris(...)`

These helpers should return triangle-oriented packets, not images.

#### 2A. Lines and boundaries

Original source:

- `draw_polyline_depth`

GPU lowering target:

- screen-space ribbon quads emitted as triangles

Concrete plan:

- Preserve near-plane clipping semantics from `draw_polyline_depth`.
- Project the surviving 3D polyline to 2D pixel coordinates.
- For each visible segment:
  - compute 2D direction
  - compute perpendicular normal
  - expand by half width in screen space
  - emit one quad as two triangles
- Assign one flat color per segment using the same mean-depth attenuation rule
  currently used before `cv2.line`.

Intentional simplification:

- Start with bevel joins or simple disconnected segment quads.
- Do not try to match OpenCV caps/joins perfectly in the first pass.

#### 2B. Crosswalks and other filled convex regions

Original source:

- `_sutherland_hodgman`
- `draw_polygon_depth`

GPU lowering target:

- triangulated projected polygons

Concrete plan:

- Preserve front-facing / behind-camera handling from `draw_polygon_depth`.
- Preserve screen rectangle clipping behavior as closely as practical.
- After projection and clipping, triangulate using a simple fan:
  - vertex `0`
  - triangles `(0, i, i+1)` for `i in 1..n-2`
- Keep one flat color for the whole region using the current mean-depth
  attenuation rule.

Intentional simplification:

- This is only for convex regions in the MVP.
- Non-convex general polygon handling is explicitly out of scope here.

#### 2C. Vehicles and traffic lights

Original source:

- `vehicle_corners_local`
- `draw_cuboids_with_occlusion`
- `draw_cuboid_at`

GPU lowering target:

- cuboid faces emitted as triangles

Concrete plan:

- Preserve cuboid local corner layout and yaw-only rotation.
- Emit each quad face as two triangles with consistent winding.
- Keep current face color logic and mean-depth attenuation.
- For traffic lights, reuse the existing fixed dimensions and world placement
  rules from `draw_cuboid_at`.

Important change:

- Replace face-level painter sorting with GPU depth buffering in the raster
  pass.
- Treat any visible differences here as expected parity review items, not
  immediate bugs, as long as the geometry is correct.

#### 2D. Heading arrows

Original source:

- `draw_heading_arrow`

GPU lowering target:

- screen-space ribbon shaft plus triangle head

Concrete plan:

- Preserve world-space arrow definition from the current function.
- Project tail/head first.
- Build a shaft ribbon exactly like a line segment.
- Build a simple isosceles triangle head in screen space aligned with the
  projected direction.
- Use a flat color, no extra style effects.

Reason:

- This matches the current semantic intent while avoiding a custom analytic
  arrow raster path.

### Stage 3: introduce Torch clip-space conversion and batch assembly

Only after primitive lowering works in CPU-side tensors should the file gain
Torch submission helpers.

Add helpers such as:

- `_pixels_to_clip_space(vertices_px, width, height)`
- `_pack_triangles_for_raster(tri_vertices_clip, tri_colors, tri_depths)`
- `_merge_pass_geometry(...)`

Representation choice for the MVP:

- Use Torch tensors on CUDA.
- Use one vertex tensor and one triangle index tensor per pass.
- Keep colors flat by repeating the same RGB value on all three vertices of a
  triangle.

Reason:

- This keeps the first nvdiffrast path simple and avoids texture logic.

### Stage 4: add a parity-first nvdiffrast raster core

This is the first point where `torch` and `nvdiffrast.torch` should be used.

Suggested internal entry points:

- `_rasterize_solid_triangle_pass(...)`
- `_compose_passes_to_canvas(...)`

MVP render-pass split:

- filled map features pass
- cuboid pass
- overlay line/arrow pass

Why multiple passes are acceptable:

- They preserve semantic clarity.
- They simplify parity debugging.
- They avoid a premature unified material system.

Expected nvdiffrast responsibilities:

- triangle rasterization
- depth test
- barycentric interpolation if needed
- optional antialiasing at pass edges later

MVP simplification:

- Start without texture sampling.
- Start with solid-color triangles only.

### Stage 5: move visual policy into explicit helpers

Right now visual behavior is implicit inside the draw functions. In the GPU
version, it should become explicit helper code so parity is easier to reason
about.

Add dedicated policy helpers:

- `_depth_attenuated_color(base_color, depth_mean, depth_max)`
- `_cuboid_face_base_colors()`
- `_traffic_light_color(is_red)`

Reason:

- This prevents visual policy from being buried inside lowering or raster code.

### Stage 6: keep CPU parity hooks during migration

Before removing any CPU code paths inside `renderer_nvdiffrast.py`, keep a
debug structure that allows comparing the lowered GPU inputs to the old CPU
render behavior.

Useful temporary hooks:

- return lowered triangles for inspection
- optionally render one primitive family at a time
- optionally bypass GPU raster and fall back to the copied CPU draw functions
  during early debugging

Reason:

- This makes parity failures explainable and local.

## Immediate edit sequence for the next implementation step

The next code-writing step after this planning phase should be small and
mechanical. A concrete order:

1. Add packet dataclasses and a packet-building helper to
   `renderer_nvdiffrast.py`.
2. Refactor `ScenarioRenderer.observe` to build the packet first, without
   changing image output yet.
3. Add primitive-lowering helpers that return triangle packets but do not call
   nvdiffrast.
4. Add a Torch-only raster stub that accepts triangle packets and raises a
   controlled `NotImplementedError` until `nvdiffrast` is installed.
5. Replace one primitive family first:
   - cuboids are the best first candidate because they already map cleanly to
     triangle faces.
6. After cuboids work, add filled regions.
7. Add line ribbons and arrow lowering last, because matching OpenCV line
   appearance is the highest visual-drift risk.

## Acceptance criteria for the first useful GPU-backed version

- `renderer_nvdiffrast.py` still preserves original scene semantics.
- A single camera can render from a `CameraScenePacket`.
- Cuboids are rendered via GPU triangles with stable depth-buffer occlusion.
- Filled regions and line-like features appear in the correct projected
  locations.
- Output differences versus `renderer.py` are explainable by the rasterization
  change, not by changed transforms or changed visual policy.
- CPU OpenCV drawing is no longer the main raster core for the primitives that
  have been migrated.

## Known risks to preserve in implementation notes

- OpenCV line appearance will not match exactly once ribbon quads replace
  `cv2.line`.
- Arrow shape will differ slightly once it becomes shaft-ribbon + triangle
  head.
- Cuboid overlaps will change where painter sorting previously disagreed with
  true depth-buffer visibility.
- Screen clipping behavior may need explicit recreation for near-parity in
  polygon and ribbon passes.

## Current repo status relevant to this plan

- `torch` is available in the current venv.
- `nvdiffrast` is not currently installed.
- Because of that, this phase intentionally stops at copied baseline +
  implementation plan, without wiring a new bridge backend yet.
