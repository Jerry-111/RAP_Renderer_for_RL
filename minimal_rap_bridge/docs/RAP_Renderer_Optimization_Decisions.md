# RAP GPU Renderer Decision Memo

## Purpose
This document records the high-level renderer decisions for the RAP migration so the reasoning is easy to revisit later with a clear link between the current pipeline, the chosen GPU path, and the MVP scope.

## Project context
The original RAP renderer already defines the important dataset-facing semantics:
- fixed transform contract: `world -> lidar -> camera`
- yaw-only object orientation for cuboids
- pinhole projection and visibility tests
- fixed camera calibration and current empirical extrinsic offsets
- semantic feature policy for lanes, boundaries, and crosswalks
- fixed color palette and depth-based color attenuation

The current bottleneck is not scene semantics. The bottleneck is the CPU rasterization path built around OpenCV primitives and painter-style face sorting.

## Main decision
We will build the GPU MVP as:
- **Torch-only runtime**
- **nvdiffrast as the raster backend**
- **minimal primitive lowering into GPU-friendly raster forms**

This means the migration is framed as:
- **preserve geometry and visual semantics**
- **replace CPU rasterization and approximate occlusion**

not as:
- a full dataset convention rewrite
- a full rendering framework migration
- a large scene abstraction redesign

## Why Torch-only
The current failed direction was mixing JAX with many conversions while still bottlenecking on OpenCV on CPU. That created runtime-boundary overhead without removing the slowest part of the pipeline.

Torch-only keeps one tensor owner for the renderer and aligns naturally with the chosen raster backend.

### Benefits
- one runtime boundary for the rendering pipeline
- fewer hidden sync/copy points
- simpler benchmarking and debugging
- directly compatible with nvdiffrast

### Risks
- less alignment with any broader JAX-only training stack
- some future interop work if JAX needs to re-enter later

### Why this is acceptable
The immediate goal is to replace the raster bottleneck cleanly. A pure Torch renderer is the simplest way to do that.

## Why nvdiffrast
nvdiffrast is the better final fit because the current RAP pipeline already owns the hard semantic parts:
- transforms
- projection
- camera behavior
- feature meaning
- color behavior

So the renderer does not need a higher-level framework to define cameras or scene conventions. It mainly needs a fast GPU raster core.

### Benefits
- close match to the current renderer boundary: we already know the geometry and camera rules, we just need fast rasterization
- leaner final architecture
- less framework overhead than a fuller rendering stack
- better fit for a migration that preserves current conventions

### Risks
- more custom glue than a higher-level framework
- parity work for lines, arrows, and clipping behavior
- less bring-up convenience than PyTorch3D

## Why not PyTorch3D as the main target
PyTorch3D is attractive for easier bring-up, richer abstractions, and debugging convenience. But in this project, many of its biggest advantages are less central because the renderer already has a mature camera/geometry contract.

PyTorch3D would still require primitive conversion and would add a broader rendering framework where the main need is actually a lower-level raster engine.

### Why this does not mean PyTorch3D is bad
PyTorch3D is still a useful development scaffold in other settings, especially when camera conventions or scene structure are still being figured out. It is just not the best final match for this renderer.

## Primitive decision
The original renderer is not a unified mesh renderer. It uses:
- projected line primitives for lanes and boundaries
- projected convex polygon fills for crosswalk-style regions
- cuboid faces for vehicles and traffic lights
- arrow graphics for heading indicators

For the GPU MVP, the semantic model stays the same, but the **raster-time representation** is lowered into GPU-friendly forms.

### Chosen raster-time representation
- **lanes / boundaries / arrow shafts** -> screen-space ribbon quads, rasterized as triangles
- **crosswalks / filled regions** -> triangulated projected polygons
- **vehicles / traffic lights** -> cuboids emitted as triangle faces
- **arrow heads** -> simple triangle geometry

## Why unify only at raster time
We are **not** unifying scene meaning. We are only unifying the draw format right before rasterization.

That helps because:
- one raster backend can handle all primitive families
- batching becomes simpler
- visibility and color logic become easier to reason about
- we avoid multiple GPU draw models that would recreate the original fragmentation in a more complex form

### Tradeoff
This does introduce parity work because OpenCV line and polygon drawing behavior will not be pixel-identical to triangle rasterization at first. But that parity cost is already unavoidable once the pipeline leaves OpenCV.

## What is preserved exactly in the MVP
The following should remain unchanged for parity-first migration:
- `world -> lidar -> camera` transform contract
- current `T_w2c` convention
- yaw-only rotation behavior
- current pinhole projection logic
- current visibility logic and thresholds
- camera intrinsics and extrinsics
- current empirical per-camera offsets
- semantic feature policy
- color palette
- depth-based color attenuation formula

## What changes in the MVP
The following implementation details change:
- OpenCV rasterization primitives are replaced
- face-level painter sorting is replaced where cuboids use GPU depth-buffer rasterization
- lines and arrows stop being drawn as OpenCV graphics and become triangle-based raster primitives
- polygon filling moves from CPU clipping/fill calls into a GPU-friendly triangle path

## MVP scope
The MVP should be a **minimal semantic-change GPU renderer**.

### Included in MVP
- one clean backend-neutral scene packet
- primitive lowering into GPU-friendly forms
- nvdiffrast raster path
- parity-first visual policy
- single-camera correctness first
- later multi-camera batching after parity is acceptable

### Explicitly not in MVP
- major dataset convention changes
- redesign of camera semantics
- large shader system redesign
- ambitious general-purpose rendering abstractions
- optimization around the old OpenCV path

## High-level rollout plan

### Phase 1: freeze the contract
Lock the semantic and geometric rules that must stay unchanged.

### Phase 2: define one scene packet
Represent one camera render request in a clear backend-neutral format.

### Phase 3: lower primitives minimally
Convert original primitives into GPU-friendly forms without changing their meaning.

### Phase 4: implement a parity-first GPU renderer
Use a small number of semantically motivated render passes rather than one monolithic redesign.

### Phase 5: port visual policy exactly
Keep colors, depth attenuation, and class behavior aligned with the original renderer.

### Phase 6: replace approximate occlusion where justified
Use depth-buffer rasterization for cuboids and treat this as a deliberate improvement.

### Phase 7: define parity gates
Judge success by stable visual behavior, not by pixel-perfect equality everywhere.

### Phase 8: optimize only after parity
After the single-camera path is stable, move on to batching, persistent GPU buffers, and reduced transfer overhead.

## Main benefits of this path
- directly attacks the current bottleneck
- keeps the dataset-facing semantics stable
- avoids wasted effort on full framework migration
- gives a clean explanation for rejecting the mixed JAX approach
- makes later performance work more meaningful because the raster core is actually on GPU

## Main risks
- line and arrow appearance may differ from OpenCV initially
- cuboid overlap behavior can shift when moving from painter sort to depth buffering
- clipping details may need careful tuning for parity
- nvdiffrast requires more renderer-owned glue than a higher-level framework

## Decision rationale in one sentence
Because the current RAP pipeline already defines its own geometry, projection, and semantic rules, the MVP should preserve those and replace only the CPU rasterization core with a lean GPU raster backend using minimal primitive lowering.
