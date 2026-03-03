# RAP Helpers Renderer Notes

This folder currently has two renderer implementations:

- `renderer.py`: original implementation (NumPy + OpenCV).
- `renderer_jax.py`: current experimental JAX variant for compute-heavy math paths.

## Current vs Original

`renderer.py` (original):
- Uses NumPy throughout transforms, projections, and drawing prep.
- Stable baseline used by default.

`renderer_jax.py` (current):
- Keeps the same `ScenarioRenderer` interface.
- Moves core math to JAX where practical (SE(3), yaw rotation, world-to-camera, projection kernels).
- Keeps NumPy at OpenCV boundaries and mutable CPU-side drawing/clipping logic.

## Why NumPy still exists in `renderer_jax.py`

- OpenCV drawing APIs require host-side NumPy arrays.
- Some clipping/raster-prep logic is mutation-heavy and not directly JAX-friendly without a larger rewrite.
- Camera/image buffers are still managed as CPU arrays.

## Backend Selection (Minimal Bridge)

From repo root:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend numpy
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend jax
```

Notes:
- `numpy` is default.
- `jax` requires `jax` and `jaxlib` installed in the runtime environment.
