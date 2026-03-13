# RAP Helpers Renderer Notes

This folder currently has two renderer implementations:

- `renderer.py`: original implementation (NumPy + OpenCV).
- `renderer_jax.py`: current experimental JAX variant for compute-heavy math paths.
- `renderer_nvdiffrast.py`: frozen copy of the original renderer used as the
  parity-first starting point for a future Torch + nvdiffrast raster backend.

## Current vs Original

`renderer.py` (original):
- Uses NumPy throughout transforms, projections, and drawing prep.
- Stable baseline used by default.

`renderer_jax.py` (current):
- Keeps the same `ScenarioRenderer` interface.
- Moves core math to JAX where practical (SE(3), yaw rotation, world-to-camera, projection kernels).
- Keeps NumPy at OpenCV boundaries and mutable CPU-side drawing/clipping logic.

`renderer_nvdiffrast.py` (planning baseline):
- Intentionally preserves the original RAP semantics and CPU behavior for now.
- Exists so GPU migration work can happen in a separate file without disturbing
  the stable NumPy path or the experimental JAX path.
- The staged migration notes live in `renderer_nvdiffrast_plan.md`.

## Why NumPy still exists in `renderer_jax.py`

- OpenCV drawing APIs require host-side NumPy arrays.
- Some clipping/raster-prep logic is mutation-heavy and not directly JAX-friendly without a larger rewrite.
- Camera/image buffers are still managed as CPU arrays.

## Backend Selection (Minimal Bridge)

From repo root:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend numpy
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend jax
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend nvdiffrast
```

Notes:
- `numpy` is default.
- `jax` requires `jax` and `jaxlib` installed in the runtime environment.
- `nvdiffrast` requires `torch`, `nvdiffrast`, and CUDA-visible PyTorch.
