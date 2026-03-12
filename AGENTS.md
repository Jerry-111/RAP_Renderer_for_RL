# AGENTS.md

## Scope

This repo is a PufferDrive-based project used to compare and align:

- the Python RAP bridge in `minimal_rap_bridge/`
- the vendored RAP renderer in `third_party/RAP/`
- the native PufferDrive visualizer built as `./visualize`

When a task is about RAP bridge behavior, start from the bridge code and the vendored RAP renderer before touching unrelated parts of the repo.

## Where Changes Usually Belong

- `minimal_rap_bridge/`
  - Bridge-specific code changes live here.
  - Main entrypoint: `minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py`
  - Use this area for bridge CLI changes, scene selection, policy wiring, video output behavior, and bridge-side debugging scripts.

- `third_party/RAP/process_data/helpers/`
  - Renderer-side code changes live here.
  - `renderer.py` is the NumPy RAP renderer.
  - `renderer_jax.py` is the JAX/GPU RAP renderer path.
  - If the issue is in camera rendering, image generation, or RAP renderer behavior, look here.

- `pufferlib/ocean/drive/`
  - Native simulator and renderer code lives here.
  - `visualize.c` backs the native `./visualize` binary.
  - `binding.c` and `drive.py` matter when native simulator state and Python bridge behavior disagree.

- `pufferlib/config/ocean/drive.ini`
  - Native renderer settings such as `goal_behavior`, `control_mode`, `init_mode`, and `max_controlled_agents` are often controlled here.
  - `./visualize` does not expose all relevant settings as CLI flags, so parity work often requires temporary edits to this file.

- `envs/pufferdrive_rap_minimal/`
  - Minimal reproducible environment setup and validation live here.
  - Use `setup_env.sh` for bootstrap and `validate_env.py` for quick checks.

## Environment

- Preferred environment:
  - `source .venv-pufferdrive-rap/bin/activate`

- Minimal bootstrap:
  - `bash envs/pufferdrive_rap_minimal/setup_env.sh`

- Native visualizer build:
  - `bash scripts/build_ocean.sh visualize local`

- Headless native rendering requires system packages:
  - `xvfb`
  - `xauth`
  - `ffmpeg`

## Sandbox And Runtime Caveats

- Do not trust sandboxed CUDA or GPU detection for this repo.
  - Codex in sandbox may not see CUDA, NVIDIA devices, or your real GPU runtime.
  - If a task depends on JAX GPU, PyTorch CUDA, native rendering, or any GPU validation, run the command outside the sandbox.

- Do not assume "GPU is unavailable" from a sandboxed failure.
  - Re-run GPU-sensitive checks outside sandbox before concluding CUDA, JAX, or torch is broken.

- Fresh dependency installation may fail inside sandbox even when the machine is fine.
  - In sandbox, `pip` or similar commands may report DNS, internet, or index access failures.
  - If bootstrap or dependency installation fails with network-style errors, re-run outside sandbox instead of debugging the repo itself.

- Headless X / native rendering checks may also need outside-sandbox execution.
  - `xvfb-run` can fail inside sandbox even after the correct packages are installed.
  - If native render smoke tests fail in sandbox, retry outside sandbox before changing code.

## Known-Good Render Alignment Settings

Use these settings when trying to align the RAP bridge and native renderer:

- `goal_behavior = 1`
- `control_mode = "control_sdc_only"`
- `init_mode = "create_all_valid"`
- `max_controlled_agents = 1`
- Bridge box source: `map_replay`

Notes:

- Native `./visualize` uses `drive.ini` for several of these settings.
- The bridge defaults do not match native timing by default.
- For timing parity with native output, the bridge should usually use:
  - `--full-episode`
  - `--episode-length 91`
  - `--video-fps 30`

Without those overrides, the bridge commonly writes only 30 frames at 10 fps, while native writes 91 frames at 30 fps.

## Default Map And Output Notes

- Default single-map smoke target:
  - `resources/drive/binaries/map_000.bin`

- Native default test shape:
  - `xvfb-run -a -s "-screen 0 1280x720x24" ./visualize --map-name resources/drive/binaries/map_000.bin`

- Bridge default single-scene behavior:
  - when only one scene is rendered, outputs are written directly under `--out-dir`
  - when multiple scenes are rendered, outputs are nested per scene

Do not assume a `map_000/` subdirectory exists for single-scene bridge runs.

## Practical Guidance For Future Agents

- If the task says "bridge change", start in `minimal_rap_bridge/`.
- If the task says "RAP renderer change", start in `third_party/RAP/process_data/helpers/`.
- If the task says "native visualize mismatch", inspect both `pufferlib/ocean/drive/visualize.c` and `pufferlib/config/ocean/drive.ini`.
- Before diagnosing install/network problems, consider sandbox restrictions first.
- Before diagnosing missing GPU/CUDA support, move the command outside sandbox first.
