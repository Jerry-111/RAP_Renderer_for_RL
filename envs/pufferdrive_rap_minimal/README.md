# PufferDrive + RAP Minimal Environment

This folder creates one reproducible Python environment for:

- `minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py`
- RAP `ScenarioRenderer` smoke rendering
- optional Torch + `nvdiffrast` GPU raster backend preparation

It is intentionally minimal and avoids deep RAP/nuPlan setup.

The RAP renderer is vendored in this repo at:

- `third_party/RAP/process_data/helpers/renderer.py`

---

## Files

- `setup_env.sh`: bootstrap script to create/install/build
- `requirements.txt`: extra packages for RAP renderer path
- `validate_env.py`: quick sanity checks

---

## Setup

From repo root:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

The script is idempotent and can be run from any working directory.

What it runs (in order):

- create/activate venv
- install core deps (`numpy`, gym stack, RAP extras)
- install `torch` (default on)
- optionally install `nvdiffrast` from source (`INSTALL_NVDIFFRAST=1`)
- clean reinstall of JAX with pinned CUDA wheel (`jax[cuda12]==0.4.30`)
- install local package in editable mode (`NO_TRAIN=1`)
- build native extensions (`python setup.py build_ext --inplace --force`)
- run validation (full Drive check when `map_000.bin` exists)

Optional custom venv path:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh /tmp/.venv-pufferdrive-rap
```

Optional custom map dir for enforced Drive check:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh /tmp/.venv-pufferdrive-rap resources/drive/binaries
```

If your node needs a custom pip index/mirror:

```bash
PIP_EXTRA_ARGS="-i https://<index>/simple --trusted-host <index-host>" \
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

Disable optional steps if desired:

```bash
INSTALL_TORCH=0 BUILD_EXT=0 RUN_VALIDATE=0 \
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

Enable the planned Torch + `nvdiffrast` renderer environment:

```bash
INSTALL_NVDIFFRAST=1 CHECK_NVDIFFRAST=1 \
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

If you need to pin a specific GPU arch for extension compilation:

```bash
INSTALL_NVDIFFRAST=1 CHECK_NVDIFFRAST=1 \
NVDIFFRAST_CUDA_ARCH_LIST=8.6 \
bash envs/pufferdrive_rap_minimal/setup_env.sh
```

Manual JAX reset (same fix used by setup):

```bash
source .venv-pufferdrive-rap/bin/activate
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
python -m pip install -U "jax[cuda12]==0.4.30"
```

Activate:

```bash
source .venv-pufferdrive-rap/bin/activate
```

---

## Validate

Basic import + renderer smoke:

```bash
python envs/pufferdrive_rap_minimal/validate_env.py
```

Include Drive runtime smoke:

```bash
python envs/pufferdrive_rap_minimal/validate_env.py --check-drive --map-dir resources/drive/binaries
```

Include Torch + `nvdiffrast` GPU validation:

```bash
python envs/pufferdrive_rap_minimal/validate_env.py --check-nvdiffrast
```

---

## Run Minimal Bridge

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --out-dir /tmp/pufferdrive_rap_minimal \
  --frames 30 \
  --map-dir resources/drive/binaries \
  --num-maps 1
```

Renderer backend options:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend numpy
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend jax
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py ... --renderer-backend nvdiffrast
```

Use the `nvdiffrast` backend only after installing it with:

```bash
INSTALL_NVDIFFRAST=1 bash envs/pufferdrive_rap_minimal/setup_env.sh
python envs/pufferdrive_rap_minimal/validate_env.py --check-nvdiffrast
```

Minimal full-scene GPU MVP example:

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --renderer-backend nvdiffrast \
  --out-dir /tmp/pufferdrive_rap_minimal_nvdiffrast \
  --frames 30 \
  --map-dir resources/drive/binaries \
  --num-agents 32
```

---

## Notes

- `setup_env.sh` installs PufferDrive with `NO_TRAIN=1` to reduce dependency load on cluster nodes.
- Native extensions are built in-place with:
  - `python setup.py build_ext --inplace --force`
- `nvdiffrast` is not listed in `requirements.txt`; it is installed separately in
  `setup_env.sh` because the recommended source install uses `--no-build-isolation`.
- The default `nvdiffrast` source in `setup_env.sh` is pinned to a known-good
  commit and can be overridden with `NVDIFFRAST_SRC=...` if needed.
- Headless visualizer tools (`xvfb`, `xauth`, `ffmpeg`) are still system-level (apt) concerns; see:
  - `envs/pufferdrive_rap_minimal/cluster-headless-setup.md`
