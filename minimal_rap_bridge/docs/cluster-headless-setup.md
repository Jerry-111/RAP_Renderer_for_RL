# Cluster Headless Setup (Kubernetes)

This guide captures the exact process for running PufferDrive visualization/inference on a headless cluster pod.

## Scope

This validates:
- Python environment setup
- Native build (`visualize`)
- Headless rendering with `xvfb`
- Pretrained policy inference
- Video export

This does **not** fully validate long training/evaluation pipelines.

## 1) Clone and enter the repo

```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
```

Note: Linux paths are case-sensitive. The folder is `PufferDrive`, not `pufferdrive`.

## 2) Create one Python environment

Use one env style at a time. Avoid stacking `conda` + `.venv` prompts for daily use.

If `uv` is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
python setup.py build_ext --inplace --force
```

## 3) Install headless runtime tools in the pod

If you have root in the container/pod, prefer apt packages:

```bash
apt-get update
apt-get install -y xvfb xauth ffmpeg
```

Why: conda `xvfb` variants (for older COS builds) may fail with missing system libs such as `libcrypto.so.10`.

## 4) Build visualizer binary

```bash
bash scripts/build_ocean.sh visualize local
```

You may see warnings; they are not fatal.

## 5) Run headless inference + visualization

The default config expects maps under `resources/drive/binaries/training/`, but a fresh checkout often only has `resources/drive/binaries/map_000.bin`.

Use an explicit map path:

```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize --map-name resources/drive/binaries/map_000.bin
```

This rolls out the pretrained policy and exports video.

## 6) Output locations

Expected output directory:

```text
resources/drive/puffer_drive_weights/video/
```

Example output file:

```text
resources/drive/puffer_drive_weights/video/map_000_topdown.mp4
```

## 7) Copy video from pod to local machine

Run this from your local machine:

```bash
kubectl cp <namespace>/<pod-name>:/root/PufferDrive/resources/drive/puffer_drive_weights/video/map_000_topdown.mp4 ./map_000_topdown.mp4
```

## 8) Common issues and fixes

`bash: uv: command not found`
- Use `python -m venv .venv` and plain `pip`.

`bash: .venv/bin/activate: No such file or directory`
- Create the venv first with `python -m venv .venv`.

`xvfb-run: command not found`
- Install apt `xvfb` (preferred in this setup).

`xvfb-run: error: xauth command not found`
- Install apt `xauth`.

`Xvfb: error while loading shared libraries: libcrypto.so.10`
- Stop using that conda `Xvfb` binary; use apt `xvfb`.

`File Not Found - resources/drive/binaries/training/map_000.bin`
- Pass `--map-name resources/drive/binaries/map_000.bin` or create the expected training path/symlink.

`LeakSanitizer` output at process exit
- `local` build enables sanitizers by design. This is a debug report, not proof that rendering failed.
- For cleaner runs:

```bash
bash scripts/build_ocean.sh visualize fast
```

## 9) Minimal smoke-test checklist

1. `python -c "import pufferlib; print('ok')"`
2. `bash scripts/build_ocean.sh visualize local`
3. `xvfb-run ... ./visualize --map-name resources/drive/binaries/map_000.bin`
4. Confirm `.mp4` exists under `resources/drive/puffer_drive_weights/video/`

