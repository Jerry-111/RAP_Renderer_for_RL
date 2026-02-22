# PufferDrive + RAP Minimal Environment

This folder creates one reproducible Python environment for:

- `minimal_rap_bridge/render_pufferdrive_to_rap.py`
- RAP `ScenarioRenderer` smoke rendering

It is intentionally minimal and avoids deep RAP/nuPlan setup.

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

This now performs an enforced validation at the end:

- imports
- RAP renderer smoke
- PufferDrive functional smoke (`Drive` init + `reset` + multiple `step`s + global state + road edges)

Setup fails if any of those checks fail.

Optional custom venv path:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh /tmp/.venv-pufferdrive-rap
```

Optional custom map dir for enforced Drive check:

```bash
bash envs/pufferdrive_rap_minimal/setup_env.sh /tmp/.venv-pufferdrive-rap resources/drive/binaries
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

---

## Run Minimal Bridge

```bash
python minimal_rap_bridge/render_pufferdrive_to_rap.py \
  --out-dir /tmp/pufferdrive_rap_minimal \
  --frames 30 \
  --map-dir resources/drive/binaries \
  --num-maps 1 \
  --num-agents 32
```

---

## Notes

- `setup_env.sh` installs PufferDrive with `NO_TRAIN=1` to reduce dependency load on cluster nodes.
- Native extensions are built in-place with:
  - `python setup.py build_ext --inplace --force`
- Headless visualizer tools (`xvfb`, `xauth`, `ffmpeg`) are still system-level (apt) concerns; see:
  - `minimal_rap_bridge/docs/cluster-headless-setup.md`
