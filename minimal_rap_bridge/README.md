# Minimal RAP Bridge

Quick runbook for rendering selected WOMD scenes with:
- the Python bridge (`render_pufferdrive_to_rap_mvp.py`)
- the native PufferDrive renderer (`./visualize`)

Use `--renderer-backend jax` to run the JAX-optimized RAP renderer copy (`process_data/helpers/renderer_jax.py`).
For environment setup, use `envs/pufferdrive_rap_minimal/setup_env.sh` (it installs JAX by default via `INSTALL_JAX=1`).

## 1) Bridge render with PyTorch policy (videos only, no JPG frames)

Run from repo root (`/root/RAP_Renderer_for_RL`):

```bash
source .venv/bin/activate
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --map-dir /jerry_slow_vol/womd_bins/validation \
  --replay-all-scenes \
  --scene-ids 18,35,63 \
  --renderer-backend jax \
  --num-maps 1 \
  --action-source policy \
  --goal-behavior 1 \
  --control-mode control_sdc_only \
  --init-mode create_all_valid \
  --write-video \
  --no-save-frames \
  --out-dir /tmp/pufferdrive_rap_mvp_s18_s35_s63
```

Outputs are written under:
- `/tmp/pufferdrive_rap_mvp_s18_s35_s63/map_018/`
- `/tmp/pufferdrive_rap_mvp_s18_s35_s63/map_035/`
- `/tmp/pufferdrive_rap_mvp_s18_s35_s63/map_063/`

Each scene folder contains per-camera MP4s (e.g., `replay_CAM_F0.mp4`).

Notes:
- `--control-mode control_sdc_only` keeps policy control on the SDC.
- With default `--box-source auto`, non-controlled actors are rendered from WOMD map trajectories (`map_replay`) so you can see surrounding traffic.
- If you set `--control-mode control_wosac` (or similar), many/all valid agents become policy-controlled.
- Default policy checkpoint is `pufferlib/resources/drive/pufferdrive_weights.pt` (WOMD baseline).
- Default is pufferl-style stochastic sampling. For deterministic rendering, add: `--policy-deterministic`.

## 2) Native renderer for the same 3 scenes (write to /tmp)

If needed, build once:

```bash
source .venv/bin/activate
bash scripts/build_ocean.sh visualize local
```

Then render scenes to `/tmp/pd_native_s18_s35_s63`:

```bash
mkdir -p /tmp/pd_native_s18_s35_s63
for sid in 018 035 063; do
  xvfb-run -a -s "-screen 0 1280x720x24" \
    ./visualize \
    --map-name /jerry_slow_vol/womd_bins/validation/map_${sid}.bin \
    --output-topdown /tmp/pd_native_s18_s35_s63/map_${sid}_topdown.mp4 \
    --output-agent /tmp/pd_native_s18_s35_s63/map_${sid}_agent.mp4
done
```

## 3) Native goal behavior control (set to mode 1)

Native `./visualize` currently does not expose `--goal-behavior` as a CLI flag.
It reads this from `pufferlib/config/ocean/drive.ini`.

Set `goal_behavior=1` before running native renders:

```bash
sed -i 's/^goal_behavior = .*/goal_behavior = 1/' pufferlib/config/ocean/drive.ini
```

Optional: restore default respawn mode afterwards:

```bash
sed -i 's/^goal_behavior = .*/goal_behavior = 0/' pufferlib/config/ocean/drive.ini
```

## 4) Profile 10 scenes (forward/step/RAP render/scene total)

Run from repo root (`/root/RAP_Renderer_for_RL`):

```bash
source .venv/bin/activate
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --map-dir /jerry_slow_vol/womd_bins/validation \
  --replay-all-scenes \
  --max-scenes 10 \
  --num-maps 1 \
  --action-source policy \
  --control-mode control_sdc_only \
  --init-mode create_all_valid \
  --no-write-video \
  --no-save-frames \
  --no-log-frame-pixels \
  --out-dir /tmp/pufferdrive_rap_profile_10
```

At the end it prints:
- `avg_inference_forward_ms`
- `avg_step_ms`
- `avg_rap_renderer_ms`
- `avg_scene_total_ms`

## 5) Profile all 75 validation scenes (comprehensive, no video output)

Use this when you want a broader benchmark across all scenes while measuring only:
- policy inference forward
- env step
- RAP renderer
- total scene time

Run from repo root (`/root/RAP_Renderer_for_RL`):

```bash
source .venv/bin/activate

OUT_DIR="/tmp/pd_profile_75scenes_goal1_sdc_policy_mapreplay"
LOG_FILE="/tmp/pd_profile_75scenes_goal1_sdc_policy_mapreplay.log"

python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --map-dir /jerry_slow_vol/womd_bins/validation \
  --replay-all-scenes \
  --max-scenes 75 \
  --num-maps 1 \
  --frames 91 \
  --episode-length 91 \
  --action-source policy \
  --policy-device cuda:0 \
  --goal-behavior 1 \
  --control-mode control_sdc_only \
  --init-mode create_all_valid \
  --box-source map_replay \
  --no-write-video \
  --no-save-frames \
  --no-log-frame-pixels \
  --out-dir "$OUT_DIR" | tee "$LOG_FILE"
```

Read the aggregate timing results:

```bash
grep -A5 "=== Timing Aggregate (Across Scenes) ===" "$LOG_FILE"
```

Optional sanity checks:

```bash
grep "^Scenes to replay:" "$LOG_FILE"
grep "selected_agents=" "$LOG_FILE" | sort | uniq -c
```

## 6) Quick backend profile (5 scenes, no frame/video output)

This mode disables frame JPGs, MP4 writing, and per-frame pixel logs via `--profile-no-io`.

Run from repo root (`/root/RAP_Renderer_for_RL`):

```bash
source .venv/bin/activate

COMMON_ARGS=(
  --map-dir /jerry_slow_vol/womd_bins/validation
  --replay-all-scenes
  --max-scenes 5
  --num-maps 1
  --full-episode
  --episode-length 91
  --action-source neutral
  --control-mode control_sdc_only
  --init-mode create_all_valid
  --box-source map_replay
  --profile-no-io
  --out-dir /tmp/pd_profile_no_io
)

# NumPy renderer baseline
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --renderer-backend numpy \
  "${COMMON_ARGS[@]}"

# JAX renderer (GPU-enforced in current bridge code)
export JAX_PLATFORMS=cuda
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --renderer-backend jax \
  "${COMMON_ARGS[@]}"
```

Note: `--renderer-backend jax` currently fails fast if CUDA cannot be initialized or no visible GPU is detected.
