# Minimal RAP Bridge

Quick runbook for rendering selected WOMD scenes with:
- the Python bridge (`render_pufferdrive_to_rap_mvp.py`)
- the native PufferDrive renderer (`./visualize`)

Use `--renderer-backend jax` to run the JAX-optimized RAP renderer copy (`process_data/helpers/renderer_jax.py`).
For environment setup, use `envs/pufferdrive_rap_minimal/setup_env.sh` (it installs JAX by default via `INSTALL_JAX=1`, with a clean reinstall of `jax[cuda12]==0.4.30`).

If your venv already had mixed/old JAX packages, reset it with:

```bash
source .venv-pufferdrive-rap/bin/activate
python -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt
python -m pip install -U "jax[cuda12]==0.4.30"
```

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

## 2) Native renderer aligned with RAP setup (write to /tmp)

If needed, build once:

```bash
source .venv/bin/activate
bash scripts/build_ocean.sh visualize local
```

Then render scenes to `/tmp/pd_native_s18_s35_s63` with RAP-aligned behavior:
- `goal_behavior = 1`
- `control_mode = "control_sdc_only"`
- `init_mode = "create_all_valid"`
- `max_controlled_agents = 1`

```bash
CFG="pufferlib/config/ocean/drive.ini"
OUT="/tmp/pd_native_s18_s35_s63"
mkdir -p "$OUT"
BACKUP=$(mktemp)
cp "$CFG" "$BACKUP"
trap 'cp "$BACKUP" "$CFG"; rm -f "$BACKUP"' EXIT

sed -i 's/^goal_behavior = .*/goal_behavior = 1/' "$CFG"
sed -i 's/^control_mode = .*/control_mode = "control_sdc_only"/' "$CFG"
sed -i 's/^init_mode = .*/init_mode = "create_all_valid"/' "$CFG"
sed -i 's/^max_controlled_agents = .*/max_controlled_agents = 1/' "$CFG"

for sid in 018 035 063; do
  ASAN_OPTIONS=detect_leaks=0 xvfb-run -a -s "-screen 0 1280x720x24" \
    ./visualize \
    --map-name /jerry_slow_vol/womd_bins/validation/map_${sid}.bin \
    --output-topdown "$OUT/map_${sid}_topdown.mp4" \
    --output-agent "$OUT/map_${sid}_agent.mp4"
done
```

`./visualize` does not currently expose these controls as CLI flags, so `drive.ini`
must be set before native renders.

## 3) Profile 10 scenes (forward/step/RAP render/scene total)

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

## 4) Profile all 75 validation scenes (comprehensive, no video output)

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

## 5) Quick backend profile (5 scenes, policy + GPU)

Use this variant when you want policy inference on GPU (`--policy-device cuda:0`) while comparing renderer backends.

```bash
cd /root/RAP_Renderer_for_RL
source .venv/bin/activate

COMMON_ARGS=(
  --map-dir /jerry_slow_vol/womd_bins/validation
  --replay-all-scenes
  --max-scenes 5
  --num-maps 1
  --full-episode
  --episode-length 91
  --action-source policy
  --policy-device cuda:0
  --control-mode control_sdc_only
  --init-mode create_all_valid
  --box-source map_replay
  --profile-no-io
)

# 1) NumPy renderer (CPU renderer + GPU policy inference)
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --renderer-backend numpy \
  --out-dir /tmp/pd_profile_numpy_gpu_policy \
  "${COMMON_ARGS[@]}"

# 2) JAX renderer (GPU-enforced renderer + GPU policy inference)
export JAX_PLATFORMS=cuda
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --renderer-backend jax \
  --out-dir /tmp/pd_profile_jax_gpu_policy \
  "${COMMON_ARGS[@]}"
```
