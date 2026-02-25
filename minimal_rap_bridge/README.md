# Minimal RAP Bridge

Quick runbook for rendering selected WOMD scenes with:
- the Python bridge (`render_pufferdrive_to_rap_mvp.py`)
- the native PufferDrive renderer (`./visualize`)

## 1) Bridge render with PyTorch policy (videos only, no JPG frames)

Run from repo root (`/root/RAP_Renderer_for_RL`):

```bash
source .venv/bin/activate
python minimal_rap_bridge/render_pufferdrive_to_rap_mvp.py \
  --map-dir /jerry_slow_vol/womd_bins/validation \
  --replay-all-scenes \
  --scene-ids 18,35,63 \
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
