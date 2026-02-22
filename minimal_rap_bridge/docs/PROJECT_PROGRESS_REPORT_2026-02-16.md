# PufferDrive → RAP Closed-Loop Parity Progress Report

**Prepared by:** Bridge Integration Workstream  
**Date:** February 16, 2026  
**Project area:** `minimal_rap_bridge` integration with native PufferDrive policy stepping

## 0) Scope of this report (full bridge project to date)

This report covers the bridge effort end-to-end, not only the most recent closed-loop work:
- initial PufferDrive state → RAP scenario conversion
- camera/intrinsics/ego alignment debugging
- replay-source and identity stability fixes
- native closed-loop policy wiring
- action/state parity instrumentation and validation
- current roadmap for robustness and scale

## 1) Objective

Build a RAP rendering pipeline that uses the **same closed-loop policy decisions** as native PufferDrive, so simulation behavior matches native dynamics while preserving RAP camera control.

Target equivalence:
- Same policy actions per step
- Same simulator state evolution per step
- Expected camera/render differences (native chase camera vs RAP sensor cameras)

## 2) What has been implemented (project-wide)

### Foundation bridge capability
- Implemented base bridge to render RAP frames from PufferDrive state.
- Added state-vs-RAP comparator workflow and native-vs-RAP video comparison tooling.
- Added baseline scripts for cleaner iteration:
  - `minimal_rap_bridge/render_pufferdrive_to_rap_baseline.py`
  - `minimal_rap_bridge/compare_state_vs_rap_baseline.py`

### Stabilization and alignment work (pre-closed-loop)
- Corrected non-neutral stepping behavior (legacy “zero action” was not neutral in discrete classic dynamics).
- Improved ego/heading stability under duplicate IDs by moving to slot-stable matching logic.
- Added ego lock/anchor behavior for replay cases where ground-truth ego track becomes invalid.
- Added map/agent rendering hygiene improvements (e.g., duplicate overlap suppression).

### Native closed-loop parity work (current phase)

### Native closed-loop policy path in Python API
- Added native policy lifecycle/step APIs in `pufferlib/ocean/drive/drive.py`:
  - `init_native_policy(...)`
  - `step_native_policy()`
  - `close_native_policy()`
- Added C bindings in `pufferlib/ocean/drive/binding.c`:
  - `vec_policy_init(...)`
  - `vec_policy_step(...)`
  - `vec_policy_close(...)`

### RAP bridge wiring for policy stepping
- Added control switch in bridge scripts:
  - `--control-source native_policy`
  - `--policy-path resources/drive/puffer_drive_weights.bin`
- Added action logging:
  - reference loop logger: `minimal_rap_bridge/log_native_policy_actions_reference.py`
  - RAP loop action logging in render pipeline
  - comparator: `minimal_rap_bridge/compare_action_logs.py`

### Reproducible one-command pipeline
- `minimal_rap_bridge/run_native_policy_pipeline.sh` now runs:
  1. state-vs-RAP panels
  2. pure RAP render
  3. RAP stitched strip video (L/F/R)
  4. native-vs-RAP visual matching
  5. action parity report (default enabled)

### Strong parity guardrails
- Defaulted pipeline to single-env-safe `--num-agents 21`
- Added runtime guard to fail when `num_envs > 1` (prevents merged multi-simulation artifacts)

### Additional deterministic checker
- Added `minimal_rap_bridge/check_policy_action_replay_consistency.py`
  - Pass A: run `step_native_policy()`, record actions + post-step state checksums
  - Pass B: replay exact actions via `env.step(actions)`, compare post-step state checksums

## 3) Evidence collected (current)

## Action parity result
From `/tmp/pd_rap_pipeline_native_check/action_log_compare.txt`:
- `row_count_match=True`
- `mismatch_rows=0`
- `first_mismatch=none`

Interpretation: per-step action decisions are identical between reference native policy loop and RAP pipeline loop for compared horizon.

## State replay parity result
From `/tmp/pd_action_replay_consistency_report.txt`:
- `frames_checked=79`
- `num_mismatch_steps=0`
- `all_steps_match=True`
- `first_mismatch=none`

Interpretation: replaying exact native-policy actions reproduces the same exposed global simulator state (`id/x/y/heading/length/width`) at each step.

## 4) Major issues solved (log)

1. **Non-neutral “zero action” induced artificial drift/spin**
- Symptom: unrealistic heading drift and trajectory instability in open-loop stepping.
- Root cause: discrete action index `0` mapped to aggressive control, not neutral.
- Fix: neutral action inference applied for stepping helper.
- Outcome: rollout behavior became physically sensible for baseline debugging.

2. **Duplicate agent IDs broke heading/ego matching**
- Symptom: apparent self-rotation artifacts and occasional ego reassignment errors.
- Root cause: ID-based mapping assumed uniqueness that does not hold in exports.
- Fix: slot-stable heading lookup and nearest-by-last-position ego resolution among same-ID candidates.
- Outcome: stable ego tracking and reduced false rotation artifacts.

3. **Ground-truth replay ego discontinuity (frame jump)**
- Symptom: abrupt ego side/camera jump when selected ego slot became invalid mid-horizon.
- Root cause: per-frame re-resolution switched to a different same-ID agent once original track ended.
- Fix: lock ego slot for GT replay and anchor camera to last valid pose when ego becomes invalid.
- Outcome: removed abrupt camera/ego identity jumps in that failure mode.

4. **Segmentation fault in native policy step (multi-env vectorized setup)**
- Symptom: crash during native policy stepping in larger vectorized configurations.
- Root cause: shared weight buffer index not reset per env during `DriveNet` initialization.
- Fix: reset `weights->idx = 0` before each `init_drivenet(...)`.
- Outcome: stable native policy stepping in tested configs.

5. **Divergent/chaotic traffic caused by merged simulations**
- Symptom: vehicle count inflated over time and interactions looked unrealistic.
- Root cause: using `num_agents` values that created `num_envs > 1`; outputs were effectively combined across environments.
- Fix: enforce single-env parity configuration (`num_agents=21` for this map/config) and hard-fail guard in pipeline when `num_envs > 1`.
- Outcome: stable per-frame actor counts and consistent single-simulation behavior.

6. **Ego mismatch instability across runs**
- Symptom: occasional ego identity mismatch vs expected native-selected ego.
- Fix: added parity control knobs (`--ego-select-mode`, `--ego-agent-index`, `--control-mode`, `--init-mode`) and lockable ego strategy.
- Outcome: reproducible ego selection in controlled runs.

## 5) What this does and does not guarantee

Guaranteed by current checks:
- Policy action parity (full action tensor checksum per step)
- Simulator state parity for exposed global state fields under exact action replay

Not guaranteed by current checks:
- Pixel-level parity with native `./visualize` output
- Parity of internal simulator fields not included in `get_global_agent_state`
- Parity when run configurations differ (map selection, seed, horizon, control/init modes, frame alignment)

## 6) Roadmap (next)

### Phase A — Measurement hardening (immediate)
1. Add optional parity checks for rewards/terminals/truncations.
2. Add first-divergence detector that dumps the earliest mismatch frame and full diagnostics.
3. Standardize a single “parity run config” file for reproducibility.

**Acceptance criteria**
- Zero mismatches on action + state + reward/terminal checks for baseline seed/map over 80 frames.

### Phase B — Visual consistency audit
1. Regenerate native headless videos with matched config/horizon.
2. Align frame indexing/horizon exactly between native and RAP outputs.
3. Produce per-frame discrepancy panels for flagged interactions (e.g., looping opposite-lane vehicle, contact behavior).

**Acceptance criteria**
- Any remaining discrepancy is attributable to documented camera/view/model differences, not action/state divergence.

### Phase C — Multi-map expansion
1. Expand evaluation from single-map to multi-map (`num_maps > 1`) with fixed seed sets.
2. Add per-map parity summaries (action parity, state parity, first-divergence frame).
3. Stratify outcomes by scenario type (dense traffic, merges, intersections, sparse highway).

**Acceptance criteria**
- Stable parity metrics across a representative map/seed matrix.

### Phase D — C-core telemetry wiring for completeness
1. Extend C/Python bindings to expose additional simulator signals beyond current global pose/size fields.
2. Add optional logging of per-step dynamics and event fields (e.g., speed, acceleration, collisions, offroad/goal events, respawn counters, control flags as available in Drive internals).
3. Add parity checks that include these newly exposed fields.

**Acceptance criteria**
- New telemetry fields are surfaced in a stable schema and included in automated parity reports.
- First-difference diagnostics report action + state + telemetry mismatch context at divergence step.

## 7) Immediate next commands (reproducible)

```bash
# Full single-env pipeline (includes action check + strip video)
bash minimal_rap_bridge/run_native_policy_pipeline.sh \
  --native-video /tmp/pd_native_vs_rap/native_agent.mp4 \
  --out-root /tmp/pd_rap_pipeline_native_singleenv \
  --frames 27 --episode-length 120 --seed 1 \
  --map-dir resources/drive/binaries --num-maps 1 --num-agents 21
```

```bash
# Deterministic action->state replay check
./.venv-pufferdrive-rap/bin/python minimal_rap_bridge/check_policy_action_replay_consistency.py \
  --frames 80 --episode-length 120 --seed 1 \
  --map-dir resources/drive/binaries --num-maps 1 --num-agents 21 \
  --control-mode control_vehicles --init-mode create_all_valid \
  --ego-select-mode native_visualize \
  --policy-path resources/drive/puffer_drive_weights.bin \
  --out-report /tmp/pd_action_replay_consistency_report.txt \
  --out-csv /tmp/pd_action_replay_consistency.csv
```

## 8) Current status summary

The core closed-loop parity objective is substantially de-risked at the **action** and **exposed state** levels in single-env mode.  
The remaining work is primarily in **visual-forensic validation** and **robustness scaling** across more scenarios.
