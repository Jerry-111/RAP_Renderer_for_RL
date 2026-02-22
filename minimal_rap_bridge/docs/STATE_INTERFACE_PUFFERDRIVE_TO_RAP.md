# State Interface: PufferDrive -> RAP

Date: 2026-02-19  
Purpose: Explicitly list (1) what state is available from PufferDrive APIs/C bindings, (2) what RAP renderer consumes, and (3) current gaps.

## 1) PufferDrive State You Can Read Today

## Python API surface (`Drive`)

- Step outputs:
  - `obs, rewards, terminals, truncations, info` from `step(...)` / `step_native_policy(...)`
  - Source: `pufferlib/ocean/drive/drive.py:281`, `pufferlib/ocean/drive/drive.py:312`
- Per-agent global state:
  - `x, y, z, heading, id, length, width`
  - Source: `pufferlib/ocean/drive/drive.py:330`
- Per-agent metadata:
  - `entity_type, respawn_count`
  - Source: `pufferlib/ocean/drive/drive.py:362`
- Per-agent ground-truth trajectories:
  - `x, y, z, heading, valid, id, is_vehicle, is_track_to_predict, scenario_id`
  - Source: `pufferlib/ocean/drive/drive.py:388`
- Road-edge geometry:
  - flattened `x, y`, per-polyline `lengths`, and `scenario_id`
  - Source: `pufferlib/ocean/drive/drive.py:428`

## Internal C state (exists, not fully exported)

`Entity` and `Drive` structs contain richer signals than the bridge currently reads, e.g.:
- dynamics: `vx, vy, vz, a_long, a_lat, jerk_long, jerk_lat, steering_angle`
- event/status: `collision_state`, `metrics_array`, `stopped`, `removed`, `respawn_timestep`, `respawn_count`
- goal/lane internals and scenario metadata
- Source: `pufferlib/ocean/drive/drive.h:168`, `pufferlib/ocean/drive/drive.h:296`

## 2) What RAP Renderer Accepts/Uses

`ScenarioRenderer.observe(scenario)` currently consumes:

- `scenario["ego_heading"]`
- `scenario["traffic_lights"]` (expects elements where `feat[1]` is red/green flag and `feat[2]` is `[x,y]`)
- `scenario["map_features"]` dict:
  - if `type` contains `LANE`: uses `polygon`
  - if `type` contains `CROSSWALK` or `SPEED_BUMP`: uses `polygon`
  - if `type` contains `BOUNDARY` or `SOLID`: uses `polyline`
- `scenario["anns"]`:
  - `gt_boxes_world` (box geometry; renderer uses box dimensions/pose)
  - `gt_names`

Sources:
- `third_party/RAP/process_data/helpers/renderer.py:704`
- `third_party/RAP/process_data/helpers/renderer.py:717`
- `third_party/RAP/process_data/helpers/renderer.py:734`
- `third_party/RAP/process_data/helpers/renderer.py:755`

Camera channels and intrinsics/extrinsics come from RAP `camera_params`, filtered by `camera_channel_list` at renderer init.
- Source: `third_party/RAP/process_data/helpers/renderer.py:63`, `third_party/RAP/process_data/helpers/renderer.py:695`

## 3) What Bridge Currently Sends to RAP

Bridge scenario payload (`make_scenario`) sets:
- `ego_heading`
- `traffic_lights = []` (currently none from PufferDrive API path)
- `map_features` from road edges (+ optional parsed lane polylines from map binary)
- `anns = {gt_boxes_world, gt_names}` from global agent state

Sources:
- `minimal_rap_bridge/render_pufferdrive_to_rap.py:764`
- `minimal_rap_bridge/render_pufferdrive_to_rap.py:790`
- `minimal_rap_bridge/render_pufferdrive_to_rap.py:438`
- `minimal_rap_bridge/render_pufferdrive_to_rap.py:698`

## 4) Discrepancies / Missing Pieces (Fact-Checked)

### A) Gettable from Python today (`implemented`)

- `Drive.step(...)` / `step_native_policy(...)` already provide `obs, rewards, terminals, truncations, info`.  
  `info` can include aggregated log metrics (`offroad_rate`, `collision_rate`, `lane_alignment_rate`, etc.) via `vec_log` + `my_log` (typically emitted on `report_interval` ticks).
  - Sources: `pufferlib/ocean/drive/drive.py:281`, `pufferlib/ocean/drive/drive.py:312`, `pufferlib/ocean/env_binding.h:558`, `pufferlib/ocean/drive/binding.c:237`
- `obs` already carries normalized policy features (ego goal/speed/collision/respawn bits, relative partner features, local road-segment features).  
  This is available now, but it is feature-engineered/normalized rather than raw simulator state.
  - Sources: `pufferlib/ocean/drive/drive.h:80`, `pufferlib/ocean/drive/drive.h:1750`
- Direct global getters already implemented: current pose/size/id, agent meta, ground-truth trajectories, road-edge polylines.
  - Sources: `pufferlib/ocean/drive/drive.py:330`, `pufferlib/ocean/drive/drive.py:362`, `pufferlib/ocean/drive/drive.py:388`, `pufferlib/ocean/drive/drive.py:428`

### B) Needs new C export/binding for direct runtime access (`not implemented`)

- Per-agent runtime dynamics/state are present in C but not exported through current Python getters:
  - `vx, vy, vz, a_long, a_lat, jerk_long, jerk_lat, steering_angle`
  - `collision_state`, `metrics_array`, `stopped`, `removed`, `respawn_timestep`
  - `goal_position_*`, `current_lane_idx`, `valid` (current-step validity)
  - Sources: `pufferlib/ocean/drive/drive.h:168`, `pufferlib/ocean/drive/drive.h:296`
- Per-agent `height` is loaded in C (`Entity.height`) but not available in current global Python state getter, so bridge currently uses an assumed box height.
  - Sources: `pufferlib/ocean/drive/drive.h:181`, `pufferlib/ocean/drive/drive.h:1676`, `minimal_rap_bridge/render_pufferdrive_to_rap.py:169`
- Current global state getter does not expose scenario identity per row (`scenario_id`), which makes multi-env mixing harder to disambiguate unless you track env slices externally.
  - Sources: `pufferlib/ocean/drive/drive.h:1676`, `pufferlib/ocean/drive/drive.py:148`

### C) No new C required, but only via Python-side parsing/reconstruction (`partially implemented`)

- Map semantics beyond road edges are available in map binaries (types include `ROAD_LANE`, `ROAD_LINE`, `STOP_SIGN`, `CROSSWALK`, `SPEED_BUMP`, `DRIVEWAY`) but are not returned by existing Python getters.
  - Sources: `pufferlib/ocean/drive/drive.h:15`, `pufferlib/ocean/drive/drive.h:20`, `pufferlib/ocean/drive/drive.h:409`
- Bridge already does this partially by parsing lane/line polylines directly from `.bin`; the same pattern can be extended for other map types.
  - Source: `minimal_rap_bridge/render_pufferdrive_to_rap.py:607`
- Some runtime signals can be approximated from consecutive global states (for example velocity from finite differences), but that is derived, not native exported state.

### D) Not gettable from current Drive state path (`not implemented`)

- Dynamic traffic-light state expected by RAP (`is_red` + position in `scenario["traffic_lights"]`) is not exposed by current Drive Python API and is not represented as a dedicated map entity type in current Drive constants.
  - Sources: `third_party/RAP/process_data/helpers/renderer.py:717`, `pufferlib/ocean/drive/drive.h:15`, `minimal_rap_bridge/render_pufferdrive_to_rap.py:790`

### E) RAP-side constraint (already implemented behavior)

- RAP renderer currently consumes a narrow schema (`ego_heading`, `traffic_lights`, `map_features`, `anns`); extra simulator fields will be ignored unless renderer code is extended.
  - Source: `third_party/RAP/process_data/helpers/renderer.py:706`

## 5) Bottom Line

For rendering parity, current Python exports are usable. For simulator-faithful state replay, the main hard gap is direct export of per-agent runtime dynamics/event flags and scenario-row identity from C.

Brief next step: implement one new C getter for per-agent runtime telemetry (`vx/vy/vz`, `collision_state`, `stopped/removed`, `height`) and wire it into the bridge before adding more RAP schema fields.
