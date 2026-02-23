#!/usr/bin/env python3
"""Fresh minimal PufferDrive -> RAP renderer bridge (renderer-only MVP)."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RAP_ROOT = REPO_ROOT / "third_party" / "RAP"

# Resolve local imports without editable installs.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RAP_ROOT) not in sys.path:
    sys.path.insert(0, str(RAP_ROOT))

from pufferlib.ocean.drive.drive import Drive  # noqa: E402
from process_data.helpers.renderer import ScenarioRenderer, camera_params  # noqa: E402


@dataclass
class BridgeConfig:
    out_dir: Path
    frames: int
    seed: int
    map_dir: str
    num_maps: int
    num_agents: int
    episode_length: int
    cameras: List[str]
    ego_agent_index: int
    include_ego_box: bool
    control_mode: str
    init_mode: str
    max_controlled_agents: int
    single_env_mode: str
    write_video: bool
    save_frames: bool
    video_fps: float
    video_codec: str
    replay_all_scenes: bool
    scene_ids: List[int] | None
    max_scenes: int | None
    full_episode: bool


def parse_scene_ids(scene_ids: str | None) -> List[int] | None:
    if scene_ids is None:
        return None
    ids = []
    for token in scene_ids.split(","):
        tok = token.strip()
        if not tok:
            continue
        ids.append(int(tok))
    if not ids:
        raise ValueError("--scene-ids was provided but no valid ids were parsed")
    return sorted(set(ids))


def parse_args() -> BridgeConfig:
    parser = argparse.ArgumentParser(description="Minimal renderer-only PufferDrive -> RAP bridge")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/pufferdrive_rap_mvp"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--map-dir", type=str, default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=64)
    parser.add_argument("--episode-length", type=int, default=120)
    parser.add_argument("--full-episode", action="store_true", help="Render full episode_length frames")
    parser.add_argument(
        "--cameras",
        type=str,
        default="CAM_F0,CAM_L0,CAM_R0",
        help="Comma-separated camera ids from RAP camera_params",
    )
    parser.add_argument("--ego-agent-index", type=int, default=0)
    parser.add_argument("--include-ego-box", action="store_true")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="control_wosac",
        choices=["control_vehicles", "control_agents", "control_wosac", "control_sdc_only"],
        help="Use control_wosac by default for denser single-scene actor sets",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="create_all_valid",
        choices=["create_all_valid", "create_only_controlled"],
    )
    parser.add_argument("--max-controlled-agents", type=int, default=128)
    parser.add_argument(
        "--single-env-mode",
        type=str,
        default="auto_max",
        choices=["auto_max", "strict"],
        help="auto_max picks the highest num_agents <= request that still yields num_envs==1",
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-camera MP4 replay videos",
    )
    parser.add_argument(
        "--save-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-frame JPGs in addition to videos",
    )
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--video-codec", type=str, default="mp4v", help="4-char codec (e.g., mp4v, avc1)")
    parser.add_argument("--replay-all-scenes", action="store_true", help="Replay all map_*.bin scenes from --map-dir")
    parser.add_argument("--scene-ids", type=str, default=None, help="Comma-separated map ids, e.g. 18,63")
    parser.add_argument("--max-scenes", type=int, default=None, help="Optional cap after filtering scenes")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise ValueError("At least one camera must be provided via --cameras")

    invalid = [c for c in cameras if c not in camera_params]
    if invalid:
        valid = ",".join(sorted(camera_params.keys()))
        raise ValueError(f"Invalid camera ids: {','.join(invalid)}. Valid camera ids are: {valid}")

    frames = args.episode_length if args.full_episode else args.frames
    scene_ids = parse_scene_ids(args.scene_ids)

    if frames <= 0:
        raise ValueError("--frames must be > 0")
    if args.episode_length < frames:
        raise ValueError("--episode-length must be >= frames (or --full-episode)")
    if args.num_maps <= 0:
        raise ValueError("--num-maps must be > 0")
    if args.num_agents <= 0:
        raise ValueError("--num-agents must be > 0")
    if args.max_controlled_agents <= 0:
        raise ValueError("--max-controlled-agents must be > 0")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be > 0")
    if len(args.video_codec) != 4:
        raise ValueError("--video-codec must be exactly 4 characters")
    if (not args.write_video) and (not args.save_frames):
        raise ValueError("At least one output must be enabled: --write-video or --save-frames")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError("--max-scenes must be > 0 when provided")
    if args.replay_all_scenes and args.num_maps != 1:
        raise ValueError("--replay-all-scenes currently requires --num-maps 1")

    return BridgeConfig(
        out_dir=args.out_dir,
        frames=frames,
        seed=args.seed,
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        cameras=cameras,
        ego_agent_index=args.ego_agent_index,
        include_ego_box=args.include_ego_box,
        control_mode=args.control_mode,
        init_mode=args.init_mode,
        max_controlled_agents=args.max_controlled_agents,
        single_env_mode=args.single_env_mode,
        write_video=args.write_video,
        save_frames=args.save_frames,
        video_fps=args.video_fps,
        video_codec=args.video_codec,
        replay_all_scenes=args.replay_all_scenes,
        scene_ids=scene_ids,
        max_scenes=args.max_scenes,
        full_episode=args.full_episode,
    )


def valid_agent_indices(state: Dict[str, np.ndarray]) -> np.ndarray:
    finite = np.isfinite(state["x"]) & np.isfinite(state["y"]) & np.isfinite(state["heading"])
    size_ok = (state["length"] > 0.0) & (state["width"] > 0.0)
    return np.flatnonzero(finite & size_ok)


def choose_ego_index(state: Dict[str, np.ndarray], preferred_idx: int) -> int:
    valid = valid_agent_indices(state)
    if valid.size == 0:
        raise RuntimeError("No valid agents available for ego selection")
    if preferred_idx in set(valid.tolist()):
        return int(preferred_idx)
    return int(valid[0])


def resolve_ego_index_by_id(state: Dict[str, np.ndarray], ego_id: int, fallback_idx: int) -> int:
    valid = set(valid_agent_indices(state).tolist())
    if not valid:
        raise RuntimeError("No valid agents available while resolving ego index")

    candidates = [int(i) for i in np.flatnonzero(state["id"] == ego_id).tolist() if int(i) in valid]
    if candidates:
        if fallback_idx in candidates:
            return int(fallback_idx)
        return int(candidates[0])

    if fallback_idx in valid:
        return int(fallback_idx)
    return int(next(iter(valid)))


def extract_boundary_polylines_abs(road_edges: Dict[str, np.ndarray]) -> List[np.ndarray]:
    polylines: List[np.ndarray] = []
    pt_idx = 0
    lengths = road_edges["lengths"]
    xs = road_edges["x"]
    ys = road_edges["y"]

    for seg_len in lengths:
        seg_len = int(seg_len)
        x_seg = xs[pt_idx : pt_idx + seg_len]
        y_seg = ys[pt_idx : pt_idx + seg_len]
        pt_idx += seg_len
        if seg_len < 2:
            continue

        poly = np.stack([x_seg, y_seg], axis=1).astype(np.float32)
        if np.isfinite(poly).all():
            polylines.append(poly)

    return polylines


def build_map_features(polylines_abs: Sequence[np.ndarray], ego_x: float, ego_y: float) -> Dict[str, Dict[str, np.ndarray]]:
    features: Dict[str, Dict[str, np.ndarray]] = {}
    ego_xy = np.array([ego_x, ego_y], dtype=np.float32)

    for idx, poly in enumerate(polylines_abs):
        rel = (poly - ego_xy).astype(np.float32)
        if rel.shape[0] < 2 or not np.isfinite(rel).all():
            continue
        features[f"boundary_{idx:05d}"] = {"type": "BOUNDARY", "polyline": rel}

    return features


def build_boxes_from_state(
    state: Dict[str, np.ndarray],
    ego_idx: int,
    include_ego_box: bool,
    default_height: float = 1.6,
) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_agent_indices(state)
    if valid.size == 0:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)

    ego_x = float(state["x"][ego_idx])
    ego_y = float(state["y"][ego_idx])
    ego_z = float(state["z"][ego_idx])
    boxes: List[List[float]] = []
    names: List[str] = []

    for i in valid:
        if (not include_ego_box) and int(i) == int(ego_idx):
            continue
        rel_x = float(state["x"][i] - ego_x)
        rel_y = float(state["y"][i] - ego_y)
        rel_z = float(state["z"][i] - ego_z)
        boxes.append(
            [
                rel_x,
                rel_y,
                rel_z,
                float(state["length"][i]),
                float(state["width"][i]),
                float(default_height),
                float(state["heading"][i]),
            ]
        )
        names.append("vehicle")

    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)
    return np.array(boxes, dtype=np.float32), np.array(names, dtype=object)


def build_scenario(
    state: Dict[str, np.ndarray],
    ego_idx: int,
    include_ego_box: bool,
    map_features: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    gt_boxes_world, gt_names = build_boxes_from_state(state, ego_idx=ego_idx, include_ego_box=include_ego_box)
    return {
        "ego_heading": float(state["heading"][ego_idx]),
        "traffic_lights": [],
        "map_features": map_features,
        "anns": {
            "gt_boxes_world": gt_boxes_world,
            "gt_names": gt_names,
        },
    }


def validate_action_contract(env: Drive, actions: np.ndarray) -> None:
    if actions.shape != env.actions.shape:
        raise ValueError(f"Action shape mismatch: expected {env.actions.shape}, got {actions.shape}")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError(f"Action dtype must be integer, got {actions.dtype}")
    if getattr(env, "dynamics_model", None) != "classic":
        raise ValueError("MVP bridge currently supports only classic dynamics")

    min_a = int(np.min(actions))
    max_a = int(np.max(actions))
    if min_a < 0 or max_a > 90:
        raise ValueError(f"Action range mismatch for classic dynamics: expected [0, 90], got [{min_a}, {max_a}]")


def make_neutral_actions(env: Drive) -> np.ndarray:
    # classic discrete neutral action index: accel=0, steer=0 -> 3*13 + 6 = 45
    actions = np.full(env.actions.shape, 45, dtype=env.actions.dtype)
    validate_action_contract(env, actions)
    return actions


def save_frame_images(out_dir: Path, frame_idx: int, rendered: Dict[str, np.ndarray]) -> None:
    for cam_id, image in rendered.items():
        path = out_dir / f"frame_{frame_idx:04d}_{cam_id}.jpg"
        ok = cv2.imwrite(str(path), image[:, :, ::-1])
        if not ok:
            raise RuntimeError(f"Failed to write image: {path}")


def log_nonzero_pixels(frame_idx: int, rendered: Dict[str, np.ndarray]) -> None:
    counts = {cam: int((img > 0).sum()) for cam, img in rendered.items()}
    summary = ", ".join(f"{cam}={count}" for cam, count in counts.items())
    print(f"[frame {frame_idx:04d}] nonzero_pixels: {summary}")


def open_video_writers(
    out_dir: Path,
    cameras: Sequence[str],
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> Dict[str, cv2.VideoWriter]:
    writers: Dict[str, cv2.VideoWriter] = {}
    fourcc = cv2.VideoWriter_fourcc(*codec)
    for cam in cameras:
        video_path = out_dir / f"replay_{cam}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path} (codec={codec})")
        writers[cam] = writer
    return writers


def write_video_frames(writers: Dict[str, cv2.VideoWriter], rendered: Dict[str, np.ndarray]) -> None:
    for cam, writer in writers.items():
        if cam not in rendered:
            raise RuntimeError(f"Rendered output missing camera: {cam}")
        writer.write(rendered[cam][:, :, ::-1])


def close_video_writers(writers: Dict[str, cv2.VideoWriter]) -> None:
    for writer in writers.values():
        writer.release()


def _create_env(cfg: BridgeConfig, num_agents: int) -> Drive:
    env = Drive(
        num_agents=num_agents,
        num_maps=cfg.num_maps,
        map_dir=cfg.map_dir,
        episode_length=cfg.episode_length,
        control_mode=cfg.control_mode,
        init_mode=cfg.init_mode,
        max_controlled_agents=cfg.max_controlled_agents,
        resample_frequency=0,
        report_interval=max(cfg.frames, 1),
    )
    env.reset(seed=cfg.seed)
    return env


def create_single_scene_env(cfg: BridgeConfig) -> tuple[Drive, int]:
    requested = int(cfg.num_agents)

    if cfg.single_env_mode == "strict":
        env = _create_env(cfg, requested)
        if int(env.num_envs) != 1:
            env.close()
            raise RuntimeError(
                f"strict mode requires num_envs==1 but got num_envs={env.num_envs} for num_agents={requested}. "
                "Use --single-env-mode auto_max to auto-select the largest single-scene agent count."
            )
        return env, requested

    # auto_max mode: choose the largest num_agents <= requested with num_envs == 1.
    low, high = 1, requested
    best = None
    while low <= high:
        mid = (low + high) // 2
        probe = _create_env(cfg, mid)
        num_envs = int(probe.num_envs)
        probe.close()
        if num_envs == 1:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    if best is None:
        raise RuntimeError(
            "Could not find any num_agents value that yields num_envs==1 for this map/config. "
            "Try another control_mode/init_mode."
        )

    env = _create_env(cfg, best)
    if int(env.num_envs) != 1:
        env.close()
        raise RuntimeError(f"single-scene allocator mismatch: expected num_envs=1, got {env.num_envs}")
    return env, best


def map_id_from_path(path: Path) -> int | None:
    name = path.stem
    if not name.startswith("map_"):
        return None
    try:
        return int(name.split("_")[1])
    except Exception:
        return None


def resolve_scenes(cfg: BridgeConfig) -> List[Tuple[str, Path]]:
    root = Path(cfg.map_dir)
    if not root.exists():
        raise FileNotFoundError(f"map-dir does not exist: {root}")

    if cfg.replay_all_scenes:
        files = sorted(root.glob("map_*.bin"))
        if cfg.scene_ids is not None:
            wanted = set(cfg.scene_ids)
            files = [p for p in files if map_id_from_path(p) in wanted]
        if cfg.max_scenes is not None:
            files = files[: cfg.max_scenes]
        if not files:
            raise RuntimeError("No scenes matched --replay-all-scenes filters")
        return [(p.stem, p) for p in files]

    if cfg.scene_ids is not None:
        if len(cfg.scene_ids) != 1:
            raise ValueError("Without --replay-all-scenes, --scene-ids must contain exactly one id")
        p = root / f"map_{cfg.scene_ids[0]:03d}.bin"
        if not p.exists():
            raise FileNotFoundError(f"Requested scene not found: {p}")
        return [(p.stem, p)]

    candidate = root / "map_000.bin"
    if candidate.exists():
        return [(candidate.stem, candidate)]

    files = sorted(root.glob("map_*.bin"))
    if files:
        return [(files[0].stem, files[0])]
    raise RuntimeError(f"No map_*.bin files found in: {root}")


def stage_single_scene_map(scene_source: Path) -> Path:
    stage_dir = Path(tempfile.mkdtemp(prefix="pd_scene_stage_"))
    target = stage_dir / "map_000.bin"
    try:
        target.symlink_to(scene_source.resolve())
    except Exception:
        shutil.copy2(scene_source, target)
    return stage_dir


def run_single_scene(cfg: BridgeConfig, scene_name: str, scene_source: Path) -> Dict[str, int | str]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    env, selected_num_agents = create_single_scene_env(cfg)
    video_writers: Dict[str, cv2.VideoWriter] = {}
    try:
        road_edges = env.get_road_edge_polylines()
        polylines_abs = extract_boundary_polylines_abs(road_edges)
        if not polylines_abs:
            raise RuntimeError("No boundary polylines found from env.get_road_edge_polylines()")

        render_width, render_height = 1920, 1120
        renderer = ScenarioRenderer(camera_channel_list=cfg.cameras, width=render_width, height=render_height)
        neutral_actions = make_neutral_actions(env)

        if cfg.write_video:
            video_writers = open_video_writers(
                out_dir=cfg.out_dir,
                cameras=cfg.cameras,
                width=render_width,
                height=render_height,
                fps=cfg.video_fps,
                codec=cfg.video_codec,
            )

        first_state = env.get_global_agent_state()
        ego_idx = choose_ego_index(first_state, cfg.ego_agent_index)
        ego_id = int(first_state["id"][ego_idx])

        print(f"\n=== Scene {scene_name} ({scene_source}) ===")
        print(f"Output directory: {cfg.out_dir}")
        print(f"Frames: {cfg.frames}, Seed: {cfg.seed}")
        print(
            f"single_env_mode={cfg.single_env_mode}, num_agents(requested={cfg.num_agents}, selected={selected_num_agents}), "
            f"num_envs={env.num_envs}, num_agents_actual={env.num_agents}"
        )
        print(
            f"control_mode={cfg.control_mode}, init_mode={cfg.init_mode}, max_controlled_agents={cfg.max_controlled_agents}, "
            f"ego_idx={ego_idx}, ego_id={ego_id}"
        )
        print(
            f"Cameras: {cfg.cameras} | write_video={cfg.write_video} codec={cfg.video_codec} fps={cfg.video_fps} | "
            f"save_frames={cfg.save_frames}"
        )

        for frame_idx in range(cfg.frames):
            state = env.get_global_agent_state()
            ego_idx = resolve_ego_index_by_id(state, ego_id=ego_id, fallback_idx=ego_idx)
            ego_x = float(state["x"][ego_idx])
            ego_y = float(state["y"][ego_idx])
            map_features = build_map_features(polylines_abs, ego_x=ego_x, ego_y=ego_y)
            if not map_features:
                raise RuntimeError(f"No map features available at frame={frame_idx}")

            scenario = build_scenario(
                state=state,
                ego_idx=ego_idx,
                include_ego_box=cfg.include_ego_box,
                map_features=map_features,
            )
            rendered = renderer.observe(scenario)
            if cfg.save_frames:
                save_frame_images(cfg.out_dir, frame_idx, rendered)
            if cfg.write_video:
                write_video_frames(video_writers, rendered)
            log_nonzero_pixels(frame_idx, rendered)

            if frame_idx < (cfg.frames - 1):
                validate_action_contract(env, neutral_actions)
                env.step(neutral_actions)

        return {
            "scene_name": scene_name,
            "selected_agents": int(selected_num_agents),
            "num_envs": int(env.num_envs),
            "frames": int(cfg.frames),
        }
    finally:
        close_video_writers(video_writers)
        env.close()


def run_bridge(cfg: BridgeConfig) -> None:
    scenes = resolve_scenes(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scenes to replay: {len(scenes)}")

    summaries: List[Dict[str, int | str]] = []
    for scene_name, scene_source in scenes:
        scene_out_dir = cfg.out_dir if len(scenes) == 1 else (cfg.out_dir / scene_name)
        stage_dir = stage_single_scene_map(scene_source)
        try:
            scene_cfg = replace(cfg, map_dir=str(stage_dir), num_maps=1, out_dir=scene_out_dir)
            summaries.append(run_single_scene(scene_cfg, scene_name=scene_name, scene_source=scene_source))
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    print("\n=== Replay Summary ===")
    for s in summaries:
        print(
            f"scene={s['scene_name']} frames={s['frames']} selected_agents={s['selected_agents']} "
            f"num_envs={s['num_envs']}"
        )


def main() -> None:
    cfg = parse_args()
    run_bridge(cfg)


if __name__ == "__main__":
    main()
