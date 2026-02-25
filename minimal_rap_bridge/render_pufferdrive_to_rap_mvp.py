#!/usr/bin/env python3
"""Fresh minimal PufferDrive -> RAP renderer bridge (renderer-only MVP)."""

from __future__ import annotations

import argparse
import importlib
import struct
import shutil
import sys
import tempfile
import warnings
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
    goal_behavior: int
    action_source: str
    policy_model_path: Path
    policy_device: str
    policy_name: str
    policy_input_size: int
    policy_hidden_size: int
    use_rnn: bool
    rnn_name: str | None
    rnn_input_size: int
    rnn_hidden_size: int
    policy_deterministic: bool
    box_source: str
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
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4096,
        help="Requested controlled-agent budget. With --single-env-mode auto_max, a large value auto-selects the max single-scene agent count.",
    )
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
        "--goal-behavior",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=respawn, 1=generate_new_goals, 2=stop",
    )
    parser.add_argument(
        "--action-source",
        type=str,
        default="neutral",
        choices=["neutral", "policy"],
        help="How to produce env actions: neutral baseline or PyTorch policy inference",
    )
    parser.add_argument(
        "--policy-model-path",
        type=Path,
        default=Path("pufferlib/resources/drive/pufferdrive_weights.pt"),
        help="Path to a PyTorch checkpoint used when --action-source policy",
    )
    parser.add_argument(
        "--policy-device",
        type=str,
        default="auto",
        help="PyTorch device for inference: auto/cpu/cuda[:id]",
    )
    parser.add_argument("--policy-name", type=str, default="Drive", help="Policy class from pufferlib.ocean.torch")
    parser.add_argument("--policy-input-size", type=int, default=64)
    parser.add_argument("--policy-hidden-size", type=int, default=256)
    parser.add_argument(
        "--use-rnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap policy with recurrent module",
    )
    parser.add_argument("--rnn-name", type=str, default="Recurrent", help="RNN wrapper class from pufferlib.ocean.torch")
    parser.add_argument("--rnn-input-size", type=int, default=256)
    parser.add_argument("--rnn-hidden-size", type=int, default=256)
    parser.add_argument(
        "--policy-deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use argmax/mean actions instead of sampling",
    )
    parser.add_argument(
        "--box-source",
        type=str,
        default="auto",
        choices=["auto", "sim", "map_replay"],
        help="Actor boxes for RAP renderer: sim=active agents only, map_replay=replay boxes from map trajectories, auto=map_replay for control_sdc_only else sim",
    )
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
    if args.action_source == "policy":
        if args.policy_input_size <= 0 or args.policy_hidden_size <= 0:
            raise ValueError("--policy-input-size and --policy-hidden-size must be > 0")
        if args.use_rnn and (args.rnn_input_size <= 0 or args.rnn_hidden_size <= 0):
            raise ValueError("--rnn-input-size and --rnn-hidden-size must be > 0 when --use-rnn")
        if not args.policy_model_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_model_path}")

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
        goal_behavior=args.goal_behavior,
        action_source=args.action_source,
        policy_model_path=args.policy_model_path,
        policy_device=args.policy_device,
        policy_name=args.policy_name,
        policy_input_size=args.policy_input_size,
        policy_hidden_size=args.policy_hidden_size,
        use_rnn=args.use_rnn,
        rnn_name=args.rnn_name if args.use_rnn else None,
        rnn_input_size=args.rnn_input_size,
        rnn_hidden_size=args.rnn_hidden_size,
        policy_deterministic=args.policy_deterministic,
        box_source=args.box_source,
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


def build_scenario_from_boxes(
    ego_heading: float,
    map_features: Dict[str, Dict[str, np.ndarray]],
    gt_boxes_world: np.ndarray,
    gt_names: np.ndarray,
) -> Dict:
    return {
        "ego_heading": float(ego_heading),
        "traffic_lights": [],
        "map_features": map_features,
        "anns": {
            "gt_boxes_world": gt_boxes_world,
            "gt_names": gt_names,
        },
    }


def _read_exact(fh, nbytes: int) -> bytes:
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"Unexpected EOF while reading {nbytes} bytes")
    return data


def load_map_replay_objects(map_bin_path: Path) -> Dict[str, object]:
    """Load actor trajectories from a WOMD binary map for renderer-side replay boxes."""
    objects: List[Dict[str, object]] = []
    with map_bin_path.open("rb") as fh:
        _scenario_id = _read_exact(fh, 16)
        sdc_track_index = struct.unpack("i", _read_exact(fh, 4))[0]
        tracks_to_predict_count = struct.unpack("i", _read_exact(fh, 4))[0]
        if tracks_to_predict_count > 0:
            _read_exact(fh, 4 * tracks_to_predict_count)

        num_objects = struct.unpack("i", _read_exact(fh, 4))[0]
        num_roads = struct.unpack("i", _read_exact(fh, 4))[0]

        for obj_idx in range(num_objects):
            _map_id = struct.unpack("i", _read_exact(fh, 4))[0]
            obj_type = struct.unpack("i", _read_exact(fh, 4))[0]
            obj_id = struct.unpack("i", _read_exact(fh, 4))[0]
            array_size = struct.unpack("i", _read_exact(fh, 4))[0]

            x = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32).copy()
            y = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32).copy()
            z = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32).copy()
            _vx = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32)
            _vy = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32)
            _vz = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32)
            heading = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.float32).copy()
            valid = np.frombuffer(_read_exact(fh, 4 * array_size), dtype=np.int32).copy()

            width, length, _height = struct.unpack("fff", _read_exact(fh, 12))
            _goal_x, _goal_y, _goal_z = struct.unpack("fff", _read_exact(fh, 12))
            _mark_as_expert = struct.unpack("i", _read_exact(fh, 4))[0]

            objects.append(
                {
                    "track_index": obj_idx,
                    "type": int(obj_type),
                    "id": int(obj_id),
                    "x": x,
                    "y": y,
                    "z": z,
                    "heading": heading,
                    "valid": valid,
                    "length": float(length),
                    "width": float(width),
                }
            )

        # Skip road records to keep parser aligned.
        for _ in range(num_roads):
            _map_id = struct.unpack("i", _read_exact(fh, 4))[0]
            _road_type = struct.unpack("i", _read_exact(fh, 4))[0]
            _road_id = struct.unpack("i", _read_exact(fh, 4))[0]
            size = struct.unpack("i", _read_exact(fh, 4))[0]
            _read_exact(fh, 4 * size * 3)  # x, y, z arrays
            _read_exact(fh, 12)  # width, length, height
            _read_exact(fh, 12)  # goal xyz
            _read_exact(fh, 4)  # mark_as_expert

    return {"sdc_track_index": int(sdc_track_index), "objects": objects}


def build_boxes_from_map_replay(
    replay_data: Dict[str, object],
    frame_idx: int,
    ego_x: float,
    ego_y: float,
    ego_z: float,
    include_ego_box: bool,
) -> tuple[np.ndarray, np.ndarray]:
    objects = replay_data["objects"]
    sdc_track_index = int(replay_data["sdc_track_index"])
    boxes: List[List[float]] = []
    names: List[str] = []

    for obj in objects:
        track_index = int(obj["track_index"])
        if (not include_ego_box) and track_index == sdc_track_index:
            continue

        arr_n = int(obj["x"].shape[0])
        if arr_n == 0:
            continue
        t = min(frame_idx, arr_n - 1)
        if int(obj["valid"][t]) != 1:
            continue

        length = float(obj["length"])
        width = float(obj["width"])
        if length <= 0.0 or width <= 0.0:
            continue

        x = float(obj["x"][t])
        y = float(obj["y"][t])
        z = float(obj["z"][t])
        heading = float(obj["heading"][t])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z) and np.isfinite(heading)):
            continue

        boxes.append([x - ego_x, y - ego_y, z - ego_z, length, width, 1.6, heading])
        obj_type = int(obj["type"])
        if obj_type == 1:
            names.append("vehicle")
        elif obj_type == 2:
            names.append("pedestrian")
        elif obj_type == 3:
            names.append("cyclist")
        else:
            names.append("agent")

    if not boxes:
        return np.zeros((0, 7), dtype=np.float32), np.array([], dtype=object)
    return np.array(boxes, dtype=np.float32), np.array(names, dtype=object)


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


def resolve_torch_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return "cuda" if torch.cuda.is_available() else "cpu"


def extract_state_dict(checkpoint: object) -> Dict[str, object]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "policy_state_dict"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    return {str(k): v for k, v in checkpoint.items()}


def strip_module_prefix(state_dict: Dict[str, object]) -> Dict[str, object]:
    cleaned: Dict[str, object] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


class ActionSource:
    def next_actions(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def on_step_result(self, terminals: np.ndarray, truncations: np.ndarray) -> None:
        del terminals, truncations

    def mode_label(self) -> str:
        raise NotImplementedError


class NeutralActionSource(ActionSource):
    def __init__(self, env: Drive):
        self.actions = make_neutral_actions(env)

    def next_actions(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return self.actions

    def mode_label(self) -> str:
        return "neutral"


class PolicyActionSource(ActionSource):
    def __init__(self, cfg: BridgeConfig, env: Drive):
        try:
            import torch
            import pufferlib.pytorch as puffer_torch
        except ImportError as exc:
            raise RuntimeError(
                "Policy action source requires PyTorch and pufferlib.pytorch. "
                "Install project deps in your active environment."
            ) from exc

        self.torch = torch
        self.sample_logits = puffer_torch.sample_logits
        self.env = env
        self.action_shape = env.actions.shape
        self.action_dtype = env.actions.dtype
        self.policy_deterministic = cfg.policy_deterministic
        self.use_rnn = cfg.use_rnn

        device_str = resolve_torch_device(cfg.policy_device)
        if device_str.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError(
                "Requested CUDA inference but torch.cuda.is_available() is False. "
                "GPU device files or permissions are not usable in this runtime."
            )
        self.device = torch.device(device_str)

        torch_module = importlib.import_module("pufferlib.ocean.torch")
        policy_cls = getattr(torch_module, cfg.policy_name)
        policy = policy_cls(env, input_size=cfg.policy_input_size, hidden_size=cfg.policy_hidden_size)

        if self.use_rnn:
            if cfg.rnn_name is None:
                raise ValueError("--rnn-name must be set when --use-rnn")
            rnn_cls = getattr(torch_module, cfg.rnn_name)
            policy = rnn_cls(env, policy, input_size=cfg.rnn_input_size, hidden_size=cfg.rnn_hidden_size)

        self.policy = policy.to(self.device)
        self.policy.eval()

        checkpoint = torch.load(cfg.policy_model_path, map_location=self.device)
        state_dict = strip_module_prefix(extract_state_dict(checkpoint))
        missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint/policy mismatch. "
                f"missing_keys={missing[:8]} unexpected_keys={unexpected[:8]} "
                "(showing up to 8 each)."
            )

        self.state: Dict[str, object] = {}
        if self.use_rnn:
            num_agents = int(env.num_agents)
            hidden_size = int(self.policy.hidden_size)
            self.state = {
                "lstm_h": torch.zeros(num_agents, hidden_size, device=self.device),
                "lstm_c": torch.zeros(num_agents, hidden_size, device=self.device),
            }

    def _deterministic_actions(self, logits: object) -> object:
        if isinstance(logits, self.torch.distributions.Normal):
            return logits.mean
        if isinstance(logits, self.torch.Tensor):
            return logits.argmax(dim=-1, keepdim=True)
        # Multi-discrete tuple/list of tensors.
        return self.torch.stack([head.argmax(dim=-1) for head in logits], dim=1)

    def next_actions(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            obs_tensor = self.torch.as_tensor(obs, device=self.device)
            logits, _ = self.policy.forward_eval(obs_tensor, self.state if self.use_rnn else None)

            if self.policy_deterministic:
                actions_tensor = self._deterministic_actions(logits)
            else:
                actions_tensor, _, _ = self.sample_logits(logits)

            actions = actions_tensor.detach().cpu().numpy().reshape(self.action_shape)
            if isinstance(logits, self.torch.distributions.Normal):
                actions = np.clip(actions, self.env.single_action_space.low, self.env.single_action_space.high)
            actions = actions.astype(self.action_dtype, copy=False)
            return actions

    def on_step_result(self, terminals: np.ndarray, truncations: np.ndarray) -> None:
        if not self.use_rnn:
            return
        done = np.asarray(terminals).astype(bool) | np.asarray(truncations).astype(bool)
        if not done.any():
            return
        done_tensor = self.torch.as_tensor(done, device=self.device)
        self.state["lstm_h"][done_tensor] = 0
        self.state["lstm_c"][done_tensor] = 0

    def mode_label(self) -> str:
        mode = "policy"
        if self.policy_deterministic:
            mode += "_deterministic"
        if self.use_rnn:
            mode += "_rnn"
        mode += f"_{self.device.type}"
        return mode


def create_action_source(cfg: BridgeConfig, env: Drive) -> ActionSource:
    if cfg.action_source == "neutral":
        return NeutralActionSource(env)
    return PolicyActionSource(cfg, env)


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
        goal_behavior=cfg.goal_behavior,
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

    # Fast path: in control_sdc_only, a single scene has one policy-controlled actor (SDC).
    # Probing large requested num_agents values is expensive and unnecessary here.
    if cfg.control_mode == "control_sdc_only":
        env = _create_env(cfg, 1)
        if int(env.num_envs) != 1:
            env.close()
            raise RuntimeError(
                f"control_sdc_only expected num_envs==1 with num_agents=1, got num_envs={env.num_envs}. "
                "Check map validity and control/init settings."
            )
        return env, 1

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
        action_source = create_action_source(cfg, env)
        obs = np.array(env.observations, copy=True)
        resolved_box_source = cfg.box_source
        if resolved_box_source == "auto":
            resolved_box_source = "map_replay" if cfg.control_mode == "control_sdc_only" else "sim"

        replay_data = None
        if resolved_box_source == "map_replay":
            replay_data = load_map_replay_objects(scene_source)

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
        print(f"action_source={action_source.mode_label()}")
        print(f"box_source={resolved_box_source}")
        print(
            f"control_mode={cfg.control_mode}, init_mode={cfg.init_mode}, max_controlled_agents={cfg.max_controlled_agents}, "
            f"ego_idx={ego_idx}, ego_id={ego_id}"
        )
        if cfg.action_source == "policy" and cfg.control_mode != "control_sdc_only":
            print(
                "[note] control_mode is not control_sdc_only, so multiple agents may be policy-controlled. "
                "Use --control-mode control_sdc_only for human-trajectory replay of non-SDC actors."
            )
        if cfg.control_mode == "control_sdc_only" and resolved_box_source == "sim":
            print(
                "[note] sim box source only includes active agents; with control_sdc_only this is typically one actor. "
                "Use --box-source map_replay to visualize other WOMD actors."
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

            if replay_data is not None:
                gt_boxes_world, gt_names = build_boxes_from_map_replay(
                    replay_data=replay_data,
                    frame_idx=frame_idx,
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_z=float(state["z"][ego_idx]),
                    include_ego_box=cfg.include_ego_box,
                )
                scenario = build_scenario_from_boxes(
                    ego_heading=float(state["heading"][ego_idx]),
                    map_features=map_features,
                    gt_boxes_world=gt_boxes_world,
                    gt_names=gt_names,
                )
            else:
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
                actions = action_source.next_actions(obs)
                validate_action_contract(env, actions)
                obs, _, terminals, truncations, _ = env.step(actions)
                action_source.on_step_result(terminals, truncations)

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
