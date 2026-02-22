#!/usr/bin/env python3
"""Validate minimal env for PufferDrive -> RAP bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RAP_ROOT = REPO_ROOT / "third_party" / "RAP"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RAP_ROOT) not in sys.path:
    sys.path.insert(0, str(RAP_ROOT))


def validate_imports() -> None:
    import cv2  # noqa: F401
    from pufferlib.ocean.drive.drive import Drive  # noqa: F401
    from process_data.helpers.renderer import ScenarioRenderer  # noqa: F401

    print("[ok] imports: cv2, Drive, ScenarioRenderer")


def validate_renderer_smoke(out_path: Path) -> None:
    import cv2
    from process_data.helpers.renderer import ScenarioRenderer

    renderer = ScenarioRenderer(camera_channel_list=["CAM_F0"], width=640, height=384)
    scenario = {
        "ego_heading": 0.0,
        "traffic_lights": [(1, True, [20.0, 0.0])],
        "map_features": {
            "edge_0": {
                "type": "BOUNDARY",
                "polyline": np.array([[0.0, -6.0], [20.0, -6.0], [40.0, -6.0]], dtype=np.float32),
            }
        },
        "anns": {
            "gt_boxes_world": np.array([[20.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0]], dtype=np.float32),
            "gt_names": np.array(["vehicle"], dtype=object),
        },
    }
    out = renderer.observe(scenario)
    image = out["CAM_F0"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image[:, :, ::-1])
    print(f"[ok] renderer smoke image: {out_path} nonzero={int((image > 0).sum())}")


def validate_drive_smoke(map_dir: str) -> None:
    from pufferlib.ocean.drive.drive import Drive

    map_path = Path(map_dir) / "map_000.bin"
    if not map_path.exists():
        raise FileNotFoundError(
            f"Expected map file not found: {map_path}. "
            "Pass --map-dir pointing to a directory containing map_000.bin."
        )

    env = Drive(
        num_agents=16,
        num_maps=1,
        map_dir=map_dir,
        episode_length=32,
        resample_frequency=0,
        report_interval=64,
        control_mode="control_agents",
        init_mode="create_all_valid",
    )
    try:
        obs, info = env.reset(seed=0)
        if not isinstance(obs, np.ndarray):
            raise RuntimeError("Drive reset did not return numpy observations.")
        if obs.shape != env.observations.shape:
            raise RuntimeError(
                f"Unexpected reset obs shape: {obs.shape} vs expected {env.observations.shape}"
            )

        state = env.get_global_agent_state()
        n = int(((state["length"] > 0) & (state["width"] > 0)).sum())
        if n <= 0:
            raise RuntimeError("Drive global state contains zero valid agents.")

        road_edges = env.get_road_edge_polylines()
        n_edges = int(len(road_edges["lengths"]))
        if n_edges <= 0:
            raise RuntimeError("Drive road-edge extraction returned zero polylines.")

        # Step a few frames to verify functional simulation path.
        step_count = 3
        for _ in range(step_count):
            actions = np.zeros_like(env.actions)
            obs, rewards, terminals, truncations, step_info = env.step(actions)
            if obs.shape != env.observations.shape:
                raise RuntimeError("Unexpected observation shape after step.")
            if rewards.shape != env.rewards.shape:
                raise RuntimeError("Unexpected rewards shape after step.")
            if terminals.shape != env.terminals.shape:
                raise RuntimeError("Unexpected terminals shape after step.")
            if truncations.shape != env.truncations.shape:
                raise RuntimeError("Unexpected truncations shape after step.")
            if not np.isfinite(rewards).all():
                raise RuntimeError("Non-finite reward values found.")

        print(
            "[ok] drive smoke: "
            f"valid_agents={n}, road_edge_polylines={n_edges}, stepped_frames={step_count}"
        )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate minimal PufferDrive+RAP env")
    parser.add_argument("--check-drive", action="store_true", help="Also instantiate Drive and read state")
    parser.add_argument("--map-dir", default="resources/drive/binaries")
    parser.add_argument("--out-path", type=Path, default=Path("/tmp/pufferdrive_rap_env_validate.jpg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_imports()
    validate_renderer_smoke(args.out_path)
    if args.check_drive:
        validate_drive_smoke(args.map_dir)
    print("[done] environment validation complete")


if __name__ == "__main__":
    main()
