"""Parity-first Torch+nvdiffrast migration baseline.

This file is a frozen copy of `renderer.py` that serves as the starting point
for a staged GPU raster migration. The intention is to preserve scene
semantics and replace only the CPU rasterization core in small steps.

Current status:
- Semantics remain identical to the original NumPy/OpenCV renderer.
- A staged nvdiffrast hybrid path exists for filled regions, cuboid batches,
  line-like overlays, and arrows.
- A full-scene GPU-backed MVP mode is available for the primitive families
  emitted by `ScenarioRenderer.observe`.
- Future work should lower the existing primitives into GPU-friendly forms
  without changing camera conventions, colors, or depth attenuation rules.
"""

import cv2
import numpy as np
from tqdm import tqdm
from numpy import array
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import time


@dataclass
class CameraState:
    camera_id: str
    width: int
    height: int
    depth_max: float
    K: np.ndarray
    T_w2c: np.ndarray
    cam_t: np.ndarray
    cam_R: np.ndarray
    lidar_pos: np.ndarray


@dataclass
class LineOverlay:
    semantic_tag: str
    points_world: np.ndarray
    color: np.ndarray
    radius: int = 8
    near: float = 1e-3
    depth_max: float = 80.0
    cull_center_xy: Optional[np.ndarray] = None
    cull_radius_xy: float = 0.0


@dataclass
class FilledRegion:
    semantic_tag: str
    points_world: np.ndarray
    fill_color: np.ndarray
    depth_max: float
    outline_color: Optional[np.ndarray] = None
    outline_radius: int = 8
    outline_near: float = 1e-3


@dataclass
class CuboidInstance:
    semantic_tag: str
    center_world: np.ndarray
    dims: np.ndarray
    color_rgb: np.ndarray
    thickness: int = -1
    anchor_mode: str = "bottom_center"


@dataclass
class CuboidBatch:
    semantic_tag: str
    bboxes_world: np.ndarray
    depth_max: float = 120.0


@dataclass
class ArrowOverlay:
    semantic_tag: str
    pos_world: np.ndarray
    yaw: float
    color_rgb: np.ndarray
    arrow_len: float = 3.0
    thickness: int = 6


@dataclass(frozen=True)
class RenderOpRef:
    kind: str
    index: int


@dataclass
class CameraScenePacket:
    camera: CameraState
    line_overlays: List[LineOverlay] = field(default_factory=list)
    filled_regions: List[FilledRegion] = field(default_factory=list)
    cuboid_instances: List[CuboidInstance] = field(default_factory=list)
    cuboid_batches: List[CuboidBatch] = field(default_factory=list)
    arrow_overlays: List[ArrowOverlay] = field(default_factory=list)
    render_order: List[RenderOpRef] = field(default_factory=list)

    def add_line_overlay(self, overlay: LineOverlay) -> None:
        self.line_overlays.append(overlay)
        self.render_order.append(RenderOpRef("line", len(self.line_overlays) - 1))

    def add_filled_region(self, region: FilledRegion) -> None:
        self.filled_regions.append(region)
        self.render_order.append(RenderOpRef("filled_region", len(self.filled_regions) - 1))

    def add_cuboid_instance(self, cuboid: CuboidInstance) -> None:
        self.cuboid_instances.append(cuboid)
        self.render_order.append(RenderOpRef("cuboid_instance", len(self.cuboid_instances) - 1))

    def add_cuboid_batch(self, cuboid_batch: CuboidBatch) -> None:
        self.cuboid_batches.append(cuboid_batch)
        self.render_order.append(RenderOpRef("cuboid_batch", len(self.cuboid_batches) - 1))

    def add_arrow_overlay(self, overlay: ArrowOverlay) -> None:
        self.arrow_overlays.append(overlay)
        self.render_order.append(RenderOpRef("arrow", len(self.arrow_overlays) - 1))


@dataclass
class LoweredTriangleBatch:
    semantic_tag: str
    source_kind: str
    vertices_px: np.ndarray
    vertex_depth: np.ndarray
    vertex_color: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(self.vertices_px.shape[0])


@dataclass
class LoweredCameraScene:
    camera: CameraState
    filled_triangle_batches: List[LoweredTriangleBatch] = field(default_factory=list)
    cuboid_triangle_batches: List[LoweredTriangleBatch] = field(default_factory=list)
    overlay_triangle_batches: List[LoweredTriangleBatch] = field(default_factory=list)

    @property
    def total_triangle_count(self) -> int:
        return sum(batch.triangle_count for batch in self.filled_triangle_batches) + \
            sum(batch.triangle_count for batch in self.cuboid_triangle_batches) + \
            sum(batch.triangle_count for batch in self.overlay_triangle_batches)


@dataclass
class SceneStaticCacheEntry:
    line_overlays: List[LineOverlay]
    filled_regions: List[FilledRegion]
    cuboid_instances: List[CuboidInstance]
    render_order: List[RenderOpRef]

def build_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4×4 SE(3) 齐次矩阵"""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3], T[:3, 3] = R, t
    return T


COLOR_TABLE = {
    'lanelines': np.array([98, 183, 249], np.uint8),  # 浅蓝
    'lanes': np.array([56, 103, 221], np.uint8),  # 深蓝
    'road_boundaries': np.array([200, 36, 35], np.uint8),  # 深红
    'crosswalks': np.array([206, 131, 63], np.uint8),  # 土黄
    'traffic_light_red': np.array([255, 0, 0], np.uint8),  # 红
    'traffic_light_yellow': np.array([255, 255, 0], np.uint8),  # 黄
    'traffic_light_green': np.array([0, 255, 0], np.uint8),  # 绿
    'traffic_light_unknown': np.array([255, 255, 255], np.uint8),  # 白
    'pedestrian': np.array( [255, 0, 255], np.uint8),  # 青
    'vehicle': np.array([0, 128, 255], np.uint8),  # 蓝
    'bicycle': np.array([255, 255, 0], np.uint8),  # 黑
}

def yaw_to_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float32)

def world_to_camera_T(lidar_pos, lidar_yaw,
                      cam2lidar_t, cam2lidar_R) -> np.ndarray:
    """
    构造世界到相机的齐次变换
    world ─► lidar ─► camera
    """
    T_w_lidar = build_se3(yaw_to_rot(lidar_yaw), lidar_pos)  # world→lidar
    T_cam_lidar = build_se3(cam2lidar_R, cam2lidar_t)  # cam→lidar (给定)
    T_w_cam = T_w_lidar @ T_cam_lidar  # world→cam
    return np.linalg.inv(T_w_cam)  # 取逆得 cam←world


def project_points_cam(points_cam: np.ndarray,
                       K: np.ndarray, img_hw) -> tuple[np.ndarray, np.ndarray]:
    """
    相机坐标系点集 → 像素坐标 & 可见 mask
    points_cam: (N,3)
    """
    x, y, z = points_cam.T
    eps_mask = z > 1e-3
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    H, W = img_hw
    uv = np.stack([u, v], axis=1)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = eps_mask & in_img
    return uv.astype(np.int32), valid


def _transform_world_to_camera(points_world: np.ndarray, T_w2c: np.ndarray) -> np.ndarray:
    return (T_w2c[:3, :3] @ points_world.T + T_w2c[:3, 3:4]).T


def _project_points_cam_float(points_cam: np.ndarray,
                              K: np.ndarray,
                              img_hw) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = points_cam.T
    eps_mask = z > 1e-3
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    H, W = img_hw
    uv = np.stack([u, v], axis=1).astype(np.float32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = eps_mask & in_img
    return uv, valid


def _project_homogeneous_float(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    uv_h = (K @ points_cam.T).T
    return (uv_h[:, :2] / uv_h[:, 2:3]).astype(np.float32)


def _depth_attenuated_color(base_color: np.ndarray,
                            depth_mean: float,
                            depth_max: float) -> np.ndarray:
    alpha = np.clip((depth_max - depth_mean) / depth_max, 0.0, 1.0)
    return (alpha * np.asarray(base_color, dtype=np.float32)).astype(np.uint8)


def _make_triangle_batch(semantic_tag: str,
                         source_kind: str,
                         vertices_px: List[np.ndarray],
                         vertex_depth: List[np.ndarray],
                         color) -> Optional[LoweredTriangleBatch]:
    if not vertices_px:
        return None
    verts = np.asarray(vertices_px, dtype=np.float32)
    depths = np.asarray(vertex_depth, dtype=np.float32)
    if isinstance(color, list):
        tri_colors = np.asarray(color, dtype=np.uint8)
    else:
        tri_colors = np.repeat(np.asarray(color, dtype=np.uint8)[None, :], verts.shape[0], axis=0)
    colors = np.repeat(tri_colors[:, None, :], 3, axis=1)
    return LoweredTriangleBatch(
        semantic_tag=semantic_tag,
        source_kind=source_kind,
        vertices_px=verts,
        vertex_depth=depths,
        vertex_color=colors,
    )


def _quad_to_triangles(quad: np.ndarray, depth: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
    return (
        [quad[[0, 1, 2]], quad[[0, 2, 3]]],
        [depth[[0, 1, 2]], depth[[0, 2, 3]]],
    )


camera_params = {'CAM_F0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.00785972, -0.02271912, 0.99971099],
                                                            [-0.99994262, 0.00745516, -0.00769211],
                                                            [-0.00727825, -0.99971409, -0.02277642]]),
                            'sensor2lidar_translation': array([1.65506747, -0.01168732, 1.49112208])},
                 'CAM_L0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.81776776, -0.0057693, 0.57551942],
                                                            [-0.57553938, -0.01377628, 0.81765802],
                                                            [0.0032112, -0.99988846, -0.01458626]]),
                            'sensor2lidar_translation': array([1.63069485, 0.11956747, 1.48117884])},
                 'CAM_L1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.93120104, 0.00261563, -0.36449662],
                                                            [0.36447127, -0.02048653, 0.93098926],
                                                            [-0.00503215, -0.99978671, -0.0200304]]),
                            'sensor2lidar_translation': array([1.29939471, 0.63819702, 1.36736822])},
                 'CAM_L2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.63520782, 0.01497516, -0.77219607],
                                                            [0.77232489, -0.00580669, 0.63520119],
                                                            [0.00502834, -0.99987101, -0.01525415]]),
                            'sensor2lidar_translation': array([-0.49561003, 0.54750373, 1.3472672])},
                 'CAM_R0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.82454901, 0.01165722, 0.56567043],
                                                            [-0.56528395, 0.02532491, -0.82450755],
                                                            [-0.02393702, -0.9996113, -0.01429199]]),
                            'sensor2lidar_translation': array([1.61828343, -0.15532203, 1.49007665])},
                 'CAM_R1': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.92684778, 0.02177016, -0.37480562],
                                                            [0.37497631, 0.00421964, -0.92702479],
                                                            [-0.01859993, -0.9997541, -0.01207426]]),
                            'sensor2lidar_translation': array([1.27299407, -0.60973112, 1.37217911])},
                 'CAM_R2': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[-0.62253245, 0.03706878, -0.78171558],
                                                            [0.78163434, -0.02000083, -0.62341618],
                                                            [-0.03874424, -0.99911254, -0.01652307]]),
                            'sensor2lidar_translation': array([-0.48771615, -0.493167, 1.35027683])},
                 'CAM_B0': {'distortion': array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
                            'intrinsics': array([[1.545e+03, 0.000e+00, 9.600e+02],
                                                 [0.000e+00, 1.545e+03, 5.600e+02],
                                                 [0.000e+00, 0.000e+00, 1.000e+00]]),
                            'sensor2lidar_rotation': array([[0.00802542, 0.01047463, -0.99991293],
                                                            [0.99989075, -0.01249671, 0.00789433],
                                                            [-0.01241293, -0.99986705, -0.01057378]]),
                            'sensor2lidar_translation': array([-0.47463312, 0.02368552, 1.4341838])}}

COLOR_TABLE = {
    'lanelines': np.array([98, 183, 249], np.uint8),  # 浅蓝
    'lanes': np.array([56, 103, 221], np.uint8),  # 深蓝
    'road_boundaries': np.array([200, 36, 35], np.uint8),  # 深红
    'crosswalks': np.array([206, 131, 63], np.uint8),  # 土黄
    'traffic_light_red': np.array([255, 0, 0], np.uint8),  # 红
    'traffic_light_yellow': np.array([255, 255, 0], np.uint8),  # 黄
    'traffic_light_green': np.array([0, 255, 0], np.uint8),  # 绿
    'traffic_light_unknown': np.array([255, 255, 255], np.uint8),  # 白
    'pedestrian': np.array( [255, 0, 255], np.uint8),  # 青
    'vehicle': np.array([0, 128, 255], np.uint8),  # 蓝
    'bicycle': np.array([255, 255, 0], np.uint8),  # 黑
}

def save_as_video(img_list, save_path):
    # 确定视频的保存路径和帧率
    fps = 10  # 可以根据需要调整帧率

    # 获取图像尺寸
    first_frame = img_list[0]
    h, w, c = first_frame['CAM_F0'].shape
    # 拼接后宽度
    total_width = w

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (total_width, h))

    for frame_dict in img_list:
        # 获取三张图像
        #img_L = frame_dict['CAM_L0']
        img_F = frame_dict['CAM_F0']
        #img_R = frame_dict['CAM_R0']

        # 确保图像格式是uint8
        #img_L = img_L.astype(np.uint8)
        img_F = img_F.astype(np.uint8)
        #img_R = img_R.astype(np.uint8)

        # 横向拼接
        # concatenated_img = np.hstack((img_L, img_F, img_R))
        # concatenated_img = concatenated_img[:, :, ::-1]  # BGR to RGB
        # 写入视频
       # video_writer.write(concatenated_img)
        video_writer.write(img_F[:, :, ::-1])

    # 释放资源
    video_writer.release()
    print(f'视频已保存到 {save_path}')

def draw_polyline_depth(canvas, polyline3d, T_w2c, K, color,
                        radius=8, seg_interval=0.5,
                        near=1e-3, depth_max=80.):
    H, W = canvas.shape[:2]

    # ---------- 1. 一次性变换 & 投影 ---------- #
    pts_cam = (T_w2c[:3, :3] @ polyline3d.T + T_w2c[:3, 3:4]).T
    z = pts_cam[:, 2]
    cam_mask = z >= near  # 在近平面前方的点
    proj_uv = (K @ pts_cam.T)[:2].T  # shape (N, 2)
    proj_uv /= z[:, None]  # (x/z, y/z)

    u, v = proj_uv[:, 0], proj_uv[:, 1]

    # ---------- 2. per-segment 处理 ---------- #
    for i in range(len(pts_cam) - 1):
        p1c, p2c = pts_cam[i].copy(), pts_cam[i + 1].copy()
        z1, z2 = z[i], z[i + 1]

        # 2-a) z-裁剪到 NEAR
        if z1 < near and z2 < near:
            continue
        if z1 < near or z2 < near:
            t = (near - z1) / (z2 - z1) if z1 < near else (near - z2) / (z1 - z2)
            inter = p1c + t * (p2c - p1c) if z1 < near else p2c + t * (p1c - p2c)
            if z1 < near:
                p1c, z1 = inter, near
            else:
                p2c, z2 = inter, near

            # 只需要为**新增的交点**再算一次投影
            p = (K @ p1c) if z1 == near and (p1c is inter) else (K @ p2c)
            if z1 == near and (p1c is inter):
                proj_uv[i] = p[:2] / p[2]
            else:
                proj_uv[i + 1] = p[:2] / p[2]
            u, v = proj_uv[:, 0], proj_uv[:, 1]  # 更新引用

        # 2-b) 端点像素
        p1 = (int(round(u[i])), int(round(v[i])))
        p2 = (int(round(u[i + 1])), int(round(v[i + 1])))

        # 快速判定“整段在画面内” → 省一次 clipLine
        inside = (
                0 <= p1[0] < W and 0 <= p1[1] < H and
                0 <= p2[0] < W and 0 <= p2[1] < H
        )
        if inside:
            p1_img, p2_img = p1, p2
        else:
            ok, p1_img, p2_img = cv2.clipLine((0, 0, W - 1, H - 1), p1, p2)
            if not ok:
                continue

        # 2-c) 着色
        depth_mean = max(min((z1 + z2) * 0.5, depth_max), 0.)
        alpha = (depth_max - depth_mean) / depth_max
        col = (alpha * color).astype(np.uint8).tolist()

        cv2.line(canvas, p1_img, p2_img, col, radius, cv2.LINE_AA)


def _sutherland_hodgman(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clip a 2‑D polygon against an axis‑aligned screen rectangle using the
    Sutherland–Hodgman algorithm.

    Parameters
    ----------
    poly : (N, 2) array_like
        Polygon vertices (x, y) in image coordinates *in order*.
    w, h : int
        Image width and height.

    Returns
    -------
    np.ndarray, shape (M, 2)
        The clipped polygon (may be empty).
    """

    def clip_edge(pts: list[np.ndarray], inside_fn, intersect_fn):
        if not pts:
            return []
        output = []
        prev = pts[-1]
        prev_inside = inside_fn(prev)
        for curr in pts:
            curr_inside = inside_fn(curr)
            if curr_inside:
                if not prev_inside:  # entering – add intersection first
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_inside:  # leaving – add intersection only
                output.append(intersect_fn(prev, curr))
            prev, prev_inside = curr, curr_inside
        return output

    # Work in float to avoid precision loss
    pts = [np.asarray(p, float) for p in poly.tolist()]

    # Left   (x >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Right  (x <= w-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[0] <= w - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((w - 1 - p[0]) / (q[0] - p[0]))
    )
    if not pts:
        return np.empty((0, 2))

    # Top    (y >= 0)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] >= 0,
        intersect_fn=lambda p, q: p + (q - p) * ((0 - p[1]) / (q[1] - p[1]))
    )
    if not pts:
        return np.empty((0, 2))

    # Bottom (y <= h-1)
    pts = clip_edge(
        pts,
        inside_fn=lambda p: p[1] <= h - 1,
        intersect_fn=lambda p, q: p + (q - p) * ((h - 1 - p[1]) / (q[1] - p[1]))
    )

    return np.asarray(pts, dtype=np.float32)


def draw_polygon_depth(canvas: np.ndarray,
                       hull3d: np.ndarray,
                       T_w2c: np.ndarray,
                       K: np.ndarray,
                       color: np.ndarray,
                       depth_max) -> None:
    """Project a convex 3‑D polygon and draw its visible part with depth shading.

    Compared with the original implementation, this version **clips** the
    projected polygon against the image boundary so that even if some of the
    polygon’s vertices are outside the frame (or behind the camera), the visible
    portion is still rendered.
    """

    # --- World → camera space ------------------------------------------------
    pts_cam = (T_w2c[:3, :3] @ hull3d.T + T_w2c[:3, 3:4]).T  # (N, 3)

    # Cull vertices that are *behind* the camera (negative z). We ignore them
    # for projection but keep their depth for α if any remain in front.
    in_front = pts_cam[:, 2] > 1e-6
    if not np.any(in_front):
        return  # whole polygon is behind camera

    pts_cam_front = pts_cam[in_front]

    # --- Perspective projection (no validity filtering yet) -----------------
    uv_h = (K @ pts_cam_front.T).T  # (M, 3) – homogeneous
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    # --- Clip against the image rectangle -----------------------------------
    h, w = canvas.shape[:2]
    poly_clipped = _sutherland_hodgman(uv, w, h)
    if poly_clipped.shape[0] < 3:
        return  # Vanishes after clipping

    hull_uv = poly_clipped.astype(np.int32)

    # --- Depth‑based alpha ---------------------------------------------------
    depth_mean = float(np.clip(pts_cam_front[:, 2].mean(), 0.0, depth_max))
    alpha = (depth_max - depth_mean) / depth_max
    col = (alpha * np.asarray(color, dtype=float)).astype(np.uint8).tolist()

    # --- Rasterisation -------------------------------------------------------
    cv2.fillConvexPoly(canvas, hull_uv, col)


def draw_cuboids_with_occlusion(canvas, bboxes, T_w2c, K, depth_max=120.0):
    """
    在一张 canvas（H×W×3）上，将所有车辆的 3D 立方体面进行深度排序后填充：
    - 使用低饱和度的“粉彩”式颜色作为每个面的基础色，
    - 并根据面到相机的平均深度做线性颜色衰减（越远越暗）。
    - bboxes: 形状为 (N, >=7) 的数组。每一行至少包含 [x, y, z, L, W, H, yaw, ...]
    - T_w2c: 4×4 世界到相机的变换矩阵
    - K:      3×3 相机内参
    - depth_max: 用于裁剪深度时的最大深度（如果 Z 超过该值，就当作 depth_max 处理）
    """
    H, W = canvas.shape[:2]

    # ---- 1) 低饱和度粉彩底色（BGR 格式） ----
    #    颜色值都在 80~150 之间，保证偏灰，但又带一点色彩
    base_face_colors = [
        (247, 37, 133),  # front 面（微暖粉色）
        (76, 201, 240),  # back  面（微暖绿色）
        (114, 9, 183),  # left  面（微暖蓝色）
        (67, 97, 238),  # right 面（微暖黄色）
        (58, 12, 163),  # top   面（微暖青色）
        (58, 12, 163),  # bottom面（微暖紫色）
    ]

    # ---- 2) 面索引定义，与 vehicle_corners_local 返回的 8 个点顺序保持一致 ----
    face_indices = [
        [0, 1, 5, 4],  # front 面
        [2, 3, 7, 6],  # back  面
        [3, 0, 4, 7],  # left  面
        [1, 2, 6, 5],  # right 面
        [3, 2, 1, 0],  # top   面
        [4, 5, 6, 7],  # bottom面
    ]

    # ---- 3) 收集所有要绘制的“面” ----
    faces_to_draw = []  # 列表中每项：{'poly': np.int32((4,2)), 'depth': float, 'base_color': (B,G,R)}

    num_vehicles = bboxes.shape[0]
    for vi in range(num_vehicles):
        info = bboxes[vi]
        pos   = info[:3]      # (x, y, z)
        L     = info[3]       # 长
        Wd    = info[4]       # 宽
        H_box = info[5]       # 高
        yaw   = info[6]       # 偏航角

        # 3.1) 局部角点，(8,3)
        corners_loc = vehicle_corners_local(L, Wd, H_box)

        # 3.2) 世界坐标系下旋转 + 平移
        R_yaw = yaw_to_rot(yaw)                           
        corners_world = (R_yaw @ corners_loc.T).T + pos   

        # 3.3) 转到相机坐标系
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T  # (8,3)

        # 3.4) 投影到像素平面，得到 uv 以及 valid mask
        uv, valid = project_points_cam(pts_cam, K, (H, W))  # uv: (8,2)，valid: (8,)

        # 如果 8 个顶点里可见的少于 4 个，就跳过这辆车
        if valid.sum() < 4:
            continue

        # 3.5) 遍历 6 个面，收集可绘制的面
        for fi, idxs in enumerate(face_indices):
            pts_cam_face = pts_cam[idxs]  # (4,3)
            # 如果这个面所有顶点都在相机后方，就跳过
            if np.all(pts_cam_face[:, 2] <= 0):
                continue

            # 计算这个面顶点的平均深度，并 clamp 到 [0, depth_max]
            z_vals = pts_cam_face[:, 2].clip(0, depth_max)
            z_mean = float(np.mean(z_vals))

            # 只要这个面有至少一个顶点有效（落在图像内），就继续
            if not np.any(valid[idxs]):
                continue

            # 顶点在图像平面上的整数像素坐标
            poly_2d = np.array([uv[j] for j in idxs], dtype=np.int32)  # (4,2), dtype=int32

            faces_to_draw.append({
                'poly': poly_2d,
                'depth': z_mean,
                'base_color': base_face_colors[fi]
            })

    # ---- 4) 根据 depth 从大（最远）到小（最近）排序 ----
    faces_to_draw.sort(key=lambda x: x['depth'], reverse=True)

    # ---- 5) 按顺序绘制所有面，并做深度衰减（越远越暗） ----
    for face in faces_to_draw:
        poly       = face['poly']         # (4,2) 的 int32
        depth_mean = face['depth']        # 平均深度
        base_B, base_G, base_R = face['base_color']

        # 线性深度衰减系数 alpha ∈ [0,1]：1 表示最近，0 表示 depth_max
        alpha = np.clip((depth_max - depth_mean) / depth_max, 0.0, 1.0)

        # 应用衰减：直接在 BGR 三个通道上乘以 alpha
        B = int(base_B * alpha)
        G = int(base_G * alpha)
        R = int(base_R * alpha)

        cv2.fillConvexPoly(canvas, poly, (B, G, R), cv2.LINE_AA)



def vehicle_corners_local(L, W, H):
    """返回 (8,3) 车辆局部坐标顶点，Z 轴向上"""
    return np.array([
        [L / 2, W / 2, H / 2],  # 0 前左上
        [L / 2, -W / 2, H / 2],  # 1 前右上
        [-L / 2, -W / 2, H / 2],  # 2 后右上
        [-L / 2, W / 2, H / 2],  # 3 后左上
        [L / 2, W / 2, -H / 2],  # 4 前左下
        [L / 2, -W / 2, -H / 2],  # 5 前右下
        [-L / 2, -W / 2, -H / 2],  # 6 后右下
        [-L / 2, W / 2, -H / 2],  # 7 后左下
    ], dtype=np.float32)


def draw_cuboids_depth(canvas,
                       cuboids_world,       # list[(8,3)]
                       T_w2c,               # 4×4
                       K,                   # 3×3
                       colors_rgb=None,     # list[(r,g,b)]，可 None
                       depth_max=120.0,
                       edge_thickness=2):
    """
    在同一张 canvas 上绘制 N 个车辆 cuboid。
    可见性由面级深度排序确保（近处自动遮挡远处）。
    """
    H, W = canvas.shape[:2]
    if colors_rgb is None:
        colors_rgb = [(200, 0, 0)] * len(cuboids_world)

    # ——— 立方体 6 个面顶点索引 ———
    faces = [
        (0, 1, 2, 3),  # top
        (4, 5, 6, 7),  # bottom
        (0, 1, 5, 4),  # front
        (2, 3, 7, 6),  # back
        (1, 2, 6, 5),  # right
        (0, 3, 7, 4)   # left
    ]

    # ------------------------------------------------
    # 1⃣️ 先把所有 cuboid 的所有面丢进列表并算平均深度
    # ------------------------------------------------
    face_buffer = []   # (z_mean, poly_int32, fill_color_bgr)

    for corners_world, base_col_rgb in zip(cuboids_world, colors_rgb):
        # 世界 → 相机
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, (H, W))

        # 任一顶点可见即可尝试该面；如果所有顶点都看不见就跳过
        for idx in faces:
            if not valid[list(idx)].any():
                continue

            pts_cam_face = pts_cam[list(idx)]
            z_mean = float(np.clip(pts_cam_face[:, 2].mean(), 0, depth_max))

            # ——— 颜色：基础色 * 深度衰减 * 朝向阴影 ———
            depth_alpha = (depth_max - z_mean) / depth_max          # 近→1 远→0
            # 用简单 Lambert 估计：normal z 分量越负越朝向相机
            n = np.cross(pts_cam_face[1] - pts_cam_face[0],
                         pts_cam_face[2] - pts_cam_face[0])
            n = n / (np.linalg.norm(n) + 1e-6)
            facing = max(-n[2], 0.0)                                # 0~1
            shade = 0.3 + 0.7 * facing                              # 侧面更暗
            shade *= 0.6 + 0.4 * depth_alpha                        # 远处整体更暗
            col_face = tuple(int(shade * c) for c in base_col_rgb)  # RGB

            poly = uv[list(idx)].astype(np.int32)                   # (4,2)
            face_buffer.append((z_mean, poly, col_face[::-1]))      # BGR

    # ------------------------------------------------
    # 2⃣️ 远 → 近 排序并填充
    # ------------------------------------------------
    face_buffer.sort(key=lambda x: x[0], reverse=True)
    for _, poly, col_bgr in face_buffer:
        cv2.fillConvexPoly(canvas, poly, col_bgr, cv2.LINE_AA)

    # ------------------------------------------------
    # 3⃣️ 最后勾勒所有棱线（可选）
    # ------------------------------------------------
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for corners_world, base_col_rgb in zip(cuboids_world, colors_rgb):
        pts_cam = (T_w2c[:3, :3] @ corners_world.T + T_w2c[:3, 3:4]).T
        uv, valid = project_points_cam(pts_cam, K, (H, W))

        for i, j in edges:
            if not (valid[i] or valid[j]):
                continue
            z_mean = np.clip((pts_cam[i, 2] + pts_cam[j, 2]) * 0.5, 0, depth_max)
            depth_alpha = (depth_max - z_mean) / depth_max
            col_edge = tuple(int(depth_alpha * c) for c in base_col_rgb[::-1])  # BGR
            p1, p2 = tuple(uv[i]), tuple(uv[j])
            ok, p1c, p2c = cv2.clipLine((0, 0, W, H), p1, p2)
            if ok:
                cv2.line(canvas, p1c, p2c, col_edge, edge_thickness, cv2.LINE_AA)

import numpy as np
import cv2

def draw_cuboid_at(canvas,
                   center_pos,   # [x, y, z], 世界坐标系中长方体底面中心位置
                   dims,         # (L, W, H)
                   T_w2c,        # 4×4 世界→相机
                   K,            # 3×3 相机内参
                   color_rgb=(0, 255, 0),
                   thickness=-1  # -1 表示填充，>0 表示画线框
                   ):
    """
    在 canvas 上，以 center_pos 作为长方体底面中心，把一个 (L, W, H) 的长方体投影并绘制到图像上。
    - canvas: uint8 图像，H×W×3
    - center_pos: 长方体底面中心在世界坐标系中的 [x, y, z]
    - dims: (L, W, H)
    - T_w2c: 4×4 世界到相机坐标变换矩阵
    - K: 3×3 相机内参
    - color_rgb: (R, G, B)
    - thickness: -1 填充，>0 画边线
    """

    H_img, W_img = canvas.shape[:2]

    # 1) 先在“Local”坐标系里得到 8 个顶点。让底面中心位于 (0,0,0)：
    L, W, H_box = dims
    # 本地坐标系下的 8 个顶点（x, y, z）
    # 底面 z=0，顶面 z=H_box
    # 这里：x 轴指向正前方，y 轴指向右方，z 轴指向上方。
    # 下面顺序便于组合各面：
    local_corners = np.array([
        [ L/2,  W/2, 0.0],  # 0: front-right-bottom
        [ L/2, -W/2, 0.0],  # 1: front-left -bottom
        [-L/2, -W/2, 0.0],  # 2: back -left -bottom
        [-L/2,  W/2, 0.0],  # 3: back -right-bottom
        [ L/2,  W/2, H_box],# 4: front-right-top
        [ L/2, -W/2, H_box],# 5: front-left -top
        [-L/2, -W/2, H_box],# 6: back -left -top
        [-L/2,  W/2, H_box] # 7: back -right-top
    ], dtype=np.float32)  # (8,3)

    # 2) 从 Local → World：直接平移到 center_pos。因为交通信号灯一般竖直不旋转，这里不考虑 yaw。
    #    如果你要让它围绕 z 轴有朝向（比如竖直柱子有朝北朝南的方向），再插入旋转矩阵就行。
    center_pos = np.array(center_pos, dtype=np.float32).reshape(3,)
    world_corners = local_corners + center_pos  # (8,3)

    # 3) World → Camera 坐标系：
    #    pts_cam = R * world + t  （R = T_w2c[:3,:3], t = T_w2c[:3,3])
    pts_cam = (T_w2c[:3, :3] @ world_corners.T + T_w2c[:3, 3:4]).T  # (8,3)

    # 4) 投影到像素面，得到 uv=(8,2) 和 valid=(8,) boolean 掩码
    uv, valid = project_points_cam(pts_cam, K, (H_img, W_img))

    # 5) 定义 6 个面用到的顶点索引（4 个点一组），按照 local_corners 的顺序
    face_idxs = [
        [0, 1, 2, 3],  # 底面 （z=0）
        [4, 5, 6, 7],  # 顶面 （z=H_box）
        [0, 1, 5, 4],  # 前面（front）
        [1, 2, 6, 5],  # 左面（left）
        [2, 3, 7, 6],  # 后面（back）
        [3, 0, 4, 7],  # 右面（right）
    ]

    # 6) 遍历每个面，如果该面的4个顶点中至少有1个 “valid”（在图像内、Z>0），就画出该面
    for idxs in face_idxs:
        # 先检查是否有任意一个顶点在相机前方且投影落在图像范围内
        if not np.any([valid[i] for i in idxs]):
            continue

        # 再检查该面所有点是否都在相机后方: 如果都在 Z<=0，就跳过
        pts_face_cam = pts_cam[idxs]  # (4,3)
        if np.all(pts_face_cam[:, 2] <= 0):
            continue

        # 取出 2D 投影坐标（整数）
        poly2d = np.array([uv[i] for i in idxs], dtype=np.int32)  # (4,2)

        # 用 OpenCV 填充或画线
        if thickness < 0:
            cv2.fillConvexPoly(canvas, poly2d, color_rgb, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [poly2d], isClosed=True, color=color_rgb, thickness=thickness, lineType=cv2.LINE_AA)

def draw_heading_arrow(canvas,
                       pos_world,  # (3,)  物体在世界坐标中的质心
                       yaw,  # 标量 (弧度)
                       T_w2c,  # (4,4) 世界→相机
                       K,  # (3,3) 内参
                       color_rgb=(255, 255, 0),
                       arrow_len=3.0,  # 以米为单位，在图上可调
                       thickness=6):
    H, W = canvas.shape[:2]
    color_bgr = tuple(int(c) for c in color_rgb)

    # ---- 1. 计算箭头两端的世界坐标 ------------------------------------------
    # “车头”方向向量（世界系）
    dir_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    p_tail_w = pos_world
    p_head_w = pos_world + dir_world * arrow_len

    # ---- 2. 世界 → 相机 -------------------------------------------------------
    pw_tail_c = T_w2c[:3, :3] @ p_tail_w + T_w2c[:3, 3]
    pw_head_c = T_w2c[:3, :3] @ p_head_w + T_w2c[:3, 3]

    # 过滤：若尾点就在相机后面（z<=0），直接跳过
    if pw_tail_c[2] <= 0 or pw_head_c[2] <= 0:
        return

    # ---- 3. 投影到像素坐标 ----------------------------------------------------
    pts_cam = np.vstack([pw_tail_c, pw_head_c])  # shape (2,3)
    uv, valid = project_points_cam(pts_cam, K, (H, W))  # uv: (2,2)

    if not valid.all():
        return

    p_tail_px, p_head_px = map(tuple, uv.astype(int))

    # ---- 4. 画箭头 ------------------------------------------------------------
    cv2.arrowedLine(canvas,
                    p_tail_px,
                    p_head_px,
                    color_bgr,
                    thickness,
                    tipLength=0.25)  # tipLength 相对箭头长度的比例


class ScenarioRenderer:
    def __init__(self,
                 camera_channel_list=['CAM_F0', 'CAM_L0', 'CAM_R0'],
                 width=1920,
                 height=1120,
                 depth_max=120.0,
                 use_gpu_full_scene: bool = False,
                 use_gpu_cuboid_instances: bool = False,
                 use_gpu_cuboid_batches: bool = False,
                 use_gpu_filled_regions: bool = False,
                 use_gpu_line_overlays: bool = False,
                 use_gpu_arrows: bool = False,
                 quality_mode: str = "parity",
                 torch_device: str = "cuda",
                 enable_nvdiffrast_antialias: bool = True):
        self.width = width
        self.height = height
        self.depth_max = depth_max
        self.use_gpu_full_scene = use_gpu_full_scene
        self.use_gpu_cuboid_instances = use_gpu_cuboid_instances or use_gpu_full_scene
        self.use_gpu_cuboid_batches = use_gpu_cuboid_batches
        self.use_gpu_filled_regions = use_gpu_filled_regions
        self.use_gpu_line_overlays = use_gpu_line_overlays
        self.use_gpu_arrows = use_gpu_arrows
        self.quality_mode = quality_mode
        if use_gpu_full_scene:
            self.use_gpu_cuboid_batches = True
            self.use_gpu_filled_regions = True
            self.use_gpu_line_overlays = True
            self.use_gpu_arrows = True
        self.torch_device = torch_device
        self.enable_nvdiffrast_antialias = enable_nvdiffrast_antialias
        self.camera_models = {}
        self._torch = None
        self._dr = None
        self._dr_ctx = None
        self._scene_static_cache: Dict[Tuple[Any, ...], SceneStaticCacheEntry] = {}
        self._line_points_torch_cache: Dict[int, Any] = {}
        self._perf_last: Dict[str, float] = {}
        self._cache_scope: Optional[Tuple[Any, ...]] = None
        for k, v in camera_params.items():
            if not k in camera_channel_list: continue
            self.camera_models[k] = v

    @staticmethod
    def _ground_points_from_poly2d(poly2d: np.ndarray) -> np.ndarray:
        return np.hstack([poly2d, np.zeros((poly2d.shape[0], 1), np.float32)])

    @staticmethod
    def _polyline_cull_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return np.zeros(2, dtype=np.float32), 0.0
        if pts.shape[1] >= 2:
            pts_xy = pts[:, :2]
        else:
            return np.zeros(2, dtype=np.float32), 0.0
        center = np.mean(pts_xy, axis=0, dtype=np.float32)
        radius = float(np.max(np.linalg.norm(pts_xy - center[None, :], axis=1))) if pts_xy.shape[0] else 0.0
        return center.astype(np.float32), radius

    @staticmethod
    def _traffic_light_world_corners(cuboid: CuboidInstance) -> np.ndarray:
        L, W, H_box = cuboid.dims
        local_corners = np.array([
            [L / 2, W / 2, 0.0],
            [L / 2, -W / 2, 0.0],
            [-L / 2, -W / 2, 0.0],
            [-L / 2, W / 2, 0.0],
            [L / 2, W / 2, H_box],
            [L / 2, -W / 2, H_box],
            [-L / 2, -W / 2, H_box],
            [-L / 2, W / 2, H_box],
        ], dtype=np.float32)
        return local_corners + cuboid.center_world.astype(np.float32).reshape(3,)

    def _lower_line_overlay_to_triangle_batch(self,
                                              camera: CameraState,
                                              overlay: LineOverlay) -> Optional[LoweredTriangleBatch]:
        pts_cam = _transform_world_to_camera(overlay.points_world, camera.T_w2c).astype(np.float32)
        proj_uv = _project_homogeneous_float(pts_cam, camera.K).copy()
        z = pts_cam[:, 2]

        tri_vertices = []
        tri_depths = []
        tri_colors = []
        half_width = max(float(overlay.radius), 1.0) * 0.5

        for i in range(len(pts_cam) - 1):
            p1c, p2c = pts_cam[i].copy(), pts_cam[i + 1].copy()
            z1, z2 = float(z[i]), float(z[i + 1])

            if z1 < overlay.near and z2 < overlay.near:
                continue
            if z1 < overlay.near or z2 < overlay.near:
                t = (overlay.near - z1) / (z2 - z1) if z1 < overlay.near else (overlay.near - z2) / (z1 - z2)
                inter = p1c + t * (p2c - p1c) if z1 < overlay.near else p2c + t * (p1c - p2c)
                if z1 < overlay.near:
                    p1c, z1 = inter, overlay.near
                    proj_uv[i] = _project_homogeneous_float(p1c[None, :], camera.K)[0]
                else:
                    p2c, z2 = inter, overlay.near
                    proj_uv[i + 1] = _project_homogeneous_float(p2c[None, :], camera.K)[0]

            p1 = proj_uv[i].copy()
            p2 = proj_uv[i + 1].copy()
            direction = p2 - p1
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                continue

            normal = np.array([-direction[1], direction[0]], dtype=np.float32) / norm
            offset = normal * half_width
            quad = np.stack([p1 + offset, p1 - offset, p2 - offset, p2 + offset], axis=0)
            depth = np.array([z1, z1, z2, z2], dtype=np.float32)
            depth_mean = float(np.clip((z1 + z2) * 0.5, 0.0, overlay.depth_max))
            color = _depth_attenuated_color(overlay.color, depth_mean, overlay.depth_max)
            tris, tri_depth = _quad_to_triangles(quad, depth)
            tri_vertices.extend(tris)
            tri_depths.extend(tri_depth)
            tri_colors.extend([color.copy(), color.copy()])

        return _make_triangle_batch(overlay.semantic_tag, "line_overlay", tri_vertices, tri_depths, tri_colors)

    def _get_world_points_torch(self, points_world: np.ndarray):
        torch, _ = self._lazy_import_torch_raster()
        cache_key = id(points_world)
        cached = self._line_points_torch_cache.get(cache_key)
        if cached is not None:
            return cached
        pts = torch.as_tensor(np.asarray(points_world, dtype=np.float32), dtype=torch.float32, device=self.torch_device)
        self._line_points_torch_cache[cache_key] = pts
        return pts

    def _build_camera_tensors(self, camera: CameraState):
        torch, _ = self._lazy_import_torch_raster()
        device = torch.device(self.torch_device)
        K = torch.as_tensor(camera.K, dtype=torch.float32, device=device)
        T = torch.as_tensor(camera.T_w2c, dtype=torch.float32, device=device)
        return {
            "R": T[:3, :3],
            "t": T[:3, 3],
            "fx": K[0, 0],
            "fy": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "W": float(camera.width),
            "H": float(camera.height),
        }

    def _collect_line_overlays_for_gpu(self, packet: CameraScenePacket) -> List[LineOverlay]:
        overlays = list(packet.line_overlays)
        for region in packet.filled_regions:
            if region.outline_color is None:
                continue
            cull_center, cull_radius = self._polyline_cull_circle(region.points_world)
            overlays.append(
                LineOverlay(
                    semantic_tag=f"{region.semantic_tag}_outline",
                    points_world=region.points_world,
                    color=region.outline_color,
                    radius=region.outline_radius,
                    near=region.outline_near,
                    depth_max=region.depth_max,
                    cull_center_xy=cull_center,
                    cull_radius_xy=cull_radius,
                )
            )
        return overlays

    def _lower_line_overlay_torch(self, overlay: LineOverlay, cam_tensors):
        torch, _ = self._lazy_import_torch_raster()
        pts_world = self._get_world_points_torch(overlay.points_world)
        if pts_world.shape[0] < 2:
            return None

        pts_cam = pts_world @ cam_tensors["R"].T + cam_tensors["t"]
        p1 = pts_cam[:-1]
        p2 = pts_cam[1:]
        z1 = p1[:, 2]
        z2 = p2[:, 2]
        near = float(overlay.near)

        visible_seg = ~((z1 < near) & (z2 < near))
        if not torch.any(visible_seg):
            return None

        p1 = p1[visible_seg]
        p2 = p2[visible_seg]
        z1 = z1[visible_seg]
        z2 = z2[visible_seg]

        denom = z2 - z1
        safe = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
        t = (near - z1) / safe
        inter = p1 + t[:, None] * (p2 - p1)

        clip_p1 = (z1 < near) & (z2 >= near)
        clip_p2 = (z2 < near) & (z1 >= near)
        p1 = torch.where(clip_p1[:, None], inter, p1)
        p2 = torch.where(clip_p2[:, None], inter, p2)
        z1 = torch.where(clip_p1, torch.full_like(z1, near), z1)
        z2 = torch.where(clip_p2, torch.full_like(z2, near), z2)

        u1 = cam_tensors["fx"] * p1[:, 0] / z1 + cam_tensors["cx"]
        v1 = cam_tensors["fy"] * p1[:, 1] / z1 + cam_tensors["cy"]
        u2 = cam_tensors["fx"] * p2[:, 0] / z2 + cam_tensors["cx"]
        v2 = cam_tensors["fy"] * p2[:, 1] / z2 + cam_tensors["cy"]
        uv1 = torch.stack([u1, v1], dim=1)
        uv2 = torch.stack([u2, v2], dim=1)

        dir_uv = uv2 - uv1
        seg_len = torch.linalg.norm(dir_uv, dim=1)
        valid_len = seg_len > 1e-6

        min_u = torch.minimum(u1, u2)
        max_u = torch.maximum(u1, u2)
        min_v = torch.minimum(v1, v2)
        max_v = torch.maximum(v1, v2)
        in_view = (max_u >= 0.0) & (min_u < cam_tensors["W"]) & (max_v >= 0.0) & (min_v < cam_tensors["H"])
        keep = valid_len & in_view
        if not torch.any(keep):
            return None

        uv1 = uv1[keep]
        uv2 = uv2[keep]
        z1 = z1[keep]
        z2 = z2[keep]
        dir_uv = dir_uv[keep]
        seg_len = seg_len[keep]

        half_width = max(float(overlay.radius), 1.0) * 0.5
        normal = torch.stack([-dir_uv[:, 1], dir_uv[:, 0]], dim=1) / seg_len[:, None]
        offset = normal * half_width

        quad = torch.stack([uv1 + offset, uv1 - offset, uv2 - offset, uv2 + offset], dim=1)
        depth = torch.stack([z1, z1, z2, z2], dim=1)

        tri_pos = torch.cat([quad[:, [0, 1, 2], :], quad[:, [0, 2, 3], :]], dim=0)
        tri_depth = torch.cat([depth[:, [0, 1, 2]], depth[:, [0, 2, 3]]], dim=0)

        depth_mean = torch.clamp((z1 + z2) * 0.5, 0.0, float(overlay.depth_max))
        alpha = torch.clamp((float(overlay.depth_max) - depth_mean) / max(float(overlay.depth_max), 1e-6), 0.0, 1.0)
        base_color = torch.as_tensor(np.asarray(overlay.color, dtype=np.float32), dtype=torch.float32, device=tri_pos.device)
        seg_color = alpha[:, None] * base_color[None, :]
        tri_color = seg_color.repeat_interleave(2, dim=0)
        tri_color = tri_color[:, None, :].expand(-1, 3, -1)
        return tri_pos, tri_depth, tri_color

    def _lower_line_overlays_torch_batched(self,
                                           overlays: List[LineOverlay],
                                           camera: CameraState,
                                           cam_tensors):
        torch, _ = self._lazy_import_torch_raster()
        device = torch.device(self.torch_device)

        p1_chunks = []
        p2_chunks = []
        near_chunks = []
        depth_max_chunks = []
        half_width_chunks = []
        color_chunks = []

        for overlay in overlays:
            points_world = np.asarray(overlay.points_world, dtype=np.float32)
            if points_world.shape[0] < 2:
                continue
            if self.quality_mode == "perf" and overlay.cull_center_xy is not None:
                center_world = np.array(
                    [float(overlay.cull_center_xy[0]), float(overlay.cull_center_xy[1]), 0.0],
                    dtype=np.float32,
                )
                center_cam = _transform_world_to_camera(center_world[None, :], camera.T_w2c)[0].astype(np.float32)
                zc = float(center_cam[2])
                if (zc + float(overlay.cull_radius_xy)) <= float(overlay.near):
                    continue
                if zc > float(overlay.near):
                    u = float(camera.K[0, 0] * center_cam[0] / zc + camera.K[0, 2])
                    v = float(camera.K[1, 1] * center_cam[1] / zc + camera.K[1, 2])
                    rpx_u = float(camera.K[0, 0] * float(overlay.cull_radius_xy) / max(zc, float(overlay.near)))
                    rpx_v = float(camera.K[1, 1] * float(overlay.cull_radius_xy) / max(zc, float(overlay.near)))
                    margin = 96.0
                    if (
                        (u + rpx_u) < -margin
                        or (u - rpx_u) > (float(camera.width) + margin)
                        or (v + rpx_v) < -margin
                        or (v - rpx_v) > (float(camera.height) + margin)
                    ):
                        continue

            pts_world = self._get_world_points_torch(points_world)
            if self.quality_mode == "perf" and pts_world.shape[0] > 64:
                # Perf mode allows slight overlay relaxation; decimate very dense polylines.
                decimate_stride = 4 if pts_world.shape[0] > 256 else 2
                pts_world = pts_world[::decimate_stride]
                if pts_world.shape[0] < 2:
                    continue
            if pts_world.shape[0] < 2:
                continue
            pts_cam = pts_world @ cam_tensors["R"].T + cam_tensors["t"]
            seg_count = int(pts_cam.shape[0]) - 1
            if seg_count <= 0:
                continue
            p1_chunks.append(pts_cam[:-1])
            p2_chunks.append(pts_cam[1:])
            near_chunks.append(torch.full((seg_count,), float(overlay.near), dtype=torch.float32, device=device))
            depth_max_chunks.append(torch.full((seg_count,), float(overlay.depth_max), dtype=torch.float32, device=device))
            half_width_chunks.append(
                torch.full((seg_count,), max(float(overlay.radius), 1.0) * 0.5, dtype=torch.float32, device=device)
            )
            base_color = torch.as_tensor(
                np.asarray(overlay.color, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            color_chunks.append(base_color[None, :].expand(seg_count, -1))

        if not p1_chunks:
            return None

        p1 = torch.cat(p1_chunks, dim=0)
        p2 = torch.cat(p2_chunks, dim=0)
        near = torch.cat(near_chunks, dim=0)
        depth_max = torch.cat(depth_max_chunks, dim=0)
        half_width = torch.cat(half_width_chunks, dim=0)
        base_color = torch.cat(color_chunks, dim=0)

        z1 = p1[:, 2]
        z2 = p2[:, 2]
        visible_seg = ~((z1 < near) & (z2 < near))
        p1 = p1[visible_seg]
        p2 = p2[visible_seg]
        z1 = z1[visible_seg]
        z2 = z2[visible_seg]
        near = near[visible_seg]
        depth_max = depth_max[visible_seg]
        half_width = half_width[visible_seg]
        base_color = base_color[visible_seg]
        if p1.shape[0] == 0:
            return None

        denom = z2 - z1
        safe = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
        t = (near - z1) / safe
        inter = p1 + t[:, None] * (p2 - p1)
        clip_p1 = (z1 < near) & (z2 >= near)
        clip_p2 = (z2 < near) & (z1 >= near)
        p1 = torch.where(clip_p1[:, None], inter, p1)
        p2 = torch.where(clip_p2[:, None], inter, p2)
        z1 = torch.where(clip_p1, near, z1)
        z2 = torch.where(clip_p2, near, z2)

        u1 = cam_tensors["fx"] * p1[:, 0] / z1 + cam_tensors["cx"]
        v1 = cam_tensors["fy"] * p1[:, 1] / z1 + cam_tensors["cy"]
        u2 = cam_tensors["fx"] * p2[:, 0] / z2 + cam_tensors["cx"]
        v2 = cam_tensors["fy"] * p2[:, 1] / z2 + cam_tensors["cy"]
        uv1 = torch.stack([u1, v1], dim=1)
        uv2 = torch.stack([u2, v2], dim=1)

        dir_uv = uv2 - uv1
        seg_len = torch.linalg.norm(dir_uv, dim=1)
        min_u = torch.minimum(u1, u2)
        max_u = torch.maximum(u1, u2)
        min_v = torch.minimum(v1, v2)
        max_v = torch.maximum(v1, v2)
        in_view = (max_u >= 0.0) & (min_u < cam_tensors["W"]) & (max_v >= 0.0) & (min_v < cam_tensors["H"])
        keep = (seg_len > 1e-6) & in_view
        uv1 = uv1[keep]
        uv2 = uv2[keep]
        z1 = z1[keep]
        z2 = z2[keep]
        seg_len = seg_len[keep]
        half_width = half_width[keep]
        depth_max = depth_max[keep]
        base_color = base_color[keep]
        dir_uv = dir_uv[keep]
        if uv1.shape[0] == 0:
            return None

        normal = torch.stack([-dir_uv[:, 1], dir_uv[:, 0]], dim=1) / seg_len[:, None]
        offset = normal * half_width[:, None]
        quad = torch.stack([uv1 + offset, uv1 - offset, uv2 - offset, uv2 + offset], dim=1)
        depth = torch.stack([z1, z1, z2, z2], dim=1)

        tri_pos = torch.cat([quad[:, [0, 1, 2], :], quad[:, [0, 2, 3], :]], dim=0)
        tri_depth = torch.cat([depth[:, [0, 1, 2]], depth[:, [0, 2, 3]]], dim=0)
        depth_mid = (z1 + z2) * 0.5
        depth_mean = torch.minimum(torch.maximum(depth_mid, torch.zeros_like(depth_mid)), depth_max)
        alpha = torch.clamp((depth_max - depth_mean) / torch.clamp(depth_max, min=1e-6), 0.0, 1.0)
        seg_color = alpha[:, None] * base_color
        tri_color = seg_color.repeat_interleave(2, dim=0)
        tri_color = tri_color[:, None, :].expand(-1, 3, -1)
        return tri_pos, tri_depth, tri_color

    def _rasterize_line_overlays_fast(self,
                                      overlays: List[LineOverlay],
                                      camera: CameraState):
        if not overlays:
            return None

        t_lower = time.perf_counter()
        torch, dr = self._lazy_import_torch_raster()
        ctx = self._get_raster_context()
        cam_tensors = self._build_camera_tensors(camera)
        lowered = self._lower_line_overlays_torch_batched(overlays, camera, cam_tensors)
        self._perf_add("lower_lines_ms", (time.perf_counter() - t_lower) * 1000.0)
        if lowered is None:
            return None

        t_raster = time.perf_counter()
        tri_pos, tri_depth, tri_color = lowered

        verts_px = tri_pos.reshape(-1, 2)
        verts_depth = tri_depth.reshape(-1)
        verts_color = tri_color.reshape(-1, 3) / 255.0

        x = (verts_px[:, 0] / max(float(camera.width), 1.0)) * 2.0 - 1.0
        y = 1.0 - (verts_px[:, 1] / max(float(camera.height), 1.0)) * 2.0
        z = torch.clamp(verts_depth / max(float(camera.depth_max), 1e-6), 0.0, 1.0) * 2.0 - 1.0
        w = torch.ones_like(z)
        pos = torch.stack([x, y, z, w], dim=1).unsqueeze(0)
        tri = torch.arange(verts_px.shape[0], device=verts_px.device, dtype=torch.int32).reshape(-1, 3)
        color = verts_color.unsqueeze(0)

        rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=[camera.height, camera.width])
        color_img, _ = dr.interpolate(color, rast, tri, rast_db=rast_db)
        use_overlay_aa = self.enable_nvdiffrast_antialias and self.quality_mode != "perf"
        if use_overlay_aa:
            color_img = dr.antialias(color_img, rast, pos, tri)
        alpha = (rast[..., 3:4] > 0).to(color_img.dtype)
        rgba = torch.cat([color_img * alpha, alpha], dim=-1)
        rgba = torch.flip(rgba[0], dims=[0])
        rgba = torch.clamp(rgba, 0.0, 1.0).mul(255.0).byte()
        self._perf_add("raster_overlay_ms", (time.perf_counter() - t_raster) * 1000.0)
        return rgba

    def _lower_filled_region_to_triangle_batch(self,
                                               camera: CameraState,
                                               region: FilledRegion) -> Optional[LoweredTriangleBatch]:
        pts_cam = _transform_world_to_camera(region.points_world, camera.T_w2c).astype(np.float32)
        in_front = pts_cam[:, 2] > 1e-6
        if not np.any(in_front):
            return None

        pts_cam_front = pts_cam[in_front]
        uv = _project_homogeneous_float(pts_cam_front, camera.K)
        poly_clipped = _sutherland_hodgman(uv, camera.width, camera.height)
        if poly_clipped.shape[0] < 3:
            return None

        depth_mean = float(np.clip(pts_cam_front[:, 2].mean(), 0.0, region.depth_max))
        color = _depth_attenuated_color(region.fill_color, depth_mean, region.depth_max)

        tri_vertices = []
        tri_depths = []
        tri_colors = []
        for i in range(1, poly_clipped.shape[0] - 1):
            tri_vertices.append(poly_clipped[[0, i, i + 1]].astype(np.float32))
            tri_depths.append(np.full(3, depth_mean, dtype=np.float32))
            tri_colors.append(color.copy())

        return _make_triangle_batch(region.semantic_tag, "filled_region", tri_vertices, tri_depths, tri_colors)

    def _lower_cuboid_instance_to_triangle_batch(self,
                                                 camera: CameraState,
                                                 cuboid: CuboidInstance) -> Optional[LoweredTriangleBatch]:
        if cuboid.thickness >= 0:
            return None

        world_corners = self._traffic_light_world_corners(cuboid)
        pts_cam = _transform_world_to_camera(world_corners, camera.T_w2c).astype(np.float32)
        uv, valid = _project_points_cam_float(pts_cam, camera.K, (camera.height, camera.width))
        face_idxs = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]

        tri_vertices = []
        tri_depths = []
        tri_colors = []
        for idxs in face_idxs:
            if not np.any(valid[idxs]):
                continue
            pts_face_cam = pts_cam[idxs]
            if np.all(pts_face_cam[:, 2] <= 0):
                continue
            quad = uv[idxs].astype(np.float32)
            depth = pts_face_cam[:, 2].astype(np.float32)
            tris, tri_depth = _quad_to_triangles(quad, depth)
            tri_vertices.extend(tris)
            tri_depths.extend(tri_depth)
            tri_colors.extend([cuboid.color_rgb.copy(), cuboid.color_rgb.copy()])

        return _make_triangle_batch(cuboid.semantic_tag, "cuboid_instance", tri_vertices, tri_depths, tri_colors)

    def _lower_cuboid_batch_to_triangle_batch(self,
                                              camera: CameraState,
                                              cuboid_batch: CuboidBatch) -> Optional[LoweredTriangleBatch]:
        face_indices = [
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [1, 2, 6, 5],
            [3, 2, 1, 0],
            [4, 5, 6, 7],
        ]
        base_face_colors = [
            np.array((247, 37, 133), dtype=np.uint8),
            np.array((76, 201, 240), dtype=np.uint8),
            np.array((114, 9, 183), dtype=np.uint8),
            np.array((67, 97, 238), dtype=np.uint8),
            np.array((58, 12, 163), dtype=np.uint8),
            np.array((58, 12, 163), dtype=np.uint8),
        ]

        tri_vertices = []
        tri_depths = []
        tri_colors = []
        for info in cuboid_batch.bboxes_world:
            pos = info[:3]
            L = info[3]
            Wd = info[4]
            H_box = info[5]
            yaw = info[6]
            corners_loc = vehicle_corners_local(L, Wd, H_box)
            corners_world = (yaw_to_rot(float(yaw)) @ corners_loc.T).T + pos
            pts_cam = _transform_world_to_camera(corners_world, camera.T_w2c).astype(np.float32)
            uv, valid = _project_points_cam_float(pts_cam, camera.K, (camera.height, camera.width))

            if valid.sum() < 4:
                continue

            for fi, idxs in enumerate(face_indices):
                pts_cam_face = pts_cam[idxs]
                if np.all(pts_cam_face[:, 2] <= 0):
                    continue
                if not np.any(valid[idxs]):
                    continue

                z_vals = pts_cam_face[:, 2].clip(0, cuboid_batch.depth_max)
                z_mean = float(np.mean(z_vals))
                color = _depth_attenuated_color(base_face_colors[fi], z_mean, cuboid_batch.depth_max)
                quad = uv[idxs].astype(np.float32)
                depth = pts_cam_face[:, 2].astype(np.float32)
                tris, tri_depth = _quad_to_triangles(quad, depth)
                tri_vertices.extend(tris)
                tri_depths.extend(tri_depth)
                tri_colors.extend([color.copy(), color.copy()])

        return _make_triangle_batch(cuboid_batch.semantic_tag, "cuboid_batch", tri_vertices, tri_depths, tri_colors)

    def _lower_arrow_to_triangle_batch(self,
                                       camera: CameraState,
                                       arrow: ArrowOverlay) -> Optional[LoweredTriangleBatch]:
        dir_world = np.array([np.cos(arrow.yaw), np.sin(arrow.yaw), 0.0], dtype=np.float32)
        p_tail_w = np.asarray(arrow.pos_world, dtype=np.float32)
        p_head_w = p_tail_w + dir_world * float(arrow.arrow_len)

        pts_cam = _transform_world_to_camera(np.vstack([p_tail_w, p_head_w]), camera.T_w2c).astype(np.float32)
        if np.any(pts_cam[:, 2] <= 0):
            return None

        uv, valid = _project_points_cam_float(pts_cam, camera.K, (camera.height, camera.width))
        if not valid.all():
            return None

        tail_px, head_px = uv
        direction = head_px - tail_px
        length_px = float(np.linalg.norm(direction))
        if length_px <= 1e-6:
            return None

        direction = direction / length_px
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        half_width = max(float(arrow.thickness), 1.0) * 0.5
        head_len = max(length_px * 0.25, float(arrow.thickness))
        head_base = head_px - direction * head_len

        shaft_quad = np.stack([
            tail_px + normal * half_width,
            tail_px - normal * half_width,
            head_base - normal * half_width,
            head_base + normal * half_width,
        ], axis=0)
        shaft_depth = np.array([pts_cam[0, 2], pts_cam[0, 2], pts_cam[1, 2], pts_cam[1, 2]], dtype=np.float32)
        head_tri = np.stack([
            head_px,
            head_base - normal * float(arrow.thickness),
            head_base + normal * float(arrow.thickness),
        ], axis=0).astype(np.float32)
        head_depth = np.full(3, pts_cam[1, 2], dtype=np.float32)
        color = np.asarray(arrow.color_rgb, dtype=np.uint8)

        tri_vertices, tri_depths = _quad_to_triangles(shaft_quad, shaft_depth)
        tri_vertices.append(head_tri)
        tri_depths.append(head_depth)
        tri_colors = [color.copy(), color.copy(), color.copy()]
        return _make_triangle_batch(arrow.semantic_tag, "arrow_overlay", tri_vertices, tri_depths, tri_colors)

    def _lower_scene_packet(self,
                            packet: CameraScenePacket,
                            include_filled_regions: bool = True,
                            include_cuboid_instances: bool = True,
                            include_cuboid_batches: bool = True,
                            include_line_overlays: bool = True,
                            include_arrows: bool = True) -> LoweredCameraScene:
        lowered = LoweredCameraScene(camera=packet.camera)

        if include_filled_regions or include_line_overlays:
            t0 = time.perf_counter()
            for region in packet.filled_regions:
                if include_filled_regions:
                    batch = self._lower_filled_region_to_triangle_batch(packet.camera, region)
                    if batch is not None:
                        lowered.filled_triangle_batches.append(batch)
                if include_line_overlays and region.outline_color is not None:
                    outline = LineOverlay(
                        semantic_tag=f"{region.semantic_tag}_outline",
                        points_world=region.points_world,
                        color=region.outline_color,
                        radius=region.outline_radius,
                        near=region.outline_near,
                        depth_max=region.depth_max,
                    )
                    outline_batch = self._lower_line_overlay_to_triangle_batch(packet.camera, outline)
                    if outline_batch is not None:
                        lowered.overlay_triangle_batches.append(outline_batch)
            self._perf_add("lower_filled_ms", (time.perf_counter() - t0) * 1000.0)

        if include_cuboid_instances:
            t0 = time.perf_counter()
            for cuboid in packet.cuboid_instances:
                batch = self._lower_cuboid_instance_to_triangle_batch(packet.camera, cuboid)
                if batch is not None:
                    lowered.cuboid_triangle_batches.append(batch)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._perf_add("lower_cuboid_instances_ms", dt_ms)
            self._perf_add("lower_cuboids_ms", dt_ms)

        if include_cuboid_batches:
            t0 = time.perf_counter()
            for cuboid_batch in packet.cuboid_batches:
                batch = self._lower_cuboid_batch_to_triangle_batch(packet.camera, cuboid_batch)
                if batch is not None:
                    lowered.cuboid_triangle_batches.append(batch)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self._perf_add("lower_cuboid_batches_ms", dt_ms)
            self._perf_add("lower_cuboids_ms", dt_ms)

        if include_line_overlays:
            t0 = time.perf_counter()
            for overlay in packet.line_overlays:
                batch = self._lower_line_overlay_to_triangle_batch(packet.camera, overlay)
                if batch is not None:
                    lowered.overlay_triangle_batches.append(batch)
            self._perf_add("lower_lines_ms", (time.perf_counter() - t0) * 1000.0)

        if include_arrows:
            t0 = time.perf_counter()
            for arrow in packet.arrow_overlays:
                batch = self._lower_arrow_to_triangle_batch(packet.camera, arrow)
                if batch is not None:
                    lowered.overlay_triangle_batches.append(batch)
            self._perf_add("lower_arrows_ms", (time.perf_counter() - t0) * 1000.0)

        return lowered

    def _lazy_import_torch_raster(self):
        if self._torch is None:
            import torch
            self._torch = torch
        if self._dr is None:
            import nvdiffrast.torch as dr
            self._dr = dr
        return self._torch, self._dr

    def _get_raster_context(self):
        torch, dr = self._lazy_import_torch_raster()
        if not torch.cuda.is_available():
            raise RuntimeError("GPU rasterization requires torch CUDA availability.")
        if self._dr_ctx is None:
            self._dr_ctx = dr.RasterizeCudaContext(device=self.torch_device)
        return self._dr_ctx

    @staticmethod
    def _pixels_to_clip_space(vertices_px: np.ndarray,
                              vertex_depth: np.ndarray,
                              width: int,
                              height: int,
                              depth_max: float) -> np.ndarray:
        x = (vertices_px[..., 0] / max(width, 1)) * 2.0 - 1.0
        y = 1.0 - (vertices_px[..., 1] / max(height, 1)) * 2.0
        z = np.clip(vertex_depth / max(depth_max, 1e-6), 0.0, 1.0) * 2.0 - 1.0
        w = np.ones_like(z, dtype=np.float32)
        return np.stack([x, y, z, w], axis=-1).astype(np.float32)

    def _pack_triangle_batches_for_raster(self,
                                          triangle_batches: List[LoweredTriangleBatch],
                                          camera: CameraState):
        if not triangle_batches:
            return None

        clip_vertices = []
        colors = []
        tri_indices = []
        vertex_offset = 0
        for batch in triangle_batches:
            batch_clip = self._pixels_to_clip_space(
                batch.vertices_px.reshape(-1, 2),
                batch.vertex_depth.reshape(-1),
                camera.width,
                camera.height,
                camera.depth_max,
            )
            batch_color = (batch.vertex_color.reshape(-1, 3).astype(np.float32) / 255.0)
            num_vertices = batch_clip.shape[0]
            num_tris = batch.triangle_count

            clip_vertices.append(batch_clip)
            colors.append(batch_color)
            tri = np.arange(vertex_offset, vertex_offset + num_vertices, dtype=np.int32).reshape(num_tris, 3)
            tri_indices.append(tri)
            vertex_offset += num_vertices

        packed = {
            "pos": np.concatenate(clip_vertices, axis=0),
            "color": np.concatenate(colors, axis=0),
            "tri": np.concatenate(tri_indices, axis=0),
        }
        return packed

    def _rasterize_solid_triangle_batches(self,
                                          triangle_batches: List[LoweredTriangleBatch],
                                          camera: CameraState,
                                          return_torch: bool = False):
        packed = self._pack_triangle_batches_for_raster(triangle_batches, camera)
        if packed is None:
            return None

        torch, dr = self._lazy_import_torch_raster()
        ctx = self._get_raster_context()
        device = torch.device(self.torch_device)

        pos = torch.as_tensor(packed["pos"], dtype=torch.float32, device=device).unsqueeze(0)
        tri = torch.as_tensor(packed["tri"], dtype=torch.int32, device=device)
        color = torch.as_tensor(packed["color"], dtype=torch.float32, device=device).unsqueeze(0)

        rast, rast_db = dr.rasterize(ctx, pos, tri, resolution=[camera.height, camera.width])
        color_img, _ = dr.interpolate(color, rast, tri, rast_db=rast_db)
        if self.enable_nvdiffrast_antialias:
            color_img = dr.antialias(color_img, rast, pos, tri)

        alpha = (rast[..., 3:4] > 0).to(color_img.dtype)
        rgba = torch.cat([color_img * alpha, alpha], dim=-1)
        # nvdiffrast returns image rows in the opposite vertical convention
        # from the original OpenCV-based RAP renderer, so flip once here.
        rgba = torch.flip(rgba[0], dims=[0])
        rgba = torch.clamp(rgba, 0.0, 1.0).mul(255.0).byte()
        if return_torch:
            return rgba
        return rgba.cpu().numpy()

    @staticmethod
    def _filter_triangle_batches(triangle_batches: List[LoweredTriangleBatch],
                                 allowed_source_kinds: set[str]) -> List[LoweredTriangleBatch]:
        return [batch for batch in triangle_batches if batch.source_kind in allowed_source_kinds]

    def _use_any_gpu_pass(self) -> bool:
        return any([
            self.use_gpu_cuboid_instances,
            self.use_gpu_cuboid_batches,
            self.use_gpu_filled_regions,
            self.use_gpu_line_overlays,
            self.use_gpu_arrows,
        ])

    def _perf_reset(self) -> None:
        self._perf_last = {}

    def _perf_add(self, key: str, ms: float) -> None:
        self._perf_last[key] = self._perf_last.get(key, 0.0) + float(ms)

    def get_perf_stats(self, reset: bool = False) -> Dict[str, float]:
        stats = dict(self._perf_last)
        if reset:
            self._perf_last = {}
        return stats

    @staticmethod
    def _composite_rgba_over_canvas(canvas: np.ndarray, rgba: Optional[np.ndarray]) -> np.ndarray:
        if rgba is None:
            return canvas
        out = canvas.copy()
        mask = rgba[..., 3] > 0
        out[mask] = rgba[..., :3][mask]
        return out

    def _composite_rgba_over_canvas_torch(self,
                                          canvas_rgb,
                                          rgba):
        if rgba is None:
            return canvas_rgb
        torch, _ = self._lazy_import_torch_raster()
        if canvas_rgb is None:
            canvas_rgb = torch.zeros(rgba.shape[0], rgba.shape[1], 3, dtype=torch.uint8, device=rgba.device)
        mask = rgba[..., 3] > 0
        canvas_rgb[mask] = rgba[..., :3][mask]
        return canvas_rgb

    def _to_numpy_rgb(self, canvas_rgb) -> np.ndarray:
        return canvas_rgb.detach().cpu().numpy()

    def _render_scene_packet_torch_stub(self, packet: CameraScenePacket):
        lowered = self._lower_scene_packet(packet)
        raise NotImplementedError(
            "Torch/nvdiffrast raster submission is not wired yet. "
            f"Lowered {lowered.total_triangle_count} triangles for camera {packet.camera.camera_id}."
        )

    def _build_camera_state(self, cam_id, cam_model, lidar_pos, lidar_yaw) -> CameraState:
        cam_t = cam_model["sensor2lidar_translation"].astype(np.float32).copy()
        cam_t[2] += 0.8
        cam_t[0] -= 2
        cam_R = cam_model["sensor2lidar_rotation"].astype(np.float32)
        K = cam_model["intrinsics"].astype(np.float32)
        T_w2c = world_to_camera_T(lidar_pos, lidar_yaw, cam_t, cam_R)
        return CameraState(
            camera_id=cam_id,
            width=self.width,
            height=self.height,
            depth_max=self.depth_max,
            K=K,
            T_w2c=T_w2c,
            cam_t=cam_t,
            cam_R=cam_R,
            lidar_pos=np.asarray(lidar_pos, dtype=np.float32),
        )

    def _append_traffic_light_primitives(self, packet: CameraScenePacket, traffic_lights) -> None:
        for feat in traffic_lights:
            is_red = feat[1]
            xy = feat[2]
            z_base = 5
            pos_world = np.array([xy[0], xy[1], z_base], dtype=np.float32)
            dims = np.array([0.5, 0.5, 1.0], dtype=np.float32)
            color_key = 'traffic_light_red' if is_red else 'traffic_light_green'
            packet.add_cuboid_instance(
                CuboidInstance(
                    semantic_tag="traffic_light",
                    center_world=pos_world,
                    dims=dims,
                    color_rgb=COLOR_TABLE[color_key].copy(),
                    thickness=-1,
                )
            )

    def _append_map_primitives(self,
                               packet: CameraScenePacket,
                               map_features: Dict[str, Dict[str, np.ndarray]],
                               lidar_pos: np.ndarray,
                               apply_distance_cull: bool = True) -> None:
        for feat in map_features.values():
            ftype = feat['type']
            if 'LANE' in ftype:
                poly2d = feat['polygon'].astype(np.float32)
                if apply_distance_cull:
                    pts_dist = np.linalg.norm(poly2d - lidar_pos[np.newaxis, :2], axis=1)
                    if np.min(pts_dist) > self.depth_max:
                        continue
                cull_center, cull_radius = self._polyline_cull_circle(poly2d)
                packet.add_line_overlay(
                    LineOverlay(
                        semantic_tag=ftype,
                        points_world=self._ground_points_from_poly2d(poly2d),
                        color=COLOR_TABLE['lanelines'].copy(),
                        radius=2,
                        depth_max=self.depth_max,
                        cull_center_xy=cull_center,
                        cull_radius_xy=cull_radius,
                    )
                )
            elif 'CROSSWALK' in ftype or 'SPEED_BUMP' in ftype:
                poly2d = feat['polygon'].astype(np.float32)
                packet.add_filled_region(
                    FilledRegion(
                        semantic_tag=ftype,
                        points_world=self._ground_points_from_poly2d(poly2d),
                        fill_color=COLOR_TABLE['crosswalks'].copy(),
                        depth_max=self.depth_max,
                        outline_color=COLOR_TABLE['lanelines'].copy(),
                    )
                )
            elif 'BOUNDARY' in ftype or 'SOLID' in ftype:
                poly2d = feat['polyline'].astype(np.float32)
                cull_center, cull_radius = self._polyline_cull_circle(poly2d)
                packet.add_line_overlay(
                    LineOverlay(
                        semantic_tag=ftype,
                        points_world=self._ground_points_from_poly2d(poly2d),
                        color=COLOR_TABLE['road_boundaries'].copy(),
                        radius=10,
                        depth_max=self.depth_max,
                        cull_center_xy=cull_center,
                        cull_radius_xy=cull_radius,
                    )
                )

    def _append_annotation_primitives(self, packet: CameraScenePacket, scenario) -> None:
        anns = scenario["anns"]
        packet.add_cuboid_batch(
            CuboidBatch(
                semantic_tag="vehicle_batch",
                bboxes_world=np.asarray(anns["gt_boxes_world"], dtype=np.float32),
                depth_max=self.depth_max,
            )
        )

    @staticmethod
    def _resolve_lidar_pose(scenario) -> tuple[np.ndarray, float]:
        ego_pose = scenario.get("ego_pose")
        if ego_pose is not None:
            pos = np.array(
                [
                    float(ego_pose.get("x", 0.0)),
                    float(ego_pose.get("y", 0.0)),
                    float(ego_pose.get("z", 0.0)),
                ],
                dtype=np.float32,
            )
            yaw = float(ego_pose.get("heading", scenario.get("ego_heading", 0.0)))
            return pos, yaw
        return np.zeros(3, dtype=np.float32), float(scenario["ego_heading"])

    @staticmethod
    def _resolve_map_features_for_scene(scenario) -> tuple[Dict[str, Dict[str, np.ndarray]], bool]:
        world_static = scenario.get("map_features_world_static")
        if world_static is not None:
            return world_static, True
        return scenario.get("map_features", {}), False

    def _make_static_cache_key(self, scenario) -> Tuple[Any, ...]:
        scene_id = scenario.get("scene_id", "__unknown_scene__")
        return (scene_id, "__shared__", self.width, self.height, float(self.depth_max), self.quality_mode)

    def _build_scene_template_packet(self,
                                     map_features: Dict[str, Dict[str, np.ndarray]],
                                     traffic_lights,
                                     lidar_pos: np.ndarray,
                                     apply_distance_cull: bool) -> CameraScenePacket:
        template_camera = CameraState(
            camera_id="__template__",
            width=self.width,
            height=self.height,
            depth_max=self.depth_max,
            K=np.eye(3, dtype=np.float32),
            T_w2c=np.eye(4, dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
            cam_R=np.eye(3, dtype=np.float32),
            lidar_pos=np.zeros(3, dtype=np.float32),
        )
        packet = CameraScenePacket(camera=template_camera)
        self._append_traffic_light_primitives(packet, traffic_lights)
        self._append_map_primitives(packet, map_features, lidar_pos, apply_distance_cull=apply_distance_cull)
        return packet

    def _get_or_build_scene_static_cache_entry(self,
                                               scenario,
                                               lidar_pos: np.ndarray,
                                               world_static_map: bool) -> SceneStaticCacheEntry:
        cache_key = self._make_static_cache_key(scenario)
        cached = self._scene_static_cache.get(cache_key)
        if cached is not None:
            return cached

        map_features, _ = self._resolve_map_features_for_scene(scenario)
        template = self._build_scene_template_packet(
            map_features=map_features,
            traffic_lights=scenario.get("traffic_lights", []),
            lidar_pos=lidar_pos,
            apply_distance_cull=not world_static_map,
        )
        entry = SceneStaticCacheEntry(
            line_overlays=template.line_overlays,
            filled_regions=template.filled_regions,
            cuboid_instances=template.cuboid_instances,
            render_order=template.render_order,
        )
        self._scene_static_cache[cache_key] = entry
        return entry

    def _build_camera_scene_packet(self,
                                   cam_id,
                                   cam_model,
                                   scenario,
                                   lidar_pos,
                                   lidar_yaw,
                                   scene_template_packet: Optional[SceneStaticCacheEntry] = None) -> CameraScenePacket:
        camera_state = self._build_camera_state(cam_id, cam_model, lidar_pos, lidar_yaw)
        if scene_template_packet is None:
            packet = CameraScenePacket(camera=camera_state)
            map_features, world_static_map = self._resolve_map_features_for_scene(scenario)
            self._append_traffic_light_primitives(packet, scenario.get("traffic_lights", []))
            self._append_map_primitives(packet, map_features, lidar_pos, apply_distance_cull=not world_static_map)
            self._append_annotation_primitives(packet, scenario)
            return packet
        packet = CameraScenePacket(
            camera=camera_state,
            line_overlays=list(scene_template_packet.line_overlays),
            filled_regions=list(scene_template_packet.filled_regions),
            cuboid_instances=list(scene_template_packet.cuboid_instances),
            cuboid_batches=[],
            arrow_overlays=[],
            render_order=list(scene_template_packet.render_order),
        )
        self._append_annotation_primitives(packet, scenario)
        return packet

    def _render_scene_packet_cpu(self,
                                 packet: CameraScenePacket,
                                 canvas: Optional[np.ndarray] = None,
                                 include_line_overlays: bool = True,
                                 include_filled_region_fill: bool = True,
                                 include_filled_region_outline: bool = True,
                                 include_cuboid_instances: bool = True,
                                 include_cuboid_batches: bool = True,
                                 include_arrows: bool = True) -> np.ndarray:
        if canvas is None:
            canvas = np.zeros((packet.camera.height, packet.camera.width, 3), dtype=np.uint8)
        else:
            canvas = canvas.copy()
        for op in packet.render_order:
            if op.kind == "cuboid_instance":
                if not include_cuboid_instances:
                    continue
                cuboid = packet.cuboid_instances[op.index]
                draw_cuboid_at(
                    canvas,
                    cuboid.center_world,
                    tuple(cuboid.dims.tolist()),
                    packet.camera.T_w2c,
                    packet.camera.K,
                    color_rgb=cuboid.color_rgb.tolist(),
                    thickness=cuboid.thickness,
                )
            elif op.kind == "line":
                if not include_line_overlays:
                    continue
                overlay = packet.line_overlays[op.index]
                draw_polyline_depth(
                    canvas,
                    overlay.points_world,
                    packet.camera.T_w2c,
                    packet.camera.K,
                    overlay.color,
                    radius=overlay.radius,
                    near=overlay.near,
                    depth_max=overlay.depth_max,
                )
            elif op.kind == "filled_region":
                region = packet.filled_regions[op.index]
                if include_filled_region_fill:
                    draw_polygon_depth(
                        canvas,
                        region.points_world,
                        packet.camera.T_w2c,
                        packet.camera.K,
                        region.fill_color,
                        region.depth_max,
                    )
                if include_filled_region_outline and region.outline_color is not None:
                    draw_polyline_depth(
                        canvas,
                        region.points_world,
                        packet.camera.T_w2c,
                        packet.camera.K,
                        region.outline_color,
                        radius=region.outline_radius,
                        near=region.outline_near,
                        depth_max=region.depth_max,
                    )
            elif op.kind == "cuboid_batch":
                if not include_cuboid_batches:
                    continue
                cuboid_batch = packet.cuboid_batches[op.index]
                draw_cuboids_with_occlusion(
                    canvas,
                    cuboid_batch.bboxes_world,
                    packet.camera.T_w2c,
                    packet.camera.K,
                    depth_max=cuboid_batch.depth_max,
                )
            elif op.kind == "arrow":
                if not include_arrows:
                    continue
                arrow = packet.arrow_overlays[op.index]
                draw_heading_arrow(
                    canvas,
                    arrow.pos_world,
                    arrow.yaw,
                    packet.camera.T_w2c,
                    packet.camera.K,
                    color_rgb=tuple(int(c) for c in arrow.color_rgb.tolist()),
                    arrow_len=arrow.arrow_len,
                    thickness=arrow.thickness,
                )
            else:
                raise ValueError(f"Unsupported render op kind: {op.kind}")
        return canvas

    def _render_scene_packet_hybrid(self, packet: CameraScenePacket) -> np.ndarray:
        use_fast_line_path = self.use_gpu_line_overlays and (self.quality_mode == "perf")
        t_lower = time.perf_counter()
        lowered = self._lower_scene_packet(
            packet,
            include_filled_regions=self.use_gpu_filled_regions,
            include_cuboid_instances=self.use_gpu_cuboid_instances,
            include_cuboid_batches=self.use_gpu_cuboid_batches,
            include_line_overlays=self.use_gpu_line_overlays and (not use_fast_line_path),
            include_arrows=self.use_gpu_arrows,
        )
        self._perf_add("lower_total_ms", (time.perf_counter() - t_lower) * 1000.0)
        canvas = np.zeros((packet.camera.height, packet.camera.width, 3), dtype=np.uint8)
        gpu_canvas = None

        if not self.use_gpu_filled_regions:
            canvas = self._render_scene_packet_cpu(
                packet,
                canvas=canvas,
                include_line_overlays=False,
                include_filled_region_fill=True,
                include_filled_region_outline=False,
                include_cuboid_instances=False,
                include_cuboid_batches=False,
                include_arrows=False,
            )

        gpu_cuboid_source_kinds = set()
        if self.use_gpu_cuboid_instances:
            gpu_cuboid_source_kinds.add("cuboid_instance")
        if self.use_gpu_cuboid_batches:
            gpu_cuboid_source_kinds.add("cuboid_batch")

        if not self.use_gpu_cuboid_instances:
            canvas = self._render_scene_packet_cpu(
                packet,
                canvas=canvas,
                include_line_overlays=False,
                include_filled_region_fill=False,
                include_filled_region_outline=False,
                include_cuboid_instances=True,
                include_cuboid_batches=False,
                include_arrows=False,
            )

        if not self.use_gpu_cuboid_batches:
            canvas = self._render_scene_packet_cpu(
                packet,
                canvas=canvas,
                include_line_overlays=False,
                include_filled_region_fill=False,
                include_filled_region_outline=False,
                include_cuboid_instances=False,
                include_cuboid_batches=True,
                include_arrows=False,
            )

        opaque_batches: List[LoweredTriangleBatch] = []
        if self.use_gpu_filled_regions:
            opaque_batches.extend(self._filter_triangle_batches(lowered.filled_triangle_batches, {"filled_region"}))
        if gpu_cuboid_source_kinds:
            opaque_batches.extend(self._filter_triangle_batches(lowered.cuboid_triangle_batches, gpu_cuboid_source_kinds))
        if opaque_batches:
            t_opaque = time.perf_counter()
            opaque_rgba = self._rasterize_solid_triangle_batches(opaque_batches, packet.camera, return_torch=True)
            self._perf_add("raster_opaque_ms", (time.perf_counter() - t_opaque) * 1000.0)
            gpu_canvas = self._composite_rgba_over_canvas_torch(gpu_canvas, opaque_rgba)

        if not self.use_gpu_line_overlays:
            canvas = self._render_scene_packet_cpu(
                packet,
                canvas=canvas,
                include_line_overlays=True,
                include_filled_region_fill=False,
                include_filled_region_outline=True,
                include_cuboid_instances=False,
                include_cuboid_batches=False,
                include_arrows=False,
            )

        if not self.use_gpu_arrows:
            canvas = self._render_scene_packet_cpu(
                packet,
                canvas=canvas,
                include_line_overlays=False,
                include_filled_region_fill=False,
                include_filled_region_outline=False,
                include_cuboid_instances=False,
                include_cuboid_batches=False,
                include_arrows=True,
            )

        if use_fast_line_path:
            fast_line_overlays = self._collect_line_overlays_for_gpu(packet)
            overlay_rgba_fast = self._rasterize_line_overlays_fast(fast_line_overlays, packet.camera)
            gpu_canvas = self._composite_rgba_over_canvas_torch(gpu_canvas, overlay_rgba_fast)

        overlay_source_kinds = set()
        if self.use_gpu_line_overlays and (not use_fast_line_path):
            overlay_source_kinds.add("line_overlay")
        if self.use_gpu_arrows:
            overlay_source_kinds.add("arrow_overlay")
        if overlay_source_kinds:
            overlay_batches = self._filter_triangle_batches(lowered.overlay_triangle_batches, overlay_source_kinds)
            t_overlay = time.perf_counter()
            overlay_rgba = self._rasterize_solid_triangle_batches(overlay_batches, packet.camera, return_torch=True)
            self._perf_add("raster_overlay_ms", (time.perf_counter() - t_overlay) * 1000.0)
            gpu_canvas = self._composite_rgba_over_canvas_torch(gpu_canvas, overlay_rgba)
        if gpu_canvas is not None:
            t_copy = time.perf_counter()
            if np.any(canvas):
                torch, _ = self._lazy_import_torch_raster()
                cpu_canvas_t = torch.as_tensor(canvas, dtype=torch.uint8, device=gpu_canvas.device)
                mask = gpu_canvas.any(dim=-1).bool()
                cpu_canvas_t[mask] = gpu_canvas[mask]
                out = self._to_numpy_rgb(cpu_canvas_t)
                self._perf_add("gpu_to_cpu_copy_ms", (time.perf_counter() - t_copy) * 1000.0)
                return out
            out = self._to_numpy_rgb(gpu_canvas)
            self._perf_add("gpu_to_cpu_copy_ms", (time.perf_counter() - t_copy) * 1000.0)
            return out
        return canvas

    def observe(self, scenario):
        self._perf_reset()
        t_obs = time.perf_counter()
        lidar_pos, lidar_yaw = self._resolve_lidar_pose(scenario)
        map_features, world_static_map = self._resolve_map_features_for_scene(scenario)
        scene_id = scenario.get("scene_id", "__unknown_scene__")
        scope = (scene_id, self.width, self.height, float(self.depth_max), self.quality_mode, bool(world_static_map))
        if self._cache_scope != scope:
            self._scene_static_cache.clear()
            self._line_points_torch_cache.clear()
            self._cache_scope = scope

        ret_dict = {}
        if world_static_map:
            t_static = time.perf_counter()
            scene_template_packet = self._get_or_build_scene_static_cache_entry(
                scenario=scenario,
                lidar_pos=lidar_pos,
                world_static_map=True,
            )
            self._perf_add("build_static_cache_ms", (time.perf_counter() - t_static) * 1000.0)
        else:
            t_static = time.perf_counter()
            template = self._build_scene_template_packet(
                map_features=map_features,
                traffic_lights=scenario.get("traffic_lights", []),
                lidar_pos=lidar_pos,
                apply_distance_cull=True,
            )
            scene_template_packet = SceneStaticCacheEntry(
                line_overlays=template.line_overlays,
                filled_regions=template.filled_regions,
                cuboid_instances=template.cuboid_instances,
                render_order=template.render_order,
            )
            self._perf_add("build_static_cache_ms", (time.perf_counter() - t_static) * 1000.0)
        for cam_id, cam_model in self.camera_models.items():
            t_build = time.perf_counter()
            packet = self._build_camera_scene_packet(
                cam_id,
                cam_model,
                scenario,
                lidar_pos,
                lidar_yaw,
                scene_template_packet=scene_template_packet,
            )
            self._perf_add("build_packet_ms", (time.perf_counter() - t_build) * 1000.0)
            if self._use_any_gpu_pass():
                ret_dict[cam_id] = self._render_scene_packet_hybrid(packet)
            else:
                ret_dict[cam_id] = self._render_scene_packet_cpu(packet)
            # import matplotlib.pyplot as plt
            # plt.imshow(canvas)
            # plt.show()
            #print("canvas shape", canvas.shape)
            # ret_dict[cam_id] = canvas
            # cuboids = []
            # for i in range(bboxes.shape[0]):
            #     info = bboxes[i]
            #     name = names[i]
            #     pos = info[:3]
            #     yaw = info[6]
            #     L = info[3]
            #     Wd = info[4]
            #     H_box = info[5]

            #     color = COLOR_TABLE['vehicle']

            #     corners_loc = vehicle_corners_local(L, Wd, H_box)
            #     R_yaw = yaw_to_rot(yaw)
            #     corners_world = (R_yaw @ corners_loc.T).T + pos
            #     cuboids.append(corners_world)

            # draw_scene_cuboids(canvas,
            #                     cuboids,
            #                     T_w2c, K, color)

            #     # if track["type"] == "BICYCLE":
            #     #     color = COLOR_TABLE['bicycle']
            #     # elif track["type"] == "PEDESTRIAN":
            #     #     color = COLOR_TABLE['pedestrian']
            #     # elif track["type"] == "VEHICLE":
            #     #     color = COLOR_TABLE['vehicle']
            #     #     Wd, H_box = H_box, Wd
            #     # else:

        self._perf_add("observe_total_ms", (time.perf_counter() - t_obs) * 1000.0)
        return ret_dict



def make_sky_ground_canvas(
    H, W,
    horizon=0.60,
    sky_top=(180,120,60),
    sky_horizon=(230,205,185),
    ground_far=(105,105,105),
    ground_near=(35,35,35),
    sun=None,                      # e.g. {'azim':0.65,'elev':0.22,'radius':70,'glow_sigma':150,'intensity':0.9}
    vignette_strength=0.22,
    noise_std=2,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    # ---------- 1) 按行生成天空/地面（明确通道维） ----------
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]   # (H,1,1)  0顶/1底
    sky_mask = (y < horizon).astype(np.float32)                  # (H,1,1)
    t_sky = np.clip(y / max(horizon, 1e-6), 0, 1)                # (H,1,1)
    t_gnd = np.clip((y - horizon) / max(1 - horizon, 1e-6), 0, 1)

    sky_top  = np.array(sky_top,  np.float32)[None, None, :]     # (1,1,3)
    sky_hori = np.array(sky_horizon, np.float32)[None, None, :]
    gnd_far  = np.array(ground_far,  np.float32)[None, None, :]
    gnd_near = np.array(ground_near, np.float32)[None, None, :]

    sky = (1 - t_sky) * sky_top + t_sky * sky_hori               # (H,1,3)
    gnd = (1 - t_gnd) * gnd_far + t_gnd * gnd_near               # (H,1,3)
    row_bg = sky_mask * sky + (1 - sky_mask) * gnd               # (H,1,3)

    # 扩展到整幅宽度，显式 (H,W,3)
    bg = np.broadcast_to(row_bg, (H, W, 3)).astype(np.float32).copy()

    # ---------- 2) 太阳（带柔光），用显式通道维 ----------
    if isinstance(sun, dict):
        az  = float(sun.get('azim', 0.5))       # 0左 1右
        el  = float(sun.get('elev', 0.18))      # 0顶 1底
        r   = int(sun.get('radius', 80))
        sgm = float(sun.get('glow_sigma', 140))
        inten = float(sun.get('intensity', 0.9))

        x0, y0 = int(W * az), int(H * el)
        yy, xx = np.ogrid[:H, :W]              # 省内存
        d2 = (xx - x0)**2 + (yy - y0)**2

        disk = (d2 <= r*r).astype(np.float32)[..., None]         # (H,W,1)
        glow = np.exp(-0.5 * d2 / (sgm*sgm)).astype(np.float32)[..., None]

        sun_col = np.array([255, 255, 255], np.float32)[None, None, :]  # (1,1,3)

        bg += disk * (sun_col - bg) * (0.8 * inten)
        bg += glow * (sun_col - bg) * (0.25 * inten)

    # ---------- 3) 暗角 ----------
    if vignette_strength > 0:
        nx = np.linspace(-1, 1, W, dtype=np.float32)
        ny = np.linspace(-1, 1, H, dtype=np.float32)
        xx, yy = np.meshgrid(nx, ny)
        rr = np.sqrt(xx*xx + yy*yy)
        vign = np.clip(1 - vignette_strength * rr*rr, 0.75, 1.0).astype(np.float32)
        bg *= vign[..., None]

    # ---------- 4) 轻噪声 ----------
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, (H, W, 3)).astype(np.float32)
        bg += noise

    return np.clip(bg, 0, 255).astype(np.uint8)
