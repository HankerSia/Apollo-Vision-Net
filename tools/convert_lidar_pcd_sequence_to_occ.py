#!/usr/bin/env python3
"""Convert a short LiDAR sequence (default 3 frames) into Apollo-Vision-Net OCC GT release format.

This script was re-created to support the workflow:
  multi-frame lidar -> aggregate (static + dynamic center) -> voxel-space morph fill -> sparse OCC.

Outputs (for the center frame id):
  - <center>_occ.npy: int32 [N,2]  (voxel_index, semantic_id)
  - <center>_flow.npy: float32 [N,2] (zeros placeholder)
  - <center>_occ_invalid.npy: int64 [M] (empty placeholder)

Grid defaults match BEVFormer OCC settings in this repo:
  point_cloud_range = [-50,-50,-5, 50,50,3]
  occupancy_size    = [0.5,0.5,0.5] -> dims 200x200x16

Notes:
  - No ego-motion/pose compensation is applied (dataset appears pose-less), so multi-frame stacking is naive.
  - If label_dir (json 3D OBB) is provided, points inside boxes are treated as dynamic;
    dynamic points are kept only from the center frame to avoid ghosting.
  - voxel_morph fill operates in voxel mask space (closing: dilation->erosion).
    - IMPORTANT: Do NOT write into data/occ_gt_release_v1_0 (that's the GT dataset). Use a separate out_dir.

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OccGridSpec:
    point_cloud_range: Tuple[float, float, float, float, float, float]
    occupancy_size: Tuple[float, float, float]

    @property
    def dims_xyz(self) -> Tuple[int, int, int]:
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        sx, sy, sz = self.occupancy_size
        xdim = int(round((x_max - x_min) / sx))
        ydim = int(round((y_max - y_min) / sy))
        zdim = int(round((z_max - z_min) / sz))
        return xdim, ydim, zdim


def read_pcd_xyz(pcd_path: str) -> np.ndarray:
    """Read ASCII PCD with x y z fields."""
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_i = None
    fields = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        if s.startswith("DATA"):
            data_i = i + 1
            break

    if data_i is None:
        raise ValueError(f"PCD missing DATA line: {pcd_path}")
    if fields is None:
        raise ValueError(f"PCD missing FIELDS line: {pcd_path}")

    try:
        xi = fields.index("x")
        yi = fields.index("y")
        zi = fields.index("z")
    except ValueError:
        raise ValueError(f"PCD does not contain x/y/z fields: {fields}")

    pts: List[List[float]] = []
    for ln in lines[data_i:]:
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) <= max(xi, yi, zi):
            continue
        try:
            pts.append([float(parts[xi]), float(parts[yi]), float(parts[zi])])
        except ValueError:
            continue

    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


@dataclass
class OrientedBox:
    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) (dx,dy,dz)
    yaw: float  # radians
    obj_type: str


def load_label_boxes(label_json_path: str) -> List[OrientedBox]:
    with open(label_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes: List[OrientedBox] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("objects", data.get("labels", []))
    else:
        items = []
    if not isinstance(items, list):
        items = []

    for obj in items:
        # L2_BEV_fisheye schema: {obj_type, psr:{position, rotation, scale}}
        psr = obj.get("psr", {}) if isinstance(obj, dict) else {}
        pos = psr.get("position") if isinstance(psr, dict) else None
        scale = psr.get("scale") if isinstance(psr, dict) else None
        rot = psr.get("rotation") if isinstance(psr, dict) else None

        # Fallback to older schemas if present.
        if pos is None:
            pos = obj.get("position", obj.get("center", obj.get("translation")))
        if scale is None:
            scale = obj.get("scale", obj.get("size", obj.get("dimensions")))
        if rot is None:
            rot = obj.get("rotation", obj.get("rotation_z", obj.get("yaw")))
        if pos is None or scale is None:
            continue

        if isinstance(rot, dict):
            yaw = float(rot.get("z", rot.get("yaw", 0.0)))
        else:
            yaw = float(rot) if rot is not None else 0.0

        center = np.asarray([pos["x"], pos["y"], pos.get("z", 0.0)] if isinstance(pos, dict) else pos, dtype=np.float32)
        size = np.asarray([scale["x"], scale["y"], scale.get("z", 0.0)] if isinstance(scale, dict) else scale, dtype=np.float32)
        obj_type = str(obj.get("obj_type", obj.get("type", obj.get("category", "Unknown"))))
        boxes.append(OrientedBox(center=center, size=size, yaw=yaw, obj_type=obj_type))

    return boxes


def points_in_oriented_boxes_mask(points_xyz: np.ndarray, boxes: List[OrientedBox]) -> Tuple[np.ndarray, List[str]]:
    """Return box index per point (-1 if none) and list of box types."""
    if points_xyz.shape[0] == 0 or not boxes:
        return np.full((points_xyz.shape[0],), -1, dtype=np.int32), []

    out = np.full((points_xyz.shape[0],), -1, dtype=np.int32)
    types = [b.obj_type for b in boxes]

    px = points_xyz[:, 0]
    py = points_xyz[:, 1]
    pz = points_xyz[:, 2]

    for bi, b in enumerate(boxes):
        cx, cy, cz = b.center.tolist()
        dx, dy, dz = b.size.tolist()
        hx, hy, hz = dx * 0.5, dy * 0.5, dz * 0.5

        c = math.cos(-b.yaw)
        s = math.sin(-b.yaw)

        x = px - cx
        y = py - cy
        z = pz - cz
        xr = x * c - y * s
        yr = x * s + y * c

        inside = (np.abs(xr) <= hx) & (np.abs(yr) <= hy) & (np.abs(z) <= hz)
        out[inside] = bi

    return out, types


def voxelize_sparse_with_semantic(
    xyz: np.ndarray,
    sem: np.ndarray,
    *,
    grid: OccGridSpec,
    fallback_class_id: int,
) -> np.ndarray:
    """Voxelize points to sparse occ with majority-vote semantic per voxel."""
    if xyz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    x_min, y_min, z_min, x_max, y_max, z_max = grid.point_cloud_range
    sx, sy, sz = grid.occupancy_size
    xdim, ydim, zdim = grid.dims_xyz

    # Range filter
    m = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] < x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] < y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] < z_max)
    )
    xyz = xyz[m]
    sem = sem[m]
    if xyz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    ix = np.floor((xyz[:, 0] - x_min) / sx).astype(np.int64)
    iy = np.floor((xyz[:, 1] - y_min) / sy).astype(np.int64)
    iz = np.floor((xyz[:, 2] - z_min) / sz).astype(np.int64)

    valid = (ix >= 0) & (ix < xdim) & (iy >= 0) & (iy < ydim) & (iz >= 0) & (iz < zdim)
    ix, iy, iz, sem = ix[valid], iy[valid], iz[valid], sem[valid]
    if ix.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    vox = ix + iy * xdim + iz * xdim * ydim

    # Majority vote
    order = np.argsort(vox)
    vox = vox[order]
    sem = sem[order]

    uniq_vox, start = np.unique(vox, return_index=True)
    out_sem = np.empty((uniq_vox.shape[0],), dtype=np.int32)
    start = start.tolist() + [vox.shape[0]]
    for i in range(len(start) - 1):
        a, b = start[i], start[i + 1]
        seg = sem[a:b]
        if seg.size == 0:
            out_sem[i] = int(fallback_class_id)
        else:
            vals, cnt = np.unique(seg.astype(np.int32), return_counts=True)
            out_sem[i] = int(vals[int(np.argmax(cnt))])

    occ = np.empty((uniq_vox.shape[0], 2), dtype=np.int32)
    occ[:, 0] = uniq_vox.astype(np.int32)
    occ[:, 1] = out_sem.astype(np.int32)
    return occ


def _sparse_to_dense_mask(vox_idx: np.ndarray, grid: OccGridSpec) -> np.ndarray:
    xdim, ydim, zdim = grid.dims_xyz
    mask = np.zeros((zdim, ydim, xdim), dtype=bool)
    if vox_idx.size == 0:
        return mask
    idx = vox_idx.astype(np.int64)
    x = idx % xdim
    y = (idx // xdim) % ydim
    z = idx // (xdim * ydim)
    valid = (x >= 0) & (x < xdim) & (y >= 0) & (y < ydim) & (z >= 0) & (z < zdim)
    mask[z[valid], y[valid], x[valid]] = True
    return mask


def _dense_mask_to_sparse(mask: np.ndarray, grid: OccGridSpec) -> np.ndarray:
    xdim, ydim, _ = grid.dims_xyz
    zz, yy, xx = np.nonzero(mask)
    vox = (xx + yy * xdim + zz * xdim * ydim).astype(np.int64)
    return np.unique(vox)


def _dilate3d(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    zdim, ydim, xdim = mask.shape
    out = mask.copy()
    for dz in range(-radius, radius + 1):
        z0 = max(0, dz)
        z1 = min(zdim, zdim + dz)
        sz0 = max(0, -dz)
        sz1 = min(zdim, zdim - dz)
        for dy in range(-radius, radius + 1):
            y0 = max(0, dy)
            y1 = min(ydim, ydim + dy)
            sy0 = max(0, -dy)
            sy1 = min(ydim, ydim - dy)
            for dx in range(-radius, radius + 1):
                x0 = max(0, dx)
                x1 = min(xdim, xdim + dx)
                sx0 = max(0, -dx)
                sx1 = min(xdim, xdim - dx)
                out[z0:z1, y0:y1, x0:x1] |= mask[sz0:sz1, sy0:sy1, sx0:sx1]
    return out


def _erode3d(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return ~_dilate3d(~mask, radius)


def voxel_morph_fill(
    *,
    xyz: np.ndarray,
    sem: np.ndarray,
    grid: OccGridSpec,
    radius: int,
    close_iters: int,
    fallback_class_id: int,
) -> np.ndarray:
    """Return sparse occ after voxel-space closing.

    Semantics:
      - original occupied voxels keep their majority-vote semantic
      - newly-added filled voxels use fallback_class_id
    """
    occ_orig = voxelize_sparse_with_semantic(xyz, sem, grid=grid, fallback_class_id=fallback_class_id)
    vox_orig = occ_orig[:, 0].astype(np.int64)
    sem_orig = occ_orig[:, 1].astype(np.int32)

    mask = _sparse_to_dense_mask(vox_orig, grid)
    filled = mask
    for _ in range(max(1, int(close_iters))):
        filled = _erode3d(_dilate3d(filled, radius), radius)

    vox_filled = _dense_mask_to_sparse(filled, grid)

    sem_map = {int(v): int(s) for v, s in zip(vox_orig.tolist(), sem_orig.tolist())}
    out_sem = np.array([sem_map.get(int(v), int(fallback_class_id)) for v in vox_filled], dtype=np.int32)
    occ = np.empty((vox_filled.shape[0], 2), dtype=np.int32)
    occ[:, 0] = vox_filled.astype(np.int32)
    occ[:, 1] = out_sem.astype(np.int32)
    return occ


def _parse_label_map(path: Optional[str]) -> Dict[str, int]:
    if not path:
        # Default mapping (can be overridden by --label_map_json)
        return {
            "Unknown": 1,
            "Car": 2,
            "Truck": 3,
            "Bus": 4,
            "Motorcycle": 5,
            "Bicycle": 6,
            "Pedestrian": 7,
            "Tricycle": 8,
            "Trimotorcycle": 8,
        }
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lidar_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--center_id", required=True, help="Center frame id, e.g. 000040")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--label_dir", default=None)
    p.add_argument("--label_map_json", default=None)
    p.add_argument("--occupied_class_id", type=int, default=1)

    p.add_argument(
        "--point_cloud_range",
        type=float,
        nargs=6,
        default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    )
    p.add_argument(
        "--occupancy_size",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
    )

    p.add_argument("--fill_method", choices=["none", "voxel_morph"], default="voxel_morph")
    p.add_argument("--morph_radius", type=int, default=1)
    p.add_argument("--morph_close_iters", type=int, default=1)

    args = p.parse_args()

    lidar_dir = args.lidar_dir
    label_dir = args.label_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    center = int(args.center_id)
    half = args.window // 2

    # frame ids: clamp >=0 (avoid negative names)
    frames = []
    for t in range(-half, half + 1):
        fid = center + t * int(args.stride)
        if fid < 0:
            continue
        frames.append(fid)

    label_map = _parse_label_map(args.label_map_json)
    grid = OccGridSpec(tuple(args.point_cloud_range), tuple(args.occupancy_size))

    static_pts: List[np.ndarray] = []
    static_sem: List[np.ndarray] = []
    dynamic_pts_center: Optional[np.ndarray] = None
    dynamic_sem_center: Optional[np.ndarray] = None

    for fid in frames:
        fid_s = f"{fid:06d}"
        pcd_path = os.path.join(lidar_dir, f"{fid_s}.pcd")
        if not os.path.exists(pcd_path):
            continue

        xyz = read_pcd_xyz(pcd_path)
        if xyz.shape[0] == 0:
            continue

        sem = np.full((xyz.shape[0],), int(args.occupied_class_id), dtype=np.int32)
        dynamic_mask = np.zeros((xyz.shape[0],), dtype=bool)

        if label_dir is not None:
            label_path = os.path.join(label_dir, f"{fid_s}.json")
            if os.path.exists(label_path):
                boxes = load_label_boxes(label_path)
                box_idx, box_types = points_in_oriented_boxes_mask(xyz, boxes)
                for bi, t in enumerate(box_types):
                    sid = int(label_map.get(t, label_map.get("Unknown", args.occupied_class_id)))
                    sem[box_idx == bi] = sid
                dynamic_mask = box_idx >= 0

        static_mask = ~dynamic_mask
        if static_mask.any():
            static_pts.append(xyz[static_mask])
            static_sem.append(sem[static_mask])

        if fid == center and dynamic_mask.any():
            dynamic_pts_center = xyz[dynamic_mask]
            dynamic_sem_center = sem[dynamic_mask]

    if len(static_pts) == 0:
        raise RuntimeError("No points found in the requested window.")

    xyz_static = np.concatenate(static_pts, axis=0)
    sem_static = np.concatenate(static_sem, axis=0)

    if dynamic_pts_center is not None and dynamic_pts_center.size:
        xyz_all = np.concatenate([xyz_static, dynamic_pts_center], axis=0)
        sem_all = np.concatenate([sem_static, dynamic_sem_center], axis=0)
    else:
        xyz_all = xyz_static
        sem_all = sem_static

    if args.fill_method == "voxel_morph":
        occ = voxel_morph_fill(
            xyz=xyz_all,
            sem=sem_all,
            grid=grid,
            radius=int(args.morph_radius),
            close_iters=int(args.morph_close_iters),
            fallback_class_id=int(args.occupied_class_id),
        )
    else:
        occ = voxelize_sparse_with_semantic(xyz_all, sem_all, grid=grid, fallback_class_id=int(args.occupied_class_id))

    # flow/invalid placeholders
    flow = np.zeros((occ.shape[0], 2), dtype=np.float32)
    occ_invalid = np.zeros((0,), dtype=np.int64)

    out_prefix = os.path.join(out_dir, f"{args.center_id}")
    np.save(out_prefix + "_occ.npy", occ)
    np.save(out_prefix + "_flow.npy", flow)
    np.save(out_prefix + "_occ_invalid.npy", occ_invalid)

    xdim, ydim, zdim = grid.dims_xyz
    print(f"Wrote aggregated frame {args.center_id} using window={args.window} stride={args.stride}")
    print(f"frames: {[f'{f:06d}' for f in frames]}")
    print(f"grid: xdim={xdim} ydim={ydim} zdim={zdim}")
    print(f"points: static={xyz_static.shape[0]} total={xyz_all.shape[0]}")
    print(f"occ: {occ.dtype} {occ.shape}")


if __name__ == "__main__":
    main()
