#!/usr/bin/env python3
"""Convert raw LiDAR point clouds (.pcd) into Apollo-Vision-Net occ GT release format.

This produces 3 files per frame:
- <frame>_occ.npy: int32 [N, 2] where each row is [voxel_index, semantic_id]
- <frame>_flow.npy: float32 [N, 2] aligned with _occ rows (2D flow, filled with zeros here)
- <frame>_occ_invalid.npy: int64 [M] voxel indices considered invalid (empty by default)

Notes
- This is a *minimal* converter from point presence to occupancy (no semantic labels).
- semantic_id is set to `occupied_class_id` for all occupied voxels.
- Voxel grid is derived from point_cloud_range + occupancy_size (same as repo configs).

Example
    python tools/convert_lidar_pcd_to_occ.py \
      --input_pcd /home/nuvo/Downloads/L2_BEV_fisheye/lidar/000000.pcd \
      --out_dir data/occ_gt_release_v1_0/custom_demo/train/scene-0000 \
      --frame_id 000
"""

from __future__ import annotations

import argparse
import os
import json
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
        xdim = int((x_max - x_min) / sx)
        ydim = int((y_max - y_min) / sy)
        zdim = int((z_max - z_min) / sz)
        return xdim, ydim, zdim

    @property
    def voxel_num(self) -> int:
        xdim, ydim, zdim = self.dims_xyz
        return xdim * ydim * zdim


def read_pcd_xyz(pcd_path: str) -> np.ndarray:
    """Read PCD (ASCII) and return xyz as float32 [N,3].

    This supports the common ASCII PCD format used in many dumps.
    If your PCD is binary/compressed, convert it to ASCII or extend this reader.
    """

    with open(pcd_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PCD header: {pcd_path}")
            header_lines.append(line)
            if line.strip().upper().startswith(b"DATA"):
                data_line = line.strip().split()
                if len(data_line) != 2:
                    raise ValueError(f"Malformed DATA line in PCD: {line!r}")
                data_type = data_line[1].lower()
                break

        header = b"".join(header_lines).decode("utf-8", errors="ignore")

        def _get_value(key: str) -> Optional[str]:
            for hl in header.splitlines():
                if hl.startswith(key + " "):
                    return hl.split(" ", 1)[1].strip()
            return None

        fields = (_get_value("FIELDS") or "").split()
        if not fields:
            fields = (_get_value("FIELD") or "").split()  # sometimes FIELD
        if not fields:
            raise ValueError("PCD header missing FIELDS/FIELD")

        if data_type != b"ascii":
            raise NotImplementedError(
                f"Only ASCII PCD is supported for now; got DATA {data_type.decode()}"
            )

        # Load remaining bytes as text.
        data_bytes = f.read()
        text = data_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return np.zeros((0, 3), dtype=np.float32)

        data = np.loadtxt(text.splitlines(), dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]

    # Map to xyz.
    field_to_idx = {name: i for i, name in enumerate(fields)}
    if not all(k in field_to_idx for k in ("x", "y", "z")):
        raise ValueError(f"PCD fields missing x/y/z: {fields}")
    xyz = data[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]].astype(np.float32)
    return xyz


def voxelize_to_sparse_occ(
    xyz: np.ndarray,
    grid: OccGridSpec,
    occupied_class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize xyz into sparse occ array.

    Returns:
      occ (int32) [N,2] : [voxel_index, semantic_id]
      vox_indices (int64) [N] : voxel_index only

    Flatten convention used here:
      voxel_index = x + y * xdim + z * xdim * ydim
    (x fastest, then y, then z)

    This matches the common dense flatten order for tensors shaped (xdim, ydim, zdim).
    If your training expects a different order, adjust here and keep it consistent.
    """

    x_min, y_min, z_min, x_max, y_max, z_max = grid.point_cloud_range
    sx, sy, sz = grid.occupancy_size
    xdim, ydim, zdim = grid.dims_xyz

    # Compute voxel coordinates.
    x = np.floor((xyz[:, 0] - x_min) / sx).astype(np.int64)
    y = np.floor((xyz[:, 1] - y_min) / sy).astype(np.int64)
    z = np.floor((xyz[:, 2] - z_min) / sz).astype(np.int64)

    valid = (x >= 0) & (x < xdim) & (y >= 0) & (y < ydim) & (z >= 0) & (z < zdim)
    x, y, z = x[valid], y[valid], z[valid]

    vox = x + y * xdim + z * xdim * ydim
    vox = np.unique(vox)  # one label per voxel

    occ = np.empty((vox.shape[0], 2), dtype=np.int32)
    occ[:, 0] = vox.astype(np.int32)
    occ[:, 1] = np.int32(occupied_class_id)
    return occ, vox


def load_label_boxes(label_json_path: str) -> List[dict]:
    """Load a label json containing a list of objects with {obj_type, psr}.

    Expected item format (example):
      {
        "obj_id": "0",
        "obj_type": "Car",
        "psr": {
          "position": {"x": ..., "y": ..., "z": ...},
          "rotation": {"x": 0, "y": 0, "z": yaw},
          "scale": {"x": length, "y": width, "z": height}
        }
      }
    """
    with open(label_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Label JSON must be a list, got {type(data)}")
    return data


def points_in_oriented_boxes_mask(
    xyz: np.ndarray,
    boxes: List[dict],
) -> Tuple[np.ndarray, List[str]]:
    """Assign each point to a box index via point-in-OBB test.

    Returns:
      point_box_idx: int64 [N], -1 means background
      box_types: list[str] aligned with boxes

    Overlaps are resolved by preferring the smallest-volume box.
    """
    if len(boxes) == 0 or xyz.shape[0] == 0:
        return -np.ones((xyz.shape[0],), dtype=np.int64), []

    box_types: List[str] = []
    centers = []
    yaws = []
    half_sizes = []
    volumes = []
    for b in boxes:
        psr = b.get("psr", {})
        pos = psr.get("position", {})
        rot = psr.get("rotation", {})
        scl = psr.get("scale", {})
        cx, cy, cz = float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))
        yaw = float(rot.get("z", 0.0))
        lx, ly, lz = float(scl.get("x", 0.0)), float(scl.get("y", 0.0)), float(scl.get("z", 0.0))

        centers.append((cx, cy, cz))
        yaws.append(yaw)
        half_sizes.append((lx / 2.0, ly / 2.0, lz / 2.0))
        volumes.append(max(lx, 1e-6) * max(ly, 1e-6) * max(lz, 1e-6))
        box_types.append(str(b.get("obj_type", "Unknown")))

    centers = np.asarray(centers, dtype=np.float32)  # [B,3]
    yaws = np.asarray(yaws, dtype=np.float32)  # [B]
    half_sizes = np.asarray(half_sizes, dtype=np.float32)  # [B,3]
    volumes = np.asarray(volumes, dtype=np.float32)

    # Prefer smaller boxes on overlap.
    order = np.argsort(volumes)

    point_box_idx = -np.ones((xyz.shape[0],), dtype=np.int64)
    for bi in order:
        c = centers[bi]
        yaw = yaws[bi]
        hs = half_sizes[bi]

        # Transform points to box local frame: translate then rotate by -yaw.
        p = xyz - c[None, :]
        cos_y = float(np.cos(-yaw))
        sin_y = float(np.sin(-yaw))
        x_local = p[:, 0] * cos_y - p[:, 1] * sin_y
        y_local = p[:, 0] * sin_y + p[:, 1] * cos_y
        z_local = p[:, 2]

        inside = (
            (np.abs(x_local) <= hs[0])
            & (np.abs(y_local) <= hs[1])
            & (np.abs(z_local) <= hs[2])
        )
        # Only assign background points.
        inside = inside & (point_box_idx < 0)
        point_box_idx[inside] = int(bi)

    return point_box_idx, box_types


def default_label_map() -> Dict[str, int]:
    """A conservative mapping from dataset obj_type strings to occ semantic ids.

    The repo's `occupancy_classes=16` does not document class names here.
    For now, we map common traffic participants into a small id set.

    You can override via --label_map_json (a dict {"Car": 1, ...}).
    """
    return {
        "Car": 1,
        "Truck": 2,
        "Bus": 3,
        "Trimotorcycle": 4,
        "Motorcycle": 4,
        "Bicycle": 5,
        "Pedestrian": 6,
        "Unknown": 1,
    }


def build_sparse_occ_from_semantic_points(
    xyz: np.ndarray,
    point_semantic: np.ndarray,
    grid: OccGridSpec,
    fallback_class_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize points and assign a semantic id per voxel (majority vote).

    point_semantic: int64 [N], -1 means unlabeled.
    """
    x_min, y_min, z_min, x_max, y_max, z_max = grid.point_cloud_range
    sx, sy, sz = grid.occupancy_size
    xdim, ydim, zdim = grid.dims_xyz

    x = np.floor((xyz[:, 0] - x_min) / sx).astype(np.int64)
    y = np.floor((xyz[:, 1] - y_min) / sy).astype(np.int64)
    z = np.floor((xyz[:, 2] - z_min) / sz).astype(np.int64)
    valid = (x >= 0) & (x < xdim) & (y >= 0) & (y < ydim) & (z >= 0) & (z < zdim)
    x, y, z = x[valid], y[valid], z[valid]
    sem = point_semantic[valid]

    vox = x + y * xdim + z * xdim * ydim

    # Aggregate semantics per voxel.
    # We'll do: per-voxel majority over labeled points; if none labeled, use fallback_class_id.
    order = np.argsort(vox)
    vox_s = vox[order]
    sem_s = sem[order]

    uniq, start_idx = np.unique(vox_s, return_index=True)
    # end idx via next start.
    end_idx = np.r_[start_idx[1:], [vox_s.shape[0]]]

    out_sem = np.empty((uniq.shape[0],), dtype=np.int32)
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        sem_block = sem_s[s:e]
        sem_block = sem_block[sem_block >= 0]
        if sem_block.size == 0:
            out_sem[i] = np.int32(fallback_class_id)
        else:
            # majority vote
            vals, cnts = np.unique(sem_block, return_counts=True)
            out_sem[i] = np.int32(vals[np.argmax(cnts)])

    occ = np.empty((uniq.shape[0], 2), dtype=np.int32)
    occ[:, 0] = uniq.astype(np.int32)
    occ[:, 1] = out_sem
    return occ, uniq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pcd", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frame_id", default=None, help="Default uses input filename stem")
    parser.add_argument(
        "--label_json",
        default=None,
        help="Optional label json path (e.g., .../label/000000.json). If provided, semantic ids will be assigned by point-in-box.",
    )
    parser.add_argument(
        "--label_map_json",
        default=None,
        help="Optional json file containing mapping dict {obj_type: semantic_id}.",
    )
    parser.add_argument(
        "--occupied_class_id",
        type=int,
        default=1,
        help="Semantic id written into *_occ.npy for all occupied voxels",
    )

    # Defaults aligned to `projects/configs/bevformer/bev_tiny_det_occ_apollo.py`.
    parser.add_argument(
        "--point_cloud_range",
        type=float,
        nargs=6,
        default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    )
    parser.add_argument(
        "--occupancy_size",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
    )

    args = parser.parse_args()

    frame_id = args.frame_id
    if frame_id is None:
        frame_id = os.path.splitext(os.path.basename(args.input_pcd))[0]

    grid = OccGridSpec(
        point_cloud_range=tuple(float(x) for x in args.point_cloud_range),
        occupancy_size=tuple(float(x) for x in args.occupancy_size),
    )

    os.makedirs(args.out_dir, exist_ok=True)

    xyz = read_pcd_xyz(args.input_pcd)
    if args.label_json is not None:
        boxes = load_label_boxes(args.label_json)
        label_map: Dict[str, int] = default_label_map()
        if args.label_map_json is not None:
            with open(args.label_map_json, "r") as f:
                user_map = json.load(f)
            if not isinstance(user_map, dict):
                raise ValueError("--label_map_json must be a json dict {obj_type: id}")
            # Merge override.
            label_map.update({str(k): int(v) for k, v in user_map.items()})

        point_box_idx, box_types = points_in_oriented_boxes_mask(xyz, boxes)
        point_sem = -np.ones((xyz.shape[0],), dtype=np.int64)
        for bi, t in enumerate(box_types):
            sem_id = int(label_map.get(t, label_map.get("Unknown", args.occupied_class_id)))
            point_sem[point_box_idx == bi] = sem_id

        occ, _vox = build_sparse_occ_from_semantic_points(
            xyz,
            point_semantic=point_sem,
            grid=grid,
            fallback_class_id=args.occupied_class_id,
        )
    else:
        occ, _vox = voxelize_to_sparse_occ(xyz, grid=grid, occupied_class_id=args.occupied_class_id)

    # Flow: zeros, aligned with occ rows.
    flow = np.zeros((occ.shape[0], 2), dtype=np.float32)

    # Invalid: empty by default.
    occ_invalid = np.zeros((0,), dtype=np.int64)

    np.save(os.path.join(args.out_dir, f"{frame_id}_occ.npy"), occ)
    np.save(os.path.join(args.out_dir, f"{frame_id}_flow.npy"), flow)
    np.save(os.path.join(args.out_dir, f"{frame_id}_occ_invalid.npy"), occ_invalid)

    xdim, ydim, zdim = grid.dims_xyz
    print(
        "\n".join(
            [
                f"Wrote frame {frame_id} to {args.out_dir}",
                f"grid: xdim={xdim} ydim={ydim} zdim={zdim} voxel_num={grid.voxel_num}",
                f"occ: {occ.dtype} {occ.shape}",
                f"flow: {flow.dtype} {flow.shape}",
                f"invalid: {occ_invalid.dtype} {occ_invalid.shape}",
            ]
        )
    )


if __name__ == "__main__":
    main()
