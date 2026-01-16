#!/usr/bin/env python3
"""Visualize a custom sparse occupancy (.npy) in the same style as vis_occ_pair_single.py.

This is for quick apples-to-apples inspection: cube voxels, fixed camera, offscreen render.

Inputs:
  - --occ_npy: sparse occ file (int32 [N,2]) as produced by this repo's occ gt release format.
  - (optional) --pcd: ASCII PCD file; if provided, raw points are rendered as small spheres.

Output:
  - A single PNG.

Example:
  python tools/occ_visualization/vis_custom_occ_single.py \
    --occ_npy data/occ_gt_release_v1_0/custom_demo/train/scene-0000_sem_batch51/000000_occ.npy \
    --pcd /home/nuvo/Downloads/L2_BEV_fisheye/lidar/000000.pcd \
    --out test/occ_dump_quick/pcd_occ_sem_batch51/000000_occ_style.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Mayavi needs to be imported before matplotlib in some environments.
from mayavi import mlab

mlab.options.offscreen = True


NUM_CLASSES = 16
POINT_CLOUD_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]


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


def read_pcd_xyz(pcd_path: str) -> np.ndarray:
    # Keep consistent with tools/convert_lidar_pcd_to_occ.py
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
            fields = (_get_value("FIELD") or "").split()
        if not fields:
            raise ValueError("PCD header missing FIELDS/FIELD")

        if data_type != b"ascii":
            raise NotImplementedError(
                f"Only ASCII PCD is supported for now; got DATA {data_type.decode()}"
            )

        data_bytes = f.read()
        text = data_bytes.decode("utf-8", errors="ignore")
        if not text.strip():
            return np.zeros((0, 3), dtype=np.float32)

        data = np.loadtxt(text.splitlines(), dtype=np.float32)
        if data.ndim == 1:
            data = data[None, :]

    field_to_idx = {name: i for i, name in enumerate(fields)}
    if not all(k in field_to_idx for k in ("x", "y", "z")):
        raise ValueError(f"PCD fields missing x/y/z: {fields}")
    return data[:, [field_to_idx["x"], field_to_idx["y"], field_to_idx["z"]]].astype(np.float32)


def obtain_points_label(occ: np.ndarray, grid: OccGridSpec):
    occ_index, occ_cls = occ[:, 0].astype(np.int64), occ[:, 1].astype(np.int64)
    xdim, ydim, zdim = grid.dims_xyz

    # Keep consistent with tools/occ_visualization/vis_occ_pair_single.py
    x = occ_index % xdim
    y = (occ_index // xdim) % ydim
    z = occ_index // (xdim * ydim)

    x_min, y_min, z_min, x_max, y_max, z_max = grid.point_cloud_range

    point_x = (x.astype(np.float32) + 0.5) / xdim * (x_max - x_min) + x_min
    point_y = (y.astype(np.float32) + 0.5) / ydim * (y_max - y_min) + y_min
    point_z = (z.astype(np.float32) + 0.5) / zdim * (z_max - z_min) + z_min

    points = np.stack([point_x, point_y, point_z], axis=1)
    labels = occ_cls.astype(np.int64)
    return points, labels


def _fixed_lut_table() -> np.ndarray:
    # Copied from vis_occ_pair_single.py
    return np.array(
        [
            [255, 158, 0, 255],
            [255, 99, 71, 255],
            [255, 140, 0, 255],
            [255, 69, 0, 255],
            [233, 150, 70, 255],
            [220, 20, 60, 255],
            [255, 61, 99, 255],
            [0, 0, 230, 255],
            [47, 79, 79, 255],
            [112, 128, 144, 255],
            [0, 207, 191, 255],
            [175, 0, 75, 255],
            [75, 0, 75, 255],
            [112, 180, 60, 255],
            [222, 184, 135, 255],
            [0, 175, 0, 255],
            [0, 0, 0, 255],
        ],
        dtype=np.uint8,
    )


def visualize_occ(points: np.ndarray, labels: np.ndarray, voxel_size: float, out_path: str, *, pcd_xyz: Optional[np.ndarray]):
    # Map labels -> scalar values in [1..17] for LUT lookup.
    point_colors = np.zeros(points.shape[0], dtype=np.float32)
    for cls_index in range(NUM_CLASSES):
        class_point = labels == cls_index
        point_colors[class_point] = cls_index + 1

    fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    occ_plot = mlab.points3d(
        x,
        y,
        z,
        point_colors,
        scale_factor=voxel_size,
        mode="cube",
        scale_mode="vector",
        opacity=1.0,
        vmin=1,
        vmax=17,
    )
    occ_plot.module_manager.scalar_lut_manager.lut.table = _fixed_lut_table()

    if pcd_xyz is not None and pcd_xyz.size:
        # Raw points as small spheres for reference (subsampled for speed outside).
        mlab.points3d(
            pcd_xyz[:, 0],
            pcd_xyz[:, 1],
            pcd_xyz[:, 2],
            scale_factor=max(voxel_size * 0.2, 0.05),
            mode="sphere",
            color=(0.0, 0.0, 0.0),
            opacity=0.15,
        )

    # Fixed camera (copied from vis_occ_pair_single.py) with robust access.
    scene = fig.scene
    cam = getattr(scene, "camera", None)
    if cam is None and hasattr(scene, "scene"):
        cam = getattr(scene.scene, "camera", None)
    if cam is not None:
        cam.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
        cam.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
        cam.view_angle = 30.0
        cam.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
        cam.clipping_range = [0.18978054185107493, 189.78054185107493]
        if hasattr(cam, "compute_view_plane_normal"):
            cam.compute_view_plane_normal()

    if hasattr(scene, "render"):
        scene.render()
    elif hasattr(scene, "scene") and hasattr(scene.scene, "render"):
        scene.scene.render()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mlab.savefig(out_path)
    mlab.close(all=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize custom sparse occupancy in vis_occ_pair_single style")
    ap.add_argument("--occ_npy", required=True, help="Sparse occ npy (int32 [N,2])")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--pcd", default="", help="Optional ASCII PCD for overlay")
    ap.add_argument(
        "--max_points_plot",
        type=int,
        default=120000,
        help="Subsample PCD points for faster rendering when --pcd is provided",
    )
    ap.add_argument(
        "--point_cloud_range",
        type=float,
        nargs=6,
        default=POINT_CLOUD_RANGE,
    )
    ap.add_argument(
        "--occupancy_size",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Coarse occ by default (0.5m) to match occ_gt_release_v1_0",
    )
    args = ap.parse_args()

    occ = np.load(args.occ_npy)
    if occ.ndim != 2 or occ.shape[1] != 2:
        raise ValueError(f"occ_npy should be [N,2], got {occ.shape}")

    grid = OccGridSpec(
        point_cloud_range=tuple(float(x) for x in args.point_cloud_range),
        occupancy_size=tuple(float(x) for x in args.occupancy_size),
    )

    points, labels = obtain_points_label(occ, grid)

    pcd_xyz = None
    if args.pcd:
        pcd_xyz = read_pcd_xyz(args.pcd)
        if pcd_xyz.shape[0] > args.max_points_plot:
            sel = np.random.RandomState(0).choice(pcd_xyz.shape[0], args.max_points_plot, replace=False)
            pcd_xyz = pcd_xyz[sel]

    voxel_size = float(args.occupancy_size[0])
    visualize_occ(points, labels, voxel_size, args.out, pcd_xyz=pcd_xyz)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
