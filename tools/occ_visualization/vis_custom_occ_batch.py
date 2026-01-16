#!/usr/bin/env python3
"""Batch visualize custom sparse occupancy in vis_occ_pair_single.py style.

This script loops over frames and calls the same Mayavi rendering logic as
`tools/occ_visualization/vis_custom_occ_single.py`.

Example:
  python tools/occ_visualization/vis_custom_occ_batch.py \
    --pcd_dir /home/nuvo/Downloads/L2_BEV_fisheye/lidar \
    --occ_dir data/occ_gt_release_v1_0/custom_demo/train/scene-0000_sem_batch51 \
    --out_dir test/occ_dump_quick/pcd_occ_sem_batch51/style_overlay \
    --start 0 --end 50
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np

# Allow running as a standalone script from anywhere.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.occ_visualization.vis_custom_occ_single import (  # type: ignore
    OccGridSpec,
    obtain_points_label,
    read_pcd_xyz,
    visualize_occ,
)


def _frame_name(i: int) -> str:
    return f"{i:06d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch visualize custom sparse occ (Mayavi cube + fixed camera)")
    ap.add_argument("--occ_dir", required=True, help="Directory containing *_occ.npy files")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--pcd_dir", default="", help="Optional directory containing .pcd files")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="Inclusive end frame index")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--max_points_plot",
        type=int,
        default=120000,
        help="Subsample PCD points for faster rendering when --pcd_dir is provided",
    )
    ap.add_argument(
        "--point_cloud_range",
        type=float,
        nargs=6,
        default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
    )
    ap.add_argument(
        "--occupancy_size",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    grid = OccGridSpec(
        point_cloud_range=tuple(float(x) for x in args.point_cloud_range),
        occupancy_size=tuple(float(x) for x in args.occupancy_size),
    )
    voxel_size = float(args.occupancy_size[0])

    for i in range(args.start, args.end + 1, args.stride):
        stem = _frame_name(i)
        occ_path = osp.join(args.occ_dir, f"{stem}_occ.npy")
        if not osp.exists(occ_path):
            print(f"[skip] missing occ: {occ_path}")
            continue

        pcd_xyz = None
        if args.pcd_dir:
            pcd_path = osp.join(args.pcd_dir, f"{stem}.pcd")
            if osp.exists(pcd_path):
                pcd_xyz = read_pcd_xyz(pcd_path)
                if pcd_xyz.shape[0] > args.max_points_plot:
                    sel = np.random.RandomState(0).choice(pcd_xyz.shape[0], args.max_points_plot, replace=False)
                    pcd_xyz = pcd_xyz[sel]
            else:
                print(f"[warn] missing pcd: {pcd_path}")

        occ = np.load(occ_path)
        if occ.ndim != 2 or occ.shape[1] != 2:
            print(f"[skip] bad occ shape {occ.shape} for {occ_path}")
            continue

        points, labels = obtain_points_label(occ, grid)
        out_path = osp.join(args.out_dir, f"{stem}.png")
        visualize_occ(points, labels, voxel_size, out_path, pcd_xyz=pcd_xyz)
        print(f"[ok] {stem} -> {out_path}")


if __name__ == "__main__":
    main()
