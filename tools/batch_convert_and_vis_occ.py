#!/usr/bin/env python3
"""Batch convert LiDAR+labels to occ and visualize multiple frames.

This glues together:
- tools/convert_lidar_pcd_to_occ.py (conversion)
- tools/vis_pcd_and_occ.py (visualization)

It is intentionally lightweight and uses subprocess so it doesn't depend on the
internal functions of those scripts.

Example
  python tools/batch_convert_and_vis_occ.py \
    --lidar_dir /home/nuvo/Downloads/L2_BEV_fisheye/lidar \
    --label_dir /home/nuvo/Downloads/L2_BEV_fisheye/label \
    --frame_start 0 --frame_end 4 \
    --out_occ_dir data/occ_gt_release_v1_0/custom_demo/train/scene-0000_sem_batch \
    --out_vis_dir test/occ_dump_quick/pcd_occ_sem_batch \
    --view 3d --color_by semantic
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List


def _frame_ids(start: int, end: int, width: int) -> List[str]:
    return [str(i).zfill(width) for i in range(start, end + 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar_dir", required=True)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--frame_start", type=int, required=True)
    parser.add_argument("--frame_end", type=int, required=True)
    parser.add_argument("--frame_width", type=int, default=6)

    parser.add_argument("--out_occ_dir", required=True)
    parser.add_argument("--out_vis_dir", required=True)

    parser.add_argument("--view", choices=["topdown", "3d"], default="3d")
    parser.add_argument("--color_by", choices=["height", "semantic"], default="semantic")
    parser.add_argument("--max_points_plot", type=int, default=120000)

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    convert_py = os.path.join(repo_root, "tools", "convert_lidar_pcd_to_occ.py")
    vis_py = os.path.join(repo_root, "tools", "vis_pcd_and_occ.py")

    os.makedirs(args.out_occ_dir, exist_ok=True)
    os.makedirs(args.out_vis_dir, exist_ok=True)

    fids = _frame_ids(args.frame_start, args.frame_end, args.frame_width)
    for fid in fids:
        pcd = os.path.join(args.lidar_dir, f"{fid}.pcd")
        if not os.path.exists(pcd):
            print(f"[skip] missing pcd: {pcd}")
            continue

        label_json = None
        if args.label_dir is not None:
            cand = os.path.join(args.label_dir, f"{fid}.json")
            if os.path.exists(cand):
                label_json = cand
            else:
                print(f"[warn] missing label json: {cand} (will fallback to occupied_class_id)")

        out_scene_dir = args.out_occ_dir
        # Convert
        cmd = [
            "python3",
            convert_py,
            "--input_pcd",
            pcd,
            "--out_dir",
            out_scene_dir,
            "--frame_id",
            fid,
        ]
        if label_json is not None:
            cmd += ["--label_json", label_json]

        subprocess.run(cmd, check=True)

        occ_npy = os.path.join(out_scene_dir, f"{fid}_occ.npy")
        out_png = os.path.join(args.out_vis_dir, f"{fid}_{args.view}_{args.color_by}.png")

        # Visualize
        vcmd = [
            "python3",
            vis_py,
            "--pcd",
            pcd,
            "--occ_npy",
            occ_npy,
            "--out",
            out_png,
            "--view",
            args.view,
            "--max_points_plot",
            str(args.max_points_plot),
        ]
        if args.color_by == "semantic":
            vcmd += ["--color_by_semantic"]
        else:
            vcmd += ["--color_by_height"]

        subprocess.run(vcmd, check=True)

    print(f"Done. occ_dir={args.out_occ_dir} vis_dir={args.out_vis_dir}")


if __name__ == "__main__":
    main()
