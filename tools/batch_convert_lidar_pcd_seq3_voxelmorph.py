#!/usr/bin/env python3
"""Batch-generate sliding-window (3-frame) voxel_morph OCC results (+ optional visualization).

Rule required by user:
  - Image #1 uses frames (000001,000002,000003) -> output center 000002
  - Image #2 uses frames (000002,000003,000004) -> output center 000003
  - ...
  - Total 32 outputs.

This script:
  1) runs tools/convert_lidar_pcd_sequence_to_occ.py with --fill_method voxel_morph
  2) optionally runs tools/vis_pcd_and_occ.py to dump png

It skips missing frames and prints a summary.

Example:
  python3 tools/batch_convert_lidar_pcd_seq3_voxelmorph.py \
    --lidar_dir /home/nuvo/Downloads/L2_BEV_fisheye/lidar \
    --label_dir /home/nuvo/Downloads/L2_BEV_fisheye/label
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    # Use the wrapper so batch stays stable even if the main converter filename changes.
    conv = repo_root / "tools" / "convert_lidar_pcd_seq3_voxelmorph.py"
    vis = repo_root / "tools" / "vis_pcd_and_occ.py"

    p = argparse.ArgumentParser()
    p.add_argument("--lidar_dir", required=True)
    p.add_argument("--label_dir", default=None)
    p.add_argument(
        "--out_dir",
        default=str(
            repo_root
            / "data"
            / "custom_occ"
            / "scene-0000_sem_seq3_morph_batch32"
        ),
    )
    p.add_argument(
        "--vis_dir",
        default=str(repo_root / "test" / "occ_dump_quick" / "pcd_occ_sem_seq3_morph_batch32"),
    )
    p.add_argument("--count", type=int, default=32)
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Output start id i. Output ids will be i..i+count-1. Each output i uses frames (i,i+1,i+2).",
    )

    p.add_argument("--morph_radius", type=int, default=1)
    p.add_argument("--morph_close_iters", type=int, default=1)
    p.add_argument("--make_vis", action="store_true", help="Also generate png via tools/vis_pcd_and_occ.py")
    p.add_argument("--view", choices=["2d", "3d"], default="3d")
    p.add_argument("--color_by_semantic", action="store_true", default=True)

    args = p.parse_args()

    lidar_dir = Path(args.lidar_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None
    out_dir = Path(args.out_dir)
    vis_dir = Path(args.vis_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    missing: list[tuple[str, list[str]]] = []
    ok_centers: list[str] = []

    for k in range(args.count):
        i = args.start + k
        out_id = f"{i:06d}"
        # To get frames (i,i+1,i+2) using a symmetric window=3, we set center=(i+1)
        center = i + 1
        center_id = f"{center:06d}"

        needed = [f"{i:06d}", f"{center:06d}", f"{(i+2):06d}"]

        cmd1 = [
            "python",
            str(conv),
            "--lidar_dir",
            str(lidar_dir),
            "--center_id",
            center_id,
            "--window",
            "3",
            "--out_dir",
            str(out_dir),
            "--fill_method",
            "voxel_morph",
            "--morph_radius",
            str(int(args.morph_radius)),
            "--morph_close_iters",
            str(int(args.morph_close_iters)),
        ]
        if label_dir is not None:
            cmd1 += ["--label_dir", str(label_dir)]

        r1 = run(cmd1)
        if r1.returncode != 0:
            # Could be missing input files or a processing error; record output for diagnosis.
            if k < 3:
                print(f"[fail] center={center_id}\n{r1.stdout}")
            missing.append((center_id, needed))
            continue

        # Converter writes using center_id filename; we rename to out_id to match required convention.
        occ_path = out_dir / f"{center_id}_occ.npy"
        flow_path = out_dir / f"{center_id}_flow.npy"
        inv_path = out_dir / f"{center_id}_occ_invalid.npy"
        if not occ_path.exists() or not flow_path.exists() or not inv_path.exists():
            missing.append((out_id, needed))
            continue

        # Rename/move to out_id
        (out_dir / f"{out_id}_occ.npy").write_bytes(occ_path.read_bytes())
        (out_dir / f"{out_id}_flow.npy").write_bytes(flow_path.read_bytes())
        (out_dir / f"{out_id}_occ_invalid.npy").write_bytes(inv_path.read_bytes())
        # Remove center-named intermediates to avoid confusion.
        try:
            occ_path.unlink()
            flow_path.unlink()
            inv_path.unlink()
        except Exception:
            pass

        ok_centers.append(out_id)

        if args.make_vis:
            out_png = vis_dir / f"{out_id}_{args.view}_semantic.png"
            cmd2 = [
                "python",
                str(vis),
                "--pcd",
                str(lidar_dir / f"{out_id}.pcd"),
                "--occ_npy",
                str(out_dir / f"{out_id}_occ.npy"),
                "--out",
                str(out_png),
                "--view",
                str(args.view),
            ]
            if args.color_by_semantic:
                cmd2 += ["--color_by_semantic"]
            r2 = run(cmd2)
            if r2.returncode != 0 or not out_png.exists():
                # Vis is optional: record but do not fail occ generation.
                missing.append((center_id + "(vis)", needed))

    print(f"Generated occ: {len(ok_centers)}/{args.count}")
    if missing:
        print(f"Skipped/failed: {len(missing)}")
        for cid, need in missing[:20]:
            print(f"  - {cid} needs {need}")


if __name__ == "__main__":
    main()
