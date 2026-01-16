#!/usr/bin/env python3
"""Visualize an input LiDAR PCD and its converted sparse occupancy.

Outputs a single PNG with 2 panels:
- Left: raw point cloud XY (top-down)
- Right: occupied voxels XY (top-down), optionally colored by height (z bin)

This is meant for quick sanity checks of `tools/convert_lidar_pcd_to_occ.py` output.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def voxel_index_to_xyz(voxel_index: np.ndarray, grid: OccGridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flattened voxel index -> integer voxel coords (x, y, z)."""
    xdim, ydim, zdim = grid.dims_xyz
    # Keep consistent with tools/occ_visualization/vis_occ_pair_single.py:
    #   x = idx % occ_xdim
    #   y = (idx // occ_xdim) % occ_ydim
    #   z = idx // (occ_xdim * occ_ydim)
    x = voxel_index % xdim
    y = (voxel_index // xdim) % ydim
    z = voxel_index // (xdim * ydim)
    return x, y, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", required=True)
    parser.add_argument("--occ_npy", required=True)
    parser.add_argument("--out", required=True, help="Output PNG path")
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
    parser.add_argument(
        "--max_points_plot",
        type=int,
        default=120000,
        help="Subsample points for faster plotting",
    )
    parser.add_argument(
        "--color_by_height",
        action="store_true",
        help="Color occupied voxels by z bin",
    )
    parser.add_argument(
        "--color_by_semantic",
        action="store_true",
        help="Color occupied voxels by semantic id (occ[:,1]) with a discrete colormap.",
    )

    parser.add_argument(
        "--semantic_legend_json",
        default=None,
        help=(
            "Optional JSON mapping to show legend labels for semantic ids. "
            "Accepts either {\"Name\": id} or {\"id\": \"Name\"}."
        ),
    )
    parser.add_argument(
        "--view",
        choices=["topdown", "3d"],
        default="topdown",
        help="Visualization view. topdown=2D XY. 3d=3D scatter with an oblique (~45°) camera.",
    )

    parser.add_argument(
        "--backend",
        choices=["matplotlib", "mayavi"],
        default="matplotlib",
        help=(
            "3D backend. matplotlib=default 3D scatter. mayavi=voxel cubes + fixed camera "
            "(aligned with tools/occ_visualization/vis_occ_pair_single.py)."
        ),
    )

    parser.add_argument(
        "--overlay_pcd",
        action="store_true",
        help="When using --view 3d, overlay the raw PCD points together with OCC in the same 3D view.",
    )

    parser.add_argument(
        "--show_axes",
        action="store_true",
        help="Show 3D axes (supported for both matplotlib and mayavi backends).",
    )

    parser.add_argument(
        "--layout",
        choices=["overlay", "side_by_side"],
        default="side_by_side",
        help=(
            "Layout. For matplotlib backend, default is side-by-side (left PCD, right OCC), "
            "matching the original script behavior. For mayavi backend, side_by_side is also supported."
        ),
    )

    parser.add_argument(
        "--camera",
        choices=["fixed", "45deg"],
        default="45deg",
        help=(
            "Camera preset for --view 3d. 45deg uses (elev,azim) for matplotlib and a generic "
            "oblique view for mayavi. fixed matches vis_occ_pair_single.py camera."
        ),
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=35.0,
        help="3D view elevation angle in degrees (matplotlib view_init).",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D view azimuth angle in degrees (matplotlib view_init).",
    )

    args = parser.parse_args()

    def _load_semantic_legend(path: Optional[str]) -> Dict[int, str]:
        if not path:
            # Default legend for our L2_BEV_fisheye mapping used by tools/convert_lidar_pcd_sequence_to_occ.py
            return {
                1: "Unknown",
                2: "Car",
                3: "Truck",
                4: "Bus",
                5: "Motorcycle",
                6: "Bicycle",
                7: "Pedestrian",
                8: "Tricycle",
            }
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out: Dict[int, str] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                # {name: id}
                if isinstance(v, (int, float)):
                    out[int(v)] = str(k)
                else:
                    # {id: name}
                    try:
                        out[int(k)] = str(v)
                    except Exception:
                        continue
        return out

    id_to_name = _load_semantic_legend(args.semantic_legend_json)

    if args.backend == "mayavi" and args.view != "3d":
        raise ValueError("--backend mayavi currently supports only --view 3d")

    grid = OccGridSpec(
        point_cloud_range=tuple(float(x) for x in args.point_cloud_range),
        occupancy_size=tuple(float(x) for x in args.occupancy_size),
    )
    x_min, y_min, z_min, x_max, y_max, z_max = grid.point_cloud_range
    sx, sy, sz = grid.occupancy_size

    xyz = read_pcd_xyz(args.pcd)
    # Clip points to the configured range for consistent visualization.
    if xyz.size:
        in_range = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] <= x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] <= y_max)
            & (xyz[:, 2] >= z_min)
            & (xyz[:, 2] <= z_max)
        )
        xyz = xyz[in_range]
    if xyz.shape[0] > args.max_points_plot:
        sel = np.random.RandomState(0).choice(xyz.shape[0], args.max_points_plot, replace=False)
        xyz_plot = xyz[sel]
    else:
        xyz_plot = xyz

    occ = np.load(args.occ_npy)
    if occ.ndim != 2 or occ.shape[1] != 2:
        raise ValueError(f"occ_npy should be [N,2], got {occ.shape}")
    vox = occ[:, 0].astype(np.int64)
    sem_id = occ[:, 1].astype(np.int64)
    vx, vy, vz = voxel_index_to_xyz(vox, grid)

    # Clip OCC voxels to valid dims (defensive) before converting to world coords.
    xdim, ydim, zdim = grid.dims_xyz
    occ_in = (
        (vx >= 0)
        & (vx < xdim)
        & (vy >= 0)
        & (vy < ydim)
        & (vz >= 0)
        & (vz < zdim)
    )
    if not np.all(occ_in):
        vx = vx[occ_in]
        vy = vy[occ_in]
        vz = vz[occ_in]
        sem_id = sem_id[occ_in]

    # Convert voxel coords -> voxel center world coords.
    occ_x = x_min + (vx.astype(np.float32) + 0.5) * sx
    occ_y = y_min + (vy.astype(np.float32) + 0.5) * sy
    occ_z = z_min + (vz.astype(np.float32) + 0.5) * sz

    if args.color_by_height and args.color_by_semantic:
        raise ValueError("Choose only one of --color_by_height or --color_by_semantic")

    if args.backend == "mayavi":
        # Render with the same cube-style and fixed camera used in
        # tools/occ_visualization/vis_occ_pair_single.py for apples-to-apples comparisons.
        # Note: Mayavi must be imported before matplotlib in some environments.
        from mayavi import mlab

        mlab.options.offscreen = True

        occ_colors_map = np.array(
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

        # Mayavi expects scalar per point for LUT indexing; we follow vis_occ_pair_single.py:
        # point_colors in [1..17].
        if args.color_by_semantic:
            # semantic ids are assumed in [0..15] (16 classes). Clamp for safety.
            c = np.clip(sem_id.astype(np.int64), 0, 15).astype(np.float32) + 1.0
        elif args.color_by_height:
            # Use height bin as pseudo-class, mapped into [1..17] for a stable LUT.
            c = (np.clip(vz.astype(np.int64), 0, 15) % 16).astype(np.float32) + 1.0
        else:
            c = np.ones_like(occ_x, dtype=np.float32)

        def _apply_camera(fig_obj):
            scene = fig_obj.scene
            cam = getattr(scene, "camera", None)
            if cam is None and hasattr(scene, "scene"):
                cam = getattr(scene.scene, "camera", None)

            # Force a consistent bounding box so auto distance/framing doesn't differ
            # between left (PCD) and right (OCC) figures.
            try:
                scene.reset_zoom()
            except Exception:
                pass

            if args.camera == "fixed":
                if cam is not None:
                    cam.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
                    cam.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
                    cam.view_angle = 30.0
                    cam.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
                    cam.clipping_range = [0.18978054185107493, 189.78054185107493]
                    if hasattr(cam, "compute_view_plane_normal"):
                        cam.compute_view_plane_normal()
            else:
                # Use a deterministic oblique view with an explicit focal point and distance
                # derived from the BEV range, so left/right look identical.
                try:
                    x0 = 0.5 * (x_min + x_max)
                    y0 = 0.5 * (y_min + y_max)
                    z0 = 0.5 * (z_min + z_max)
                    # diagonal of the box as a stable distance baseline
                    diag = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2 + (z_max - z_min) ** 2) ** 0.5
                    mlab.view(azimuth=-60, elevation=35, distance=1.2 * diag, focalpoint=(x0, y0, z0))
                except Exception:
                    try:
                        mlab.view(azimuth=-60, elevation=35, distance="auto", focalpoint="auto")
                    except Exception:
                        pass

            if hasattr(scene, "render"):
                scene.render()
            elif hasattr(scene, "scene") and hasattr(scene.scene, "render"):
                scene.scene.render()

        def _maybe_axes():
            if not args.show_axes:
                return
            mlab.axes(
                xlabel="x",
                ylabel="y",
                zlabel="z",
                nb_labels=5,
                ranges=[x_min, x_max, y_min, y_max, z_min, z_max],
            )
            mlab.outline(
                extent=(x_min, x_max, y_min, y_max, z_min, z_max),
                color=(0.2, 0.2, 0.2),
                line_width=1.0,
            )

        def _draw_occ():
            p = mlab.points3d(
                occ_x,
                occ_y,
                occ_z,
                c,
                scale_factor=float(max(sx, sy, sz)),
                mode="cube",
                scale_mode="vector",
                opacity=1.0,
                vmin=1,
                vmax=17,
            )
            p.module_manager.scalar_lut_manager.lut.table = occ_colors_map
            return p

        def _draw_pcd():
            mlab.points3d(
                xyz_plot[:, 0],
                xyz_plot[:, 1],
                xyz_plot[:, 2],
                scale_factor=float(max(sx, sy, sz) * 0.25),
                mode="sphere",
                color=(0.0, 0.0, 0.0),
                opacity=0.25,
            )

        if args.layout == "side_by_side":
            # Two separate Mayavi figures (left PCD, right OCC), later concatenated.
            fig_left = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            _draw_pcd()
            _maybe_axes()
            _apply_camera(fig_left)
            left_img = mlab.screenshot(figure=fig_left, mode="rgb", antialiased=True)

            fig_right = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
            _draw_occ()
            _maybe_axes()
            _apply_camera(fig_right)
            right_img = mlab.screenshot(figure=fig_right, mode="rgb", antialiased=True)

            # Compose side-by-side and save.
            import imageio

            sbs = np.concatenate([left_img, right_img], axis=1)
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            imageio.imwrite(args.out, sbs)
            mlab.close(all=True)
            print(f"Saved: {args.out}")
            return

        # overlay
        fig = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
        _draw_occ()
        if args.overlay_pcd:
            _draw_pcd()
        _maybe_axes()
        _apply_camera(fig)

        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        mlab.savefig(args.out)
        mlab.close(all=True)
        print(f"Saved: {args.out}")
        return

    # Import matplotlib lazily (some envs are headless).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6), dpi=150)
    if args.view == "3d":
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        ax0 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")

        # Left: raw points
        ax0.scatter(
            xyz_plot[:, 0],
            xyz_plot[:, 1],
            xyz_plot[:, 2],
            s=0.2,
            c="k",
            alpha=0.35,
            depthshade=False,
        )
        ax0.set_title(f"PCD 3D (N={xyz.shape[0]})")

        # Right: occupied voxels
        if args.color_by_semantic:
            # Discrete colormap for semantic ids.
            uniq = np.unique(sem_id)
            # keep a stable ordering
            uniq = np.sort(uniq)
            # Map ids to 0..K-1
            id_to_k = {int(s): i for i, s in enumerate(uniq.tolist())}
            k = np.vectorize(lambda s: id_to_k[int(s)])(sem_id).astype(np.int64)
            cmap = plt.get_cmap("tab20", len(uniq))
            sc = ax1.scatter(
                occ_x,
                occ_y,
                occ_z,
                s=1.2,
                c=k.astype(np.float32),
                cmap=cmap,
                alpha=0.9,
                depthshade=False,
                vmin=-0.5,
                vmax=len(uniq) - 0.5,
            )
            cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04, ticks=np.arange(len(uniq)))
            labels = []
            for sid in uniq.tolist():
                sid_i = int(sid)
                name = id_to_name.get(sid_i)
                labels.append(f"{sid_i}:{name}" if name else str(sid_i))
            cbar.ax.set_yticklabels(labels)
            cbar.set_label("semantic id")

            # Also add an on-plot legend (useful when saving images)
            try:
                handles = []
                for j, sid in enumerate(uniq.tolist()):
                    rgba = cmap(j / max(1, len(uniq) - 1))
                    sid_i = int(sid)
                    name = id_to_name.get(sid_i)
                    lab = f"{sid_i}:{name}" if name else str(sid_i)
                    handles.append(Patch(facecolor=rgba, edgecolor="k", label=lab))
                ax1.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
            except Exception:
                pass
        elif args.color_by_height:
            sc = ax1.scatter(
                occ_x,
                occ_y,
                occ_z,
                s=1.2,
                c=vz.astype(np.float32),
                cmap="viridis",
                alpha=0.9,
                depthshade=False,
            )
            cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label("z bin")
        else:
            ax1.scatter(
                occ_x,
                occ_y,
                occ_z,
                s=1.2,
                c="tab:orange",
                alpha=0.9,
                depthshade=False,
            )
        ax1.set_title(f"Occupied voxels 3D (N={occ.shape[0]})")

        for ax in (ax0, ax1):
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=args.elev, azim=args.azim)
            # Make the scene look less stretched.
            try:
                ax.set_box_aspect((x_max - x_min, y_max - y_min, (z_max - z_min) * 3.0))
            except Exception:
                pass
    else:
        axes = fig.subplots(1, 2)

        from matplotlib.patches import Patch

        # Left: raw points
        ax = axes[0]
        ax.scatter(xyz_plot[:, 0], xyz_plot[:, 1], s=0.2, c="k", alpha=0.6)
        ax.set_title(f"PCD XY (N={xyz.shape[0]})")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Right: occupied voxels
        ax = axes[1]
        if args.color_by_semantic:
            uniq = np.unique(sem_id)
            uniq = np.sort(uniq)
            id_to_k = {int(s): i for i, s in enumerate(uniq.tolist())}
            k = np.vectorize(lambda s: id_to_k[int(s)])(sem_id).astype(np.int64)
            cmap = plt.get_cmap("tab20", len(uniq))
            sc = ax.scatter(occ_x, occ_y, s=1.0, c=k.astype(np.float32), cmap=cmap, alpha=0.9, vmin=-0.5, vmax=len(uniq) - 0.5)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(len(uniq)))
            labels = []
            for sid in uniq.tolist():
                sid_i = int(sid)
                name = id_to_name.get(sid_i)
                labels.append(f"{sid_i}:{name}" if name else str(sid_i))
            cbar.ax.set_yticklabels(labels)
            cbar.set_label("semantic id")

            try:
                handles = []
                for j, sid in enumerate(uniq.tolist()):
                    rgba = cmap(j / max(1, len(uniq) - 1))
                    sid_i = int(sid)
                    name = id_to_name.get(sid_i)
                    lab = f"{sid_i}:{name}" if name else str(sid_i)
                    handles.append(Patch(facecolor=rgba, edgecolor="k", label=lab))
                ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
            except Exception:
                pass
        elif args.color_by_height:
            sc = ax.scatter(occ_x, occ_y, s=1.0, c=vz.astype(np.float32), cmap="viridis", alpha=0.9)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("z bin")
        else:
            ax.scatter(occ_x, occ_y, s=1.0, c="tab:orange", alpha=0.9)
        ax.set_title(f"Occupied voxels XY (N={occ.shape[0]})")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)

    print(f"Saved visualization: {args.out}")


if __name__ == "__main__":
    main()
