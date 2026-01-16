"""Simulate nuScenes-map GT generation (vector map) and visualize in BEV.

This script:
1) Loads one sample from a *_infos_temporal_*.pkl (must contain map_location).
2) Builds lidar2global from lidar2ego + ego2global.
3) Uses MapTR-style VectorizedLocalMap to generate vector GT:
   - divider (road_divider/lane_divider)
   - ped_crossing
   - boundary (road/lane contours)
4) Plots polylines in LiDAR local coordinates (same BEV frame as det/occ).

Output: a PNG with legend.

Example:
  python tools/simulate_and_vis_map_gt.py \
    --data-root data/nuscenes \
    --infos data/nuscenes/nuscenes_with_maploc_infos_temporal_train.pkl \
    --index 0 \
    --out tools_outputs/map_gt_vis_idx0.png
"""

import argparse
import os
import sys

import mmcv
import numpy as np
from pyquaternion import Quaternion

# For headless servers.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.nuscenes_det_occ_map_dataset import VectorizedLocalMap


LABEL2NAME = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
}
LABEL2COLOR = {
    0: '#1f77b4',  # blue
    1: '#ff7f0e',  # orange
    2: '#2ca02c',  # green
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--infos', type=str, required=True)
    parser.add_argument('--index', type=int, default=0,
                        help='Single index to visualize (ignored if --start/--count is set).')
    parser.add_argument('--start', type=int, default=None,
                        help='Start index (inclusive) for batch export.')
    parser.add_argument('--count', type=int, default=None,
                        help='Number of frames to export for batch mode.')
    parser.add_argument('--out', type=str, default='tools_outputs/map_gt_vis.png',
                        help='Output PNG path (single mode) or output directory (batch mode).')
    parser.add_argument('--patch-h', type=float, default=100.0)
    parser.add_argument('--patch-w', type=float, default=100.0)
    parser.add_argument('--fixed-pts', type=int, default=80)
    parser.add_argument('--per-class', action='store_true',
                        help='In single mode: also export per-class-only PNGs next to --out.')
    parser.add_argument('--only', type=str, default=None,
                        choices=['divider', 'ped_crossing', 'boundary'],
                        help='Only render one class (single mode).')
    args = parser.parse_args()

    data = mmcv.load(args.infos)
    vmap = VectorizedLocalMap(
        dataroot=args.data_root,
        patch_size=(args.patch_h, args.patch_w),
        map_classes=('divider', 'ped_crossing', 'boundary'),
        fixed_ptsnum_per_line=args.fixed_pts,
    )

    def render_one(index: int, out_path: str, only_label: int = None, title_suffix: str = ''):
        info = data['infos'][index]
        if 'map_location' not in info:
            raise RuntimeError(
                'infos pkl missing map_location. Please regenerate infos with updated converter.'
            )

        # Build lidar2global.
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = np.array(info['ego2global_translation'])

        lidar2global = ego2global @ lidar2ego

        anns = vmap.gen_vectorized_samples(
            location=info['map_location'],
            lidar2global_translation=list(lidar2global[:3, 3]),
            lidar2global_rotation=list(Quaternion(matrix=lidar2global).q),
        )

        labels = anns['gt_vecs_label']
        inst = anns['gt_vecs_pts_loc']

        # Convert to fixed points for visualization.
        if len(labels) > 0:
            pts = inst.fixed_num_sampled_points.numpy()  # [N, fixed_pts, 2]
        else:
            pts = np.zeros((0, args.fixed_pts, 2), dtype=np.float32)

        # Plot (pure GT in LiDAR-local BEV).
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        base_title = f"Map GT (idx={index}) @ {info['map_location']}"
        if title_suffix:
            base_title += f" | {title_suffix}"
        ax.set_title(base_title)

        half_w = args.patch_w / 2
        half_h = args.patch_h / 2
        ax.set_xlim([-half_w, half_w])
        ax.set_ylim([-half_h, half_h])
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.2)

        ax.scatter([0.0], [0.0], c='k', s=20, label='ego')

        drawn_labels = []
        for i in range(len(labels)):
            lab = int(labels[i])
            if only_label is not None and lab != only_label:
                continue
            color = LABEL2COLOR.get(lab, '#7f7f7f')
            xy = pts[i]
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2, alpha=0.9)
            drawn_labels.append(lab)

        handles = []
        for lab in sorted(set(int(x) for x in drawn_labels)):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=LABEL2COLOR.get(lab, '#7f7f7f'),
                    lw=3,
                    label=LABEL2NAME.get(lab, str(lab)),
                )
            )
        handles.append(plt.Line2D([0], [0], marker='o', color='k', lw=0, label='ego'))
        ax.legend(handles=handles, loc='upper right')

        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        return info['map_location'], labels

    # Decide batch vs single.
    if args.start is not None or args.count is not None:
        if args.start is None or args.count is None:
            raise ValueError('Batch mode requires both --start and --count.')
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        end = args.start + args.count
        print(f'Batch export: [{args.start}, {end}) -> {out_dir}')
        for idx in range(args.start, end):
            out_path = os.path.join(out_dir, f'map_gt_vis_idx{idx:06d}.png')
            loc, labels = render_one(idx, out_path)
            if idx == args.start:
                # print one sample stats for sanity
                print('first_frame_map_location:', loc)
                print('first_frame_num_vecs:', len(labels))
                if len(labels) > 0:
                    uniq, cnt = np.unique(np.array(labels), return_counts=True)
                    print('first_frame_label_hist:', {int(k): int(v) for k, v in zip(uniq, cnt)})
    else:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        only_label = None
        only_suffix = ''
        if args.only is not None:
            name2label = {'divider': 0, 'ped_crossing': 1, 'boundary': 2}
            only_label = name2label[args.only]
            only_suffix = f'only={args.only}'

        loc, labels = render_one(args.index, args.out, only_label=only_label, title_suffix=only_suffix)
        print('saved:', args.out)
        print('map_location:', loc)
        print('num_vecs:', len(labels))
        if len(labels) > 0:
            uniq, cnt = np.unique(np.array(labels), return_counts=True)
            print('label_hist:', {int(k): int(v) for k, v in zip(uniq, cnt)})

        if args.per_class and args.only is None:
            base, ext = os.path.splitext(args.out)
            render_one(args.index, f'{base}.divider{ext}', only_label=0, title_suffix='only=divider')
            render_one(args.index, f'{base}.ped_crossing{ext}', only_label=1, title_suffix='only=ped_crossing')
            render_one(args.index, f'{base}.boundary{ext}', only_label=2, title_suffix='only=boundary')
            print('saved per-class views:', f'{base}.divider{ext}', f'{base}.ped_crossing{ext}', f'{base}.boundary{ext}')


if __name__ == '__main__':
    main()
