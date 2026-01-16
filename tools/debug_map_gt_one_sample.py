"""Quick sanity check for nuScenes-map vector GT generation.

This script loads one entry from a nuscenes infos pkl and runs the
VectorizedLocalMap GT generation logic (divider/ped_crossing/boundary).

It is intentionally lightweight and does not require the training pipeline.

Expected:
- infos must contain `map_location` (regenerate with updated converter).
- dataset root must contain `maps/` (nuScenes map expansion).

Usage (example):
  python tools/debug_map_gt_one_sample.py \
    --data-root data/nuscenes \
    --infos data/nuscenes/nuscenes_infos_temporal_train.pkl \
    --index 0
"""

import argparse
import os
import sys

import mmcv
import numpy as np
from pyquaternion import Quaternion

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.nuscenes_det_occ_map_dataset import VectorizedLocalMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--infos', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--patch-h', type=float, default=100.0)
    parser.add_argument('--patch-w', type=float, default=100.0)
    args = parser.parse_args()

    data = mmcv.load(args.infos)
    info = data['infos'][args.index]

    assert 'map_location' in info, 'infos pkl missing map_location; regenerate infos first.'

    # Mirror how datasets build lidar2global.
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = np.array(info['ego2global_translation'])

    lidar2global = ego2global @ lidar2ego

    vmap = VectorizedLocalMap(
        dataroot=args.data_root,
        patch_size=(args.patch_h, args.patch_w),
        map_classes=('divider', 'ped_crossing', 'boundary'),
        fixed_ptsnum_per_line=50,
    )

    anns = vmap.gen_vectorized_samples(
        location=info['map_location'],
        lidar2global_translation=list(lidar2global[:3, 3]),
        lidar2global_rotation=list(Quaternion(matrix=lidar2global).q),
    )

    labels = anns['gt_vecs_label']
    inst = anns['gt_vecs_pts_loc']

    print('map_location:', info['map_location'])
    print('num_vecs:', len(labels))
    if len(labels) > 0:
        # Show class histogram.
        uniq, cnt = np.unique(np.array(labels), return_counts=True)
        print('label_hist:', {int(k): int(v) for k, v in zip(uniq, cnt)})
        # Show first polyline points.
        pts = inst.fixed_num_sampled_points[0].numpy()
        print('first_vec_pts_shape:', pts.shape)
        print('first_vec_pts_head:', pts[:5])


if __name__ == '__main__':
    main()
