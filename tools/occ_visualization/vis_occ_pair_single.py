#!/usr/bin/env python3
"""Render a single-frame Occupancy GT vs Pred comparison image.

This script consumes the folder layout produced by
`NuScenesDataset.evaluate_occ_iou(..., show_dir=...)`:

  <show_dir>/thre_XX.XX/<scene_name>/{images,occ_gts,occ_preds}/

and creates a PNG that stacks:
  [ RGB (stitched) ]
  [ OCC GT | OCC Pred ]

It uses Mayavi offscreen rendering (same style as create_video_gt_pred_rgb.py).

Demo (recommended; aligned with occ dump folders):

    python tools/occ_visualization/vis_occ_pair_single.py \
        --occ_root test/occ_dump_quick --thre 0.25 --scene scene-0916 --frame 0 \
        --dataroot data/nuscenes --version v1.0-mini

Notes:
    - Output will be written to:
            test/occ_dump_quick/thre_0.25/scene-0916/visualization_occ_single/000.png
    - If dump folder `images/000.png` is missing, the script will automatically
        stitch the 6 nuScenes cameras from `--dataroot/--version`.
    - When you run evaluation via `tools/dist_test.sh ... --eval iou` (or `iou bbox`),
        the occ IoU metrics are also saved to a JSON file:
            <show_dir>/thre_XX.XX/occ_eval/metrics_summary.json
        If you did not pass `--show_dir`, it will follow the test-time folder:
            test/<exp_name>/<timestamp>/thre_XX.XX/occ_eval/metrics_summary.json
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import imageio
import numpy as np

# Mayavi needs to be imported before matplotlib in some environments.
from mayavi import mlab  # noqa: E402

mlab.options.offscreen = True

import matplotlib.pyplot as plt  # noqa: E402
from nuscenes.nuscenes import NuScenes  # noqa: E402


NUM_CLASSES = 16
POINT_CLOUD_RANGE = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize occupancy GT vs Pred for one frame.')
    # Keep backwards-compat (--root) and add a more consistent alias (--occ_root)
    # to match other aligned visualization scripts in this repo.
    p.add_argument('--root', default='', help='show_dir root used in dist_test (--show_dir).')
    p.add_argument('--occ_root', default='', help='Alias of --root (preferred).')
    p.add_argument('--thre', type=float, default=0.25, help='occupancy threshold, e.g. 0.25')
    p.add_argument('--scene', required=True, help='scene name, e.g. scene-0916')
    p.add_argument('--frame', type=int, required=True, help='frame index to visualize (integer).')
    p.add_argument('--out', default='', help='output png path; default under <root>/.../visualization_occ_single/')
    p.add_argument(
        '--dataroot',
        default='data/nuscenes',
        help='nuScenes dataroot (only used when dump images/{frame}.png is missing).',
    )
    p.add_argument(
        '--version',
        default='v1.0-mini',
        help='nuScenes version (only used when dump images/{frame}.png is missing).',
    )
    p.add_argument('--occ_resolution', choices=['coarse', 'fine'], default='coarse')
    p.add_argument('--no_ego', action='store_true', help='do not render ego-car voxels')
    args = p.parse_args()

    # Resolve root from either flag.
    args.root = args.root or args.occ_root
    if not args.root:
        p.error('one of --root or --occ_root is required')
    return args


def _find_sample_token_by_scene_and_frame(nusc: NuScenes, scene_name: str, frame_idx: int) -> str:
    if frame_idx < 0:
        raise ValueError(f'frame_idx must be >= 0. Got {frame_idx}')
    first_sample_token = None
    for sc in nusc.scene:
        if sc.get('name') == scene_name:
            first_sample_token = sc['first_sample_token']
            break
    if first_sample_token is None:
        raise KeyError(f'Cannot find scene name {scene_name!r} in nuScenes {nusc.version}.')
    token = first_sample_token
    for _ in range(frame_idx):
        rec = nusc.get('sample', token)
        token = rec['next']
        if token == '':
            raise IndexError(f'frame_idx={frame_idx} is out of range for scene {scene_name}.')
    return token


def _load_or_build_input_rgb(
    *,
    rgb_path: str,
    scene_name: str,
    frame_idx: int,
    dataroot: str,
    version: str,
):
    """Load stitched surround image if available, else build from nuScenes 6 cams."""
    if osp.exists(rgb_path):
        return imageio.imread(rgb_path)

    # Build a horizontal mosaic from the 6 cameras.
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    sample_token = _find_sample_token_by_scene_and_frame(nusc, scene_name, frame_idx)
    sample = nusc.get('sample', sample_token)
    cams = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ]

    imgs = []
    for cam in cams:
        if cam not in sample['data']:
            continue
        sd = nusc.get('sample_data', sample['data'][cam])
        img_fp = osp.join(dataroot, sd['filename'])
        if osp.exists(img_fp):
            imgs.append(imageio.imread(img_fp))

    if len(imgs) == 0:
        return None

    # Resize all images to the same height for clean stitching.
    import cv2
    h = min(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        scale = h / im.shape[0]
        w = int(im.shape[1] * scale)
        resized.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def _occ_grid_params(occ_resolution: str):
    if occ_resolution == 'coarse':
        occupancy_size = [0.5, 0.5, 0.5]
        voxel_size = 0.5
    else:
        occupancy_size = [0.2, 0.2, 0.2]
        voxel_size = 0.2

    occ_xdim = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / occupancy_size[0])
    occ_ydim = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / occupancy_size[1])
    occ_zdim = int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / occupancy_size[2])
    voxel_num = occ_xdim * occ_ydim * occ_zdim
    return occupancy_size, voxel_size, occ_xdim, occ_ydim, occ_zdim, voxel_num


def generate_the_ego_car(voxel_size: float, num_classes: int):
    ego_range = [-2, -1, -1.5, 2, 1, 0]
    ego_voxel_size = [0.5, 0.5, 0.5]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])

    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)

    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_x, ego_point_y, ego_point_z), axis=-1)

    ego_points_label = (np.ones((ego_point_xyz.shape[0])) * num_classes).astype(np.uint8)
    ego_points_flow = np.zeros((ego_point_xyz.shape[0], 2))

    return {'point': ego_point_xyz, 'label': ego_points_label, 'flow': ego_points_flow}


def obtain_points_label(occ, voxel_num, occ_xdim, occ_ydim, occ_zdim):
    occ_index, occ_cls = occ[:, 0].astype(np.int64), occ[:, 1].astype(np.int64)

    # We only render occupied voxels (occ_index), no need to create dense grid.
    points = []
    for i in range(len(occ_index)):
        indice = int(occ_index[i])
        x = indice % occ_xdim
        y = (indice // occ_xdim) % occ_ydim
        z = indice // (occ_xdim * occ_ydim)
        point_x = (x + 0.5) / occ_xdim * (POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) + POINT_CLOUD_RANGE[0]
        point_y = (y + 0.5) / occ_ydim * (POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) + POINT_CLOUD_RANGE[1]
        point_z = (z + 0.5) / occ_zdim * (POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) + POINT_CLOUD_RANGE[2]
        points.append([point_x, point_y, point_z])

    points = np.stack(points) if len(points) else np.zeros((0, 3), dtype=np.float32)
    labels = occ_cls.astype(np.int64)
    return points, labels


def visualize_occ(points, labels, ego_dict, voxel_size: float, add_ego_car: bool):
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

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    point_colors = np.zeros(points.shape[0])
    for cls_index in range(NUM_CLASSES):
        class_point = labels == cls_index
        point_colors[class_point] = cls_index + 1

    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))
    lidar_plot = mlab.points3d(
        x,
        y,
        z,
        point_colors,
        scale_factor=voxel_size,
        mode='cube',
        scale_mode='vector',
        opacity=1.0,
        vmin=1,
        vmax=17,
    )
    lidar_plot.module_manager.scalar_lut_manager.lut.table = occ_colors_map

    if add_ego_car and ego_dict is not None:
        ego_point_xyz = ego_dict['point']
        ego_color = np.linalg.norm(ego_point_xyz, axis=-1)
        ego_color = ego_color / (ego_color.max() + 1e-6)
        mlab.points3d(
            ego_point_xyz[:, 0],
            ego_point_xyz[:, 1],
            ego_point_xyz[:, 2],
            ego_color,
            colormap='rainbow',
            scale_factor=voxel_size,
            mode='cube',
            opacity=1.0,
            scale_mode='none',
        )

    # fixed camera (same as create_video_gt_pred_rgb.py)
    scene = figure
    scene.scene.z_plus_view()
    scene.scene.camera.position = [-1.1612566981665453, -63.271696093007456, 33.06645769267362]
    scene.scene.camera.focal_point = [-0.0828344205684326, -0.029545161654287222, -1.078433202901462]
    scene.scene.camera.view_angle = 30.0
    scene.scene.camera.view_up = [-0.011200801911309498, 0.4752037522484654, 0.879804487306994]
    scene.scene.camera.clipping_range = [0.18978054185107493, 189.78054185107493]
    scene.scene.camera.compute_view_plane_normal()
    scene.scene.render()

    img = mlab.screenshot()
    mlab.close()
    return img


def main() -> None:
    args = _parse_args()

    thre_str = f"thre_{args.thre:.2f}"
    base_dir = osp.join(args.root, thre_str, args.scene)

    occ_gt_path = osp.join(base_dir, 'occ_gts', f'{args.frame:03d}_occ.npy')
    occ_pred_path = osp.join(base_dir, 'occ_preds', f'{args.frame:03d}_occ.npy')
    rgb_path = osp.join(base_dir, 'images', f'{args.frame:03d}.png')

    if not osp.exists(occ_gt_path):
        raise FileNotFoundError(f'GT occ not found: {occ_gt_path}')
    if not osp.exists(occ_pred_path):
        raise FileNotFoundError(f'Pred occ not found: {occ_pred_path}')

    occ_gt = np.load(occ_gt_path)
    occ_pred = np.load(occ_pred_path)

    occupancy_size, voxel_size, occ_xdim, occ_ydim, occ_zdim, voxel_num = _occ_grid_params(args.occ_resolution)

    ego_dict = None
    if not args.no_ego:
        ego_dict = generate_the_ego_car(voxel_size=voxel_size, num_classes=NUM_CLASSES)

    gt_points, gt_labels = obtain_points_label(occ_gt, voxel_num, occ_xdim, occ_ydim, occ_zdim)
    pred_points, pred_labels = obtain_points_label(occ_pred, voxel_num, occ_xdim, occ_ydim, occ_zdim)

    occ_gt_img = visualize_occ(gt_points, gt_labels, ego_dict, voxel_size=voxel_size, add_ego_car=not args.no_ego)
    occ_pred_img = visualize_occ(pred_points, pred_labels, ego_dict, voxel_size=voxel_size, add_ego_car=not args.no_ego)

    rgb_img = _load_or_build_input_rgb(
        rgb_path=rgb_path,
        scene_name=args.scene,
        frame_idx=args.frame,
        dataroot=args.dataroot,
        version=args.version,
    )

    if args.out:
        out_path = args.out
    else:
        out_dir = osp.join(base_dir, 'visualization_occ_single')
        os.makedirs(out_dir, exist_ok=True)
        out_path = osp.join(out_dir, f'{args.frame:03d}.png')

    # Use GridSpec so the top RGB panel is centered and aligned with the two bottom panels
    # (matplotlib's subplot + tight_layout can look off-center with mixed grids).
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 10), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.0, 1.0],
        hspace=0.05,
        wspace=0.02,
    )

    # Top: span both columns.
    ax_rgb = fig.add_subplot(gs[0, :])
    ax_rgb.axis('off')
    if rgb_img is not None:
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title('RGB (stitched)', fontsize=12)
    else:
        ax_rgb.text(
            0.5,
            0.5,
            'RGB image not found',
            ha='center',
            va='center',
            transform=ax_rgb.transAxes,
        )

    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.axis('off')
    ax_gt.imshow(occ_gt_img)
    ax_gt.set_title('OCC GT', fontsize=12)

    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.axis('off')
    ax_pred.imshow(occ_pred_img)
    ax_pred.set_title('OCC Pred', fontsize=12)

    # Manual margins (more stable than tight_layout for this layout).
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print('saved:', out_path)


if __name__ == '__main__':
    main()
