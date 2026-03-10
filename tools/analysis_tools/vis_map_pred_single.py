"""Visualize predicted and GT vector-map results for a single sample.

Usage example:
  python tools/analysis_tools/vis_map_pred_single.py \
    --data-root data/nuscenes \
    --infos data/nuscenes/nuscenes_infos_temporal_val.pkl \
    --results test/bev_tiny_det_map_apollo/Tue_Mar_10_15_19_32_2026/map_results.pkl \
    --index 50 \
    --out tools_outputs/det_map_vis_epoch12/map_idx050.png
"""

from __future__ import annotations

import argparse
import os
import sys

import mmcv
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.nuscenes_det_occ_map_dataset import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    _scene_name_to_log_location,
)


LABEL2NAME = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
}

LABEL2COLOR = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#2ca02c',
}


def _load_input_mosaic_from_nuscenes(nusc: NuScenes, dataroot: str, sample_token: str):
    import os.path as osp
    import cv2
    from PIL import Image

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
        tok = sample['data'].get(cam)
        if not tok:
            imgs.append(None)
            continue
        sd = nusc.get('sample_data', tok)
        p = osp.join(dataroot, sd['filename'])
        if not osp.exists(p):
            imgs.append(None)
            continue
        imgs.append(np.array(Image.open(p).convert('RGB')))

    avail = [im for im in imgs if im is not None]
    if not avail:
        return None

    h = min(im.shape[0] for im in avail)
    panels = []
    for cam, im in zip(cams, imgs):
        if im is None:
            w = int(h * 16 / 9)
            panel = np.full((h, w, 3), 200, dtype=np.uint8)
            cv2.putText(panel, f'{cam} (missing)', (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 2, cv2.LINE_AA)
            panels.append(panel)
            continue
        scale = h / im.shape[0]
        w = max(1, int(im.shape[1] * scale))
        panel = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (w, 55), (255, 255, 255), thickness=-1)
        cv2.putText(panel, cam, (15, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(panel)

    col_ws = [max(panels[i].shape[1] for i in [0, 3]),
              max(panels[i].shape[1] for i in [1, 4]),
              max(panels[i].shape[1] for i in [2, 5])]

    def pad_to_w(img, w):
        if img.shape[1] == w:
            return img
        pad = np.full((img.shape[0], w - img.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([img, pad], axis=1)

    top_row = np.concatenate([
        pad_to_w(panels[0], col_ws[0]),
        pad_to_w(panels[1], col_ws[1]),
        pad_to_w(panels[2], col_ws[2]),
    ], axis=1)
    bot_row = np.concatenate([
        pad_to_w(panels[3], col_ws[0]),
        pad_to_w(panels[4], col_ws[1]),
        pad_to_w(panels[5], col_ws[2]),
    ], axis=1)
    return np.concatenate([top_row, bot_row], axis=0)


def _compose_with_input_image(*, bev_path: str, out_path: str, input_rgb, top_height: int = 260) -> None:
    from PIL import Image

    bev = Image.open(bev_path).convert('RGBA')
    if input_rgb is None:
        bev.save(out_path)
        return

    top = Image.fromarray(input_rgb).convert('RGBA')
    target_w = bev.size[0]
    # Match the BEV width exactly so the top 2x3 camera mosaic spans the
    # same width as the lower BEV plot, without center cropping.
    scale = target_w / max(1, top.size[0])
    resized_h = max(1, int(round(top.size[1] * scale)))
    top_strip = top.resize((target_w, resized_h), resample=Image.BILINEAR)

    # Keep `top_height` as a lower bound only; this avoids very tiny top strips
    # when the BEV figure is narrow, but still preserves the full camera mosaic.
    if top_height > 0 and resized_h < top_height:
        canvas = Image.new('RGBA', (target_w, top_height), (255, 255, 255, 255))
        oy = (top_height - resized_h) // 2
        canvas.paste(top_strip, (0, oy), top_strip)
        top_strip = canvas

    gap = 6
    out = Image.new('RGBA', (target_w, top_strip.size[1] + gap + bev.size[1]), (255, 255, 255, 255))
    out.paste(top_strip, (0, 0))
    out.paste(bev, (0, top_strip.size[1] + gap), bev)
    out.save(out_path)


def _build_lidar2global(info: dict) -> np.ndarray:
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])

    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = np.array(info['ego2global_translation'])
    return ego2global @ lidar2ego


def _load_gt(vmap: VectorizedLocalMap, info: dict):
    location = info.get('map_location', None)
    if location is None:
        scene_name = info.get('scene_name', None)
        if not scene_name:
            raise KeyError('Missing map_location/scene_name in infos record.')
        location = _scene_name_to_log_location(
            scene_name,
            dataroot=vmap.data_root,
            version='v1.0-trainval',
        ) or scene_name

    lidar2global = _build_lidar2global(info)
    anns = vmap.gen_vectorized_samples(
        location=location,
        lidar2global_translation=list(lidar2global[:3, 3]),
        lidar2global_rotation=list(Quaternion(matrix=lidar2global).q),
    )
    labels = np.asarray(anns['gt_vecs_label'], dtype=np.int64)
    pts_obj = anns['gt_vecs_pts_loc']
    if isinstance(pts_obj, LiDARInstanceLines):
        pts = pts_obj.fixed_num_sampled_points.cpu().numpy()
    else:
        pts = np.asarray(pts_obj, dtype=np.float32)
    return labels, pts


def _load_pred(result: dict):
    pts = np.asarray(result['vectors'], dtype=np.float32)
    scores = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
    labels = np.asarray(result['labels'], dtype=np.int64).reshape(-1)
    return labels, scores, pts


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize one map prediction sample.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--infos', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--patch-h', type=float, default=100.0)
    parser.add_argument('--patch-w', type=float, default=100.0)
    parser.add_argument('--fixed-pts', type=int, default=20)
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--with-input', action='store_true')
    parser.add_argument('--input-height', type=int, default=260)
    args = parser.parse_args()

    infos = mmcv.load(args.infos)['infos']
    results = mmcv.load(args.results)
    info = infos[args.index]
    result = results[args.index]

    vmap = VectorizedLocalMap(
        dataroot=args.data_root,
        patch_size=(args.patch_h, args.patch_w),
        map_classes=('divider', 'ped_crossing', 'boundary'),
        fixed_ptsnum_per_line=args.fixed_pts,
    )

    gt_labels, gt_pts = _load_gt(vmap, info)
    pred_labels, pred_scores, pred_pts = _load_pred(result)

    keep = pred_scores >= float(args.score_thr)
    if int(args.topk) > 0 and keep.any():
        keep_idx = np.where(keep)[0]
        ranked = keep_idx[np.argsort(pred_scores[keep_idx])[::-1][: args.topk]]
        keep = np.zeros_like(keep, dtype=bool)
        keep[ranked] = True

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_xlim([-args.patch_w / 2, args.patch_w / 2])
    ax.set_ylim([-args.patch_h / 2, args.patch_h / 2])
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.scatter([0.0], [0.0], c='k', s=20, label='ego')

    for i in range(len(gt_labels)):
        lab = int(gt_labels[i])
        color = LABEL2COLOR.get(lab, '#7f7f7f')
        ax.plot(gt_pts[i, :, 0], gt_pts[i, :, 1], color=color, linewidth=1.6, alpha=0.30)

    for i in np.where(keep)[0]:
        lab = int(pred_labels[i])
        color = LABEL2COLOR.get(lab, '#7f7f7f')
        ax.plot(pred_pts[i, :, 0], pred_pts[i, :, 1], color=color, linewidth=2.5, alpha=0.95)
        ax.text(
            float(pred_pts[i, 0, 0]),
            float(pred_pts[i, 0, 1]),
            f'{LABEL2NAME.get(lab, lab)}:{pred_scores[i]:.2f}',
            fontsize=7,
            color=color,
            alpha=0.9,
        )

    handles = [
        plt.Line2D([0], [0], color=LABEL2COLOR[k], lw=3, label=LABEL2NAME[k])
        for k in sorted(LABEL2NAME)
    ]
    handles.extend([
        plt.Line2D([0], [0], color='k', lw=1.6, alpha=0.30, label='GT (faint)'),
        plt.Line2D([0], [0], color='k', lw=2.5, alpha=0.95, label='Pred (kept)'),
        plt.Line2D([0], [0], marker='o', color='k', lw=0, label='ego'),
    ])
    ax.legend(handles=handles, loc='upper right')
    ax.set_title(
        f"Map prediction idx={args.index} | scene={info.get('scene_name', 'n/a')} | "
        f"token={info['token'][:8]}...\nkept={int(keep.sum())}/{len(pred_scores)} @ score>={args.score_thr}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=220)
    plt.close(fig)

    if args.with_input:
        input_rgb = None
        try:
            nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
            input_rgb = _load_input_mosaic_from_nuscenes(nusc, args.data_root, info['token'])
        except Exception:
            input_rgb = None

        with_input_out = os.path.splitext(args.out)[0] + '_with_input.png'
        _compose_with_input_image(
            bev_path=args.out,
            out_path=with_input_out,
            input_rgb=input_rgb,
            top_height=args.input_height,
        )
        print('saved:', with_input_out)

    print('saved:', args.out)


if __name__ == '__main__':
    main()