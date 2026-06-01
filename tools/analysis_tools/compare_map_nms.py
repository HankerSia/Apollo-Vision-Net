#!/usr/bin/env python3
"""Compare original and GenMapFuse-style merged map predictions for one nuScenes sample.

The script renders two panels for a chosen sample token:
  - Original predictions (after score threshold + top-k filtering)
  - Class-wise NMS predictions (with a tighter centerline threshold)

Each panel includes the 6-camera input mosaic on top and the BEV map below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.nuscenes_det_occ_map_dataset import (
    LiDARInstanceLines,
    _scene_name_to_log_location,
)
from projects.mmdet3d_plugin.datasets.nuscenes_det_mapv2_dataset import (
    VectorizedLocalMapV2,
)


LABEL2NAME = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
    3: 'centerline',
}

LABEL2COLOR = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#d62728',
}

NAME2LABEL = {name: label for label, name in LABEL2NAME.items()}


def _load_input_mosaic_from_nuscenes(nusc: NuScenes, dataroot: str, sample_token: str):
    import cv2
    import os.path as osp
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
            cv2.putText(panel, f'{cam} (missing)', (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            panels.append(panel)
            continue
        scale = h / im.shape[0]
        w = max(1, int(im.shape[1] * scale))
        panel = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (w, 55), (255, 255, 255), thickness=-1)
        cv2.putText(panel, cam, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(panel)

    col_ws = [max(panels[i].shape[1] for i in [0, 3]), max(panels[i].shape[1] for i in [1, 4]), max(panels[i].shape[1] for i in [2, 5])]

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
    scale = target_w / max(1, top.size[0])
    resized_h = max(1, int(round(top.size[1] * scale)))
    top_strip = top.resize((target_w, resized_h), resample=Image.BILINEAR)

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


def _load_gt(vmap: VectorizedLocalMapV2, info: dict, version: str):
    location = info.get('map_location', None)
    if location is None:
        scene_name = info.get('scene_name', None)
        if not scene_name:
            raise KeyError('Missing map_location/scene_name in infos record.')
        location = _scene_name_to_log_location(scene_name, dataroot=vmap.data_root, version=version) or scene_name

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


def _load_pred(result):
    if isinstance(result, dict) and 'vectors' in result and 'scores' in result and 'labels' in result:
        pts = np.asarray(result['vectors'], dtype=np.float32)
        scores = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
        labels = np.asarray(result['labels'], dtype=np.int64).reshape(-1)
        return labels, scores, pts

    if isinstance(result, dict) and 'vectors' in result:
        vectors = result['vectors']
        labels = []
        scores = []
        pts = []
        for vec in vectors:
            pts.append(np.asarray(vec['pts'], dtype=np.float32))
            scores.append(float(vec.get('confidence_level', 1.0)))
            if 'type' in vec:
                labels.append(int(vec['type']))
            else:
                labels.append(int(NAME2LABEL.get(vec.get('cls_name', 'divider'), 0)))
        if len(pts) == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32), np.zeros((0, 0, 2), dtype=np.float32)
        return np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float32), np.asarray(pts, dtype=np.float32)

    raise TypeError(f'Unsupported result format: {type(result)!r}')


def _set_equal_length(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if pts_a.shape != pts_b.shape:
        raise ValueError(f'Expected same point shape, got {pts_a.shape} vs {pts_b.shape}')
    return pts_a, pts_b


def _line_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    pts_a, pts_b = _set_equal_length(pts_a, pts_b)
    forward = np.linalg.norm(pts_a - pts_b, axis=-1).mean()
    reverse = np.linalg.norm(pts_a - pts_b[::-1], axis=-1).mean()
    return float(min(forward, reverse))


def _line_heading_cosine(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    vec_a = pts_a[-1] - pts_a[0]
    vec_b = pts_b[-1] - pts_b[0]
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 1.0
    cos_val = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return abs(np.clip(cos_val, -1.0, 1.0))


def _heading_deg(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    dx, dy = points[-1] - points[0]
    return math.degrees(math.atan2(dy, dx))


def _projection_interval(points: np.ndarray, heading_deg_value: float) -> tuple[float, float]:
    rad = np.deg2rad(heading_deg_value)
    axis = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    proj = points @ axis
    return float(np.min(proj)), float(np.max(proj))


def _overlap_ratio(a: np.ndarray, b: np.ndarray, heading_deg_value: float) -> float:
    a0, a1 = _projection_interval(a, heading_deg_value)
    b0, b1 = _projection_interval(b, heading_deg_value)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return float(overlap / denom)


def _resample_to_count(points: np.ndarray, count: int) -> np.ndarray:
    from shapely.geometry import LineString

    if len(points) == 0:
        return points
    if count <= 2:
        return np.vstack([points[0], points[-1]]) if len(points) > 1 else points.copy()
    line = LineString(points)
    if line.length == 0.0:
        return np.repeat(points[:1], count, axis=0)
    distances = np.linspace(0.0, float(line.length), count, dtype=np.float64)
    return np.array([line.interpolate(float(d)).coords[0] for d in distances], dtype=np.float64)


def _align_to_reference(ref_pts: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if _line_distance(ref_pts, pts) <= _line_distance(ref_pts, pts[::-1]):
        return pts
    return pts[::-1]


def _merge_cluster(cluster_pts: Sequence[np.ndarray], cluster_scores: Sequence[float]) -> np.ndarray:
    ref_pts = cluster_pts[0]
    count = max(8, min(48, max(len(pts) for pts in cluster_pts)))
    aligned_pts = []
    for pts in cluster_pts:
        aligned = _align_to_reference(ref_pts, pts)
        aligned_pts.append(_resample_to_count(aligned, count))
    weights = np.asarray([max(1e-3, float(s)) for s in cluster_scores], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-6)
    stacked = np.stack(aligned_pts, axis=0)
    merged = np.tensordot(weights, stacked, axes=(0, 0))
    return merged.astype(np.float32)


def _fit_centerline_cluster(cluster_pts: Sequence[np.ndarray], cluster_scores: Sequence[float]) -> np.ndarray:
    if not cluster_pts:
        return np.zeros((0, 2), dtype=np.float32)

    anchor_idx = int(np.argmax([len(pts) for pts in cluster_pts]))
    anchor = cluster_pts[anchor_idx]
    if len(anchor) < 2:
        return anchor.astype(np.float32)

    weighted_points = []
    for pts, score in zip(cluster_pts, cluster_scores):
        aligned = _align_to_reference(anchor, pts)
        if len(aligned) < 2:
            continue
        repeat = max(1, int(round(float(score) * 10.0)))
        weighted_points.append(np.repeat(aligned, repeat, axis=0))

    if not weighted_points:
        return anchor.astype(np.float32)

    stacked = np.concatenate(weighted_points, axis=0)
    center = stacked.mean(axis=0)
    centered = stacked - center
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    if np.dot(direction, anchor[-1] - anchor[0]) < 0:
        direction = -direction
    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    t_all = centered @ direction
    n_all = centered @ normal

    unique_t = np.unique(np.round(t_all, 3))
    degree = min(3, max(1, len(unique_t) - 1))
    if degree <= 0:
        return anchor.astype(np.float32)

    try:
        poly_coeff = np.polyfit(t_all, n_all, deg=degree)
    except np.linalg.LinAlgError:
        return anchor.astype(np.float32)

    anchor_centered = anchor - center
    anchor_t = anchor_centered @ direction
    low = float(np.min(anchor_t))
    high = float(np.max(anchor_t))
    if high - low < 1.0:
        low = float(np.quantile(t_all, 0.05))
        high = float(np.quantile(t_all, 0.95))
    if high - low < 1e-3:
        return anchor.astype(np.float32)

    count = max(8, min(32, max(len(pts) for pts in cluster_pts)))
    samples_t = np.linspace(low, high, count, dtype=np.float64)
    samples_n = np.polyval(poly_coeff, samples_t)
    centerline = np.array(
        [center + direction * t + normal * n for t, n in zip(samples_t, samples_n)],
        dtype=np.float64,
    )
    return centerline.astype(np.float32)


def _guided_centerline_cluster(cluster_pts: Sequence[np.ndarray], cluster_scores: Sequence[float]) -> np.ndarray:
    if not cluster_pts:
        return np.zeros((0, 2), dtype=np.float32)

    guide_idx = int(np.argmax(cluster_scores))
    guide = _align_to_reference(cluster_pts[guide_idx], cluster_pts[guide_idx])
    if len(guide) < 2:
        return guide.astype(np.float32)

    from shapely.geometry import LineString, Point

    guide_line = LineString(guide)
    sample_count = max(8, min(48, max(len(pts) for pts in cluster_pts)))
    sample_distances = np.linspace(0.0, float(guide_line.length), sample_count, dtype=np.float64)

    aggregated = []
    for distance in sample_distances:
        guide_pt = np.array(guide_line.interpolate(float(distance)).coords[0], dtype=np.float64)
        aligned_points = []
        weights = []
        for pts, score in zip(cluster_pts, cluster_scores):
            if len(pts) < 2:
                continue
            line = LineString(_align_to_reference(guide, pts))
            projected = line.interpolate(float(line.project(Point(float(guide_pt[0]), float(guide_pt[1])))))
            aligned_points.append(np.array(projected.coords[0], dtype=np.float64))
            weights.append(max(1.0, float(score)))
        if not aligned_points:
            aggregated.append(guide_pt)
            continue
        aligned_arr = np.asarray(aligned_points, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        mean_pt = np.average(aligned_arr, axis=0, weights=weights_arr)
        aggregated.append(mean_pt)

    return np.asarray(aggregated, dtype=np.float32)


def _centerline_merge(
    labels: np.ndarray,
    scores: np.ndarray,
    pts: np.ndarray,
    *,
    merge_distance_m: float,
    angle_thr_deg: float,
    overlap_ratio_thr: float,
    fuse_method: str,
):
    centerline_cls = NAME2LABEL['centerline']
    center_idxs = np.where(labels == centerline_cls)[0]
    if len(center_idxs) == 0:
        return labels, scores, pts, np.ones((len(labels),), dtype=bool)

    order = center_idxs[np.argsort(scores[center_idxs])[::-1]]
    used = np.zeros((len(order),), dtype=bool)
    keep_labels: List[int] = []
    keep_scores: List[float] = []
    keep_pts: List[np.ndarray] = []
    keep_mask = np.zeros((len(labels),), dtype=bool)
    min_cos = math.cos(math.radians(float(angle_thr_deg)))

    for i, idx in enumerate(order):
        if used[i]:
            continue
        cluster = [int(idx)]
        cluster_scores = [float(scores[idx])]
        cluster_pts = [pts[idx]]
        used[i] = True
        for j in range(i + 1, len(order)):
            if used[j]:
                continue
            other_idx = int(order[j])
            if (
                _line_distance(pts[idx], pts[other_idx]) <= merge_distance_m
                and _line_heading_cosine(pts[idx], pts[other_idx]) >= min_cos
                and _overlap_ratio(pts[idx], pts[other_idx], _heading_deg(pts[idx])) >= overlap_ratio_thr
            ):
                used[j] = True
                cluster.append(other_idx)
                cluster_scores.append(float(scores[other_idx]))
                cluster_pts.append(pts[other_idx])

        if fuse_method == 'fit':
            merged_pts = _fit_centerline_cluster(cluster_pts, cluster_scores)
        elif fuse_method == 'guided':
            merged_pts = _guided_centerline_cluster(cluster_pts, cluster_scores)
        else:
            merged_pts = _merge_cluster(cluster_pts, cluster_scores)
        keep_labels.append(centerline_cls)
        keep_scores.append(max(cluster_scores))
        keep_pts.append(merged_pts)
        keep_mask[np.asarray(cluster, dtype=np.int64)] = True

    other_idxs = np.where(labels != centerline_cls)[0]
    if len(other_idxs) > 0:
        keep_labels.extend(labels[other_idxs].tolist())
        keep_scores.extend(scores[other_idxs].tolist())
        keep_pts.extend([pts[i] for i in other_idxs])
        keep_mask[other_idxs] = True

    return (
        np.asarray(keep_labels, dtype=np.int64),
        np.asarray(keep_scores, dtype=np.float32),
        np.asarray(keep_pts, dtype=np.float32),
        keep_mask,
    )


def _classwise_nms(labels: np.ndarray, scores: np.ndarray, pts: np.ndarray, *, centerline_thr: float, other_thr: float):
    keep_mask = np.zeros((len(scores),), dtype=bool)
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        if len(idxs) == 0:
            continue
        order = idxs[np.argsort(scores[idxs])[::-1]]
        kept: List[int] = []
        thr = centerline_thr if int(cls) == NAME2LABEL['centerline'] else other_thr
        for idx in order:
            if all(_line_distance(pts[idx], pts[kept_idx]) >= thr for kept_idx in kept):
                kept.append(int(idx))
                keep_mask[idx] = True
    return labels[keep_mask], scores[keep_mask], pts[keep_mask], keep_mask


def _apply_score_filter(labels, scores, pts, *, score_thr: float, topk: int):
    keep = scores >= float(score_thr)
    if int(topk) > 0 and keep.any():
        keep_idx = np.where(keep)[0]
        ranked = keep_idx[np.argsort(scores[keep_idx])[::-1][: int(topk)]]
        keep = np.zeros_like(keep, dtype=bool)
        keep[ranked] = True
    return labels[keep], scores[keep], pts[keep], keep


def _render_bev_panel(*, info: dict, title: str, labels: np.ndarray, scores: np.ndarray, pts: np.ndarray,
                      gt_labels: np.ndarray, gt_pts: np.ndarray, input_rgb, out_path: str,
                      score_thr: float, with_gt: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.scatter([0.0], [0.0], c='k', s=20, label='ego')

    if with_gt:
        for i in range(len(gt_labels)):
            lab = int(gt_labels[i])
            color = LABEL2COLOR.get(lab, '#7f7f7f')
            ax.plot(gt_pts[i, :, 0], gt_pts[i, :, 1], color=color, linewidth=1.6, alpha=0.25)

    for i in range(len(labels)):
        lab = int(labels[i])
        color = LABEL2COLOR.get(lab, '#7f7f7f')
        ax.plot(pts[i, :, 0], pts[i, :, 1], color=color, linewidth=2.5, alpha=0.95)
        ax.text(
            float(pts[i, 0, 0]),
            float(pts[i, 0, 1]),
            f'{LABEL2NAME.get(lab, lab)}:{scores[i]:.2f}',
            fontsize=7,
            color=color,
            alpha=0.9,
        )

    handles = [plt.Line2D([0], [0], color=LABEL2COLOR[k], lw=3, label=LABEL2NAME[k]) for k in sorted(LABEL2NAME)]
    handles.extend([
        plt.Line2D([0], [0], color='k', lw=1.6, alpha=0.25, label='GT (faint)'),
        plt.Line2D([0], [0], color='k', lw=2.5, alpha=0.95, label='Pred (kept)'),
        plt.Line2D([0], [0], marker='o', color='k', lw=0, label='ego'),
    ])
    ax.legend(handles=handles, loc='upper right')
    ax.set_title(
        f'{title} | scene={info.get("scene_name", "n/a")} | token={str(info.get("token", "n/a"))[:8]}...\n'
        f'kept={len(labels)} @ score>={score_thr}'
    )
    fig.tight_layout()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_bev = tmp.name
    try:
        fig.savefig(temp_bev, dpi=220)
    finally:
        plt.close(fig)

    try:
        _compose_with_input_image(bev_path=temp_bev, out_path=out_path, input_rgb=input_rgb, top_height=260)
    finally:
        try:
            os.remove(temp_bev)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare original and NMS map predictions for one sample.')
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--infos', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--sample-token', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--score-thr', type=float, default=0.35)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--centerline-nms-thr', type=float, default=1.0)
    parser.add_argument('--centerline-merge-distance-m', type=float, default=0.6)
    parser.add_argument('--centerline-merge-angle-thr', type=float, default=8.0)
    parser.add_argument('--centerline-merge-overlap-thr', type=float, default=0.35)
    parser.add_argument('--centerline-fuse-method', choices=['avg', 'fit', 'guided'], default='guided')
    parser.add_argument('--other-nms-thr', type=float, default=1.5)
    parser.add_argument('--mode', choices=['nms', 'merge'], default='merge')
    parser.add_argument('--fixed-pts', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    infos = mmcv.load(args.infos)['infos']
    info = next((item for item in infos if item.get('token') == args.sample_token), None)
    if info is None:
        raise KeyError(f'Could not find sample token in infos: {args.sample_token}')

    results = mmcv.load(args.results)
    result = next((item for item in results['results'] if item.get('sample_token') == args.sample_token), None)
    if result is None:
        raise KeyError(f'Could not find sample token in results: {args.sample_token}')

    gt_labels = np.zeros((0,), dtype=np.int64)
    gt_pts = np.zeros((0, args.fixed_pts, 2), dtype=np.float32)
    vmap = VectorizedLocalMapV2(
        dataroot=args.data_root,
        patch_size=(100.0, 100.0),
        map_classes=('divider', 'ped_crossing', 'boundary', 'centerline'),
        fixed_ptsnum_per_line=args.fixed_pts,
    )
    gt_labels, gt_pts = _load_gt(vmap, info, args.version)

    pred_labels, pred_scores, pred_pts = _load_pred(result)
    pred_labels, pred_scores, pred_pts, _ = _apply_score_filter(
        pred_labels, pred_scores, pred_pts, score_thr=args.score_thr, topk=args.topk
    )

    if args.mode == 'merge':
        nms_labels, nms_scores, nms_pts, keep_mask = _centerline_merge(
            pred_labels, pred_scores, pred_pts,
            merge_distance_m=args.centerline_merge_distance_m,
            angle_thr_deg=args.centerline_merge_angle_thr,
            overlap_ratio_thr=args.centerline_merge_overlap_thr,
            fuse_method=args.centerline_fuse_method,
        )
        mode_name = (
            f'GenMapFuse {args.centerline_fuse_method} @ {args.centerline_merge_distance_m}m / '
            f'{args.centerline_merge_angle_thr}deg / overlap>{args.centerline_merge_overlap_thr}'
        )
    else:
        nms_labels, nms_scores, nms_pts, keep_mask = _classwise_nms(
            pred_labels, pred_scores, pred_pts,
            centerline_thr=args.centerline_nms_thr,
            other_thr=args.other_nms_thr,
        )
        mode_name = f'class-wise NMS @ {args.centerline_nms_thr}'

    kept_before = int(len(pred_scores))
    kept_after = int(len(nms_scores))
    print(f'original kept={kept_before}, after_{args.mode}={kept_after}')
    print('centerline kept before/after:', int(np.sum(pred_labels == NAME2LABEL['centerline'])), int(np.sum(nms_labels == NAME2LABEL['centerline'])))

    input_rgb = None
    try:
        nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
        input_rgb = _load_input_mosaic_from_nuscenes(nusc, args.data_root, args.sample_token)
    except Exception:
        input_rgb = None

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    original_panel = str(Path(args.out).with_suffix('')) + '_original.png'
    nms_panel = str(Path(args.out).with_suffix('')) + '_nms.png'

    _render_bev_panel(
        info=info,
        title='Original predictions',
        labels=pred_labels,
        scores=pred_scores,
        pts=pred_pts,
        gt_labels=gt_labels,
        gt_pts=gt_pts,
        input_rgb=input_rgb,
        out_path=original_panel,
        score_thr=args.score_thr,
    )

    _render_bev_panel(
        info=info,
        title=f'After {mode_name}',
        labels=nms_labels,
        scores=nms_scores,
        pts=nms_pts,
        gt_labels=gt_labels,
        gt_pts=gt_pts,
        input_rgb=input_rgb,
        out_path=nms_panel,
        score_thr=args.score_thr,
    )

    from PIL import Image, ImageDraw

    orig = Image.open(original_panel).convert('RGBA')
    nms = Image.open(nms_panel).convert('RGBA')
    width = max(orig.size[0], nms.size[0])
    height = orig.size[1] + nms.size[1] + 40
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    canvas.paste(orig, (0, 0), orig)
    canvas.paste(nms, (0, orig.size[1] + 40), nms)
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'Original', fill=(0, 0, 0, 255))
    draw.text((16, orig.size[1] + 48), f'{mode_name}', fill=(0, 0, 0, 255))
    canvas.save(args.out)

    try:
        os.remove(original_panel)
    except OSError:
        pass
    try:
        os.remove(nms_panel)
    except OSError:
        pass

    print('saved:', args.out)


if __name__ == '__main__':
    main()