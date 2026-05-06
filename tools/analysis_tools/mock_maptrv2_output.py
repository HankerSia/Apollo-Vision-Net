#!/usr/bin/env python3
"""Generate a GT-based mock MapTRv2 output bundle.

Example:
    python tools/analysis_tools/mock_maptrv2_output.py \
        --data-root data/nuscenes \
        --infos data/nuscenes/nuscenes_infos_temporal_val.pkl \
        --index 0 \
        --output-dir tools_outputs/mock_maptrv2_from_gt

Outputs:
    - mock_maptrv2_head_outputs.pt
    - mock_nuscmap_results.json
    - mock_metadata.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mmcv
import numpy as np
import torch
from pyquaternion import Quaternion


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.datasets.nuscenes_det_mapv2_dataset import VectorizedLocalMapV2
from projects.mmdet3d_plugin.datasets.nuscenes_det_occ_map_dataset import _scene_name_to_log_location


LABEL2NAME = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
    3: 'centerline',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a GT-based mock MapTRv2 output bundle.')
    parser.add_argument('--data-root', type=Path, default=Path('data/nuscenes-mini'), help='nuScenes data root.')
    parser.add_argument('--infos', type=Path, default=Path('data/nuscenes-mini/nuscenes_with_maploc_infos_temporal_val.pkl'), help='Temporal infos pkl.')
    parser.add_argument('--index', type=int, default=0, help='Sample index in infos.')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory to save mock outputs.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch-size', type=int, default=1, help='Mock batch size.')
    parser.add_argument('--num-decoder-layers', type=int, default=6, help='Number of decoder layers.')
    parser.add_argument('--num-vec-one2one', type=int, default=50, help='Number of one-to-one vectors.')
    parser.add_argument('--num-vec-one2many', type=int, default=300, help='Number of one-to-many vectors.')
    parser.add_argument('--num-pts', type=int, default=20, help='Points per vector.')
    parser.add_argument('--bev-h', type=int, default=50, help='Mock BEV height.')
    parser.add_argument('--bev-w', type=int, default=50, help='Mock BEV width.')
    parser.add_argument('--embed-dims', type=int, default=256, help='Embedding channels.')
    parser.add_argument('--num-cams', type=int, default=6, help='Camera count for mock pv seg.')
    parser.add_argument('--feat-h', type=int, default=32, help='PV feature height.')
    parser.add_argument('--feat-w', type=int, default=88, help='PV feature width.')
    parser.add_argument('--score-thr', type=float, default=0.35, help='Confidence threshold for exported vectors.')
    parser.add_argument('--jitter-std', type=float, default=0.0, help='Std of per-point xy jitter in meters.')
    parser.add_argument('--layer-decay', type=float, default=0.92, help='Per-layer decay for logits and geometric quality.')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Probability of dropping a GT vector from the exported result.')
    parser.add_argument('--false-positive-count', type=int, default=0, help='How many low-confidence false positives to inject.')
    parser.add_argument('--map-range', type=float, nargs=4, default=[-50.0, -50.0, 50.0, 50.0], metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'), help='BEV xy range for normalization.')
    return parser.parse_args()


def build_lidar2global(info: dict) -> np.ndarray:
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info['lidar2ego_translation'], dtype=np.float64)

    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(info['ego2global_translation'], dtype=np.float64)
    return ego2global @ lidar2ego


def resolve_map_location(info: dict, data_root: Path) -> str:
    location = info.get('map_location')
    if location:
        return location
    scene_name = info.get('scene_name')
    if not scene_name:
        raise KeyError('Missing both map_location and scene_name in infos record.')
    location = _scene_name_to_log_location(scene_name, dataroot=str(data_root), version='v1.0-trainval')
    if not location:
        raise KeyError(f'Unable to infer map location from scene_name={scene_name!r}')
    return location


def load_gt_sample(args: argparse.Namespace) -> Tuple[dict, np.ndarray, np.ndarray]:
    infos_obj = mmcv.load(str(args.infos))
    infos = infos_obj['infos'] if isinstance(infos_obj, dict) and 'infos' in infos_obj else infos_obj
    info = infos[args.index]
    map_location = resolve_map_location(info, args.data_root)
    lidar2global = build_lidar2global(info)

    vmap = VectorizedLocalMapV2(
        dataroot=str(args.data_root),
        patch_size=(100.0, 100.0),
        map_classes=('divider', 'ped_crossing', 'boundary', 'centerline'),
        fixed_ptsnum_per_line=args.num_pts,
    )
    anns = vmap.gen_vectorized_samples(
        location=map_location,
        lidar2global_translation=list(lidar2global[:3, 3]),
        lidar2global_rotation=list(Quaternion(matrix=lidar2global).q),
    )
    labels = np.asarray(anns['gt_vecs_label'], dtype=np.int64)
    gt_obj = anns['gt_vecs_pts_loc']
    if len(labels) > 0:
        pts = gt_obj.fixed_num_sampled_points.cpu().numpy().astype(np.float32)
    else:
        pts = np.zeros((0, args.num_pts, 2), dtype=np.float32)
    info = dict(info)
    info['map_location'] = map_location
    return info, labels, pts


def normalize_points(points_xy: np.ndarray, map_range: Sequence[float]) -> np.ndarray:
    xmin, ymin, xmax, ymax = [float(value) for value in map_range]
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)
    out = points_xy.copy()
    out[..., 0] = (out[..., 0] - xmin) / width
    out[..., 1] = (out[..., 1] - ymin) / height
    return np.clip(out, 0.0, 1.0)


def bbox_from_points_norm(points_norm: np.ndarray) -> np.ndarray:
    xmin = points_norm[:, 0].min()
    xmax = points_norm[:, 0].max()
    ymin = points_norm[:, 1].min()
    ymax = points_norm[:, 1].max()
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, xmax - xmin, ymax - ymin], dtype=np.float32)


def add_point_noise(points_xy: np.ndarray, rng: np.random.Generator, jitter_std: float) -> np.ndarray:
    noise = rng.normal(0.0, jitter_std, size=points_xy.shape).astype(np.float32)
    return points_xy + noise


def make_false_positive(class_id: int, num_pts: int, rng: np.random.Generator) -> np.ndarray:
    center_x = float(rng.uniform(-25.0, 25.0))
    center_y = float(rng.uniform(-25.0, 25.0))
    drift = float(rng.uniform(-4.0, 4.0))
    t = np.linspace(-1.0, 1.0, num_pts, dtype=np.float32)
    if class_id == 1:
        width = float(rng.uniform(2.0, 5.0))
        height = float(rng.uniform(1.5, 3.0))
        x = center_x + width * np.sign(np.sin(np.pi * t))
        y = center_y + height * t
    elif class_id == 3:
        x = center_x + 12.0 * t
        y = center_y + 2.0 * np.sin(np.pi * t) + drift
    else:
        x = center_x + 10.0 * t
        y = center_y + drift + 0.5 * np.sin(2.0 * np.pi * t)
    return np.stack([x, y], axis=-1).astype(np.float32)


def rasterize_bev(points_list: List[np.ndarray], bev_h: int, bev_w: int, map_range: Sequence[float]) -> torch.Tensor:
    mask = np.zeros((1, bev_h, bev_w), dtype=np.float32)
    xmin, ymin, xmax, ymax = [float(value) for value in map_range]
    width = max(xmax - xmin, 1e-6)
    height = max(ymax - ymin, 1e-6)
    radius = 1
    for points in points_list:
        gx = np.clip(np.round((points[:, 0] - xmin) / width * (bev_w - 1)).astype(np.int64), 0, bev_w - 1)
        gy = np.clip(np.round((points[:, 1] - ymin) / height * (bev_h - 1)).astype(np.int64), 0, bev_h - 1)
        for idx in range(max(len(gx) - 1, 0)):
            x0, y0 = int(gx[idx]), int(gy[idx])
            x1, y1 = int(gx[idx + 1]), int(gy[idx + 1])
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                xx = int(round(x0 + (x1 - x0) * step / steps))
                yy = int(round(y0 + (y1 - y0) * step / steps))
                mask[0, max(yy - radius, 0): min(yy + radius + 1, bev_h), max(xx - radius, 0): min(xx + radius + 1, bev_w)] = 1.0
    return torch.from_numpy(mask)


def make_mock_outputs(args: argparse.Namespace, info: dict, gt_labels: np.ndarray, gt_pts: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    num_layers = args.num_decoder_layers
    num_classes = len(LABEL2NAME)
    one2one = args.num_vec_one2one
    one2many = args.num_vec_one2many
    num_pts = args.num_pts

    all_cls_scores = torch.full((num_layers, args.batch_size, one2one, num_classes), -8.0, dtype=torch.float32)
    all_bbox_preds = torch.zeros((num_layers, args.batch_size, one2one, 4), dtype=torch.float32)
    all_pts_preds = torch.zeros((num_layers, args.batch_size, one2one, num_pts, 2), dtype=torch.float32)
    one2many_cls = torch.full((num_layers, args.batch_size, one2many, num_classes), -8.5, dtype=torch.float32)
    one2many_bbox = torch.zeros((num_layers, args.batch_size, one2many, 4), dtype=torch.float32)
    one2many_pts = torch.zeros((num_layers, args.batch_size, one2many, num_pts, 2), dtype=torch.float32)

    kept_vectors: List[Dict] = []
    one2one_xy: List[np.ndarray] = []

    gt_count = min(len(gt_labels), one2one)
    for vec_index in range(gt_count):
        label = int(gt_labels[vec_index])
        gt_xy = gt_pts[vec_index]
        dropped = bool(rng.random() < args.drop_prob)
        if dropped:
            continue

        noisy_xy = add_point_noise(gt_xy, rng, args.jitter_std)
        one2one_xy.append(noisy_xy)
        score_base = float(np.clip(rng.uniform(0.82, 0.97), 0.0, 0.999))
        kept_vectors.append(
            {
                'pts': np.round(noisy_xy, 3).tolist(),
                'pts_num': int(noisy_xy.shape[0]),
                'cls_name': LABEL2NAME[label],
                'type': label,
                'confidence_level': round(score_base, 4),
            }
        )

        for layer in range(num_layers):
            layer_quality = args.layer_decay ** float(num_layers - layer - 1)
            layer_xy = gt_xy + (noisy_xy - gt_xy) * layer_quality
            layer_norm = normalize_points(layer_xy, args.map_range)
            layer_bbox = bbox_from_points_norm(layer_norm)
            logit = np.log((score_base * layer_quality) / max(1.0 - score_base * layer_quality, 1e-4))
            all_cls_scores[layer, 0, vec_index, label] = float(logit)
            all_bbox_preds[layer, 0, vec_index] = torch.from_numpy(layer_bbox)
            all_pts_preds[layer, 0, vec_index] = torch.from_numpy(layer_norm)

    for fp_index in range(min(args.false_positive_count, max(one2one - len(kept_vectors), 0))):
        vec_index = gt_count + fp_index
        if vec_index >= one2one:
            break
        label = int(rng.integers(0, num_classes))
        fp_xy = make_false_positive(label, num_pts, rng)
        fp_score = float(rng.uniform(max(0.05, args.score_thr - 0.2), args.score_thr + 0.02))
        if fp_score >= args.score_thr:
            kept_vectors.append(
                {
                    'pts': np.round(fp_xy, 3).tolist(),
                    'pts_num': int(fp_xy.shape[0]),
                    'cls_name': LABEL2NAME[label],
                    'type': label,
                    'confidence_level': round(fp_score, 4),
                }
            )
        for layer in range(num_layers):
            layer_norm = normalize_points(fp_xy, args.map_range)
            one_fp_bbox = bbox_from_points_norm(layer_norm)
            logit = np.log(fp_score / max(1.0 - fp_score, 1e-4)) - 0.15 * float(num_layers - layer - 1)
            all_cls_scores[layer, 0, vec_index, label] = float(logit)
            all_bbox_preds[layer, 0, vec_index] = torch.from_numpy(one_fp_bbox)
            all_pts_preds[layer, 0, vec_index] = torch.from_numpy(layer_norm)

    total_o2m = min(one2many, max(len(kept_vectors) * 6, len(kept_vectors)))
    for vec_index in range(total_o2m):
        src = kept_vectors[vec_index % max(len(kept_vectors), 1)] if kept_vectors else None
        if src is None:
            break
        label = int(src['type'])
        src_xy = np.asarray(src['pts'], dtype=np.float32)
        noisy_xy = add_point_noise(src_xy, rng, args.jitter_std * 1.6)
        for layer in range(num_layers):
            layer_norm = normalize_points(noisy_xy, args.map_range)
            layer_bbox = bbox_from_points_norm(layer_norm)
            score = max(float(src['confidence_level']) - 0.12 - 0.03 * (vec_index % 6), 0.05)
            logit = np.log(score / max(1.0 - score, 1e-4)) - 0.1 * float(num_layers - layer - 1)
            one2many_cls[layer, 0, vec_index, label] = float(logit)
            one2many_bbox[layer, 0, vec_index] = torch.from_numpy(layer_bbox)
            one2many_pts[layer, 0, vec_index] = torch.from_numpy(layer_norm)

    bev_embed = torch.randn((args.bev_h * args.bev_w, args.batch_size, args.embed_dims), dtype=torch.float32)
    seg = rasterize_bev(one2one_xy, args.bev_h, args.bev_w, args.map_range).unsqueeze(0)
    pv_seg = torch.zeros((args.batch_size, args.num_cams, 1, args.feat_h, args.feat_w), dtype=torch.float32)

    head_outputs = {
        'bev_embed': bev_embed,
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'all_pts_preds': all_pts_preds,
        'enc_cls_scores': None,
        'enc_bbox_preds': None,
        'enc_pts_preds': None,
        'seg': seg,
        'pv_seg': pv_seg,
        'one2many_outs': {
            'all_cls_scores': one2many_cls,
            'all_bbox_preds': one2many_bbox,
            'all_pts_preds': one2many_pts,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
            'seg': None,
            'pv_seg': None,
        },
    }

    export_json = {
        'meta': {
            'use_lidar': False,
            'use_camera': True,
            'use_radar': False,
            'use_map': True,
            'use_external': True,
        },
        'results': [
            {
                'sample_token': info['token'],
                'vectors': [vector for vector in kept_vectors if float(vector['confidence_level']) >= args.score_thr],
            }
        ],
    }

    metadata = {
        'source_infos': str(args.infos),
        'source_index': int(args.index),
        'sample_token': info['token'],
        'scene_name': info.get('scene_name'),
        'map_location': info.get('map_location'),
        'gt_vectors_total': int(len(gt_labels)),
        'exported_vectors': int(len(export_json['results'][0]['vectors'])),
        'num_vec_one2one': int(args.num_vec_one2one),
        'num_vec_one2many': int(args.num_vec_one2many),
        'num_pts': int(args.num_pts),
        'jitter_std': float(args.jitter_std),
        'drop_prob': float(args.drop_prob),
        'false_positive_count': int(args.false_positive_count),
        'score_thr': float(args.score_thr),
    }
    return head_outputs, export_json, metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info, gt_labels, gt_pts = load_gt_sample(args)
    head_outputs, export_json, metadata = make_mock_outputs(args, info, gt_labels, gt_pts)

    pt_path = args.output_dir / 'mock_maptrv2_head_outputs.pt'
    json_path = args.output_dir / 'mock_nuscmap_results.json'
    meta_path = args.output_dir / 'mock_metadata.json'

    torch.save(head_outputs, pt_path)
    json_path.write_text(json.dumps(export_json, indent=2), encoding='utf-8')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'saved: {pt_path}')
    print(f'saved: {json_path}')
    print(f'saved: {meta_path}')
    print(f"sample_token: {metadata['sample_token']}")
    print(f"map_location: {metadata['map_location']}")
    print(f"gt_vectors_total: {metadata['gt_vectors_total']}")
    print(f"exported_vectors: {metadata['exported_vectors']}")


if __name__ == '__main__':
    main()