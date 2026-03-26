#!/usr/bin/env python3
# Example:
# python tools/analysis_tools/project_det_map_to_pv_single.py \
#   --data-root data/nuscenes \
#   --infos data/nuscenes/nuscenes_infos_temporal_val.pkl \
#   --det-results test/bev_tiny_det_map_apollo/Mon_Mar_16_14_58_05_2026/pts_bbox/results_nusc.json \
#   --map-results test/bev_tiny_det_map_apollo/Mon_Mar_16_14_58_05_2026/map_results.pkl \
#   --subset-index 20 \
#   --out-dir tools_outputs/mon_mar_16_145805_vis/projected_idx020_groundz

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import mmcv
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, view_points, BoxVisibility
from pyquaternion import Quaternion


CAM_ORDER = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT',
    'CAM_BACK',
    'CAM_BACK_RIGHT',
]

MAP_LABEL2NAME = {
    0: 'divider',
    1: 'ped_crossing',
    2: 'boundary',
}

MAP_LABEL2COLOR_BGR = {
    0: (255, 128, 0),
    1: (0, 165, 255),
    2: (0, 200, 0),
}

DET_NAME2COLOR_BGR = {
    'car': (0, 220, 0),
    'truck': (0, 180, 255),
    'construction_vehicle': (0, 120, 255),
    'bus': (255, 80, 80),
    'trailer': (255, 180, 0),
    'barrier': (180, 180, 0),
    'motorcycle': (255, 0, 255),
    'bicycle': (255, 0, 128),
    'pedestrian': (0, 0, 255),
    'traffic_cone': (80, 255, 255),
}

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Project one det+map sample back to camera images.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--infos', required=True)
    parser.add_argument('--det-results', required=True)
    parser.add_argument('--map-results', required=True)
    parser.add_argument('--subset-index', type=int, required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--det-score-thr', type=float, default=0.35)
    parser.add_argument('--map-score-thr', type=float, default=0.35)
    parser.add_argument('--map-topk', type=int, default=30)
    parser.add_argument('--map-thickness', type=int, default=3)
    parser.add_argument('--box-thickness', type=int, default=2)
    parser.add_argument(
        '--map-z',
        type=float,
        default=None,
        help='Map polyline z in lidar coordinates. Default uses ground-plane estimate: -lidar2ego_translation[2].',
    )
    return parser.parse_args()


def resolve_data_path(data_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    normalized = [part for part in path.parts if part not in ('', '.')]
    for start in range(len(normalized)):
        candidate = data_root.joinpath(*normalized[start:])
        if candidate.exists():
            return candidate
    return data_root.joinpath(*normalized)


def build_token_to_info_index(infos: List[dict]) -> Dict[str, int]:
    return {info['token']: idx for idx, info in enumerate(infos)}


def lidar2img_from_cam_info(cam_info: dict) -> np.ndarray:
    lidar2cam_r = np.linalg.inv(np.asarray(cam_info['sensor2lidar_rotation'], dtype=np.float64))
    lidar2cam_t = np.asarray(cam_info['sensor2lidar_translation'], dtype=np.float64) @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4, dtype=np.float64)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam_rt.T


def resample_polyline(points_xy: np.ndarray, sample_dist: float = 0.2) -> np.ndarray:
    if points_xy.shape[0] < 2:
        return points_xy
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    total = float(seg.sum())
    if total <= 1e-6:
        return points_xy
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.arange(0.0, total + sample_dist, sample_dist, dtype=np.float64)
    out = []
    for target in targets:
        idx = int(np.searchsorted(cum, target, side='right') - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        local = target - cum[idx]
        ratio = 0.0 if seg[idx] <= 1e-6 else local / seg[idx]
        pt = points_xy[idx] + ratio * (points_xy[idx + 1] - points_xy[idx])
        out.append(pt)
    return np.asarray(out, dtype=np.float64)


def split_visible_polyline(uv: np.ndarray, valid: np.ndarray) -> Iterable[np.ndarray]:
    chunk: List[np.ndarray] = []
    for point, is_valid in zip(uv, valid):
        if is_valid:
            chunk.append(point)
            continue
        if len(chunk) >= 2:
            yield np.asarray(chunk, dtype=np.int32)
        chunk = []
    if len(chunk) >= 2:
        yield np.asarray(chunk, dtype=np.int32)


def draw_map_polyline_on_image(
    image: np.ndarray,
    polyline_xy: np.ndarray,
    lidar2img: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    z_value: float,
) -> None:
    polyline_xy = np.asarray(polyline_xy, dtype=np.float64)
    if polyline_xy.ndim != 2 or polyline_xy.shape[0] < 2:
        return
    if polyline_xy.shape[1] == 2:
        polyline_xyz = np.concatenate(
            [polyline_xy, np.full((polyline_xy.shape[0], 1), z_value, dtype=np.float64)],
            axis=1,
        )
    else:
        polyline_xyz = polyline_xy[:, :3]
    sampled = resample_polyline(polyline_xyz[:, :2], sample_dist=0.2)
    if sampled.shape[0] < 2:
        return
    sampled_xyz = np.concatenate(
        [sampled, np.full((sampled.shape[0], 1), z_value, dtype=np.float64)],
        axis=1,
    )
    pts_4d = np.concatenate([sampled_xyz, np.ones((sampled_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (lidar2img @ pts_4d.T).T
    depth = proj[:, 2]
    uv = proj[:, :2] / np.clip(depth[:, None], a_min=1e-6, a_max=None)
    h, w = image.shape[:2]
    valid = (
        (depth > 1e-6)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < w - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < h - 1)
    )
    for chunk in split_visible_polyline(np.round(uv).astype(np.int32), valid):
        cv2.polylines(image, [chunk], False, color, thickness, lineType=cv2.LINE_AA)


def detection_box_from_result(det: dict) -> Box:
    velocity = det.get('velocity', [0.0, 0.0])
    return Box(
        center=np.asarray(det['translation'], dtype=np.float64),
        size=np.asarray(det['size'], dtype=np.float64),
        orientation=Quaternion(det['rotation']),
        label=np.nan,
        score=float(det.get('detection_score', 0.0)),
        velocity=np.asarray([velocity[0], velocity[1], 0.0], dtype=np.float64),
        name=det.get('detection_name', 'object'),
    )


def transform_box_global_to_camera(box: Box, cam_info: dict) -> Box:
    box = Box(box.center.copy(), box.wlh.copy(), Quaternion(box.orientation.elements), box.label, box.score, box.velocity.copy(), box.name, box.token)
    box.translate(-np.asarray(cam_info['ego2global_translation'], dtype=np.float64))
    box.rotate(Quaternion(cam_info['ego2global_rotation']).inverse)
    box.translate(-np.asarray(cam_info['sensor2ego_translation'], dtype=np.float64))
    box.rotate(Quaternion(cam_info['sensor2ego_rotation']).inverse)
    return box


def draw_detection_box(
    image: np.ndarray,
    box: Box,
    intrinsic: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    label: str,
) -> None:
    corners = box.corners()
    if np.all(corners[2, :] <= 1e-6):
        return
    corners_2d = view_points(corners, intrinsic, normalize=True)[:2, :].T
    corners_2d = np.round(corners_2d).astype(np.int32)
    for i0, i1 in BOX_EDGES:
        pt0 = tuple(corners_2d[i0])
        pt1 = tuple(corners_2d[i1])
        cv2.line(image, pt0, pt1, color, thickness, lineType=cv2.LINE_AA)
    text_org = tuple(corners_2d[0] + np.array([2, -4]))
    cv2.putText(image, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def make_mosaic(images_by_cam: Dict[str, np.ndarray]) -> np.ndarray:
    panels = []
    heights = [img.shape[0] for img in images_by_cam.values()]
    target_h = min(heights)
    for cam in CAM_ORDER:
        img = images_by_cam[cam]
        scale = target_h / img.shape[0]
        target_w = max(1, int(round(img.shape[1] * scale)))
        panel = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (target_w, 36), (255, 255, 255), -1)
        cv2.putText(panel, cam, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(panel)
    col_ws = [max(panels[i].shape[1] for i in pair) for pair in ((0, 3), (1, 4), (2, 5))]

    def pad_w(img: np.ndarray, width: int) -> np.ndarray:
        if img.shape[1] >= width:
            return img
        pad = np.full((img.shape[0], width - img.shape[1], 3), 255, dtype=np.uint8)
        return np.concatenate([img, pad], axis=1)

    top = np.concatenate([pad_w(panels[0], col_ws[0]), pad_w(panels[1], col_ws[1]), pad_w(panels[2], col_ws[2])], axis=1)
    bottom = np.concatenate([pad_w(panels[3], col_ws[0]), pad_w(panels[4], col_ws[1]), pad_w(panels[5], col_ws[2])], axis=1)
    return np.concatenate([top, bottom], axis=0)


def project_one_index(
    subset_index: int,
    args: argparse.Namespace,
    infos: List[dict],
    token_to_info_idx: Dict[str, int],
    det_results: Dict[str, List[dict]],
    map_results: List[dict],
    out_dir: Path,
) -> dict:
    cam_out_dir = out_dir / 'cams'
    cam_out_dir.mkdir(parents=True, exist_ok=True)

    det_tokens = list(det_results.keys())
    if subset_index < 0 or subset_index >= len(det_tokens):
        raise IndexError(f'subset-index={subset_index} out of range for det results ({len(det_tokens)})')

    token = det_tokens[subset_index]
    if token not in token_to_info_idx:
        raise KeyError(f'token {token} not found in infos file')
    info_idx = token_to_info_idx[token]
    info = infos[info_idx]
    map_z = float(args.map_z) if args.map_z is not None else -float(info['lidar2ego_translation'][2])

    if subset_index < 0 or subset_index >= len(map_results):
        raise IndexError(f'subset-index={subset_index} out of range for map results ({len(map_results)})')
    map_pred = map_results[subset_index]

    map_scores = np.asarray(map_pred['scores'], dtype=np.float32)
    map_labels = np.asarray(map_pred['labels'], dtype=np.int64)
    map_vectors = np.asarray(map_pred['vectors'], dtype=np.float32)
    keep = map_scores >= float(args.map_score_thr)
    if args.map_topk > 0 and keep.any():
        keep_idx = np.where(keep)[0]
        top_idx = keep_idx[np.argsort(map_scores[keep_idx])[::-1][:args.map_topk]]
        keep = np.zeros_like(keep, dtype=bool)
        keep[top_idx] = True

    det_entries = [det for det in det_results[token] if float(det.get('detection_score', 0.0)) >= float(args.det_score_thr)]
    det_boxes = [detection_box_from_result(det) for det in det_entries]

    images_by_cam: Dict[str, np.ndarray] = {}
    metadata = {
        'subset_index': subset_index,
        'current_val_index': info_idx,
        'token': token,
        'scene_name': info.get('scene_name'),
        'map_location': info.get('map_location'),
        'det_kept': len(det_boxes),
        'map_kept': int(keep.sum()),
        'map_z_lidar': map_z,
        'cameras': {},
    }

    for cam in CAM_ORDER:
        cam_info = info['cams'][cam]
        img_path = resolve_data_path(Path(args.data_root), cam_info['data_path'])
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'failed to load image: {img_path}')

        lidar2img = lidar2img_from_cam_info(cam_info)
        intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float64)
        imsize = (image.shape[1], image.shape[0])

        for det, box in zip(det_entries, det_boxes):
            cam_box = transform_box_global_to_camera(box, cam_info)
            if not box_in_image(cam_box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
                continue
            color = DET_NAME2COLOR_BGR.get(det['detection_name'], (0, 255, 255))
            label = f"{det['detection_name']}:{float(det['detection_score']):.2f}"
            draw_detection_box(image, cam_box, intrinsic, color, args.box_thickness, label)

        for idx in np.where(keep)[0]:
            color = MAP_LABEL2COLOR_BGR.get(int(map_labels[idx]), (255, 255, 0))
            draw_map_polyline_on_image(image, map_vectors[idx], lidar2img, color, args.map_thickness, map_z)

        out_path = cam_out_dir / f'{cam}.jpg'
        cv2.imwrite(str(out_path), image)
        images_by_cam[cam] = image
        metadata['cameras'][cam] = {
            'input_image': str(img_path),
            'output_image': str(out_path),
        }

    mosaic = make_mosaic(images_by_cam)
    mosaic_path = out_dir / 'projected_idx_overlay_grid.jpg'
    cv2.imwrite(str(mosaic_path), mosaic)
    metadata['mosaic'] = str(mosaic_path)

    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f'saved mosaic: {mosaic_path}')
    print(f'saved cams dir: {cam_out_dir}')
    print(f'subset_index={subset_index} token={token} current_val_index={info_idx}')
    return metadata


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    infos = mmcv.load(args.infos)['infos']
    token_to_info_idx = build_token_to_info_index(infos)

    det_results = json.load(open(args.det_results, 'r', encoding='utf-8'))['results']
    map_results = mmcv.load(args.map_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_one_index(
        subset_index=args.subset_index,
        args=args,
        infos=infos,
        token_to_info_idx=token_to_info_idx,
        det_results=det_results,
        map_results=map_results,
        out_dir=out_dir,
    )


if __name__ == '__main__':
    main()