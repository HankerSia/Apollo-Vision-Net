from __future__ import annotations

import json
import os
from functools import partial
from multiprocessing import Pool
from os import path as osp
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from shapely.geometry import LineString
from terminaltables import AsciiTable

from .tpfp import custom_tpfp_gen


def _format_res_gt_by_class_worker(args):
    """Multiprocessing worker for `format_res_gt_by_classes`.

    Must be top-level (picklable).
    """

    (
        class_id,
        gen_results,
        annotations,
        num_fixed_sample_pts,
        num_pred_pts_per_instance,
        eval_use_same_gt_sample_num_flag,
    ) = args

    out = [
        get_cls_results(
            g,
            a,
            num_sample=num_fixed_sample_pts,
            num_pred_pts_per_instance=num_pred_pts_per_instance,
            eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
            class_id=class_id,
        )
        for g, a in zip(gen_results, annotations)
    ]

    if len(out) == 0:
        gens, gts = (), ()
    else:
        gens, gts = zip(*out)
    return int(class_id), gens, gts


def average_precision(recalls: np.ndarray, precisions: np.ndarray, mode: str = 'area') -> np.ndarray:
    """Compute AP given precision-recall curve.

    This mirrors the common OpenMMLab implementation.
    """

    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros((num_scales,), dtype=np.float32)

    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))

        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])

        for i in range(num_scales):
            idx = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, idx + 1] - mrec[i, idx]) * mpre[i, idx + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                ap[i] += precs.max() if precs.size > 0 else 0.0
        ap /= 11.0
    else:
        raise ValueError('mode must be "area" or "11points"')

    return ap[0] if no_scale else ap


def _resample_line(pts: np.ndarray, num: int) -> np.ndarray:
    if num <= 0:
        raise ValueError(f'num must be > 0, got {num}')
    if pts.shape[0] == num:
        return pts

    if pts.shape[0] < 2:
        # Degenerate line: repeat the only point.
        p = pts[0] if pts.shape[0] == 1 else np.zeros((2,), dtype=np.float32)
        return np.repeat(p[None, :], num, axis=0)

    line = LineString(pts)
    if line.length <= 1e-6:
        return np.repeat(np.array(line.coords[0], dtype=np.float32)[None, :], num, axis=0)

    distances = np.linspace(0.0, float(line.length), num)
    sampled = np.array([list(line.interpolate(d).coords)[0] for d in distances], dtype=np.float32)
    return sampled


def get_cls_results(
    gen_results: dict,
    annotations: dict,
    num_sample: int = 100,
    num_pred_pts_per_instance: int = 20,
    eval_use_same_gt_sample_num_flag: bool = True,
    class_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-class predictions and GT for a single sample.

    Returns:
        cls_gens: (N, D+1), last dim is confidence score.
        cls_gts: (M, D)
    """

    pred_list: List[np.ndarray] = []
    pred_scores: List[float] = []

    for res in gen_results.get('vectors', []):
        if int(res.get('type', -1)) != int(class_id):
            continue
        pts = np.asarray(res.get('pts', []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue

        if eval_use_same_gt_sample_num_flag:
            sampled = _resample_line(pts, num_sample)
        else:
            # MapTR assumes fixed-length predictions. If not, resample to fixed.
            sampled = _resample_line(pts, num_pred_pts_per_instance)

        pred_list.append(sampled)
        pred_scores.append(float(res.get('confidence_level', 0.0)))

    if len(pred_list) == 0:
        d = (num_sample if eval_use_same_gt_sample_num_flag else num_pred_pts_per_instance) * 2
        cls_gens = np.zeros((0, d + 1), dtype=np.float32)
    else:
        arr = np.stack(pred_list, axis=0).reshape(len(pred_list), -1)
        scores = np.asarray(pred_scores, dtype=np.float32)[:, None]
        cls_gens = np.concatenate([arr, scores], axis=1)

    gt_list: List[np.ndarray] = []
    for ann in annotations.get('vectors', []):
        if int(ann.get('type', -1)) != int(class_id):
            continue
        pts = np.asarray(ann.get('pts', []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        sampled = _resample_line(pts, num_sample)
        gt_list.append(sampled)

    if len(gt_list) == 0:
        cls_gts = np.zeros((0, num_sample * 2), dtype=np.float32)
    else:
        cls_gts = np.stack(gt_list, axis=0).reshape(len(gt_list), -1)

    return cls_gens, cls_gts


def format_res_gt_by_classes(
    result_path: str,
    gen_results: Sequence[dict],
    annotations: Sequence[dict],
    cls_names: Sequence[str],
    num_pred_pts_per_instance: int = 20,
    eval_use_same_gt_sample_num_flag: bool = True,
    pc_range: Optional[Sequence[float]] = None,
    nproc: int = 8,
) -> Tuple[Dict[str, Tuple[np.ndarray, ...]], Dict[str, Tuple[np.ndarray, ...]]]:
    """Pre-format predictions and GT by class.

    Output format matches MapTR's evaluator inputs.

    Returns:
        cls_gens: dict[class_name] -> tuple(per-sample arrays)
        cls_gts:  dict[class_name] -> tuple(per-sample arrays)
    """

    del pc_range  # only used for visualization in MapTR; keep signature aligned

    assert len(gen_results) == len(annotations), 'Pred/GT must have same number of samples'

    num_fixed_sample_pts = 100

    # Keep a small cache file next to result json (optional, for debugging).
    output_dir = osp.join(*osp.split(result_path)[:-1])
    mmcv.mkdir_or_exist(output_dir)
    cache_path = osp.join(output_dir, 'cls_formatted.pkl')

    cls_gens: Dict[str, Tuple[np.ndarray, ...]] = {}
    cls_gts: Dict[str, Tuple[np.ndarray, ...]] = {}

    if nproc and nproc > 1:
        args = [
            (
                class_id,
                gen_results,
                annotations,
                num_fixed_sample_pts,
                num_pred_pts_per_instance,
                eval_use_same_gt_sample_num_flag,
            )
            for class_id in range(len(cls_names))
        ]
        with Pool(processes=nproc) as pool:
            out = pool.map(_format_res_gt_by_class_worker, args)
        # out is list[(class_id, gens, gts)]
        out = sorted(out, key=lambda x: x[0])
        for class_id, clsname in enumerate(cls_names):
            _, gens, gts = out[class_id]
            cls_gens[clsname] = gens
            cls_gts[clsname] = gts
    else:
        for class_id, clsname in enumerate(cls_names):
            _, gens, gts = _format_res_gt_by_class_worker(
                (
                    class_id,
                    gen_results,
                    annotations,
                    num_fixed_sample_pts,
                    num_pred_pts_per_instance,
                    eval_use_same_gt_sample_num_flag,
                )
            )
            cls_gens[clsname] = gens
            cls_gts[clsname] = gts

    mmcv.dump([cls_gens, cls_gts], cache_path)
    return cls_gens, cls_gts


def eval_map(
    gen_results: dict,
    annotations: dict,
    cls_gens: Dict[str, Tuple[np.ndarray, ...]],
    cls_gts: Dict[str, Tuple[np.ndarray, ...]],
    threshold: float,
    cls_names: Sequence[str],
    logger=None,
    pc_range: Optional[Sequence[float]] = None,
    metric: str = 'chamfer',
    num_pred_pts_per_instance: int = 20,
    nproc: int = 8,
):
    """Evaluate map mAP for a given threshold (MapTR protocol)."""

    del gen_results, annotations, pc_range, num_pred_pts_per_instance

    eval_results = []

    for clsname in cls_names:
        per_sample_gens = cls_gens[clsname]
        per_sample_gts = cls_gts[clsname]

        # Compute tp/fp for each sample
        tpfp_fn = partial(custom_tpfp_gen, threshold=threshold, metric=metric)

        if nproc and nproc > 1:
            with Pool(processes=nproc) as pool:
                tpfp = pool.starmap(tpfp_fn, zip(per_sample_gens, per_sample_gts))
        else:
            tpfp = [tpfp_fn(g, gt) for g, gt in zip(per_sample_gens, per_sample_gts)]

        tp_list, fp_list = zip(*tpfp) if len(tpfp) else ([], [])

        num_gts = int(sum(gt.shape[0] for gt in per_sample_gts))

        # Flatten detections across samples
        dets = np.vstack(per_sample_gens) if len(per_sample_gens) else np.zeros((0, 1), dtype=np.float32)
        num_dets = int(dets.shape[0])

        if num_dets == 0:
            recalls = np.zeros((0,), dtype=np.float32)
            precisions = np.zeros((0,), dtype=np.float32)
            ap = np.float32(0.0)
        else:
            scores = dets[:, -1]
            order = np.argsort(-scores)
            tp = np.hstack(tp_list)[order]
            fp = np.hstack(fp_list)[order]

            tp_cum = np.cumsum(tp, axis=0)
            fp_cum = np.cumsum(fp, axis=0)
            eps = np.finfo(np.float32).eps
            recalls = tp_cum / max(num_gts, eps)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, eps)
            ap = average_precision(recalls, precisions, mode='area')

        eval_results.append(
            {
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap,
            }
        )

    aps = [r['ap'] for r in eval_results if r['num_gts'] > 0]
    mean_ap = float(np.mean(aps)) if len(aps) else 0.0

    _print_map_summary(mean_ap, eval_results, class_name=list(cls_names), logger=logger)

    return mean_ap, eval_results


def _print_map_summary(mean_ap: float, results: List[dict], class_name: List[str], logger=None):
    if logger == 'silent':
        return

    header = ['class', 'gts', 'dets', 'recall', 'ap']
    table_data = [header]

    for name, res in zip(class_name, results):
        recall = float(res['recall'][-1]) if getattr(res['recall'], 'size', 0) > 0 else 0.0
        ap = float(res['ap']) if not isinstance(res['ap'], np.ndarray) else float(res['ap'].item())
        table_data.append([name, int(res['num_gts']), int(res['num_dets']), f'{recall:.3f}', f'{ap:.3f}'])

    table_data.append(['mAP', '', '', '', f'{mean_ap:.3f}'])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print('\n' + table.table)
