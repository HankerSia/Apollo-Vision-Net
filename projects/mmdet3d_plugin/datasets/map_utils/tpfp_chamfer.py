import warnings
from typing import Literal, Optional

import numpy as np
from scipy.spatial import distance
from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString
from shapely.strtree import STRtree

try:
    # Shapely>=1.8
    from shapely.errors import ShapelyDeprecationWarning  # type: ignore
except Exception:  # pragma: no cover
    ShapelyDeprecationWarning = Warning


Metric = Literal['chamfer', 'iou']


def custom_polyline_score(
    pred_lines: np.ndarray,
    gt_lines: np.ndarray,
    linewidth: float = 1.0,
    metric: Metric = 'chamfer',
) -> np.ndarray:
    """Compute pairwise similarity scores between predicted and GT polylines.

    Protocol (MapTR compatible):
    - metric='chamfer': returns negative symmetric Chamfer distance, so higher is better.
    - metric='iou': returns IoU between buffered polylines, so higher is better.

    Args:
        pred_lines: (N, P, 2) float array.
        gt_lines: (M, P, 2) float array.
        linewidth: buffer radius (meters) used for intersection prefilter and IoU.
        metric: 'chamfer' or 'iou'.

    Returns:
        score_matrix: (N, M)
    """

    if metric == 'iou':
        linewidth = 1.0

    num_preds = int(pred_lines.shape[0])
    num_gts = int(gt_lines.shape[0])

    if metric == 'chamfer':
        score_matrix = np.full((num_preds, num_gts), -100.0, dtype=np.float64)
    elif metric == 'iou':
        score_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    else:
        raise ValueError(f'Unsupported metric: {metric}')

    if num_preds == 0 or num_gts == 0:
        return score_matrix

    pred_polys = [
        LineString(line).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for line in pred_lines
    ]
    gt_polys = [
        LineString(line).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for line in gt_lines
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
        tree = STRtree(pred_polys)

    # Shapely 1.8 (used here) supports tree.query(geom) returning geometries.
    index_by_id = {id(g): i for i, g in enumerate(pred_polys)}

    for gt_id, gt_poly in enumerate(gt_polys):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
            candidates = tree.query(gt_poly)

        for cand in candidates:
            if not cand.intersects(gt_poly):
                continue
            pred_id = index_by_id[id(cand)]

            if metric == 'chamfer':
                dist_mat = distance.cdist(pred_lines[pred_id], gt_lines[gt_id], 'euclidean')
                ab = dist_mat.min(axis=1).mean()
                ba = dist_mat.min(axis=0).mean()
                score_matrix[pred_id, gt_id] = -float((ab + ba) / 2.0)
            else:
                inter = cand.intersection(gt_poly).area
                union = cand.union(gt_poly).area
                score_matrix[pred_id, gt_id] = float(inter / union) if union > 0 else 0.0

    return score_matrix
