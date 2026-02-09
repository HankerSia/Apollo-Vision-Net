from __future__ import annotations

import numpy as np

from .tpfp_chamfer import custom_polyline_score


def custom_tpfp_gen(
    gen_lines: np.ndarray,
    gt_lines: np.ndarray,
    threshold: float = 0.5,
    metric: str = 'chamfer',
):
    """Compute per-detection TP/FP flags for one sample.

    This matches MapTR's evaluation protocol:
    - For chamfer: match if symmetric chamfer distance <= threshold.
    - For iou: match if IoU >= threshold.

    Args:
        gen_lines: (N, K+1) where last dim is score and the rest is flattened points.
        gt_lines: (M, K) flattened points.

    Returns:
        tp, fp: (N,), float32 arrays with 0/1.
    """

    num_gens = int(gen_lines.shape[0])
    num_gts = int(gt_lines.shape[0])

    tp = np.zeros((num_gens,), dtype=np.float32)
    fp = np.zeros((num_gens,), dtype=np.float32)

    if num_gens == 0:
        return tp, fp
    if num_gts == 0:
        fp[...] = 1.0
        return tp, fp

    gen_scores = gen_lines[:, -1]

    # reshape back to (N, P, 2)
    pred_pts = gen_lines[:, :-1].reshape(num_gens, -1, 2)
    gt_pts = gt_lines.reshape(num_gts, -1, 2)

    score_mat = custom_polyline_score(pred_pts, gt_pts, linewidth=2.0, metric=metric)  # (N, M)

    if metric == 'chamfer':
        # score is negative distance, so threshold becomes -thr
        match_thr = -float(threshold)
        matched = score_mat.max(axis=1) >= match_thr
    else:
        match_thr = float(threshold)
        matched = score_mat.max(axis=1) >= match_thr

    best_gt = score_mat.argmax(axis=1)

    # Sort by score descending
    order = np.argsort(-gen_scores)
    gt_covered = np.zeros((num_gts,), dtype=bool)

    for i in order.tolist():
        if not matched[i]:
            fp[i] = 1.0
            continue
        g = int(best_gt[i])
        if not gt_covered[g]:
            gt_covered[g] = True
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    return tp, fp
