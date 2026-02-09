"""Map-vector evaluation utilities (MapTR protocol compatible).

This package provides a lightweight, self-contained implementation of the
vector-map mAP evaluation used by MapTR (Chamfer / IoU matching).

The goal is protocol alignment rather than speed.
"""

from .mean_ap import eval_map, format_res_gt_by_classes  # noqa: F401
