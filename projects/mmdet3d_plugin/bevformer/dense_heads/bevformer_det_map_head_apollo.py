# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

"""Det+Map head (Apollo)

This is a minimal, low-impact scaffold to replace the existing occ head with a
MapTR-aligned vector-map head while keeping BEVFormer temporal `prev_bev`
mechanism unchanged.

Design goals:
- New file only (no modification to existing core logic).
- Provide the same critical output key `bev_embed` used by BEVFormer for
  temporal caching.
- Expose MapTR-style hooks (`get_map_results`) in a way that's easy to extend.

NOTE: The map branch is intentionally stubbed in this first milestone. It will
be filled with MapTR's query/decoder + vectorization logic once data/pipeline is
wired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:
    linear_sum_assignment = None

try:
    from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString  # type: ignore
except Exception:
    LineString = None

try:
    from shapely.strtree import STRtree  # type: ignore
except Exception:
    STRtree = None

try:
    # Shapely>=1.8
    from shapely.errors import ShapelyDeprecationWarning  # type: ignore
except Exception:
    ShapelyDeprecationWarning = Warning

from mmcv.runner import auto_fp16, get_dist_info
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.utils import build_from_cfg
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid

from .bevformer_head import BEVFormerHead
from ...maptr.dense_heads.maptr_loss_head import MapTRLossHead, pts_to_bbox_normalized


@HEADS.register_module()
class BEVFormerDetMapHeadApollo(BEVFormerHead):
    """A minimal det+map head that *only* guarantees BEV extraction and a
    predictable inference output contract.

    Expected by BEVFormer detector:
    - Calling the head like `outs = head(img_feats, img_metas, prev_bev)`.
    - `outs` must contain `bev_embed` (Tensor of shape [bs, bev_h*bev_w, C] or
      equivalent that the detector caches as `prev_bev`).

    For now we keep det/map predictions empty to avoid breaking training/test
    scripts while we wire map GT + evaluator.
    """

    def __init__(
        self,
        *args,
        transformer: Dict[str, Any],
        bev_h: int,
        bev_w: int,
        embed_dims: int = 256,
        positional_encoding: Optional[Dict[str, Any]] = None,
        real_h: Optional[float] = None,
        real_w: Optional[float] = None,
        enable_det: bool = True,
        enable_map: bool = True,
        **kwargs,
    ) -> None:
        """Det+Map head.

        This head is meant to be det-evaluable by reusing the standard
        `BEVFormerHead` (a `DETRHead`) detection pipeline, while adding a map
        branch for MapTR-style vector map prediction.

        Note: The BEVFormer detector in this repo calls `pts_bbox_head.loss`
        with a "joint" signature (including `pts_feats/occ_gts/flow_gts/outs`).
        We keep a wide `loss(...)` below to stay compatible.
        """

        # --- Transformer config massage ---
        # Our config uses `det_decoder` + `map_decoder`. `BEVFormerHead` expects
        # `decoder` inside its transformer cfg. Also, `BEVFormerHead` does not
        # know `map_decoder`, so we pop it before calling super().__init__.
        # Pop det_map-only keys that upstream DETRHead/BEVFormerHead doesn't accept.
        # Keep backward-compatible config keys (`enabling_*`).
        if 'enabling_det' in kwargs and ('enable_det' not in kwargs):
            enable_det = bool(kwargs.pop('enabling_det'))
        else:
            kwargs.pop('enabling_det', None)
        if 'enabling_map' in kwargs and ('enable_map' not in kwargs):
            enable_map = bool(kwargs.pop('enabling_map'))
        else:
            kwargs.pop('enabling_map', None)

        occupancy_size = kwargs.pop('occupancy_size', None)
        point_cloud_range = kwargs.pop('point_cloud_range', None)

        # MapTR-official loss/target/assigner path (strategy B).
        self.map_loss_impl = str(kwargs.pop('map_loss_impl', 'apollo_simple'))
        map_gt_shift_pts_pattern = str(kwargs.pop('map_gt_shift_pts_pattern', 'v2'))
        map_dir_interval = int(kwargs.pop('map_dir_interval', 1))
        map_sync_cls_avg_factor = bool(kwargs.pop('map_sync_cls_avg_factor', True))
        map_bg_cls_weight = float(kwargs.pop('map_bg_cls_weight', 0.1))
        map_loss_cls_cfg = kwargs.pop('map_loss_cls', None)
        map_loss_bbox_cfg = kwargs.pop('map_loss_bbox', None)
        map_loss_iou_cfg = kwargs.pop('map_loss_iou', None)
        map_loss_pts_cfg = kwargs.pop('map_loss_pts', None)
        map_loss_dir_cfg = kwargs.pop('map_loss_dir', None)
        map_assigner_cfg = kwargs.pop('map_assigner', None)
        map_sampler_cfg = kwargs.pop('map_sampler', None)

        transformer_cfg = dict(transformer)
        map_decoder_cfg = transformer_cfg.pop('map_decoder', None)
        det_decoder_cfg = transformer_cfg.pop('det_decoder', None)
        if ('decoder' not in transformer_cfg) and (det_decoder_cfg is not None):
            transformer_cfg['decoder'] = det_decoder_cfg

        self.enable_det = bool(enable_det)
        self.enable_map = bool(enable_map)

        # Initialize the detection part via the official BEVFormerHead.
        # This builds encoder+decoder, query embedding, bbox coder, det losses.
        super().__init__(
            *args,
            transformer=transformer_cfg,
            bev_h=bev_h,
            bev_w=bev_w,
            **kwargs,
        )

        # Keep explicit values for map branch; if not provided, reuse BEVFormerHead's.
        # (BEVFormerHead computes self.real_h/self.real_w from bbox_coder.pc_range.)
        if real_h is not None:
            self.real_h = float(real_h)
        if real_w is not None:
            self.real_w = float(real_w)

        # Optional point cloud / map range for MapTR assigner.
        # Prefer explicit config value; else reuse det pc_range if available.
        if point_cloud_range is None:
            point_cloud_range = getattr(self, 'pc_range', None)
        self.point_cloud_range = point_cloud_range

        # Kept for compatibility with earlier configs.
        self.occupancy_size = occupancy_size

        # --- Map decoder (optional) ---
        self.map_decoder_num_layers = int(map_decoder_cfg.get('num_layers', 1)) if isinstance(map_decoder_cfg, dict) else 1
        self.map_decoder = None
        if map_decoder_cfg is not None:
            try:
                self.map_decoder = build_from_cfg(map_decoder_cfg, TRANSFORMER_LAYER_SEQUENCE)
            except Exception:
                self.map_decoder = None

        # MapTR-head style switches.
        # - Official MapTR uses point-level queries: num_query = num_vec * num_pts_per_vec.
        # - Decoder refines reference points only when reg_branches is passed.
        self.map_with_box_refine = bool(kwargs.get('map_with_box_refine', True))
        self.map_use_point_queries = bool(kwargs.get('map_use_point_queries', self.map_loss_impl == 'maptr_official'))
        self.map_query_embed_type = str(kwargs.get('map_query_embed_type', 'instance_pts'))

        # MapTR-style learned reference points for map queries.
        # This mirrors the DETR/BEVFormer pattern where reference_points are
        # predicted from query_pos then refined layer-by-layer.
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
        nn.init.xavier_uniform_(self.map_reference_points.weight)
        nn.init.constant_(self.map_reference_points.bias, 0.0)

        # --- Map branch (MapTR-aligned, minimal) ---
        # We predict a fixed number of vector instances per sample.
        self.num_map_vec = int(kwargs.get('num_map_vec', 50))
        # Prefer config-provided fixed points number; otherwise default to 20.
        self.map_num_pts = int(kwargs.get('map_num_pts', kwargs.get('fixed_ptsnum_per_line', 20)))
        self.map_num_classes = int(kwargs.get('map_num_classes', 3))

        # Point-level query protocol (official MapTRHead): num_query = num_vec * num_pts_per_vec
        self.map_num_query = int(self.num_map_vec * self.map_num_pts)

        # Map points decoding: normalize raw outputs to metric coordinates.
        # Default: sigmoid -> [0,1] then scale to pc_range in meters.
        self.map_pts_normalize = str(kwargs.get('map_pts_normalize', 'sigmoid'))

        # Map matching/loss settings (MapTR-style: one-to-one matching + focal cls).
        self.map_matcher_type = str(kwargs.get('map_matcher_type', 'hungarian'))
        self.map_cost_cls = float(kwargs.get('map_cost_cls', 1.0))
        self.map_cost_pts = float(kwargs.get('map_cost_pts', 1.0))
        # Point cost used in matching (assignment only; no grad):
        # - 'chamfer': symmetric chamfer distance in meters (MapTR eval-aligned)
        # - 'l1': mean L1 over fixed sampled points
        # - 'iou': buffered-polyline IoU on CPU (slow; optional)
        self.map_pts_cost_type = str(kwargs.get('map_pts_cost_type', 'chamfer'))
        self.map_iou_linewidth = float(kwargs.get('map_iou_linewidth', 1.0))

        # Losses
        # NOTE:
        # In some environments, mmcv.ops sigmoid focal loss expects integer
        # class targets (LongTensor) and will assert if passed one-hot floats.
        # For MapTR-style multi-label one-hot targets, use a pure PyTorch
        # implementation to avoid dtype/shape constraints.
        self.map_focal_gamma = float(kwargs.get('map_focal_gamma', 2.0))
        self.map_focal_alpha = float(kwargs.get('map_focal_alpha', 0.25))
        self.map_focal_weight = float(kwargs.get('map_focal_weight', 1.0))
        self.map_pts_weight = float(kwargs.get('map_pts_weight', 1.0))
        self.loss_map_pts = build_loss(dict(type='L1Loss', loss_weight=1.0))

        # Official MapTR-style assigner/targets/loss (decoupled from forward).
        # Note: expects dataset to pass `gt_map_vecs_pts_loc` as LiDARInstanceLines.
        self.maptr_loss_head: Optional[MapTRLossHead] = None
        if self.enable_map and (self.map_loss_impl == 'maptr_official'):
            if self.map_pts_normalize != 'sigmoid':
                raise ValueError(
                    'map_loss_impl=maptr_official expects map_pts_normalize="sigmoid" '
                    f'but got {self.map_pts_normalize!r}'
                )
            # Default to MapTR tiny config semantics; allow cfg override.
            if map_loss_pts_cfg is None:
                map_loss_pts_cfg = dict(type='PtsL1Loss', loss_weight=float(self.map_pts_weight))
            if map_loss_dir_cfg is None:
                map_loss_dir_cfg = dict(type='PtsDirCosLoss', loss_weight=0.005)
            if map_loss_cls_cfg is None:
                map_loss_cls_cfg = dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0)
            if map_loss_bbox_cfg is None:
                map_loss_bbox_cfg = dict(type='L1Loss', loss_weight=0.0)
            if map_loss_iou_cfg is None:
                map_loss_iou_cfg = dict(type='GIoULoss', loss_weight=0.0)
            if map_assigner_cfg is None:
                map_assigner_cfg = dict(
                    type='MapTRAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                    pts_cost=dict(type='OrderedPtsL1Cost', weight=float(self.map_pts_weight)),
                    pc_range=self.point_cloud_range,
                )
            self.maptr_loss_head = MapTRLossHead(
                num_classes=self.map_num_classes,
                pc_range=self.point_cloud_range,
                num_pts_per_vec=self.map_num_pts,
                num_pts_per_gt_vec=self.map_num_pts,
                dir_interval=map_dir_interval,
                gt_shift_pts_pattern=map_gt_shift_pts_pattern,
                sync_cls_avg_factor=map_sync_cls_avg_factor,
                bg_cls_weight=map_bg_cls_weight,
                loss_cls=map_loss_cls_cfg,
                loss_bbox=map_loss_bbox_cfg,
                loss_iou=map_loss_iou_cfg,
                loss_pts=map_loss_pts_cfg,
                loss_dir=map_loss_dir_cfg,
                assigner=map_assigner_cfg,
                sampler=map_sampler_cfg,
            )

        # Map query embeddings + heads.
        hidden = self.embed_dims

        # (A) Vector-level lightweight fallback (kept for robustness)
        self.map_query_embed = nn.Embedding(self.num_map_vec, self.embed_dims)
        self.map_cls_head = nn.Sequential(
            nn.Linear(self.embed_dims, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.map_num_classes),
        )
        self.map_pts_head = nn.Sequential(
            nn.Linear(self.embed_dims, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.map_num_pts * 2),
        )

        # (B) Official MapTRHead-style point queries + cls/reg branches.
        # Only used when map_use_point_queries=True and map_decoder is available.
        self.map_instance_embedding: Optional[nn.Embedding] = None
        self.map_pts_embedding: Optional[nn.Embedding] = None
        self.map_point_query_embedding: Optional[nn.Embedding] = None
        self.map_cls_branches: Optional[nn.ModuleList] = None
        self.map_reg_branches: Optional[nn.ModuleList] = None

        if self.map_use_point_queries and (self.map_decoder is not None):
            num_pred = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)

            def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
                return nn.ModuleList([copy.deepcopy(module) for _ in range(int(n))])

            # cls branch: per-vector, computed by pooling point-query features.
            cls_branch: List[nn.Module] = []
            num_reg_fcs = int(getattr(self, 'num_reg_fcs', 2))
            for _ in range(max(0, num_reg_fcs)):
                cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(nn.Linear(self.embed_dims, self.map_num_classes))
            fc_cls = nn.Sequential(*cls_branch)

            # reg branch: per-point query, outputs 2D offsets.
            reg_branch: List[nn.Module] = []
            for _ in range(max(0, num_reg_fcs)):
                reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(nn.Linear(self.embed_dims, 2))
            reg_branch = nn.Sequential(*reg_branch)

            if self.map_with_box_refine:
                self.map_cls_branches = _get_clones(fc_cls, num_pred)
                self.map_reg_branches = _get_clones(reg_branch, num_pred)
            else:
                self.map_cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
                self.map_reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

            # Point-query embeddings. Official MapTR uses instance+pts embeddings (2*embed_dims).
            if self.map_query_embed_type == 'instance_pts':
                self.map_instance_embedding = nn.Embedding(self.num_map_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts, self.embed_dims * 2)
            else:
                # all_pts: direct embedding for all point queries
                self.map_point_query_embedding = nn.Embedding(self.map_num_query, self.embed_dims * 2)

        # Keep attributes that BEVFormer detector sometimes expects.
        self.fp16_enabled = False
        # Optional point cloud / map range handled above.

        # Map points loss (training only).
        # Keep it simple and registry-free so smoke training can run without
        # requiring MapTR's full losses/assigners stack.
        self.map_pts_loss_type = str(kwargs.get('map_pts_loss_type', 'smooth_l1'))

        # Debug flags (safe defaults).
        self.debug_nan = bool(kwargs.get('debug_nan', False))
        self._nan_debug_printed = 0
        self.debug_map_refine = bool(kwargs.get('debug_map_refine', False))
        self.debug_map_gt = bool(kwargs.get('debug_map_gt', True))

    def _tensor_finite_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        """Return finite ratio + min/max for debugging (no grad, CPU friendly)."""
        with torch.no_grad():
            finite = torch.isfinite(x)
            ratio = float(finite.float().mean().detach().cpu())
            # Use masked min/max to avoid NaN affecting statistics.
            if finite.any():
                xf = x[finite]
                mn = float(xf.min().detach().cpu())
                mx = float(xf.max().detach().cpu())
            else:
                mn = float('nan')
                mx = float('nan')
        return {'finite_ratio': ratio, 'min': mn, 'max': mx, 'shape': tuple(x.shape), 'dtype': str(x.dtype), 'device': str(x.device)}

    def _maybe_log_nan(self, name: str, x: torch.Tensor) -> None:
        if not self.debug_nan:
            return
        if self._nan_debug_printed >= 20:
            return
        if not torch.isfinite(x).all():
            stats = self._tensor_finite_stats(x)
            print(f'[det_map][nan] {name}:', stats)
            self._nan_debug_printed += 1

    def _map_range_xy(self):
        """Return (x_min, x_max, y_min, y_max) for map vectors in meters."""
        pc = self.point_cloud_range
        if isinstance(pc, (list, tuple)) and len(pc) == 6:
            x_min, y_min, _z_min, x_max, y_max, _z_max = pc
            return float(x_min), float(x_max), float(y_min), float(y_max)
        # Fallback to BEVFormerHead's real_w/real_h centered at 0.
        x_max = float(getattr(self, 'real_w', 100.0)) / 2.0
        y_max = float(getattr(self, 'real_h', 100.0)) / 2.0
        return -x_max, x_max, -y_max, y_max

    def _decode_map_pts(self, pts_raw: torch.Tensor) -> torch.Tensor:
        """Decode raw point predictions to metric XY in the same convention as GT."""
        x_min, x_max, y_min, y_max = self._map_range_xy()

        if self.map_pts_normalize == 'tanh':
            pts = pts_raw.tanh()
            x = pts[..., 0] * (x_max - x_min) / 2.0
            y = pts[..., 1] * (y_max - y_min) / 2.0
        else:
            # default sigmoid
            pts01 = pts_raw.sigmoid()
            x = pts01[..., 0] * (x_max - x_min) + x_min
            y = pts01[..., 1] * (y_max - y_min) + y_min

        # Clip to patch to match GT clamping.
        x = torch.clamp(x, min=x_min, max=x_max)
        y = torch.clamp(y, min=y_min, max=y_max)
        return torch.stack((x, y), dim=-1)

    def _denormalize_map_pts01(self, pts01: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0,1] XY to metric XY."""
        x_min, x_max, y_min, y_max = self._map_range_xy()
        x = pts01[..., 0] * (x_max - x_min) + x_min
        y = pts01[..., 1] * (y_max - y_min) + y_min
        x = torch.clamp(x, min=x_min, max=x_max)
        y = torch.clamp(y, min=y_min, max=y_max)
        return torch.stack((x, y), dim=-1)

    def _polyline_iou_cost_matrix_cpu(
        self,
        pred_pts: torch.Tensor,
        gt_pts: torch.Tensor,
        linewidth: float,
    ) -> torch.Tensor:
        """Compute IoU-based cost matrix on CPU with Shapely + STRtree.

        Returns cost in [0, 1] where lower is better: cost = 1 - IoU.

        Note: intended for debugging / small-scale experiments.
        """
        if LineString is None or STRtree is None:
            raise RuntimeError('Shapely (with STRtree) is required for IoU map matching cost')

        pred_np = pred_pts.detach().cpu().numpy()
        gt_np = gt_pts.detach().cpu().numpy()
        num_pred = int(pred_np.shape[0])
        num_gt = int(gt_np.shape[0])

        # Default to cost=1 (IoU=0) for all pairs.
        cost = torch.ones((num_pred, num_gt), dtype=torch.float32)
        if num_pred == 0 or num_gt == 0:
            return cost.to(device=pred_pts.device, dtype=pred_pts.dtype)

        pred_polys = [
            LineString(line).buffer(
                linewidth,
                cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre,
            )
            for line in pred_np
        ]
        gt_polys = [
            LineString(line).buffer(
                linewidth,
                cap_style=CAP_STYLE.flat,
                join_style=JOIN_STYLE.mitre,
            )
            for line in gt_np
        ]

        pred_area = [float(p.area) for p in pred_polys]
        gt_area = [float(g.area) for g in gt_polys]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
            tree = STRtree(pred_polys)

        # Shapely 1.8: query returns geometries. Shapely 2.x: query may return indices.
        index_by_id = {id(g): i for i, g in enumerate(pred_polys)}

        for gi, gpoly in enumerate(gt_polys):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
                cands = tree.query(gpoly)

            # If candidates are indices (Shapely 2), convert to ints.
            if len(cands) > 0 and not hasattr(cands[0], 'intersects'):
                cand_inds = [int(x) for x in cands]
            else:
                cand_inds = [index_by_id[id(c)] for c in cands]

            for pi in cand_inds:
                ppoly = pred_polys[pi]
                if not ppoly.intersects(gpoly):
                    continue
                inter = float(ppoly.intersection(gpoly).area)
                union = pred_area[pi] + gt_area[gi] - inter
                iou = (inter / union) if union > 0 else 0.0
                cost[pi, gi] = float(1.0 - iou)

        return cost.to(device=pred_pts.device, dtype=pred_pts.dtype)

    def _sigmoid_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float,
        alpha: float,
        avg_factor: Optional[float] = None,
    ) -> torch.Tensor:
        """Sigmoid focal loss supporting one-hot float targets.

        Args:
            logits: [N, C]
            targets: [N, C] in {0,1}
        """
        targets = targets.to(dtype=logits.dtype)
        prob = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = ce * ((1.0 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if avg_factor is None:
            return loss.mean()
        denom = float(max(avg_factor, 1.0))
        return loss.sum() / denom

    def _match_map_vectors(
        self,
        pred_logits: torch.Tensor,
        pred_pts: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_pts: torch.Tensor,
    ) -> torch.Tensor:
        """Match predicted vectors to GT vectors (one-to-one).

        Returns:
            matched_pred_inds (LongTensor[k]), matched_gt_inds (LongTensor[k])
        """
        num_pred = int(pred_logits.size(0))
        num_gt = int(gt_labels.size(0))
        if num_gt == 0 or num_pred == 0:
            return pred_logits.new_zeros((0,), dtype=torch.long), pred_logits.new_zeros((0,), dtype=torch.long)

        # --- Classification cost: negative probability for the GT class ---
        with torch.no_grad():
            prob = pred_logits.sigmoid()  # [num_pred, C]
            cls_cost = -prob[:, gt_labels.long()]  # [num_pred, num_gt]

            # --- Point cost (assignment only): align with MapTR eval as much as possible ---
            num_pts = int(min(pred_pts.size(1), gt_pts.size(1)))
            if self.map_pts_cost_type == 'l1':
                # Direction-invariant mean L1 over points.
                pp = pred_pts[:, :num_pts, :].unsqueeze(1)  # [num_pred, 1, P, 2]
                gp_fwd = gt_pts[:, :num_pts, :].unsqueeze(0)    # [1, num_gt, P, 2]
                gp_rev = gt_pts[:, :num_pts, :].flip(dims=[1]).unsqueeze(0)  # [1, num_gt, P, 2]
                pts_cost_fwd = (pp - gp_fwd).abs().mean(dim=(2, 3))  # [num_pred, num_gt]
                pts_cost_rev = (pp - gp_rev).abs().mean(dim=(2, 3))
                pts_cost = torch.minimum(pts_cost_fwd, pts_cost_rev)
            elif self.map_pts_cost_type == 'iou':
                # MapTR-eval-like IoU between buffered polylines.
                # NOTE: CPU + Shapely; intended for debugging / small-scale runs.
                pts_cost = self._polyline_iou_cost_matrix_cpu(
                    pred_pts[:, :num_pts, :],
                    gt_pts[:, :num_pts, :],
                    linewidth=self.map_iou_linewidth,
                )
            else:
                # default: symmetric Chamfer distance in meters (direction-invariant by definition)
                # dist: [num_pred, num_gt, P, P]
                pp = pred_pts[:, :num_pts, :].unsqueeze(1)
                gp = gt_pts[:, :num_pts, :].unsqueeze(0)
                dist = torch.cdist(pp, gp, p=2)
                ab = dist.min(dim=-1).values.mean(dim=-1)  # [num_pred, num_gt]
                ba = dist.min(dim=-2).values.mean(dim=-1)  # [num_pred, num_gt]
                pts_cost = 0.5 * (ab + ba)

            cost = self.map_cost_cls * cls_cost + self.map_cost_pts * pts_cost

            cost_cpu = cost.detach().cpu()

            if (self.map_matcher_type == 'hungarian') and (linear_sum_assignment is not None):
                row_ind, col_ind = linear_sum_assignment(cost_cpu)
                matched_pred = pred_logits.new_tensor(row_ind, dtype=torch.long)
                matched_gt = pred_logits.new_tensor(col_ind, dtype=torch.long)
                return matched_pred, matched_gt

            # Greedy fallback (SciPy not available or matcher not set to hungarian)
            # Take lowest-cost pairs without conflicts.
            flat_cost = cost_cpu.reshape(-1)
            order = torch.argsort(flat_cost)
            matched_pred = []
            matched_gt = []
            used_pred = set()
            used_gt = set()
            for idx in order.tolist():
                p = int(idx // num_gt)
                g = int(idx % num_gt)
                if p in used_pred or g in used_gt:
                    continue
                used_pred.add(p)
                used_gt.add(g)
                matched_pred.append(p)
                matched_gt.append(g)
                if len(used_gt) >= num_gt or len(used_pred) >= num_pred:
                    break
            if len(matched_pred) == 0:
                return pred_logits.new_zeros((0,), dtype=torch.long), pred_logits.new_zeros((0,), dtype=torch.long)
            return pred_logits.new_tensor(matched_pred, dtype=torch.long), pred_logits.new_tensor(matched_gt, dtype=torch.long)

    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        img_metas: List[dict],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward.

        Args:
            mlvl_feats: list of multi-level image features. Each element has
                shape [bs, num_cam, C, H, W].
            img_metas: list of dict.
            prev_bev: cached BEV from previous frame.
            only_bev: if True, only return bev features.

        Returns:
            outs dict. Always includes `bev_embed`.
        """
        # Detection forward (encoder + det decoder) from BEVFormerHead.
        # If only_bev=True, BEVFormerHead returns a Tensor (bev_embed).
        if self.enable_det:
            outs = super().forward(mlvl_feats, img_metas, prev_bev=prev_bev, only_bev=only_bev)
            if only_bev:
                return outs
        else:
            # No det: only obtain BEV from encoder.
            outs = super().forward(mlvl_feats, img_metas, prev_bev=prev_bev, only_bev=only_bev)
            if only_bev:
                return outs

        assert isinstance(outs, dict)
        bev_embed = outs.get('bev_embed', None)
        if isinstance(bev_embed, torch.Tensor):
            self._maybe_log_nan('bev_embed', bev_embed)

        # Ensure map keys exist.
        outs.setdefault('map_cls_logits', None)
        outs.setdefault('map_pts', None)
        outs.setdefault('map_pts_norm', None)

        if self.enable_map:
            if not hasattr(self, '_run_cfg_logged'):
                self._run_cfg_logged = 0
            if self._run_cfg_logged < 1:
                rank, _ = get_dist_info()
                if rank == 0:
                    num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)
                    logging.getLogger('mmdet').info(
                        '[det_map][run_cfg] '
                        'map_loss_impl=%s map_use_point_queries=%s map_with_box_refine=%s '
                        'map_decoder=%s num_layers=%d map_query_embed_type=%s '
                        'num_map_vec=%d map_num_pts=%d map_num_query=%d '
                        'map_pts_normalize=%s map_pts_cost_type=%s',
                        self.map_loss_impl,
                        bool(self.map_use_point_queries),
                        bool(self.map_with_box_refine),
                        type(self.map_decoder).__name__ if self.map_decoder is not None else None,
                        num_dec,
                        self.map_query_embed_type,
                        int(self.num_map_vec),
                        int(self.map_num_pts),
                        int(self.map_num_query),
                        self.map_pts_normalize,
                        self.map_pts_cost_type,
                    )
                self._run_cfg_logged += 1

            # Simple per-vector query features derived from BEV global context.
            # NOTE: In this codebase, `bev_embed` can be either:
            # - [bs, bev_h*bev_w, C]
            # - [bev_h*bev_w, bs, C]  (common in transformer implementations)
            # We normalize to [bs, bev_tokens, C] for the map branch only.
            if isinstance(bev_embed, torch.Tensor) and bev_embed.dim() == 3:
                bs_from_metas = len(img_metas) if img_metas is not None else None
                if (bs_from_metas is not None) and (bev_embed.size(0) == bs_from_metas):
                    bev_embed_btc = bev_embed
                elif (bs_from_metas is not None) and (bev_embed.size(1) == bs_from_metas):
                    bev_embed_btc = bev_embed.permute(1, 0, 2).contiguous()
                else:
                    # Heuristic fallback: treat the smaller dim as batch.
                    if bev_embed.size(0) <= bev_embed.size(1):
                        bev_embed_btc = bev_embed
                    else:
                        bev_embed_btc = bev_embed.permute(1, 0, 2).contiguous()

                bs = bev_embed_btc.size(0)
                bev_global = bev_embed_btc.mean(dim=1)  # [bs, C]
            else:
                # Fallback: shouldn't happen for BEVFormer.
                bs = mlvl_feats[0].shape[0]
                bev_global = mlvl_feats[0].new_zeros((bs, self.embed_dims))

            # Prefer running a proper MapTR-style decoder if available.
            ran_decoder = False
            if (
                (self.map_decoder is not None)
                and self.map_use_point_queries
                and (self.map_cls_branches is not None)
                and (self.map_reg_branches is not None)
            ):
                try:
                    num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)

                    # Build point-query embeddings (num_query, 2C) following official MapTRHead.
                    if (self.map_query_embed_type == 'instance_pts') and (self.map_instance_embedding is not None) and (self.map_pts_embedding is not None):
                        pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)  # [1, num_pts, 2C]
                        inst_embeds = self.map_instance_embedding.weight.unsqueeze(1)  # [num_vec, 1, 2C]
                        object_query_embeds = (pts_embeds + inst_embeds).flatten(0, 1)  # [num_query, 2C]
                    elif self.map_point_query_embedding is not None:
                        object_query_embeds = self.map_point_query_embedding.weight
                    else:
                        raise RuntimeError('map point-query embeddings are not initialized')

                    if object_query_embeds.size(0) != self.map_num_query:
                        raise RuntimeError(
                            f'map_num_query mismatch: expected {self.map_num_query} but got {object_query_embeds.size(0)}'
                        )

                    # Split to query_pos/query content.
                    query_pos, query = torch.split(object_query_embeds.to(bev_embed_btc.dtype), self.embed_dims, dim=1)
                    query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [bs, num_query, C]
                    query = query.unsqueeze(0).expand(bs, -1, -1)  # [bs, num_query, C]

                    # Learned reference points in [0,1]: (bs, num_query, 2)
                    init_reference = self.map_reference_points(query_pos).sigmoid()

                    # Decoder inputs.
                    q = query.permute(1, 0, 2).contiguous()
                    qpos = query_pos.permute(1, 0, 2).contiguous()
                    bev_value = bev_embed_btc.permute(1, 0, 2).contiguous()  # (num_bev, bs, C)
                    spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=q.device, dtype=torch.long)
                    level_start_index = torch.tensor([0], device=q.device, dtype=torch.long)

                    dec_states, dec_references = self.map_decoder(
                        q,
                        key=None,
                        value=bev_value,
                        query_pos=qpos,
                        reference_points=init_reference,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        reg_branches=self.map_reg_branches if self.map_with_box_refine else None,
                        cls_branches=None,
                    )

                    # dec_states: [num_dec, num_query, bs, C]
                    # dec_references: [num_dec, bs, num_query, 2]
                    if (not isinstance(dec_states, torch.Tensor)) or (dec_states.dim() != 4):
                        raise RuntimeError(f'Unexpected map decoder states shape: {getattr(dec_states, "shape", None)}')
                    if (not isinstance(dec_references, torch.Tensor)) or (dec_references.dim() != 4):
                        raise RuntimeError(f'Unexpected map decoder references shape: {getattr(dec_references, "shape", None)}')

                    if self.debug_map_refine and self.map_with_box_refine:
                        if not hasattr(self, '_dbg_refine_printed'):
                            self._dbg_refine_printed = 0
                        if self._dbg_refine_printed < 1:
                            with torch.no_grad():
                                d0 = (dec_references[0] - init_reference).abs().mean().item()
                                deltas = [d0]
                                for i in range(1, int(dec_references.size(0))):
                                    deltas.append((dec_references[i] - dec_references[i - 1]).abs().mean().item())
                            msg = ' '.join([f'd{i}={v:.6f}' for i, v in enumerate(deltas)])
                            rank, _ = get_dist_info()
                            if rank == 0:
                                logging.getLogger('mmdet').info(
                                    '[det_map][map_refine] absmean(ref_delta): %s', msg)
                            self._dbg_refine_printed += 1

                    all_cls: List[torch.Tensor] = []
                    all_pts01: List[torch.Tensor] = []
                    all_bbox01: List[torch.Tensor] = []

                    for lvl in range(min(int(dec_states.size(0)), num_dec)):
                        hs_l = dec_states[lvl].permute(1, 0, 2).contiguous()  # [bs, num_query, C]

                        if lvl == 0:
                            reference = init_reference
                        else:
                            reference = dec_references[lvl - 1]

                        # Point regression (normalized coords in [0,1])
                        tmp = self.map_reg_branches[lvl](hs_l)
                        tmp[..., 0:2] = tmp[..., 0:2] + inverse_sigmoid(reference)
                        pts01 = tmp[..., 0:2].sigmoid()

                        pts01 = pts01.view(bs, self.num_map_vec, self.map_num_pts, 2)
                        bbox01 = pts_to_bbox_normalized(pts01)

                        # Vector classification (pool over points like official MapTRHead)
                        hs_vec = hs_l.view(bs, self.num_map_vec, self.map_num_pts, -1).mean(dim=2)
                        cls_logits = self.map_cls_branches[lvl](hs_vec)

                        all_cls.append(cls_logits)
                        all_pts01.append(pts01)
                        all_bbox01.append(bbox01)

                    all_cls_t = torch.stack(all_cls, dim=0)
                    all_pts01_t = torch.stack(all_pts01, dim=0)
                    all_bbox01_t = torch.stack(all_bbox01, dim=0)

                    outs['map_cls_logits'] = all_cls_t[-1]
                    outs['map_pts_norm'] = all_pts01_t[-1]
                    outs['map_pts'] = self._denormalize_map_pts01(all_pts01_t[-1])
                    outs['map_preds_dicts'] = {
                        'all_cls_scores': all_cls_t,
                        'all_bbox_preds': all_bbox01_t,
                        'all_pts_preds': all_pts01_t,
                        'enc_cls_scores': None,
                        'enc_bbox_preds': None,
                        'enc_pts_preds': None,
                    }
                    ran_decoder = True
                except Exception as e:
                    if not hasattr(self, '_map_decoder_fail_printed'):
                        self._map_decoder_fail_printed = 0
                    if self._map_decoder_fail_printed < 3:
                        print('[det_map][map_decoder][point_queries] failed, falling back:', repr(e))
                        self._map_decoder_fail_printed += 1
                    ran_decoder = False

            if (not ran_decoder) and (self.map_decoder is not None):
                try:
                    # Prepare query/query_pos: (num_query, bs, C)
                    # Use DETR-style formulation: content query starts from 0,
                    # while positional query carries the learned embedding.
                    qpos = self.map_query_embed.weight.unsqueeze(1).expand(self.num_map_vec, bs, -1)
                    q = torch.zeros_like(qpos)

                    # BEV memory as value: (num_bev, bs, C)
                    # Note: CustomMSDeformableAttention assumes (num_key, bs, C)
                    # when batch_first=False (default).
                    bev_value = bev_embed_btc.permute(1, 0, 2).contiguous()
                    spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=q.device, dtype=torch.long)
                    level_start_index = torch.tensor([0], device=q.device, dtype=torch.long)

                    # Learned reference points in [0,1]: (bs, num_query, 2)
                    refp = self.map_reference_points(qpos.permute(1, 0, 2)).sigmoid()

                    dec_out = self.map_decoder(
                        q,
                        key=None,
                        value=bev_value,
                        query_pos=qpos,
                        reference_points=refp,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                    )
                    # dec_out may be (layers, num_query, bs, C) or (out, ref)
                    if isinstance(dec_out, tuple) and len(dec_out) >= 1:
                        dec_feats = dec_out[0]
                    else:
                        dec_feats = dec_out

                    if isinstance(dec_feats, torch.Tensor) and dec_feats.dim() == 4:
                        # [num_dec, num_query, bs, C] -> [num_dec, bs, num_query, C]
                        dec_layers = dec_feats.permute(0, 2, 1, 3).contiguous()
                    elif isinstance(dec_feats, torch.Tensor) and dec_feats.dim() == 3:
                        # [num_query, bs, C] -> [1, bs, num_query, C]
                        dec_layers = dec_feats.permute(1, 0, 2).unsqueeze(0).contiguous()
                    else:
                        raise RuntimeError(f'Unexpected map decoder output type/shape: {type(dec_feats)} {getattr(dec_feats, "shape", None)}')

                    # Strict MapTR-style multi-layer outputs.
                    # cls: [num_dec, bs, num_vec, num_cls]
                    all_cls = self.map_cls_head(dec_layers)
                    # pts_raw: [num_dec, bs, num_vec, num_pts, 2]
                    pts_raw = self.map_pts_head(dec_layers).view(
                        dec_layers.size(0),
                        bs,
                        self.num_map_vec,
                        self.map_num_pts,
                        2,
                    )
                    all_pts01 = pts_raw.sigmoid() if self.map_pts_normalize != 'tanh' else (pts_raw.tanh() + 1.0) * 0.5

                    # Keep last-layer convenience outputs for inference utilities.
                    outs['map_cls_logits'] = all_cls[-1]
                    outs['map_pts_norm'] = all_pts01[-1]
                    outs['map_pts'] = self._denormalize_map_pts01(all_pts01[-1])

                    # Build MapTR-style preds_dicts for loss.
                    all_pts01_flat = all_pts01.reshape(-1, self.num_map_vec, self.map_num_pts, 2)
                    all_bbox01_flat = pts_to_bbox_normalized(all_pts01_flat)
                    all_bbox01 = all_bbox01_flat.view(dec_layers.size(0), bs, self.num_map_vec, 4)
                    outs['map_preds_dicts'] = {
                        'all_cls_scores': all_cls,
                        'all_bbox_preds': all_bbox01,
                        'all_pts_preds': all_pts01,
                        'enc_cls_scores': None,
                        'enc_bbox_preds': None,
                        'enc_pts_preds': None,
                    }
                    ran_decoder = True
                except Exception as e:
                    # Decoder failed: fall back, but don't hide the root cause.
                    if not hasattr(self, '_map_decoder_fail_printed'):
                        self._map_decoder_fail_printed = 0
                    if self._map_decoder_fail_printed < 3:
                        print('[det_map][map_decoder] failed, falling back:', repr(e))
                        self._map_decoder_fail_printed += 1
                    ran_decoder = False

            if not ran_decoder:
                # Lightweight fallback: per-vector features from BEV global context.
                q = self.map_query_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # [bs, num_vec, C]
                q = q + bev_global.unsqueeze(1)
                cls_last = self.map_cls_head(q)  # [bs, num_vec, num_cls]
                pts_raw = self.map_pts_head(q).view(bs, self.num_map_vec, self.map_num_pts, 2)
                # MapTR-official loss expects normalized pts in [0,1].
                pts01_last = pts_raw.sigmoid() if self.map_pts_normalize != 'tanh' else (pts_raw.tanh() + 1.0) * 0.5

                outs['map_cls_logits'] = cls_last
                outs['map_pts_norm'] = pts01_last
                outs['map_pts'] = self._denormalize_map_pts01(pts01_last)

                # Strict MapTR-style multi-layer preds_dicts for loss.
                # If map decoder is unavailable, repeat the single-layer preds
                # so MapTRLossHead still sees [num_dec, bs, ...].
                if self.enable_map and (self.map_loss_impl == 'maptr_official'):
                    num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)
                    all_cls = cls_last.unsqueeze(0).repeat(num_dec, 1, 1, 1)
                    all_pts01 = pts01_last.unsqueeze(0).repeat(num_dec, 1, 1, 1, 1)
                    all_pts01_flat = all_pts01.reshape(-1, self.num_map_vec, self.map_num_pts, 2)
                    all_bbox01_flat = pts_to_bbox_normalized(all_pts01_flat)
                    all_bbox01 = all_bbox01_flat.view(num_dec, bs, self.num_map_vec, 4)
                    outs['map_preds_dicts'] = {
                        'all_cls_scores': all_cls,
                        'all_bbox_preds': all_bbox01,
                        'all_pts_preds': all_pts01,
                        'enc_cls_scores': None,
                        'enc_bbox_preds': None,
                        'enc_pts_preds': None,
                    }

            # Build MapTR-style preds_dicts for loss (kept separate from det outs).
            # (Decoder path already fills outs['map_preds_dicts'].)
            if self.enable_map and (self.map_loss_impl == 'maptr_official') and ('map_preds_dicts' not in outs):
                if isinstance(outs.get('map_cls_logits', None), torch.Tensor) and isinstance(outs.get('map_pts_norm', None), torch.Tensor):
                    cls_last = outs['map_cls_logits']
                    pts01_last = outs['map_pts_norm']
                    num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)
                    all_cls = cls_last.unsqueeze(0).repeat(num_dec, 1, 1, 1)
                    all_pts01 = pts01_last.unsqueeze(0).repeat(num_dec, 1, 1, 1, 1)
                    all_pts01_flat = all_pts01.reshape(-1, self.num_map_vec, self.map_num_pts, 2)
                    all_bbox01_flat = pts_to_bbox_normalized(all_pts01_flat)
                    all_bbox01 = all_bbox01_flat.view(num_dec, bs, self.num_map_vec, 4)
                    outs['map_preds_dicts'] = {
                        'all_cls_scores': all_cls,
                        'all_bbox_preds': all_bbox01,
                        'all_pts_preds': all_pts01,
                        'enc_cls_scores': None,
                        'enc_bbox_preds': None,
                        'enc_pts_preds': None,
                    }

            if isinstance(outs['map_cls_logits'], torch.Tensor):
                self._maybe_log_nan('map_cls_logits', outs['map_cls_logits'])
            if isinstance(outs['map_pts'], torch.Tensor):
                self._maybe_log_nan('map_pts', outs['map_pts'])

        return outs

    # --- MapTR-aligned inference hook ---
    def get_map_results(self, outs: Dict[str, Any], img_metas: List[dict], **kwargs) -> List[Dict[str, Any]]:
        """Convert network outputs to map results format.

        This is intentionally a stub: returns empty predictions for each sample.
        Later we will replace this with MapTR's vector decoding and class
        formatting (e.g., divider / boundary / ped_crossing etc.).
        """
        bs = len(img_metas)
        if not isinstance(outs, dict):
            return [dict(vectors=[], scores=[], labels=[]) for _ in range(bs)]

        cls_logits = outs.get('map_cls_logits', None)
        pts = outs.get('map_pts', None)
        if (not isinstance(cls_logits, torch.Tensor)) or (not isinstance(pts, torch.Tensor)):
            return [dict(vectors=[], scores=[], labels=[]) for _ in range(bs)]

        # cls_logits: [bs, num_vec, num_cls]
        # pts:       [bs, num_vec, num_pts, 2]
        results: List[Dict[str, Any]] = []
        with torch.no_grad():
            for i in range(bs):
                li = cls_logits[i]
                pi = pts[i]

                # MapTR-style heads often use sigmoid + focal loss.
                prob = li.sigmoid()
                scores, labels = prob.max(dim=-1)

                results.append(
                    dict(
                        vectors=pi.detach().cpu().numpy(),
                        scores=scores.detach().cpu().numpy(),
                        labels=labels.detach().cpu().numpy(),
                        cls_logits=li.detach().cpu().numpy(),
                    ))
        return results

    # --- Loss stubs (to be implemented once map_gts are wired) ---
    def loss(
        self,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        pts_feats=None,
        occ_gts=None,
        flow_gts=None,
        outs: Optional[Dict[str, Any]] = None,
        img_metas: Optional[List[dict]] = None,
        gt_map_vecs_label=None,
        gt_map_vecs_pts_loc=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Loss for det+map (placeholder).

        Contract (current milestone):
        - Accept MapTR-style GT keys from dataset/pipeline:
          `gt_map_vecs_label`, `gt_map_vecs_pts_loc`.
        - Return at least one stable map-related loss key so training loops
          expecting non-empty loss dicts won't choke.

        We intentionally do *not* implement MapTR decoding/loss yet.
        """

        outs = outs or {}
        if not isinstance(outs, dict):
            outs = {}

        # Anchor (keeps training stable if any branch missing).
        bev = outs.get('bev_embed', None)
        if isinstance(bev, torch.Tensor):
            loss_anchor = bev.new_zeros(())
        else:
            loss_anchor = torch.zeros((), device='cpu')

        losses: Dict[str, torch.Tensor] = {}

        # ---- Detection losses ----
        if self.enable_det and (gt_bboxes_3d is not None) and (gt_labels_3d is not None):
            try:
                det_losses = super().loss(
                    gt_bboxes_list=gt_bboxes_3d,
                    gt_labels_list=gt_labels_3d,
                    preds_dicts=outs,
                    gt_bboxes_ignore=None,
                    img_metas=img_metas,
                )
                losses.update(det_losses)
            except Exception as e:
                # Keep training alive for smoke; surface the failure.
                print('[det_map][det] loss failed:', repr(e))
                if getattr(self, 'debug_nan', False):
                    import traceback

                    traceback.print_exc()
                losses['loss_det_failed'] = loss_anchor

        # ---- Map GT sanity checks / lightweight stats ----
        # We keep these checks user-friendly and non-fatal by default.
        # If you want hard asserts later, we can add a config flag.
        missing_map_gt = (gt_map_vecs_label is None) or (gt_map_vecs_pts_loc is None)

        # Normalize GT container types.
        # Depending on the dataset/pipeline, these may come as:
        # - list[Tensor] / list[LiDARInstanceLines] (batch)
        # - Tensor / LiDARInstanceLines (single sample)
        # We normalize to per-batch lists so the code below can treat them
        # uniformly.
        if not missing_map_gt:
            if not isinstance(gt_map_vecs_label, (list, tuple)):
                gt_map_vecs_label = [gt_map_vecs_label]
            if not isinstance(gt_map_vecs_pts_loc, (list, tuple)):
                gt_map_vecs_pts_loc = [gt_map_vecs_pts_loc]

        if self.enable_map and (not missing_map_gt) and getattr(self, 'debug_map_gt', True):
            # Expected: batch lists, one entry per sample.
            # - gt_map_vecs_label: list[Tensor[num_lines]]
            # - gt_map_vecs_pts_loc: list[Tensor[num_lines, num_pts, 2]] (or LiDARInstanceLines)
            try:
                bs = len(img_metas) if img_metas is not None else None
                n_label = len(gt_map_vecs_label) if hasattr(gt_map_vecs_label, '__len__') else None
                n_pts = len(gt_map_vecs_pts_loc) if hasattr(gt_map_vecs_pts_loc, '__len__') else None

                # Only print occasionally to avoid spamming.
                # (The runner will call loss() every iteration.)
                if not hasattr(self, '_map_gt_debug_printed'):
                    self._map_gt_debug_printed = 0

                if self._map_gt_debug_printed < 1:
                    print('[det_map][gt] bs(img_metas)=', bs, 'len(labels)=', n_label, 'len(pts)=', n_pts)
                    for i in range(min(1, n_label or 0)):
                        lab_i = gt_map_vecs_label[i]
                        pts_i = gt_map_vecs_pts_loc[i]
                        # LiDARInstanceLines case
                        if hasattr(pts_i, 'fixed_num_sampled_points'):
                            pts_tensor = pts_i.fixed_num_sampled_points
                        else:
                            pts_tensor = pts_i

                        if hasattr(lab_i, 'detach'):
                            uniq = sorted(set(lab_i.detach().cpu().tolist()))
                            num_lines = int(lab_i.numel())
                        else:
                            uniq = None
                            num_lines = None

                        pts_shape = tuple(getattr(pts_tensor, 'shape', ()))
                        print(f'[det_map][gt] sample{i}: num_lines={num_lines}, labels_uniq={uniq}, pts_shape={pts_shape}')
                    self._map_gt_debug_printed += 1
            except Exception as e:
                # Non-fatal: keep training going.
                if not hasattr(self, '_map_gt_debug_fail_printed'):
                    self._map_gt_debug_fail_printed = 0
                if self._map_gt_debug_fail_printed < 3:
                    print('[det_map][gt] sanity check failed:', repr(e))
                    self._map_gt_debug_fail_printed += 1

        # ---- MapTR-official map loss (assigner/targets/loss) ----
        if self.enable_map and (not missing_map_gt) and (self.map_loss_impl == 'maptr_official'):
            map_preds = outs.get('map_preds_dicts', None)
            if (self.maptr_loss_head is not None) and isinstance(map_preds, dict):
                try:
                    map_loss_dict = self.maptr_loss_head.loss(
                        gt_vecs_list=gt_map_vecs_pts_loc,
                        gt_labels_list=gt_map_vecs_label,
                        preds_dicts=map_preds,
                        gt_bboxes_ignore=None,
                    )
                    # Prefix to avoid clashing with det keys.
                    losses['loss_map_cls'] = map_loss_dict.get('loss_cls', loss_anchor)
                    losses['loss_map_bbox'] = map_loss_dict.get('loss_bbox', loss_anchor)
                    losses['loss_map_iou'] = map_loss_dict.get('loss_iou', loss_anchor)
                    losses['loss_map_pts'] = map_loss_dict.get('loss_pts', loss_anchor)
                    losses['loss_map_dir'] = map_loss_dict.get('loss_dir', loss_anchor)

                    # Aggregated map loss (bbox/iou are typically 0-weight in MapTR configs).
                    losses['loss_map'] = (
                        losses['loss_map_cls']
                        + losses['loss_map_bbox']
                        + losses['loss_map_iou']
                        + losses['loss_map_pts']
                        + losses['loss_map_dir']
                    )

                    # Keep intermediate decoder layer losses for parity with MapTR logs.
                    for k, v in map_loss_dict.items():
                        if k.startswith('d'):
                            losses[f'loss_map_{k}'] = v
                except Exception as e:
                    print('[det_map][maptr_official] map loss failed:', repr(e))
                    if getattr(self, 'debug_nan', False):
                        import traceback
                        traceback.print_exc()
                    losses['loss_map'] = loss_anchor
                    losses['loss_map_cls'] = loss_anchor
                    losses['loss_map_pts'] = loss_anchor
                    losses['loss_map_dir'] = loss_anchor
            else:
                losses['loss_map'] = loss_anchor
                losses['loss_map_cls'] = loss_anchor
                losses['loss_map_pts'] = loss_anchor
                losses['loss_map_dir'] = loss_anchor
        elif self.enable_map:
            # Fallback to legacy keys (kept to avoid breaking old experiments).
            losses.setdefault('loss_map', loss_anchor)
            losses.setdefault('loss_map_cls', loss_anchor)
            losses.setdefault('loss_map_pts', loss_anchor)

        if self.enable_map and missing_map_gt:
            # Keep training loop stable but make it visible in logs.
            losses['loss_map_missing_gt'] = loss_anchor
        return losses
