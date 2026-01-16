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

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmdet.models import HEADS


@HEADS.register_module()
class BEVFormerDetMapHeadApollo(nn.Module):
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
        transformer: Dict[str, Any],
        bev_h: int,
        bev_w: int,
        real_h: float,
        real_w: float,
        embed_dims: int = 256,
        positional_encoding: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # We deliberately import build helpers lazily to match existing codebase
        # style and avoid extra deps.
        from mmcv.cnn.bricks.transformer import build_positional_encoding
        from mmdet.models.utils import build_transformer

        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.real_h = float(real_h)
        self.real_w = float(real_w)
        self.embed_dims = int(embed_dims)

        # BEV queries & pos encoding mimic BEVFormer occ/det heads.
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        if positional_encoding is None:
            # A sane default consistent with many BEVFormer configs.
            positional_encoding = dict(type='LearnedPositionalEncoding', num_feats=self.embed_dims // 2, row_num_embed=self.bev_h, col_num_embed=self.bev_w)
        else:
            positional_encoding = dict(positional_encoding)
            positional_encoding.setdefault('row_num_embed', self.bev_h)
            positional_encoding.setdefault('col_num_embed', self.bev_w)

        self.positional_encoding = build_positional_encoding(positional_encoding)

        # Transformer is required for BEV extraction.
        # In this codebase, `PerceptionTransformer` is registered under
        # mmdet's TRANSFORMER registry (not mmcv's layer-sequence registry).
        self.transformer = build_transformer(transformer)

        # --- Det branch (stub) ---
        self.enable_det = kwargs.get('enable_det', True)

        # --- Map branch (stub, MapTR-aligned) ---
        self.enable_map = kwargs.get('enable_map', True)

        # Minimal map prediction heads ("先简后全")
        # We predict a fixed number of vector instances per sample.
        self.num_map_vec = int(kwargs.get('num_map_vec', 50))
        # Prefer config-provided fixed points number; otherwise default to 20.
        self.map_num_pts = int(kwargs.get('map_num_pts', kwargs.get('fixed_ptsnum_per_line', 20)))
        self.map_num_classes = int(kwargs.get('map_num_classes', 3))

        # A tiny MLP head from a per-query embedding to cls/pts.
        hidden = self.embed_dims
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

        # Keep attributes that BEVFormer detector sometimes expects.
        self.fp16_enabled = False

        # Debug flags (safe defaults).
        self.debug_nan = bool(kwargs.get('debug_nan', False))
        self._nan_debug_printed = 0

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
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        if isinstance(bev_embed, torch.Tensor):
            self._maybe_log_nan('bev_embed', bev_embed)

            # NOTE: We expect upstream transformer to keep bev_embed finite.
            # If this triggers, treat it as a bug and fix in transformer.

        if only_bev:
            return bev_embed

        # Keep a BEVFormer-compat output dict.
        outs: Dict[str, Any] = {
            'bev_embed': bev_embed,
            # Det keys (kept for compatibility, but set to None for now).
            'all_cls_scores': None,
            'all_bbox_preds': None,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            # Map keys (new).
            # Minimal map preds (will be replaced by MapTR decoder later).
            'map_cls_logits': None,
            'map_pts': None,
        }

        if self.enable_map:
            # Simple per-vector query features derived from BEV global context.
            # bev_embed: [bs, bev_h*bev_w, C] in this codebase.
            if isinstance(bev_embed, torch.Tensor) and bev_embed.dim() == 3:
                bev_global = bev_embed.mean(dim=1)  # [bs, C]
            else:
                # Fallback: shouldn't happen for BEVFormer.
                bev_global = mlvl_feats[0].new_zeros((bs, self.embed_dims))

            q = self.map_query_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # [bs, num_vec, C]
            q = q + bev_global.unsqueeze(1)
            outs['map_cls_logits'] = self.map_cls_head(q)  # [bs, num_vec, 3]
            pts = self.map_pts_head(q).view(bs, self.num_map_vec, self.map_num_pts, 2)
            outs['map_pts'] = pts

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
        return [dict(vectors=[], scores=[], labels=[]) for _ in range(bs)]

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

        # Make a tiny differentiable tensor tied to the graph so DDP doesn't
        # complain about unused parameters when map is enabled.
        outs = outs or {}
        bev = outs.get('bev_embed', None)
        if isinstance(bev, torch.Tensor):
            # Avoid NaN propagation: 0.0 * NaN is still NaN.
            loss_anchor = bev.new_zeros(())
        else:
            # Fallback for edge cases (shouldn't happen in normal training).
            loss_anchor = torch.zeros((), device='cpu')

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

        if self.enable_map and (not missing_map_gt):
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

                if self._map_gt_debug_printed < 3:
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
                print('[det_map][gt] sanity check failed:', repr(e))

        # ---- Minimal map loss ("先简后全") ----
        # Align first-K predicted vectors with first-K GT vectors.
        # This is *not* MapTR matching; it's just to validate end-to-end gradients.
        if self.enable_map and (not missing_map_gt):
            map_cls = outs.get('map_cls_logits', None)
            map_pts = outs.get('map_pts', None)
            if isinstance(map_cls, torch.Tensor) and isinstance(map_pts, torch.Tensor):
                # If upstream BEV has NaNs (common in early CPU smoke tests),
                # avoid propagating NaNs into cls/pts losses.
                if (not torch.isfinite(map_cls).all()) or (not torch.isfinite(map_pts).all()):
                    loss_cls = loss_anchor
                    loss_pts = loss_anchor
                else:
                    # Convert GT pts into tensors.
                    gt_pts_list = []
                    for pts_i in gt_map_vecs_pts_loc:
                        if hasattr(pts_i, 'fixed_num_sampled_points'):
                            gt_pts_list.append(pts_i.fixed_num_sampled_points)
                        else:
                            gt_pts_list.append(pts_i)

                    # Defensive: sometimes upstream provides batch size meta=1, but
                    # `gt_map_vecs_*` may still be wrapped inconsistently.
                    # We'll safely index by `b` only if available.
                    gt_lab_list = gt_map_vecs_label

                    loss_cls = map_cls.sum() * 0.0
                    loss_pts = map_pts.sum() * 0.0

                    for b in range(map_cls.shape[0]):
                        if b >= len(gt_lab_list) or b >= len(gt_pts_list):
                            continue

                        gt_lab = gt_lab_list[b]
                        gt_pts = gt_pts_list[b]

                        if not isinstance(gt_lab, torch.Tensor) or not isinstance(gt_pts, torch.Tensor):
                            continue

                        # Ensure shapes: [num_lines], [num_lines, num_pts, 2]
                        if gt_pts.dim() != 3 or gt_pts.size(-1) != 2:
                            continue

                        # Optionally clamp GT points count to prediction points count.
                        num_pts = min(gt_pts.shape[1], map_pts.shape[2])
                        gt_pts = gt_pts[:, :num_pts, :]

                        K = min(gt_lab.numel(), map_cls.shape[1])
                        if K <= 0:
                            continue

                        pred_logits = map_cls[b, :K, :]  # [K, C]
                        tgt_labels = gt_lab[:K].long().to(pred_logits.device)
                        loss_cls = loss_cls + F.cross_entropy(pred_logits, tgt_labels)

                        pred_pts = map_pts[b, :K, :num_pts, :]  # [K, P, 2]
                        tgt_pts = gt_pts[:K].to(pred_pts.device)
                        loss_pts = loss_pts + F.l1_loss(pred_pts, tgt_pts)

                # Average over batch for stability.
                loss_cls = loss_cls / max(1, map_cls.shape[0])
                loss_pts = loss_pts / max(1, map_pts.shape[0])
            else:
                # If forward didn't produce map preds, keep it stable.
                loss_cls = loss_anchor
                loss_pts = loss_anchor
        else:
            loss_cls = loss_anchor
            loss_pts = loss_anchor

        losses = {
            'loss_map_placeholder': loss_anchor,
            'loss_map_cls': loss_cls,
            'loss_map_pts': loss_pts,
        }
        if self.enable_map and missing_map_gt:
            # Keep training loop stable but make it visible in logs.
            losses['loss_map_missing_gt'] = loss_anchor
        return losses
