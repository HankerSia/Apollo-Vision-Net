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
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.utils import build_from_cfg
from mmdet.models import HEADS, build_loss

from .bevformer_head import BEVFormerHead


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
        self.map_decoder = None
        if map_decoder_cfg is not None:
            try:
                self.map_decoder = build_from_cfg(map_decoder_cfg, TRANSFORMER_LAYER_SEQUENCE)
            except Exception:
                self.map_decoder = None

        # --- Map branch (MapTR-aligned, minimal) ---
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
        # Optional point cloud / map range handled above.

        # Map points loss (training only).
        # Keep it simple and registry-free so smoke training can run without
        # requiring MapTR's full losses/assigners stack.
        self.map_pts_loss_type = str(kwargs.get('map_pts_loss_type', 'smooth_l1'))

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

        if self.enable_map:
            # Simple per-vector query features derived from BEV global context.
            # bev_embed: [bs, bev_h*bev_w, C] in this codebase.
            if isinstance(bev_embed, torch.Tensor) and bev_embed.dim() == 3:
                bs = bev_embed.size(0)
                bev_global = bev_embed.mean(dim=1)  # [bs, C]
            else:
                # Fallback: shouldn't happen for BEVFormer.
                bs = mlvl_feats[0].shape[0]
                bev_global = mlvl_feats[0].new_zeros((bs, self.embed_dims))

            # Prefer running a proper MapTR-style decoder if available.
            ran_decoder = False
            if self.map_decoder is not None:
                try:
                    # Prepare query: (num_query, bs, C)
                    q = self.map_query_embed.weight.unsqueeze(1).expand(self.num_map_vec, bs, -1)
                    # Provide default reference points (normalized center) so
                    # decoders expecting reference_points won't error.
                    refp = q.new_full((bs, self.num_map_vec, 2), 0.5)
                    dec_out = self.map_decoder(q, reference_points=refp)
                    # dec_out may be (layers, num_query, bs, C) or (out, ref)
                    if isinstance(dec_out, tuple) and len(dec_out) >= 1:
                        dec_feats = dec_out[0]
                    else:
                        dec_feats = dec_out

                    if isinstance(dec_feats, torch.Tensor) and dec_feats.dim() == 4:
                        # take last layer: [layers, num_query, bs, C]
                        last = dec_feats[-1]
                    else:
                        # already [num_query, bs, C]
                        last = dec_feats

                    # to [bs, num_query, C]
                    last = last.permute(1, 0, 2)
                    outs['map_cls_logits'] = self.map_cls_head(last)
                    pts = self.map_pts_head(last).view(bs, self.num_map_vec, self.map_num_pts, 2)
                    outs['map_pts'] = pts
                    ran_decoder = True
                except Exception:
                    # Decoder not fully wired; fall back to simple MLP below.
                    ran_decoder = False

            if not ran_decoder:
                # Lightweight fallback: per-vector features from BEV global context.
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
                if (not torch.isfinite(map_cls).all()) or (not torch.isfinite(map_pts).all()):
                    loss_cls = loss_anchor
                    loss_pts = loss_anchor
                else:
                    # Convert GT pts into tensors and normalize container types.
                    gt_pts_list = []
                    for pts_i in gt_map_vecs_pts_loc:
                        if hasattr(pts_i, 'fixed_num_sampled_points'):
                            gt_pts_list.append(pts_i.fixed_num_sampled_points)
                        else:
                            gt_pts_list.append(pts_i)

                    gt_lab_list = gt_map_vecs_label

                    loss_cls = map_cls.new_zeros(())
                    loss_pts = map_pts.new_zeros(())

                    bs = map_cls.shape[0]
                    for b in range(bs):
                        if b >= len(gt_lab_list) or b >= len(gt_pts_list):
                            continue

                        gt_lab = gt_lab_list[b]
                        gt_pts = gt_pts_list[b]
                        if not isinstance(gt_lab, torch.Tensor) or not isinstance(gt_pts, torch.Tensor):
                            continue
                        if gt_pts.numel() == 0 or gt_lab.numel() == 0:
                            continue

                        # Ensure shapes: [num_lines], [num_lines, num_pts, 2]
                        if gt_pts.dim() != 3 or gt_pts.size(-1) != 2:
                            continue

                        # Predictions for this sample
                        pred_logits = map_cls[b]  # [num_q, C]
                        pred_pts = map_pts[b]     # [num_q, P, 2]
                        device = pred_logits.device

                        # Simple alignment: match the first K predictions with
                        # the first K GT instances. This is only for smoke
                        # training to verify gradients/end-to-end plumbing.
                        k = int(min(pred_logits.size(0), gt_lab.size(0), pred_pts.size(0), gt_pts.size(0)))
                        if k <= 0:
                            continue

                        # cls loss
                        tgt_labels = gt_lab[:k].long().to(device)
                        loss_cls = loss_cls + F.cross_entropy(pred_logits[:k], tgt_labels)

                        # pts loss
                        num_pts = int(min(gt_pts.shape[1], pred_pts.shape[1]))
                        if num_pts <= 0:
                            continue
                        pred_pts_k = pred_pts[:k, :num_pts, :]
                        tgt_pts_k = gt_pts[:k, :num_pts, :].to(device)
                        if self.map_pts_loss_type == 'l1':
                            loss_pts = loss_pts + F.l1_loss(pred_pts_k, tgt_pts_k, reduction='mean')
                        else:
                            loss_pts = loss_pts + F.smooth_l1_loss(pred_pts_k, tgt_pts_k, reduction='mean')

                    # normalize by batch size
                    loss_cls = loss_cls / max(1, bs)
                    loss_pts = loss_pts / max(1, bs)
            else:
                loss_cls = loss_anchor
                loss_pts = loss_anchor
        else:
            loss_cls = loss_anchor
            loss_pts = loss_anchor

        losses.update({
            'loss_map_placeholder': loss_anchor,
            'loss_map_cls': loss_cls,
            'loss_map_pts': loss_pts,
        })
        if self.enable_map and missing_map_gt:
            # Keep training loop stable but make it visible in logs.
            losses['loss_map_missing_gt'] = loss_anchor
        return losses
