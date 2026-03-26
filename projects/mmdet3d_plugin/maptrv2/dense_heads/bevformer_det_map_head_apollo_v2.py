from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16, get_dist_info
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from ...bevformer.dense_heads.bevformer_head import BEVFormerHead
from ...bevformer.dense_heads.bevformer_det_map_head_apollo import BEVFormerDetMapHeadApollo
from ...maptr.dense_heads.maptr_loss_head import pts_to_bbox_normalized


@HEADS.register_module()
class BEVFormerDetMapHeadApolloV2(BEVFormerDetMapHeadApollo):
    """BEVFormer det + MapTRv2-style map head.

    This keeps the existing detection trunk untouched and upgrades only the map
    branch with:
    - decoupled decoder support
    - one-to-one / one-to-many query split
    - optional lightweight BEV/PV auxiliary segmentation heads
    """

    def __init__(
        self,
        *args,
        map_num_vec_one2one: int = 50,
        map_num_vec_one2many: int = 300,
        map_k_one2many: int = 6,
        map_lambda_one2many: float = 1.0,
        map_aux_seg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.map_num_vec_one2one = int(map_num_vec_one2one)
        self.map_num_vec_one2many = int(map_num_vec_one2many)
        self.map_k_one2many = int(map_k_one2many)
        self.map_lambda_one2many = float(map_lambda_one2many)
        self.map_aux_seg = dict(map_aux_seg or {})

        total_num_map_vec = self.map_num_vec_one2one + self.map_num_vec_one2many
        kwargs['num_map_vec'] = int(total_num_map_vec)

        super().__init__(*args, **kwargs)

        self.num_map_vec = int(total_num_map_vec)
        self.map_num_query = int(self.num_map_vec * self.map_num_pts)

        self.map_aux_seg_use = bool(self.map_aux_seg.get('use_aux_seg', False))
        self.map_aux_seg_bev = bool(self.map_aux_seg.get('bev_seg', False))
        self.map_aux_seg_pv = bool(self.map_aux_seg.get('pv_seg', False))
        self.map_aux_seg_classes = int(self.map_aux_seg.get('seg_classes', 1))
        self.map_aux_seg_loss_weight = float(self.map_aux_seg.get('loss_weight', 1.0))
        self.map_aux_seg_pos_weight = float(self.map_aux_seg.get('pos_weight', 2.0))
        self.map_aux_seg_radius = int(self.map_aux_seg.get('radius', 1))
        self.map_aux_pv_loss_weight = float(self.map_aux_seg.get('pv_loss_weight', self.map_aux_seg_loss_weight))
        self.map_aux_pv_pos_weight = float(self.map_aux_seg.get('pv_pos_weight', self.map_aux_seg_pos_weight))
        self.map_aux_pv_radius = int(self.map_aux_seg.get('pv_radius', self.map_aux_seg_radius))

        self.map_seg_head: Optional[nn.Module] = None
        self.map_pv_seg_head: Optional[nn.Module] = None
        self.map_seg_loss: Optional[nn.Module] = None
        self.map_pv_seg_loss: Optional[nn.Module] = None
        if self.map_aux_seg_use:
            if self.map_aux_seg_bev:
                self.map_seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.map_aux_seg_classes, kernel_size=1),
                )
                self.map_seg_loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self.map_aux_seg_pos_weight], dtype=torch.float32)
                )
            if self.map_aux_seg_pv:
                self.map_pv_seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.map_aux_seg_classes, kernel_size=1),
                )
                self.map_pv_seg_loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([self.map_aux_pv_pos_weight], dtype=torch.float32)
                )

    def _build_maptrv2_self_attn_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.map_num_vec_one2many <= 0:
            return None
        attn_mask = torch.zeros((self.num_map_vec, self.num_map_vec), device=device, dtype=torch.bool)
        attn_mask[self.map_num_vec_one2one :, : self.map_num_vec_one2one] = True
        attn_mask[: self.map_num_vec_one2one, self.map_num_vec_one2one :] = True
        return attn_mask

    def _repeat_gt_vecs(self, gt_vecs, repeat_times: int):
        if repeat_times <= 1:
            return gt_vecs
        if torch.is_tensor(gt_vecs):
            reps = [repeat_times] + [1] * (gt_vecs.ndim - 1)
            return gt_vecs.repeat(*reps)
        if hasattr(gt_vecs, 'instance_list'):
            instance_list = list(getattr(gt_vecs, 'instance_list', [])) * repeat_times
            return gt_vecs.__class__(
                instance_line_list=instance_list,
                sample_dist=getattr(gt_vecs, 'sample_dist', 1),
                num_samples=getattr(gt_vecs, 'num_samples', 250),
                padding=getattr(gt_vecs, 'padding', False),
                fixed_num=getattr(gt_vecs, 'fixed_num', -1),
                padding_value=getattr(gt_vecs, 'padding_value', -10000),
                patch_size=getattr(gt_vecs, 'patch_size', None),
            )
        return gt_vecs

    def _repeat_gt_for_one2many(self, gt_labels_list, gt_vecs_list):
        repeat_times = max(int(self.map_k_one2many), 1)
        rep_labels = []
        rep_vecs = []
        for labels, vecs in zip(gt_labels_list, gt_vecs_list):
            if torch.is_tensor(labels):
                rep_labels.append(labels.repeat(repeat_times))
            else:
                rep_labels.append(labels)
            rep_vecs.append(self._repeat_gt_vecs(vecs, repeat_times))
        return rep_labels, rep_vecs

    def _gt_vecs_to_points_tensor(self, gt_vecs) -> Optional[torch.Tensor]:
        if gt_vecs is None:
            return None
        if hasattr(gt_vecs, 'fixed_num_sampled_points'):
            pts = gt_vecs.fixed_num_sampled_points
        else:
            pts = gt_vecs
        if not torch.is_tensor(pts):
            return None
        if pts.ndim == 2 and (pts.size(-1) % 2 == 0):
            pts = pts.view(pts.size(0), -1, 2)
        if pts.ndim != 3 or pts.size(-1) != 2:
            return None
        return pts.to(dtype=torch.float32)

    def _draw_points_to_mask(self, mask: torch.Tensor, pts: torch.Tensor) -> None:
        if pts.numel() == 0:
            return
        x_min, x_max, y_min, y_max = self._map_range_xy()
        width = max(float(x_max - x_min), 1e-6)
        height = max(float(y_max - y_min), 1e-6)

        x = (pts[..., 0] - x_min) / width
        y = (pts[..., 1] - y_min) / height
        gx = torch.clamp((x * (self.bev_w - 1)).round().long(), 0, self.bev_w - 1)
        gy = torch.clamp((y * (self.bev_h - 1)).round().long(), 0, self.bev_h - 1)

        radius = max(int(self.map_aux_seg_radius), 0)
        for idx in range(max(int(gx.numel()) - 1, 0)):
            x0 = int(gx[idx].item())
            y0 = int(gy[idx].item())
            x1 = int(gx[idx + 1].item())
            y1 = int(gy[idx + 1].item())
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                xx = int(round(x0 + (x1 - x0) * step / steps))
                yy = int(round(y0 + (y1 - y0) * step / steps))
                x_lo = max(xx - radius, 0)
                x_hi = min(xx + radius + 1, self.bev_w)
                y_lo = max(yy - radius, 0)
                y_hi = min(yy + radius + 1, self.bev_h)
                mask[y_lo:y_hi, x_lo:x_hi] = 1.0

        if gx.numel() == 1:
            xx = int(gx[0].item())
            yy = int(gy[0].item())
            x_lo = max(xx - radius, 0)
            x_hi = min(xx + radius + 1, self.bev_w)
            y_lo = max(yy - radius, 0)
            y_hi = min(yy + radius + 1, self.bev_h)
            mask[y_lo:y_hi, x_lo:x_hi] = 1.0

    def _build_bev_seg_targets(self, gt_map_vecs_pts_loc, device: torch.device) -> Optional[torch.Tensor]:
        if gt_map_vecs_pts_loc is None:
            return None
        if not isinstance(gt_map_vecs_pts_loc, (list, tuple)):
            gt_map_vecs_pts_loc = [gt_map_vecs_pts_loc]

        targets = torch.zeros(
            (len(gt_map_vecs_pts_loc), 1, self.bev_h, self.bev_w),
            dtype=torch.float32,
            device=device,
        )
        for b, gt_vecs in enumerate(gt_map_vecs_pts_loc):
            pts = self._gt_vecs_to_points_tensor(gt_vecs)
            if pts is None or pts.numel() == 0:
                continue
            pts = pts.to(device=device)
            for line_pts in pts:
                finite_mask = torch.isfinite(line_pts).all(dim=-1)
                valid = line_pts[finite_mask]
                if valid.size(0) == 0:
                    continue
                self._draw_points_to_mask(targets[b, 0], valid)
        return targets

    def _draw_projected_points_to_mask(
        self,
        mask: torch.Tensor,
        pts_uv: torch.Tensor,
        feat_h: int,
        feat_w: int,
        img_h: float,
        img_w: float,
        radius: int,
    ) -> None:
        if pts_uv.numel() == 0:
            return

        u = pts_uv[:, 0] / max(float(img_w - 1.0), 1.0)
        v = pts_uv[:, 1] / max(float(img_h - 1.0), 1.0)
        gx = torch.clamp((u * (feat_w - 1)).round().long(), 0, feat_w - 1)
        gy = torch.clamp((v * (feat_h - 1)).round().long(), 0, feat_h - 1)

        for idx in range(max(int(gx.numel()) - 1, 0)):
            x0 = int(gx[idx].item())
            y0 = int(gy[idx].item())
            x1 = int(gx[idx + 1].item())
            y1 = int(gy[idx + 1].item())
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for step in range(steps + 1):
                xx = int(round(x0 + (x1 - x0) * step / steps))
                yy = int(round(y0 + (y1 - y0) * step / steps))
                x_lo = max(xx - radius, 0)
                x_hi = min(xx + radius + 1, feat_w)
                y_lo = max(yy - radius, 0)
                y_hi = min(yy + radius + 1, feat_h)
                mask[y_lo:y_hi, x_lo:x_hi] = 1.0

        if gx.numel() == 1:
            xx = int(gx[0].item())
            yy = int(gy[0].item())
            x_lo = max(xx - radius, 0)
            x_hi = min(xx + radius + 1, feat_w)
            y_lo = max(yy - radius, 0)
            y_hi = min(yy + radius + 1, feat_h)
            mask[y_lo:y_hi, x_lo:x_hi] = 1.0

    def _project_line_to_image(self, pts_xy: torch.Tensor, lidar2img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pts_xy.numel() == 0:
            empty = pts_xy.new_zeros((0, 2))
            return empty, pts_xy.new_zeros((0,), dtype=torch.bool)
        pts_xyz1 = torch.cat(
            [
                pts_xy,
                pts_xy.new_zeros((pts_xy.size(0), 1)),
                pts_xy.new_ones((pts_xy.size(0), 1)),
            ],
            dim=-1,
        )
        proj = pts_xyz1 @ lidar2img.transpose(0, 1)
        depth = proj[:, 2]
        valid = depth > 1e-5
        uv = proj[:, :2] / torch.clamp(depth[:, None], min=1e-5)
        return uv, valid

    def _build_pv_seg_targets(
        self,
        gt_map_vecs_pts_loc,
        img_metas: Optional[List[dict]],
        pred_seg: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if gt_map_vecs_pts_loc is None or img_metas is None:
            return None
        if not isinstance(gt_map_vecs_pts_loc, (list, tuple)):
            gt_map_vecs_pts_loc = [gt_map_vecs_pts_loc]
        if len(gt_map_vecs_pts_loc) != len(img_metas):
            return None

        bs, num_cam, _num_cls, feat_h, feat_w = pred_seg.shape
        targets = torch.zeros((bs, num_cam, 1, feat_h, feat_w), dtype=torch.float32, device=pred_seg.device)
        radius = max(int(self.map_aux_pv_radius), 0)

        for b, (gt_vecs, meta) in enumerate(zip(gt_map_vecs_pts_loc, img_metas)):
            pts = self._gt_vecs_to_points_tensor(gt_vecs)
            if pts is None or pts.numel() == 0:
                continue

            lidar2img_list = meta.get('lidar2img', None)
            pad_shape_list = meta.get('pad_shape', meta.get('img_shape', None))
            if lidar2img_list is None or pad_shape_list is None:
                continue

            pts = pts.to(device=pred_seg.device)
            for cam_idx, lidar2img_np in enumerate(lidar2img_list):
                if cam_idx >= num_cam:
                    break
                lidar2img = torch.as_tensor(lidar2img_np, dtype=pts.dtype, device=pred_seg.device)
                img_h, img_w = pad_shape_list[cam_idx][:2]

                for line_pts in pts:
                    finite_mask = torch.isfinite(line_pts).all(dim=-1)
                    valid_line = line_pts[finite_mask]
                    if valid_line.size(0) == 0:
                        continue
                    uv, valid_depth = self._project_line_to_image(valid_line, lidar2img)
                    visible = valid_depth.clone()
                    visible &= uv[:, 0] >= 0
                    visible &= uv[:, 0] <= float(img_w - 1)
                    visible &= uv[:, 1] >= 0
                    visible &= uv[:, 1] <= float(img_h - 1)
                    uv = uv[visible]
                    if uv.size(0) == 0:
                        continue
                    self._draw_projected_points_to_mask(
                        targets[b, cam_idx, 0],
                        uv,
                        feat_h=feat_h,
                        feat_w=feat_w,
                        img_h=float(img_h),
                        img_w=float(img_w),
                        radius=radius,
                    )

        return targets

    def _normalize_bev_embed(self, bev_embed: torch.Tensor, img_metas: Optional[List[dict]]):
        bs_from_metas = len(img_metas) if img_metas is not None else None
        if (bs_from_metas is not None) and (bev_embed.size(0) == bs_from_metas):
            bev_embed_btc = bev_embed
        elif (bs_from_metas is not None) and (bev_embed.size(1) == bs_from_metas):
            bev_embed_btc = bev_embed.permute(1, 0, 2).contiguous()
        else:
            bev_embed_btc = bev_embed if bev_embed.size(0) <= bev_embed.size(1) else bev_embed.permute(1, 0, 2).contiguous()
        return bev_embed_btc

    def _split_map_preds(
        self,
        all_cls_t: torch.Tensor,
        all_bbox01_t: torch.Tensor,
        all_pts01_t: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        one2one = {
            'all_cls_scores': all_cls_t[:, :, : self.map_num_vec_one2one, :],
            'all_bbox_preds': all_bbox01_t[:, :, : self.map_num_vec_one2one, :],
            'all_pts_preds': all_pts01_t[:, :, : self.map_num_vec_one2one, :, :],
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
        }
        if self.map_num_vec_one2many <= 0:
            return one2one, None
        one2many = {
            'all_cls_scores': all_cls_t[:, :, self.map_num_vec_one2one :, :],
            'all_bbox_preds': all_bbox01_t[:, :, self.map_num_vec_one2one :, :],
            'all_pts_preds': all_pts01_t[:, :, self.map_num_vec_one2one :, :, :],
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
        }
        return one2one, one2many

    @auto_fp16(apply_to=('mlvl_feats',))
    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        img_metas: List[dict],
        prev_bev: Optional[torch.Tensor] = None,
        only_bev: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.enable_det:
            outs = BEVFormerHead.forward(self, mlvl_feats, img_metas, prev_bev=prev_bev, only_bev=only_bev)
            if only_bev:
                return outs
        else:
            outs = BEVFormerHead.forward(self, mlvl_feats, img_metas, prev_bev=prev_bev, only_bev=only_bev)
            if only_bev:
                return outs

        assert isinstance(outs, dict)
        bev_embed = outs.get('bev_embed', None)
        if isinstance(bev_embed, torch.Tensor):
            self._maybe_log_nan('bev_embed', bev_embed)

        outs.setdefault('map_cls_logits', None)
        outs.setdefault('map_pts', None)
        outs.setdefault('map_pts_norm', None)

        if not self.enable_map:
            return outs

        if not hasattr(self, '_run_cfg_logged_v2'):
            self._run_cfg_logged_v2 = 0
        if self._run_cfg_logged_v2 < 1:
            rank, _ = get_dist_info()
            if rank == 0:
                logging.getLogger('mmdet').info(
                    '[det_map_v2][run_cfg] one2one=%d one2many=%d k=%d lambda=%.3f decoder=%s',
                    self.map_num_vec_one2one,
                    self.map_num_vec_one2many,
                    self.map_k_one2many,
                    self.map_lambda_one2many,
                    type(self.map_decoder).__name__ if self.map_decoder is not None else None,
                )
            self._run_cfg_logged_v2 += 1

        if not isinstance(bev_embed, torch.Tensor) or bev_embed.dim() != 3:
            return outs

        bev_embed_btc = self._normalize_bev_embed(bev_embed, img_metas)
        bs = bev_embed_btc.size(0)
        bev_global = bev_embed_btc.mean(dim=1)
        ran_decoder = False

        if (
            (self.map_decoder is not None)
            and self.map_use_point_queries
            and (self.map_cls_branches is not None)
            and (self.map_reg_branches is not None)
        ):
            try:
                num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)
                if (
                    self.map_query_embed_type == 'instance_pts'
                    and (self.map_instance_embedding is not None)
                    and (self.map_pts_embedding is not None)
                ):
                    pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
                    inst_embeds = self.map_instance_embedding.weight.unsqueeze(1)
                    object_query_embeds = (pts_embeds + inst_embeds).flatten(0, 1)
                elif self.map_point_query_embedding is not None:
                    object_query_embeds = self.map_point_query_embedding.weight
                else:
                    raise RuntimeError('map point-query embeddings are not initialized')

                query_pos, query = torch.split(object_query_embeds.to(bev_embed_btc.dtype), self.embed_dims, dim=1)
                query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
                query = query.unsqueeze(0).expand(bs, -1, -1)
                init_reference = self.map_reference_points(query_pos).sigmoid()

                q = query.permute(1, 0, 2).contiguous()
                qpos = query_pos.permute(1, 0, 2).contiguous()
                bev_value = bev_embed_btc.permute(1, 0, 2).contiguous()
                spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=q.device, dtype=torch.long)
                level_start_index = torch.tensor([0], device=q.device, dtype=torch.long)
                self_attn_mask = self._build_maptrv2_self_attn_mask(q.device)

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
                    self_attn_mask=self_attn_mask,
                    num_vec=self.num_map_vec,
                    num_pts_per_vec=self.map_num_pts,
                )

                all_cls = []
                all_pts01 = []
                all_bbox01 = []
                for lvl in range(min(int(dec_states.size(0)), num_dec)):
                    hs_l = dec_states[lvl].permute(1, 0, 2).contiguous()
                    reference = init_reference if lvl == 0 else dec_references[lvl - 1]
                    tmp = self.map_reg_branches[lvl](hs_l)
                    tmp[..., 0:2] = tmp[..., 0:2] + inverse_sigmoid(reference)
                    pts01 = tmp[..., 0:2].sigmoid()
                    pts01 = pts01.view(bs, self.num_map_vec, self.map_num_pts, 2)
                    bbox01 = pts_to_bbox_normalized(pts01)
                    hs_vec = hs_l.view(bs, self.num_map_vec, self.map_num_pts, -1).mean(dim=2)
                    cls_logits = self.map_cls_branches[lvl](hs_vec)
                    all_cls.append(cls_logits)
                    all_pts01.append(pts01)
                    all_bbox01.append(bbox01)

                all_cls_t = torch.stack(all_cls, dim=0)
                all_pts01_t = torch.stack(all_pts01, dim=0)
                all_bbox01_t = torch.stack(all_bbox01, dim=0)
                one2one_preds, one2many_preds = self._split_map_preds(all_cls_t, all_bbox01_t, all_pts01_t)

                outs['map_cls_logits'] = one2one_preds['all_cls_scores'][-1]
                outs['map_pts_norm'] = one2one_preds['all_pts_preds'][-1]
                outs['map_pts'] = self._denormalize_map_pts01(one2one_preds['all_pts_preds'][-1])
                outs['map_preds_dicts'] = one2one_preds
                outs['one2many_outs'] = one2many_preds
                ran_decoder = True
            except Exception as e:
                if not hasattr(self, '_maptrv2_decoder_fail_printed'):
                    self._maptrv2_decoder_fail_printed = 0
                if self._maptrv2_decoder_fail_printed < 3:
                    print('[det_map_v2][map_decoder] failed, falling back:', repr(e))
                    self._maptrv2_decoder_fail_printed += 1
                ran_decoder = False

        if not ran_decoder:
            q = self.map_query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
            q = q + bev_global.unsqueeze(1)
            cls_last = self.map_cls_head(q)
            pts_raw = self.map_pts_head(q).view(bs, self.num_map_vec, self.map_num_pts, 2)
            pts01_last = pts_raw.sigmoid() if self.map_pts_normalize != 'tanh' else (pts_raw.tanh() + 1.0) * 0.5

            num_dec = int(getattr(self.map_decoder, 'num_layers', 0) or self.map_decoder_num_layers or 1)
            all_cls_t = cls_last.unsqueeze(0).repeat(num_dec, 1, 1, 1)
            all_pts01_t = pts01_last.unsqueeze(0).repeat(num_dec, 1, 1, 1, 1)
            all_bbox01_t = pts_to_bbox_normalized(all_pts01_t.reshape(-1, self.num_map_vec, self.map_num_pts, 2))
            all_bbox01_t = all_bbox01_t.view(num_dec, bs, self.num_map_vec, 4)
            one2one_preds, one2many_preds = self._split_map_preds(all_cls_t, all_bbox01_t, all_pts01_t)

            outs['map_cls_logits'] = one2one_preds['all_cls_scores'][-1]
            outs['map_pts_norm'] = one2one_preds['all_pts_preds'][-1]
            outs['map_pts'] = self._denormalize_map_pts01(one2one_preds['all_pts_preds'][-1])
            outs['map_preds_dicts'] = one2one_preds
            outs['one2many_outs'] = one2many_preds

        if self.map_aux_seg_use and (self.map_seg_head is not None):
            seg_bev = bev_embed_btc.view(bs, self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2).contiguous()
            outs['map_seg'] = self.map_seg_head(seg_bev)
        else:
            outs['map_seg'] = None

        if self.map_aux_seg_use and (self.map_pv_seg_head is not None):
            feat = mlvl_feats[-1]
            bs_f, num_cam, _c, feat_h, feat_w = feat.shape
            pv = self.map_pv_seg_head(feat.flatten(0, 1))
            outs['map_pv_seg'] = pv.view(bs_f, num_cam, -1, feat_h, feat_w)
        else:
            outs['map_pv_seg'] = None

        if isinstance(outs['map_cls_logits'], torch.Tensor):
            self._maybe_log_nan('map_cls_logits', outs['map_cls_logits'])
        if isinstance(outs['map_pts'], torch.Tensor):
            self._maybe_log_nan('map_pts', outs['map_pts'])
        return outs

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
        losses = super().loss(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_feats=pts_feats,
            occ_gts=occ_gts,
            flow_gts=flow_gts,
            outs=outs,
            img_metas=img_metas,
            gt_map_vecs_label=gt_map_vecs_label,
            gt_map_vecs_pts_loc=gt_map_vecs_pts_loc,
            **kwargs,
        )

        outs = outs or {}
        if (
            not self.enable_map
            or self.map_num_vec_one2many <= 0
            or self.map_lambda_one2many <= 0
            or gt_map_vecs_label is None
            or gt_map_vecs_pts_loc is None
            or self.maptr_loss_head is None
        ):
            return losses

        one2many_preds = outs.get('one2many_outs', None)
        if not isinstance(one2many_preds, dict):
            one2many_preds = None

        if not isinstance(gt_map_vecs_label, (list, tuple)):
            gt_map_vecs_label = [gt_map_vecs_label]
        if not isinstance(gt_map_vecs_pts_loc, (list, tuple)):
            gt_map_vecs_pts_loc = [gt_map_vecs_pts_loc]

        rep_labels, rep_vecs = self._repeat_gt_for_one2many(gt_map_vecs_label, gt_map_vecs_pts_loc)
        loss_anchor = next(iter(losses.values())) if len(losses) > 0 else one2many_preds['all_cls_scores'].sum() * 0.0

        if isinstance(one2many_preds, dict):
            try:
                one2many_loss = self.maptr_loss_head.loss(
                    gt_vecs_list=rep_vecs,
                    gt_labels_list=rep_labels,
                    preds_dicts=one2many_preds,
                    gt_bboxes_ignore=None,
                )
                losses['loss_map_o2m_cls'] = self.map_lambda_one2many * one2many_loss.get('loss_cls', loss_anchor)
                losses['loss_map_o2m_bbox'] = self.map_lambda_one2many * one2many_loss.get('loss_bbox', loss_anchor)
                losses['loss_map_o2m_iou'] = self.map_lambda_one2many * one2many_loss.get('loss_iou', loss_anchor)
                losses['loss_map_o2m_pts'] = self.map_lambda_one2many * one2many_loss.get('loss_pts', loss_anchor)
                losses['loss_map_o2m_dir'] = self.map_lambda_one2many * one2many_loss.get('loss_dir', loss_anchor)
                losses['loss_map_o2m'] = (
                    losses['loss_map_o2m_cls']
                    + losses['loss_map_o2m_bbox']
                    + losses['loss_map_o2m_iou']
                    + losses['loss_map_o2m_pts']
                    + losses['loss_map_o2m_dir']
                )
                if 'loss_map' in losses:
                    losses['loss_map'] = losses['loss_map'] + losses['loss_map_o2m']
                else:
                    losses['loss_map'] = losses['loss_map_o2m']
            except Exception as e:
                print('[det_map_v2][one2many] map loss failed:', repr(e))
                losses['loss_map_o2m'] = loss_anchor * 0.0

        if self.map_aux_seg_use and self.map_aux_seg_bev and self.map_seg_loss is not None:
            pred_seg = outs.get('map_seg', None)
            if isinstance(pred_seg, torch.Tensor):
                seg_targets = self._build_bev_seg_targets(gt_map_vecs_pts_loc, pred_seg.device)
                if seg_targets is not None:
                    pos_weight = self.map_seg_loss.pos_weight
                    if pos_weight.device != pred_seg.device:
                        self.map_seg_loss.pos_weight = pos_weight.to(pred_seg.device)
                    loss_map_seg = self.map_seg_loss(pred_seg.float(), seg_targets.float())
                    loss_map_seg = loss_map_seg * self.map_aux_seg_loss_weight
                    losses['loss_map_seg'] = loss_map_seg
                    if 'loss_map' in losses:
                        losses['loss_map'] = losses['loss_map'] + loss_map_seg
                    else:
                        losses['loss_map'] = loss_map_seg

        if self.map_aux_seg_use and self.map_aux_seg_pv and self.map_pv_seg_loss is not None:
            pred_pv_seg = outs.get('map_pv_seg', None)
            if isinstance(pred_pv_seg, torch.Tensor):
                pv_targets = self._build_pv_seg_targets(gt_map_vecs_pts_loc, img_metas, pred_pv_seg)
                if pv_targets is not None:
                    pos_weight = self.map_pv_seg_loss.pos_weight
                    if pos_weight.device != pred_pv_seg.device:
                        self.map_pv_seg_loss.pos_weight = pos_weight.to(pred_pv_seg.device)
                    loss_map_pv_seg = self.map_pv_seg_loss(pred_pv_seg.float(), pv_targets.float())
                    loss_map_pv_seg = loss_map_pv_seg * self.map_aux_pv_loss_weight
                    losses['loss_map_pv_seg'] = loss_map_pv_seg
                    if 'loss_map' in losses:
                        losses['loss_map'] = losses['loss_map'] + loss_map_pv_seg
                    else:
                        losses['loss_map'] = loss_map_pv_seg

        return losses
