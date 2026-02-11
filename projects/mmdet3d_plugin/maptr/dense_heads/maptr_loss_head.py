import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.bbox import build_assigner, build_sampler
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.models import build_loss

# Ensure custom MapTR registries are loaded (LOSSES / MATCH_COST / ASSIGNERS).
from .. import assigners as _maptr_assigners  # noqa: F401
from .. import losses as _maptr_losses  # noqa: F401


def normalize_2d_bbox(bboxes: torch.Tensor, pc_range) -> torch.Tensor:
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts: torch.Tensor, pc_range) -> torch.Tensor:
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes: torch.Tensor, pc_range) -> torch.Tensor:
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return bboxes


def denormalize_2d_pts(pts: torch.Tensor, pc_range) -> torch.Tensor:
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return new_pts


class MapTRLossHead(nn.Module):
    """MapTR official target assignment + loss, decoupled from forward().

    This module reuses MapTR's assigner/cost + loss definitions and expects
    MapTR-style vector-map GT containers (e.g. LiDARInstanceLines) in
    `gt_vecs_list`.

    Inputs:
      preds_dicts:
        - all_cls_scores: [num_dec, bs, num_vec, num_classes]
        - all_bbox_preds: [num_dec, bs, num_vec, 4] (normalized cxcywh)
        - all_pts_preds:  [num_dec, bs, num_vec, num_pts, 2] (normalized [0,1])
    """

    def __init__(
        self,
        num_classes: int,
        pc_range,
        num_pts_per_vec: int,
        num_pts_per_gt_vec: Optional[int] = None,
        dir_interval: int = 1,
        gt_shift_pts_pattern: str = 'v2',
        sync_cls_avg_factor: bool = True,
        bg_cls_weight: float = 0.1,
        loss_cls: Optional[dict] = None,
        loss_bbox: Optional[dict] = None,
        loss_iou: Optional[dict] = None,
        loss_pts: Optional[dict] = None,
        loss_dir: Optional[dict] = None,
        assigner: Optional[dict] = None,
        sampler: Optional[dict] = None,
        code_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.cls_out_channels = int(num_classes)
        self.pc_range = pc_range

        self.num_pts_per_vec = int(num_pts_per_vec)
        self.num_pts_per_gt_vec = int(num_pts_per_gt_vec) if num_pts_per_gt_vec is not None else int(num_pts_per_vec)
        self.dir_interval = int(dir_interval)

        self.gt_shift_pts_pattern = str(gt_shift_pts_pattern)
        self.sync_cls_avg_factor = bool(sync_cls_avg_factor)
        self.bg_cls_weight = float(bg_cls_weight)

        self.code_weights = code_weights if code_weights is not None else [1.0, 1.0, 1.0, 1.0]
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)

        self.loss_cls = build_loss(
            loss_cls
            or dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0)
        )
        self.loss_bbox = build_loss(loss_bbox or dict(type='L1Loss', loss_weight=0.0))
        self.loss_iou = build_loss(loss_iou or dict(type='GIoULoss', loss_weight=0.0))
        self.loss_pts = build_loss(loss_pts or dict(type='PtsL1Loss', loss_weight=5.0))
        self.loss_dir = build_loss(loss_dir or dict(type='PtsDirCosLoss', loss_weight=0.005))

        self.assigner = build_assigner(
            assigner
            or dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5.0),
                pc_range=pc_range,
            )
        )

        self.sampler = build_sampler(sampler or dict(type='PseudoSampler'), context=self)

    def _get_gt_shifts_pts(self, gt_vecs) -> torch.Tensor:
        if self.gt_shift_pts_pattern == 'v0':
            return gt_vecs.shift_fixed_num_sampled_points
        if self.gt_shift_pts_pattern == 'v1':
            return gt_vecs.shift_fixed_num_sampled_points_v1
        if self.gt_shift_pts_pattern == 'v2':
            return gt_vecs.shift_fixed_num_sampled_points_v2
        if self.gt_shift_pts_pattern == 'v3':
            return gt_vecs.shift_fixed_num_sampled_points_v3
        if self.gt_shift_pts_pattern == 'v4':
            return gt_vecs.shift_fixed_num_sampled_points_v4
        raise NotImplementedError

    def _get_target_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        pts_pred: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_shifts_pts: torch.Tensor,
        gt_bboxes_ignore=None,
    ):
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]

        assign_result, order_index = self.assigner.assign(
            bbox_pred,
            cls_score,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_shifts_pts,
            gt_bboxes_ignore,
        )

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]

        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pts_targets,
            pts_weights,
            pos_inds,
            neg_inds,
        )

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        assert gt_bboxes_ignore_list is None, 'Only supports gt_bboxes_ignore=None'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        pts_preds: torch.Tensor,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(
                pts_preds,
                size=(self.num_pts_per_gt_vec),
                mode='linear',
                align_corners=True,
            )
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )

        dir_weights = pts_weights[:, :-self.dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:, self.dir_interval :, :] - denormed_pts_preds[:, : -self.dir_interval, :]
        pts_targets_dir = pts_targets[:, self.dir_interval :, :] - pts_targets[:, : -self.dir_interval, :]

        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def loss(
        self,
        gt_vecs_list,
        gt_labels_list,
        preds_dicts: Dict[str, torch.Tensor],
        gt_bboxes_ignore=None,
    ) -> Dict[str, torch.Tensor]:
        assert gt_bboxes_ignore is None, 'Only supports gt_bboxes_ignore=None'

        # Match official MapTRHead behavior (@force_fp32(apply_to=('preds_dicts'))):
        # ensure assignment + loss computations run in FP32 even under AMP.
        all_cls_scores = preds_dicts['all_cls_scores'].float()
        all_bbox_preds = preds_dicts['all_bbox_preds'].float()
        all_pts_preds = preds_dicts['all_pts_preds'].float()

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_vecs_list = copy.deepcopy(gt_vecs_list)

        gt_bboxes_list = [gt.bbox.to(device) for gt in gt_vecs_list]
        gt_shifts_pts_list = [self._get_gt_shifts_pts(gt).to(device) for gt in gt_vecs_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_pts_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict: Dict[str, torch.Tensor] = {}
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(
            losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_pts[:-1], losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1

        return loss_dict


def pts_to_bbox_normalized(pts01: torch.Tensor) -> torch.Tensor:
    """MapTR transform_method='minmax' but for already-normalized pts.

    Args:
        pts01: [bs, num_vec, num_pts, 2] with x/y in [0,1]

    Returns:
        bbox_cxcywh: [bs, num_vec, 4] in normalized cxcywh.
    """
    pts_x = pts01[..., 0]
    pts_y = pts01[..., 1]

    xmin = pts_x.min(dim=2, keepdim=True)[0]
    xmax = pts_x.max(dim=2, keepdim=True)[0]
    ymin = pts_y.min(dim=2, keepdim=True)[0]
    ymax = pts_y.max(dim=2, keepdim=True)[0]

    bbox_xyxy = torch.cat([xmin, ymin, xmax, ymax], dim=2)
    bbox_cxcywh = bbox_xyxy_to_cxcywh(bbox_xyxy)
    return bbox_cxcywh
