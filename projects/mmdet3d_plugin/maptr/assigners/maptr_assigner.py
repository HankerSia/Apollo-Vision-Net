import torch
import torch.nn.functional as F

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])
    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    new_pts[..., 1:2] = (pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    return new_pts


@BBOX_ASSIGNERS.register_module()
class MapTRAssigner(BaseAssigner):
    """MapTR one-to-one assigner (Hungarian)."""

    def __init__(
        self,
        cls_cost=dict(type='ClassificationCost', weight=1.0),
        reg_cost=dict(type='BBoxL1Cost', weight=1.0),
        iou_cost=dict(type='IoUCost', weight=0.0),
        pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
        pc_range=None,
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pts_cost = build_match_cost(pts_cost)
        self.pc_range = pc_range

    def assign(
        self,
        bbox_pred,
        cls_pred,
        pts_pred,
        gt_bboxes,
        gt_labels,
        gt_pts,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        assert gt_bboxes_ignore is None, 'Only supports gt_bboxes_ignore=None'
        assert bbox_pred.shape[-1] == 4, 'Only support bbox pred shape is 4 dims'

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), None

        cls_cost = self.cls_cost(cls_pred, gt_labels)

        normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :4], normalized_gt_bboxes[:, :4])

        _, num_orders, num_pts_per_gtline, _ = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)

        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(
                pts_pred.permute(0, 2, 1),
                size=(num_pts_per_gtline),
                mode='linear',
                align_corners=True,
            )
            pts_pred_interpolated = pts_pred_interpolated.permute(0, 2, 1).contiguous()
        else:
            pts_pred_interpolated = pts_pred

        pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)
        pts_cost_ordered = pts_cost_ordered.view(num_bboxes, num_gts, num_orders)
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
        iou_cost = self.iou_cost(bboxes, gt_bboxes)

        cost = cls_cost + reg_cost + iou_cost + pts_cost

        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels), order_index
