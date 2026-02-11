# Copyright (c) OpenMMLab. All rights reserved.

import functools

import mmcv
import torch
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.models import weighted_loss
from mmdet.models.builder import LOSSES
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
    else:
        if reduction == 'mean':
            loss = loss.sum()
            loss = loss / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


@mmcv.jit(derivate=True, coderize=True)
def custom_weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
    else:
        if reduction == 'mean':
            loss = loss.permute(1, 0, 2, 3).contiguous()
            loss = loss.sum((1, 2, 3))
            loss = loss / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def custom_weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weighted_dir_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_smooth_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1), 1, 1)
    assert pred.size() == target.size()
    loss = smooth_l1_loss(pred, target, reduction='none')
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_l1_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1), 1, 1)
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    if target.numel() == 0:
        return pred.sum() * 0
    num_samples, num_dir, _ = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir)).flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


@LOSSES.register_module()
class OrderedPtsSmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * ordered_pts_smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_dir


@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


@LOSSES.register_module()
class OrderedPtsL1Loss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss_bbox = self.loss_weight * ordered_pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_bbox


@MATCH_COST.register_module()
class OrderedPtsSmoothL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_gts, num_orders, _, _ = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1).unsqueeze(1).repeat(1, num_gts * num_orders, 1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1).unsqueeze(0).repeat(bbox_pred.size(0), 1, 1)
        bbox_cost = smooth_l1_loss(bbox_pred, gt_bboxes, reduction='none').sum(-1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class PtsL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_gts, _, _ = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.view(num_gts, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class OrderedPtsL1Cost(object):
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        num_gts, num_orders, _, _ = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class MyChamferDistanceCost:
    def __init__(self, loss_src_weight=1.0, loss_dst_weight=1.0):
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def __call__(self, src, dst, src_weight=1.0, dst_weight=1.0):
        src_expand = src.unsqueeze(1).repeat(1, dst.shape[0], 1, 1)
        dst_expand = dst.unsqueeze(0).repeat(src.shape[0], 1, 1, 1)
        distance = torch.cdist(src_expand, dst_expand)
        src2dst_distance = torch.min(distance, dim=3)[0]
        dst2src_distance = torch.min(distance, dim=2)[0]
        loss_src = (src2dst_distance * src_weight).mean(-1)
        loss_dst = (dst2src_distance * dst_weight).mean(-1)
        loss = loss_src * self.loss_src_weight + loss_dst * self.loss_dst_weight
        return loss


@mmcv.jit(derivate=True, coderize=True)
def chamfer_distance(
    src,
    dst,
    src_weight=1.0,
    dst_weight=1.0,
    reduction='mean',
    avg_factor=None,
):
    distance = torch.cdist(src, dst)
    src2dst_distance, indices1 = torch.min(distance, dim=2)
    dst2src_distance, indices2 = torch.min(distance, dim=1)

    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)

    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            raise ValueError('MyCDLoss can not be used with reduction=`none`')
        elif reduction_enum == 1:
            loss_src = loss_src.mean(-1).mean()
            loss_dst = loss_dst.mean(-1).mean()
        elif reduction_enum == 2:
            loss_src = loss_src.mean(-1).sum()
            loss_dst = loss_dst.mean(-1).sum()
        else:
            raise NotImplementedError
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss_src = loss_src.mean(-1).sum() / (avg_factor + eps)
            loss_dst = loss_dst.mean(-1).sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss_src, loss_dst, indices1, indices2


@LOSSES.register_module()
class MyChamferDistance(nn.Module):
    def __init__(self, reduction='mean', loss_src_weight=1.0, loss_dst_weight=1.0):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(
        self,
        source,
        target,
        src_weight=1.0,
        dst_weight=1.0,
        avg_factor=None,
        reduction_override=None,
        return_indices=False,
        **kwargs,
    ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, reduction, avg_factor=avg_factor
        )

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight
        loss_pts = loss_source + loss_target

        if return_indices:
            return loss_pts, indices1, indices2
        else:
            return loss_pts
