# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

import copy
from tkinter import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
import math

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from .bevformer_occupancy_head import BEVFormerOccupancyHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding,build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16
from mmdet3d.models.builder import build_loss, build_head
from mmcv.ops import points_in_boxes_part
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.models.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from projects.mmdet3d_plugin.models.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from projects.mmdet3d_plugin.models.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss

from mmcv.runner import get_dist_info

@HEADS.register_module()
class BEVFormerOccupancyHeadApollo(BEVFormerOccupancyHead):
    def __init__(self,
                 *args,
                 group_detr=1,
                 occ_tsa=None,
                 positional_encoding_occ=None,
                 balance_cls_weight=False,
                 loss_lovasz=None,
                 loss_affinity=None,
                 **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        loss_type = {
            'FocalLoss': 'focal_loss',
            'CustomFocalLoss': 'CustomFocalLoss',
            'CrossEntropyLoss': 'ce_loss'
        }
        kwargs['occ_loss_type'] = loss_type[kwargs['loss_occupancy']['type']]
        super().__init__(*args, **kwargs)
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.occ_zdim*self.occ_dims, kernel_size=1),
            nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.occ_zdim*self.occ_dims, self.occ_zdim*self.occ_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
            nn.ReLU(inplace=True),
        )
        if occ_tsa:
            self.upsample_layer = nn.Sequential(
                    nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU(inplace=True),
                )
            self.occ_tsa = build_transformer_layer_sequence(occ_tsa)
            self.occ_tsa_head = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.occ_zdim*self.occ_dims, kernel_size=1),
                nn.BatchNorm2d(self.occ_zdim*self.occ_dims),
                nn.ReLU(inplace=True)
            )
            if positional_encoding_occ is not None:
                    positional_encoding_occ['row_num_embed'] = self.occ_xdim
                    positional_encoding_occ['col_num_embed'] = self.occ_ydim

                    self.positional_encoding_occ = build_positional_encoding(
                        positional_encoding_occ)
                    assert 'num_feats' in positional_encoding_occ
                    num_feats = positional_encoding_occ['num_feats']
                    assert num_feats * 2 == self.embed_dims, 'embed_dims should' \

                    f' be exactly 2 times of num_feats. Found {self.embed_dims}' \

                    f' and {num_feats}.'
            else:
                self.positional_encoding_occ = None
        else:
            self.occ_tsa = None

        self.predict_free_voxel = (self.occupancy_classes == len(nusc_class_names))
        self.loss_lovasz = loss_lovasz
        self.loss_affinity = loss_affinity
        # if self.loss_affinity is not None and not self.predict_free_voxel:
        #     raise ValueError('Affinity loss can only be used when predicting free voxels')
        if balance_cls_weight:
            self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies + 0.001))
        else:
            self.class_weights = torch.ones(17)/17  # FIXME hardcode 17

    def upsample_tsa_occ(self, feat_flatten, spatial_shapes, level_start_index, bev_for_occ, bs, seq_len, **kwargs):
        bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w)
        upsampled_bev_embed = self.upsample_layer(bev_for_occ)
        bev_queries = upsampled_bev_embed.flatten(2).permute(2, 0, 1)
        dtype = feat_flatten.dtype
        if self.positional_encoding_occ is not None:
            occ_bev_mask = torch.zeros((bs, self.occ_xdim, self.occ_ydim),
                                        device=bev_queries.device).to(dtype)
            query_pos = self.positional_encoding_occ(occ_bev_mask).to(dtype)
            query_pos = query_pos.flatten(2).permute(0, 2, 1)
        else:
            query_pos = None
        bev_embed = self.occ_tsa(
            bev_queries,
            feat_flatten,
            feat_flatten,
            query_pos=query_pos,
            bev_h=self.occ_xdim,
            bev_w=self.occ_ydim,
            bev_pos=torch.zeros_like(bev_queries).permute(1, 0, 2),  # fake
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=torch.zeros((bs, 2), device=bev_queries.device).to(dtype),  # fake
            **kwargs
        )
        occ_pred = self.occ_proj(bev_embed)
        occ_pred = occ_pred.view(bs * seq_len, self.occ_xdim*self.occ_ydim, self.occ_zdim, self.occ_dims)
        occ_pred = occ_pred.permute(0, 2, 1, 3)
        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        return occ_pred

    def upsample_occ(self, bev_for_occ, bs, seq_len):
        bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w)
        occ_pred = self.upsample_layer(bev_for_occ)
        occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_dims, self.occ_zdim, self.occ_xdim, self.occ_ydim)
        occ_pred = occ_pred.permute(0, 2, 3, 4, 1)
        occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_zdim*self.occ_xdim*self.occ_ydim, self.occ_dims)
        return occ_pred

    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.transformer.decoder is not None:  # 3D detectio query
            object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:  # NOTE: Only difference to bevformer head
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        if isinstance(prev_bev, list) or isinstance(prev_bev, tuple):
            prev_bevs = [f.permute(1, 0, 2) for f in prev_bev]
            prev_bev = prev_bev[-1]
        elif torch.is_tensor(prev_bev) or prev_bev is None:
            prev_bevs = None
        else:
            raise NotImplementedError
        
        if only_bev:  # only use encoder to obtain BEV features
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        elif self.only_occ:
            bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            # bev_embed: (bs, num_query, embed_dims)
            if prev_bevs is not None:
                bev_for_occ = torch.cat((*prev_bevs, bev_embed), dim=1)
                seq_len = len(prev_bevs) + 1
            else:
                bev_for_occ = bev_embed
                seq_len = 1
            
            if self.occ_head_type == 'mlp':
                if self.use_fine_occ:
                    occ_pred = self.occ_proj(bev_for_occ)
                    occ_pred = occ_pred.view(self.bev_h*self.bev_w, bs * seq_len, self.occ_zdim//2, self.occ_dims)
                    occ_pred = occ_pred.permute(1, 3, 2, 0)
                    occ_pred = occ_pred.view(bs * seq_len, self.occ_dims, self.occ_zdim//2, self.bev_h, self.bev_w)
                    occ_pred = self.up_sample(occ_pred)  # (bs*seq_len, C, occ_z, occ_y, occ_x)
                    occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
                    outputs_occupancy = self.occ_branches(occ_pred)
                    outputs_flow = None
                else:
                    occ_pred = self.occ_proj(bev_for_occ)
                    occ_pred = occ_pred.view(self.bev_h*self.bev_w, bs * seq_len, self.occ_zdim, self.occ_dims)
                    occ_pred = occ_pred.permute(1, 2, 0, 3) # bs*seq_len, z, x*y, dim
                    if self.with_occupancy_flow:
                        occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

                    occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
                    outputs_occupancy = self.occ_branches(occ_pred)

                    if self.predict_flow:
                        outputs_flow = self.flow_branches(occ_pred)
                    else:
                        outputs_flow = None
                    
            elif self.occ_head_type == 'cnn':
                # bev_for_occ.shape: (bs, num_query, embed_dims)
                bev_for_occ = bev_for_occ.view(bs, 1, self.bev_h, self.bev_w, self.embed_dims)
                bev_for_occ = bev_for_occ.permute(0, 2, 3, 1, 4).flatten(3)  # (bs, bev_h, bev_w, -1)
                occ_pred = self.occ_proj(bev_for_occ)
                if self.use_fine_occ:
                    occ_pred = occ_pred.view(bs, self.bev_h, self.bev_w, self.occ_zdim//2, self.occ_dims)
                    occ_pred = occ_pred.permute(0, 4, 3, 1, 2)
                    occ_pred = self.up_sample(occ_pred)  # (bs, C, d, h, w)
                else:
                    occ_pred = occ_pred.view(bs, self.bev_h, self.bev_w, self.occ_zdim, self.occ_dims)
                    occ_pred = occ_pred.permute(0, 4, 3, 1, 2)  # (bs, occ_dims, z_dim, bev_h, bev_w)
                outputs_occupancy, outputs_flow = self.occ_seg_head(occ_pred)
                outputs_occupancy = outputs_occupancy.reshape(bs, self.occupancy_classes, -1)
                outputs_occupancy = outputs_occupancy.permute(0, 2, 1)
                if outputs_flow is not None:
                    outputs_flow = outputs_flow.reshape(bs, -1, 2)
            else:
                raise NotImplementedError

            # bev_embed = bev_embed.permute(1, 0, 2)  # (num_query, bs, embed_dims)
            outs = {
                'bev_embed': bev_embed,
                'all_cls_scores': None,
                'all_bbox_preds': None,
                'occupancy_preds': outputs_occupancy,
                'flow_preds': outputs_flow,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_occupancy_preds': None
            }

            return outs

        else:
            # import pdb; pdb.set_trace()
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                return_intermediate=True if self.occ_tsa else False
            )

        if self.occ_tsa:
            bev_embed, hs, init_reference, inter_references, feat_flatten, spatial_shapes, level_start_index = outputs
        else:
            bev_embed, hs, init_reference, inter_references = outputs

        # bev_embed: [bev_h*bev_w, bs, embed_dims]
        # hs:  (num_dec_layers, num_query, bs, embed_dims)
        # init_reference: (bs, num_query, 3)  in (0, 1)
        # inter_references:  (num_dec_layers, bs, num_query, 3)  in (0, 1)
        if prev_bevs is not None:
            bev_for_occ = torch.cat((*prev_bevs, bev_embed), dim=1)
            seq_len = len(prev_bevs) + 1
        else:
            bev_for_occ = bev_embed
            seq_len = 1
        if self.occ_tsa is None:
            occ_pred = self.upsample_occ(bev_for_occ, bs, seq_len) # bs*seq_len, z*x*y, num_classes
        else:
            occ_pred = self.upsample_tsa_occ(
                feat_flatten, 
                spatial_shapes, 
                level_start_index, 
                bev_for_occ, 
                bs, 
                seq_len,
                img_metas=img_metas)

        if self.with_occupancy_flow:
            occ_pred = self.occupancy_aggregation(occ_pred.view(bs, seq_len, self.occ_zdim, self.occ_xdim, self.occ_ydim, self.occ_dims))

        occ_pred = occ_pred.reshape(bs * seq_len, -1, self.occ_dims)
        outputs_occupancy = self.occ_branches(occ_pred)

        if self.predict_flow:
            outputs_flow = self.flow_branches(occ_pred)
        else:
            outputs_flow = None

        # if self.with_color_render:
        #     outputs_color = self.color_branches(occ_pred)
        #     color_in_cams = self.voxel2image(outputs_color)
        #     occupancy_in_cams = self.voxel2image(outputs_occupancy)
        #     image_pred = self.render_image(color_in_cams, occupancy_in_cams)

        hs = hs.permute(0, 2, 1, 3)  # (num_dec_layers, bs, num_query, embed_dims)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'occupancy_preds': outputs_occupancy,
            'flow_preds': outputs_flow,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_occupancy_preds': None
        }

        return outs

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    occupancy_preds,
                    flow_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    gt_occupancy=None,
                    gt_flow=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
                for 3d: shape is [bs, num_query, 10]   (cx, cy, w, l, cz, h, sin(theta), cos(theta), vx, vy)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                for 3d: tensor.shape = (num_gt_box, 9)
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)  # bs
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)  # (num_query)
        label_weights = torch.cat(label_weights_list, 0)  # (num_query)
        bbox_targets = torch.cat(bbox_targets_list, 0)  # (num_query, 9)
        bbox_weights = torch.cat(bbox_weights_list, 0)  # (num_query, 10)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)  # (bs*num_query, 10)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))  # (bs*num_query, 10)

        # transfer gt anno(cx, cy, cz, w, l, h, rot, vx, vy) to (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy)
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)  # (num_query, 10)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)  # remove empty
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # loss occupancy
        if occupancy_preds is not None:
            if self.occ_loss_type == 'focal_loss':
                num_pos_occ = torch.sum(gt_occupancy < self.occupancy_classes)
                occ_avg_factor = num_pos_occ * 1.0
                loss_occupancy = self.loss_occupancy(occupancy_preds, gt_occupancy, avg_factor=occ_avg_factor)
            # occupancy_preds: (bs*seq_len*occ_zdim*occ_ydim*occ_xdim, num_classes) -> (bs*seq_len, num_classes, occ_ydim, occ_xdim, occ_zdim)
            occupancy_preds = occupancy_preds.view(num_imgs, self.occ_zdim, self.occ_xdim, self.occ_ydim, -1)
            occupancy_preds = occupancy_preds.permute(0, 4, 2, 3, 1).contiguous()
            # gt_occupancy: (bs*seq_len*occ_zdim*occ_ydim*occ_xdim) -> (bs*seq_len, occ_ydim, occ_xdim, occ_zdim)
            gt_occupancy = gt_occupancy.view(num_imgs, self.occ_zdim, self.occ_xdim, self.occ_ydim)
            gt_occupancy = gt_occupancy.permute(0, 2, 3, 1).contiguous()

            if self.occ_loss_type == 'CustomFocalLoss':
                loss_occupancy = self.loss_occupancy(occupancy_preds, gt_occupancy, self.class_weights.type_as(occupancy_preds), ignore_index=255)
            elif self.occ_loss_type == 'ce_loss':
                loss_occupancy = CE_ssc_loss(occupancy_preds, gt_occupancy, self.class_weights.type_as(occupancy_preds), ignore_index=255)

            if self.loss_lovasz:
                loss_lovasz_softmax = self.loss_lovasz['loss_weight'] * lovasz_softmax(
                    torch.softmax(occupancy_preds, dim=1), gt_occupancy)
            else:
                loss_lovasz_softmax = torch.zeros_like(loss_occupancy)

            if self.loss_affinity:
                if self.loss_affinity['loss_voxel_sem_scal_weight'] != 0:
                    loss_sem_scal = self.loss_affinity['loss_voxel_sem_scal_weight'] * \
                        sem_scal_loss(occupancy_preds, gt_occupancy)
                else:
                    loss_sem_scal = torch.zeros_like(loss_occupancy)
                if self.loss_affinity['loss_voxel_geo_scal_weight'] != 0:
                    loss_geo_scal = self.loss_affinity['loss_voxel_geo_scal_weight'] * \
                        geo_scal_loss(occupancy_preds, gt_occupancy, non_empty_idx=self.occupancy_classes-1)
                else:
                    loss_geo_scal = torch.zeros_like(loss_occupancy)
            else:
                loss_sem_scal = torch.zeros_like(loss_occupancy)
                loss_geo_scal = torch.zeros_like(loss_occupancy)
        else:
            loss_occupancy = torch.zeros_like(loss_cls)
            loss_lovasz_softmax = torch.zeros_like(loss_cls)
            loss_sem_scal = torch.zeros_like(loss_cls)
            loss_geo_scal = torch.zeros_like(loss_cls)

        # loss flow: TODO only calculate foreground object flow
        if flow_preds is not None:
            object_mask = gt_occupancy < 10  # 0-9: object 10-15: background
            num_pos_flow = torch.sum(object_mask)
            flow_avg_factor = num_pos_flow * 1.0
            loss_flow = self.loss_flow(flow_preds[object_mask], gt_flow[object_mask], avg_factor=flow_avg_factor)
        else:
            loss_flow = torch.zeros_like(loss_cls)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_occupancy = torch.nan_to_num(loss_occupancy)
            loss_flow = torch.nan_to_num(loss_flow)
        return loss_cls, loss_bbox, loss_occupancy, loss_flow, loss_lovasz_softmax, loss_sem_scal, loss_geo_scal

    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             point_coords,
             occ_gts,
             flow_gts,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']  # (num_dec_layers, bs, num_query, 10)
        all_bbox_preds = preds_dicts['all_bbox_preds']
        occupancy_preds = preds_dicts['occupancy_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 11)
        flow_preds = preds_dicts['flow_preds']  # (bs, occ_zdim*occ_ydim*occ_xdim, 2)
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_occupancy_preds = preds_dicts['enc_occupancy_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None and enc_occupancy_preds is None

        bs = len(gt_bboxes_list)
        if isinstance(gt_bboxes_list[0], list):
            # if has temporal frames
            seq_len = len(gt_bboxes_list[0])
            temporal_gt_bboxes_list = gt_bboxes_list
            temporal_gt_labels_list = gt_labels_list
            gt_bboxes_list = [gt_bboxes[-1] for gt_bboxes in gt_bboxes_list]
            gt_labels_list = [gt_labels[-1] for gt_labels in gt_labels_list]
        else:
            seq_len = 1
            temporal_gt_bboxes_list = [[gt_bboxes] for gt_bboxes in gt_bboxes_list]
            temporal_gt_labels_list = [[gt_labels] for gt_labels in gt_labels_list]

        # assert seq_len == len(point_coords[0])
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device


        if occ_gts:
            if not self.predict_free_voxel:
                gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes).to(device)
            else:
                gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*(self.occupancy_classes-1)).to(device)
            for sample_id in range(len(temporal_gt_bboxes_list)):
                for frame_id in range(seq_len):
                    occ_gt = occ_gts[sample_id][frame_id].long()
                    gt_occupancy[sample_id*seq_len+frame_id][[occ_gt[:, 0]]] = occ_gt[:, 1]

            if flow_preds is not None:
                gt_flow = torch.zeros((bs*seq_len, self.voxel_num, 2)).to(device)
                for sample_id in range(len(temporal_gt_bboxes_list)):
                    for frame_id in range(seq_len):
                        occ_gt = occ_gts[sample_id][frame_id].long()
                        flow_gt_sparse = flow_gts[sample_id][frame_id]  # only store the flow of occupied grid
                        gt_flow[sample_id*seq_len+frame_id][[occ_gt[:, 0]]] = flow_gt_sparse

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]  

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        gt_occupancy = gt_occupancy.view(-1) 
        occupancy_preds = occupancy_preds.view(-1, self.occupancy_classes)  

        if flow_preds is not None:
            flow_preds = flow_preds.view(-1, self.flow_gt_dimension)
            gt_flow = gt_flow.view(-1, self.flow_gt_dimension)
        else:
            flow_preds, gt_flow = None, None

        all_gt_occupancy_list = [None for _ in range(num_dec_layers - 1)] + [gt_occupancy]
        all_occupancy_preds = [None for _ in range(num_dec_layers - 1)] + [occupancy_preds]
        all_gt_flow_list = [None for _ in range(num_dec_layers - 1)] + [gt_flow]
        all_flow_preds = [None for _ in range(num_dec_layers - 1)] + [flow_preds]

        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        loss_dict['loss_occupancy'] = 0
        loss_dict['loss_flow'] = 0
        loss_dict['lovasz_softmax'] = 0
        loss_dict['loss_sem_scal'] = 0
        loss_dict['loss_geo_scal'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores =  all_cls_scores[:, :,group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :,group_query_start:group_query_end, :]
            
            losses_cls, losses_bbox, losses_occupancy, losses_flow, \
            losses_lovasz_softmax, losses_sem_scal, losses_geo_scal = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_occupancy_preds,
                all_flow_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list,
                all_gt_occupancy_list,
                all_gt_flow_list)
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr
            # loss from other decoder layers
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1
            loss_dict['loss_occupancy'] = losses_occupancy[-1]
            loss_dict['loss_flow'] = losses_flow[-1]
            loss_dict['lovasz_softmax'] = losses_lovasz_softmax[-1]
            loss_dict['loss_sem_scal'] = losses_sem_scal[-1]
            loss_dict['loss_geo_scal'] = losses_geo_scal[-1]
        return loss_dict