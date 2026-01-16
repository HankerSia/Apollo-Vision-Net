# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)  # (num_points_in_pillar, H, W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # (num_points_in_pillar, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # (bs, num_points_in_pillar, H*W, 3)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):

        # Optional debug for smoke tests / NaN triage.
        # Enabled via config: model.debug_nan=True (already used by repo probes).
        try:
            debug_nan = bool(getattr(self, 'debug_nan', False))
        except Exception:
            debug_nan = False
        if debug_nan:
            try:
                _len = len(img_metas) if hasattr(img_metas, '__len__') else 'NA'
                print(f"[bevformer][meta] point_sampling: type(img_metas)={type(img_metas)} len={_len}")
                if isinstance(img_metas, list) and len(img_metas) > 0:
                    print(f"[bevformer][meta] point_sampling: type(img_metas[0])={type(img_metas[0])}")
                    if isinstance(img_metas[0], dict):
                        _keys = list(img_metas[0].keys())
                        print(f"[bevformer][meta] point_sampling: img_metas[0] keys={_keys}")
                        if 'lidar2img' in img_metas[0]:
                            import numpy as _np
                            _l2i0 = _np.asarray(img_metas[0]['lidar2img'])
                            print(f"[bevformer][meta] point_sampling: img_metas[0].lidar2img shape={_l2i0.shape} dtype={_l2i0.dtype}")
                    elif isinstance(img_metas[0], list):
                        print(f"[bevformer][meta] point_sampling: img_metas[0] is list len={len(img_metas[0])}")
                        if len(img_metas[0]) > 0 and isinstance(img_metas[0][0], dict):
                            _keys = list(img_metas[0][0].keys())
                            print(f"[bevformer][meta] point_sampling: img_metas[0][0] keys={_keys}")
                            if 'lidar2img' in img_metas[0][0]:
                                import numpy as _np
                                _l2i00 = _np.asarray(img_metas[0][0]['lidar2img'])
                                print(f"[bevformer][meta] point_sampling: img_metas[0][0].lidar2img shape={_l2i00.shape} dtype={_l2i00.dtype}")
            except Exception as e:
                print(f"[bevformer][meta] point_sampling debug failed: {e}")

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        # Robustify meta contract:
        # - expected: (B, num_cam, 4, 4)
        # - sometimes: (B, 4, 4) where each entry is already a stacked
        #   (num_cam,4,4) but got collapsed by numpy due to object dtype.
        if lidar2img.ndim == 3 and lidar2img.shape[-2:] == (4, 4):
            try:
                lidar2img = np.stack([np.asarray(x) for x in lidar2img], axis=0)
            except Exception:
                pass
        if debug_nan:
            try:
                print(f"[bevformer][meta] point_sampling: np.asarray(lidar2img) shape={lidar2img.shape} dtype={lidar2img.dtype}")
            except Exception:
                pass
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        if debug_nan:
            try:
                print(f"[bevformer][meta] point_sampling: lidar2img tensor shape={tuple(lidar2img.shape)} dtype={lidar2img.dtype} device={lidar2img.device}")
            except Exception:
                pass
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)
        # Smoke/debug robustness: if lidar2img arrived with an unexpected
        # number of cameras (often due to inconsistent meta shaping), infer
        # the camera dimension from total elements.
        if (lidar2img.dim() == 3 and lidar2img.numel() == B * 6 * 4 * 4):
            # (B,4,4) and each element is actually (6,4,4) but collapsed.
            pass
        if lidar2img.numel() != B * num_cam * 4 * 4:
            try:
                inferred = int(lidar2img.numel() // (B * 4 * 4))
                if inferred > 0 and (lidar2img.numel() == B * inferred * 4 * 4):
                    num_cam = inferred
            except Exception:
                pass

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # img_shape contract robustness:
        # Most pipelines provide `img_shape` as a list with length=num_cams,
        # where each entry is (H, W, C). Some minimal/smoke paths may miss it.
        # For point sampling normalization we only need image height/width.
        img_shape0 = None
        try:
            img_shape0 = img_metas[0].get('img_shape', None)
        except Exception:
            img_shape0 = None

        H_img = None
        W_img = None
        if isinstance(img_shape0, (list, tuple)) and len(img_shape0) > 0:
            # Prefer per-camera shape.
            try:
                H_img = int(img_shape0[0][0])
                W_img = int(img_shape0[0][1])
            except Exception:
                H_img, W_img = None, None
        elif isinstance(img_shape0, (list, tuple)) and len(img_shape0) >= 2 and isinstance(img_shape0[0], (int, np.integer)):
            # Sometimes stored as (H, W, C) without per-cam list.
            try:
                H_img = int(img_shape0[0])
                W_img = int(img_shape0[1])
            except Exception:
                H_img, W_img = None, None

        # As a last resort (e.g., smoke tests), derive from projected points.
        # Note: this doesn't recover the true sensor resolution; it just gives a
        # consistent normalization to keep the forward pass alive.
        if (H_img is None) or (W_img is None) or (H_img <= 0) or (W_img <= 0):
            if debug_nan:
                print("[bevformer][meta] point_sampling: missing img_shape; using fallback H=W=1 for normalization")
            H_img, W_img = 1, 1

        reference_points_cam[..., 0] /= float(W_img)
        reference_points_cam[..., 1] /= float(H_img)

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_z=None,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        # Normalize img_metas contract for point_sampling.
        # In this repo, some pipelines pass temporal metas as list[list[dict]]
        # (bs, len_queue). point_sampling expects list[dict] (bs) where each
        # dict has 'lidar2img' shaped (num_cam,4,4).
        img_metas_for_sampling = kwargs.get('img_metas', None)
        if isinstance(img_metas_for_sampling, list) and len(img_metas_for_sampling) > 0 and isinstance(img_metas_for_sampling[0], list):
            # Heuristic: treat as (bs, len_queue) and use the latest/current
            # frame metas for projection.
            try:
                img_metas_for_sampling = [m_list[-1] for m_list in img_metas_for_sampling]
            except Exception:
                pass

        # Optional: mirror model.debug_nan into encoder for meta debug prints.
        if hasattr(self, 'debug_nan') is False:
            try:
                self.debug_nan = bool(kwargs.get('debug_nan', False))
            except Exception:
                pass

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas_for_sampling)

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d  # .clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 6
        # assert set(operation_order) == set(
        #     ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query