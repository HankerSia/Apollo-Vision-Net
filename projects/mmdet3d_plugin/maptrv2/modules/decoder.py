import copy
import warnings

import torch
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence
from mmdet.models.utils.transformer import inverse_sigmoid


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRv2Decoder(TransformerLayerSequence):
    """MapTRv2 decoder with iterative reference refinement."""

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


@TRANSFORMER_LAYER.register_module()
class MapTRv2DecoupledDetrTransformerDecoderLayer(BaseTransformerLayer):
    """MapTRv2 decoupled decoder layer.

    The first self-attention mixes points inside each vector, while the second
    self-attention mixes vectors across the same point index.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        num_vec=50,
        num_pts_per_vec=20,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='LN'),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        assert len(operation_order) == 8
        assert set(operation_order) == {'self_attn', 'norm', 'cross_attn', 'ffn'}
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(
                f'Use same attn_mask in all attentions in {self.__class__.__name__}',
                stacklevel=2,
            )
        else:
            assert len(attn_masks) == self.num_attn

        num_vec = int(kwargs.get('num_vec', self.num_vec))
        num_pts_per_vec = int(kwargs.get('num_pts_per_vec', self.num_pts_per_vec))
        self_attn_mask = kwargs.get('self_attn_mask', None)

        for layer in self.operation_order:
            if layer == 'self_attn':
                n_pts, n_batch, n_dim = query.shape
                if attn_index == 0:
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=self_attn_mask,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)
                else:
                    query = (
                        query.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(1, 2)
                    )
                    query_pos = (
                        query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(1, 2)
                    )
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs,
                    )
                    query = (
                        query.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(0, 1)
                    )
                    query_pos = (
                        query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                        .flatten(0, 1)
                    )

                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
