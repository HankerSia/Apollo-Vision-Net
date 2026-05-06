_base_ = ['./bev_tiny_det_map_apollo.py']

map_classes = ['divider', 'ped_crossing', 'boundary', 'centerline']

map_num_vec_one2one = 50
map_num_vec_one2many = 300
map_num_pts = 20

model = dict(
    pts_bbox_head=dict(
        type='BEVFormerDetMapHeadApolloV2',
        map_num_classes=len(map_classes),
        map_num_vec_one2one=map_num_vec_one2one,
        map_num_vec_one2many=map_num_vec_one2many,
        map_k_one2many=6,
        map_lambda_one2many=1.0,
        num_map_vec=map_num_vec_one2one + map_num_vec_one2many,
        map_num_pts=map_num_pts,
        map_aux_seg=dict(
            use_aux_seg=True,
            bev_seg=True,
            pv_seg=True,
            seg_classes=1,
            loss_weight=1.0,
            pos_weight=2.0,
            radius=1,
            pv_loss_weight=1.0,
            pv_pos_weight=2.0,
            pv_radius=1,
        ),
        transformer=dict(
            map_decoder=dict(
                _delete_=True,
                type='MapTRv2Decoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    _delete_=True,
                    type='MapTRv2DecoupledDetrTransformerDecoderLayer',
                    num_vec=map_num_vec_one2one + map_num_vec_one2many,
                    num_pts_per_vec=map_num_pts,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
                ),
            ),
        ),
    ),
)

data = dict(
    train=dict(
        type='CustomNuScenesDetMapV2Dataset',
        map_classes=map_classes,
    ),
    val=dict(
        type='CustomNuScenesDetMapV2Dataset',
        map_classes=map_classes,
    ),
    test=dict(
        type='CustomNuScenesDetMapV2Dataset',
        map_classes=map_classes,
    ),
)

evaluation = dict(
    interval=10,
    map_metric=['chamfer', 'iou'],
)
