_base_ = ['./bev_tiny_det_map_apollo.py']

# Single-variable ablation: remove centerline only.
# Keep MapTRv2 decoder, one-to-many branch, and aux-seg settings unchanged.

data_root = 'data/nuscenes/'
map_classes = ['divider', 'ped_crossing', 'boundary']
map_ann_file = 'data/nuscenes/nuscenes_map_anns_val_no_centerline.json'
map_train_info_file = data_root + 'nuscenes_map_infos_temporal_train.pkl'
map_val_info_file = data_root + 'nuscenes_map_infos_temporal_val.pkl'
map_auto_prepare = dict(
    enabled=True,
    root_path=data_root,
    out_dir=data_root,
    canbus='data',
    extra_tag='nuscenes',
    version='v1.0',
    max_sweeps=10,
    point_cloud_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
    splits=['trainval'],
    required_files=[map_train_info_file, map_val_info_file],
)

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
        ann_file=map_train_info_file,
        map_classes=map_classes,
        map_ann_file=map_ann_file,
    ),
    val=dict(
        type='CustomNuScenesDetMapV2Dataset',
        ann_file=map_val_info_file,
        map_classes=map_classes,
        map_ann_file=map_ann_file,
    ),
    test=dict(
        type='CustomNuScenesDetMapV2Dataset',
        ann_file=map_val_info_file,
        map_classes=map_classes,
        map_ann_file=map_ann_file,
    ),
)

evaluation = dict(
    interval=10,
    save_best='NuscMap_chamfer/mAP',
    rule='greater',
    map_metric=['chamfer', 'iou'],
)