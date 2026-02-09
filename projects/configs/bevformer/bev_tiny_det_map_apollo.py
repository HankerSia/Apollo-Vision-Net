# tiny model ResNet50
# det + map (MapTR-aligned scaffold)

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/schedules/cyclic_20e.py',
    '../_base_/default_runtime.py'
]

# Smoke-test only: keep runtime short.
# Override the schedule's default runner (usually 20 epochs).
runner = dict(type='EpochBasedRunner', max_epochs=2)

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# det+map may not use all parameters every iteration (branch-conditional losses).
find_unused_parameters = True

# Workaround: this env's `torch.utils.tensorboard` import fails because
# `distutils.version` is missing. For smoke-train we only keep text logging.
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.2, 0.2, 8]
occupancy_size = [0.5, 0.5, 0.5]  # kept for compatibility of some code paths

img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=True,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3
# GroupMultiheadAttention requires `num_query % group == 0` during training.
# We keep `group=1` for both stages to ensure correctness.
group_detr = 1

model = dict(
    type='BEVFormer',
    # 为了本地 CPU smoke test（1-batch forward_train）先关闭 grid mask，
    # 否则 grid_mask 内部强制 .cuda() 会触发 cpu/cuda 混用。
    use_grid_mask=False,
    video_test_mode=True,
    # Det+Map: do NOT use occ gts.
    use_occ_gts=False,
    only_occ=False,
    only_det=False,
    pretrained=dict(img='ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth'),
    img_backbone=dict(
        type='DLA',
        levels=[1, 1, 1, 2, 2, 1],
        channels=[16, 32, 64, 128, 256, 512],
        out_features=['level3', 'level4', 'level5']),
    img_neck=dict(
        type='SECONDFPNV2',
        in_channels=[128, 256, 512],
        upsample_strides=[0.5, 1, 2],
        out_channels=[256, 256, 256]),
    pts_bbox_head=(
        # det+map scaffold (not yet det-evaluable)
        dict(
            type='BEVFormerDetMapHeadApollo',
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            num_classes=10,
            in_channels=_dim_,
            embed_dims=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                voxel_size=voxel_size,
                num_classes=10),
            point_cloud_range=point_cloud_range,
            occupancy_size=occupancy_size,
            transformer=dict(
                type='PerceptionTransformer',
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                encoder=dict(
                    type='BEVFormerEncoder',
                    num_layers=3,
                    pc_range=point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=_dim_,
                                num_levels=1,
                                attn_logits_clamp=20.0,
                                debug_attn_nan=False),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=point_cloud_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=_dim_,
                                    num_points=8,
                                    num_levels=_num_levels_,
                                    attn_logits_clamp=20.0,
                                    debug_attn_nan=False),
                            )
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm'))),
                det_decoder=dict(
                    type='DetectionTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='GroupMultiheadAttention',
                                group=group_detr,
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm'))),
                map_decoder=dict(
                    type='MapTRDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                        ],
                        feedforward_channels=_ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm'))),
            ),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0),
            enabling_det=True,
            enabling_map=True,
            real_h=100.0,
            real_w=100.0,
        )
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

# --- pipelines ---

dataset_type = 'CustomNuScenesDetMapDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline_det_map = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # Ensure LoadAnnotations3D (from mmdet3d) can append to bbox/label field lists.
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_attr_label=False),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    # Map GT is computed on-the-fly by the det_map dataset (MapTR-aligned).
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'gt_map_vecs_label', 'gt_map_vecs_pts_loc', 'img'])
]

train_pipeline = train_pipeline_det_map


test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

# Override dataset settings from the base dataset config.
data = dict(
    # For temporal BEVFormer, each dataset item already contains a queue of
    # frames stacked into `img` (see CustomNuScenesDataset.union2one). Keep
    # batch size 1 for minimal forward smoke tests.
    samples_per_gpu=1,
    workers_per_gpu=4,
    # Required by projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        use_occ_gts=False,
        test_mode=False,
        queue_length=queue_length,
        fixed_ptsnum_per_line=20,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        use_occ_gts=False,
        test_mode=True,
        queue_length=queue_length,
        fixed_ptsnum_per_line=20,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        use_occ_gts=False,
        test_mode=True,
        queue_length=queue_length,
        fixed_ptsnum_per_line=20,
        box_type_3d='LiDAR'),
)
