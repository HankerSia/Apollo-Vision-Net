# tiny model ResNet50
# det + map (MapTR-aligned scaffold)

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

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
    pts_bbox_head=dict(
        type='BEVFormerDetMapHeadApollo',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=_dim_,
        embed_dims=_dim_,
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
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # Decoder left out for the scaffold head.
            decoder=None,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        enabling_det=True,
        enabling_map=True,
        real_h=100.0,
        real_w=100.0,
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
# For now, keep a minimal pipeline to validate head registration.
# Map GT loading will be added in the next milestone.

dataset_type = 'CustomNuScenesDetMapDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
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
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
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
        test_mode=True,
        queue_length=queue_length,
        fixed_ptsnum_per_line=20,
        box_type_3d='LiDAR'),
)
