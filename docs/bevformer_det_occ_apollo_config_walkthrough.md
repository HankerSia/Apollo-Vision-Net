# BEVFormer Det+Occ Apollo Config 详细梳理

---

## 1. 数据 pipeline & GT 加载

**配置片段**
```python
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    ...,
    dict(type='LoadOccupancyGT'),
    ...
]
```

**参数流转**
- pipeline 由 config 传入 dataset 构造，`LoadOccupancyGT` 负责读取 `.npy` 稀疏体素标签，转为 dense grid。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/datasets/pipelines/loading.py
# 功能：将稀疏体素标签npy加载为dense grid，供occupancy分支loss使用
class LoadOccupancyGT(object):
    def __call__(self, results):
        occ = np.load(results['occ_gt_path'])  # (N, 2)
        dense_grid = np.full(grid_shape, fill_value=ignore_idx)
        dense_grid[occ[:, 0]] = occ[:, 1]
        results['occ_gts'] = dense_grid
        return results
```

**训练/推理调用链**
- 训练/推理入口：dataset -> pipeline -> batch['occ_gts']

---

## 2. 模型构建与参数注入

**配置片段**
```python
model = dict(
    type='BEVFormer',
    img_backbone=dict(type='DLA', ...),
    img_neck=dict(type='SECONDFPNV2', ...),
    pts_bbox_head=dict(
        type='BEVFormerOccupancyHeadApollo',
        group_detr=11,
        occupancy_classes=16,
        transformer=..., ...)
)
```

**参数流转**
- config 通过 `build_detector` 递归传递，注册到各自类的 `__init__`，每一级参数均可追溯。
    - `build_detector` (mmdet3d/apis/train.py) 负责递归构建模型各子模块。
    - `img_backbone`、`img_neck`、`pts_bbox_head` 分别传入对应 build_xxx 工厂函数，最终注册到各自类。

**BEVFormer 主体结构与成员分析**
```python
# 文件: projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
class BEVFormer(MVXTwoStageDetector):
    def __init__(self, ..., img_backbone=None, img_neck=None, pts_bbox_head=None, ...):
        super().__init__(...)
        self.img_backbone = build_backbone(img_backbone)
        self.img_neck = build_neck(img_neck)
        self.pts_bbox_head = build_head(pts_bbox_head)
        # 其他成员: pts_bbox_head, img_backbone, img_neck, ...

    def extract_img_feat(self, img, img_metas, len_queue=None):
        # 图像特征提取主流程
        img = img.view(-1, C, H, W)
        feats = self.img_backbone(img)
        feats = self.img_neck(feats)
        ...
        return img_feats

    def forward_pts_train(self, ...):
        # 训练主流程
        img_feats = self.extract_img_feat(img, img_metas)
        outs = self.pts_bbox_head(img_feats, img_metas, ...)
        loss_inputs = ... # 组装 loss 输入
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, img, img_metas, ...):
        # 推理主流程
        img_feats = self.extract_img_feat(img, img_metas)
        outs = self.pts_bbox_head(img_feats, img_metas, ...)
        results = self.pts_bbox_head.get_bboxes(*outs, img_metas, ...)
        return results
```


### 2.1 继承关系分析与MVXTwoStageDetector代码解析

BEVFormer 继承自 MVXTwoStageDetector，后者又继承自 BaseDetector，形成如下继承链：

```
BaseDetector (mmdet3d/core)
  └── MVXTwoStageDetector (projects/mmdet3d_plugin/core/)
        └── BEVFormer (projects/mmdet3d_plugin/bevformer/detectors/bevformer.py)
```

**MVXTwoStageDetector 主要作用**
- 作为多模态（Multi-View/Modal eXtension）检测器基类，支持多传感器输入（如图像、点云等），并定义了常用的 forward、extract_feat 等接口。
- 负责管理 backbone、neck、head 等子模块的注册与调用。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/core/mvx_two_stage.py
# 功能：多模态两阶段检测器基类，管理backbone/neck/head，定义特征提取与前向接口
class MVXTwoStageDetector(BaseDetector):
    def __init__(self, img_backbone=None, img_neck=None, pts_backbone=None, pts_neck=None, pts_bbox_head=None, img_roi_head=None, ...):
        super().__init__()
        if img_backbone is not None:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        if pts_backbone is not None:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)
        if pts_bbox_head is not None:
            self.pts_bbox_head = build_head(pts_bbox_head)
        ...

    def extract_img_feat(self, img, img_metas):
        # 图像特征提取接口，供子类重载
        if self.with_img_backbone:
            x = self.img_backbone(img)
            if self.with_img_neck:
                x = self.img_neck(x)
            return x
        else:
            return None
    ...
```

**小结**
- BEVFormer 通过继承 MVXTwoStageDetector，获得了多模态特征提取、backbone/neck/head 注册与 forward 接口的通用实现。
BEVFormer(MVXTwoStageDetector) 主要重载了 MVXTwoStageDetector(Base3DDetector) 的以下方法：

extract_img_feat(self, img, img_metas, len_queue=None)
forward_pts_train(self, ...)
simple_test(self, img, img_metas, ...)
实际调用时：

如果 BEVFormer 实现了 extract_img_feat、forward_pts_train、simple_test，则优先调用 BEVFormer 的同名方法（即重载）。
其他未重载的方法（如 MVXTwoStageDetector 的 extract_pts_feat、forward_train、forward_img_train 等）则直接调用父类实现。


**训练/推理主调用链（详细）**


**训练入口（主流程调用链）：**
1. `build_detector(config)`
2. -> `BEVFormer.__init__`（自动调用父类 MVXTwoStageDetector.__init__ 完成参数注册）
3. -> `forward_pts_train`（BEVFormer重载，主训练流程）
4. -> `extract_img_feat`（BEVFormer重载，优先调用子类实现）
5. -> `img_backbone` -> `img_neck`（特征提取）
6. -> `pts_bbox_head.forward`（head前向）
7. -> `pts_bbox_head.loss`（loss计算）

**推理入口（主流程调用链）：**
1. `build_detector(config)`
2. -> `BEVFormer.__init__`（自动调用父类 MVXTwoStageDetector.__init__ 完成参数注册）
3. -> `simple_test`（BEVFormer重载，主推理流程）
4. -> `extract_img_feat`（BEVFormer重载，优先调用子类实现）
5. -> `img_backbone` -> `img_neck`（特征提取）
6. -> `pts_bbox_head.forward`（head前向）
7. -> `pts_bbox_head.get_bboxes`（结果解码）

> 说明：BEVFormer 会重载 MVXTwoStageDetector 的 extract_img_feat、forward_pts_train、simple_test 等方法，实际调用时优先用 BEVFormer 的实现，未重载的则用父类逻辑。

**参数流转举例**
- config['img_backbone'] -> `build_backbone` -> `DLA.__init__`
- config['img_neck'] -> `build_neck` -> `SECONDFPNV2.__init__`
- config['pts_bbox_head'] -> `build_head` -> `BEVFormerOccupancyHeadApollo.__init__`

**小结**
- BEVFormer 作为主 detector，聚合 backbone、neck、head，所有参数均由 config 递归注入。
- 训练/推理主流程清晰，便于追踪每一级参数和调用链。

---


## 3. 主干特征提取（分模型细化）

### 3.1 图像主干网络（Backbone: DLA）

**配置片段**
```python
img_backbone=dict(
    type='DLA',
    levels=[1, 1, 1, 2, 2, 1],
    channels=[16, 32, 64, 128, 256, 512],
    out_features=['level3', 'level4', 'level5']
)
```

**参数流转**
- config 里的 img_backbone 字典传入 `build_backbone`，注册到 DLA 类。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/models/backbones/dla.py
# 功能：DLA主干网络，提取多尺度图像特征
class DLA(BaseModule):
    def __init__(self, levels, channels, out_features, ...):
        # 构建多层 DLA 主干
        ...
    def forward(self, x):
        # 输入: x [B*N, C, H, W]
        # 输出: 多层特征
        ...
```

**训练/推理调用链**
- 训练/推理：`BEVFormer.extract_img_feat` -> `self.img_backbone(img)`

---

### 3.2 特征融合网络（Neck: SECONDFPNV2）

**配置片段**
```python
img_neck=dict(
    type='SECONDFPNV2',
    in_channels=[128, 256, 512],
    upsample_strides=[0.5, 1, 2],
    out_channels=[256, 256, 256]
)
```

**参数流转**
- config 里的 img_neck 字典传入 `build_neck`，注册到 SECONDFPNV2 类。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/models/necks/secondfpnv2.py
# 功能：FPN结构neck，融合多尺度特征
class SECONDFPNV2(BaseModule):
    def __init__(self, in_channels, upsample_strides, out_channels, ...):
        # 构建多层 FPN 融合
        ...
    def forward(self, feats):
        # 输入: 多层 backbone 特征
        # 输出: 融合后的特征
        ...
```

**训练/推理调用链**
- 训练/推理：`BEVFormer.extract_img_feat` -> `self.img_neck(feats)`

---

### 3.3 主干特征提取总流程

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/bevformer/detectors/bevformer.py
# 功能：提取图像特征，串联backbone和neck
def extract_img_feat(self, img, img_metas, len_queue=None):
    img = img.view(-1, C, H, W)
    feats = self.img_backbone(img)  # DLA
    feats = self.img_neck(feats)    # SECONDFPNV2
    ...
    return img_feats
```

**训练/推理调用链**
- 训练：`BEVFormer.forward_pts_train` -> `extract_img_feat` -> DLA -> SECONDFPNV2
- 推理：`BEVFormer.simple_test` -> `extract_img_feat` -> DLA -> SECONDFPNV2

---

## 4. Transformer结构（Encoder/Decoder详细）
**主结构与参数流转**

**配置片段**
```python
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
                dict(type='TemporalSelfAttention', embed_dims=_dim_, num_levels=1),
                dict(type='SpatialCrossAttention', pc_range=point_cloud_range,
                     deformable_attention=dict(type='MSDeformableAttention3D', embed_dims=_dim_, num_points=8, num_levels=_num_levels_),
                     embed_dims=_dim_)
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    ),
    decoder=dict(
        type='DetectionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='GroupMultiheadAttention', group=group_detr, embed_dims=_dim_, num_heads=8, dropout=0.1),
                dict(type='CustomMSDeformableAttention', embed_dims=_dim_, num_levels=1)
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        )
    )
)
```

**参数流转与结构说明**
- config['transformer'] 传入 `build_transformer`，注册到 PerceptionTransformer。
- PerceptionTransformer 负责聚合 encoder/decoder，参数递归传递到 BEVFormerEncoder/DetectionTransformerDecoder 及其 transformerlayers。
- encoder/decoder 的 transformerlayers 支持多层堆叠，operation_order 控制每层注意力/归一化/FFN顺序。
- attn_cfgs 支持多种注意力类型（TemporalSelfAttention、SpatialCrossAttention、GroupMultiheadAttention、CustomMSDeformableAttention）。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/bevformer/modules/transformer.py
# 功能：Transformer主控，聚合encoder/decoder，完成BEV特征聚合与目标解码
class PerceptionTransformer(BaseModule):
    def __init__(self, encoder=None, decoder=None, embed_dims=256, ...):
        super().__init__()
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder) if decoder is not None else None
        self.embed_dims = embed_dims
        ...
    def forward(self, mlvl_feats, bev_query, object_query_embeds, bev_h, bev_w, ...):
        # 1. encoder: BEV特征聚合
        # 2. decoder: 检测/分割query解码
        ...
        return ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/encoder.py
# 功能：堆叠BEVFormerLayer，完成时序/空间BEV特征编码
class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(self, num_layers=3, transformerlayers=None, ...):
        super().__init__(transformerlayers, num_layers, ...)
        ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/encoder.py
# 功能：单层BEVFormer编码器，支持时序/空间注意力与FFN
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    def __init__(self, attn_cfgs, feedforward_channels, ...):
        # attn_cfgs: TemporalSelfAttention, SpatialCrossAttention
        # operation_order: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/decoder.py
# 功能：堆叠DetrTransformerDecoderLayer，完成目标/occupancy解码
class DetectionTransformerDecoder(TransformerLayerSequence):
    def __init__(self, num_layers=6, transformerlayers=None, ...):
        super().__init__(transformerlayers, num_layers, ...)
        ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/decoder.py
# 功能：单层decoder，支持分组多头注意力与可变形注意力
class DetrTransformerDecoderLayer(BaseModule):
    def __init__(self, attn_cfgs, feedforward_channels, ...):
        # attn_cfgs: GroupMultiheadAttention, CustomMSDeformableAttention
        ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/group_attention.py
# 功能：分组多头注意力，提升目标区分能力
class GroupMultiheadAttention(BaseModule):
    def __init__(self, embed_dims, num_heads, group=11, ...):
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, ...)
        self.group = group
    def forward(self, query, ...):
        # 按 group 分组处理 query
        ...

# 文件: projects/mmdet3d_plugin/bevformer/modules/decoder.py
# 功能：自定义可变形注意力，提升空间建模能力
class CustomMSDeformableAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, ...):
        # 可变形注意力实现
        ...
```

**结构与调用链小结**
- PerceptionTransformer 作为 transformer 总控，聚合 encoder/decoder。
- encoder（BEVFormerEncoder）堆叠 BEVFormerLayer，支持时序/空间注意力。
- decoder（DetectionTransformerDecoder）堆叠 DetrTransformerDecoderLayer，支持分组多头注意力与可变形注意力。
- 所有参数均由 config 递归注入，结构高度可定制。
- 训练/推理主链：`pts_bbox_head.forward` -> `self.transformer(...)` -> `encoder.forward` -> `decoder.forward`


## 5. 检测/分割分支（前向与继承链分析）

**配置片段**
```python
pts_bbox_head=dict(
    type='BEVFormerOccupancyHeadApollo',
    group_detr=11,
    occupancy_classes=16,
    occ_dims=128,
    num_classes=10,
    code_size=8,
    code_weights=[1.0, ...],
    transformer=...,
    loss_cls=dict(type='FocalLoss', ...),
    loss_bbox=dict(type='L1Loss', ...),
    loss_occupancy=dict(type='FocalLoss', ...),
    ...
)
```

**参数流转**
- group_detr、occupancy_classes、transformer、loss_cls、loss_bbox、loss_occupancy 等传入 head。

**关键代码（前向主流程）**
```python
# 文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py
# 功能：head前向，组装query并调用transformer，输出检测/occupancy结果
def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
    ...
    outputs = self.transformer(
        mlvl_feats,
        bev_queries,
        object_query_embeds,
        self.bev_h,
        self.bev_w,
        grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        bev_pos=bev_pos,
        reg_branches=self.reg_branches if self.with_box_refine else None,
        cls_branches=self.cls_branches if self.as_two_stage else None,
        img_metas=img_metas,
        prev_bev=prev_bev,
        return_intermediate=True if self.occ_tsa else False
    )
    ...
```

**训练/推理调用链**
- 训练：`BEVFormer.forward_pts_train` -> `pts_bbox_head.forward`
- 推理：`BEVFormer.simple_test` -> `pts_bbox_head.forward`
- 训练/推理：`pts_bbox_head.forward` -> `self.transformer(...)` -> `encoder.forward` -> `decoder.forward`

### 5.1 继承链结构

1. **BEVFormerOccupancyHeadApollo**  
   文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py  
   继承自：BEVFormerOccupancyHead

2. **BEVFormerOccupancyHead**  
   文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head.py  
   继承自：DETRHead

3. **DETRHead**  
   文件: mmdet/models/dense_heads/detr_head.py  
   继承自：AnchorFreeHead（未展开，属于 mmdet）

### 5.2 参数传递与实现分析

#### 1. BEVFormerOccupancyHeadApollo

**实现片段**
```python
# 文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py
class BEVFormerOccupancyHeadApollo(BEVFormerOccupancyHead):
    def __init__(self, *args, group_detr=1, occ_tsa=None, positional_encoding_occ=None, balance_cls_weight=False, loss_lovasz=None, loss_affinity=None, **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']
        ...
        super().__init__(*args, **kwargs)
        ...
```
- 参数传递：config 里的 group_detr、num_query、loss_occupancy、occupancy_classes 等通过 kwargs 传递给父类。
- 额外处理：对 num_query 做分组扩展，对 loss_occupancy 类型做映射。

#### 2. BEVFormerOccupancyHead

**实现片段**
```python
# 文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head.py
class BEVFormerOccupancyHead(DETRHead):
    def __init__(self, *args, with_box_refine=False, as_two_stage=False, transformer=None, bbox_coder=None, num_cls_fcs=2, code_weights=None, bev_h=30, bev_w=30, occupancy_size=[0.5,0.5,0.5], point_cloud_range=[...], loss_occupancy=None, occ_dims=16, occupancy_classes=1, only_occ=False, ...):
        ...
        super().__init__(*args, transformer=transformer, **kwargs)
        ...
```
- 参数传递：config 里的 transformer、loss_occupancy、occupancy_classes、occ_dims、code_weights 等通过 kwargs 继续传递给 DETRHead。
- 额外处理：初始化分支、位置编码、体素参数等。

#### 3. DETRHead

**实现片段**
```python
# 文件: mmdet/models/dense_heads/detr_head.py
class DETRHead(AnchorFreeHead):
    def __init__(self, num_classes, in_channels, num_query=100, num_reg_fcs=2, transformer=None, sync_cls_avg_factor=False, positional_encoding=dict(...), loss_cls=dict(...), loss_bbox=dict(...), loss_iou=dict(...), train_cfg=dict(...), test_cfg=dict(...), init_cfg=None, **kwargs):
        ...
        self.transformer = build_transformer(transformer)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        ...
```
- 参数传递：transformer、loss_cls、loss_bbox、loss_iou、num_query、num_classes、in_channels 等全部由 config 或父类传递。
- 额外处理：初始化 transformer、loss、位置编码、分支等。

### 5.3 关键参数流转举例

- `group_detr`（apollo.py）→ 扩展 `num_query` → 传递给 occupancy_head.py
- `transformer`（config/occupancy_head.py）→ 传递给 DETRHead → build_transformer
- `loss_occupancy`（config/apollo.py）→ 传递给 occupancy_head.py → 传递给 DETRHead
- `occupancy_classes`、`occ_dims`、`code_weights` 等参数层层传递，最终用于分支初始化和 loss 计算。

### 5.4 典型 forward/loss 调用链

- `BEVFormerOccupancyHeadApollo.forward`（apollo.py）
  - 组装 query、调用 transformer、分割/检测分支
- `BEVFormerOccupancyHead.forward`（occupancy_head.py）
  - 组装分支、调用 DETRHead.forward
- `DETRHead.forward`（detr_head.py）
  - 处理 transformer、分支、loss

### 5.5 总结

- 每一层都通过 super().__init__ 传递参数，config 里的所有关键参数（如 transformer、loss、分支结构、类别数、体素参数等）都能层层流转到最底层实现。
- 继承链清晰，便于扩展和定制分支行为。

---

def upsample_occ(self, bev_for_occ, bs, seq_len):



## 6. Loss 计算

**配置片段**
```python
pts_bbox_head=dict(
    loss_cls=dict(type='FocalLoss', ...),
    loss_bbox=dict(type='L1Loss', ...),
    loss_occupancy=dict(type='FocalLoss', ...),
    loss_affinity=dict(type='affinity_loss', ...),
    ...
)
```

**参数流转**
- loss_cls/loss_bbox/loss_occupancy/loss_affinity 等传入 head。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py
# 功能：head loss计算，分组处理各类loss
def loss(self, gt_bboxes_list, gt_labels_list, ..., preds_dicts, ...):
    for group_index in range(self.group_detr):
        group_cls_scores =  all_cls_scores[:, :,group_query_start:group_query_end, :]
        group_bbox_preds = all_bbox_preds[:, :,group_query_start:group_query_end, :]
        losses_cls, losses_bbox, losses_occupancy, ... = multi_apply(
            self.loss_single, group_cls_scores, group_bbox_preds, ...)
        loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
        ...
```

**训练/推理调用链**
- 训练：`pts_bbox_head.loss` -> `loss_single` -> 各类 loss
- 推理：只用 forward，不计算 loss

---

## 7. 位置编码

**配置片段**
```python
positional_encoding=dict(
    type='LearnedPositionalEncoding',
    num_feats=_pos_dim_,
    row_num_embed=bev_h_,
    col_num_embed=bev_w_,
)
```

**参数流转**
- num_feats、row_num_embed、col_num_embed 传入 positional_encoding。

**关键代码**
```python
# 文件: mmdet/models/utils/positional_encoding.py
# 功能：可学习位置编码，生成BEV空间位置特征
class LearnedPositionalEncoding(BaseModule):
    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, ...):
        ...
    def forward(self, mask):
        # 输出 shape: [bs, num_feats*2, h, w]
```

**训练/推理调用链**
- 训练/推理：`pts_bbox_head.forward` -> positional_encoding

---

## 8. 体素 GT 还原

**配置片段**
- pipeline 里 LoadOccupancyGT 输出 occ_gts，loss 里转 dense grid。

**关键代码**
```python
# 文件: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py
# 功能：将稀疏体素GT转为dense grid，便于loss计算
if occ_gts:
    gt_occupancy = (torch.ones((bs*seq_len, self.voxel_num), dtype=torch.long)*self.occupancy_classes).to(device)
    for sample_id in range(len(temporal_gt_bboxes_list)):
        for frame_id in range(seq_len):
            occ_gt = occ_gts[sample_id][frame_id].long()
            gt_occupancy[sample_id*seq_len+frame_id][[occ_gt[:, 0]]] = occ_gt[:, 1]
```

**训练/推理调用链**
- 训练：`pts_bbox_head.loss` -> 稀疏体素转 dense grid
- 推理：只用 forward，不处理 occ_gts

---

## 9. 典型训练正向主流程（伪代码+配置映射）

```python
# 1. 数据 pipeline
img, occ_gts = pipeline(sample)  # config: train_pipeline

# 2. 特征提取
img_feats = model.extract_img_feat(img, img_metas)  # config: img_backbone, img_neck

# 3. Head 前向
outputs = model.pts_bbox_head.forward(img_feats, img_metas, ...)  # config: pts_bbox_head, transformer, group_detr, occupancy_classes

# 4. Loss 计算
loss_dict = model.pts_bbox_head.loss(
    gt_bboxes_list, gt_labels_list, ..., occ_gts, ..., outputs, ...)  # config: loss_cls, loss_bbox, loss_occupancy, loss_affinity
```

**训练调用链**
- dataset/pipeline -> BEVFormer.forward_pts_train -> extract_img_feat -> pts_bbox_head.forward -> transformer.encoder/decoder -> 检测/分割分支 -> pts_bbox_head.loss

**推理调用链**
- dataset/pipeline -> BEVFormer.simple_test -> extract_img_feat -> pts_bbox_head.forward -> transformer.encoder/decoder -> 检测/分割分支

---
