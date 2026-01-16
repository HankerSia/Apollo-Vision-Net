# Apollo-Vision-Net（BEVFormer det+occ）源码导读与 Shape 实测闭环

本文面向“想把工程吃透”的读者：不止解释 $shape$，而是把 **配置 → 数据 → 源码 → 实测输出** 串成一条可复用的排障/理解路径。

本文基于 `projects/configs/bevformer/bev_tiny_det_occ_apollo.py`，并使用 nuScenes `v1.0-mini` 跑 `tools/debug_shapes_v1mini.py` 得到的真实打印结果。

---

## 0. 先给结论：你应该记住的 I/O “契约”（带实测）

为了让后文不在细节里迷路，先把关键张量的**入口/出口契约**列出来（均为推理 `forward_test`、batch=1）。

### 0.1 输入侧（数据管线输出给模型的关键 meta）

来自 `img_metas`（每个 batch 里有 6 个相机视角）：

```text
ori_shape: [(450, 800, 3) x6]
img_shape: [(480, 800, 3) x6]
pad_shape: [(480, 800, 3) x6]
scale_factor: 1.0
```

### 0.2 Backbone+Neck 后（送入 head/transformer 的多相机特征）

进入 `pts_bbox_head` 的第 1 个 level（本 config 只有 1 个 level）：

```text
mlvl_feats[0] = (B=1, Ncam=6, C=256, H=28, W=48)
```

### 0.3 Transformer/Decoder（用于 det 的 BEV 与 query 输出）

head 内部 `outs`（由 hook 在 `pts_bbox_head` forward() 中捕获）：

```text
outs.bev_embed      = (bev_h*bev_w=2500, bs=1, embed_dims=256)
outs.all_cls_scores = (num_layers=6, bs=1, num_query=900, num_classes=10)
outs.all_bbox_preds = (num_layers=6, bs=1, num_query=900, code_size=8)
```

### 0.4 Occupancy（体素 logits 与后处理输出）

```text
outs.occupancy_preds (logits) = (bs=1, Nvoxel=640000, occ_classes=16)

final occ_results['occupancy_preds'] (postproc) = (Nvoxel=640000, 2)
```

> 这里 “16 vs 2” 是一个非常常见的误区：前者是训练/forward 阶段的**分类 logits 维度**，后者是工程里 `get_occupancy_prediction()` 的**离散编码输出格式**。

---

## 1. 工程结构与调用入口：从 `Detector.simple_test()` 开始

把整体调用链固定下来，后面每个 shape 才有归属。

### 1.1 Detector：图像特征提取与形状恢复

源码位置：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

你关心的两个函数：

- `extract_img_feat()`：
  - 把输入 `img` 从 `(B, Ncam, 3, H, W)` reshape 成 `(B*Ncam, 3, H, W)` 喂给 backbone/neck
  - backbone/neck 跑完后再 reshape 回 `(B, Ncam, C, H', W')`
- `simple_test()` / `simple_test_pts()`：
  - 调用 `pts_bbox_head` 产出 det 结果
  - 同时（本工程里）产出 occ 的后处理结果

> 建议你阅读时抓住一个核心：**图像特征在 detector 内部会经历“BN 合并 → 网络 → BN 拆开”的形状往返**。这也是 `mlvl_feats` 为什么经常看到 `B, Ncam` 两个维度并存。

### 1.2 代码对照：`extract_img_feat()` 的 reshape 往返

下面代码摘自 `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py::extract_img_feat()`（省略无关分支），它决定了你后面看到的 `(B, Ncam, C, H', W')` 到底怎么来的：

```python
elif img.dim() == 5 and img.size(0) > 1:
    B, N, C, H, W = img.size()
    img = img.reshape(B * N, C, H, W)

img_feats = self.img_backbone(img)
...
if self.with_img_neck:
    img_feats = self.img_neck(img_feats)

for img_feat in img_feats:
    BN, C, H, W = img_feat.size()
    img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
return img_feats_reshaped
```

逐段解释（把“思路”讲透）：

1. **合并 batch 与相机维**：`img.reshape(B*N, C, H, W)`
   - 这样 backbone/neck 就可以把“每张图”当成一个普通 batch 去跑。
2. **backbone/neck 不知道 num_cam 的存在**：它们只看到 `BN=B*N`。
3. **网络跑完再拆回相机维**：`view(B, BN/B, C, H', W')`
   - 所以当你实测到 `mlvl_feats[0]=(1,6,256,28,48)` 时，本质上就是：
  - `B=1`
  - `BN/B = 6`（相机数）
  - `C=256, H'=28, W'=48`（neck 最终输出）

---

## 2. 数据管线：`ori_shape/img_shape/pad_shape` 到底分别表示什么？

这一段的目标不是复述概念，而是把“为什么本例是 450→480”讲清楚。

### 2.1 含义定义（mmcv/mmdet 通用约定）

- `ori_shape`：原图读入尺寸
- `img_shape`：完成 resize/crop 等几何增强后的尺寸（pad 前）
- `pad_shape`：完成 padding 后的最终输入尺寸
- `scale_factor`：resize 缩放比例（仅 pad 时通常就是 1.0）

### 2.2 本工程 config 中谁会改变这些字段？

配置位置：`projects/configs/bevformer/bev_tiny_det_occ_apollo.py` 的 `test_pipeline`

其中与形状相关的典型算子包括：

- `RandomScaleImageMultiViewImage`：会改变 `img_shape` 与 `scale_factor`
- `PadMultiViewImage(size_divisor=32)`：会改变 `pad_shape`（把 H/W pad 到能被 32 整除）

### 2.3 v1.0-mini 实测解读

本次实测：`scale_factor=1.0` 且 `img_shape==pad_shape==(480,800,3)`，因此可以判定：

- 450→480 主要来自 padding（至少在这帧样本上）
- 并非“网络 stride 除出来”的结果

---

## 3. DLA Backbone：取哪几层、为什么输出是这些

### 3.1 配置与源码位置

config（摘要）：

```python
img_backbone=dict(
  type='DLA',
  levels=[1, 1, 1, 2, 2, 1],
  channels=[16, 32, 64, 128, 256, 512],
  out_features=['level3', 'level4', 'level5'])
```

源码位置：`projects/mmdet3d_plugin/models/backbones/dla.py`

### 3.2 读源码要看什么

建议用“输入/输出契约”的方式读：

- 输入：来自 detector reshape 后的 `(B*Ncam, 3, H, W)`
- 输出：包含多个 stage / level 的特征（本工程选择 `level3/4/5`）

为什么通常用 `level3/4/5`：

- 更低层（`level1/2`）空间分辨率大但语义弱、成本高
- 更高层语义强但分辨率低
- 通过 neck 做对齐融合，再统一喂给 transformer

---

## 4. SECONDFPNV2 Neck：多尺度如何对齐、为什么最终是 C=256

### 4.1 配置与源码位置

config（摘要）：

```python
img_neck=dict(
  type='SECONDFPNV2',
  in_channels=[128, 256, 512],
  upsample_strides=[0.5, 1, 2],
  out_channels=[256, 256, 256])
```

源码位置：`projects/mmdet3d_plugin/models/necks/second_fpnv2.py`

### 4.2 forward 的真实语义（按代码分支解释）

`SECONDFPNV2.forward()` 的核心逻辑可以归纳为：

1. 对每个输入 level 建一个 `deblock`：可能是 ConvTranspose2d（上采样）、Conv2d（下采样）或保持
2. 把对齐后的多尺度特征在 channel 维拼接：`torch.cat([...], dim=1)`
3. 通过 `last_conv` 把拼接后的通道压回 256
4. 返回 list（通常长度为 1），形成 `mlvl_feats`

`upsample_strides=[0.5, 1, 2]` 的关键点：

- `2`：上采样 ×2（transpose conv）
- `1`：不变
- `0.5`：代码里会走“下采样”分支（把 1/0.5 取整为 2）

这解释了为什么它不是“把所有尺度都拉到最大分辨率”的传统 FPN，而是使用一套目标尺度对齐策略。

### 4.3 代码对照：`SECONDFPNV2.forward()` 的 4 行主逻辑

源码：`projects/mmdet3d_plugin/models/necks/second_fpnv2.py::SECONDFPNV2.forward()`

```python
ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

if len(ups) > 1:
  out = torch.cat(ups, dim=1)
else:
  out = ups[0]
out = self.last_conv(out)
return [out]
```

读这段代码要抓住两个关键点：

1. `deblocks` 把 `level3/4/5` 的多尺度特征**对齐到同一个 (H, W)**（通过上/下采样分支）。
2. `torch.cat(..., dim=1)` 发生在 **channel 维**：多尺度在通道上拼接。
3. `last_conv` 再把 `sum(out_channels)` 压回到 **256 通道**。
4. `return [out]` 解释了为什么你最终拿到的 `mlvl_feats` 是一个 **list，长度为 1**（即 `_num_levels_=1` 的直观体现）。

---

## 5. `mlvl_feats[0] = (1, 6, 256, 28, 48)`：它从哪里来？怎么定位？

### 5.1 这个张量出现在调用链的哪个位置

`mlvl_feats` 是 detector 完成 backbone+neck 后送入 head 的特征列表。

关联源码：

- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`：`extract_img_feat()` 产出 `img_feats` / `mlvl_feats`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`：`forward()` 入参接收 `mlvl_feats`

### 5.2 为什么不建议“用 stride 除法猜尺寸”

你很容易按 $480/16=30,\;800/16=50$ 去猜特征图大小，但工程里实际是 `28×48`。

更可靠的解释是：

- backbone 各 stage 的 stride/pad 组合
- neck 上下采样的对齐策略（及可能的裁剪/对齐到偶数等）
- 以及不同实现对边界的处理差异

因此本文采用的策略是：**不猜，直接在 head 入口处 hook 打印**。

### 5.3 实测

```text
mlvl_feats[0] = (1, 6, 256, 28, 48)
```

维度含义：

- `B=1`：batch
- `Ncam=6`：6 相机
- `C=256`：neck 输出通道（`last_conv` 压到 256）
- `H=28, W=48`：进入 transformer 之前的特征空间尺寸

---

## 6. PerceptionTransformer：从多相机 features 到 BEV embedding

源码位置：`projects/mmdet3d_plugin/bevformer/modules/transformer.py`

这一段的目标是解释两件事：

1. `mlvl_feats` 如何变成变形注意力需要的 `feat_flatten/spatial_shapes`
2. BEV queries（50×50=2500）如何生成并输出 `bev_embed`

### 6.1 `mlvl_feats` 的 flatten 与维度重排（按代码语句对照）

进入 `get_bev_features()` 时，单个 level 的 shape 是：

```text
(bs, num_cam, c, h, w)
```

代码里会做类似（概念级别）操作：

- `feat.flatten(3)`：把 `(h,w)` 展平成 `h*w`
- `permute(...)`：把相机维提到最前以便后续对每个 cam 做 attention

你最终会得到 encoder 需要的 `feat_flatten`，并同步得到：

- `spatial_shapes=[(h,w), ...]`
- `level_start_index`

这些变量为 deformable attention 提供“每个 level 的空间布局”。

### 6.1.1 代码对照：`get_bev_features()` 里最关键的 flatten/permute

源码：`projects/mmdet3d_plugin/bevformer/modules/transformer.py::PerceptionTransformer.get_bev_features()`

```python
for lvl, feat in enumerate(mlvl_feats):
  bs, num_cam, c, h, w = feat.shape
  spatial_shape = (h, w)
  feat = feat.flatten(3).permute(1, 0, 3, 2)
  spatial_shapes.append(spatial_shape)
  feat_flatten.append(feat)

feat_flatten = torch.cat(feat_flatten, 2)
...
feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
```

逐段解释（对应你关心的 shape 链路）：

1. `feat.shape = (bs, num_cam, c, h, w)`
   - 对应你实测到的 `mlvl_feats[0]=(1,6,256,28,48)`。
2. `feat.flatten(3)`：把 `(h,w)` 展平为 `h*w`
   - shape 变为 `(bs, num_cam, c, h*w)`。
3. `.permute(1, 0, 3, 2)`：把相机维 `num_cam` 放到最前面
   - shape 变为 `(num_cam, bs, h*w, c)`。
4. 最后一行 `feat_flatten.permute(0, 2, 1, 3)`
   - shape 变为 `(num_cam, h*w, bs, c)`，也就是注释写的 `(num_cam, H*W, bs, embed_dims)`。

> 常见误区：
> - 很多人把 `h*w` 当作“batch 维”，其实它只是把空间维展平，后面 attention 会把它当作序列长度。
> - `num_cam` 被放到最前面，是为了跨视角注意力里更自然地建模相机维（实现上也更方便）。

### 6.2 BEV queries 与 `bev_embed` 的 I/O

BEV 网格来自 config：`bev_h=50, bev_w=50`，因此 query 数是：

$$N_{bev}=50\times 50=2500$$

工程内约定（也是 BEVFormer 常见约定）：

```text
bev_queries: (N_bev, bs, embed_dims)
bev_embed:   (N_bev, bs, embed_dims)
```

### 6.2.1 代码对照：BEV queries 如何扩 batch

同一函数里，BEV query 的 batch 扩展非常直接：

```python
bs = mlvl_feats[0].size(0)
bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)  # (num, bs, embed_dims)
```

对应实测 `outs.bev_embed=(2500,1,256)`，其中 `2500=bev_h*bev_w`。

### 6.3 v1.0-mini 实测

```text
outs.bev_embed = (2500, 1, 256)
```

---

## 7. GroupDETR：query 如何“成组”，训练/推理为什么 query 数不同

这一节只讲“工程里真正发生了什么”，避免停留在概念层。

### 7.1 配置入口：group attention + num_query

config 里有两处共同决定 GroupDETR 行为：

1) `group_detr` 参与 decoder attention：

```python
dict(
  type='GroupMultiheadAttention',
  group=group_detr,
  embed_dims=256,
  num_heads=8,
)
```

2) head 初始化会把 query 数乘上 group：

- base `num_query=900`
- effective train-time queries：`group_detr * num_query = 11 * 900 = 9900`

### 7.2 源码关键逻辑：推理时裁剪回 900

源码位置：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`

在 `BEVFormerOccupancyHeadApollo.forward()` 中有显式裁剪：

```python
if not self.training:
    object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
```

逐段解释：

1. `self.query_embedding.weight` 里存的是“训练时的全部 query”（这里已经把 `num_query` 乘上 `group_detr` 了）。
2. 推理时通过 slice 把 query 数裁剪回 `num_query // group_detr`。
3. 因此你在 `forward_test` 实测得到的 `outs.all_cls_scores` / `outs.all_bbox_preds` 的 `num_query` 维，才会是 `900` 而不是 `9900`。

这句话决定了：

- 训练：query=9900（更充分监督）
- 推理：query=900（更快）

### 7.3 decoder 输出张量的标准组织方式（与 DETR 对齐）

同一文件中，transformer decoder 的典型输出为：

```text
hs: (num_layers, num_query, bs, embed_dims)
```

随后会经 `permute` 变成：

```text
(num_layers, bs, num_query, embed_dims)
```

并逐层生成：

- `all_cls_scores`：`(num_layers, bs, num_query, num_classes)`
- `all_bbox_preds`：`(num_layers, bs, num_query, code_size)`

### 7.4 v1.0-mini 实测（forward_test，query=900）

```text
outs.all_cls_scores = (6, 1, 900, 10)
outs.all_bbox_preds = (6, 1, 900, 8)
```

对照解释：

- `6`：decoder 层数（`decoder.num_layers=6`）
- `900`：推理裁剪后的 query 数
- `10`：nuScenes 10 类
- `8`：3D box code size

### 7.5 深挖 det（1）：`outs` 字典是怎么从 `hs` 变出来的？

到这里为止，我们只“看到”了 `outs.all_cls_scores/all_bbox_preds` 的 shape，但还缺少一层：它们是怎么在 head 里构造出来的。

源码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`（forward 中后段，接 transformer 输出之后）

```python
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
  ...
  outputs_coord = tmp
  outputs_classes.append(outputs_class)
  outputs_coords.append(outputs_coord)

outputs_classes = torch.stack(outputs_classes)
outputs_coords = torch.stack(outputs_coords)

outs = {
  'bev_embed': bev_embed,
  'all_cls_scores': outputs_classes,
  'all_bbox_preds': outputs_coords,
  ...
}
```

读这段代码的“思路”是：

1. `hs` 的每一层 `hs[lvl]` 都是一个 decoder layer 的输出，形状是 `(bs, num_query, embed_dims)`。
2. `cls_branches[lvl] / reg_branches[lvl]` 是“每层一个 head”的 DETR 经典做法。
3. `reference`（初始/逐层更新的 reference points）会参与 box 回归的反归一化/偏移（`inverse_sigmoid` + 若干维度上的加法与 sigmoid）。
4. `torch.stack(outputs_classes)` 把每层的结果堆回去，因此你最终看到：
   - `all_cls_scores = (num_layers, bs, num_query, num_classes)`
   - `all_bbox_preds = (num_layers, bs, num_query, code_size)`

### 7.6 深挖 det（2）：`get_bboxes()` 如何把 (L, B, Q, …) 变成最终 3D boxes

det 的“对外输出协议”不是 `all_cls_scores`，而是 `boxes_3d/scores_3d/labels_3d`。
这一步发生在 detector 的 `simple_test_pts()` 里：

- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py::simple_test_pts()`
  - `bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)`

而 `get_bboxes()` 的实现在基类里：

源码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head.py::get_bboxes()`

```python
preds_dicts = self.bbox_coder.decode(preds_dicts)

for i in range(num_samples):
  preds = preds_dicts[i]
  bboxes = preds['bboxes']  # shape: (max_num=300, 9)
  ...
  scores = preds['scores']
  labels = preds['labels']
  ret_list.append([bboxes, scores, labels])
```

读这段代码要抓住两点：

1. **真正的解码/筛选逻辑都在 `bbox_coder.decode()` 里**。
   - 也就是说：`all_cls_scores/all_bbox_preds` → (top-k + decode) → `preds['bboxes/scores/labels']`。
2. 这里注释写死了 `max_num=300`，这也是你最终 det 结果里“每帧最多 300 个框”的来源（常见配置是 NMS-free 的 top-k）。

补一个“实测闭环”：在 v1.0-mini 的 debug 输出里，你会看到 `scores_3d/labels_3d` 的 shape 就是 `(300,)`，这与 `max_num=300` 完全一致。

同时你可能注意到运行时会打印来自 `nms_free_coder.py` 的 warning，这也侧面印证了：**top-k / index 解码发生在 bbox coder 的 decode 阶段**（即 `self.bbox_coder.decode(preds_dicts)` 这一行）。

下面把 `bbox_coder.decode()` 这一步直接“写透”。

### 7.7 写透 det：`NMSFreeCoder.decode()` / `decode_single()` 逐段代码对照

#### 7.7.1 先定位配置：为什么 top-k 是 300

config 在：`projects/configs/bevformer/bev_tiny_det_occ_apollo.py`：

- `bbox_coder=dict(type='NMSFreeCoder', ..., max_num=300, num_classes=10)`

所以最终每帧“最多保留 300 个框”的硬来源就是 `max_num=300`。

#### 7.7.2 源码位置

coco-like（但用于 3D）NMS-free coder 实现在：

- `projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py::NMSFreeCoder`

#### 7.7.3 `decode()`：只取 decoder 最后一层

```python
all_cls_scores = preds_dicts['all_cls_scores'][-1]
all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
```

这两行解释了一个常见疑问：

- head forward 里我们得到 `outs.all_cls_scores.shape = (L, B, Q, C)`（你实测是 `(6,1,900,10)`），但 decode 时实际只用最后一层：
  - `all_cls_scores.shape = (B, Q, C) = (1,900,10)`

然后对 batch 逐个 sample 做 `decode_single()`：

```python
for i in range(batch_size):
   predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
```

#### 7.7.4 `decode_single()`：把 `query×class` 展平后做全局 top-k

```python
cls_scores = cls_scores.sigmoid()
scores, indexs = cls_scores.view(-1).topk(max_num)
labels = indexs % self.num_classes
bbox_index = indexs // self.num_classes
bbox_preds = bbox_preds[bbox_index]
```

逐行对照（也对应你看到的 warning 行）：

1. `sigmoid()`：由于分类头是 sigmoid/focal loss 体系，需要把 logits 变成概率。
2. `view(-1)`：把 `(Q, C)` 展平。
  - 以你的实测为例：$Q=900$、$C=10$，展平长度是 $9000$。
3. `topk(max_num)`：在 **所有 query 的所有类别** 的二维网格里做全局 top-k。
  - 因为 config 里 `max_num=300`，所以这里直接产出 `scores.shape == (300,)`。
4. `labels = indexs % num_classes`：展平索引的余数部分就是类别 id。
5. `bbox_index = indexs // num_classes`：商就是 query id（第几个 query）。
6. `bbox_preds = bbox_preds[bbox_index]`：从 `(Q, box_dim)` gather 成 `(300, box_dim)`。

这一步就是“NMS-free”的精髓：

- 不需要 NMS。
- 用 DETR 的 query 作为候选集，直接 top-k 选出最高分的 `query×class` 对。

#### 7.7.5 `denormalize_bbox()` 与 `post_center_range` 过滤

```python
final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
...
mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)
boxes3d = final_box_preds[mask]
scores = final_scores[mask]
labels = final_preds[mask]
```

- `denormalize_bbox()`：把网络输出的归一化 box 参数还原到真实 3D 坐标系（依赖 `pc_range`）。
- `post_center_range`：只用 box center 做范围裁剪。

#### 7.7.6 和你的输出 shape “闭环”

- `max_num=300` → `.topk(300)` → `scores_3d/labels_3d` 理论上就是 `(<=300,)`。
- 你这次 v1.0-mini 实测最终恰好还是 `(300,)`，说明 range filter 在该样本里没有把 top-k 中的 box 裁掉（或裁掉后数量仍保持 300 的量级）。

---

## 8. Occupancy Head：体素网格、logits 形状与后处理形状为何不同

源码位置：

- head 主体：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`
- 后处理入口一般在 detector 的 `simple_test_pts()`（同第 1 节）调用链中

### 8.1 体素网格尺寸从 config 推导（并与实测闭环）

config：

- `occupancy_size=[0.5, 0.5, 0.5]`
- `point_cloud_range=[-50, -50, -5, 50, 50, 3]`

则：

$$
x=\frac{50-(-50)}{0.5}=200,\quad
y=\frac{50-(-50)}{0.5}=200,\quad
z=\frac{3-(-5)}{0.5}=16
$$

总 voxel：

$$N_{voxel}=200\times200\times16=640000$$

### 8.2 v1.0-mini 实测：logits

```text
outs.occupancy_preds = (1, 640000, 16)
```

解读：

- `640000`：体素数
- `16`：`occupancy_classes`（每个体素的分类 logits 维度）

### 8.2.1 深挖 occ（1）：从 `bev_embed` 到 `(bs, Nvoxel, occ_dims)` 的展平

occ 分支最容易困惑的点是：它最终要输出的是**体素网格**，但 transformer 输出的是 **BEV token 序列**（`(N_bev, bs, embed_dims)`）。
因此 head 里一定要做一次“从 BEV 到体素”的上采样/reshape。

源码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`

最直接的一条路径是 `upsample_occ()`：

```python
bev_for_occ = bev_for_occ.permute(1, 2, 0).contiguous().view(bs*seq_len, -1, self.bev_h, self.bev_w)
occ_pred = self.upsample_layer(bev_for_occ)
occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_dims, self.occ_zdim, self.occ_xdim, self.occ_ydim)
occ_pred = occ_pred.permute(0, 2, 3, 4, 1)
occ_pred = occ_pred.contiguous().view(bs*seq_len, self.occ_zdim*self.occ_xdim*self.occ_ydim, self.occ_dims)
return occ_pred
```

这段代码回答了三个关键问题：

1. `bev_for_occ.permute(1,2,0)`：把 `(N_bev, bs, C)` 转成 `(bs, C, N_bev)`，再 reshape 成 `(bs, C, bev_h, bev_w)`，恢复 BEV 平面。
2. `upsample_layer`：把 `bev_h×bev_w` 的 BEV 平面提升到 `occ_xdim×occ_ydim`，同时引入 `occ_zdim`（具体实现取决于该层结构）。
3. 最后 `view(..., Nvoxel, occ_dims)`：把三维体素 `(z,x,y)` 展平成 `Nvoxel=z*x*y`，得到你实测看到的 `Nvoxel=640000`。

### 8.2.2 写透 occ：`upsample_layer` 的模块结构（为什么 z 维“藏在通道里”）

上面这段 `upsample_occ()` 里把关键实现“黑盒”留给了 `upsample_layer`。这里把它拆开讲清楚。

源码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py::BEVFormerOccupancyHeadApollo.__init__()`

默认（`occ_tsa=None`）分支是一个 `nn.Sequential`：

```python
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
```

把它理解成三步：

1. **BEV 空间上采样 2×**：`ConvTranspose2d(stride=2)` 把 `(bev_h, bev_w)` 变成 `(2*bev_h, 2*bev_w)`。
2. **把 z 维折叠进通道**：`1×1 Conv` 把 `embed_dims → occ_zdim*occ_dims`。
   - 这一步非常关键：
   - occ 的高度轴 $z$ 并不是通过显式 3D 卷积生成的，而是“编码在 channel 里”，后续再用 `view(..., occ_dims, occ_zdim, ...)` 拆出来。
3. **再次空间上采样 2×**：第二次反卷积把空间变成 `(4*bev_h, 4*bev_w)`。

因此 `upsample_occ()` 里这行 reshape 才成立：

```python
occ_pred = occ_pred.view(bs*seq_len, occ_dims, occ_zdim, occ_xdim, occ_ydim)
```

也就是说：

- `occ_xdim, occ_ydim` 的空间分辨率来自 **两次 stride=2 的反卷积**（总倍率 4×）。
- `occ_zdim` 来自 **通道拆分**，因为此时 feature map 的 channel 数就是 `occ_zdim*occ_dims`。

> 额外分支：如果 config 里开启 `occ_tsa`，`upsample_layer` 会变成“只做空间上采样”的版本（通道仍保持 `embed_dims`），随后再用 `occ_tsa_head(1×1 Conv)` 把 `embed_dims → occ_zdim*occ_dims`；并且引入一个 occ TSA transformer 去融合 image features。

> 常见误区：
> - `permute(0,2,3,4,1)` 看起来像“乱换维度”，其实它是在把 `occ_dims`（类别/通道）放到最后，方便后续按体素做分类。
> - `Nvoxel` 的顺序是 `z*x*y` 还是 `x*y*z` 在不同项目里可能不同；这里以代码里 `view(self.occ_zdim*self.occ_xdim*self.occ_ydim, ...)` 为准。

### 8.3 为什么最终输出变成 `(640000, 2)`（实现相关的编码格式）

本工程在推理路径中会对 logits 做后处理（函数名通常类似 `get_occupancy_prediction()`），把 logits 转成离散预测并编码输出。

因此你在最终 `occ_results` 中看到的往往是：

```text
occupancy_preds = (640000, 2)
```

这一维 `2` 是**工程当前实现**定义的编码格式（例如“类别 id + mask/置信度/占用标记”等；具体含义以实现为准），不等同于 logits 的 `occ_classes=16`。

### 8.3.1 深挖 occ（2）：`get_occupancy_prediction()` 如何把 logits 变成 `(Nvoxel, 2)`

源码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head.py::get_occupancy_prediction()`

```python
occupancy_preds = occ_results['occupancy_preds']

occupancy_preds = occupancy_preds.reshape(-1, self.occupancy_classes)
occupancy_preds = occupancy_preds.sigmoid()
occupancy_preds = torch.cat(
   (occupancy_preds, torch.ones_like(occupancy_preds)[:, :1] * occ_threshold),
   dim=-1)
occ_class = occupancy_preds.argmax(dim=-1)
occ_index, = torch.where(occ_class < self.occupancy_classes)
occ_class = occ_class[occ_index[:]]
occupancy_preds = torch.stack([occ_index, occ_class], dim=-1)
```

这段后处理的关键思想是：

1. 把所有体素展平成二维：`(-1, occupancy_classes)`。
2. 过 `sigmoid()` 得到概率（`focal_loss` 分支）。
3. 手动拼了一个“阈值列”，并用 `argmax` 同时实现：
  - “如果没有任何类别概率超过阈值，就落到阈值列（相当于 empty/ignore）”
  - “否则取最大概率的类别”
4. 最终用 `torch.stack([occ_index, occ_class], dim=-1)` 输出一个稀疏编码：
  - 第 1 列：体素 index
  - 第 2 列：体素类别

因此最终形状是 `(Nocc, 2)`，其中 `Nocc` 可能小于 `Nvoxel`（只保留非 empty/有效体素）。

本工程这次 v1.0-mini 的 debug 输出里看到的是 `(640000, 2)`，说明在当前配置/阈值/类别定义下，该实现返回了“按实现定义的编码结果”（并没有明显稀疏到远小于 `Nvoxel` 的数量级）。

### 8.4 代码对照：occ 后处理是在 `Detector.simple_test()` 里触发的

源码：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py::BEVFormer.simple_test()`

```python
new_prev_bev, bbox_pts, occ_results = result
if occ_results['occupancy_preds'] is not None:
    occ_results = self.pts_bbox_head.get_occupancy_prediction(occ_results, occ_threshold)
```

读这段代码要抓住：

- `simple_test_pts()` 返回的 `occ_results['occupancy_preds']` 还是 head forward() 的原始输出（通常是 logits）。
- 在 `simple_test()` 里，如果不为空，就会调用 `get_occupancy_prediction()` 把它转换成最终可渲染/可评估的格式。
- 这就是为什么你看到“同一个 key：`occupancy_preds`”，但形状会从 `(1, 640000, 16)` 变成 `(640000, 2)`。

---

## 9. forward_test 的最终输出结构：det 与 occ 各返回什么

这是“对外 API 视角”的总结：你从 `model(return_loss=False, **data)` / `forward_test` 拿到的东西长什么样。

### 9.1 v1.0-mini 实测摘要

```text
outputs 是 tuple(len=2)

[0] det: list(len=1)
  [0] dict(keys=['pts_bbox'])
    pts_bbox: dict(keys=['boxes_3d', 'scores_3d', 'labels_3d'])

[1] occ: dict(keys=['occupancy_preds', 'flow_preds'])
  occupancy_preds: Tensor(shape=(640000, 2), device=cuda)
  flow_preds: None
```

### 9.2 对应到工程职责

- det：遵循 mmdet3d 标准输出协议，供 nuScenes eval/可视化直接消费
- occ：工程自定义结果协议，供 occ eval/渲染消费

---

## 10. 用于“实测闭环”的最小脚本：为什么推荐用 hook 而不是在 forward 里 print

脚本文件：`tools/debug_shapes_v1mini.py`

### 10.1 这个脚本做了什么

1. 读取 config
2. 构建 v1.0-mini dataset/dataloader（batch=1）
3. 构建 model
4. forward_test 跑一帧
5. 通过 hook 抓取关键内部张量（避免污染训练/测试代码）

### 10.2 为什么 hook 是工程里更稳的做法

- 不会破坏分布式脚本（`pdb`/print 很容易把多卡卡死或刷爆日志）
- 不需要维护“debug 分支”模型代码
- 关键 shape 可以集中在一个工具脚本里长期复用

### 10.3 运行方式

```bash
python3 tools/debug_shapes_v1mini.py \
  --config projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
  --dataroot data/nuscenes \
  --version v1.0-mini \
  --device cuda
```

---

## 11. 常见坑与排障顺序（工程化版本）

### 11.1 推荐排查顺序

1. 先确认 `img_metas`：到底是 resize 还是 pad（本例是 pad：450→480）
2. 固定 head 入口处的 `mlvl_feats` shape（本例是 1×6×256×28×48）
3. 再看 transformer 输出（`bev_embed`/`all_cls_scores`/`all_bbox_preds`）
4. 最后再看 occ：先 logits，再后处理

### 11.2 强烈不建议

- 在 forward 里插 `pdb.set_trace()`（分布式场景极易挂死）
- 纯靠“stride 除法”猜 feature map 尺寸

---

## 附：本文涉及的关键文件索引

- Config：`projects/configs/bevformer/bev_tiny_det_occ_apollo.py`
- Detector：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
- Backbone(DLA)：`projects/mmdet3d_plugin/models/backbones/dla.py`
- Neck(SECONDFPNV2)：`projects/mmdet3d_plugin/models/necks/second_fpnv2.py`
- Transformer：`projects/mmdet3d_plugin/bevformer/modules/transformer.py`
- Head(det+occ)：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`
- Debug script：`tools/debug_shapes_v1mini.py`

