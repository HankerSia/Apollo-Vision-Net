# BEV Sparse 多任务（检测 + MapTR + FlashOcc）在 nuScenes 上的配置与源码走读（基于 OE3.7）

本文基于配置文件：
- `samples/ai_toolchain/horizon_model_train_sample/scripts/configs/bev/bev_sparse_det_maptr_flashocc_henet_tinym_nuscenes.py`

目标
- 将本配置中涉及的三类任务（检测 / 在线矢量地图 MapTR / 占用预测 FlashOcc）从“数据准备、整体模型架构、到配置项与源码实现的逐项对应”做一次可复现的走读。

说明（阅读提示）
- OE3.7 的大部分模型与训练逻辑封装在 Python 包 `hat` 中（随官方 wheel 发布）。为便于对照阅读，我已把发布的 `horizon_torch_samples` wheel 解包，源码位于：
  - `extracted/horizon_torch_samples/hat/`
- 文中所有指向实现的位置均基于上述解压目录。
- 如果你的目的是跑通训练/导出，请确保本机环境与 OE3.7 提供的 wheel（Python/Torch/CUDA 版本）一致；如果仅为阅读或写技术博客，则不必局部运行。

---

## 1. 运行与环境：先讲清楚“什么能跑”

这份示例配置文件在开头直接 `import horizon_plugin_pytorch as horizon` 以及大量 `from hat... import ...`，因此运行训练/导出前需要：

1) **Python 版本匹配**
- OE3.7 包名里带 `py310`，并包含 `horizon_plugin_pytorch-...-cp310-...whl`，通常意味着官方预期环境是 Python 3.10。

2) **依赖安装路径**
- `hat` 的源码随 `horizon_torch_samples` wheel 分发（纯 Python）。
- `horizon_plugin_pytorch` 是带 CUDA/Torch 绑定的二进制 wheel（需要 Python 版本/torch/cu 版本匹配）。

如果你只是做“读代码 + 写博客/解析”，不需要安装运行；如果你要跑训练，建议创建一个与 OE3.7 wheel 匹配的 conda env。

---

## 2. 数据：目录约定、准备内容、生成流程

配置里关键路径（摘自配置头部变量）：

- `data_rootdir = "tmp_data/occ3d_nuscenes/bev_occ_new/v1.0-trainval"`
  - 训练/验证集期望在此目录下存在：
    - `train_lmdb/`
    - `val_lmdb/`

- `meta_rootdir = "./tmp_data/nuscenes/meta"`
  - nuScenes 的 meta/地图相关文件输出目录

- `anchor_file = "./tmp_data/nuscenes/nuscenes_kmeans900.npy"`
  - 检测 head 的锚点（稀疏 anchor / memory bank 初始 anchor）

- `map_anchor_file = "./tmp_orig_data/nuscenes/kmeans_map_100.npy"`
  - 地图 head 的锚点（在线矢量地图 query anchor）

- `sd_map_path = "./tmp_data/nuscenes/osm"`
  - 静态地图/OSM（用于矢量地图 GT 或辅助）

### 2.0 先把“这份配置到底期望什么数据”说清楚

这份配置的 `train_dataset/val_dataset` 是 `NuscenesSparseMapDataset`，它**读取的是打包好的 LMDB**，并且训练时会用到三类 GT：

1) **检测 GT（3D box）**：来自 packed sample 里的 `gt_bboxes` / `gt_boxes` / `gt_names` 等字段
2) **在线矢量地图 GT（MapTR）**：在 `NuscenesMapDataset.vectormap_pipeline()` 里在线生成，写入 `gt_labels_map`、`gt_instances`（`LiDARInstanceLines`）
3) **占用 GT（Occ3D）**：本配置 `with_lidar_occ=True`，会在 `NuscenesBevDataset.__getitem__()` 里生成 `gt_occ_info = sample._get_lidar_occ(lidar2global)`

所以要“能直接跑这份配置”，你的 `train_lmdb/val_lmdb` 必须满足两个硬条件：

- LMDB 的 sample 格式要能被 `NuscenesSample` 解析（它读取的是 `sample["cam"]` 列表，而不是 `sample["cams"]` 字典）
- sample 里必须已经包含 occ3d 的 bytes：`voxel_semantics` / `mask_lidar` / `mask_camera`（否则 `gt_occ_info` 生成会直接报错）

### 2.1 nuScenes 基础信息与 gt database（通用）

通用数据预处理工具脚本：
- `scripts/tools/create_data.py`

核心调用链（nuScenes 分支）：
- `hat.data.datasets.nuscenes_dataset.create_nuscenes_infos`
- `hat.data.datasets.nuscenes_dataset.create_nuscenes_groundtruth_database`

你通常会像这样调用：

```bash
conda activate apollo_vnet
cd /home/nuvo/Downloads/OE3.7/horizon_j6_open_explorer_v3.7.0-py310_20251215
python samples/ai_toolchain/horizon_model_train_sample/scripts/tools/create_data.py \
  --dataset nuscenes \
  --root-dir /path/to/nuscenes \
  --extra-tag nuscenes \
  --out-dir ./tmp_data/nuscenes
```

脚本位置：
- `scripts/tools/create_data.py`

### 2.2 FlashOcc / Occ3D 相关 info 生成

该示例是 “det + map + occ”，并且显式使用了 occ3d 相关路径，因此通常还需要生成包含占用 GT 路径等信息的 nuScenes info：

- `scripts/tools/dataset_converters/create_data_flashocc.py`
  - CLI 参数（脚本里只有两个）：
    - `-s/--src-data-path`：nuScenes 根目录
    - `-o/--out-dir`：输出目录
  - 输出文件：
    - `{out_dir}/nuscenes_infos_train.pkl`
    - `{out_dir}/nuscenes_infos_val.pkl`
  - 关键字段回填逻辑（脚本最后的 `add_ann_adj_info()`）：
    - 写入 `scene_token` / `scene_name`
    - 写入 `occ_path = "occ3d/gts/{scene_name}/{sample_token}"`
    - 注意：它会把 `ann_infos` 最终覆盖为 `get_gt(info)` 的返回值 `(gt_boxes, gt_labels)`（tuple），不再是 annotation dict 列表

实现位置（脚本本身）：
- `scripts/tools/dataset_converters/create_data_flashocc.py`

### 2.3 生成 camera 参数 / homography / reference points（BEV 采样几何）

工具脚本：
- `scripts/tools/gen_reference_points_nusc.py`

它会：
1) 生成 camera intrinsic、sensor2ego 等 npy（按城市或全集）
2) 生成 homography
3) 初始化模型并根据配置选择 `ref_gen` / `ref_gen_bevformer` 生成 reference points

CLI 参数（来自脚本 `parse_args()`）：
- `--data-path`：nuScenes 根目录
- `--save-path`：输出根目录（脚本会自动创建 `${save_path}/${model_name}/...`）
- `--save-by-city`：按城市分别落盘
- `--version`：如 `v1.0-mini` / `v1.0-trainval`
- `-c/--config`：配置文件路径（用于 init_model + 选择 `gen_ref_type`）

输出目录结构（与脚本实现一致）：
- `${save_path}/${model_name}/...` 下会生成（至少）
  - `sensor2ego_translation.npy`
  - `sensor2ego_rotation.npy`
  - `camera_intrinsic.npy`
  - `homography.npy`
  - 以及 reference points generator 写出的若干 `.npy`

实现位置：
- `scripts/tools/gen_reference_points_nusc.py`
- 其依赖：
  - `scripts/tools/gen_camera_param_nusc.py`
  - `scripts/tools/homography_generator.py`
  - `scripts/tools/reference_points_generator.py`

### 2.4 打包成 LMDB（train_lmdb/val_lmdb）

配置中的 dataloader 直接读取 `train_lmdb` / `val_lmdb`。

这里要分清两条 pack 路线（非常关键）：

#### 路线 A（与本配置对齐）：`NuscenesPacker` → sample 里是 `cam` 列表

脚本入口：
- `scripts/tools/datasets/nuscenes_packer.py`

内部 packer：
- `hat.data.datasets.nuscenes_dataset.NuscenesPacker`

它产出的 sample 里 camera 字段是 `cam`（列表），能被 `NuscenesSample.get_cam_by_name()` 正常解析。

**但**：本配置需要 occ3d（`with_lidar_occ=True`），而 `scripts/tools/datasets/nuscenes_packer.py` 的 CLI 默认不暴露 `need_occ`。
你必须保证 pack 时把 `voxel_semantics/mask_lidar/mask_camera` 写进 sample（来源是 `occ3d/gts/{scene}/{token}/labels.npz`）。

最稳妥的做法（不改脚本）：直接按 `NuscenesPacker.__init__()` 的参数写一个最小 pack 调用：

```bash
conda activate apollo_vnet
cd /home/nuvo/Downloads/OE3.7/horizon_j6_open_explorer_v3.7.0-py310_20251215
python - << 'PY'
from hat.data.datasets.nuscenes_dataset import NuscenesPacker

root = "/path/to/nuscenes"
out_root = "./tmp_data/occ3d_nuscenes/bev_occ_new/v1.0-trainval"

for split in ["train", "val"]:
    pack_path = f"{out_root}/{split}_lmdb"
    packer = NuscenesPacker(
        version="v1.0-trainval",
        src_data_dir=root,
        target_data_dir=pack_path,
        split_name=split,
        num_workers=20,
        pack_type="lmdb",
        only_lidar=False,
        need_occ=True,
    )
    packer()
PY
```

#### 路线 B（用于 occ-only 数据集）：`Occ3dNuscenesPacker` → sample 里是 `cams` 字典

脚本入口：
- `scripts/tools/datasets/occ3d_nuscenes_packer.py`

内部 packer：
- `hat.data.datasets.occ3d_nuscenes_dataset.Occ3dNuscenesPacker`

它的 sample 结构用的是 `cams`（字典），与 `NuscenesSample` 期望的 `cam`（列表）不同，因此**不适合直接喂给本配置的 `NuscenesSparseMapDataset`**。

另一个容易踩坑的点：脚本参数里写的是 `--anno-file kitti_train.json`，但实现里实际是 `open(annFile, "rb")` + `pickle.load()`，也就是你传的 `--anno-file` 更像是一个 `*_infos_train.pkl` 结构的 pickle。

实现位置：
- `scripts/tools/datasets/nuscenes_packer.py`
- `scripts/tools/datasets/occ3d_nuscenes_packer.py`
- `extracted/horizon_torch_samples/hat/data/datasets/nuscenes_dataset.py`
- `extracted/horizon_torch_samples/hat/data/datasets/occ3d_nuscenes_dataset.py`

### 2.5 占用 GT（Occ3D labels.npz）的硬约定：必须存在且 shape 固定

无论你走路线 A 还是 B，occ GT 最终都来自 nuScenes 根目录下的：

- `occ3d/gts/{scene_name}/{sample_token}/labels.npz`

并且 `labels.npz` 必须包含：
- `semantics`
- `mask_lidar`
- `mask_camera`

在 `hat` 的实现里，这三个数组会以 `np.uint8` 读出并 reshape 为 `(200, 200, 16)`。

### 2.6 batch 里的关键字段与 shape：`collate_nuscenes` 的“view 维度摊平规则”

本配置 dataloader 的 `collate_fn=collate_nuscenes`（实现：`extracted/horizon_torch_samples/hat/data/collates/nusc_collates.py`）。

- 对 camera 相关 key（`img/ego2img/lidar2img/cam2ego/camera_intrinsic`）：
  - 会把每个 sample 的 6 个 view **直接摊平后 stack**，所以通常得到 `(B*6, ...)` 形状
- 对 `gt_labels_map/gt_instances/gt_pv_seg_mask/osm_vectors`：
  - 这些 key 在 `list_keys` 里，会保留为 list（长度为 `B`），由 head/criterion 自己处理对齐
- `gt_occ_info`：
  - 是一个 dict，默认走 `default_collate`，通常会变成 `dict[str, Tensor]`

### 2.7 训练集真正用的 Dataset：NuscenesSparseMapDataset

本配置训练集使用：
- `type="NuscenesSparseMapDataset"`

实现位置：
- `extracted/horizon_torch_samples/hat/data/datasets/nuscenes_map_dataset.py`
  - `class NuscenesMapDataset`：负责 vectormap GT（静态地图矢量化、aux seg 等）
  - `class NuscenesSparseMapDataset`：训练态/测试态的 `prepare_train_data/prepare_test_data` 封装

两个细节（建议你读代码时带着这个结论看）：

1) **`sd_map_path` / OSM 在本配置里默认没有启用**
- 配置虽然定义了 `sd_map_path = "./tmp_data/nuscenes/osm"`，但 `train_dataset/val_dataset` 并没有传 `sd_map_path=...`
- 因此 `VectorizedLocalMap(sd_map_path=None)`，`osm_vectors/osm_mask` 会一直是 `None`
- 如果你要启用 OSM 辅助，只需要在 dataset dict 里补一行：`sd_map_path=sd_map_path`

2) **`NuscenesSparseMapDataset` 不是时序 dataset：它最终只返回 `prepare_train_data(...)[0]`**
- `prepare_train_data()` 内部会构造一个队列（`queue_length`），但 `NuscenesSparseMapDataset.__getitem__()` 最后只取 `data[0]` 返回
- 所以本配置下 `MapTRCriterion.forward()` 走的是非时序分支（batch 里不会有 `seq_meta`）

---

## 3. 总体模型设计：SparseBevFusionMultitaskOE（三头多任务）

配置最顶层模型：
- `model = dict(type="SparseBevFusionMultitaskOE", backbone=..., neck=..., head=task_heads, depth_branch=...)`

实现位置：
- `extracted/horizon_torch_samples/hat/models/structures/sparse_multitask.py`
  - `class SparseBevFusionMultitaskOE`

### 3.1 forward 主逻辑（多 head 路由）

`SparseBevFusionMultitaskOE.forward(data)` 的关键点：

1) `extract_feat(data["img"])` 得到 backbone/neck 的特征金字塔 `feature_maps`
2) 遍历 `head`（ModuleDict）：分别调用 det / om(map) / occ head
3) 对输出进行 `_post_process`（按训练阶段/quant_analysis 做裁剪等）

代码位置：
- `extracted/horizon_torch_samples/hat/models/structures/sparse_multitask.py`

#### 3.1.1 `SparseBevFusionMultitaskOE` 的实现代码走读（按 `forward()` 从上到下）

实现位置：
- `extracted/horizon_torch_samples/hat/models/structures/sparse_multitask.py`

它的 forward 主线就是：**抽特征** → **按 head 类型路由调用** → **统一做 `_post_process`（注入 depth 辅助监督）**。

关键实现片段（省略无关分支）：

```python
if not self.only_lidar:
  feature_maps = self.extract_feat(data["img"])
else:
  feature_maps = None

lidar_feature = self.forward_lidar_feature(data) if self.lidar_net else None

model_outs = OrderedDict()
for head_name, head in self.head.items():
  head_outputs = self.forward_single_head(
    head, feature_maps, data, lidar_feature, self.compiler_model
  )
  if isinstance(head_outputs, dict):
    model_outs.update({f"{head_name}.{k}": v for k, v in head_outputs.items()})
  else:
    model_outs[head_name] = head_outputs

out = self._post_process(feature_maps, model_outs, data)
```

两个实现细节（建议对照源码验证）：

1) **路由完全由 head 的“类”决定（不是由 head_name 决定）**
- det/occ：`SparseBEVOEHead`、`FlashOccHead` 共享调用签名：`head(feature_maps=..., metas=data, lidar_feature=..., compiler_model=...)`
- map：`SparseMapPerceptionDecoder` 使用调用签名：`head(mlvl_feats=feature_maps, data=data, lidar_feature=...)`

2) **DenseDepth 的 loss 注入点不在 head，而在结构体 `_post_process()`**
- `SparseBevFusionMultitaskOE` 继承自 `SparseBevFusionOE`，其 `_post_process()` 在：
  - `extracted/horizon_torch_samples/hat/models/structures/sparseLIF.py`
- 训练态（且非 tracing 且非 `preparing_qat`）会额外加：

```python
depths = self.depth_branch(feature_maps, data)
model_outs["loss_dense_depth"] = self.depth_branch.loss(depths, data)
```

### 3.2 Backbone：HENet

配置里 backbone：
- `type="HENet"`

实现位置：
- `extracted/horizon_torch_samples/hat/models/backbones/henet.py`

结构要点：
- `patch_embed`：开头两层 stride-2 conv（或其他 stem 变体）把输入降采样
- `stages`：4 个 stage，每个 stage 是 `BasicHENetStageBlock`
- `downsample_block`：stage 间的 S2DDown
- `include_top=False` 时返回多尺度特征 `outs`（供检测/BEV heads 使用）
- 量化相关：`QuantStub/DeQuantStub` + `hnn.Interpolate`，并提供 `set_qconfig()/fuse_model()`

#### 3.2.1 `HENet.forward()`：它输出的 stride/通道为什么能对上 `MMFPN`

这份配置里 `MMFPN` 的入参是：
- `in_strides=[2,4,8,16,32]`
- `in_channels=[64,64,128,192,384]`

而 `HENet(include_top=False)` 的实现里，**第 0 个 stage 会额外 append 一个上采样特征**，从而把输出做成 5 个尺度（包含 stride=2）。

关键实现（`extracted/horizon_torch_samples/hat/models/backbones/henet.py`）：

```python
x = self.patch_embed(x)
outs = []
for idx in range(len(self.stages)):
  x = self.stages[idx](x)
  if not self.include_top:
    x_normed = self.stage_norm[idx](x)
    if idx == 0:
      outs.append(self.up(x_normed))
    outs.append(x_normed)
  if idx < len(self.stages) - 1:
    x = self.downsample_block[idx](x)
return outs
```

结合本配置参数，你可以把输出理解为：
- `outs[0]`：stride=2，C=64（stage0 的上采样输出）
- `outs[1]`：stride=4，C=64（stage0 原输出）
- `outs[2]`：stride=8，C=128
- `outs[3]`：stride=16，C=192
- `outs[4]`：stride=32，C=384

### 3.3 Neck：MMFPN（主干多尺度融合）

配置里：
- `neck=dict(type="MMFPN", ...)`

实现位置：
- `extracted/horizon_torch_samples/hat/models/necks/mm_fpn.py`

典型作用：
- 将 backbone 多 stage 输出做跨层融合，输出给检测/地图/占用 head

#### 3.3.1 `MMFPN.forward()`：先 1x1 对齐通道，再 top-down 相加融合

实现位置：
- `extracted/horizon_torch_samples/hat/models/necks/mm_fpn.py`

它做的事情非常标准：
1) 对每个输入尺度做 `conv_extract(1x1)`，把通道对齐到 `fix_out_channel`（本配置为 256）
2) 从大 stride → 小 stride 逐层上采样并相加（add 用 `nn.quantized.FloatFunctional`，便于量化）
3) 每层再过一个 `fpn_conv(3x3)`

关键实现片段：

```python
in_features = features[self.src_min_stride_idx :]
fpn_fuse = [self.conv_extract[i](in_features[i]) for i in range(len(in_features))]

for idx in range(len(in_features) - 1, 0, -1):
  cur_feat = self.upscale[idx - 1](fpn_fuse[idx])
  fpn_fuse[idx - 1] = self.conv_add[...].add(fpn_fuse[idx - 1], cur_feat)

fpn_fuse = [self.fpn_conv[i](fpn_fuse[i]) for i in range(len(fpn_fuse))]
return fpn_fuse
```

结合本配置：
- `out_strides=[4,8,16,32]`，因此会丢掉 `HENet` 输出的 stride=2 特征，只融合 stride=4..32
- `MMFPN` 的输出是 4 层 list（每层 `C=256`），后续 det/om/occ 都会按各自的 `feat_indices/level_index/bev_feat_index` 取用

### 3.4 Depth branch：DenseDepthNetOE（辅助监督）

配置里：
- `depth_branch=dict(type="DenseDepthNetOE", ...)`

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py`

说明：
- 配置注释写着 “for auxiliary supervision only”。它通常用于 view transformer / depth supervision 的辅助 loss。

#### 3.4.1 `DenseDepthNetOE.forward/loss()`：用 LiDAR 点投影构造稀疏 depth GT

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py`

`forward()` 的核心逻辑：
1) 从 `metas["camera_intrinsic"][...,0,0]` 取 focal（取不到就用 `equal_focal`）
2) 对 `feature_maps[:num_depth_layers]` 做 `1x1 conv` 得到 depth，再 `exp()` 确保为正
3) 用 focal 做尺度校正：`depth *= focal / equal_focal`

关键实现片段：

```python
focal = metas["camera_intrinsic"][..., 0, 0]
focal = focal.reshape(-1) if focal is not None else self.equal_focal
for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
  depth = self.depth_layers[i](self.dequant(feat).float()).exp()
  depth = depth.transpose(0, -1) * focal / self.equal_focal
  depth = depth.transpose(0, -1)
  depths.append(depth)
```

`loss()` 的关键点在 `_get_gt()`：它会拿 `metas["points"]`（LiDAR 点）和 `metas["lidar2img"]` 把点投影到每个相机的像素平面，再按每个预测分辨率下采样生成稀疏 depth map（未命中为 `-1`），最后只在 `gt>0` 的位置计算 L1。

注意：本配置里 depth loss 是在结构体 `_post_process()` 注入（见 3.1.1），不是由 det/om/occ head 自己加的。

---

## 4. 三个任务 head：Det / OM(MapTR) / Occ(FlashOcc)

配置里：
- `task_heads = OrderedDict(det=det_head, om=om_head, occ=occ_head)`

`SparseBevFusionMultitaskOE.forward()` 会逐个调用（见上节）。

---

### 4.1 Det head：SparseBEVOEHead（稀疏 BEV 检测）

配置块：`det_head = dict(type="SparseBEVOEHead", ...)`

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/head.py`

你在配置里能看到几个关键子模块：

1) **instance_bank / memory bank**
- 配置：`instance_bank=dict(type="MemoryBankOE", num_anchor=..., anchor=anchor_file, ...)`
- 实现：
  - `MemoryBankOE`：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/memory_bank.py`
  - 基类 `InstanceBankOE`：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/instance_bank.py`

2) **anchor_encoder**
- 配置：`type="SparseBEVOEEncoder"`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/det_blocks.py`

3) **deformable_model（跨视角/跨尺度聚合）**
- 配置：`type="DeformableFeatureAggregationOE"`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py`

4) **refine_layer（每层 refine）**
- 配置：`type="SparseBEVOERefinementModule"`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/det_blocks.py`

5) **target（训练标签分配、DN 等）**
- 配置：`type="SparseBEVOETarget"`，包含 dn 噪声、权重等
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/target.py`

6) **decoder（把 head 输出转为可视化/metric 的格式）**
- 配置：`decoder=dict(type="SparseBEVOEDecoder")`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/decoder.py`

配置里的 `operation_order` 非常关键：它定义了 decoder layer 内执行顺序（deformable/ffn/norm/refine/...）。这与很多 Sparse/DETR 类 head 设计一致。

#### 4.1.1 `SparseBEVOEHead` 的实现代码走读（按 `forward()` 从上到下）

> 说明：`head.py` 是随 OE3.7 发布的源码文件；为避免整文件拷贝带来的版权风险，这里只引用少量关键片段，并给出逐段实现解析。建议你对照文件逐行阅读：
> - `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/head.py`

##### (A) `__init__()`：用 `operation_order` 动态“拼装”一条 decoder 执行链

这个 head 的核心设计点是：**不把 decoder 写死为固定层结构**，而是把每个原子算子（interaction / deformable / ffn / refine / norm ...）放进 `op_map`，再按配置给的 `operation_order` 顺序组装成 `self.layers`。

关键代码片段（`head.py` 约 L112-L172）：

```python
self.op_map = {
  "temp_interaction": [temp_instance_interaction, 0],
  "interaction": [instance_interaction, 0],
  "norm": [norm_layer, 0],
  "ffn": [ffn, 0],
  "deformable": [deformable_model, 0],
  "refine": [refine_layer, 0],
}
for op in self.operation_order:
  self.layers.append(self.op_map[op][0] if self.op_map[op][1] == 0 else copy.deepcopy(self.op_map[op][0]))
  self.op_map[op][1] += 1
```

实现含义：
- 同一个算子（例如 `norm`、`interaction`）在 `operation_order` 里出现多次时，首次使用复用同一个 module，后续出现会 `deepcopy` 生成新实例。
- 这样一来，“一层 decoder”并不是一个 class，而是 `operation_order` 中按固定模式重复的若干算子块。

##### (B) `forward()` 输入输出与 batch 维度

`feature_maps` 在这份工程里通常是多尺度 list，每个 tensor shape 常见为：
- `feature_maps[k]`：`(B * num_views, C, H, W)`

`SparseBEVOEHead` 会用 `num_views` 把它折回 batch：
- `batch_size = feature_maps[0].shape[0] // self.num_views`

如果配置了 `level_index`，它会先按 index 选出 head 要用的那几层 FPN 特征（并在非编译模式下 `.float()`）。

##### (C) `instance_bank.get()`：把“query/anchor/temporal cache/DN cache”一次性取齐

`instance_bank.get(batch_size, metas, dn_metas, compiler_model)` 会返回四个核心张量：
- `instance_feature`：`(B, N, D)`，当前 frame 的 instance/query feature
- `anchor`：`(B, N, S)`，对应的 anchor/state（后续被 refine 更新）
- `temp_instance_feature`：`(B, N_temp, D)`，缓存的 temporal instances（可能为 None）
- `temp_anchor`：`(B, N_temp, S)`

其中 `S` 就是后面 regression state 的维度；在 `loss()` 里你能看到它默认只取 `reg[..., :len(self.reg_weights)]`（默认 reg_weights 长度为 10）。

##### (D) DN（Denoising training）：把 DN anchors 拼到 queries 后面 + 构造 attention mask 隔离 DN

当 `enable_dn=True` 且训练时，head 会让 `target.get_dn_anchors()` 生成一组 dn anchors，并把它们拼到 `anchor/instance_feature` 的末尾；同时构造一个 `attn_mask`，让“正常 queries”与 “dn queries” 的 self-attn 互不干扰。

关键代码片段（`head.py` 约 L232-L294）：

```python
anchor = torch.cat([anchor, dn_anchor], dim=1)
instance_feature = torch.cat([
  instance_feature,
  instance_feature.new_zeros(batch_size, num_dn_anchor, instance_feature.shape[-1]),
], dim=1)
attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
attn_mask[:num_free_instance, :num_free_instance] = False
attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
```

实现含义：
- `num_free_instance = N`，`num_dn_anchor = N_dn`，最终实例数变成 `N + N_dn`。
- `attn_mask` 的左上角（free↔free）放开，右下角用 `dn_attn_mask` 控制 dn 内部的可见性，交叉区域默认 mask 掉。

##### (E) 投影矩阵 `projection_mat`：从像素坐标系变换到可采样坐标（并做量化 stub）

`gen_projection_mat()` 会根据每个 view 的图像尺度构造一个 4x4 的 “view normalize” 矩阵，把像素坐标映射到 `[-1, 1]` 这类 grid-sample 坐标，再与 `metas[projection_mat_key]`（默认 `lidar2img`）相乘。

关键点：forward 里会把 `projection_mat` reshape 成 `(B, V, 4, 4)` 并喂给 `QuantStub`：
- `projection_mat = self.mat_quant_stub(projection_mat.view(bs, -1, 4, 4).float())`

##### (F) 主循环：按 `operation_order` 执行 interaction / deformable / ffn / refine

核心循环在（`head.py` 约 L361-L487）。这里每类 op 对应的语义是：

1) `temp_interaction` / `interaction`
- 都是基于 `DecoupledMultiheadAttention` 的 attention（实现里用 `fc_before/fc_after` 做 feature 维度变换）
- `temp_interaction` 用 temporal cache 做 cross-attn；`interaction` 是当前 queries 的 self-attn

2) `deformable`
- 把 `(instance_feature, anchor, anchor_embed, feature_maps, projection_mat)` 送进 `DeformableFeatureAggregationOE` 做跨视角特征聚合

3) `refine`
- `refine_layer(instance_feature, anchor, anchor_embed, ...)` 返回更新后的 `anchor`（也就是 reference/state）以及 `cls` 与 `qt`（quality）
- 每次 refine 都会 `prediction.append(anchor)`，所以最终输出是 “多层 deep supervision” 的 list

典型片段（`head.py` 约 L398-L447）：

```python
elif op == "deformable":
  instance_feature = self.layers[i](instance_feature, anchor, anchor_embed, feature_maps, projection_mat)
elif op == "refine":
  anchor, cls, qt = self.layers[i](instance_feature, anchor, anchor_embed, return_cls=..., last_layer=...)
  prediction.append(self.dequant(anchor))
  classification.append(cls if i != last else self.dequant(cls))
```

##### (G) “单帧 decoder → 时序 decoder”的分界：`num_single_frame_decoder`

当 `len(prediction) == num_single_frame_decoder` 时，会触发一次 `instance_bank.update(...)`：
- 用当前 frame 的输出更新/对齐 temporal bank
- 若还启用了 temporal DN，会额外调用 `target.update_dn(...)` 更新 dn targets

这也是你在配置里看到 `num_single_frame_decoder` 以及 `operation_order` 前半段没有 `temp_interaction` 的原因：它相当于先跑若干层 “单帧” 提案，再进入 “时序融合” 的层。

##### (H) 输出、cache 与训练分支

forward 的返回分两类：
- `compiler_model=True`：只返回最后一层的 `classification/prediction/quality` + `feature`（用于编译/导出）
- 否则：返回每层的 list（用于 deep supervision），并在非 tracing 情况下 cache 到 instance bank

训练时的关键分支（`head.py` 约 L540-L556）：
- 满足 `self.training=True` 且非 tracing 且不是 `preparing_qat` 才会走 `self.loss(output, metas)`；否则走 `post_process()`（解码出最终 boxes）

##### (I) `loss()`：每层 decoder 都算一次 cls/reg（以及可选 quality 的 cns/yns）

`parse_labels()` 约定了 `data[self.gt_key]` 的格式：每个元素是 `N_gt x 10`，其中：
- `labels[:, 9]` 是类别
- `labels[:, :9]` 是回归 target

每层 loss 的要点（`head.py` 约 L579-L697）：
- 调 `self.target(cls, reg, gt_cls, gt_reg, None)` 得到 `cls_target/reg_target/cls_weights/reg_weights`
- `mask` 用于筛掉无效回归项，并通过 `reduce_mean(sum(mask))` 得到 `num_pos`（DDP 同步）
- 如果配置了 `cls_threshold_to_reg`，还会用分类置信度再筛一遍回归 mask
- `qt`（quality）不为 None 时：
  - `cns_target = exp(-||pos_err||_2)`（越准越接近 1）
  - `yns_target` 用 yaw 的余弦相似度判断是否同向

DN loss：如果 forward 里产生了 `dn_prediction/dn_classification`，`prepare_for_dn_loss()` 会把 `dn_valid_mask` 展平并筛出正样本，再对每个 decoder 层计算一次 dn 的 cls/reg。

---

### 4.2 OM head：SparseMapPerceptionDecoder + MapTRCriterion（在线矢量地图）

配置块：`om_head = dict(type="SparseMapPerceptionDecoder", ..., decoder=..., criterion=..., post_process=...)`

实现位置：
- 总入口：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/sparse_decoder.py`
  - `class SparseMapPerceptionDecoder`

它内部又引用了：

1) **SparseOMOEHead（decoder 网络主体）**
- 配置：`decoder=dict(type="SparseOMOEHead", ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/sparse_head.py`

2) **InstanceBankOE（map anchor bank）**
- 配置：`instance_bank=dict(type="InstanceBankOE", anchor=map_anchor_file, ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/instance_bank.py`

3) **DeformableFeatureAggregationOEv2（特征聚合 v2）**
- 配置：`type="DeformableFeatureAggregationOEv2"`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py`

4) **SparsePoint3DKeyPointsGenerator / SparsePoint3DRefinementModule（关键点采样与 refine）**
- 配置：`kps_generator=dict(type="SparsePoint3DKeyPointsGenerator", ...)`
- 配置：`refine_layer=dict(type="SparsePoint3DRefinementModule", ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/blocks.py`

5) **Criterion：MapTRCriterion（loss/assigner/aux seg）**
- 配置：`criterion=dict(type="MapTRCriterion", assigner=..., loss_cls=..., loss_pts=..., ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/criterion.py`

6) **Assigner：MapTRAssigner**
- 配置：`assigner=dict(type="MapTRAssigner", cls_cost=..., pts_cost=..., ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/assigner.py`

7) **Loss / Cost：PtsL1Loss、PtsDirCosLoss、OrderedPtsL1Cost**
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/map_loss.py`

8) **PostProcess：MapTRPostProcess**
- 配置：`post_process=dict(type="MapTRPostProcess", ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/maptr/postprocess.py`

#### 4.2.0 `SparseMapPerceptionDecoder`：从 raw decoder 输出到 criterion/postprocess 的输入字典

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/maptr/sparse_decoder.py`

这个模块可以视为 MapTR 的“包装器”：
- `self.decoder(...)` 负责网络前向（输出 `classification/prediction` 列表）
- `get_outputs(...)` 负责把点集预测转换成 bbox/pts 两种视角，并拆成 one2one/one2many
- `_post_process(...)` 负责训练/推理分流：训练走 `criterion`，推理走 `post_process`

关键代码路径（省略无关分支）：

```python
outputs = self.sparse_decoder(mlvl_feats, data, lidar_feature, is_deploy=self.is_deploy)
if self.is_deploy or get_value("preparing_qat"):
  return outputs

outputs = self.get_outputs(
  outputs["classification"],
  outputs["prediction"],
  outputs.get("outputs_pv_seg", None),
  outputs.get("dense_depth", None),
)
return self._post_process(data, outputs)
```

`get_outputs()` 里最重要的是 `transform_box(reference)`：
- 先把点集 reshape 成 `(B, num_vec, num_pts, 2)`
- 再按 `transform_method="minmax"` 得到 bbox：`[xmin,ymin,xmax,ymax]`

##### PV aux seg（本配置启用）

你的 `map_aux_seg_cfg` 是 `use_aux_seg=True` 且 `pv_seg=True`，所以 `sparse_decoder()` 会额外算 `outputs_pv_seg`，并把 `(B*num_cam, ...)` 重新 view 回 `(B, num_cam, ...)`：

```python
bs = mlvl_feats[0].shape[0] // self.num_cam
outputs_pv_seg = self.pv_seg_head_layers[i](feats)
outputs_pv_seg = outputs_pv_seg.view(bs, self.num_cam, -1, feat_h, feat_w)
```

#### 4.2.1 MapTR 的 GT 是怎么来的：`vectormap_pipeline()` → `gt_instances/gt_labels_map/(gt_pv_seg_mask)`

很多人读 MapTR loss 时最迷糊的是：**`gt_instances` 到底是什么类型？里面有哪些属性？怎么跟 `gt_shift_pts_pattern` 对上？**
这份工程的答案完全在 dataset 里。

数据生成入口：
- `NuscenesMapDataset.vectormap_pipeline(example)`
  - 文件：`extracted/horizon_torch_samples/hat/data/datasets/nuscenes_map_dataset.py`

它做了这些事：

1) 选择“用 lidar pose 还是 ego pose”来生成 map GT
- 如果 `use_lidar_gt=True`（本配置是 True）：用 `example["lidar2global"]` 推导 `translation/rotation/homo2img`
- 否则用 `example["meta"]["ego2global_translation/rotation"]` 与 `example["ego2img"]`

2) 调 `VectorizedLocalMap.gen_vectorized_samples(...)` 生成矢量 GT
- `self.vector_map` 在 `NuscenesMapDataset.__init__()` 里创建（patch_size 来自 `pc_range`）
- 输出里最重要的字段：
  - `gt_vecs_pts_loc`：`LiDARInstanceLines`（每个 instance 是一条 LineString）
  - `gt_vecs_label`：每条线的类别 label（0/1/2...）
  - `gt_pv_semantic_mask`：若 `aux_seg["pv_seg"]=True`，会生成多尺度、按 camera 投影的 PV seg mask

3) 写回到训练样本字段（供 MapTR head/criterion 消费）
- `example["gt_labels_map"] = to_tensor(gt_vecs_label)`
- `example["gt_instances"] = gt_vecs_pts_loc`（原样保留 `LiDARInstanceLines`，而不是直接 flatten 成 tensor）
- 若存在 PV seg：`example["gt_pv_seg_mask"] = [to_tensor(mask) for mask in gt_pv_semantic_mask]`

`LiDARInstanceLines` 这个类型，是 `MapTRCriterion` 里“所有 GT 几何”的来源：
- `.bbox`：给 bbox 分支/辅助
- `.fixed_num_sampled_points`：给点集回归
- `.shift_fixed_num_sampled_points_*`：给 `gt_shift_pts_pattern`（包括本配置的 `sparsedrive`）

#### 4.2.2 MapTRCriterion：核心 loss 计算流程（按 decoder 层做 deep supervision）

`MapTRCriterion.forward(preds_dicts, data)` 的实现是“可以直接对照源码逐行走完”的典型 MapTR loss：

**(1) 从 batch 里取 GT**
- 入口字段：`data["gt_instances"]`、`data["gt_labels_map"]`
  - 其中 `gt_instances` 在 dataset 里是 `LiDARInstanceLines`（由 `NuscenesMapDataset.vectormap_pipeline()` 写入）

**(2) 把 `LiDARInstanceLines` 展开成三套监督：bbox / pts / shifted-pts**
- `gt_bboxes_list`：用 `LiDARInstanceLines.bbox`
- `gt_pts_list`：用 `LiDARInstanceLines.fixed_num_sampled_points`
- `gt_shifts_pts_list`：由 `gt_shift_pts_pattern` 决定调用哪个属性（本配置是 `sparsedrive`，对应 `shift_fixed_num_sampled_points_sparsedrive`）

**(3) `valid_mask`：过滤空 GT 样本，避免 assign/loss 崩溃**
- 当某个 batch item 的 `gt_instances.instance_list` 为空，会被 `valid_mask=False` 标记
- 随后 `filter_feats()` 会对 `all_cls_scores/all_bbox_preds/all_pts_preds` 以及 GT list 做一致过滤

**(4) decoder 多层 deep supervision：逐层调用 `loss_single()`**
- 预测字典 key（来自 `SparseOMOEHead` 输出）：
  - `all_cls_scores` / `all_bbox_preds` / `all_pts_preds`
  - `enc_cls_scores` / `enc_bbox_preds` / `enc_pts_preds`（如果有 encoder proposal）
- `loss_single()` 内部会：
  - 调 `get_targets()` → `_get_target_single()` → `assigner.assign()` 生成 `labels/bbox_targets/pts_targets`
  - 分类 loss：按 `num_total_pos/neg` 算 `cls_avg_factor`，并用 `reduce_mean()` 做 DDP sync（若启用）
  - 点集回归：先 `normalize_2d_pts()`，必要时对 pred pts 做 `F.interpolate()` 到 `num_pts_per_gt_vec`
  - 方向 loss：在**反归一化后的坐标系**里，用 `dir_interval` 做差分

**(5) loss 字典的 key 命名（看日志/对齐权重非常重要）**
- 最后一层 decoder：`loss_cls/loss_pts/loss_dir`
- 其余层：`d0.loss_cls/d0.loss_pts/d0.loss_dir`、`d1...` 依次类推
- 若启用 `aux_seg`：可能额外出现 `loss_seg`、`loss_pv_seg`
- 若启用 encoder proposal：可能额外出现 `enc_loss_cls/enc_losses_pts/enc_losses_dir`

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/maptr/criterion.py`

> 读代码抓手：建议从 `forward()` 开始，再顺着 `loss_single()`、`_get_target_single()`、`assigner.assign()` 跟下去。

#### 4.2.3 MapTRAssigner：如何做匹配

`MapTRAssigner.assign()` 是标准“cost + Hungarian”的一对一匹配（在 map 任务里尤其关键）：

1) **分类 cost**
- `cls_cost = self.cls_cost(cls_pred, gt_labels)`

2) **点集 cost（带 order / shift 的最小化）**
- 先把 GT pts `normalize_2d_pts(gt_pts, pc_range)`
- 如果 pred 的 `num_pts_per_vec != gt 的 num_pts_per_gt_vec`：对 pred 做 `F.interpolate()` 对齐
- `pts_cost_ordered = self.pts_cost(pts_pred_interpolated, normalized_gt_pts)`
  - 这里的 pts cost 是“带 order 维度”的（shape 会被 reshape 成 `[num_query, num_gt, num_orders]`）
- `pts_cost, order_index = torch.min(pts_cost_ordered, dim=2)`
  - `order_index` 会在 `MapTRCriterion._get_target_single()` 里用来选中正确的 shift 版本

3) **Hungarian（CPU）**
- `cost = (cls_cost + pts_cost).detach().cpu()`
- `linear_sum_assignment(cost)`（依赖 `scipy`）输出匹配 `(query_idx, gt_idx)`

4) **输出给 criterion 的关键信息**
- `assigned_gt_inds`：0 表示负样本，`>0` 表示匹配到的 GT（1-based）
- `assigned_labels`
- `order_index`：让每个正样本选择“哪一种 shift/order 的 GT pts”

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/maptr/assigner.py`

---

### 4.3 Occ head：FlashOccHead（LSS view transformer + BEV encoder + occ decoder）

配置块：`occ_head = dict(type="FlashOccHead", view_transformer=..., bev_encoder=..., bev_decoder=...)`

实现位置：
- 入口：`extracted/horizon_torch_samples/hat/models/task_modules/flashocc/decoder.py`
  - `class FlashOccHead`
  - `class FlashOccDetDecoder`

关键子模块：

1) **ViewTransformer：LSSTransformer**
- 配置：`view_transformer=dict(type="LSSTransformer", ...)`
- 实现：`extracted/horizon_torch_samples/hat/models/task_modules/view_fusion/view_transformer.py`

2) **BevEncoder（BEV 特征编码）**
- 配置：`bev_encoder=dict(type="BevEncoder", backbone=HENet(in_channels=64,...), neck=BiFPN(...))`
- 实现：
  - `BevEncoder`：`extracted/horizon_torch_samples/hat/models/task_modules/view_fusion/encoder.py`
  - `BiFPN`：`extracted/horizon_torch_samples/hat/models/necks/bifpn.py`

3) **BEVOCCHead2D / occ decoder**
- 配置：`bev_decoder=dict(type="FlashOccDetDecoder", occ_head=dict(type="BEVOCCHead2D", ...), loss_occ=...)`
- 实现：
  - `FlashOccDetDecoder`：`extracted/horizon_torch_samples/hat/models/task_modules/flashocc/decoder.py`
  - `BEVOCCHead2D`：`extracted/horizon_torch_samples/hat/models/task_modules/flashocc/bev_occ_head.py`

#### 4.3.0 `FlashOccHead/FlashOccDetDecoder`：forward 与 loss 的实现要点

实现位置：
- `extracted/horizon_torch_samples/hat/models/task_modules/flashocc/decoder.py`

`FlashOccHead.forward()` 的主线是“取一层 PV feature → LSS lift-splat 到 BEV → encoder → decoder”：

```python
feat = feature_maps[self.bev_feat_index]
bev_feat, aux_data = self.view_transformer(feat, metas, compiler_model)
if self.bev_upscale > 1:
  bev_feat = self.resize(bev_feat)
bev_feat, metas = self._transform(bev_feat, metas)
if self.bev_encoder:
  bev_feat = self.bev_encoder(bev_feat, metas)
pred, result = self.bev_decoder([bev_feat] if not isinstance(bev_feat, Sequence) else bev_feat, metas)
```

输出分流：
- `compiler_model=True`：直接返回 `pred`（导出/编译路径）
- `training=True`：返回 loss dict（来自 `FlashOccDetDecoder._post_process`）
- `eval`：返回 `[result["occ_pre"]]`（离散语义体素）

`FlashOccDetDecoder` 的 loss 计算依赖 `data["gt_occ_info"]`：
- `voxel_semantics: (B,Dx,Dy,Dz)`
- `mask_camera/mask_lidar: (B,Dx,Dy,Dz)`
- `occ_pred: (B,Dx,Dy,Dz,n_cls)` 会 reshape 成 `(B*Dx*Dy*Dz, n_cls)` 并按 mask 做 CE

```python
voxel_semantics = voxel_semantics.reshape(-1)
preds = occ_pred.reshape(-1, self.num_classes)
mask_camera = mask_camera.reshape(-1)
num_total_samples = mask_camera.sum()
loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
```

#### 4.3.1 这份配置里 occ GT 到底怎么进 batch：`with_lidar_occ=True` → `gt_occ_info`

本配置 `train_dataset/val_dataset` 明确设置：`with_ego_occ=False, with_lidar_occ=True`。

对应的数据流是：
- `NuscenesBevDataset.__getitem__()`（父类链：`NuscenesSparseMapDataset` → `NuscenesMapDataset` → `NuscenesBevDataset`）
  - 在返回前执行：`data["gt_occ_info"] = sample._get_lidar_occ(data["lidar2global"])`
- `NuscenesSample._get_lidar_occ()` 会：
  - 从 packed sample bytes 里 decode 出 `(200,200,16)` 的 `voxel_semantics/mask_lidar/mask_camera`
  - 计算 `lidar2camego`，并调用 `_get_occ_ego2lidar()` 把 ego 坐标系 occ 映射到 lidar/cam-ego 对齐坐标
  - 返回一个 dict，包含（至少）：
    - `ego_voxel_semantics`
    - `voxel_semantics`
    - `mask_lidar`
    - `mask_camera`
    - `lidar2camego`

实现位置：
- `extracted/horizon_torch_samples/hat/data/datasets/nuscenes_dataset.py`

---

## 5. 配置文件：从每个 `type` 定位到实现代码

这一节把配置文件里所有形如 `type="..."` 的模块，逐个定位到 `hat/` 解压源码的**定义位置（文件 + 行号）**，便于“按图索骥”读代码。

### 5.1 配置中所有 `type` → 实现位置（自动抽取）

<details>
<summary>展开查看（共 81 个 `type`）</summary>

说明：这里的实现位置以 OE3.7 工程根目录为基准（即文中约定的 `extracted/horizon_torch_samples/hat/...`）。

#### models
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `AsymmetricFFNOE` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py:900` |
| `BevEncoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/view_fusion/encoder.py:19` |
| `BEVOCCHead2D` | class | `extracted/horizon_torch_samples/hat/models/task_modules/flashocc/bev_occ_head.py:36` |
| `BiFPN` | class | `extracted/horizon_torch_samples/hat/models/necks/bifpn.py:421` |
| `CrossEntropyLoss` | class | `extracted/horizon_torch_samples/hat/models/losses/cross_entropy_loss.py:102` |
| `DeformableFeatureAggregationOE` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py:171` |
| `DeformableFeatureAggregationOEv2` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py:997` |
| `DenseDepthNetOE` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/blocks.py:728` |
| `FlashOccDetDecoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/flashocc/decoder.py:22` |
| `FlashOccHead` | class | `extracted/horizon_torch_samples/hat/models/task_modules/flashocc/decoder.py:195` |
| `Float2Calibration` | class | `extracted/horizon_torch_samples/hat/models/model_convert/converters.py:292` |
| `Float2QAT` | class | `extracted/horizon_torch_samples/hat/models/model_convert/converters.py:94` |
| `FocalLoss` | class | `extracted/horizon_torch_samples/hat/models/losses/focal_loss.py:18` |
| `GaussianFocalLoss` | class | `extracted/horizon_torch_samples/hat/models/losses/focal_loss.py:303` |
| `HbirModule` | class | `extracted/horizon_torch_samples/hat/models/ir_modules/hbir_module.py:32` |
| `HbmModule` | class | `extracted/horizon_torch_samples/hat/models/ir_modules/hbm_module.py:48` |
| `HENet` | class | `extracted/horizon_torch_samples/hat/models/backbones/henet.py:19` |
| `InstanceBankOE` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/instance_bank.py:48` |
| `L1Loss` | class | `extracted/horizon_torch_samples/hat/models/losses/l1_loss.py:14` |
| `LoadCheckpoint` | class | `extracted/horizon_torch_samples/hat/models/model_convert/ckpt_converters.py:18` |
| `LSSTransformer` | class | `extracted/horizon_torch_samples/hat/models/task_modules/view_fusion/view_transformer.py:369` |
| `MapTRAssigner` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/assigner.py:75` |
| `MapTRCriterion` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/criterion.py:97` |
| `MapTRPostProcess` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/postprocess.py:166` |
| `MemoryBankOE` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/memory_bank.py:21` |
| `MMFPN` | class | `extracted/horizon_torch_samples/hat/models/necks/mm_fpn.py:17` |
| `ModelConvertPipeline` | class | `extracted/horizon_torch_samples/hat/models/model_convert/pipelines.py:21` |
| `OrderedPtsL1Cost` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/map_loss.py:364` |
| `PtsDirCosLoss` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/map_loss.py:221` |
| `PtsL1Loss` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/map_loss.py:268` |
| `SimpleLoss` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/map_loss.py:406` |
| `SparseBevFusionMultitaskOE` | class | `extracted/horizon_torch_samples/hat/models/structures/sparse_multitask.py:33` |
| `SparseBEVFusionMultitaskOEIrInfer` | class | `extracted/horizon_torch_samples/hat/models/structures/sparse_multitask.py:160` |
| `SparseBEVOEDecoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/decoder.py:11` |
| `SparseBEVOEEncoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/det_blocks.py:234` |
| `SparseBEVOEHead` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/head.py:27` |
| `SparseBEVOEKeyPointsGenerator` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/det_blocks.py:155` |
| `SparseBEVOERefinementModule` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/det_blocks.py:22` |
| `SparseBEVOETarget` | class | `extracted/horizon_torch_samples/hat/models/task_modules/sparsebevoe/target.py:14` |
| `SparseMapPerceptionDecoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/sparse_decoder.py:22` |
| `SparseOEPoint3DEncoder` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/blocks.py:21` |
| `SparseOMOEHead` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/sparse_head.py:20` |
| `SparsePoint3DKeyPointsGenerator` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/blocks.py:50` |
| `SparsePoint3DRefinementModule` | class | `extracted/horizon_torch_samples/hat/models/task_modules/maptr/blocks.py:181` |

#### data
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `BevBBoxRotation` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:628` |
| `BgrToYuv444` | class | `extracted/horizon_torch_samples/hat/data/transforms/common.py:553` |
| `DistStreamBatchSampler` | class | `extracted/horizon_torch_samples/hat/data/samplers/dist_stream_sampler.py:48` |
| `MultiViewsGridMask` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:511` |
| `MultiViewsImgCrop` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:210` |
| `MultiViewsImgFlip` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:295` |
| `MultiViewsImgResize` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:92` |
| `MultiViewsImgRotate` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:391` |
| `MultiViewsImgTransformWrapper` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:65` |
| `MultiViewsPhotoMetricDistortion` | class | `extracted/horizon_torch_samples/hat/data/transforms/multi_views.py:530` |
| `Normalize` | class | `extracted/horizon_torch_samples/hat/data/transforms/detection.py:1000` |
| `NuscenesFromImage` | class | `extracted/horizon_torch_samples/hat/data/datasets/nuscenes_dataset.py:2928` |
| `NuscenesSparseMapDataset` | class | `extracted/horizon_torch_samples/hat/data/datasets/nuscenes_map_dataset.py:1964` |
| `PILToTensor` | class | `extracted/horizon_torch_samples/hat/data/transforms/common.py:225` |

#### engine
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `BasicBatchProcessor` | class | `extracted/horizon_torch_samples/hat/engine/processors/processor.py:194` |
| `Calibrator` | class | `extracted/horizon_torch_samples/hat/engine/calibrator.py:26` |
| `distributed_data_parallel_trainer` | launcher | `extracted/horizon_torch_samples/hat/engine/ddp_trainer.py:445` |
| `MultiBatchProcessor` | class | `extracted/horizon_torch_samples/hat/engine/processors/processor.py:491` |
| `Predictor` | class | `extracted/horizon_torch_samples/hat/engine/predictor.py:27` |

#### callbacks
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `Checkpoint` | class | `extracted/horizon_torch_samples/hat/callbacks/checkpoint.py:119` |
| `CosineAnnealingLrUpdater` | class | `extracted/horizon_torch_samples/hat/callbacks/lr_updater.py:738` |
| `ExponentialMovingAverage` | class | `extracted/horizon_torch_samples/hat/callbacks/exponential_moving_average.py:23` |
| `FreezeBNStatistics` | class | `extracted/horizon_torch_samples/hat/callbacks/freeze_bn.py:12` |
| `GradClip` | class | `extracted/horizon_torch_samples/hat/callbacks/grad_scale.py:17` |
| `MetricUpdater` | class | `extracted/horizon_torch_samples/hat/callbacks/metric_updater.py:207` |
| `StatsMonitor` | class | `extracted/horizon_torch_samples/hat/callbacks/monitor.py:22` |
| `Validation` | class | `extracted/horizon_torch_samples/hat/callbacks/validation.py:26` |

#### metrics
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `LossShow` | class | `extracted/horizon_torch_samples/hat/metrics/loss_show.py:17` |
| `MeanIOU` | class | `extracted/horizon_torch_samples/hat/metrics/mean_iou.py:19` |
| `NuscenesMapMetric` | class | `extracted/horizon_torch_samples/hat/metrics/nuscenes_map_metric.py:29` |
| `NuscenesMetric` | class | `extracted/horizon_torch_samples/hat/metrics/nuscenes_metric.py:69` |

#### visualize
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `NuscenesMapViz` | class | `extracted/horizon_torch_samples/hat/visualize/nuscenes_map.py:21` |
| `NuscenesMultitaskViz` | class | `extracted/horizon_torch_samples/hat/visualize/nuscenes.py:197` |
| `OccViz` | class | `extracted/horizon_torch_samples/hat/visualize/occ.py:48` |

#### profiler
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `QuantAnalysis` | class | `extracted/horizon_torch_samples/hat/profiler/quant_analysis.py:26` |

#### utils
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `HbirExporter` | class | `extracted/horizon_torch_samples/hat/utils/hbdk4/hbir_exporter.py:29` |

#### core
| Config `type` | Kind | 实现位置 |
|---|---|---|
| `FocalLossCost` | class | `extracted/horizon_torch_samples/hat/core/match_costs/match_cost.py:36` |

</details>

---

## 6. 训练/校准/QAT/导出：配置里“trainer/predictor/exporter”怎么串起来

该配置不只定义了 `model`，还定义了多套 runner：

- `float_trainer`：浮点训练
- `calibration_trainer`：量化校准
- `float_predictor` / `calibration_predictor`：评估/推理
- `hbir_exporter`：导出 HBIR
- `compile_cfg`：编译为 HBM
- `hbir_infer_model`：IR 推理封装
- `quant_analysis_solver`：量化敏感性分析

### 6.1 训练入口脚本

训练工具：
- `scripts/tools/train.py`

它要求你指定 stage（对应配置里的 `{stage}_trainer`）：

```bash
conda activate apollo_vnet
cd /home/nuvo/Downloads/OE3.7/horizon_j6_open_explorer_v3.7.0-py310_20251215
python samples/ai_toolchain/horizon_model_train_sample/scripts/tools/train.py \
  --stage float \
  --config samples/ai_toolchain/horizon_model_train_sample/scripts/configs/bev/bev_sparse_det_maptr_flashocc_henet_tinym_nuscenes.py \
  --device-ids 0,1,2,3
```

实现位置：
- `scripts/tools/train.py`

### 6.2 DDP trainer 的实现在哪里？

配置里 `float_trainer = dict(type="distributed_data_parallel_trainer", ...)`。

实现位置：
- `extracted/horizon_torch_samples/hat/engine/ddp_trainer.py`
  - `class DistributedDataParallelTrainer`

### 6.3 ModelConvertPipeline / Float2QAT / Float2Calibration

配置里多处出现：
- `model_convert_pipeline=dict(type="ModelConvertPipeline", converters=[...])`
- `Float2QAT` / `Float2Calibration` / `LoadCheckpoint`

实现位置：
- `extracted/horizon_torch_samples/hat/models/model_convert/pipelines.py`
- `extracted/horizon_torch_samples/hat/models/model_convert/converters.py`

这部分负责：
- 加载 checkpoint
- 将 float 模型转换到 calibration 或 QAT 图
- 应用 `qconfig_setter` 做量化配置

### 6.4 训练/评估/可视化/导出：其余 `type` 的“职责划分”

这些模块的实现位置都已在 [5.1](#51-配置中所有-type--实现位置自动抽取) 给出；这里补一份“看到配置就知道该去哪读”的职责速查：

- Loop/Runner：`Predictor`（推理/评估循环）、`Calibrator`（量化校准循环）、`distributed_data_parallel_trainer`（DDP 训练 launcher）
- BatchProcessor：`BasicBatchProcessor`（单 batch train/eval 基本逻辑）、`MultiBatchProcessor`（多 batch/多任务封装）
- Callbacks：
  - `Checkpoint`（保存/加载 ckpt）、`Validation`（按 interval 跑 val）、`MetricUpdater`（把 metric 更新挂进 loop）
  - `CosineAnnealingLrUpdater`（lr schedule）、`GradClip`（梯度裁剪）、`FreezeBNStatistics`（冻结 BN 统计）
  - `ExponentialMovingAverage`（EMA）、`StatsMonitor`（日志/监控）
- Metrics：`NuscenesMetric`（检测评估）、`NuscenesMapMetric`（矢量地图评估）、`MeanIOU`（占用/语义 iou）、`LossShow`（loss 汇总展示）
- Viz：`NuscenesMultitaskViz`（三任务可视化）、`NuscenesMapViz`（地图可视化）、`OccViz`（占用可视化）
- Quant/Export：`QuantAnalysis`（量化敏感性分析）、`HbirExporter`（导出 HBIR）、`HbirModule` / `HbmModule`（IR 模块封装）

---

## 7. 建议的阅读路径（想把代码吃透就按这个走）

1) 先通读配置，明确三头的输入输出与 loss 入口：
- `scripts/configs/bev/bev_sparse_det_maptr_flashocc_henet_tinym_nuscenes.py`

2) 顺着模型 forward：
- `SparseBevFusionMultitaskOE.forward`（structures）
- det head：`SparseBEVOEHead`
- map head：`SparseMapPerceptionDecoder` → `SparseOMOEHead` → `MapTRCriterion`
- occ head：`FlashOccHead` → `LSSTransformer` → `FlashOccDetDecoder`

3) 最后再看数据：
- `NuscenesMapDataset` / `NuscenesSparseMapDataset`
- occ3d packer/dataset

---

## 8. 你接下来可以让我补全的内容

这篇文章已经把“配置→实现代码位置”全部定位出来，并把三大任务 head 的关键执行链条讲清楚。

如果你希望我再把文章进一步“补全到可以直接照着跑”（包括：`anno-file` 的格式说明、LMDB 生成的完整命令、以及 `BEVOCCHead2D` 等尚未在文中展开的类的逐段走读），我可以继续把缺的实现类都 grep 出来并补齐到同一篇文档里。