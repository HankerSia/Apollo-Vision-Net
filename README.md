前言：本文围绕 **det（3D 检测）** 与 **map（MapTR 实时地图生成）** 两个感知任务并行模型设计，系统梳理 Apollo-Vision-Net 中 det+map 多任务模型的“设计 → 配置 → 源码实现 → 评测闭环 → 复现结果”。目标是让读者能够对照文档完成模型复现、测试，并明确每个 cfg 字段对应的源码位置、推理输出的接口规范与设计理由。

## 目录

- [0.复现准备（环境、数据、权重）](#section-0-prereq)
- [1.目标、边界与关键工程决策](#section-1-goals)
- [2.总体设计：det 与 map 的数据流与输出接口与格式规范](#section-2-contract)
- [3.配置设计：如何在 cfg 中表达 det+map（以及如何映射到代码）](#section-3-config)
- [4.实现改动：按模块拆解实际代码如何落地](#section-4-impl)
- [5.评测协议与落盘产物：bbox 与 map 各自怎么评、产物在哪里](#section-5-eval)
- [6.复现命令与测试结果（含 2026-02-09 离线复测）](#section-6-tests)
- [7.常见坑与排障清单](#section-7-debug)
- [8.后续迭代路线](#section-8-roadmap)
- [9.工程化对账表与回归清单](#section-9-audit)

---

<a id="section-0-prereq"></a>
## 0.复现准备（环境、数据、权重）

这一节的目标是：让**第一次接触本工程**的人也能按步骤把 det+map 多任务链路跑通。

### 0.1 代码与环境（建议版本）

coda环境名不限，下文用 `apollo_vnet` 作为示例。我自己环境实际安装步骤如下：
```bash
conda create -n apollo_vnet python=3.8

conda activate apollo_vnet
export PYTHONPATH=$PYTHONPATH:"/" #报错ModuleNotFoundError: No module named 'tools'时执行
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

pip install mmdet==2.14.0

pip install mmsegmentation==0.14.1

# 源码安装mmdet3d-v0.17.1版本
cd bevformer
# 下载mmdetection3d   github加速代理https://ghproxy.com/
git clone https://github.com/open-mmlab/mmdetection3d.git
# 进入mmdetection3d目录
cd mmdetection3d
# 切换v0.17.1
git checkout v0.17.1
# 安装mmdet3d-v0.17.1版本
python setup.py install

pip install numba==0.48.0

pip install numpy==1.21.0 # 报错 numpy has no module "long"时执行

pip install lyft_dataset_sdk
```
如果你本机已经安装了 mmdet3d（或正在使用 MapTR 仓库内的 mmdet3d），建议先打印一次实际 import 的来源，避免“import 到了另一个 mmdet3d”导致的 registry/PIPELINES 不一致：

```bash
python -c "import mmdet3d; print('mmdet3d from:', mmdet3d.__file__)"
```

### 0.2 数据准备：nuScenes + can_bus（必需）

将 nuScenes v1.0 数据与 can_bus 按如下组织（与 `docs/prepare_dataset.md` 一致）：

```
Apollo-Vision-Net
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
```

### 0.3 生成时序 infos（必需）：nuscenes_infos_temporal_{train,val}.pkl

det+map 配置默认使用时序 infos（`nuscenes_infos_temporal_train.pkl / nuscenes_infos_temporal_val.pkl`）。若你还没有这些文件，执行：

```bash
conda activate apollo_vnet
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data
```

执行完成后应至少包含：

- `data/nuscenes/nuscenes_infos_temporal_train.pkl`
- `data/nuscenes/nuscenes_infos_temporal_val.pkl`

### 0.4 MapTR 风格训练数据（GT）如何生成：在线矢量化（训练时实时生成）

本工程的 det+map 配置**不要求你提前离线生成一份“maptr 训练标注 pkl/json”**。

训练阶段 map GT 的生成逻辑是：

- **输入**：nuScenes 原始地图（`data/nuscenes/maps`）+ 每个 sample 的位姿（`lidar2ego_*` / `ego2global_*`）+ 当前场景的地图 location（`map_location` 或可由 `scene_name` 推导）。
- **处理**：以 `pc_range` 推导的 `patch_size` 为裁剪窗口，在全局地图中裁剪并提取 divider/ped_crossing/boundary 三类几何，再变换到 LiDAR 局部坐标系。
- **输出（写回 sample）**：
  - `gt_map_vecs_label`：每条 polyline 的类别（0/1/2）
  - `gt_map_vecs_pts_loc`：每条 polyline 的点序列（在评测/训练协议里会重采样为固定点数）

对应源码入口（强烈建议读一遍，避免“照文档跑但不理解 GT 从哪来”）：

- dataset：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`
  - `prepare_train_data(...)`：在 pipeline 完成后调用 `self._add_vectormap_gt(example, input_dict)` 把 map GT 注入到样本。
  - `_add_vectormap_gt(...)`：计算 `lidar2global`，然后调用 `self.vector_map.gen_vectorized_samples(...)` 完成在线矢量化。

简化版关键代码（与当前实现一致，便于你定位调试点）：

```python
# projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py
example = self.pipeline(input_dict)
example = self._add_vectormap_gt(example, input_dict)
```

补充说明（避免被 pipeline 顺序误导）：

- 由于注入发生在 `self.pipeline(...)` 之后，`gt_map_vecs_*` 可能并不是由 pipeline 的 `Collect3D/CustomCollect3D` “收集出来”的，而是 dataset 在返回前**额外补充**到 `example` 里。
- 因此：
  - 如果你的收集算子是严格版（缺 key 直接 `KeyError`），不要在 `Collect3D/CustomCollect3D.keys` 里硬性要求 `gt_map_vecs_*`；
  - 本工程使用的 `CustomCollect3D` 允许 keys 缺失跳过（见 `docs/changes/2026-01-15-det-map-port.md` 的说明），用于兼容这种“pipeline 后注入”的在线 GT。

```python
# projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py
anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
example['gt_map_vecs_label'] = ...
example['gt_map_vecs_pts_loc'] = ...
```

#### 0.4.1 在线 GT 生成对 infos 的字段要求（不满足会直接报错）

`_add_vectormap_gt(...)` 至少需要以下字段（来自 `get_data_info(index)` 或 `data_infos[index]`）：

- 地图 location：`map_location`（优先）或 `scene_name`（可推导为真实 location）
- 位姿：`lidar2ego_rotation` / `lidar2ego_translation` / `ego2global_rotation` / `ego2global_translation`

如果你遇到类似报错：

- `Missing map_location/scene_name in input_dict`：说明 infos 不包含 location 信息，通常需要重新生成 infos 或检查 converter。
- `Unknown nuScenes map location`：说明 location 不在 nuScenes 支持的 4 个地图名里（boston-seaport / singapore-xxx）。

#### 0.4.2 固定点数协议：fixed_ptsnum_per_line

cfg 里的 `fixed_ptsnum_per_line`（例如 20）不仅影响 pred 的格式，也影响 GT 的重采样协议。在线矢量化得到的 shapely LineString 会被重采样为固定点数，保证训练/评测的输入输出可对齐。

### 0.5 权重准备（必需）

本文的 smoke 配置 `projects/configs/bevformer/bev_tiny_det_map_apollo.py` 引用了预训练权重与待评测 checkpoint：

- 预训练权重（示例路径）：
  - `ckpts/depth_pretrained_dla34-y1urdmir-20210422_165446-model_final-remapped_bev.pth`
- 待评测 checkpoint：
  - 你可以用自己的训练产物（如 `work_dirs/.../epoch_1.pth`），也可以复用已有实验目录下的 checkpoint。

如果缺少权重，`tools/test.py` 会在 `load_checkpoint` 阶段直接报错；这是最常见的“照文档跑不起来”的原因之一。

### 0.6 快速自检（推荐）

在正式跑 `dist_test` 前，建议先确认 cfg 能 build dataset 与 model（排除 registry 未注册、数据路径错误）：

```bash
conda activate apollo_vnet
python -c "from mmcv import Config; from mmdet3d.datasets import build_dataset; from mmdet3d.models import build_model; cfg=Config.fromfile('projects/configs/bevformer/bev_tiny_det_map_apollo.py'); ds=build_dataset(cfg.data.val); m=build_model(cfg.model, test_cfg=cfg.get('test_cfg')); print(type(ds), 'evaluate_map' in dir(ds), type(m))"
```

额外建议：在训练前先验证一次“在线 map GT 是否真的生成且被 pipeline 收集到”。

```bash
conda activate apollo_vnet
python - <<'PY'
from mmcv import Config
from mmdet3d.datasets import build_dataset

cfg = Config.fromfile('projects/configs/bevformer/bev_tiny_det_map_apollo.py')
ds = build_dataset(cfg.data.train)
sample = ds[0]

keys = list(sample.keys())
print('sample keys:', keys)
assert 'gt_map_vecs_label' in sample, 'missing gt_map_vecs_label'
assert 'gt_map_vecs_pts_loc' in sample, 'missing gt_map_vecs_pts_loc'

lab = sample['gt_map_vecs_label'].data
pts = sample['gt_map_vecs_pts_loc'].data
print('gt_map_vecs_label type:', type(lab))
print('gt_map_vecs_pts_loc type:', type(pts))

if hasattr(pts, 'fixed_num_sampled_points'):
  t = pts.fixed_num_sampled_points
  print('fixed_num_sampled_points:', t.shape)
else:
  print('gt_map_vecs_pts_loc tensor shape:', getattr(pts, 'shape', None))
PY
```

若提示 dataset/head “not in registry”，优先检查 cfg 是否启用了 `plugin=True`，以及你的 `PYTHONPATH` 是否包含仓库根目录。

<a id="section-1-goals"></a>
## 1.目标、边界与关键工程决策

### 1.1 目标

- 默认多任务：同一次推理同时产出 det 与 map 两条结果。
- 同一次测试闭环：同一次 `dist_test` 同时跑通
  - nuScenes bbox evaluator（NDS/mAP/ATE/ASE/AOE/AVE/AAE 等）
  - MapTR 协议 map evaluator（Chamfer mAP + IoU mAP）

### 1.2 非目标（当前阶段不做）

- 不追求 map 任务的最终精度：当前以“链路闭环 + 协议对齐”为第一优先级。
- 不一次性引入 MapTR 全量 head/assigner/loss/CUDA op：先以 scaffold 骨架保证可训练、可推理、可评测。

### 1.3 关键工程决策（必须遵守）

1) **`bbox_results` 与 `map_results` 必须严格分离**
- bbox evaluator 只接受检测结构（`boxes_3d/scores_3d/labels_3d`），map 的 dict 混进来就会导致 bbox evaluator 误解析并报错（典型：`KeyError: 'boxes_3d'`）。

2) 默认就是 det+map；不再保留“stage 分支”
- 推理层面默认返回 `{'bbox_results': ..., 'map_results': ...}`。
- 是否启用 det/map 分支由 head 内部开关控制（cfg 中 `enabling_det/enabling_map`），而不是靠外层脚本分 stage。

3) 离线评测是否可靠，关键在 registry，不在 mmdet3d 来源
- 离线失败最常见原因：cfg 的 plugin/custom_imports 没有执行，导致自定义 dataset/head 没注册。
- 所以离线入口必须复刻 `tools/test.py` 的 plugin 导入逻辑（本工程用 `tools/eval_map_offline.py` 固化）。

运行约束：本文所有命令默认以激活你的 conda 环境为前置（下文示例为 `conda activate apollo_vnet`）。

### 1.4 相较 Apollo-Vision-Net baseline 的改动总览（按闭环拆解）

这里的 “baseline” 指 Apollo-Vision-Net 里 **仅 det（或 det+occ）** 能跑通测试闭环，但 **map（向量地图）** 还未形成“推理产物 → 落盘 → 评测”的完整链路的状态。

- **输出接口规范层（首要）**
  - BEVFormer 推理返回值从“单一 bbox 结果”升级为“bbox_results + map_results 并行返回”。
  - 强制 `bbox_results` 与 `map_results` 分离，避免 bbox evaluator 误解析 map dict。

- **DDP 收集层（multi-gpu test collect）**
  - DDP 测试收集逻辑扩展为支持 `map_results` 的独立收集，避免与 bbox/occ/mask 的 tmpdir 分片互相覆盖。

- **测试入口层（tools/test.py）**
  - 增加 `map_results.pkl` 的 dump（与 bbox 结果分开落盘）。
  - 当 `--eval` 包含 `chamfer/iou` 时，调用 dataset 的 `evaluate_map(...)` 完成 MapTR 协议评测。

- **Dataset & Evaluator 层（MapTR protocol）**
  - Dataset 增加 `evaluate_map/format_map_results/_format_map_gt`：
    - pred：把 `map_results` 统一写成 MapTR 兼容的 `nuscmap_results.json`。
    - gt：必要时自动生成 `nuscenes_map_anns_val.json`。
  - 引入/复用 `map_utils` 的 chamfer/iou mAP 计算协议（阈值、match 规则、加速策略）。

- **配置层（cfg）**
  - 新增 det+map 的 smoke 配置 `projects/configs/bevformer/bev_tiny_det_map_apollo.py`。
  - 通过 head 内部开关 `enabling_det/enabling_map` 控制分支启用，不再靠外层 stage 分支。

- **离线复测入口（可复现性）**
  - 新增离线 map 评测脚本 `tools/eval_map_offline.py`，固化“plugin import/registry 注册”的前置条件，避免离线评测随机失败。

<a id="section-2-contract"></a>
## 2.总体设计：det 与 map 的数据流与输出接口与格式规范

### 2.1 概览：数据流

BEVFormer detector 负责把同一次 forward 的输出拆成两条并行产物：

- det：走原有 `get_bboxes` → 生成 nuScenes evaluator 需要的结构
- map：调用 head 的 `get_map_results` → 生成 MapTR evaluator 能格式化的结构

### 2.2 推理输出接口规范（关键）

BEVFormer 在推理阶段返回：

```python
({
  'bbox_results': bbox_results,   # list[dict(pts_bbox=...)] 或 None
  'map_results': map_results,     # list[dict(vectors,scores,labels,...)] 或 None
}, occ_results)
```

其中：

- `bbox_results`
  - 维持 MMDet3D/nuScenes evaluator 的传统结构。
  - 下游只允许 bbox evaluator 读取它。
- `map_results`
  - 与 `bbox_results` 并行、同样按 sample 对齐。
  - 最小字段：
    - `vectors`: `(N, P, 2)` 的 polyline 点序列（P 通常为 `fixed_ptsnum_per_line`）
    - `scores`: `(N,)`（由 `sigmoid(map_cls_logits).max(-1)` 得到）
    - `labels`: `(N,)`（同上，取 argmax 类别）
    - 可选：`cls_logits`

备注：当前阶段 map 分支是 scaffold，`vectors` 的坐标语义还没有严格对齐 MapTR（例如是否映射到米制坐标系、是否裁剪到 pc_range 等），所以指标偏低是预期现象；本文关注的是链路闭环与协议对齐。

### 2.3 为什么这个契约能保证“评测闭环不互相污染”

- bbox evaluator 只看 `bbox_results`，并且读取的是固定 key（`boxes_3d/...`）。
- map evaluator 只看 `map_results`，并且由 dataset 的 `format_map_results` 写成 MapTR json，再交给 `map_utils` 计算。
- 因为两条链路从 detector 返回时就分开，所以不会出现“map dict 被当 bbox dict 解析”的问题。

<a id="section-3-config"></a>
## 3.配置设计：如何在 cfg 中表达 det+map（以及如何映射到代码）

本次闭环使用的配置：`projects/configs/bevformer/bev_tiny_det_map_apollo.py`。

### 3.1 cfg 里最重要的 3 类配置

1) plugin 与 registry（不然很多类找不到）

- `plugin=True`
- `plugin_dir='projects/mmdet3d_plugin/'`

2) 模型结构（det+map head + 双 decoder）

- `model.type='BEVFormer'`
- `model.pts_bbox_head.type='BEVFormerDetMapHeadApollo'`
- `model.pts_bbox_head.transformer.det_decoder=...`
- `model.pts_bbox_head.transformer.map_decoder=...`（`type='MapTRDecoder'`）
- `model.pts_bbox_head.enabling_det=True` / `enabling_map=True`

3) dataset 与 map 评测需要的上下文

- `dataset_type='CustomNuScenesDetMapDataset'`
- `fixed_ptsnum_per_line=20`（点数协议）
- `queue_length=3`（时序 BEV）

### 3.2 配置 → 代码映射（你改 cfg 时最常用的对照表）

- `model.type='BEVFormer'`
  - 代码：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
  - 作用：推理时拆分并返回 `bbox_results/map_results`。

- `model.pts_bbox_head.type='BEVFormerDetMapHeadApollo'`
  - 代码：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`
  - 作用：det 分支复用 `BEVFormerHead`，map 分支提供 decoder 优先、MLP fallback，并实现 `get_map_results`。

- `transformer.map_decoder.type='MapTRDecoder'`
  - 代码：`projects/mmdet3d_plugin/maptr/modules/decoder.py`
  - 作用：让 map decoder 至少“能 build、能跑通接口”，当前阶段不依赖上游 CUDA op。

- `dataset_type='CustomNuScenesDetMapDataset'`
  - 代码：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`
  - 作用：提供 `evaluate_map/format_map_results/_format_map_gt`，把 `map_results` 评成 MapTR 协议的 mAP。

### 3.3 新增/关键 cfg 字段清单（改配置前先看这个）

- **detector 级**
  - `model.only_det=False`：允许 forward 同时产出 det + map（否则 map 分支不会走到）。
  - `model.video_test_mode=True` / `queue_length=3`：时序 BEV；影响 `prev_bev` 缓存逻辑。

- **head 级（det+map 分支开关 + 输出规模）**
  - `model.pts_bbox_head.enabling_det/enabling_map`：控制 det/map 分支是否启用。
  - `fixed_ptsnum_per_line=20`：map 评测协议中的每条 polyline 固定采样点数（同时影响 GT 生成与 pred 格式化）。
  - （可选）`map_num_classes/map_num_pts/num_map_vec`：当前 scaffold 支持这些参数（不配则走默认值）。

- **transformer 级（双 decoder）**
  - `transformer.det_decoder`：用于 det。
  - `transformer.map_decoder`：用于 map（`type='MapTRDecoder'`）。当前实现会“能 build 则用；失败就 fallback 到 MLP”。

- **dataset 级（map evaluator 的关键上下文）**
  - `dataset_type='CustomNuScenesDetMapDataset'`：别名类，实际继承 `CustomNuScenesDetOccMapDataset`。
  - `map_eval_nproc`：map 评测多进程数（默认 8；离线复测可用 `--nproc 0` 强制单进程排障）。
  - `pc_range/map_ann_file/eval_use_same_gt_sample_num_flag`：控制 MapTR 协议评测行为与 GT json 的位置。

### 3.3 训练相关：为什么要 `find_unused_parameters=True`

- det+map 多任务 + scaffold 阶段，某些参数可能在某些 iteration 不参与反向（例如 map decoder 构建失败时走 MLP fallback，或某些 loss 暂时占位）。
- DDP 默认会把“未使用参数”当成错误，因此 cfg 中开启 `find_unused_parameters=True` 能让 smoke 阶段更稳。

<a id="section-4-impl"></a>
## 4.实现改动：按模块拆解实际代码如何落地

这一节不追求把每行代码抄出来，而是把“改动点 → 关键函数 → 输入输出 → 作用”讲清楚，方便你二次开发时快速下手。

### 4.1 detector：BEVFormer 推理返回 det+map

- 文件：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
- 关键函数：
  - `forward_test(...)`：调用 `simple_test(...)` 后返回 `{'bbox_results': ..., 'map_results': ...}`。
  - `simple_test_pts(...)`：
    - det：`self.pts_bbox_head.get_bboxes(...)` → `bbox3d2result(...)`
    - map：若 head 存在 `get_map_results` 则调用；异常只打印一次并回退为 `None`，防止刷屏导致日志不可读。

说明：`forward_test(...)` 的真实返回形态是：

- `bbox_results`：`simple_test(...)` 内部把每个 sample 的 `pts_bbox` 包装成 `dict(pts_bbox=...)`，所以最终是 `list[dict(pts_bbox=...)]`。
- `map_results`：由 head 的 `get_map_results(outs, img_metas)` 返回，结构为 `list[dict(vectors,scores,labels,...)]`。

### 4.2 DDP 测试收集：把 map 当成独立产物收集

- 文件：`projects/mmdet3d_plugin/bevformer/apis/test.py`
- 关键点：
  - `custom_multi_gpu_test(...)` 识别 model 返回的 dict，并把 `map_results` 收集到独立 list。
  - CPU collect 时为不同产物使用不同的 tmpdir 后缀，避免与 bbox/mask/occ/flow 的分片结果混写。
    - 注意：当前实现会在 tmpdir 名上按序追加后缀（例如先 `_mask` 再 `_map`），因此最终目录名可能呈现为 `..._mask_map` 之类的组合形式。

### 4.3 测试入口：dump map_results.pkl + 按需触发 map evaluator

- 文件：`tools/test.py`
- 关键点：
  - `map_results.pkl` 单独落盘到 `jsonfile_prefix/map_results.pkl`。
  - `--eval` 中包含 `chamfer/iou` 才会调用 `dataset.evaluate_map(...)`。
  - bbox 与 map 各自走各自 evaluator，不互相干扰。

### 4.4 dist_test 脚本：不再覆盖用户显式传入的 --eval

- 文件：`tools/dist_test.sh`
- 关键点：
  - 扫描 `${@:4}` 是否含 `--eval`。
  - 只有未显式提供时才注入默认 `--eval iou bbox`。

### 4.5 dataset：把 map_results 写成 MapTR json 并评测

- 文件：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`
- 关键方法：
  - `format_map_results(results, jsonfile_prefix)`：
    - 逐 sample 写 `nuscmap_results.json`，并把 `labels` 映射到 `divider/ped_crossing/boundary`。
  - `evaluate_map(results, metric, jsonfile_prefix, ...)`：
    - 调用 `format_map_results` 得到 pred json
    - 调用 `_evaluate_map_single(...)` 用 `map_utils` 计算 AP/mAP。
  - `_format_map_gt()`：
    - 当 `map_ann_file` 不存在时自动生成 GT json（默认 `data/nuscenes/nuscenes_map_anns_val.json`）。

说明：以下为工程上需注意的两个关键细节：

- **样本顺序对齐要求**：`format_map_results` 通过 `enumerate(results)` 读取 `self.data_infos[sample_id]['token']` 生成 `sample_token`，因此 `results` 必须与数据集 / dataloader 的样本顺序严格一致（本工程的 DDP 收集逻辑已保证此点）。

- **map_results 的输入容错**：dataset 在 `_map_det_to_vector_list` 中做统一归一化，支持两种输入形式：
  1) `{'vectors': (N,P,2), 'scores': (N,), 'labels': (N,)}`
  2) `{'vectors': [ {'pts':..., 'label':..., 'score':...}, ... ]}`
  若缺失字段或结果为空，函数会返回空列表以保证评测过程不会中断。

### 4.6 MapTR evaluator：map_utils（Chamfer/IoU mAP）

- 目录：`projects/mmdet3d_plugin/datasets/map_utils/`
- 关键点：
  - Chamfer 阈值 `[0.5, 1.0, 1.5]`。
  - IoU 阈值 `0.5..0.95 step 0.05`（polyline buffer 后求 IoU）。
  - 按置信度排序，一个 GT 只匹配一次。
  - multiprocessing 的 worker 需要可 pickling（已按顶层函数规避常见报错）。

### 4.7 det+map head：先保证闭环，再逐步替换为完整 MapTR

- 文件：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`
- 核心思路：
  - det：继承 `BEVFormerHead`，最大化复用现有 det 训练/推理。
  - map：
    - 优先尝试构建并运行 `map_decoder`（cfg 中 `transformer.map_decoder`）
    - 失败则走 MLP fallback（用 BEV global 特征 + learnable query embedding 预测 `map_cls_logits/map_pts`）
  - 推理输出通过 `get_map_results(outs, img_metas)` 标准化为 `(vectors,scores,labels)`。

关键张量形状（便于二次开发与替换为完整 MapTR）：

- head forward 产物：
  - `outs['bev_embed']`：`[bs, bev_h*bev_w, C]`
  - `outs['map_cls_logits']`：`[bs, num_map_vec, map_num_classes]`
  - `outs['map_pts']`：`[bs, num_map_vec, map_num_pts, 2]`

- `get_map_results(...)`：
  - `prob = map_cls_logits.sigmoid()`，然后 `scores, labels = prob.max(dim=-1)`。
  - 每个 sample 返回一个 dict：
    - `vectors`: `map_pts[i]`（numpy）
    - `scores`: `scores[i]`（numpy）
    - `labels`: `labels[i]`（numpy）
    - `cls_logits`: `map_cls_logits[i]`（numpy，可选）

### 4.8 轻量 MapTRDecoder：让 cfg 能 build

- 目录：`projects/mmdet3d_plugin/maptr/`
- 作用：注册 `MapTRDecoder`，让 `build_from_cfg(..., TRANSFORMER_LAYER_SEQUENCE)` 能工作。

### 4.9 离线 map 评测入口：把“正确导入 plugin”固化成脚本

- 文件：`tools/eval_map_offline.py`
- 关键点：
  - Apollo root 放入 `sys.path` 最前。
  - 复刻 cfg 的 plugin/custom_imports 导入逻辑，确保 registry 注册。
  - 打印 `mmdet3d.__file__` 帮你确认实际 import 的来源。

<a id="section-5-eval"></a>
## 5.评测协议与落盘产物：bbox 与 map 各自怎么评、产物在哪里

### 5.1 bbox（nuScenes detection）

- 主要产物：
  - `pts_bbox/metrics_summary.json`
  - `pts_bbox/results_nusc.json`

### 5.2 map（MapTR protocol）

- 主要产物：
  - `map_results.pkl`：推理的原始 map 结果（并行于 bbox）。
  - `nuscmap_results.json`：由 `format_map_results` 写出的 MapTR 协议预测 json。
  - `data/nuscenes/nuscenes_map_anns_val.json`：GT json（若不存在会自动生成）。

<a id="section-6-tests"></a>
## 6.复现命令与测试结果（含 2026-02-09 离线复测）

说明：smoke 阶段指标偏低是正常现象，本节只关注“链路闭环正确”。

### 6.1 dist_test：同一次跑 bbox + map

```bash
conda activate apollo_vnet
PORT=29513 bash tools/dist_test.sh \
  projects/configs/bevformer/bev_tiny_det_map_apollo.py \
  work_dirs/smoke_det_map_retrain_20260206_v4/epoch_1.pth 1 \
  --eval bbox chamfer iou
```

历史输出目录示例：
- `test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/`

提示：`tools/dist_test.sh` 只有在你**没有显式传入** `--eval` 时才会默认注入 `--eval iou bbox`，所以要跑 map 指标必须显式加 `chamfer/iou`。

bbox 指标（读取 `pts_bbox/metrics_summary.json`）：
- `mean_ap = 0.0`
- `nd_score = 0.01823157683703287`

map 指标（MapTR 协议，控制台输出或离线复测一致）：
- `NuscMap_chamfer/mAP = 4.098719873329375e-06`
- `NuscMap_iou/mAP = 0.0`

### 6.2 离线复测（2026-02-09）：不依赖 dist_test 的 map 评测

输入：`test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/map_results.pkl`。

chamfer：

```bash
conda activate apollo_vnet
python tools/eval_map_offline.py \
  projects/configs/bevformer/bev_tiny_det_map_apollo.py \
  test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/map_results.pkl \
  --eval chamfer --nproc 0 \
  --out-dir test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/map_eval_offline_retest_20260209_chamfer
```

关键输出：
- `mmdet3d.__file__ = /home/nuvo/MapTR/mmdetection3d/mmdet3d/__init__.py`
- `NuscMap_chamfer/mAP = 4.098719873329375e-06`

iou：

```bash
conda activate apollo_vnet
python tools/eval_map_offline.py \
  projects/configs/bevformer/bev_tiny_det_map_apollo.py \
  test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/map_results.pkl \
  --eval iou --nproc 0 \
  --out-dir test/bev_tiny_det_map_apollo/Fri_Feb__6_17_42_22_2026/map_eval_offline_retest_20260209_iou
```

关键输出：
- `NuscMap_iou/mAP = 0.0`

<a id="section-7-debug"></a>
## 7.常见坑与排障清单

下表列出训练/评测中常见的“症状 → 可能原因 → 快速修复”对照，便于现场排障：

| 症状 | 可能原因 | 修复 |
|---:|---|---|
| 缺少 `gt_map_vecs_label` / `gt_map_vecs_pts_loc`（训练样本中） | infos 中缺 `map_location` 或位姿字段；或 dataset 在 pipeline 后注入但 Collect3D 严格要求 keys | 重新生成 `nuscenes_infos_temporal_*.pkl`（确保包含 `map_location` 与位姿）；或使用允许缺键的 `CustomCollect3D`；运行自检脚本验证 `ds[0]` 中存在键 |
| 报 `Missing map_location/scene_name` | infos converter 未写入 location 字段 | 检查 `tools/create_data.py` 的 converter 参数并重新生成 infos |
| 报 `Unknown nuScenes map location` | location 值拼写错误或不在 nuScenes 支持的地图名列表 | 校正 infos 中的 location 值为 nuScenes 的标准名称（例如 `boston-seaport`） |
| pipeline/Collect3D 报 `KeyError`（因缺少 map 字段） | 收集算子在注入之前运行且严格要求 keys | 不在 `Collect3D.keys` 中强制加入 `gt_map_vecs_*`；使用本工程的 `CustomCollect3D` 或调整注入时序 |
| map 评测 mAP = 0 或结果全 0 | 预测和 GT 的坐标系/采样数不匹配（`fixed_ptsnum_per_line`）或裁剪窗口不一致 | 检查 cfg 中 `fixed_ptsnum_per_line` 与 `pc_range`；打印并比对 `gt_map_vecs_pts_loc` 与预测 `vectors` 的形状和坐标系 |
| 离线评测报 registry/ModuleNotFound（自定义 dataset/head 未注册） | `plugin` 没启用或 import 路径错误；import 到另一个 mmdet3d | 确认 cfg 中 `plugin=True` 且 `plugin_dir='projects/mmdet3d_plugin/'`；运行 `python -c "import mmdet3d; print(mmdet3d.__file__)"` 验证来源 |
| Shapely 报 `TopologicalError`/invalid geometry | 地图数据中存在自相交等不合法几何 | 在生成前对 geometry 做 `geom = geom.buffer(0)` 修复，或跳过该 patch 并记录 |

1) bbox evaluator 报 `KeyError: 'boxes_3d'`
- 原因：把 map 结果混进 bbox 的结构。
- 解决：严格保持 `bbox_results` 与 `map_results` 分离，并分开落盘。

2) 离线 build_dataset 报“xxx not in registry”
- 原因：没有执行 cfg 的 plugin/custom_imports 导入。
- 解决：使用 `tools/eval_map_offline.py`，或在你自己的脚本里复刻 `tools/test.py` 的导入逻辑。

3) `dist_test.sh` 覆盖 `--eval`
- 原因：脚本硬编码追加默认 eval。
- 解决：本工程已修复为“未传 `--eval` 才注入默认”。

4) Shapely `TopologicalError` / invalid geometry
- 解决：对 geometry 做 `buffer(0)` 修复；仍失败则跳过，保证评测不中断。

5) 多进程评测 pickling 失败
- 解决：`map_utils` 中 worker 需要定义在模块顶层；必要时将 `map_eval_nproc` 设置为 0 先单进程验证。

<a id="section-8-roadmap"></a>
## 8.后续迭代路线

1) 把 map scaffold（MLP fallback）替换为完整 MapTR head/assigner/loss，并把 loss 与 GT 对齐到论文实现。
2) 为 map 指标增加回归基线（固定输入 pkl 的离线评测 + 指标落盘），便于 CI 做 diff。
3) 把 smoke 配置升级为可收敛的训练配置（epochs、lr、data aug、map loss 权重与采样策略）。

---

<a id="section-9-audit"></a>
## 9.工程化对账表与回归清单

本节目标是把“代码改动 → 文档位置 → 如何回归验证”做成对账表，便于后续持续迭代时快速检查：

### 9.1 文件/模块对账表（建议作为 PR 自检清单）

| 文件/目录 | 主要职责 | 关键符号/产物 | 文档对应章节 |
|---|---|---|---|
| `projects/configs/bevformer/bev_tiny_det_map_apollo.py` | det+map smoke cfg（模型结构、数据管线、评测开关） | `plugin=True`、`dataset_type`、`fixed_ptsnum_per_line`、`--eval bbox chamfer iou` | §3 配置设计、§6 复现命令 |
| `projects/mmdet3d_plugin/__init__.py` | plugin 入口：导入并注册自定义 dataset/head/pipeline/maptr 等 | registry 可见性（避免 not in registry） | §3.1 plugin 与 registry |
| `projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py` | dataset：在线生成 map GT；map 评测与格式化 | `prepare_train_data/_add_vectormap_gt/evaluate_map/format_map_results`；注入 `gt_map_vecs_*` | §0.4 在线 GT、§4.5 dataset、§5 map 产物 |
| `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py` | pipeline：`CustomCollect3D` 对缺失 keys 容错 | `CustomCollect3D`（缺键跳过） | §0.4（pipeline 注入时序说明）、§7 排障 |
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` | detector：forward/test 输出拆分 det 与 map；训练透传 map GT | 推理返回 `{bbox_results, map_results}`；训练透传 `gt_map_vecs_*` | §2 输出接口规范、§4.1 detector |
| `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py` | head：det 分支复用 + map scaffold；`get_map_results` 标准化输出 | `outs['map_cls_logits']/'map_pts'` → `vectors/scores/labels` | §4.7 head、§2.2 map_results 规范 |
| `projects/mmdet3d_plugin/bevformer/apis/test.py` | DDP 测试收集：map 作为独立产物收集 | multi-gpu collect，tmpdir 后缀隔离 | §4.2 DDP 收集 |
| `tools/test.py` | 测试入口：落盘 bbox + map；按 `--eval` 触发 evaluator | `map_results.pkl`；调用 `evaluate_map` | §4.3 测试入口、§5 产物 |
| `tools/dist_test.sh` | 分布式测试脚本：不覆盖用户显式 `--eval` | `--eval` 注入逻辑 | §4.4 dist_test |
| `projects/mmdet3d_plugin/datasets/map_utils/` | MapTR 协议 evaluator（Chamfer/IoU mAP） | `NuscMap_chamfer/mAP`、`NuscMap_iou/mAP` | §4.6 map_utils、§6 指标 |
| `projects/mmdet3d_plugin/maptr/` | 轻量 MapTR 组件（用于 build decoder / scaffold） | `MapTRDecoder` | §4.8 MapTRDecoder |
| `tools/eval_map_offline.py` | 离线 map 评测：固化 plugin 导入，避免 registry 漏导入 | `--eval chamfer/iou`；打印 `mmdet3d.__file__` | §4.9 离线入口、§6.2 离线复测 |

### 9.2 最小回归清单（建议每次改动后跑一遍）

1) 确认 import 来源与 plugin 生效（避免“导入了另一个 mmdet3d/没注册到自定义组件”）：

```bash
conda activate apollo_vnet
python -c "import mmdet3d; print('mmdet3d from:', mmdet3d.__file__)"
```

2) cfg 能 build dataset/model（最快排除 registry 与数据路径问题）：

```bash
conda activate apollo_vnet
python -c "from mmcv import Config; from mmdet3d.datasets import build_dataset; from mmdet3d.models import build_model; cfg=Config.fromfile('projects/configs/bevformer/bev_tiny_det_map_apollo.py'); ds=build_dataset(cfg.data.val); m=build_model(cfg.model, test_cfg=cfg.get('test_cfg')); print(type(ds), 'evaluate_map' in dir(ds), type(m))"
```

3) 在线 map GT 真的生成（训练样本必须包含 `gt_map_vecs_*`）：

```bash
conda activate apollo_vnet
python - <<'PY'
from mmcv import Config
from mmdet3d.datasets import build_dataset

cfg = Config.fromfile('projects/configs/bevformer/bev_tiny_det_map_apollo.py')
ds = build_dataset(cfg.data.train)
sample = ds[0]
assert 'gt_map_vecs_label' in sample and 'gt_map_vecs_pts_loc' in sample
print('ok: online map gt exists')
PY
```

4) 端到端测试闭环（同一次 dist_test 同时产出 bbox + map，并能评测）：

```bash
conda activate apollo_vnet
PORT=29513 bash tools/dist_test.sh \
  projects/configs/bevformer/bev_tiny_det_map_apollo.py \
  /path/to/your_ckpt.pth 1 \
  --eval bbox chamfer iou
```

5) 离线 map 复测（排除 dist_test/分布式收集干扰）：

```bash
conda activate apollo_vnet
python tools/eval_map_offline.py \
  projects/configs/bevformer/bev_tiny_det_map_apollo.py \
  /path/to/map_results.pkl \
  --eval chamfer --nproc 0
```
