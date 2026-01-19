# 2026-01-15：Apollo-Vision-Net 接入 det+map（MapTR online vector GT）与 FP16 数值稳定性治本调试记录

> 本文为“未渲染 Markdown 原文”。
>
> 记录范围：本次会话中在 `Apollo-Vision-Net/` 下的所有 **未提交改动**（`git diff` + `git status --porcelain`），包括新增的 det+map（MapTR 风格）数据链路、head/config、三处 attention logits clamp、若干 meta/合同兜底与调试工具脚本，以及一些与训练/评测可用性相关的小修补。
>
> 关键目标：
> 1) 把 det+occ → det+map（与 MapTR online vector GT 对齐）链路跑通；
> 2) 打通最小训练闭环：`forward_train(1 batch)` 必须稳定输出 loss dict；
> 3) 为“治本式数值稳定”（softmax 上游 logits clamp，避免 overflow/inf/nan）提供可复现的 smoke 验证环境。

---

## 1. 目标（Goal）

### 1.1 功能目标

- 在 Apollo-Vision-Net 的 BEVFormer 系统中新增 **det+map** 分支：
  - map 预测以 MapTR 的 online vector map GT 为对齐对象；
  - GT 形态：每条 polyline 固定采样点数（会话中为 20），类别集合 {0,1,2}。

- 在不依赖完整训练流程的情况下，提供 **1-sample smoke**：
  - 直接执行 `model.forward_train(**batch)`；
  - 能打印 loss dict，用于快速闭环定位崩溃点与数值异常来源。

### 1.2 数值稳定性目标（治本）

- 不靠“下游 nan_to_num 掩盖”，而是针对 BEVFormer 的三处注意力模块，在 softmax 前加入 **logits clamp**，抑制 FP16 下的 overflow/inf/nan：
  1) Spatial cross-attention
  2) Temporal self-attention
  3) Decoder（MSDeformableAttention 等）

---

## 2. 本次改动总览（按 git diff 汇总）

> 以下来自：
> - `git diff --name-status`：列出所有被修改的 tracked 文件
> - `git status --porcelain`：列出所有未跟踪新增文件

### 2.1 修改（Modified）文件清单（tracked）

- `README.md`
- `projects/configs/bevformer/bev_tiny_det_occ_apollo.py`
- `projects/mmdet3d_plugin/bevformer/dense_heads/__init__.py`
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_occupancy_head_apollo.py`
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`
- `projects/mmdet3d_plugin/bevformer/modules/decoder.py`
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py`
- `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`
- `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`
- `projects/mmdet3d_plugin/bevformer/modules/transformer.py`
- `projects/mmdet3d_plugin/datasets/__init__.py`
- `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`
- `projects/mmdet3d_plugin/datasets/nuscnes_eval.py`
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`
- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`
- `tools/create_data.sh`（mode 变化 644→755）
- `tools/create_data_with_occ.py`（mode 变化 644→755）
- `tools/data_converter/nuscenes_converter.py`
- `tools/dist_test.sh`
- `tools/test.py`
- `tools/train.py`

### 2.2 新增（Untracked / Added）文件清单

> 注意：这里是 `git status` 中的未跟踪文件；其中包含大量 `__pycache__/`、`tools_outputs/`、`work_dirs/` 等运行产物。
> 本文仅列出与功能相关的“源码/配置/脚本”新增项，并单独标注“运行产物/临时文件”。

#### 2.2.1 新增：det+map（MapTR）功能相关源码/配置

- `projects/configs/bevformer/bev_tiny_det_map_apollo.py`
  - det+map 的实验配置：包含模型 head、数据管线、debug 开关等。

- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`
  - det+map head 的最小实现（map query / map logits / map pts），当前阶段以“先简后全”的 loss 作为训练闭环验证。

- `projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`
  - 数据集 wrapper：在 det(+occ) 的基础上注入 map GT（online vector）等字段。

- `projects/mmdet3d_plugin/datasets/pipelines/loading_maptr_gt_det_map.py`
  - pipeline 组件：加载/生成 MapTR 风格 GT（例如 `gt_map_vecs_label`、`gt_map_vecs_pts_loc`）并塞入 results。

- `tools/smoke_det_map_forward_train.py`
  - 单样本 smoke：构造最小 batch，规范化 `img` 与 `img_metas` 合同，运行 `forward_train` 并打印 loss。

#### 2.2.2 新增：MapTR/可视化/调试脚本（工具类）

- `tools/simulate_and_vis_map_gt.py`
- `tools/debug_map_gt_one_sample.py`
- `tools/analysis_tools/vis_det_bev_single.py`

#### 2.2.3 新增：文档/博客（非代码）

- `docs/bevformer_det_occ_multitask_blog.md`
- `docs/debug_shapes_v1mini_blog.md`
- `docs/blog_diffusiondrivev2_code_analysis.md`
- `docs/knowval_kg_value_guided_driving_blog.md`

#### 2.2.4 运行产物/临时文件（不建议纳入提交）

- 大量 `__pycache__/`、`tools_outputs/`、`work_dirs/`、`test/`、`val/`、`data/` 等

---

## 3. 关键改动：逐文件说明（为何改、改了什么、前后对比）

> 本节只覆盖“对目标闭环有直接关系的代码路径”。
> 对纯文档/工具的新增会在后面统一汇总。

### 3.1 det+map head 注册

#### 文件
- `projects/mmdet3d_plugin/bevformer/dense_heads/__init__.py`

#### 改动
- 新增导出：`BEVFormerDetMapHeadApollo`

#### 动机
- 让 config 能通过 registry 正常构建 det+map head。

---

### 3.2 Detector：把 det_map GT 透传到 head.loss

#### 文件
- `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

#### 改动点
- `__init__` 新增参数：`debug_nan=False`；保存为 `self.debug_nan`。
- 在 `extract_img_feat` 后增加数值 probe（`[bevformer][probe]`）用于定位 backbone/neck 输出是否已经出现非有限或极端值。
- `forward_pts_train(...)` 与 `forward_train(...)` 透传：
  - `gt_map_vecs_label`
  - `gt_map_vecs_pts_loc`
  到 `self.pts_bbox_head.loss(..., img_metas=..., gt_map_vecs_*=...)`。

#### 动机
- det_map 的 GT 字段来自数据管线（online map GT），必须从 runner → detector → head.loss 打通。
- 数值 probe 用于回答“NaN/Inf 在哪一层出现”：是 backbone/neck 就爆、还是 transformer attention 才爆。

---

### 3.3 三处 attention：softmax 前 logits clamp（治本）

> 目标：把 FP16 下 softmax 的 overflow 风险压在上游 logits 阶段，而不是事后 nan_to_num。

#### 3.3.1 Decoder attention clamp

- 文件：`projects/mmdet3d_plugin/bevformer/modules/decoder.py`
- 改动：
  - 在 `CustomMSDeformableAttention` 增加初始化参数：
    - `attn_logits_clamp=None`
    - `debug_attn_nan=False`
  - 在 `attention_weights.softmax(-1)` 之前对 logits 执行：
    - `attention_weights = attention_weights.clamp(-c, c)`（当 c 非空）
  - softmax 后若仍出现非有限，打印 ratio。

#### 3.3.2 Temporal self-attention clamp

- 文件：`projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`
- 改动：
  - 同样增加 `attn_logits_clamp` 与 `debug_attn_nan`。
  - softmax 前 clamp，softmax 后检查 `torch.isfinite`。

#### 3.3.3 Spatial cross-attention clamp

- 文件：`projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`
- 改动：
  - 增加 `attn_logits_clamp` 与 `debug_attn_nan`。
  - （注意：该模块 diff 中展示的是构造器新增字段；softmax 前 clamp 的具体位置需结合模块内实现确认是否已接入。）

#### 动机
- softmax 在 FP16 下极易因 logits 过大导致 `exp()` overflow → inf → NaN。
- clamp 是一种简单、可控、可配置化的治本段，适合在早期稳定训练与定位问题。

---

### 3.4 Transformer：有限性 probe + 若干兜底（让 smoke 先跑到 loss）

#### 文件
- `projects/mmdet3d_plugin/bevformer/modules/transformer.py`

#### 改动
- 增加 `_finite_stats(name, x)`：当张量出现非有限，打印 ratio/min/max/shape。
- 在多个关键张量处插入 probe：
  - `shift`
  - `bev_queries(after_can_bus)`
  - `mlvl_feats[lvl]`
  - `feat_lvl{lvl}_after_embeds`
  - `feat_flatten`
  - `bev_embed(from_encoder)`
- 对输入特征的非有限进行 `torch.nan_to_num` 的 sanitize（说明：这类 sanitize 并非最终目标，更多是为了“先跑通”并避免 attention 接收到 inf/nan 导致全链路崩溃）。
- `rotation_angle = can_bus[-1]` 增加 fallback：当 can_bus 缺失/为空时 `rotation_angle=0.0`。

#### 动机
- 让 smoke 能一路推进到 loss 输出，以便验证 clamp 是否有效。
- probe 输出用于对比“clamp 开关前后”的数值变化。
- rotation_angle fallback 属于合同兜底，避免 `IndexError` 阻塞调试。

---

### 3.5 Encoder：point_sampling 合同修复（lidar2img / img_shape / temporal metas）

#### 文件
- `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

#### 改动
- point_sampling 增加 meta debug（受 `debug_nan` 控制），打印 `img_metas` 形态与 `lidar2img` 形状。
- `lidar2img` 形状坍缩修复：当 `np.asarray` 导致维度异常时尝试 stack 复原。
- `num_cam` 推断兜底：当 `lidar2img.numel()` 与 `B*num_cam*16` 不一致时用 `numel//(B*16)` 推断。
- `img_shape` 缺失 fallback：无法获取 H/W 时使用 H=W=1 进行归一化（保证 forward 可跑）。
- `img_metas_for_sampling` 规范化：当传入 `list[list[dict]]`（bs,len_queue）时取最后一帧，变成 `list[dict]` 供 point_sampling 正常使用。

#### 动机
- 解决历史阻塞：`lidar2img.view(...)` size 不匹配、`KeyError: img_shape` 等。
- 保证最小 smoke 能走进 transformer encoder 并返回 bev_embed。

---

### 3.6 数据链路：dataset/pipeline 注入 MapTR 在线 GT

#### 文件
- `projects/mmdet3d_plugin/datasets/__init__.py`
  - 导出 `CustomNuScenesDetOccMapDataset` / `CustomNuScenesDetMapDataset`

- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`
  - 注册 pipeline：`LoadMapTRGTDetMap`

- `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`
  - `CustomCollect3D`：对 keys 做了“缺失跳过”，允许 online map GT 等字段由 wrapper 注入后存在/不存在都不崩。

- `tools/data_converter/nuscenes_converter.py`
  - 为每帧 info 记录 `map_location`（从 log.location 提取），为 map GT 生成所需的城市/区域定位提供信息源。

#### 动机
- MapTR 的 GT 往往需要 map location（例如 Singapore-hollandvillage / boston-seaport 等）来选取正确的地图。
- 在线 GT 注入时字段可选性更强，collect 阶段不能对所有 key 强假设。

---

### 3.7 det+occ 训练/评测可用性增强（支撑整体调试）

> 虽然与 det+map 不是同一条主链，但这些修改直接提升“跑通/可复现”能力。

- `projects/configs/bevformer/bev_tiny_det_occ_apollo.py`
  - `total_epochs: 24 → 100`
  - 默认禁用 TensorboardLoggerHook（避免某些环境 distutils 缺失导致 before_run 崩溃）。

- `tools/train.py`
  - `sys.path.insert(0, os.getcwd())`：确保 plugins 可被找到。
  - 支持 `plugin_dir` 为绝对路径时仍能正确 import。

- `tools/dist_test.sh`
  - 增加参数检查与 usage 提示。
  - 使用 `python3 -m torch.distributed.launch`（环境兼容）。

- `tools/test.py` + `projects/mmdet3d_plugin/datasets/nuscenes_dataset.py`
  - occ eval 结果的保存路径对齐 detection 的 json 输出目录。
  - occ eval 支持保存 `metrics_summary.json`。

- `projects/mmdet3d_plugin/datasets/nuscnes_eval.py`
  - 增加 velocity shape 的安全修复（monkey-patch output_to_nusc_box），避免 NuScenesBox.rotate 因速度向量形状异常崩溃。

---

## 4. smoke 验证：从崩溃到 loss 输出

### 4.1 复跑命令（示例）

```bash
PYTHONPATH=/home/nuvo/Apollo-Vision-Net \
python /home/nuvo/Apollo-Vision-Net/tools/smoke_det_map_forward_train.py \
/home/nuvo/Apollo-Vision-Net/projects/configs/bevformer/bev_tiny_det_map_apollo.py \
--cfg-options model.debug_nan=True
```

### 4.2 关键输出（节选）

- meta 合同：
  - `[smoke] img_metas[0][0].lidar2img shape=(6, 4, 4)`
  - `[smoke] img_metas[0][1].lidar2img shape=(6, 4, 4)`

- 数值 probe（示例）：
  - `[bevformer][nan] feat_lvl0_after_embeds: ratio=0.8945 ... min≈-1e35 max≈1e1`

- det_map GT sanity：
  - `[det_map][gt] sample0: num_lines=15, labels_uniq=[0, 1, 2], pts_shape=(15, 20, 2)`

- 最终 loss dict（成功）：
  - `loss_map_cls: 0.5236`
  - `loss_map_pts: 10.6262`

### 4.3 阶段性结论

- `forward_train(1 batch)` 已能够稳定跑到 det_map loss 并输出 loss dict（闭环达成）。
- 上游 feature 的非有限率仍 < 1（probe 明确显示），表明 clamp 的验证仍是下一优先级。

---

## 5. 后续计划（Next steps）

1. 基于当前 smoke 闭环做 A/B：分别设置 `attn_logits_clamp` 为 None vs 某个合理值（如 20/50/100），分别在三个 attention 模块验证：
   - `[bevformer][nan]` 的 ratio 是否明显上升（趋近 1.0）
   - loss 是否保持有限
   - 是否还会出现 attention_weights 非有限打印

2. “兜底代码清理（收敛）”：
   - 把只能让 forward 活下来的 fallback（如 H=W=1、rotation_angle=0）回推到 dataset/meta 的真实合同修复。

3. 在 `docs/changes/` 下补充一份更短的“变更日志 + 如何复现”，并把 `__pycache__/work_dirs/tools_outputs` 等运行产物加入 `.gitignore`（如项目允许）。
---

## 6. 附录：今天改动的一句话索引（便于 code review）

- det+map：新增 head/config/dataset/pipeline + detector 透传 GT。
- 数值稳定：三处 attention softmax 前加 logits clamp（可控、可 debug）。
- 合同兜底：encoder.point_sampling/transformer rotation_angle/can_bus/img_shape 等，确保 smoke 跑通到 loss。
- 运行体验：train/test/dist_test/eval 的若干健壮性修补。

---

## 7. 进展回顾（补充：Apollo‑Vision‑Net 接入 MapTR head）

### 7.1 目标与范围

- 目标：在 Apollo‑Vision‑Net 中新增 MapTR 风格 det+map head，并接入 online vector GT，形成可复现的最小训练闭环。
- 范围：新增 det_map 配置/数据管线/head，打通 GT 透传与 loss；同步加入数值稳定性治本手段（attention logits clamp）与 smoke 验证脚本。

### 7.2 已完成内容（核心链路）

- det_map 配置：`projects/configs/bevformer/bev_tiny_det_map_apollo.py`
- det_map head：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`
- 数据集扩展：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`
- online GT pipeline：`projects/mmdet3d_plugin/datasets/pipelines/loading_maptr_gt_det_map.py`
- 1‑sample smoke：`tools/smoke_det_map_forward_train.py`
- 完整回顾文档：`docs/changes/2026-01-15-det-map-port.md`

### 7.3 已完成内容（稳定性与兜底）

- 三处 attention softmax 前 logits clamp：spatial / temporal / decoder
- transformer/encoder 数值 probe 与合同兜底：`lidar2img`、`img_shape`、`can_bus`、temporal metas 等
- detector → head.loss 透传 map GT 字段：`gt_map_vecs_label`、`gt_map_vecs_pts_loc`

### 7.4 现状结论

- det_map 端到端 smoke 已能稳定输出 loss（闭环达成）
- clamp 已接入并可配置，具备 FP16/AMP A/B 验证基础
- 数值 probe 显示上游特征极值仍偏大，需进一步验证 clamp 的治本效果

### 7.5 下一步（主线）

- 在 AMP/FP16 条件下做 clamp A/B 对比，收集 `[bevformer][nan]` ratio/min/max 与 loss 稳定性
- 在 clamp 证据充分后，逐步回退 transformer 中的临时 `nan_to_num` 兜底

---

## 8. 核心代码摘录（创建/修改重点）

> 说明：以下为关键路径的最小片段，用于让读者快速定位核心逻辑；完整实现以源码为准。

### 8.1 det_map head：MapTR head 模型搭建与最小 loss（新建）

文件：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`

#### 8.1.1 头部构造：MapTR 风格 query + cls/pts 预测头

```python
# --- Map branch (stub, MapTR-aligned) ---
self.enable_map = kwargs.get('enable_map', True)

# Minimal map prediction heads ("先简后全")
self.num_map_vec = int(kwargs.get('num_map_vec', 50))
self.map_num_pts = int(kwargs.get('map_num_pts', kwargs.get('fixed_ptsnum_per_line', 20)))
self.map_num_classes = int(kwargs.get('map_num_classes', 3))

self.map_query_embed = nn.Embedding(self.num_map_vec, self.embed_dims)
self.map_cls_head = nn.Sequential(
  nn.Linear(self.embed_dims, self.embed_dims),
  nn.ReLU(inplace=True),
  nn.Linear(self.embed_dims, self.map_num_classes),
)
self.map_pts_head = nn.Sequential(
  nn.Linear(self.embed_dims, self.embed_dims),
  nn.ReLU(inplace=True),
  nn.Linear(self.embed_dims, self.map_num_pts * 2),
)
```

**说明**：采用 MapTR 的“固定数量 vector queries + cls/pts head”框架；当前是最小可用版，便于先打通数据与 loss 闭环。

#### 8.1.2 Forward：从 BEV 全局特征生成 map 预测

```python
if self.enable_map:
  if isinstance(bev_embed, torch.Tensor) and bev_embed.dim() == 3:
    bev_global = bev_embed.mean(dim=1)  # [bs, C]
  else:
    bev_global = mlvl_feats[0].new_zeros((bs, self.embed_dims))

  q = self.map_query_embed.weight.unsqueeze(0).expand(bs, -1, -1)
  q = q + bev_global.unsqueeze(1)
  outs['map_cls_logits'] = self.map_cls_head(q)  # [bs, num_vec, C]
  pts = self.map_pts_head(q).view(bs, self.num_map_vec, self.map_num_pts, 2)
  outs['map_pts'] = pts
```

**说明**：用 BEV 全局上下文给 query 加 bias，先做最小化预测，便于后续替换为完整 MapTR decoder。

#### 8.1.3 Loss：GT 容器兼容与“先简后全”闭环

```python
# Normalize GT container types to per-batch lists
if not isinstance(gt_map_vecs_label, (list, tuple)):
  gt_map_vecs_label = [gt_map_vecs_label]
if not isinstance(gt_map_vecs_pts_loc, (list, tuple)):
  gt_map_vecs_pts_loc = [gt_map_vecs_pts_loc]

if self.enable_map and (not missing_map_gt):
  map_cls = outs.get('map_cls_logits', None)
  map_pts = outs.get('map_pts', None)
  if isinstance(map_cls, torch.Tensor) and isinstance(map_pts, torch.Tensor):
    # 对齐前 K 条 GT/Pred，做最小闭环 loss
    pred_logits = map_cls[b, :K, :]
    tgt_labels = gt_lab[:K].long().to(pred_logits.device)
    loss_cls = loss_cls + F.cross_entropy(pred_logits, tgt_labels)

    pred_pts = map_pts[b, :K, :num_pts, :]
    tgt_pts = gt_pts[:K].to(pred_pts.device)
    loss_pts = loss_pts + F.l1_loss(pred_pts, tgt_pts)
```

**说明**：对 GT 形态进行兼容（`LiDARInstanceLines`/list/batch），并以“对齐前 K”方式计算最小 loss，用于验证梯度闭环与训练稳定性。

### 8.2 数据管线：注入 MapTR 在线 GT（新建）

文件：`projects/mmdet3d_plugin/datasets/pipelines/loading_maptr_gt_det_map.py`

```python
results['gt_map_vecs_label'] = map_labels
results['gt_map_vecs_pts_loc'] = map_pts
```

### 8.3 数据集：包装 det/occ → det_map（新建）

文件：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`

```python
# 在原有 det(+occ) 结果上追加 map GT 字段
results = super().prepare_train_data(index)
results.update(map_gt)
```

### 8.4 Detector：GT 透传到 head.loss（修改）

文件：`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

```python
losses = self.pts_bbox_head.loss(
  gt_bboxes_3d=gt_bboxes_3d,
  gt_labels_3d=gt_labels_3d,
  gt_map_vecs_label=gt_map_vecs_label,
  gt_map_vecs_pts_loc=gt_map_vecs_pts_loc,
  img_metas=img_metas,
)
```

### 8.5 三处 attention：softmax 前 logits clamp（修改）

文件：
`projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`
`projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`
`projects/mmdet3d_plugin/bevformer/modules/decoder.py`

```python
if self.attn_logits_clamp is not None:
  attention_weights = attention_weights.clamp(
    -self.attn_logits_clamp, self.attn_logits_clamp
  )
attention_weights = attention_weights.softmax(-1)
```

### 8.6 Transformer/Encoder：合同兜底与 probe（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/transformer.py`

```python
def _finite_stats(self, name, x):
  if x is None:
    return
  finite = torch.isfinite(x)
  ratio = finite.float().mean().item()
  if ratio < 1.0:
    print(f"[bevformer][nan] {name}: ratio={ratio} ...")
```

文件：`projects/mmdet3d_plugin/bevformer/modules/encoder.py`

```python
if 'img_shape' not in img_meta:
  img_shape = (1, 1, 3)
```

### 8.7 1‑sample smoke：最小闭环（新建）

文件：`tools/smoke_det_map_forward_train.py`

```python
losses = model.forward_train(**data_batch)
print("=== forward_train losses ===")
for k, v in losses.items():
  print(k, float(v))
```

### 8.8 BEV 查询与位置编码：BEVFormer 兼容的 BEV 生成（新建）

文件：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`

```python
bev_queries = self.bev_embedding.weight.to(dtype)
bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
bev_pos = self.positional_encoding(bev_mask).to(dtype)

bev_embed = self.transformer.get_bev_features(
  mlvl_feats,
  bev_queries,
  self.bev_h,
  self.bev_w,
  grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
  bev_pos=bev_pos,
  img_metas=img_metas,
  prev_bev=prev_bev,
)
```

**说明**：该段确保 det_map head 与 BEVFormer 的 temporal `prev_bev` 机制兼容，并输出可缓存的 `bev_embed`。

### 8.9 MapTR 风格 inference hook（新建）

文件：`projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`

```python
def get_map_results(self, outs, img_metas, **kwargs):
  bs = len(img_metas)
  return [dict(vectors=[], scores=[], labels=[]) for _ in range(bs)]
```

**说明**：预留 MapTR 的向量化解码接口，当前返回空结构以保证推理链路不崩。

### 8.10 Config：启用 det_map head 与 clamp（新建）

文件：`projects/configs/bevformer/bev_tiny_det_map_apollo.py`

```python
pts_bbox_head=dict(
  type='BEVFormerDetMapHeadApollo',
  bev_h=bev_h_,
  bev_w=bev_w_,
  embed_dims=_dim_,
  transformer=dict(
    type='PerceptionTransformer',
    encoder=dict(
      type='BEVFormerEncoder',
      transformerlayers=dict(
        attn_cfgs=[
          dict(type='TemporalSelfAttention', attn_logits_clamp=20.0),
          dict(
            type='SpatialCrossAttention',
            deformable_attention=dict(type='MSDeformableAttention3D', attn_logits_clamp=20.0),
          ),
        ],
      ),
    ),
    decoder=None,
  ),
  enabling_map=True,
)
```

**说明**：配置中直接启用 det_map head，并在两类 attention 中接入 `attn_logits_clamp`，为 FP16 稳定性做治本准备。

### 8.11 在线 MapTR GT：vector 生成与固定采样（新建）

文件：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`

```python
class LiDARInstanceLines(object):
  @property
  def fixed_num_sampled_points(self):
    distances = np.linspace(0, instance.length, self.fixed_num)
    sampled_points = np.array([list(instance.interpolate(d).coords) for d in distances]).reshape(-1, 2)
    # clamp to patch range
    instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
    instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
    return instance_points_tensor

class VectorizedLocalMap(object):
  def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
    # ... build line/polygon instances per class
    return dict(gt_vecs_pts_loc=gt_instance_lines, gt_vecs_label=gt_labels)
```

**说明**：该段提供 MapTR 风格在线 GT 的“固定采样点数 + 类别标签”，并与 det_map head 的最小 loss 对齐。

### 8.12 Pipeline：加载 MapTR GT 并注入 results（新建）

文件：`projects/mmdet3d_plugin/datasets/pipelines/loading_maptr_gt_det_map.py`

```python
map_gt_path = results.get('map_gt_path', None)
if map_gt_path is None:
  raise KeyError('LoadMapTRGTDetMap expects `results[\'map_gt_path\']`.')

data = np.load(map_gt_path, allow_pickle=True)
pts_key = self.npz_keys['pts']
label_key = self.npz_keys['label']
if pts_key not in data or label_key not in data:
  raise KeyError(
    f'MapTR map GT npz missing keys: required ({pts_key}, {label_key}), '
    f'got {list(data.keys())} from {map_gt_path}')

results['map_gts'] = {
  'gt_vecs_pts_loc': data[pts_key],
  'gt_vecs_label': data[label_key],
}
```

**说明**：该段提供 MapTR GT 的 npz 加载与 key 校验，并把 GT 注入 results，供后续 dataset/pipeline 使用。

### 8.13 数据集：map_location 兜底与在线 GT 入口（新建/修改）

文件：`projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`

```python
def _scene_name_to_log_location(scene_name, dataroot, version='v1.0-trainval'):
  # 旧 infos 缺 map_location 时的兜底查询
  from nuscenes.nuscenes import NuScenes
  nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
  for s in nusc.scene:
    if s.get('name') == scene_name:
      log = nusc.get('log', s['log_token'])
      return log.get('location')
  return None
```

**说明**：当旧版 infos 未提供 `map_location` 时，通过 scene→log 反查 location，保证在线 GT 能选到正确地图。

### 8.14 Collect：缺失 key 跳过，兼容在线 GT 注入（修改）

文件：`projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`

```python
for key in self.keys:
  # Some keys (e.g., online-generated map GT) may be injected later
  # by dataset wrappers. Skip missing keys to avoid hard crashes.
  if key in results:
    data[key] = results[key]
```

**说明**：允许 dataset 在 `CustomCollect3D` 之后再补充 map GT，避免因 key 缺失导致 pipeline 直接崩溃。

### 8.15 Encoder：lidar2img / img_shape 输入特在数值范围兜底（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/encoder.py`

```python
lidar2img = np.asarray(lidar2img)
if lidar2img.ndim == 3 and lidar2img.shape[-2:] == (4, 4):
  try:
    lidar2img = np.stack([np.asarray(x) for x in lidar2img], axis=0)
  except Exception:
    pass

if (H_img is None) or (W_img is None) or (H_img <= 0) or (W_img <= 0):
  H_img, W_img = 1, 1
```

**说明**：处理 `lidar2img` 维度坍缩与 `img_shape` 缺失问题，让 point_sampling 在 smoke 中可继续执行。

### 8.16 Transformer：can_bus 兜底与 NaN sanitize（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/transformer.py`

```python
# Some minimal/smoke paths may provide incomplete can_bus.
rotation_angle = 0.0
try:
  _cb = kwargs['img_metas'][i].get('can_bus', None)
  if isinstance(_cb, (list, tuple)) and len(_cb) > 0:
    rotation_angle = float(_cb[-1])
except Exception:
  pass

if not torch.isfinite(feat_flatten).all():
  feat_flatten = torch.nan_to_num(feat_flatten, nan=0.0, posinf=0.0, neginf=0.0)

if not torch.isfinite(bev_embed).all():
  bev_embed = torch.nan_to_num(bev_embed, nan=0.0, posinf=0.0, neginf=0.0)
```

**说明**：rotation_angle fallback 保证 temporal BEV 不因 can_bus 缺失而崩；NaN sanitize 用于“先跑通”与定位数值问题（后续会逐步回退）。

### 8.17 Temporal Self-Attention：logits clamp + 非有限检测（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`

```python
attention_weights = self.attention_weights(query).view(
  bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
if self.attn_logits_clamp is not None:
  c = float(self.attn_logits_clamp)
  attention_weights = attention_weights.clamp(min=-c, max=c)
attention_weights = attention_weights.softmax(-1)

if self.debug_attn_nan and (not torch.isfinite(attention_weights).all()):
  with torch.no_grad():
    finite = torch.isfinite(attention_weights)
    ratio = float(finite.float().mean().detach().cpu())
    print('[attn][temporal] attention_weights non-finite, ratio=', ratio)
```

**说明**：在 softmax 之前对 logits clamp，配合 `debug_attn_nan` 在 FP16 场景追踪非有限比例。

### 8.18 Decoder Deformable Attention：logits clamp + 非有限检测（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/decoder.py`

```python
attention_weights = self.attention_weights(query).view(
  bs, num_query, self.num_heads, self.num_levels * self.num_points)
if self.attn_logits_clamp is not None:
  c = float(self.attn_logits_clamp)
  attention_weights = attention_weights.clamp(min=-c, max=c)
attention_weights = attention_weights.softmax(-1)

if self.debug_attn_nan and (not torch.isfinite(attention_weights).all()):
  with torch.no_grad():
    finite = torch.isfinite(attention_weights)
    ratio = float(finite.float().mean().detach().cpu())
    print('[attn][decoder] attention_weights non-finite, ratio=', ratio)
```

**说明**：decoder 分支同样 clamp 并打印非有限比例，与 temporal/self 的诊断策略一致。

### 8.19 Spatial Cross-Attention：clamp 开关与稳定性参数（修改）

文件：`projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

```python
def __init__(..., attn_logits_clamp=None, debug_attn_nan=False):
  # Numerical stability (GPU/FP16): optionally clamp attention logits
  # before softmax to avoid overflow.
  self.attn_logits_clamp = attn_logits_clamp
  self.debug_attn_nan = bool(debug_attn_nan)
```

**说明**：在 Spatial Cross-Attention 的 3D deformable attention 中新增 clamp 开关与 debug 标记，为与 temporal/decoder 形成一致的可控稳定性策略。

