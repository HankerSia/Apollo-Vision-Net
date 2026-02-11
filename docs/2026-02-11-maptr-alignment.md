# MapTR 对齐与 BEVFormer det+map 改动说明（2026-02-11）

本文总结了为使 Apollo-Vision-Net 的 det+map 路径更贴近官方 MapTR 实现所做的更改，改进运行态可观测性，并新增本地 smoke 验证工具的工作。

包含的提交：

- `5993635` — maptr: vendor assigner/loss head/losses（与官方对齐）
- `6322848` — dataset: 添加 `LiDARInstanceLines.bbox` 与 shift v1/v2
- `266d2d9` — bevformer: MapTR-official map loss 路径 与 `run_cfg` 日志
- `18e45da` — tools/docs: 改进 smoke 前向脚本、控制模型打印、添加走读文档

---

## 1) MapTR Vendoring 与 Loss 对齐

**为什么**
- 降低与官方 MapTR 在 assignment / matching / loss 上的语义差异，这些差异在 AMP/FP16 下尤其会导致匹配或数值不稳定的问题。

**怎么改（实现要点）**
- Vendor（移植并注册）了轻量的 MapTR 模块子集：`MapTRAssigner`、`MapTRLossHead`、以及相关的 pts/dir loss 和匹配 cost。它们实现了基于 cls/reg/iou/pts 综合成本的 Hungarian 匹配，并支持 OrderedPtsL1 / Chamfer 风格的成本。
- 在 `MapTRLossHead.loss()` 入口处将模型预测强制转换为 FP32（即对 `preds_dicts` 的预测项调用 `.float()`），以模拟官方的 `@force_fp32(apply_to=('preds_dicts'))`，避免 AMP 引起的数值/精度差异。

**关键文件**
- `projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py`
- `projects/mmdet3d_plugin/maptr/dense_heads/maptr_loss_head.py`（loss 入口的 FP32 cast）
- `projects/mmdet3d_plugin/maptr/losses/map_loss.py`

**效果与验证**
- 语义上与官方 MapTR 对齐（Hungarian 匹配 + cls/reg/pts 成本，支持 OrderedPtsL1Cost / Chamfer 等）。
- 快速验证（smoke）：
```bash
export PYTHONPATH=$PWD
python tools/smoke_det_map_forward_train.py projects/configs/bevformer/bev_tiny_det_map_apollo.py
```
- 检查输出中 `loss_map_*` 以及中间 decoder 层的 `d*.loss_*` 是否与预期一致，特别是在使用 AMP/FP16 时关注数值稳定性。

提交快照：`5993635`

---

## 2) 数据层：`LiDARInstanceLines.bbox` 与 Shift 协议（v0/v1/v2）

**为什么**
- MapTR 依赖对每条 polyline 的固定采样点與 shift 协议计算有序点（ordered pts）损失并进行匹配；assigner 也需要 bbox 来计算 reg/iou 成本。因此数据与 head 在点数与 shift 语义上必须严格一致。

**怎么改（实现要点）**
- 在数据类中新增：
  - `LiDARInstanceLines.bbox`：返回每条线的 [xmin, ymin, xmax, ymax]，并 clamp 到 patch 范围。
  - `shift_fixed_num_sampled_points`、`shift_fixed_num_sampled_points_v1`、`shift_fixed_num_sampled_points_v2`：支持闭合/非闭合 polyline 的 shift，含 padding 与可选子采样逻辑。

**关键文件**
- `projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`

**效果与验证**
- 确保配置中的 `map_num_pts` 与数据集的 `fixed_ptsnum_per_line` 一致，避免点数不一致导致插值或匹配错误。
- 验证：运行 smoke 脚本，查看 head 中一次性打印的 `gt_map_vecs_pts_loc` 的 shape 与 `gt_map_vecs_label` 的数量。

提交快照：`6322848`

---

## 3) BEVFormer det+map Head：MapTR-official 路径 + 运行态一次性日志

**为什么**
- 仅看 config 无法完全判断训练时实际走的分支；需要在运行时确认 map 分支的关键开关（是否使用 point queries、是否做 box refine、decoder 类型/层数、pts normalize/pts cost type 等）。

**怎么改（实现要点）**
- 当 `map_loss_impl == 'maptr_official'` 时，`BEVFormerDetMapHeadApollo` 会调用 `MapTRLossHead.loss()`，并把返回的 loss key 前缀为 `loss_map_*`，以便在日志和指标中分离 det 与 map。
- 新增 rank0-only 且仅打印一次的 `[det_map][run_cfg]` 日志行，打印包括 `map_loss_impl`、`map_use_point_queries`、`map_with_box_refine`、`map_decoder`（类型）、`num_layers`、`map_query_embed_type`、`num_map_vec`、`map_num_pts`、`map_num_query`、`map_pts_normalize`、`map_pts_cost_type` 等关键项。
- 添加 `DetMapTextLoggerHook`，在训练日志中把 `det:` 與 `map:` 指标分行显示，并隐藏接近 0 的被禁用 loss，以提高可读性。

**关键文件**
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`（`[det_map][run_cfg]` 日志與 `maptr_official` 分支）
- `projects/mmdet3d_plugin/bevformer/hooks/det_map_text_logger_hook.py`

**效果與验证**
- 运行训练或 smoke 時，日志包含类似的行：
```
[det_map][run_cfg] map_loss_impl=maptr_official map_use_point_queries=True map_with_box_refine=True map_decoder=MapTRDecoder num_layers=6 map_num_pts=20 map_pts_cost_type=chamfer
det: loss: 0.1234, loss_cls: 0.0123, ...
map: loss_map: 0.2345, loss_map_pts: 0.2200, loss_map_dir: 0.0145
```
- 验证：使用 smoke 脚本或短跑训练，检查日志中是否出现 `[det_map][run_cfg]` 以及 `det:` / `map:` 输出。

提交快照：`266d2d9`

---

## 4) Tools 与文档：smoke 脚本增强与走读文档

**为什么**
- 需要快速且可靠的本地 sanity-check 工具来发现 shape/NaN/契约问题，并将实现细节写成文档以便团队复现。

**怎么改（实现要点）**
- 改进 `tools/smoke_det_map_forward_train.py`：兼容多种 `img` / `img_metas` 布局、注册轻量 forward hook 以定位 NaN，并在需要时把 `gt_bboxes_3d` tensor 包装为 `LiDARInstance3DBoxes`。
- 在 `tools/train.py` 中加入 `log_model_summary` 开关以控制是否打印模型结构。
- 添加走读文档：`docs/bev_sparse_det_maptr_flashocc_henet_tinym_nuscenes.md`，说明配置到实现的对应关系與注意事项。

**效果與验证**
- 使用 smoke 脚本可快速发现前向/后向流程中的不一致或数值问题，节省调试时间。
- 验证（示例命令）：
```
export PYTHONPATH=$PWD
python tools/smoke_det_map_forward_train.py projects/configs/bevformer/bev_tiny_det_map_apollo.py
# 控制是否打印模型结构
python tools/train.py projects/configs/bevformer/bev_tiny_det_map_apollo.py --cfg-options log_model_summary=False
```

提交快照：`18e45da`

---

## 附注與下一步建議

- 所有更改已推送到远端仓库（各提交 id 如上）。如果需要将这些改动合并为更少的提交（例如将 vendor 与 head 修改合并为单个“MapTR alignment”提交），我可以帮你执行 rebase + squash 再推。

- 如果你希望我把这份中文说明换成博客模板（带 YAML frontmatter）、或生成示例日志截图供发布，我可以继续处理。

*生成时间：2026-02-11*
# MapTR Alignment & BEVFormer det+map changes (2026-02-11)

This note summarizes the changes applied to bring Apollo-Vision-Net's det+map path closer to the official MapTR implementation, improve runtime observability, and add local smoke-validation tooling.

Commits included:

- `5993635` — maptr: vendor assigner/loss head/losses (official-aligned)
- `6322848` — dataset: add LiDARInstanceLines bbox and shift v1/v2
- `266d2d9` — bevformer: MapTR-official map loss path + run_cfg logging
- `18e45da` — tools/docs: improve smoke forward_train, gate model print, add walkthrough doc

---

## 1) MapTR vendoring & loss alignment

Why
- Reduce semantic differences with official MapTR (assignment / matching / loss) that can cause mismatch or numeric instability—especially under AMP/FP16.

What was done
- Vendored a lightweight subset of MapTR modules: `MapTRAssigner`, `MapTRLossHead`, and MapTR-related loss/cost implementations. These implement Hungarian matching using combined cls/reg/iou/pts costs and support OrderedPtsL1/Chamfer-style costs.
- `MapTRLossHead.loss()` casts model predictions to FP32 at entry (mimicking `@force_fp32(apply_to=('preds_dicts'))`) to align AMP behavior with official training.

Key files
- `projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py`
- `projects/mmdet3d_plugin/maptr/dense_heads/maptr_loss_head.py` (FP32 cast at loss entry)
- `projects/mmdet3d_plugin/maptr/losses/map_loss.py`

Effect / how to validate
- Matching semantics and loss terms now correspond to MapTR configs. To validate quickly, run the smoke forward script and inspect `loss_map_*` / layer-wise `d*.loss_*` values:

```bash
export PYTHONPATH=$PWD
python tools/smoke_det_map_forward_train.py projects/configs/bevformer/bev_tiny_det_map_apollo.py
```

---

## 2) Dataset: `LiDARInstanceLines.bbox` and shift protocols

Why
- MapTR relies on fixed-sampled point sequences + shift protocols to compute ordered pts cost and do matching. The assigner also needs bbox data for reg/iou costs. Dataset and head must agree on point counts and shift semantics.

What was done
- Added `LiDARInstanceLines.bbox` that returns per-line [xmin,ymin,xmax,ymax] clamped to the patch range.
- Implemented `shift_fixed_num_sampled_points`, `shift_fixed_num_sampled_points_v1`, and `shift_fixed_num_sampled_points_v2` to support closed/open polylines, padding, and optional subsampling.

Key file
- `projects/mmdet3d_plugin/datasets/nuscenes_det_occ_map_dataset.py`

Effect / how to validate
- Ensure `map_num_pts` in the config matches `fixed_ptsnum_per_line` in the dataset. The smoke script prints the shapes and counts of `gt_map_vecs_pts_loc` / `gt_map_vecs_label` once per run.

---

## 3) BEVFormer det+map head: MapTR-official path and one-shot runtime logging

Why
- Config alone is insufficient to guarantee runtime branching. To avoid “reading config and guessing running mode”, we added a lightweight runtime print that shows which MapTR-related features are active.

What was done
- When `map_loss_impl == 'maptr_official'`, the head calls `MapTRLossHead.loss()` and prefixes returned keys as `loss_map_*` so det vs map losses are separated.
- Added a rank0-only, one-time log line tagged `[det_map][run_cfg]` that prints: `map_loss_impl`, `map_use_point_queries`, `map_with_box_refine`, `map_decoder` (type), `num_layers`, `map_query_embed_type`, `num_map_vec`, `map_num_pts`, `map_num_query`, `map_pts_normalize`, `map_pts_cost_type`, etc.
- Added `DetMapTextLoggerHook` which prints `det:` and `map:` metrics on separate lines and suppresses near-zero disabled losses for readability.

Key files
- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py` (run_cfg log + maptr_official branch)
- `projects/mmdet3d_plugin/bevformer/hooks/det_map_text_logger_hook.py`

Effect / how to validate
- Start a train or smoke run and look for the `[det_map][run_cfg]` line in the logs; the hook will display `det:` and `map:` lines showing active losses cleanly.

Example log excerpt (expected):
```
[det_map][run_cfg] map_loss_impl=maptr_official map_use_point_queries=True map_with_box_refine=True map_decoder=MapTRDecoder num_layers=6 map_query_embed_type=... map_num_pts=20 map_pts_cost_type=chamfer
det: loss: 0.1234, loss_cls: 0.0123, ...
map: loss_map: 0.2345, loss_map_pts: 0.2200, loss_map_dir: 0.0145
```

---

## 4) Tools & docs: smoke script improvements and walkthrough

Why
- Want a quick, robust way to sanity-check forward/backward paths locally without heavy runners or checkpoints, and document the mapping from config → code.

What was done
- `tools/smoke_det_map_forward_train.py` improved to normalize `img`/`img_metas` layouts, register light forward hooks to trace NaNs, and wrap `gt_bboxes_3d` tensors into `LiDARInstance3DBoxes` when needed.
- `tools/train.py` gains `log_model_summary` config switch to avoid noisy model dumps.
- Added a walkthrough doc: `docs/bev_sparse_det_maptr_flashocc_henet_tinym_nuscenes.md` describing config → code mappings and dataset expectations.

How to use the smoke script

```bash
export PYTHONPATH=$PWD
python tools/smoke_det_map_forward_train.py projects/configs/bevformer/bev_tiny_det_map_apollo.py
```

---

## Notes & next steps

- Commits are pushed to the repository; the four commits listed at the top contain the full code changes. If you prefer a different commit grouping (for example squash vendor + head into a single "maptr alignment" commit), we can rebase and squash before publishing a formal changelog.

- If you'd like, I can produce a blog-ready post (with YAML frontmatter for your blog engine), or export the example logs/screenshots for the post. Let me know the desired blog layout and I will adapt the markdown accordingly.

---

*Generated: 2026-02-11*