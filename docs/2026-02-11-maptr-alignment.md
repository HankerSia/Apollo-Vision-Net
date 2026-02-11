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