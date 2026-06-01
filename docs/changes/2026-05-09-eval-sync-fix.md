近期功能更新与调试测试报告（2026-05-09）

本文档记录了最近对训练期评估同步逻辑与地图分支（map）相关实现所做的修复、验证步骤与排查建议，便于快速了解改动影响与后期复现。



已实施的修复（要点）
- 模型val异常修复：

  问题背景与目标
  训练过程中观察到多卡分布式训练在评估阶段出现不一致步调：评估操作（尤其 map 评估）仅由主卡（rank 0）完整执行并负责保存最佳模型，而其他卡在收集完中间结果后过早返回训练循环并参与下一轮 loss 的通信（如 all_reduce/reduce_mean）。若主卡仍在进行耗时评估，下一轮通信会缺失该卡参与，最终可能触发 NCCL 超时（长期等待通信导致的 timeout）。

  为解决该问题，本次更新目标是：在不依赖全局 NCCL barrier 的前提下保证所有非主卡在主卡完成评估并写入状态后，才进入下一轮可能触发分布式通信的训练步骤；同时修正 map 分支在损失记账与 decoder 行为上的若干问题，避免配置与实际执行路径不一致。

  实现方式：在 `projects/mmdet3d_plugin/core/evaluation/eval_hooks.py` 中实现基于文件系统的主从同步机制：主卡在评估与 `save_best` 完成后写入完成标记（`.eval_hook_sync/<run>.done`），若评估异常写入 `.fail`；非主卡在 CPU 侧轮询该目录等待 `.done` 或 `.fail`，期间不参与任何 NCCL 通信，从而避免在主卡仍评估时触发下一轮通信。此方案规避了使用 `dist.barrier()`（或其它会触发 NCCL 流量的同步手段），减少因评估耗时导致的通信超时风险。

  验证与测试建议

  ```bash
  python3 -m py_compile projects/mmdet3d_plugin/core/evaluation/eval_hooks.py
  ```

  - 小规模分布式回归测试（建议在 2 卡或 4 卡上复现）：使用 `torch.distributed.run`（或集群上常用的 launcher）启动短训练任务并开启验证，观察非主卡是否在主卡完成评估并写入 `.eval_hook_sync/` 前保持等待状态。例如：

  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 tools/train.py configs/your_config.py --validate
  ```

  在运行时关注点：
    - 日志中是否出现主卡写 `.eval_hook_sync/*.done`（或 `.fail`）的记录；
    - 非主卡是否进入等待（可以在训练日志或在节点上检查 `.eval_hook_sync/` 目录）；
    - 是否再现此前的 NCCL 超时（若修复生效，应不会因评估未完成而缺失卡导致长时间 NCCL timeout）。


- MapTRv2 / BEVFormer 相关改动：
  - `projects/mmdet3d_plugin/maptrv2/dense_heads/bevformer_det_map_head_apollo_v2.py`：将 MapTRv2 decoder 设为严格模式，新增 decoder 合约校验（例如 `map_num_query == num_map_vec * map_num_pts`、分支数量覆盖校验、`dec_states/dec_references` 维度校验、提示 `return_intermediate=True` 缺失等）；fallback 仅允许白名单初始化类异常，其他异常一律抛出以避免静默回退导致实验配置与实际执行不一致；修复 `loss_anchor` 在 `one2many_preds is None` 时可能的二次报错。
  - `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py`：将 map 主分支的聚合总项调整为仅用于日志展示（`map_main_total`），避免同时参与 `_parse_losses` 导致重复累计。
  - `projects/mmdet3d_plugin/maptrv2/dense_heads/bevformer_det_map_head_apollo_v2.py`：统一 map 损失记账方式，保留实际参与反传的 `loss_map_*` 项，新增用于观察的 `map_total` 汇总项。
  - `projects/mmdet3d_plugin/bevformer/hooks/det_map_text_logger_hook.py`：logger 兼容新增的 `map_*` 汇总字段，保证输出一致性。

受影响文件（供快速查阅）
- [projects/mmdet3d_plugin/core/evaluation/eval_hooks.py](projects/mmdet3d_plugin/core/evaluation/eval_hooks.py)
- [projects/mmdet3d_plugin/maptrv2/dense_heads/bevformer_det_map_head_apollo_v2.py](projects/mmdet3d_plugin/maptrv2/dense_heads/bevformer_det_map_head_apollo_v2.py)
- [projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py](projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_det_map_head_apollo.py)
- [projects/mmdet3d_plugin/bevformer/hooks/det_map_text_logger_hook.py](projects/mmdet3d_plugin/bevformer/hooks/det_map_text_logger_hook.py)




