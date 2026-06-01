
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os
import os.path as osp
import time

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _eval_sync_paths(self, runner):
        sync_dir = osp.join(runner.work_dir, '.eval_hook_sync')
        eval_id = f'epoch_{runner.epoch}_iter_{runner.iter}'
        return (
            sync_dir,
            osp.join(sync_dir, f'{eval_id}.done'),
            osp.join(sync_dir, f'{eval_id}.fail'),
        )

    def _wait_for_rank0_eval(self, runner, done_path, fail_path):
        timeout = float(os.environ.get('AVN_EVAL_SYNC_TIMEOUT', 24 * 60 * 60))
        poll_interval = float(os.environ.get('AVN_EVAL_SYNC_POLL_INTERVAL', 5))
        start_time = time.time()
        while True:
            if osp.exists(done_path):
                return
            if osp.exists(fail_path):
                raise RuntimeError(
                    f'rank 0 evaluation failed; see log and {fail_path}')
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f'timed out waiting for rank 0 evaluation sync: {done_path}')
            time.sleep(poll_interval)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        sync_dir, done_path, fail_path = self._eval_sync_paths(runner)
        if runner.rank == 0:
            mmcv.mkdir_or_exist(sync_dir)
            for path in (done_path, fail_path):
                if osp.exists(path):
                    os.remove(path)

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            try:
                print('\n')
                runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

                bbox_predictions = results['bbox_results']
                map_predictions = results.get('map_results', None)
                occupancy_results = results['occupancy_results']
                flow_results = results['flow_results']
                eval_results = {}

                if occupancy_results is not None:
                    self.dataloader.dataset.evaluate_occ_iou(occupancy_results,
                                                             flow_results,
                                                             occ_threshold=0.25,
                                                             runner=runner)
                if bbox_predictions is not None:
                    bbox_eval_kwargs = dict(getattr(self, 'eval_kwargs', {}))
                    bbox_eval_kwargs.pop('map_metric', None)
                    bbox_results = self.dataloader.dataset.evaluate(
                        bbox_predictions, logger=runner.logger, **bbox_eval_kwargs)
                    eval_results.update(bbox_results)

                if map_predictions is not None and hasattr(self.dataloader.dataset, 'evaluate_map'):
                    map_eval_kwargs = dict(getattr(self, 'eval_kwargs', {}))
                    map_metric = map_eval_kwargs.pop('map_metric', 'chamfer')
                    map_results = self.dataloader.dataset.evaluate_map(
                        map_predictions,
                        metric=map_metric,
                        logger=runner.logger,
                        **map_eval_kwargs)
                    eval_results.update(map_results)

                for name, val in eval_results.items():
                    runner.log_buffer.output[name] = val
                if eval_results:
                    runner.log_buffer.ready = True

                if self.save_best and eval_results:
                    key_indicator = getattr(self, 'key_indicator', self.save_best)
                    if key_indicator == 'auto':
                        key_indicator = next(iter(eval_results))
                        self.key_indicator = key_indicator
                    if key_indicator in eval_results:
                        self._save_ckpt(runner, eval_results[key_indicator])

                with open(done_path, 'w') as f:
                    f.write('done\n')
            except Exception as e:
                with open(fail_path, 'w') as f:
                    f.write(repr(e))
                raise
        else:
            self._wait_for_rank0_eval(runner, done_path, fail_path)
