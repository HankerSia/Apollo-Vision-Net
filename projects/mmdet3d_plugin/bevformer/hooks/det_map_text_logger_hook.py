from __future__ import annotations

from collections import OrderedDict

from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook


@HOOKS.register_module()
class DetMapTextLoggerHook(TextLoggerHook):
    """Text logger that prints det/map losses on separate lines.

    - Keeps the default prefix (epoch/iter, lr, eta, time, memory).
    - Splits metrics into: total/misc, det losses, map losses.
    - Suppresses metrics that are effectively zero (typically disabled losses).
    """

    def __init__(
        self,
        by_epoch=True,
        interval=10,
        ignore_last=True,
        reset_flag=False,
        interval_exp_name=1000,
        out_dir=None,
        out_suffix=(".log.json", ".log", ".py"),
        keep_local=True,
        file_client_args=None,
        *,
        zero_eps: float = 1e-12,
    ):
        super().__init__(
            by_epoch=by_epoch,
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            interval_exp_name=interval_exp_name,
            out_dir=out_dir,
            out_suffix=out_suffix,
            keep_local=keep_local,
            file_client_args=file_client_args,
        )
        self.zero_eps = float(zero_eps)

    def _log_info(self, log_dict, runner):
        # Preserve upstream exp_name printing behavior.
        if runner.meta is not None and "exp_name" in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                self.by_epoch and self.end_of_epoch(runner)
            ):
                runner.logger.info(f'Exp name: {runner.meta["exp_name"]}')

        prefix = self._format_prefix(log_dict, runner)
        misc_items, det_items, map_items = self._split_items(log_dict)

        if misc_items:
            runner.logger.info(prefix + ", ".join(misc_items))
        else:
            runner.logger.info(prefix.rstrip(", "))

        if det_items:
            runner.logger.info("det: " + ", ".join(det_items))
        if map_items:
            runner.logger.info("map: " + ", ".join(map_items))

    def _format_prefix(self, log_dict, runner) -> str:
        if log_dict["mode"] == "train":
            lr = log_dict.get("lr")
            if isinstance(lr, dict):
                lr_str = " ".join([f"lr_{k}: {v:.3e}" for k, v in lr.items()])
            else:
                lr_str = f"lr: {lr:.3e}" if lr is not None else ""

            if self.by_epoch:
                prefix = (
                    f'Epoch [{log_dict["epoch"]}]'
                    f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
                )
            else:
                prefix = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'

            if lr_str:
                prefix += f"{lr_str}, "

            if "time" in log_dict:
                # keep upstream ETA behavior via parent's time accumulation
                self.time_sec_tot += (log_dict["time"] * self.interval)
                time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                import datetime

                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                prefix += (
                    f"eta: {eta_str}, "
                    f"time: {log_dict['time']:.3f}, "
                    f"data_time: {log_dict['data_time']:.3f}, "
                )
                if "memory" in log_dict:
                    prefix += f"memory: {log_dict['memory']}, "
            return prefix

        # val/test
        if self.by_epoch:
            return (
                f'Epoch({log_dict["mode"]}) '
                f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            )
        return f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

    def _split_items(self, log_dict):
        skip_keys = {
            "mode",
            "Epoch",
            "iter",
            "lr",
            "time",
            "data_time",
            "memory",
            "epoch",
        }

        def should_print(name, val) -> bool:
            if name in skip_keys:
                return False
            if isinstance(val, (float, int)):
                if abs(float(val)) < self.zero_eps:
                    return False
            return True

        def fmt(name, val) -> str:
            if isinstance(val, float):
                val = f"{val:.4f}"
            return f"{name}: {val}"

        det_items = []
        map_items = []
        misc_items = []

        # Keep overall loss/grad_norm near the front in misc.
        priority_misc = ["loss", "grad_norm"]
        for key in priority_misc:
            if key in log_dict and should_print(key, log_dict[key]):
                misc_items.append(fmt(key, log_dict[key]))

        for name, val in log_dict.items():
            if not should_print(name, val):
                continue
            if name in priority_misc:
                continue

            is_map = name.startswith("loss_map") or ".loss_map" in name
            is_det = (
                (name.startswith("loss_") and not name.startswith("loss_map"))
                or (".loss_" in name and ".loss_map" not in name)
            )

            if is_map:
                map_items.append(fmt(name, val))
            elif is_det:
                det_items.append(fmt(name, val))
            else:
                misc_items.append(fmt(name, val))

        return misc_items, det_items, map_items
