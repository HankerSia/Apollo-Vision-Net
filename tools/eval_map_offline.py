#!/usr/bin/env python3

"""Offline MapTR-style map evaluation for Apollo-Vision-Net.

Why this script exists:

- This workspace may expose `mmdet3d` from MapTR (editable install at
    `/home/nuvo/MapTR/mmdetection3d`) and/or other installs. That is fine.
- The *real* reason offline `build_dataset(...)` often fails is that plugin
    modules were not imported, so Apollo's custom datasets/heads are not
    registered (e.g. `CustomNuScenesDetMapDataset`).

This script ensures:

1) Apollo repo root is first on `sys.path` (so `projects.*` is importable)
2) The plugin specified by the config is imported (same logic as `tools/test.py`)
3) Then runs `dataset.evaluate_map(...)` on a dumped `map_results.pkl`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


def _ensure_apollo_root_first() -> str:
    apollo_root = str(Path(__file__).resolve().parents[1])
    if sys.path[0] != apollo_root:
        # Put repo root at the very front to make local `projects.*` importable.
        sys.path.insert(0, apollo_root)
    return apollo_root


def _ensure_mmdet3d_importable(apollo_root: str) -> Optional[str]:
    """Ensure `import mmdet3d` works.

    In this workspace, `mmdet3d` is often provided by an editable install of
    `/home/nuvo/MapTR/mmdetection3d`. If it's not installed, we fall back to
    temporarily adding that path.

    Returns:
        added_path: the path we added to sys.path, if any.
    """

    try:
        import mmdet3d  # noqa: F401
        return None
    except Exception:
        pass

    # Fallback: known location in this environment.
    candidate = '/home/nuvo/MapTR/mmdetection3d'
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
        return candidate

    # Last resort: no known source.
    raise ModuleNotFoundError(
        'Cannot import mmdet3d. Install mmdetection3d (mmdet3d) into the current '
        'env, or ensure it is discoverable via PYTHONPATH.')


def _import_plugin_from_cfg(cfg, config_path: str) -> None:
    """Mirror plugin import logic used by `tools/test.py`."""

    if getattr(cfg, 'custom_imports', None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib

        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            module_dir = os.path.dirname(plugin_dir)
        else:
            module_dir = os.path.dirname(config_path)

        module_dir_parts = module_dir.split('/')
        module_path = module_dir_parts[0]
        for m in module_dir_parts[1:]:
            module_path = module_path + '.' + m

        importlib.import_module(module_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Offline map evaluation (MapTR protocol) without dist_test')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        'map_pkl',
        help='path to dumped map_results.pkl (from dist_test/tools/test.py)')
    parser.add_argument(
        '--eval',
        nargs='+',
        default=['chamfer'],
        choices=['chamfer', 'iou'],
        help='map metrics to evaluate')
    parser.add_argument(
        '--out-dir',
        default=None,
        help='output directory for formatted json/cache (default: alongside pkl)')
    parser.add_argument(
        '--split',
        default='val',
        choices=['train', 'val', 'test'],
        help='which dataset split to build from config (default: val)')
    parser.add_argument(
        '--nproc',
        type=int,
        default=None,
        help='override map_eval_nproc (default: use config/dataset)')

    return parser.parse_args()


def main() -> None:
    apollo_root = _ensure_apollo_root_first()
    added_mmdet3d_path = _ensure_mmdet3d_importable(apollo_root)

    from mmcv import Config
    import mmcv

    cfg = Config.fromfile(os.path.abspath(args.config))

    # Ensure plugin datasets/heads are registered.
    _import_plugin_from_cfg(cfg, os.path.abspath(args.config))

    # Build dataset from cfg.
    from mmdet3d.datasets import build_dataset

    split_cfg = getattr(cfg.data, args.split)
    dataset = build_dataset(split_cfg)

    if args.nproc is not None:
        # Dataset stores this as self.map_eval_nproc.
        if hasattr(dataset, 'map_eval_nproc'):
            dataset.map_eval_nproc = int(args.nproc)

    map_results = mmcv.load(args.map_pkl)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.map_pkl)), 'map_eval_offline')

    print('[eval_map_offline] apollo_root:', apollo_root)
    if added_mmdet3d_path is not None:
        print('[eval_map_offline] added mmdet3d search path:', added_mmdet3d_path)

    # Show which mmdet3d we actually imported.
    import mmdet3d as imported_mmdet3d
    print('[eval_map_offline] mmdet3d.__file__:', getattr(imported_mmdet3d, '__file__', None))

    if not hasattr(dataset, 'evaluate_map'):
        raise RuntimeError(
            f'Dataset type {type(dataset)} has no evaluate_map(). '
            'Did the plugin import run correctly?')

    metrics = dataset.evaluate_map(
        map_results,
        metric=args.eval,
        jsonfile_prefix=out_dir,
    )

    print('[eval_map_offline] metrics:')
    print(metrics)


if __name__ == '__main__':
    args = parse_args()
    main()
