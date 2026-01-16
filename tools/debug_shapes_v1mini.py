#!/usr/bin/env python3
"""Quick shape sanity-check for Apollo-Vision-Net (v1.0-mini).

This script:
  1) Loads the config (pipelines, dataset)
  2) Builds the dataset + dataloader (batch=1)
  3) Builds the model
  4) Runs a single forward pass (no grad)

It prints the key tensor/meta shapes so you can trace:
  ori_shape -> img_shape/pad -> (B, Nc, C, H, W) mlvl_feats
  -> BEVFormer head outputs (det + occ)

Usage (example):
    python tools/debug_shapes_v1mini.py \
      --config projects/configs/bevformer/bev_tiny_det_occ_apollo.py \
      --dataroot data/nuscenes \
      --version v1.0-mini

Notes:
  - This is a debug utility; it doesn't run evaluation.
  - It assumes the plugin path in your config is correct.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import DataContainer

# Ensure we can import the local plugin from repo root when running as
# `python tools/debug_shapes_v1mini.py`.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Ensure the mmdet3d plugin (datasets/models) is registered.
# This mirrors how `tools/train.py` / `tools/test.py` load the plugin.
import projects.mmdet3d_plugin  # noqa: F401

from mmdet3d.datasets import build_dataset
from mmdet3d.datasets import build_dataloader
from mmdet3d.models import build_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', default='', help='Optional checkpoint to load')
    p.add_argument('--dataroot', default='data/nuscenes')
    p.add_argument('--version', default='v1.0-mini')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def _print_one_img_meta(img_meta: dict):
    keys = ['ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'filename']
    for k in keys:
        if k in img_meta:
            v = img_meta[k]
            if k == 'filename' and isinstance(v, (list, tuple)):
                # multi-view filenames can be long
                v = [osp.basename(x) for x in v]
            print(f'  - {k}: {v}')


def _summarize(obj, prefix: str = '', max_list: int = 3):
    """Recursively summarize nested outputs.

    - Tensors: print shape/dtype/device
    - dict/list/tuple: recurse (truncate long lists)
    """
    if torch.is_tensor(obj):
        return f"{prefix}Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    if isinstance(obj, dict):
        lines = [f"{prefix}dict(keys={list(obj.keys())})"]
        for k, v in obj.items():
            lines.append(_summarize(v, prefix + f"  [{k}] ", max_list=max_list))
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        tname = type(obj).__name__
        lines = [f"{prefix}{tname}(len={len(obj)})"]
        for i, v in enumerate(obj[:max_list]):
            lines.append(_summarize(v, prefix + f"  [{i}] ", max_list=max_list))
        if len(obj) > max_list:
            lines.append(f"{prefix}  ... (truncated)")
        return "\n".join(lines)
    return f"{prefix}{type(obj).__name__}({obj})"


def main():
    args = _parse_args()

    cfg = Config.fromfile(args.config)

    # Override dataset root/version for debug.
    for split in ['train', 'val', 'test']:
        if 'data' in cfg and split in cfg.data:
            cfg.data[split].data_root = args.dataroot
            # Version is usually encoded in ann_file / dataset settings; not all dataset
            # classes accept a `version` kwarg, so only set it when present.
            if 'version' in cfg.data[split]:
                cfg.data[split].version = args.version

    # Build dataset/dataloader (use test split pipeline).
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    # Build model
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Hook: capture the first level's mlvl_feats shape when entering pts_bbox_head
    dbg = {}

    def _hook_pts_head(module, inputs):
        # Some heads take (mlvl_feats, img_metas, ...)
        if not inputs:
            return
        mlvl_feats = inputs[0]
        if isinstance(mlvl_feats, (list, tuple)) and len(mlvl_feats) > 0 and torch.is_tensor(mlvl_feats[0]):
            dbg['mlvl_feats0'] = tuple(mlvl_feats[0].shape)

    if hasattr(model, 'pts_bbox_head'):
        model.pts_bbox_head.register_forward_pre_hook(_hook_pts_head)

    def _hook_pts_head_out(module, inputs, output):
        # output should be a dict returned by BEVFormerOccupancyHeadApollo.forward
        if not isinstance(output, dict):
            return
        for k in ['bev_embed', 'all_cls_scores', 'all_bbox_preds', 'occupancy_preds', 'flow_preds']:
            v = output.get(k, None)
            if torch.is_tensor(v):
                dbg[f'outs.{k}'] = tuple(v.shape)
            elif v is None:
                dbg[f'outs.{k}'] = None
        # For GroupDETR, cls/bbox are often stacked as (num_dec_layers, bs, num_query, ...)

    if hasattr(model, 'pts_bbox_head'):
        model.pts_bbox_head.register_forward_hook(_hook_pts_head_out)

    batch = next(iter(data_loader))

    def unwrap_dc(x):
        # mmcv wraps tensors/metadata in DataContainer for collate.
        if isinstance(x, DataContainer):
            return x.data
        return x

    # Move tensors to device, unwrapping DataContainer as needed.
    def to_device(x):
        x = unwrap_dc(x)
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(to_device(v) for v in x)
        return x

    batch = to_device(batch)

    # Extract img + img_metas in a robust way.
    img = unwrap_dc(batch.get('img', None))
    img_metas = unwrap_dc(batch.get('img_metas', None))

    print('\n### Batch keys')
    print(sorted(list(batch.keys())))

    # Common shapes after unwrap:
    #   img: [Tensor] where Tensor is (B, Ncam, C, H, W)
    #   img_metas: [[dict]] (outer batch, inner queue length)
    if isinstance(img, (list, tuple)) and len(img) == 1 and torch.is_tensor(img[0]):
        img = img[0]
    if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
        img_metas = img_metas[0]

    print('\n### img tensor')
    if torch.is_tensor(img):
        print('img.shape =', tuple(img.shape))
    elif isinstance(img, (list, tuple)) and len(img) == 1 and torch.is_tensor(img[0]):
        print('img[0].shape =', tuple(img[0].shape))
    else:
        print('img is not a Tensor, type=', type(img))

    print('\n### img_metas (first sample)')
    if isinstance(img_metas, (list, tuple)) and len(img_metas) > 0:
        # BEVFormer often uses a temporal queue: img_metas[t] is a dict.
        # We'll show the first timestep's meta.
        if isinstance(img_metas[0], dict):
            _print_one_img_meta(img_metas[0])
        elif isinstance(img_metas[0], (list, tuple)) and len(img_metas[0]) > 0 and isinstance(img_metas[0][0], dict):
            _print_one_img_meta(img_metas[0][0])
    else:
        print('img_metas is not a list/tuple, type=', type(img_metas))

    # Run forward. Different wrappers exist; `forward_test` is most stable.
    with torch.no_grad():
        # Ensure `img` is a Tensor and `img_metas` is list-of-list-of-dict as expected.
        # Unwrap img to a Tensor.
        # Common patterns:
        #   img is Tensor(B, N, C, H, W)
        #   img is [Tensor(...)]
        #   img is [[Tensor(...)]], if a queue dimension is wrapped as list
        while isinstance(img, (list, tuple)) and len(img) == 1:
            img = img[0]
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = img_metas[0]

        # BEVFormer forward_test expects img_metas to be nested (batch, queue).
        # Ensure forward_test sees: img=[img_tensor], img_metas=[[meta_dict]]
        if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], dict):
            img_metas_btq = [[img_metas[0]]]
        elif (
            isinstance(img_metas, list)
            and len(img_metas) > 0
            and isinstance(img_metas[0], list)
            and len(img_metas[0]) > 0
            and isinstance(img_metas[0][0], dict)
        ):
            img_metas_btq = [[img_metas[0][0]]]
        else:
            img_metas_btq = [[img_metas]]

        outputs = model(return_loss=False, rescale=True, img=[img], img_metas=img_metas_btq)

    if 'mlvl_feats0' in dbg:
        print('\n### hooked mlvl_feats[0]')
        print('mlvl_feats[0].shape =', dbg['mlvl_feats0'])

    # Head internal outputs (from transformer+decoder)
    if any(k.startswith('outs.') for k in dbg.keys()):
        print('\n### hooked pts_bbox_head outs (key tensor shapes)')
        for k in ['outs.bev_embed', 'outs.all_cls_scores', 'outs.all_bbox_preds', 'outs.occupancy_preds', 'outs.flow_preds']:
            if k in dbg:
                print(f'{k} = {dbg[k]}')

    print('\n### model outputs type')
    print(type(outputs))
    if isinstance(outputs, (list, tuple)):
        print('len(outputs)=', len(outputs))

    print('\n### model outputs summary (truncated)')
    print(_summarize(outputs, max_list=2))

    print('\nDone.')


if __name__ == '__main__':
    main()
