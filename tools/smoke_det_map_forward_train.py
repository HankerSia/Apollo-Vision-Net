"""One-batch smoke test for det+map training forward.

Why:
- `tools/test.py` requires a checkpoint.
- We want a tiny, direct `forward_train` run to validate that the new
  attention logits clamp wiring doesn't break model construction and keeps
  losses finite.

Usage (example):
  PYTHONPATH=$PWD python tools/smoke_det_map_forward_train.py \
    projects/configs/bevformer/bev_tiny_det_map_apollo.py

Optional:
  --cfg-options model.debug_nan=True
"""

import argparse

import torch
import numpy as np

from mmcv import Config

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Ensure Apollo-Vision-Net custom datasets/models are registered.
import projects.mmdet3d_plugin  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--cfg-options', nargs='+', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict({
            k: eval(v) if isinstance(v, str) else v
            for k, v in (opt.split('=', 1) for opt in args.cfg_options)
        })

    # Make it deterministic & tiny
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0

    # Build dataset and fetch one sample
    dataset = build_dataset(cfg.data.train)
    data = dataset[0]

    # Collate a single sample into the format expected by forward_train
    # mmdet's DataContainer is usually already in the sample; model's
    # data_preprocessor/collate logic isn't used here, so we keep it simple.
    # This script intentionally targets the existing repo conventions.
    
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.train()

    # Move tensors to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Sanitize transformer embeddings if they contain non-finite values
    try:
        transformer = getattr(model.pts_bbox_head, 'transformer', None)
        if transformer is not None:
            if hasattr(transformer, 'cams_embeds') and isinstance(transformer.cams_embeds, torch.Tensor):
                if not torch.isfinite(transformer.cams_embeds).all():
                    with torch.no_grad():
                        torch.nn.init.normal_(transformer.cams_embeds, mean=0.0, std=0.01)
                        transformer.cams_embeds.data = torch.nan_to_num(
                            transformer.cams_embeds.data, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(transformer, 'level_embeds') and isinstance(transformer.level_embeds, torch.Tensor):
                if not torch.isfinite(transformer.level_embeds).all():
                    with torch.no_grad():
                        torch.nn.init.normal_(transformer.level_embeds, mean=0.0, std=0.01)
                        transformer.level_embeds.data = torch.nan_to_num(
                            transformer.level_embeds.data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass

    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            return [to_device(t) for t in x]
        return x

    # Many keys are DataContainer; unwrap via `.data` if present.
    batch = {}
    for k, v in data.items():
        if hasattr(v, 'data'):
            v = v.data
        batch[k] = to_device(v)

    # Normalize img_metas format.
    # Expected by BEVFormer: img_metas is list[ list[dict] ] (bs, len_queue)
    # Some datasets provide list[dict] or a single dict.
    if 'img_metas' in batch:
        metas = batch['img_metas']
        if isinstance(metas, dict):
            # If it's a dict-of-metas keyed by timestep, convert to an ordered list.
            if all(isinstance(v, dict) for v in metas.values()):
                try:
                    metas = [metas[k] for k in sorted(metas.keys())]
                except Exception:
                    metas = list(metas.values())
            else:
                metas = [metas]
        if isinstance(metas, list) and (len(metas) > 0) and isinstance(metas[0], dict):
            metas = [metas]  # bs=1
        # BEVFormer transformer expects can_bus to exist in each meta.
        # Some of our det+map pipelines don't include it for the smoke sample,
        # but training forward assumes it for ego-motion compensation.
        if isinstance(metas, list) and len(metas) > 0 and isinstance(metas[0], list):
            for t in range(len(metas[0])):
                if isinstance(metas[0][t], dict) and ('can_bus' not in metas[0][t]):
                    metas[0][t]['can_bus'] = [0.0] * 18
                # Encoder point_sampling expects per-frame calibration as an array
                # with shape (num_cam, 4, 4). Some pipelines may store
                # (num_cam*num_levels, 4, 4) because they duplicate per FPN level.
                # In this repo's encoder, it reads img_meta['lidar2img'] and
                # treats axis=1 as num_cam.
                if isinstance(metas[0][t], dict):
                    if 'lidar2img' not in metas[0][t]:
                        metas[0][t]['lidar2img'] = [
                            np.eye(4, dtype=np.float32) for _ in range(6)
                        ]
                    else:
                        l2i = metas[0][t]['lidar2img']
                        # Try to coerce to numpy array for shape inspection.
                        try:
                            l2i_np = np.asarray(l2i)
                            # If it's flattened across FPN levels (e.g. 24x4x4),
                            # convert to per-cam (6x4x4) by taking the first level.
                            if l2i_np.ndim == 3 and l2i_np.shape[0] == 24:
                                l2i_np = l2i_np.reshape(4, 6, 4, 4)[0]
                            # If it's (num_levels, num_cam, 4, 4), take first level.
                            if l2i_np.ndim == 4 and l2i_np.shape[0] == 4 and l2i_np.shape[1] == 6:
                                l2i_np = l2i_np[0]
                            metas[0][t]['lidar2img'] = l2i_np
                        except Exception:
                            # Keep original if conversion fails.
                            pass
        batch['img_metas'] = metas
        # Debug: summarize normalized img_metas structure
        try:
            if isinstance(metas, list):
                bs = len(metas)
                tlen = len(metas[0]) if bs > 0 and isinstance(metas[0], list) else None
                key_sample = None
                if bs > 0 and isinstance(metas[0], list) and tlen and isinstance(metas[0][0], dict):
                    key_sample = list(metas[0][0].keys())[:8]
                print(f"[smoke] img_metas: bs={bs}, len_queue={tlen}, keys(sample)={key_sample}")
        except Exception as e:
            print(f"[smoke] img_metas summary failed: {e}")

    def _normalize_lidar2img_in_place(metas_list, num_cam: int = 6):
        """Force meta['lidar2img'] to (num_cam,4,4) numpy float32.

        Handles common layouts:
        - missing: fill identity
        - (num_cam,4,4)
        - (num_levels*num_cam,4,4) e.g. 24x4x4 -> take first level
        - (num_levels,num_cam,4,4) -> take first level
        - (len_queue,num_cam,4,4) -> take first timestep
        - (len_queue,num_levels,num_cam,4,4) -> take [0,0]
        """
        if not (isinstance(metas_list, list) and len(metas_list) == 1 and isinstance(metas_list[0], list)):
            return
        for t in range(len(metas_list[0])):
            m = metas_list[0][t]
            if not isinstance(m, dict):
                continue
            if 'lidar2img' not in m:
                m['lidar2img'] = np.stack([np.eye(4, dtype=np.float32) for _ in range(num_cam)], axis=0)
                continue
            try:
                l2i_np = np.asarray(m['lidar2img'], dtype=np.float32)
            except Exception:
                continue

            # Strip time dim if present
            if l2i_np.ndim == 4 and l2i_np.shape[0] > 1 and l2i_np.shape[1] == num_cam and l2i_np.shape[2:] == (4, 4):
                l2i_np = l2i_np[0]
            if l2i_np.ndim == 5 and l2i_np.shape[0] > 1 and l2i_np.shape[2] == num_cam and l2i_np.shape[3:] == (4, 4):
                # (len_queue, num_levels, num_cam, 4, 4)
                l2i_np = l2i_np[0, 0]

            # Strip level dim if present
            if l2i_np.ndim == 4 and l2i_np.shape[0] == 4 and l2i_np.shape[1] == num_cam and l2i_np.shape[2:] == (4, 4):
                l2i_np = l2i_np[0]

            # Flattened across levels
            if l2i_np.ndim == 3 and l2i_np.shape[0] == 4 * num_cam and l2i_np.shape[1:] == (4, 4):
                l2i_np = l2i_np.reshape(4, num_cam, 4, 4)[0]

            # If still not (num_cam,4,4), fall back to identity
            if not (l2i_np.ndim == 3 and l2i_np.shape[0] == num_cam and l2i_np.shape[1:] == (4, 4)):
                l2i_np = np.stack([np.eye(4, dtype=np.float32) for _ in range(num_cam)], axis=0)

            m['lidar2img'] = l2i_np

    # Ensure lidar2img matches encoder.point_sampling expectations.
    if 'img_metas' in batch:
        _normalize_lidar2img_in_place(batch['img_metas'], num_cam=6)

    # Some codepaths (history bev) expect img_metas to match the temporal length
    # of prev_img/img. If we duplicated frames later, we must duplicate metas too.
    # Also, BEVFormer encoder.point_sampling expects lidar2img per frame with shape
    # (num_cam, 4, 4) (no time dimension). If a pipeline provides time-stacked
    # lidar2img (len_queue, num_cam, 4, 4), keep only the current frame.
    if 'img_metas' in batch and isinstance(batch['img_metas'], list) and len(batch['img_metas']) == 1:
        if isinstance(batch['img_metas'][0], list):
            for t in range(len(batch['img_metas'][0])):
                m = batch['img_metas'][0][t]
                if isinstance(m, dict) and 'lidar2img' in m:
                    try:
                        l2i_np = np.asarray(m['lidar2img'])
                        if l2i_np.ndim == 4 and l2i_np.shape[1] == 6 and l2i_np.shape[2:] == (4, 4):
                            # likely (len_queue, num_cam, 4, 4)
                            m['lidar2img'] = l2i_np[0]
                    except Exception:
                        pass

    # BEVFormer expects temporal queue shape:
    #   img: (bs, len_queue, num_cams, C, H, W)
    # Some pipelines yield (bs, num_cams, C, H, W) for single frame.
    if 'img' in batch and isinstance(batch['img'], torch.Tensor):
        # Common cases:
        # - (num_cams, C, H, W)  -> add bs
        # - (bs, num_cams, C, H, W) -> add len_queue
        if batch['img'].dim() == 4:
            batch['img'] = batch['img'].unsqueeze(0)
        if batch['img'].dim() == 5:
            batch['img'] = batch['img'].unsqueeze(1)

    # BEVFormer forward_train slices prev_img = img[:, :-1].
    # If len_queue==1, prev_img is empty and downstream reshape breaks.
    # Duplicate the single frame to create a minimal history of length 2.
    if 'img' in batch and isinstance(batch['img'], torch.Tensor) and batch['img'].dim() == 6:
        if batch['img'].size(1) == 1:
            batch['img'] = torch.cat([batch['img'], batch['img']], dim=1)
            # keep metas in sync with len_queue
            if 'img_metas' in batch and isinstance(batch['img_metas'], list) and len(batch['img_metas']) == 1:
                if isinstance(batch['img_metas'][0], list) and len(batch['img_metas'][0]) == 1:
                    batch['img_metas'][0] = [batch['img_metas'][0][0], batch['img_metas'][0][0]]

    # Re-normalize lidar2img after potential meta duplication.
    if 'img_metas' in batch:
        _normalize_lidar2img_in_place(batch['img_metas'], num_cam=6)

    # Debug: print final lidar2img shapes to ensure encoder.point_sampling contract.
    if 'img_metas' in batch and isinstance(batch['img_metas'], list) and len(batch['img_metas']) == 1 and isinstance(batch['img_metas'][0], list):
        for ti, m in enumerate(batch['img_metas'][0]):
            if isinstance(m, dict) and 'lidar2img' in m:
                try:
                    l2i_np = np.asarray(m['lidar2img'])
                    print(f"[smoke] img_metas[0][{ti}].lidar2img shape={l2i_np.shape} dtype={l2i_np.dtype}")
                except Exception as e:
                    print(f"[smoke] img_metas[0][{ti}].lidar2img: failed to inspect ({e})")

    # ---- NaN/Inf tracing helpers ----
    def _tensor_stats(t):
        if not isinstance(t, torch.Tensor):
            return None
        if t.numel() == 0:
            return None
        finite = torch.isfinite(t)
        ratio = float((~finite).float().mean().detach().cpu())
        t_min = float(t.min().detach().cpu()) if t.numel() > 0 else None
        t_max = float(t.max().detach().cpu()) if t.numel() > 0 else None
        return ratio, t_min, t_max, tuple(t.shape), str(t.dtype), str(t.device)

    def _print_stats(tag, t):
        stats = _tensor_stats(t)
        if stats is None:
            return
        ratio, t_min, t_max, shape, dtype, device = stats
        if ratio > 0:
            print(f"[nan-trace] {tag}: nonfinite_ratio={ratio:.6f} shape={shape} dtype={dtype} device={device} min={t_min} max={t_max}")

    # Check input images before any forward.
    if isinstance(batch.get('img', None), torch.Tensor):
        stats = _tensor_stats(batch['img'])
        if stats is not None:
            ratio, t_min, t_max, shape, dtype, device = stats
            print(f"[nan-trace] input.img: nonfinite_ratio={ratio:.6f} shape={shape} dtype={dtype} device={device} min={t_min} max={t_max}")

    # Register lightweight forward hooks to locate first non-finite activations.
    nan_trace_state = {'count': 0, 'max': 8}

    def _hook_fn(name):
        def _inner(module, inp, out):
            if nan_trace_state['count'] >= nan_trace_state['max']:
                return
            def _scan(obj, prefix):
                if isinstance(obj, torch.Tensor):
                    stats = _tensor_stats(obj)
                    if stats is None:
                        return
                    ratio, t_min, t_max, shape, dtype, device = stats
                    if ratio > 0:
                        print(f"[nan-trace] {prefix}: nonfinite_ratio={ratio:.6f} shape={shape} dtype={dtype} device={device} min={t_min} max={t_max}")
                        nan_trace_state['count'] += 1
                elif isinstance(obj, (list, tuple)):
                    for i, v in enumerate(obj):
                        _scan(v, f"{prefix}[{i}]")
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        _scan(v, f"{prefix}.{k}")
            _scan(inp, f"{name}.in")
            _scan(out, f"{name}.out")
        return _inner

    hooks = []
    # Focus on common NaN sources: backbone/neck/transformer/encoder/pts head
    hook_modules = {
        'img_backbone': getattr(model, 'img_backbone', None),
        'img_neck': getattr(model, 'img_neck', None),
        'pts_bbox_head': getattr(model, 'pts_bbox_head', None),
    }
    for n, m in hook_modules.items():
        if m is not None:
            hooks.append(m.register_forward_hook(_hook_fn(n)))

    # Deeper hooks inside transformer/encoder layers (if present)
    try:
        transformer = getattr(model.pts_bbox_head, 'transformer', None)
        if transformer is not None:
            hooks.append(transformer.register_forward_hook(_hook_fn('pts_bbox_head.transformer')))
            # Inspect transformer embedding params for non-finite values.
            try:
                for name, param in [
                    ('transformer.cams_embeds', getattr(transformer, 'cams_embeds', None)),
                    ('transformer.level_embeds', getattr(transformer, 'level_embeds', None)),
                ]:
                    if isinstance(param, torch.Tensor):
                        stats = _tensor_stats(param)
                        if stats is not None:
                            ratio, t_min, t_max, shape, dtype, device = stats
                            print(f"[nan-trace] {name}: nonfinite_ratio={ratio:.6f} shape={shape} dtype={dtype} device={device} min={t_min} max={t_max}")
            except Exception:
                pass
            encoder = getattr(transformer, 'encoder', None)
            if encoder is not None:
                hooks.append(encoder.register_forward_hook(_hook_fn('pts_bbox_head.transformer.encoder')))
                layers = getattr(encoder, 'layers', None)
                if layers is not None:
                    for i, layer in enumerate(layers):
                        hooks.append(layer.register_forward_hook(_hook_fn(f'encoder.layers[{i}]')))
    except Exception:
        pass

    # Inspect img feature levels before transformer to localize NaNs.
    try:
        if isinstance(batch.get('img', None), torch.Tensor) and 'img_metas' in batch:
            img_tensor = batch['img']
            if img_tensor.dim() == 6:
                len_queue = img_tensor.size(1)
                img_cur = img_tensor[:, -1, ...]
                metas_cur = [each[len_queue - 1] for each in batch['img_metas']]
            else:
                img_cur = img_tensor
                metas_cur = batch['img_metas']
            feats = model.extract_feat(img=img_cur, img_metas=metas_cur, points=None)
            if isinstance(feats, (list, tuple)):
                for i, f in enumerate(feats):
                    stats = _tensor_stats(f)
                    if stats is not None:
                        ratio, t_min, t_max, shape, dtype, device = stats
                        print(f"[nan-trace] img_feats[{i}]: nonfinite_ratio={ratio:.6f} shape={shape} dtype={dtype} device={device} min={t_min} max={t_max}")
    except Exception as e:
        print(f"[nan-trace] extract_feat check failed: {e}")

    with torch.no_grad():
        losses = model.forward_train(**batch)

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Print a compact summary
    print('=== forward_train losses ===')
    for k, v in losses.items():
        if torch.is_tensor(v):
            print(f'{k}: {float(v.detach().cpu())}')
        else:
            print(f'{k}: {v}')


if __name__ == '__main__':
    main()
