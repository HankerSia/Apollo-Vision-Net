#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse an MMDetection/MMEngine JSON log (JSON lines) and plot loss curves.
Saves PNG to the work_dirs folder by default.
"""
import json
import os
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def parse_log(path):
    train = defaultdict(lambda: {'xs': [], 'ys': []})
    val = defaultdict(lambda: {'xs': [], 'ys': []})

    # Read and collect train entries first so we can compute iters_per_epoch
    train_entries = []  # list of tuples (epoch, iter_in_epoch, {loss_key: value})
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                mode = d.get('mode', '')
                if mode == 'train':
                    epoch = d.get('epoch', None)
                    iter_in_epoch = d.get('iter', None)
                    losses = {}
                    for k, v in d.items():
                        if is_number(v) and 'loss' in k:
                            losses[k] = v
                    train_entries.append((epoch, iter_in_epoch, losses))
                elif mode == 'val':
                    epoch = d.get('epoch', None)
                    for k, v in d.items():
                        if is_number(v) and 'loss' in k:
                            x = epoch if epoch is not None else (len(val[k]['xs']) + 1)
                            val[k]['xs'].append(x)
                            val[k]['ys'].append(v)
    except FileNotFoundError:
        raise

    # Determine iterations per epoch from the largest 'iter' value seen per epoch
    iters_per_epoch = None
    per_epoch_max = {}
    for epoch, iter_in_epoch, _ in train_entries:
        if epoch is not None and iter_in_epoch is not None:
            try:
                it = int(iter_in_epoch)
            except Exception:
                continue
            per_epoch_max[int(epoch)] = max(per_epoch_max.get(int(epoch), 0), it)
    if per_epoch_max:
        iters_per_epoch = max(per_epoch_max.values())

    # Populate train data using global iteration:
    if iters_per_epoch is None:
        # fallback: sequential counter if no epoch/iter info available
        step = 0
        for _, _, losses in train_entries:
            step += 1
            for k, v in losses.items():
                train[k]['xs'].append(step)
                train[k]['ys'].append(v)
    else:
        last_step = 0
        for epoch, iter_in_epoch, losses in train_entries:
            if epoch is not None and iter_in_epoch is not None:
                try:
                    e = int(epoch)
                    it = int(iter_in_epoch)
                    step = (e - 1) * int(iters_per_epoch) + it
                except Exception:
                    last_step += 1
                    step = last_step
            else:
                last_step += 1
                step = last_step
            last_step = max(last_step, step)
            for k, v in losses.items():
                train[k]['xs'].append(step)
                train[k]['ys'].append(v)
    return train, val


def smooth(y, window=20):
    if len(y) <= 1:
        return y
    if len(y) < window:
        # small smoothing: simple running mean with smaller window
        window = max(1, len(y) // 4)
    arr = np.array(y, dtype=float)
    if window <= 1:
        return arr.tolist()
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    res = (cumsum[window:] - cumsum[:-window]) / float(window)
    pad = [float('nan')] * (window - 1)
    return pad + res.tolist()


def choose_keys(train_data):
    # prefer these keys if present
    priority = ['loss', 'loss_cls', 'loss_bbox', 'loss_map', 'loss_map_pts', 'loss_map_seg']
    keys = []
    for k in priority:
        if k in train_data:
            keys.append(k)
    # add other loss keys (limit to keep plot readable)
    for k in train_data:
        if k in keys:
            continue
        keys.append(k)
        if len(keys) >= 8:
            break
    return keys


def plot(train_data, val_data, out_png, title=None):
    if not train_data and not val_data:
        raise RuntimeError('no loss data found in log')

    plt.figure(figsize=(10, 6))

    keys = choose_keys(train_data)
    all_vals = []
    for k in keys:
        ys = train_data[k]['ys'] if k in train_data else []
        all_vals += [v for v in ys if v is not None and not math.isnan(v)]

    # If wide dynamic range, use log scale
    use_log = False
    if all_vals:
        mx = max(all_vals)
        mn_candidates = [v for v in all_vals if v > 0]
        if mn_candidates:
            mn = min(mn_candidates)
            if mn > 0 and mx / mn > 100:
                use_log = True

    for k in keys:
        data = train_data.get(k)
        if not data or not data['xs']:
            continue
        xs = data['xs']
        ys = data['ys']
        ys_s = smooth(ys, window=20)
        plt.plot(xs, ys_s, label=k)

    # plot remaining non-priority keys if any (already limited by choose_keys)

    # validation losses as markers
    for k, v in val_data.items():
        if not v['xs']:
            continue
        plt.scatter(v['xs'], v['ys'], marker='x', s=40, label=f'{k} (val)')

    plt.xlabel('global iteration (batch)')
    plt.ylabel('loss')
    if title:
        plt.title(title)
    if use_log:
        plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print('Saved:', out_png)


def main():
    p = argparse.ArgumentParser(description='Plot loss curves from JSON log (json lines)')
    p.add_argument('--log', type=str, default='work_dirs/bev_tiny_det_mapv2/20260327_014052.log.json')
    p.add_argument('--out', type=str, default='work_dirs/bev_tiny_det_mapv2/loss_plot_20260327_014052.png')
    p.add_argument('--title', type=str, default=None)
    args = p.parse_args()

    train_data, val_data = parse_log(args.log)
    plot(train_data, val_data, args.out, title=args.title)


if __name__ == '__main__':
    main()
