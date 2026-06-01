#!/usr/bin/env python3
"""Batch-export aligned detection and map visualizations for one result folder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export a sequence of det/map visualization images.')
    parser.add_argument(
        '--base',
        required=True,
        help='Result folder containing pts_bbox/results_nusc.json and map_results.pkl')
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--infos', default='data/nuscenes/nuscenes_infos_temporal_val.pkl')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--scene', default=None, help='Optional scene_name to filter frames, e.g. scene-0103')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--det-out', required=True)
    parser.add_argument('--map-out', required=True)
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    det_results = os.path.join(args.base, 'pts_bbox', 'results_nusc.json')
    map_results_pkl = os.path.join(args.base, 'map_results.pkl')
    map_results_json = os.path.join(args.base, 'nuscmap_results.json')
    if os.path.exists(map_results_pkl):
        map_results = map_results_pkl
    elif os.path.exists(map_results_json):
        map_results = map_results_json
    else:
        raise FileNotFoundError(
            f'Cannot find map results under {args.base!r}. Expected one of: '
            f'{map_results_pkl!r}, {map_results_json!r}'
        )

    os.makedirs(args.det_out, exist_ok=True)
    os.makedirs(args.map_out, exist_ok=True)

    with open(det_results) as f:
        det = json.load(f)

    tokens = list(det['results'].keys())
    if args.scene:
        if not os.path.exists(args.infos):
            raise FileNotFoundError(
                f'--scene was set to {args.scene!r}, but infos file does not exist: {args.infos!r}'
            )
        import mmcv

        infos = mmcv.load(args.infos)['infos']
        tokens = [info['token'] for info in infos if info.get('scene_name') == args.scene and info.get('token') in det['results']]

    tokens = tokens[:args.count]
    if not tokens:
        raise RuntimeError(f'No tokens found for scene={args.scene!r} in {args.base!r}')

    det_root = os.path.join(args.det_out, args.scene) if args.scene else args.det_out
    map_root = os.path.join(args.map_out, args.scene) if args.scene else args.map_out
    os.makedirs(det_root, exist_ok=True)
    os.makedirs(map_root, exist_ok=True)

    for i, token in enumerate(tokens):
        out_png = os.path.join(det_root, f'{i:03d}_{token[:8]}_det.png')
        cmd = [
            args.python, 'tools/analysis_tools/vis_det_bev_single.py',
            '--dataroot', args.data_root,
            '--version', args.version,
            '--results', det_results,
            '--sample_token', token,
            '--out', out_png,
        ]
        print('RUN DET:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

    for i, token in enumerate(tokens):
        out_png = os.path.join(map_root, f'{i:03d}_{token[:8]}_map.png')
        cmd = [
            args.python, 'tools/analysis_tools/vis_map_pred_single.py',
            '--data-root', args.data_root,
            '--version', args.version,
            '--results', map_results,
            '--sample-token', token,
            '--out', out_png,
            '--with-input',
        ]
        if os.path.exists(args.infos):
            cmd.extend(['--infos', args.infos])
        print('RUN MAP:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
