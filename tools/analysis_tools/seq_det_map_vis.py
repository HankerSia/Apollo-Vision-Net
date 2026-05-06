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
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--det-out', required=True)
    parser.add_argument('--map-out', required=True)
    parser.add_argument('--python', default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    det_results = os.path.join(args.base, 'pts_bbox', 'results_nusc.json')
    map_results = os.path.join(args.base, 'map_results.pkl')

    os.makedirs(args.det_out, exist_ok=True)
    os.makedirs(args.map_out, exist_ok=True)

    with open(det_results) as f:
        det = json.load(f)

    tokens = list(det['results'].keys())[:args.count]

    for i, token in enumerate(tokens):
        out_png = os.path.join(args.det_out, f'{i:03d}_{token[:8]}_det.png')
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

    for i in range(min(args.count, len(tokens))):
        out_png = os.path.join(args.map_out, f'{i:03d}_map.png')
        cmd = [
            args.python, 'tools/analysis_tools/vis_map_pred_single.py',
            '--data-root', args.data_root,
            '--version', args.version,
            '--infos', args.infos,
            '--results', map_results,
            '--index', str(i),
            '--out', out_png,
        ]
        print('RUN MAP:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
