#!/usr/bin/env python3

import argparse
from os import path as osp
import sys

sys.path.append('.')

from data_converter.nuscenes_maptrv2_converter import create_nuscenes_map_infos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate MapTRv2-style nuScenes offline map infos inside Apollo-Vision-Net.'
    )
    parser.add_argument('--root-path', type=str, required=True, help='Local nuScenes dataset root.')
    parser.add_argument('--canbus', type=str, required=True, help='Local nuScenes can bus root.')
    parser.add_argument('--out-dir', type=str, required=True, help='Directory to write generated pkl files.')
    parser.add_argument('--extra-tag', type=str, default='nuscenes', help='Output file prefix.')
    parser.add_argument('--version', type=str, default='v1.0', help='nuScenes version prefix, e.g. v1.0.')
    parser.add_argument('--max-sweeps', type=int, default=10, help='Number of lidar sweeps per sample.')
    parser.add_argument(
        '--point-cloud-range',
        type=float,
        nargs=6,
        default=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
        help='MapTRv2 local map range used to build offline annotation.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['trainval'],
        choices=['trainval', 'test', 'mini'],
        help='Which nuScenes split groups to generate.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    version_map = {
        'trainval': f'{args.version}-trainval',
        'test': f'{args.version}-test',
        'mini': f'{args.version}-mini',
    }

    generated = []
    for split in args.splits:
        version = version_map[split]
        print(
            f'[create_maptrv2_map_infos] generating split={split} version={version} '
            f'root={args.root_path} out={args.out_dir}',
            flush=True,
        )
        generated.extend(
            create_nuscenes_map_infos(
                root_path=args.root_path,
                out_path=args.out_dir,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=version,
                max_sweeps=args.max_sweeps,
                point_cloud_range=list(args.point_cloud_range),
            )
        )

    print('[create_maptrv2_map_infos] generated files:')
    for path in generated:
        print(path)


if __name__ == '__main__':
    main()