#!/usr/bin/env python3
"""Extract a nuScenes det+map subset by scene budget.

This tool works on existing nuScenes info pickles. It selects whole scenes
until the requested storage budget is reached and writes filtered train/val
info files. By default it only counts camera image files, which matches the
current camera-only det+mapv2 pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


CAMERA_TYPES = (
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
)


def log(message: str) -> None:
    print(f'[subset] {message}', flush=True)


def load_pickle(path: Path):
    log(f'Loading pickle: {path}')
    with path.open('rb') as handle:
        data = pickle.load(handle)
    infos = data.get('infos', []) if isinstance(data, dict) else []
    log(f'Loaded pickle: {path} infos={len(infos)}')
    return data


def dump_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    infos = obj.get('infos', []) if isinstance(obj, dict) else []
    log(f'Writing pickle: {path} infos={len(infos)}')
    with path.open('wb') as handle:
        pickle.dump(obj, handle)


def resolve_path(path: str, data_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    normalized_parts = tuple(part for part in candidate.parts if part not in ('', '.'))
    normalized = Path(*normalized_parts) if normalized_parts else Path()

    direct = data_root / normalized
    if direct.exists():
        return direct

    # Some infos store paths like ./data/nuscenes/samples/... while data_root
    # already points at .../data/nuscenes. Strip any duplicated leading prefix
    # and keep the first suffix that exists under data_root.
    for start in range(len(normalized_parts)):
        suffix = Path(*normalized_parts[start:])
        candidate_under_root = data_root / suffix
        if candidate_under_root.exists():
            return candidate_under_root

    return direct


def file_size(path: str, data_root: Path) -> int:
    resolved = resolve_path(path, data_root)
    if not resolved.exists():
        return 0
    return resolved.stat().st_size


def relative_to_root(path: str, data_root: Path) -> Path:
    resolved = resolve_path(path, data_root).resolve()
    try:
        return resolved.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f'Path is outside data root: {resolved}') from exc


def materialize_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == 'copy':
        shutil.copy2(src, dst)
        return
    if mode == 'symlink':
        dst.symlink_to(src)
        return
    if mode == 'hardlink':
        os.link(src, dst)
        return

    raise ValueError(f'Unsupported materialize mode: {mode}')


def materialize_paths(paths: Sequence[Path], data_root: Path, output_root: Path, mode: str) -> None:
    total = len(paths)
    if total == 0:
        log(f'No files to materialize into {output_root}')
        return

    log(f'Materializing {total} files into {output_root} with mode={mode}')
    for index, rel_path in enumerate(paths, start=1):
        src = data_root / rel_path
        if not src.exists():
            if index <= 5 or index == total:
                log(f'Skipping missing file {index}/{total}: {src}')
            continue
        dst = output_root / rel_path
        materialize_file(src, dst, mode)
        if index == 1 or index == total or index % 500 == 0:
            log(f'Materialized {index}/{total}: {rel_path}')


def collect_required_paths(
    infos: Iterable[dict],
    data_root: Path,
    include_lidar: bool,
    include_sweeps: bool,
) -> Set[Path]:
    required_paths: Set[Path] = set()
    for info in infos:
        for cam_name in CAMERA_TYPES:
            cam_info = info.get('cams', {}).get(cam_name)
            if cam_info and cam_info.get('data_path'):
                required_paths.add(relative_to_root(cam_info['data_path'], data_root))

        if include_lidar and info.get('lidar_path'):
            required_paths.add(relative_to_root(info['lidar_path'], data_root))

        if include_sweeps:
            for sweep in info.get('sweeps', []):
                sweep_path = sweep.get('data_path')
                if sweep_path:
                    required_paths.add(relative_to_root(sweep_path, data_root))
    return required_paths


def collect_map_paths(data_root: Path) -> List[Path]:
    maps_root = data_root / 'maps'
    if not maps_root.exists():
        return []
    return [path.relative_to(data_root) for path in maps_root.rglob('*') if path.is_file()]


def collect_extra_metadata_paths(data_root: Path) -> List[Path]:
    extra_files = []
    for name in (
        'nuscenes_map_anns_val.json',
        'nuscenes_map_anns_train.json',
        'nuscenes_map_anns_test.json',
    ):
        path = data_root / name
        if path.exists():
            extra_files.append(path.relative_to(data_root))
    return extra_files


def group_by_scene(infos: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for info in infos:
        grouped[info['scene_token']].append(info)
    return grouped


def scene_size_bytes(
    scene_infos: List[dict],
    data_root: Path,
    include_lidar: bool,
    include_sweeps: bool,
) -> int:
    total = 0
    for info in scene_infos:
        for cam_name in CAMERA_TYPES:
            cam_info = info.get('cams', {}).get(cam_name)
            if cam_info and cam_info.get('data_path'):
                total += file_size(cam_info['data_path'], data_root)

        if include_lidar and info.get('lidar_path'):
            total += file_size(info['lidar_path'], data_root)

        if include_sweeps:
            for sweep in info.get('sweeps', []):
                sweep_path = sweep.get('data_path')
                if sweep_path:
                    total += file_size(sweep_path, data_root)
    return total


def summarize_split(
    infos: List[dict],
    data_root: Path,
    include_lidar: bool,
    include_sweeps: bool,
) -> List[dict]:
    grouped = group_by_scene(infos)
    summary = []
    for scene_token, scene_infos in grouped.items():
        scene_name = scene_infos[0].get('scene_name', scene_token)
        summary.append(
            {
                'scene_token': scene_token,
                'scene_name': scene_name,
                'frames': len(scene_infos),
                'bytes': scene_size_bytes(
                    scene_infos,
                    data_root,
                    include_lidar=include_lidar,
                    include_sweeps=include_sweeps,
                ),
            }
        )
    return summary


def select_scenes(
    scene_summary: List[dict],
    target_bytes: int,
    budget_policy: str,
) -> Tuple[List[dict], int]:
    if not scene_summary:
        return [], 0

    def score(total_bytes: int) -> Tuple[int, int, int]:
        if budget_policy == 'under':
            over = 1 if total_bytes > target_bytes else 0
            gap = (total_bytes - target_bytes) if total_bytes > target_bytes else (target_bytes - total_bytes)
            return (over, gap, -min(total_bytes, target_bytes))

        return (abs(target_bytes - total_bytes), 0 if total_bytes >= target_bytes else 1, -total_bytes)

    selected: List[dict] = []
    used = 0
    for scene in scene_summary:
        candidate_used = used + scene['bytes']
        if not selected or score(candidate_used) <= score(used):
            selected.append(scene)
            used = candidate_used

    if not selected:
        first_scene = scene_summary[0]
        return [first_scene], first_scene['bytes']

    return selected, used


def filter_infos_by_scene(infos: List[dict], selected_scene_tokens: set[str]) -> List[dict]:
    return [info for info in infos if info['scene_token'] in selected_scene_tokens]


def write_manifest(
    output_dir: Path,
    target_bytes: int,
    split_name: str,
    selected_scenes: List[dict],
    selected_infos: List[dict],
    used_bytes: int,
    materialize_mode: str,
    materialized_file_count: int,
) -> None:
    manifest = {
        'split': split_name,
        'target_bytes': target_bytes,
        'selected_bytes': used_bytes,
        'selected_scene_count': len(selected_scenes),
        'selected_frame_count': len(selected_infos),
        'materialize_mode': materialize_mode,
        'materialized_file_count': materialized_file_count,
        'selected_scenes': selected_scenes,
    }
    with (output_dir / f'{split_name}_manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def process_split(
    split_name: str,
    pkl_path: Path,
    data_root: Path,
    output_dir: Path,
    target_bytes: int,
    include_lidar: bool,
    include_sweeps: bool,
    materialize_mode: str,
    budget_policy: str,
) -> None:
    log(f'Processing split={split_name} target={target_bytes / (1024 ** 3):.2f} GiB')
    data = load_pickle(pkl_path)
    infos = list(data['infos'])

    scene_summary = summarize_split(
        infos,
        data_root=data_root,
        include_lidar=include_lidar,
        include_sweeps=include_sweeps,
    )
    scene_summary.sort(key=lambda item: item['scene_name'])
    log(f'Split={split_name} scenes={len(scene_summary)} frames={len(infos)} before selection')

    selected_scenes, used_bytes = select_scenes(scene_summary, target_bytes, budget_policy)
    selected_scene_tokens = {scene['scene_token'] for scene in selected_scenes}
    selected_infos = filter_infos_by_scene(infos, selected_scene_tokens)
    log(
        f'Selected split={split_name} scenes={len(selected_scenes)} '
        f'frames={len(selected_infos)} size={used_bytes / (1024 ** 3):.2f} GiB '
        f'policy={budget_policy} order=sequential'
    )

    subset = {
        'infos': selected_infos,
        'metadata': data.get('metadata', {}),
    }

    required_paths = collect_required_paths(
        selected_infos,
        data_root=data_root,
        include_lidar=include_lidar,
        include_sweeps=include_sweeps,
    )
    log(f'Split={split_name} requires {len(required_paths)} materialized sample files')
    materialize_paths(sorted(required_paths), data_root, output_dir, materialize_mode)

    out_pkl = output_dir / pkl_path.name
    dump_pickle(subset, out_pkl)
    write_manifest(
        output_dir=output_dir,
        target_bytes=target_bytes,
        split_name=split_name,
        selected_scenes=selected_scenes,
        selected_infos=selected_infos,
        used_bytes=used_bytes,
        materialize_mode=materialize_mode,
        materialized_file_count=len(required_paths),
    )

    print(
        f'[{split_name}] scenes={len(selected_scenes)} frames={len(selected_infos)} '
        f'bytes={used_bytes / (1024 ** 3):.2f} GiB files={len(required_paths)} -> {out_pkl}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extract a nuScenes det+map subset by scene budget.'
    )
    parser.add_argument('--data-root', required=True, help='nuScenes root directory')
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to write subset pickles and manifests',
    )
    parser.add_argument(
        '--target-gb',
        type=float,
        required=True,
        help='Target storage budget in GiB',
    )
    parser.add_argument(
        '--train-pkl',
        type=str,
        default='',
        help='Path to nuscenes_infos_temporal_train.pkl',
    )
    parser.add_argument(
        '--val-pkl',
        type=str,
        default='',
        help='Path to nuscenes_infos_temporal_val.pkl',
    )
    parser.add_argument(
        '--include-lidar',
        action='store_true',
        help='Count lidar keyframe files in the size budget',
    )
    parser.add_argument(
        '--include-sweeps',
        action='store_true',
        help='Count sweep files in the size budget',
    )
    parser.add_argument(
        '--materialize-mode',
        choices=('copy', 'symlink', 'hardlink'),
        default='copy',
        help='How to materialize selected files under output-dir',
    )
    parser.add_argument(
        '--budget-policy',
        choices=('under', 'closest'),
        default='closest',
        help='Scene selection policy: stay under budget or allow slight overrun to get closer',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f'data_root={data_root}')
    log(f'output_dir={output_dir}')

    target_bytes = int(args.target_gb * (1024 ** 3))
    log(f'target={args.target_gb:.2f} GiB materialize_mode={args.materialize_mode} budget_policy={args.budget_policy}')

    splits: List[Tuple[str, Path]] = []
    if args.train_pkl:
        splits.append(('train', Path(args.train_pkl).expanduser().resolve()))
    if args.val_pkl:
        splits.append(('val', Path(args.val_pkl).expanduser().resolve()))

    if not splits:
        raise SystemExit('Provide at least one of --train-pkl or --val-pkl.')

    total_weight = 0
    split_summaries = []
    for split_name, pkl_path in splits:
        data = load_pickle(pkl_path)
        infos = list(data['infos'])
        scene_summary = summarize_split(
            infos,
            data_root=data_root,
            include_lidar=args.include_lidar,
            include_sweeps=args.include_sweeps,
        )
        split_weight = sum(scene['bytes'] for scene in scene_summary)
        split_summaries.append((split_name, pkl_path, split_weight))
        total_weight += split_weight
        log(f'Estimated split={split_name} size={split_weight / (1024 ** 3):.2f} GiB')

    if total_weight == 0:
        raise SystemExit('No file sizes were detected; check data-root and pkl paths.')

    shared_paths = collect_map_paths(data_root)
    shared_paths.extend(collect_extra_metadata_paths(data_root))
    log(f'Materializing {len(shared_paths)} shared map/metadata files')
    materialize_paths(shared_paths, data_root, output_dir, args.materialize_mode)

    for split_name, pkl_path, split_weight in split_summaries:
        split_target = target_bytes * split_weight / total_weight
        log(f'Assigned split={split_name} target={split_target / (1024 ** 3):.2f} GiB')
        process_split(
            split_name=split_name,
            pkl_path=pkl_path,
            data_root=data_root,
            output_dir=output_dir,
            target_bytes=int(split_target),
            include_lidar=args.include_lidar,
            include_sweeps=args.include_sweeps,
            materialize_mode=args.materialize_mode,
            budget_policy=args.budget_policy,
        )

    log('Subset extraction finished')


if __name__ == '__main__':
    main()