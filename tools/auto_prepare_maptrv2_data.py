from os import path as osp

import mmcv

from data_converter.nuscenes_maptrv2_converter import create_nuscenes_map_infos


REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def _resolve_repo_path(path):
    if path is None:
        return None
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(REPO_ROOT, path))


def _emit(message, logger=None):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _infer_base_info_path(map_info_path):
    directory = osp.dirname(map_info_path)
    filename = osp.basename(map_info_path)
    if '_map_infos_temporal_' not in filename:
        raise ValueError(f'Cannot infer base info path from {map_info_path}')
    return osp.join(directory, filename.replace('_map_infos_temporal_', '_infos_temporal_'))


def _needs_annotation_merge(info_path):
    if not osp.exists(info_path):
        return True

    data = mmcv.load(info_path)
    infos = data.get('infos', [])
    if not infos:
        return True

    first = infos[0]
    required_det_keys = ('gt_boxes', 'gt_names', 'num_lidar_pts')
    has_det = all(key in first for key in required_det_keys)
    has_map = isinstance(first.get('annotation', None), dict)
    return not (has_det and has_map)


def _merge_base_infos_with_map_annotations(base_info_path, map_info_path, out_path):
    base_data = mmcv.load(base_info_path)
    map_data = mmcv.load(map_info_path)

    map_infos_by_token = {info['token']: info for info in map_data['infos']}
    merged_infos = []
    missing_tokens = []
    for base_info in base_data['infos']:
        token = base_info['token']
        map_info = map_infos_by_token.get(token)
        if map_info is None:
            missing_tokens.append(token)
            continue

        merged_info = dict(base_info)
        merged_info['annotation'] = map_info.get('annotation', {})
        if 'map_location' in map_info:
            merged_info['map_location'] = map_info['map_location']
        merged_infos.append(merged_info)

    if missing_tokens:
        preview = ', '.join(missing_tokens[:3])
        raise KeyError(f'Missing map annotations for {len(missing_tokens)} tokens, e.g. {preview}')

    merged_data = dict(base_data)
    merged_data['infos'] = merged_infos
    mmcv.dump(merged_data, out_path)


def maybe_prepare_maptrv2_data(cfg, logger=None):
    settings = cfg.get('map_auto_prepare', None)
    if not settings or not settings.get('enabled', False):
        return

    required_files = list(settings.get('required_files', []))
    if not required_files:
        for key in ('map_train_info_file', 'map_val_info_file'):
            value = cfg.get(key, None)
            if value:
                required_files.append(value)

    resolved_required = [_resolve_repo_path(path) for path in required_files]
    missing_required = [path for path in resolved_required if not osp.exists(path)]
    merge_required = [path for path in resolved_required if osp.exists(path) and _needs_annotation_merge(path)]
    if not missing_required and not merge_required:
        return

    root_path = _resolve_repo_path(settings.get('root_path'))
    out_dir = _resolve_repo_path(settings.get('out_dir', settings.get('root_path')))
    canbus = _resolve_repo_path(settings.get('canbus', 'data'))
    if root_path is None or not osp.isdir(root_path):
        raise FileNotFoundError(f'map_auto_prepare root_path does not exist: {root_path}')
    if canbus is None or not osp.isdir(canbus):
        raise FileNotFoundError(f'map_auto_prepare canbus path does not exist: {canbus}')

    version_prefix = settings.get('version', 'v1.0')
    version_map = {
        'trainval': f'{version_prefix}-trainval',
        'test': f'{version_prefix}-test',
        'mini': f'{version_prefix}-mini',
    }
    splits = settings.get('splits', ['trainval'])
    point_cloud_range = list(settings.get('point_cloud_range', cfg.get('point_cloud_range', [])))

    if missing_required:
        _emit('[map_auto_prepare] generating missing offline map infos...', logger=logger)
        for split in splits:
            version = version_map[split]
            _emit(
                f'[map_auto_prepare] split={split} version={version} root={root_path} out={out_dir}',
                logger=logger,
            )
            create_nuscenes_map_infos(
                root_path=root_path,
                out_path=out_dir,
                can_bus_root_path=canbus,
                info_prefix=settings.get('extra_tag', 'nuscenes'),
                version=version,
                max_sweeps=int(settings.get('max_sweeps', 10)),
                point_cloud_range=point_cloud_range,
            )

    if merge_required:
        _emit('[map_auto_prepare] merging base det infos with offline map annotations...', logger=logger)

    for required_path in merge_required:
        base_info_path = _infer_base_info_path(required_path)
        if not osp.exists(base_info_path):
            raise FileNotFoundError(f'Base det info file does not exist: {base_info_path}')
        _emit(
            f'[map_auto_prepare] merging base det infos with offline map annotations: '
            f'{osp.basename(base_info_path)} + {osp.basename(required_path)}',
            logger=logger,
        )
        _merge_base_infos_with_map_annotations(base_info_path, required_path, required_path)

    still_invalid = [path for path in resolved_required if _needs_annotation_merge(path)]
    if still_invalid:
        raise RuntimeError(f'map_auto_prepare failed to produce merged offline infos: {still_invalid}')

    _emit('[map_auto_prepare] map infos ready.', logger=logger)