from os import path as osp
import os
from concurrent.futures import ThreadPoolExecutor

import mmcv
import networkx as nx
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box


class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def _get_centerline(self, patch_box, patch_angle, layer_name, return_token=False):
        if layer_name not in ['lane', 'lane_connector']:
            raise ValueError(f'{layer_name} is not a centerline layer')

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.get_patch_coord(patch_box, patch_angle)
        records = getattr(self.map_api, layer_name)

        centerline_dict = {}
        for record in records:
            if record['polygon_token'] is None:
                continue
            polygon = self.map_api.extract_polygon(record['polygon_token'])
            if not polygon.is_valid:
                continue

            new_polygon = polygon.intersection(patch)
            if new_polygon.is_empty:
                continue

            centerline = list(self.map_api.discretize_lanes([record['token']], 0.5).values())[0]
            centerline = LineString(np.array(centerline)[:, :2].round(3))
            if centerline.is_empty:
                continue
            centerline = centerline.intersection(patch)
            if centerline.is_empty:
                continue

            centerline = to_patch_coord(centerline, patch_angle, patch_x, patch_y)
            centerline_dict[record['token']] = dict(
                centerline=centerline,
                token=record['token'],
                incoming_tokens=self.map_api.get_incoming_lane_ids(record['token']),
                outgoing_tokens=self.map_api.get_outgoing_lane_ids(record['token']),
            )
        return centerline_dict


def to_patch_coord(geometry, patch_angle, patch_x, patch_y):
    geometry = affinity.rotate(geometry, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
    geometry = affinity.affine_transform(geometry, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return geometry


def get_available_scenes(nusc):
    available_scenes = []
    print(f'total scene num: {len(nusc.scene)}')
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        scene_not_exist = False
        while True:
            lidar_path, _, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
            break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print(f'exist scene num: {len(available_scenes)}')
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except Exception:
        return np.zeros(18)

    can_bus = []
    last_pose = pose_list[0]
    for pose in pose_list:
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])
    can_bus.extend([0.0, 0.0])
    return np.array(can_bus)


def obtain_sensor2top(nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type='lidar'):
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:
        data_path = data_path.split(f'{os.getcwd()}/')[-1]
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    translation = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    translation -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = rotation.T
    sweep['sensor2lidar_translation'] = translation
    return sweep


def obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']

    lidar2global = ego2global @ lidar2ego
    lidar2global_translation = list(lidar2global[:3, 3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']
    patch_h = point_cloud_range[4] - point_cloud_range[1]
    patch_w = point_cloud_range[3] - point_cloud_range[0]
    patch_size = (patch_h, patch_w)
    vector_map = VectorizedLocalMap(nusc_maps[location], map_explorer[location], patch_size)
    info['annotation'] = vector_map.gen_vectorized_samples(lidar2global_translation, lidar2global_rotation)
    return info


def _attach_vectormap(nusc_maps, map_explorer, point_cloud_range, info):
    return obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range)


class VectorizedLocalMap(object):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1,
    }

    def __init__(
        self,
        nusc_map,
        map_explorer,
        patch_size,
        map_classes=['divider', 'ped_crossing', 'boundary', 'centerline'],
        line_classes=['road_divider', 'lane_divider'],
        ped_crossing_classes=['ped_crossing'],
        contour_classes=['road_segment', 'lane'],
        centerline_classes=['lane_connector', 'lane'],
        use_simplify=True,
    ):
        super().__init__()
        self.nusc_map = nusc_map
        self.map_explorer = map_explorer
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.centerline_classes = centerline_classes
        self.patch_size = patch_size

    def gen_vectorized_samples(self, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        map_dict = {'divider': [], 'ped_crossing': [], 'boundary': [], 'centerline': []}

        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for _, instances in line_instances_dict.items():
                    for instance in instances:
                        map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes)
                for instance in self.ped_poly_geoms_to_instances(ped_geom):
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes)
                for instance in self.poly_geoms_to_instances(polygon_geom):
                    map_dict[vec_class].append(np.array(instance.coords))
            elif vec_class == 'centerline':
                centerline_geom = self.get_centerline_geom(patch_box, patch_angle, self.centerline_classes)
                for instance in self.centerline_geoms_to_instances(centerline_geom):
                    map_dict[vec_class].append(np.array(instance.coords))
            else:
                raise ValueError(f'WRONG vec_class: {vec_class}')
        return map_dict

    def get_centerline_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                layer_centerline_dict = self.map_explorer._get_centerline(patch_box, patch_angle, layer_name, return_token=False)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                map_geom.update(layer_centerline_dict)
        return map_geom

    def get_map_geom(self, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                map_geom[layer_name] = self.get_divider_line(patch_box, patch_angle, layer_name)
            elif layer_name in self.polygon_classes:
                map_geom[layer_name] = self.get_contour_line(patch_box, patch_angle, layer_name)
            elif layer_name in self.ped_crossing_classes:
                map_geom[layer_name] = self.get_ped_crossing_line(patch_box, patch_angle)
        return map_geom

    def get_divider_line(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_line_layers:
            raise ValueError(f'{layer_name} is not a line layer')
        if layer_name == 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer.map_api, layer_name)
        for record in records:
            line = self.map_explorer.map_api.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name):
        if layer_name not in self.map_explorer.map_api.non_geometric_polygon_layers:
            raise ValueError(f'{layer_name} is not a polygonal layer')

        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        records = getattr(self.map_explorer.map_api, layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.map_explorer.map_api.extract_polygon(token) for token in record['polygon_tokens']]
                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        else:
            for record in records:
                polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        return polygon_list

    def get_ped_crossing_line(self, patch_box, patch_angle):
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self.map_explorer.get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer.map_api, 'ped_crossing')
        for record in records:
            polygon = self.map_explorer.map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
        return polygon_list

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = {}
        for line_type, lines in line_geom.items():
            line_instances_dict[line_type] = self._one_type_line_geom_to_instances(lines)
        return line_instances_dict

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        for line in line_geom:
            if line.is_empty:
                continue
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    line_instances.append(single_line)
            elif line.geom_type == 'LineString':
                line_instances.append(line)
            else:
                raise NotImplementedError
        return line_instances

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom['ped_crossing']
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom['road_segment']
        lanes = polygon_geom['lane']
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_instances(results)

    def centerline_geoms_to_instances(self, geoms_dict):
        centerline_geoms_list, _ = self.union_centerline(geoms_dict)
        return self._one_type_line_geom_to_instances(centerline_geoms_list)

    def union_centerline(self, centerline_geoms):
        pts_graph = nx.DiGraph()
        for value in centerline_geoms.values():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx in range(len(single_geom_pts) - 1):
                        pts_graph.add_edge(tuple(single_geom_pts[idx]), tuple(single_geom_pts[idx + 1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx in range(len(centerline_pts) - 1):
                    pts_graph.add_edge(tuple(centerline_pts[idx]), tuple(centerline_pts[idx + 1]))
            else:
                raise NotImplementedError

            for pred in value['incoming_tokens']:
                if pred in centerline_geoms.keys():
                    pred_geom = centerline_geoms[pred]['centerline']
                    pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1] if pred_geom.geom_type == 'MultiLineString' else np.array(pred_geom.coords).round(3)[-1]
                    pts_graph.add_edge(tuple(pred_pt), tuple(start_pt))

            for succ in value['outgoing_tokens']:
                if succ in centerline_geoms.keys():
                    succ_geom = centerline_geoms[succ]['centerline']
                    succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0] if succ_geom.geom_type == 'MultiLineString' else np.array(succ_geom.coords).round(3)[0]
                    pts_graph.add_edge(tuple(end_pt), tuple(succ_pt))

        roots = [v for v, degree in pts_graph.in_degree() if degree == 0]
        leaves = [v for v, degree in pts_graph.out_degree() if degree == 0]
        all_paths = []
        for root in roots:
            all_paths.extend(nx.all_simple_paths(pts_graph, root, leaves))

        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path).simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_graph


def _fill_trainval_infos(
    nusc,
    nusc_can_bus,
    nusc_maps,
    map_explorer,
    train_scenes,
    val_scenes,
    test=False,
    max_sweeps=10,
    point_cloud_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
    num_workers=None,
):
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    base_infos = []
    for sample in mmcv.track_iter_progress(nusc.sample):
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']

        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, _, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,
            'sweeps': [],
            'cams': {},
            'map_location': map_location,
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        frame_idx = 0 if sample['next'] == '' else frame_idx + 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            _, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if sd_rec['prev'] == '':
                break
            sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        info['sweeps'] = sweeps

        if sample['scene_token'] in train_scenes:
            base_infos.append((True, info))
        else:
            base_infos.append((False, info))

    if num_workers is None:
        num_workers = min(8, max(1, os.cpu_count() or 1))

    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            infos = list(
                executor.map(
                    lambda item: (
                        item[0],
                        _attach_vectormap(nusc_maps, map_explorer, point_cloud_range, item[1]),
                    ),
                    base_infos,
                )
            )
    else:
        infos = [
            (is_train, _attach_vectormap(nusc_maps, map_explorer, point_cloud_range, info))
            for is_train, info in base_infos
        ]

    for is_train, info in infos:
        if is_train:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_nuscenes_map_infos(
    root_path,
    out_path,
    can_bus_root_path,
    info_prefix,
    version='v1.0-trainval',
    max_sweeps=10,
    point_cloud_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
    num_workers=None,
):
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits

    print(version, root_path)
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
    maps = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
    nusc_maps = {}
    map_explorer = {}
    for loc in maps:
        nusc_maps[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
        map_explorer[loc] = CNuScenesMapExplorer(nusc_maps[loc])

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown version')

    available_scenes = get_available_scenes(nusc)
    available_scene_names = [scene['name'] for scene in available_scenes]
    train_scenes = list(filter(lambda name: name in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda name: name in available_scene_names, val_scenes))
    train_scenes = set(available_scenes[available_scene_names.index(name)]['token'] for name in train_scenes)
    val_scenes = set(available_scenes[available_scene_names.index(name)]['token'] for name in val_scenes)

    test = 'test' in version
    if test:
        print(f'test scene: {len(train_scenes)}')
    else:
        print(f'train scene: {len(train_scenes)}, val scene: {len(val_scenes)}')

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc,
        nusc_can_bus,
        nusc_maps,
        map_explorer,
        train_scenes,
        val_scenes,
        test,
        max_sweeps=max_sweeps,
        point_cloud_range=point_cloud_range,
        num_workers=num_workers,
    )

    metadata = dict(version=version)
    mmcv.mkdir_or_exist(out_path)
    if test:
        print(f'test sample: {len(train_nusc_infos)}')
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path, f'{info_prefix}_map_infos_temporal_test.pkl')
        mmcv.dump(data, info_path)
        return [info_path]

    print(f'train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}')
    data = dict(infos=train_nusc_infos, metadata=metadata)
    info_train_path = osp.join(out_path, f'{info_prefix}_map_infos_temporal_train.pkl')
    mmcv.dump(data, info_train_path)
    data['infos'] = val_nusc_infos
    info_val_path = osp.join(out_path, f'{info_prefix}_map_infos_temporal_val.pkl')
    mmcv.dump(data, info_val_path)
    return [info_train_path, info_val_path]