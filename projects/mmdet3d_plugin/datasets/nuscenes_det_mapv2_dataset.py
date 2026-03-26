import networkx as nx
import numpy as np

from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString
from shapely.errors import TopologicalError

from .nuscenes_det_occ_map_dataset import CustomNuScenesDetMapDataset
from .nuscenes_det_occ_map_dataset import VectorizedLocalMap


class VectorizedLocalMapV2(VectorizedLocalMap):
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'centerline': 3,
        'others': -1,
    }

    def __init__(
        self,
        dataroot,
        patch_size,
        map_classes=('divider', 'ped_crossing', 'boundary', 'centerline'),
        line_classes=('road_divider', 'lane_divider'),
        ped_crossing_classes=('ped_crossing',),
        contour_classes=('road_segment', 'lane'),
        centerline_classes=('lane_connector', 'lane'),
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
    ):
        super().__init__(
            dataroot=dataroot,
            patch_size=patch_size,
            map_classes=map_classes,
            line_classes=line_classes,
            ped_crossing_classes=ped_crossing_classes,
            contour_classes=contour_classes,
            sample_dist=sample_dist,
            num_samples=num_samples,
            padding=padding,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
        )
        self.centerline_classes = list(centerline_classes)

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            elif vec_class == 'centerline':
                centerline_geom = self.get_centerline_geom(patch_box, patch_angle, self.centerline_classes, location)
                centerline_instances = self.centerline_geoms_to_instances(centerline_geom)
                for centerline in centerline_instances:
                    vectors.append((centerline, self.CLASS2LABEL.get('centerline', -1)))
            else:
                raise ValueError(f'Unsupported vec_class: {vec_class}')

        gt_instance = []
        gt_labels = []
        for instance, typ in vectors:
            if typ != -1:
                gt_instance.append(instance)
                gt_labels.append(typ)

        if len(gt_instance) == 0:
            gt_instance_lines = []
        else:
            gt_instance_lines = self._build_instance_lines(gt_instance)

        return dict(gt_vecs_pts_loc=gt_instance_lines, gt_vecs_label=gt_labels)

    def _build_instance_lines(self, gt_instance):
        return self._instance_lines_cls(
            gt_instance,
            self.sample_dist,
            self.num_samples,
            self.padding,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )

    @property
    def _instance_lines_cls(self):
        from .nuscenes_det_occ_map_dataset import LiDARInstanceLines

        return LiDARInstanceLines

    def get_centerline_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name not in self.centerline_classes:
                continue
            layer_centerline_dict = self._get_centerline(patch_box, patch_angle, layer_name, location)
            if layer_centerline_dict:
                map_geom.update(layer_centerline_dict)
        return map_geom

    def _get_centerline(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.centerline_classes:
            raise ValueError(f'{layer_name} is not a centerline layer')

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        map_api = self.map_explorer[location].map_api
        records = getattr(map_api, layer_name)

        centerline_dict = {}
        for record in records:
            polygon_token = record.get('polygon_token', None)
            if polygon_token is None:
                continue
            polygon = map_api.extract_polygon(polygon_token)
            try:
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                new_polygon = polygon.intersection(patch)
            except TopologicalError:
                try:
                    new_polygon = polygon.buffer(0).intersection(patch)
                except Exception:
                    continue

            if new_polygon.is_empty:
                continue

            centerline = list(map_api.discretize_lanes([record['token']], 0.5).values())[0]
            centerline = LineString(np.array(centerline)[:, :2].round(3))
            if centerline.is_empty:
                continue
            centerline = centerline.intersection(patch)
            if centerline.is_empty:
                continue
            centerline = self._patch_transform(centerline, patch_angle, patch_box)

            centerline_dict[record['token']] = dict(
                centerline=centerline,
                token=record['token'],
                incoming_tokens=map_api.get_incoming_lane_ids(record['token']),
                outgoing_tokens=map_api.get_outgoing_lane_ids(record['token']),
            )
        return centerline_dict

    def centerline_geoms_to_instances(self, geoms_dict):
        centerline_geoms_list, _ = self.union_centerline(geoms_dict)
        return self._one_type_line_geom_to_instances(centerline_geoms_list)

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []
        for line in line_geom:
            if line.is_empty:
                continue
            if line.geom_type == 'MultiLineString':
                for single_line in line.geoms:
                    if not single_line.is_empty:
                        line_instances.append(single_line)
            elif line.geom_type == 'LineString':
                line_instances.append(line)
            else:
                raise NotImplementedError
        return line_instances

    def union_centerline(self, centerline_geoms):
        pts_graph = nx.DiGraph()
        if not centerline_geoms:
            return [], pts_graph

        for value in centerline_geoms.values():
            centerline_geom = value['centerline']
            if centerline_geom.is_empty:
                continue
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx in range(max(len(single_geom_pts) - 1, 0)):
                        pts_graph.add_edge(tuple(single_geom_pts[idx]), tuple(single_geom_pts[idx + 1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx in range(max(len(centerline_pts) - 1, 0)):
                    pts_graph.add_edge(tuple(centerline_pts[idx]), tuple(centerline_pts[idx + 1]))
            else:
                continue

            for pred in value['incoming_tokens']:
                if pred not in centerline_geoms:
                    continue
                pred_geom = centerline_geoms[pred]['centerline']
                if pred_geom.is_empty:
                    continue
                if pred_geom.geom_type == 'MultiLineString':
                    pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1]
                else:
                    pred_pt = np.array(pred_geom.coords).round(3)[-1]
                pts_graph.add_edge(tuple(pred_pt), tuple(start_pt))

            for succ in value['outgoing_tokens']:
                if succ not in centerline_geoms:
                    continue
                succ_geom = centerline_geoms[succ]['centerline']
                if succ_geom.is_empty:
                    continue
                if succ_geom.geom_type == 'MultiLineString':
                    succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0]
                else:
                    succ_pt = np.array(succ_geom.coords).round(3)[0]
                pts_graph.add_edge(tuple(end_pt), tuple(succ_pt))

        roots = [node for node, degree in pts_graph.in_degree() if degree == 0]
        leaves = [node for node, degree in pts_graph.out_degree() if degree == 0]
        if not roots or not leaves:
            centerline_instances = []
            for value in centerline_geoms.values():
                geom = value['centerline']
                if geom.geom_type == 'MultiLineString':
                    centerline_instances.extend([g for g in geom.geoms if not g.is_empty])
                elif geom.geom_type == 'LineString' and not geom.is_empty:
                    centerline_instances.append(geom)
            return centerline_instances, pts_graph

        all_paths = []
        for root in roots:
            all_paths.extend(nx.all_simple_paths(pts_graph, root, leaves))

        final_centerline_paths = []
        for path in all_paths:
            if len(path) < 2:
                continue
            merged_line = LineString(path).simplify(0.2, preserve_topology=True)
            if not merged_line.is_empty:
                final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_graph


@DATASETS.register_module()
class CustomNuScenesDetMapV2BaseDataset(CustomNuScenesDetMapDataset):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('use_occ_gts', False)
        super().__init__(*args, **kwargs)
        patch_h = self.pc_range[4] - self.pc_range[1]
        patch_w = self.pc_range[3] - self.pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.vector_map = VectorizedLocalMapV2(
            kwargs['data_root'],
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=self.fixed_num,
            padding_value=self.padding_value,
        )


@DATASETS.register_module()
class CustomNuScenesDetMapV2Dataset(CustomNuScenesDetMapV2BaseDataset):
    pass