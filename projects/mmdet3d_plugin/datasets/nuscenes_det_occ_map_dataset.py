import numpy as np
import torch
import copy
import os
import json
from os import path as osp

import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor

from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .nuscenes_dataset import CustomNuScenesDataset

# Reuse MapTR's online nuScenes-map vectorization logic.
# Keeping it self-contained here avoids a hard dependency on the MapTR repo.
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from shapely.errors import TopologicalError


def _scene_name_to_log_location(scene_name: str, dataroot: str, version: str = 'v1.0-trainval'):
    """Map nuScenes scene name (e.g., 'scene-0061') to log location.

    This is a lightweight runtime fallback for legacy infos that don't store
    `map_location`.
    """
    if not isinstance(scene_name, str) or not scene_name.startswith('scene-'):
        return None
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        for s in nusc.scene:
            if s.get('name') == scene_name:
                log = nusc.get('log', s['log_token'])
                return log.get('location')
    except Exception:
        return None
    return None


class LiDARInstanceLines(object):
    """Line instances in LiDAR coordinates.

    This is a lightly adapted copy of MapTR's implementation.
    """

    def __init__(
        self,
        instance_line_list,
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=None,
    ):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def fixed_num_sampled_points(self):
        if self.fixed_num is None or self.fixed_num <= 0:
            raise ValueError(
                f'LiDARInstanceLines.fixed_num must be > 0, got {self.fixed_num}. '
                'Set dataset fixed_num (fixed_ptsnum_per_line) to a positive value.'
            )
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            if instance is None or instance.is_empty:
                # Keep a fully padded line to avoid crashing downstream.
                instance_points_list.append(np.zeros((self.fixed_num, 2), dtype=np.float32))
                continue
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = (
                np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]
                )
                .reshape(-1, 2)
            )
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x
        )
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y
        )
        return instance_points_tensor


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
        dataroot,
        patch_size,
        map_classes=('divider', 'ped_crossing', 'boundary'),
        line_classes=('road_divider', 'lane_divider'),
        ped_crossing_classes=('ped_crossing',),
        contour_classes=('road_segment', 'lane'),
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            'boston-seaport',
            'singapore-hollandvillage',
            'singapore-onenorth',
            'singapore-queenstown',
        ]
        self.vec_classes = list(map_classes)
        self.line_classes = list(line_classes)
        self.ped_crossing_classes = list(ped_crossing_classes)
        self.polygon_classes = list(contour_classes)

        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, location
                )
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, location
                )
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, location
                )
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
            else:
                raise ValueError(f'Unsupported vec_class: {vec_class}')

        gt_instance = []
        gt_labels = []
        for instance, typ in vectors:
            if typ != -1:
                gt_instance.append(instance)
                gt_labels.append(typ)

        if len(gt_instance) == 0:
            # Keep an empty list; caller will handle it.
            gt_instance_lines = []
        else:
            gt_instance_lines = LiDARInstanceLines(
                gt_instance,
                self.sample_dist,
                self.num_samples,
                self.padding,
                self.fixed_num,
                self.padding_value,
                patch_size=self.patch_size,
            )

        return dict(
            gt_vecs_pts_loc=gt_instance_lines,
            gt_vecs_label=gt_labels,
        )

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _patch_transform(self, geom, patch_angle, patch_box):
        # Rotate around patch center and translate so that patch center is (0,0).
        x, y, _, _ = patch_box
        geom = affinity.rotate(geom, -patch_angle, origin=(x, y), use_radians=False)
        geom = affinity.translate(geom, xoff=-x, yoff=-y)
        return geom

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError(f'layer_name={layer_name} is not a line layer')

        records = getattr(self.map_explorer[location].map_api, layer_name)
        line_list = []
        patch = box(
            patch_box[0] - patch_box[3] / 2,
            patch_box[1] - patch_box[2] / 2,
            patch_box[0] + patch_box[3] / 2,
            patch_box[1] + patch_box[2] / 2,
        )
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record['line_token'])
            try:
                line = line.intersection(patch)
            except TopologicalError:
                continue
            line = self._patch_transform(line, patch_angle, patch_box)
            line_list.append(line)
        return line_list

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError(f'layer_name={layer_name} is not a polygon layer')

        records = getattr(self.map_explorer[location].map_api, layer_name)
        polygons = []
        patch = box(
            patch_box[0] - patch_box[3] / 2,
            patch_box[1] - patch_box[2] / 2,
            patch_box[0] + patch_box[3] / 2,
            patch_box[1] + patch_box[2] / 2,
        )
        for record in records:
            if 'polygon_tokens' in record:
                polys = [
                    self.map_explorer[location].map_api.extract_polygon(token)
                    for token in record['polygon_tokens']
                ]
                poly = ops.unary_union(polys)
            else:
                poly = self.map_explorer[location].map_api.extract_polygon(
                    record['polygon_token']
                )
            # Some nuScenes map polygons are invalid (self-intersection, etc.).
            # Shapely may throw TopologicalError during intersection; try to
            # repair with buffer(0) and skip if still failing.
            try:
                if not poly.is_valid:
                    poly = poly.buffer(0)
                poly = poly.intersection(patch)
            except TopologicalError:
                try:
                    poly = poly.buffer(0).intersection(patch)
                except Exception:
                    continue
            poly = self._patch_transform(poly, patch_angle, patch_box)
            polygons.append(poly)
        return polygons

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        records = getattr(self.map_explorer[location].map_api, 'ped_crossing')
        polygons = []
        patch = box(
            patch_box[0] - patch_box[3] / 2,
            patch_box[1] - patch_box[2] / 2,
            patch_box[0] + patch_box[3] / 2,
            patch_box[1] + patch_box[2] / 2,
        )
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record['polygon_token'])
            try:
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygon = polygon.intersection(patch)
            except TopologicalError:
                try:
                    polygon = polygon.buffer(0).intersection(patch)
                except Exception:
                    continue
            polygon = self._patch_transform(polygon, patch_angle, patch_box)
            polygons.append(polygon)
        return polygons

    def line_geoms_to_instances(self, map_geom):
        line_instances_dict = {}
        for layer_name, geoms in map_geom:
            instances = []
            for line in geoms:
                if line.is_empty:
                    continue
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        instances.append(single_line)
                elif line.geom_type == 'LineString':
                    instances.append(line)
            line_instances_dict[layer_name] = instances
        return line_instances_dict

    def ped_poly_geoms_to_instances(self, ped_geom):
        # Represent pedestrian crossing polygons by their boundary.
        ped_instances = []
        for _, ped_polys in ped_geom:
            for poly in ped_polys:
                if poly.is_empty:
                    continue
                if poly.geom_type == 'MultiPolygon':
                    for p in poly.geoms:
                        ped_instances.append(p.exterior)
                elif poly.geom_type == 'Polygon':
                    ped_instances.append(poly.exterior)
        return ped_instances

    def poly_geoms_to_instances(self, polygon_geom):
        # Convert road/lane polygons into contour line instances.
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)

        if union_segments.is_empty:
            return []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])

        results = []
        for poly in union_segments.geoms:
            ext = poly.exterior
            lines = ext.intersection(local_patch)
            # IMPORTANT:
            # Do NOT linemerge contour segments here. In complex junctions,
            # `linemerge` may stitch nearby-but-unrelated segments and create
            # polylines with large point-to-point jumps (visualized as
            # "cross-connecting" boundaries). Keep segments split, consistent
            # with MapTR's original implementation.
            if lines.is_empty:
                continue
            if isinstance(lines, MultiLineString):
                for seg in lines.geoms:
                    if not seg.is_empty and seg.geom_type == 'LineString':
                        results.append(seg)
            elif lines.geom_type == 'LineString':
                results.append(lines)
            else:
                # Rarely, intersection may yield GeometryCollection. Keep only LineStrings.
                try:
                    for g in getattr(lines, 'geoms', []):
                        if not g.is_empty and g.geom_type == 'LineString':
                            results.append(g)
                except Exception:
                    pass
        return results


@DATASETS.register_module()
class CustomNuScenesDetOccMapDataset(CustomNuScenesDataset):
    """NuScenes dataset for multi-task (det + occ + map vectors).

    - Detection fields stay in `gt_bboxes_3d` / `gt_labels_3d`.
    - Occupancy fields (if enabled) stay in `occ_gts` / `flow_gts`.
    - Map vector GT is added as `gt_map_vecs_pts_loc` / `gt_map_vecs_label`.

    It requires infos to contain `map_location` (added in the converter).
    """

    def __init__(
        self,
        map_classes=('divider', 'ped_crossing', 'boundary'),
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
        map_ann_file=None,
        pc_range=None,
        eval_use_same_gt_sample_num_flag=True,
        map_eval_nproc=8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.MAPCLASSES = list(map_classes)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line

        # ---- MapTR-aligned evaluation settings ----
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.map_eval_nproc = int(map_eval_nproc) if map_eval_nproc is not None else 0
        default_pc_range = getattr(self, 'point_cloud_range', [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0])
        self.pc_range = list(pc_range) if pc_range is not None else list(default_pc_range)
        if len(self.pc_range) != 6:
            raise ValueError(f'pc_range must have 6 numbers, got {self.pc_range}')

        if map_ann_file is None:
            # Keep MapTR-compatible default filename under the dataset root.
            data_root = kwargs.get('data_root', 'data/nuscenes/')
            map_ann_file = osp.join(data_root, 'nuscenes_map_anns_val.json')
        self.map_ann_file = map_ann_file

        patch_h = self.pc_range[4] - self.pc_range[1]
        patch_w = self.pc_range[3] - self.pc_range[0]
        self.patch_size = (patch_h, patch_w)

        self.vector_map = VectorizedLocalMap(
            kwargs['data_root'],
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
        )

    # ---------------- MapTR-style map evaluation ----------------

    def evaluate_map(
        self,
        results,
        metric='chamfer',
        logger=None,
        jsonfile_prefix=None,
        **kwargs,
    ):
        """Evaluate vector-map predictions using MapTR protocol.

        Args:
            results: list of per-sample map predictions.
            metric: 'chamfer', 'iou' or list of them.
            jsonfile_prefix: output directory (same style as detection).

        Returns:
            dict: metric name -> value
        """

        if results is None:
            return {}
        if jsonfile_prefix is None:
            jsonfile_prefix = osp.join('test', 'map_results')

        result_path = self.format_map_results(results, jsonfile_prefix=jsonfile_prefix)
        return self._evaluate_map_single(result_path, metric=metric, logger=logger)

    def _evaluate_map_single(self, result_path: str, metric='chamfer', logger=None):
        from .map_utils.mean_ap import eval_map
        from .map_utils.mean_ap import format_res_gt_by_classes

        result_path = osp.abspath(result_path)

        if not osp.exists(self.map_ann_file):
            self._format_map_gt()

        with open(result_path, 'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']

        with open(self.map_ann_file, 'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']

        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path,
            gen_results,
            annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
            nproc=max(self.map_eval_nproc, 0),
        )

        metrics = metric if isinstance(metric, (list, tuple)) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported, allowed: {allowed_metrics}')

        detail = {}
        for m in metrics:
            if m == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            else:
                thresholds = np.linspace(
                    0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
                ).tolist()

            cls_aps = np.zeros((len(thresholds), len(self.MAPCLASSES)), dtype=np.float64)
            for i, thr in enumerate(thresholds):
                mAP, cls_ap = eval_map(
                    gen_results,
                    annotations,
                    cls_gens,
                    cls_gts,
                    threshold=float(thr),
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=m,
                    nproc=max(self.map_eval_nproc, 0),
                )
                for j in range(len(self.MAPCLASSES)):
                    cls_aps[i, j] = float(cls_ap[j]['ap'])

            for j, name in enumerate(self.MAPCLASSES):
                detail[f'NuscMap_{m}/{name}_AP'] = float(cls_aps.mean(axis=0)[j])
            detail[f'NuscMap_{m}/mAP'] = float(cls_aps.mean(axis=0).mean())

        return detail

    def format_map_results(self, results, jsonfile_prefix: str):
        """Write predictions to MapTR-compatible json format."""
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')

        pred_annos = []
        for sample_id, det in enumerate(results):
            sample_token = self.data_infos[sample_id]['token']
            pred_anno = {'sample_token': sample_token}

            vec_list = self._map_det_to_vector_list(det)

            pred_vec_list = []
            for vec in vec_list:
                label = int(vec['label'])
                if label < 0 or label >= len(self.MAPCLASSES):
                    continue
                pred_vec_list.append(
                    dict(
                        pts=np.asarray(vec['pts'], dtype=np.float32),
                        pts_num=int(len(vec['pts'])),
                        cls_name=self.MAPCLASSES[label],
                        type=label,
                        confidence_level=float(vec.get('score', 0.0)),
                    )
                )

            pred_anno['vectors'] = pred_vec_list
            pred_annos.append(pred_anno)

        nusc_submissions = {'meta': getattr(self, 'modality', {}), 'results': pred_annos}
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _map_det_to_vector_list(self, det):
        """Normalize map prediction for a single sample into a list of vectors."""
        if det is None:
            return []

        # Already vector-list format.
        if isinstance(det, dict) and isinstance(det.get('vectors', None), list):
            if len(det['vectors']) == 0:
                return []
            if isinstance(det['vectors'][0], dict):
                return det['vectors']

        if not isinstance(det, dict):
            raise TypeError(f'Unsupported map det type: {type(det)}')

        vectors = det.get('vectors', None)
        scores = det.get('scores', None)
        labels = det.get('labels', None)

        if vectors is None or scores is None or labels is None:
            return []

        vectors = np.asarray(vectors)
        scores = np.asarray(scores).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        if vectors.ndim != 3 or vectors.shape[-1] != 2:
            raise ValueError(f'Expected vectors with shape (N,P,2), got {vectors.shape}')

        out = []
        n = int(min(vectors.shape[0], scores.shape[0], labels.shape[0]))
        for i in range(n):
            out.append(
                {
                    'label': int(labels[i]),
                    'score': float(scores[i]),
                    'pts': vectors[i],
                }
            )
        return out

    def _format_map_gt(self):
        """Generate GT map annotations json compatible with MapTR evaluator."""
        assert self.map_ann_file is not None
        mmcv.mkdir_or_exist(osp.dirname(self.map_ann_file))

        gt_annos = []
        dataset_length = len(self)
        prog_bar = mmcv.ProgressBar(dataset_length)

        for sample_id in range(dataset_length):
            info = self.data_infos[sample_id]
            sample_token = info['token']

            location = self._resolve_map_location_from_info(info)

            lidar2ego = np.eye(4)
            lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = np.array(info['lidar2ego_translation'])

            ego2global = np.eye(4)
            ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
            ego2global[:3, 3] = np.array(info['ego2global_translation'])

            lidar2global = ego2global @ lidar2ego
            lidar2global_translation = list(lidar2global[:3, 3])
            lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

            anns_results = self.vector_map.gen_vectorized_samples(
                location, lidar2global_translation, lidar2global_rotation
            )

            gt_labels = anns_results.get('gt_vecs_label', [])
            gt_lines = []
            gt_inst = anns_results.get('gt_vecs_pts_loc', [])
            if isinstance(gt_inst, LiDARInstanceLines):
                gt_lines = gt_inst.instance_list

            gt_vec_list = []
            for gt_label, gt_vec in zip(gt_labels, gt_lines):
                gt_label = int(gt_label)
                if gt_label < 0 or gt_label >= len(self.MAPCLASSES):
                    continue
                pts = np.asarray(list(gt_vec.coords), dtype=np.float32)
                gt_vec_list.append(
                    dict(
                        pts=pts,
                        pts_num=int(pts.shape[0]),
                        cls_name=self.MAPCLASSES[gt_label],
                        type=gt_label,
                    )
                )

            gt_annos.append({'sample_token': sample_token, 'vectors': gt_vec_list})
            prog_bar.update()

        mmcv.dump({'GTs': gt_annos}, self.map_ann_file)

    def _resolve_map_location_from_info(self, info: dict):
        """Resolve nuScenes map location for one sample info."""
        if info.get('map_location', None) is not None:
            return info['map_location']

        scene_name = info.get('scene_name', None)
        if isinstance(scene_name, str) and scene_name:
            location = scene_name
            if scene_name.startswith('scene-'):
                version = None
                if isinstance(getattr(self, 'metadata', None), dict):
                    version = self.metadata.get('version', None)
                version = version or 'v1.0-trainval'
                resolved = _scene_name_to_log_location(scene_name, self.data_root, version=version)
                if resolved is not None:
                    location = resolved
            return location

        raise KeyError('Missing map_location/scene_name in data_infos for map GT formatting')

    def _add_vectormap_gt(self, example, input_dict):
        # Prefer map_location from infos; fall back to nuScenes scene name.
        # Some info pkls (e.g., legacy temporal infos) don't include
        # `map_location`. In that case, use the nuScenes default mapping:
        # scene_name like "singapore-onenorth" is accepted by nuscenes-map.
        if 'map_location' in input_dict:
            location = input_dict['map_location']
        elif 'scene_name' in input_dict:
            # Some converters store scene_name as "scene-XXXX".
            # Try to map it back to a real nuScenes map location via metadata.
            scene_name = input_dict['scene_name']
            location = scene_name
            if scene_name.startswith('scene-') and hasattr(self, 'metadata'):
                scene_map_loc = None
                # 1) preferred: explicit mapping dict
                if isinstance(self.metadata, dict):
                    scene_map_loc = self.metadata.get('scene_map_location', None)
                    if isinstance(scene_map_loc, dict) and scene_name in scene_map_loc:
                        location = scene_map_loc[scene_name]
                    else:
                        scene_map_loc = None
                # 2) fallback: match in metadata['scenes'] list
                if scene_map_loc is None and isinstance(self.metadata, dict):
                    scenes = self.metadata.get('scenes', None)
                    if isinstance(scenes, (list, tuple)):
                        for s in scenes:
                            if s.get('name') == scene_name and 'log_location' in s:
                                location = s['log_location']
                                break
                # 3) last resort: query nuScenes tables to get log location.
                if location == scene_name:
                    version = None
                    if isinstance(getattr(self, 'metadata', None), dict):
                        version = self.metadata.get('version', None)
                    # NuScenes constructor expects a concrete version string.
                    version = version or 'v1.0-trainval'
                    resolved = _scene_name_to_log_location(scene_name, self.data_root, version=version)
                    if resolved is not None:
                        location = resolved
        else:
            raise KeyError(
                'Missing `map_location`/`scene_name` in input_dict. '
                'Please regenerate infos with map location fields.'
            )

        if location not in self.vector_map.map_explorer:
            raise KeyError(
                f'Unknown nuScenes map location: {location}. '
                'Expected one of: ' + ', '.join(self.vector_map.MAPS)
            )

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = np.array(input_dict['lidar2ego_translation'])

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = np.array(input_dict['ego2global_translation'])

        lidar2global = ego2global @ lidar2ego
        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        anns_results = self.vector_map.gen_vectorized_samples(
            location, lidar2global_translation, lidar2global_rotation
        )

        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])

        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except Exception:
                # empty tensor/list - preserve for test
                pass

        # Keep detection GT keys untouched; add new keys for map.
        example['gt_map_vecs_label'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_map_vecs_pts_loc'] = DC(gt_vecs_pts_loc, cpu_only=True)
        return example

    def prepare_train_data(self, index):
        # Important: add vector-map GT *before* Collect/FormatBundle.
        # The parent implementation runs the pipeline and (typically) ends with
        # CustomCollect3D, which will raise KeyError if keys don't exist.
        # Build input_dict from data_infos so we retain scene_name/location.
        info = self.data_infos[index]
        input_dict = self.get_data_info(index)
        # Propagate scene_name if base get_data_info doesn't.
        if 'scene_name' not in input_dict and 'scene_name' in info:
            input_dict['scene_name'] = info['scene_name']
        if 'map_location' not in input_dict and 'map_location' in info:
            input_dict['map_location'] = info['map_location']
        # Map GT needs lidar2ego_* to compute lidar2global.
        if 'lidar2ego_rotation' not in input_dict and 'lidar2ego_rotation' in info:
            input_dict['lidar2ego_rotation'] = info['lidar2ego_rotation']
        if 'lidar2ego_translation' not in input_dict and 'lidar2ego_translation' in info:
            input_dict['lidar2ego_translation'] = info['lidar2ego_translation']
        # mmdet3d's LoadAnnotations3D expects these lists to exist.
        # Some of our temporal infos/pipelines don't initialize them early.
        input_dict.setdefault('bbox3d_fields', [])
        input_dict.setdefault('pts_mask_fields', [])
        input_dict.setdefault('pts_seg_fields', [])
        input_dict.setdefault('mask_fields', [])
        input_dict.setdefault('seg_fields', [])
        example = self.pipeline(input_dict)
        if example is None:
            return None

        # Ensure temporal meta format: base pipeline may output a single meta
        # dict, while BEVFormer/union2one expects a dict-of-metas keyed by time.
        if 'img_metas' in example and isinstance(example['img_metas'], DC):
            metas = example['img_metas'].data
            if isinstance(metas, dict) and 'can_bus' in metas:
                # Wrap single-timestep meta into {0: meta_dict, 1: meta_dict}
                # so temporal loops that expect at least two steps (prev+curr)
                # can index safely.
                example['img_metas'] = DC({0: metas, 1: copy.deepcopy(metas)}, cpu_only=True)

        example = self._add_vectormap_gt(example, input_dict)

        # BEVFormer in this repo expects temporal queue in `img` with shape
        # [num_cam, T, C, H, W] so that after dataloader stack it becomes
        # [bs, T, num_cam, C, H, W]. The base dataset returns [num_cam, C, H, W]
        # for a single timestamp (queue folding happens in CustomNuScenesDataset).
        # For det+map scaffolding, wrap the single frame as T=1 to satisfy
        # forward_train/obtain_history_bev without changing detector code.
        try:
            # If the base dataset already produced a temporal queue folded into
            # one sample (img shape [num_cam, T, C, H, W]), keep it.
            # If it's a single frame ([num_cam, C, H, W]), wrap as T=1.
            if 'img' in example and isinstance(example['img'], DC):
                img_t = example['img'].data
                if isinstance(img_t, torch.Tensor) and img_t.dim() == 4:
                    # Produce [T, num_cam, C, H, W] so that after dataloader
                    # stacking it becomes [bs, T, num_cam, C, H, W], matching
                    # BEVFormer.obtain_history_bev expectations.
                    img_tc = torch.stack([img_t, img_t], dim=0)  # [2, num_cam, C, H, W]
                    example['img'] = DC(img_tc, cpu_only=False, stack=True)
        except Exception:
            pass

        # NOTE: do NOT touch img_metas structure here.
        # CustomNuScenesDataset.union2one relies on a dict-of-metas format
        # `{timestep: meta_dict}`. Rewriting it can corrupt meta dict contents.
        return example


@DATASETS.register_module()
class CustomNuScenesDetMapDataset(CustomNuScenesDetOccMapDataset):
    """NuScenes dataset for det+map only (det_map).

    This is a thin alias of `CustomNuScenesDetOccMapDataset` that keeps naming
    explicit for det+map experiments. It does NOT change behavior; the
    difference is intended to be controlled by config (e.g., whether the
    pipeline loads `occ_gts`).
    """

    pass
