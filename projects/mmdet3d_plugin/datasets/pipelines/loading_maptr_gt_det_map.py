import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMapTRGTDetMap(object):
    """Load MapTR-style vector-map ground truth for det+map.

    This is intentionally *minimal* and purely additive to avoid impacting the
    existing det+occ pipeline.

    Expected input in `results`:
        - results['map_gt_path']: path to a .npz file containing MapTR GT.

    Expected .npz keys (MapTR / VectorizedLocalMap output convention):
        - gt_vecs_pts_loc: array-like, typically shape (N, S, P, 2) or (N, P, 2)
          depending on padding/fixed-num setting.
        - gt_vecs_label: array-like, shape (N,)

    Output:
        - results['map_gts'] as a dict with keys:
            - 'gt_vecs_pts_loc'
            - 'gt_vecs_label'

    Label order MUST match MapTR:
        0: divider (road_divider / lane_divider)
        1: ped_crossing
        2: boundary (contours)
    """

    def __init__(self, npz_keys=None):
        # Allow customization if user stores different key names.
        if npz_keys is None:
            npz_keys = dict(
                pts='gt_vecs_pts_loc',
                label='gt_vecs_label',
            )
        self.npz_keys = npz_keys

    def __call__(self, results):
        map_gt_path = results.get('map_gt_path', None)
        if map_gt_path is None:
            raise KeyError('LoadMapTRGTDetMap expects `results[\'map_gt_path\']`.')

        with mmcv.FileClient(backend='disk') as _:
            pass

        data = np.load(map_gt_path, allow_pickle=True)

        pts_key = self.npz_keys['pts']
        label_key = self.npz_keys['label']
        if pts_key not in data or label_key not in data:
            raise KeyError(
                f'MapTR map GT npz missing keys: required ({pts_key}, {label_key}), '
                f'got {list(data.keys())} from {map_gt_path}')

        map_gts = {
            'gt_vecs_pts_loc': data[pts_key],
            'gt_vecs_label': data[label_key],
        }
        results['map_gts'] = map_gts
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(npz_keys={self.npz_keys})'
